"""Minimal OpenAI-compatible server for a local Qwen3-VL checkpoint.

This intentionally supports the subset used by the assessment framework:
text-only or text-plus-data-URL image messages and deterministic chat
completions.  It keeps model deployment isolated from the framework core.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import asynccontextmanager
import hashlib
from io import BytesIO
import json
import os
import tempfile
import time
from typing import Any
from urllib.request import urlopen

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import uvicorn
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 512
    temperature: float = 0.0
    response_format: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


def decode_data_image(url: str) -> Image.Image:
    if not url.startswith("data:image/") or ";base64," not in url:
        raise ValueError("Only base64 image data URLs are supported")
    payload = url.split(";base64,", 1)[1]
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


def decode_video_url(url: str, maximum_frames: int) -> list[Image.Image]:
    """Uniformly decode a submitted video into a bounded chronological image list."""
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temporary_path = handle.name
        with urlopen(url, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    capture = cv2.VideoCapture(temporary_path)
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to decode submitted video URL: {url}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Submitted video has no decodable frames: {url}")
        indices = np.linspace(
            0, frame_count - 1, min(maximum_frames, frame_count), dtype=int,
        ).tolist()
        images = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Unable to decode submitted video frame {index}")
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return images
    finally:
        capture.release()
        os.unlink(temporary_path)


def normalize_messages(
    messages: list[dict[str, Any]], maximum_video_frames: int = 24,
) -> tuple[list[dict[str, Any]], int]:
    normalized = []
    video_frame_count = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            # Newer multimodal Transformers processors iterate every message's
            # content as typed blocks.  Keeping a plain string here makes the
            # processor iterate characters and fail on system/text messages.
            normalized.append({
                "role": message["role"],
                "content": [{"type": "text", "text": content}],
            })
            continue
        converted = []
        for item in content:
            if item.get("type") == "text":
                converted.append({"type": "text", "text": str(item.get("text", ""))})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                converted.append({"type": "image", "image": decode_data_image(image_url)})
            elif item.get("type") == "video_url":
                video_url = item.get("video_url", {}).get("url", "")
                frames = decode_video_url(video_url, maximum_video_frames)
                video_frame_count += len(frames)
                converted.extend({"type": "image", "image": frame} for frame in frames)
            else:
                raise ValueError(f"Unsupported multimodal content item: {item.get('type')}")
        normalized.append({"role": message["role"], "content": converted})
    return normalized, video_frame_count


def build_app(
    model_path: str, served_model_name: str, maximum_video_frames: int = 24,
    device_map: str = "single", max_memory_gib: int | None = None,
    maximum_completion_tokens: int = 2048,
) -> FastAPI:
    if maximum_completion_tokens < 1:
        raise ValueError("maximum_completion_tokens must be positive")
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["processor"] = AutoProcessor.from_pretrained(model_path)
        placement: Any = "auto" if device_map == "auto" else {"": "cuda"}
        maximum_memory = None
        if device_map == "auto" and max_memory_gib is not None:
            maximum_memory = {
                index: f"{max_memory_gib}GiB" for index in range(torch.cuda.device_count())
            }
        state["model"] = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=placement,
            max_memory=maximum_memory,
            attn_implementation="sdpa",
        ).eval()
        yield
        state.clear()
        torch.cuda.empty_cache()

    app = FastAPI(title="Qwen3-VL OpenAI-compatible local server", lifespan=lifespan)

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": served_model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest) -> dict[str, Any]:
        if request.model != served_model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
        try:
            messages, video_frame_count = normalize_messages(
                request.messages, maximum_video_frames,
            )
            processor = state["processor"]
            model = state["model"]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            effective_max_tokens = min(
                max(int(request.max_tokens), 1), maximum_completion_tokens,
            )
            schema = None
            prefix_allowed_tokens_fn = None
            response_format = request.response_format or {}
            if response_format.get("type") == "json_schema":
                schema_wrapper = response_format.get("json_schema")
                if not isinstance(schema_wrapper, dict):
                    raise ValueError("json_schema response format requires a json_schema object")
                schema = schema_wrapper.get("schema")
                if not isinstance(schema, dict):
                    raise ValueError("json_schema response format requires json_schema.schema")
                prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
                    processor.tokenizer,
                    JsonSchemaParser(schema),
                )
            started = time.monotonic()
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": effective_max_tokens,
                "do_sample": request.temperature > 0,
                "temperature": max(float(request.temperature), 1e-5),
            }
            if prefix_allowed_tokens_fn is not None:
                generation_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            trimmed = [
                output[len(input_ids):]
                for input_ids, output in zip(inputs.input_ids, generated)
            ]
            text = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]
            schema_sha256 = None
            json_control_character_normalized = False
            if schema is not None:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as error:
                    if "Invalid control character" not in str(error):
                        raise
                    try:
                        parsed = json.loads(text, strict=False)
                    except json.JSONDecodeError as relaxed_error:
                        diagnostic = {
                            "strict_error": str(error),
                            "relaxed_error": str(relaxed_error),
                            "generated_characters": len(text),
                            "completion_tokens": int(sum(
                                item.numel() for item in trimmed
                            )),
                            "generated_tail": text[-600:],
                            "generated_text_base64": base64.b64encode(
                                text.encode()
                            ).decode(),
                        }
                        raise ValueError(
                            json.dumps(diagnostic, ensure_ascii=False)
                        ) from relaxed_error
                    text = json.dumps(
                        parsed, ensure_ascii=False, separators=(",", ":"),
                    )
                    json_control_character_normalized = True
                validate_json_schema(instance=parsed, schema=schema)
                schema_sha256 = hashlib.sha256(
                    json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            prompt_tokens = int(inputs.input_ids.numel())
            completion_tokens = int(sum(item.numel() for item in trimmed))
            finish_reason = (
                "length" if completion_tokens >= effective_max_tokens else "stop"
            )
            return {
                "id": f"qwen3vl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": served_model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "generation_latency_s": time.monotonic() - started,
                    "video_frames_sampled": video_frame_count,
                    "requested_max_tokens": int(request.max_tokens),
                    "effective_max_tokens": effective_max_tokens,
                    "json_schema_constrained": schema is not None,
                    "json_schema_sha256": schema_sha256,
                    "json_control_character_normalized": (
                        json_control_character_normalized
                    ),
                },
            }
        except (ValueError, KeyError, RuntimeError, JsonSchemaValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="qwen3-vl-8b-sop")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18902)
    parser.add_argument("--maximum-video-frames", type=int, default=24)
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    parser.add_argument("--max-memory-gib", type=int)
    parser.add_argument("--maximum-completion-tokens", type=int, default=2048)
    args = parser.parse_args()
    uvicorn.run(
        build_app(
            args.model_path, args.served_model_name, args.maximum_video_frames,
            args.device_map, args.max_memory_gib, args.maximum_completion_tokens,
        ),
        host=args.host, port=args.port, workers=1,
    )


if __name__ == "__main__":
    main()
