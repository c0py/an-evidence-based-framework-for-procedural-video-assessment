"""Explicit, auditable chat-completion transport for the memory manager only."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def write_new_json(path: Path, value: dict) -> None:
    """Exclusive creation: never overwrite a prior artifact."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


class GraphMemoryChatClient:
    def __init__(self, base_url: str, model: str, audit_dir: Path, *,
                 api_key_env: str = "QWEN_API_KEY", max_tokens: int = 4096,
                 timeout_s: float = 900, max_request_chars: int = 120000):
        if not base_url.startswith(("http://", "https://")) or not model:
            raise ValueError("Explicit model and chat-completion endpoint are required")
        if max_tokens < 1 or timeout_s <= 0 or max_request_chars < 1000:
            raise ValueError("Invalid transport limits")
        self.base_url, self.model, self.audit_dir = base_url.rstrip("/"), model, audit_dir
        self.api_key_env = api_key_env
        self.max_tokens, self.timeout_s = max_tokens, timeout_s
        self.max_request_chars = max_request_chars
        self.calls = 0

    def __call__(self, messages: list[dict], schema: dict) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": 0.0,
                   "max_tokens": self.max_tokens,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "event_memory_update", "strict": True, "schema": schema}}}
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(body) > self.max_request_chars:
            raise ValueError("Memory request exceeds budget; reduce batch/candidate limits")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.calls += 1
        stem = f"{self.calls:06d}"
        # Audit payload excludes authorization headers/environment values.
        write_new_json(self.audit_dir / f"{stem}.request.json", payload)
        request = Request(self.base_url + "/chat/completions", data=body.encode("utf-8"),
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {os.environ.get(self.api_key_env, 'EMPTY')}"},
                          method="POST")
        started = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw_text = response.read().decode("utf-8")
            write_new_json(self.audit_dir / f"{stem}.response.json",
                           {"elapsed_s": time.monotonic() - started, "raw_response": raw_text})
            response = json.loads(raw_text)
            choice = response["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete memory update; refusing truncated output")
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Expected JSON string content")
            proposal = json.loads(content)
            if not isinstance(proposal, dict):
                raise ValueError("Expected graph update object")
            write_new_json(self.audit_dir / f"{stem}.usage.json",
                           {"usage": response.get("usage"), "elapsed_s": time.monotonic() - started})
            return proposal
        except HTTPError as exc:
            write_new_json(self.audit_dir / f"{stem}.error.json",
                           {"type": "HTTPError", "status": exc.code,
                            "elapsed_s": time.monotonic() - started})
            raise RuntimeError(f"Memory backend HTTP {exc.code}; no graph update committed") from exc
        except Exception as exc:
            write_new_json(self.audit_dir / f"{stem}.error.json",
                           {"type": type(exc).__name__, "elapsed_s": time.monotonic() - started})
            raise
