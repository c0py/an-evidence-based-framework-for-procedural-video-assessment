#!/usr/bin/env python3
"""Bounded frozen-Qwen phase pair on public CholecT50 development frames.

Arms differ only by the presence of a five-frame atomic-fact timeline.  The
plugin never exposes a phase or action-triplet prediction to Qwen.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler
import json
import math
import mimetypes
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen


PHASES = {
    0: "preparation",
    1: "calot_triangle_dissection",
    2: "clipping_and_cutting",
    3: "gallbladder_dissection",
    4: "gallbladder_packaging",
    5: "cleaning_and_coagulation",
    6: "gallbladder_extraction",
}
ARMS = ("vision_only", "vision_plus_atomic_facts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-per-phase", type=int, default=3)
    parser.add_argument("--maximum-window-span", type=int, default=50)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="qwen3-vl-32b-sop")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_facts(path: Path) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {(record["video"], int(record["frame_id"])): record for record in records}
    if len(indexed) != len(records):
        raise RuntimeError("FACTS.jsonl contains duplicate canonical frame keys")
    return records, indexed


def freeze_cases(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    protocol_path = args.output_dir / "FROZEN_PROTOCOL.json"
    cases_path = args.output_dir / "CASES.json"
    if protocol_path.exists() or cases_path.exists():
        if not (protocol_path.exists() and cases_path.exists()):
            raise RuntimeError("partial frozen protocol exists")
        protocol = json.loads(protocol_path.read_text())
        cases = json.loads(cases_path.read_text())
        if protocol["facts_sha256"] != sha256_file(args.facts):
            raise RuntimeError("FACTS.jsonl changed after protocol freeze")
        return protocol, cases

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fact_records, _ = load_facts(args.facts)
    frame_ids_by_video: dict[str, list[int]] = {}
    image_by_key = {}
    for record in fact_records:
        video = record["video"]
        frame = int(record["frame_id"])
        frame_ids_by_video.setdefault(video, []).append(frame)
        image_by_key[(video, frame)] = record["image"]
    for video in frame_ids_by_video:
        frame_ids_by_video[video].sort()

    phase_by_key = {}
    label_hashes = {}
    for label_path in sorted(args.labels_dir.glob("VID*.json")):
        label_hashes[label_path.name] = sha256_file(label_path)
        payload = json.loads(label_path.read_text())
        for frame, rows in payload["annotations"].items():
            phases = {int(row[14]) for row in rows}
            if len(phases) != 1:
                raise RuntimeError(f"inconsistent phase at {label_path.stem}/{frame}")
            phase_by_key[(label_path.stem, int(frame))] = next(iter(phases))

    candidates: dict[int, dict[str, list[dict]]] = {phase: {} for phase in PHASES}
    for video, frame_ids in sorted(frame_ids_by_video.items()):
        position = {frame: index for index, frame in enumerate(frame_ids)}
        for frame in frame_ids:
            key = (video, frame)
            if key not in phase_by_key:
                continue
            index = position[frame]
            if index < 2 or index + 2 >= len(frame_ids):
                continue
            window = frame_ids[index - 2 : index + 3]
            if window[-1] - window[0] > args.maximum_window_span:
                continue
            phase = phase_by_key[key]
            rank = hashlib.sha256(f"{phase}:{video}:{frame}".encode()).hexdigest()
            candidates[phase].setdefault(video, []).append(
                {"video": video, "anchor_frame": frame, "window_frames": window, "rank": rank}
            )

    cases = []
    for phase in PHASES:
        best_by_video = []
        for video, items in candidates[phase].items():
            best_by_video.append(min(items, key=lambda item: item["rank"]))
        chosen = sorted(best_by_video, key=lambda item: item["rank"])[: args.cases_per_phase]
        if len(chosen) != args.cases_per_phase:
            raise RuntimeError(f"phase {phase} has only {len(chosen)} eligible distinct videos")
        for item in chosen:
            case_id = f"p{phase}_{item['video']}_{item['anchor_frame']:06d}"
            cases.append(
                {
                    "case_id": case_id,
                    "video": item["video"],
                    "anchor_frame": item["anchor_frame"],
                    "window_frames": item["window_frames"],
                    "window_span": item["window_frames"][-1] - item["window_frames"][0],
                    "images": [image_by_key[(item["video"], frame)] for frame in item["window_frames"]],
                    "ground_truth_phase_id": phase,
                    "ground_truth_phase_name": PHASES[phase],
                    "selection_rank": item["rank"],
                }
            )
    cases.sort(key=lambda item: (item["ground_truth_phase_id"], item["selection_rank"]))
    selection_digest = hashlib.sha256(
        json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    protocol = {
        "status": "frozen_public_development_protocol",
        "formal_confirmation": False,
        "selection_consumed_phase_labels_for_balancing": True,
        "selection_did_not_consume_visual_plugin_scores": True,
        "cases_per_phase": args.cases_per_phase,
        "total_cases": len(cases),
        "maximum_window_span": args.maximum_window_span,
        "window_frames": 5,
        "center_frame_index_one_based": 3,
        "facts_sha256": sha256_file(args.facts),
        "label_hashes": label_hashes,
        "selection_digest": selection_digest,
        "arms": list(ARMS),
        "shared_frozen_qwen": True,
        "temperature": 0.0,
        "plugin_exposes_phase": False,
        "plugin_exposes_triplet_head": False,
        "primary_metric": "paired_accuracy_difference_facts_minus_vision",
        "secondary_metrics": ["macro_f1", "per_phase_accuracy", "paired_exact_sign_test"],
        "decision_rule": "Proceed only if facts improve both accuracy and macro-F1 without a large concentrated failure mode.",
    }
    atomic_write_json(cases_path, cases)
    atomic_write_json(protocol_path, protocol)
    return protocol, cases


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def fact_timeline(case: dict, facts: dict[tuple[str, int], dict]) -> str:
    lines = [
        "Atomic facts predicted by a held-out specialist model. They are fallible evidence, not ground truth:"
    ]
    for index, frame in enumerate(case["window_frames"], start=1):
        record = facts[(case["video"], int(frame))]
        lines.append(f"Frame {index} (source frame {frame}):")
        for component in ("instrument", "verb", "target"):
            entries = record["facts"][component][:2]
            text = ", ".join(f"{entry['name']} ({entry['score']:.3f})" for entry in entries)
            lines.append(f"  {component}: {text}")
    return "\n".join(lines)


def messages_for(case: dict, arm: str, facts: dict[tuple[str, int], dict]) -> list[dict]:
    taxonomy = "; ".join(f"{index}={name}" for index, name in PHASES.items())
    system = (
        "You assess laparoscopic cholecystectomy workflow. Classify the surgical phase at the CENTER "
        "(third) frame using the five chronological images. Use temporal context, but answer for the "
        f"center frame only. Phase taxonomy: {taxonomy}. When atomic facts are supplied, treat them as "
        "fallible model observations and arbitrate them against the images. Return only the required JSON."
    )
    content = []
    for index, image in enumerate(case["images"], start=1):
        content.append({"type": "text", "text": f"Chronological frame {index}:"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(Path(image))}})
    instruction = "Select exactly one phase for chronological frame 3."
    if arm == "vision_plus_atomic_facts":
        instruction += "\n\n" + fact_timeline(case, facts)
    content.append({"type": "text", "text": instruction})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "phase_id": {"type": "integer", "minimum": 0, "maximum": 6},
            "phase_name": {"type": "string", "enum": list(PHASES.values())},
            "rationale": {"type": "string"},
        },
        "required": ["phase_id", "phase_name", "rationale"],
        "additionalProperties": False,
    }


def post_completion(args: argparse.Namespace, messages: list[dict]) -> tuple[dict, float]:
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "surgical_phase", "strict": True, "schema": response_schema()},
        },
    }
    request = Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"},
        method="POST",
    )
    started = time.monotonic()
    with urlopen(request, timeout=args.timeout_s) as response:
        body = json.loads(response.read())
    return body, time.monotonic() - started


def model_health(args: argparse.Namespace) -> dict:
    with urlopen(args.base_url.rstrip("/") + "/models", timeout=30) as response:
        payload = json.loads(response.read())
    models = [item["id"] for item in payload.get("data", [])]
    if args.model not in models:
        raise RuntimeError(f"model {args.model!r} unavailable: {models}")
    return payload


def classification_metrics(truth: list[int], prediction: list[int | None]) -> dict:
    accuracy = sum(p == t for p, t in zip(prediction, truth)) / len(truth)
    per_phase = {}
    f1s = []
    for phase in PHASES:
        tp = sum(t == phase and p == phase for t, p in zip(truth, prediction))
        fp = sum(t != phase and p == phase for t, p in zip(truth, prediction))
        fn = sum(t == phase and p != phase for t, p in zip(truth, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_phase[str(phase)] = {
            "name": PHASES[phase], "support": sum(t == phase for t in truth),
            "accuracy": recall, "precision": precision, "f1": f1,
        }
        f1s.append(f1)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s), "per_phase": per_phase}


def exact_sign_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def evaluate(args: argparse.Namespace, cases: list[dict]) -> dict:
    truth = [int(case["ground_truth_phase_id"]) for case in cases]
    predictions: dict[str, list[int | None]] = {arm: [] for arm in ARMS}
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "truth": case["ground_truth_phase_id"]}
        for arm in ARMS:
            path = args.output_dir / "responses" / f"{case['case_id']}__{arm}.json"
            if not path.exists():
                raise RuntimeError(f"missing response: {path}")
            payload = json.loads(path.read_text())
            parsed = payload.get("parsed")
            prediction = None
            if isinstance(parsed, dict):
                candidate = parsed.get("phase_id")
                if candidate in PHASES and parsed.get("phase_name") == PHASES[candidate]:
                    prediction = int(candidate)
            predictions[arm].append(prediction)
            row[arm] = prediction
        rows.append(row)
    correct = {
        arm: [prediction == target for prediction, target in zip(predictions[arm], truth)]
        for arm in ARMS
    }
    gains = sum((not a) and b for a, b in zip(correct["vision_only"], correct["vision_plus_atomic_facts"]))
    losses = sum(a and (not b) for a, b in zip(correct["vision_only"], correct["vision_plus_atomic_facts"]))
    metrics = {arm: classification_metrics(truth, predictions[arm]) for arm in ARMS}
    result = {
        "status": "public_development_pair_complete",
        "formal_confirmation": False,
        "cases": len(cases),
        "metrics": metrics,
        "paired": {
            "accuracy_difference_facts_minus_vision": (
                metrics["vision_plus_atomic_facts"]["accuracy"] - metrics["vision_only"]["accuracy"]
            ),
            "macro_f1_difference_facts_minus_vision": (
                metrics["vision_plus_atomic_facts"]["macro_f1"] - metrics["vision_only"]["macro_f1"]
            ),
            "facts_gain_cases": gains,
            "facts_harm_cases": losses,
            "both_correct": sum(a and b for a, b in zip(correct["vision_only"], correct["vision_plus_atomic_facts"])),
            "both_wrong": sum((not a) and (not b) for a, b in zip(correct["vision_only"], correct["vision_plus_atomic_facts"])),
            "exact_two_sided_sign_p": exact_sign_p(gains, losses),
        },
        "case_predictions": rows,
        "limitations": [
            "Balanced sample selection consumed public development phase labels.",
            "This small pair is a development gate, not an independent confirmation.",
            "The complete official dataset is required for a formal video-level cross-validation experiment.",
        ],
    }
    atomic_write_json(args.output_dir / "EVALUATION.json", result)
    return result


def main() -> None:
    args = parse_args()
    protocol, cases = freeze_cases(args)
    print(json.dumps({"frozen_protocol": protocol, "case_ids": [case["case_id"] for case in cases]}, indent=2))
    if args.freeze_only:
        return
    if not args.base_url:
        raise ValueError("--base-url is required unless --freeze-only is used")
    health = model_health(args)
    _, facts = load_facts(args.facts)
    responses = args.output_dir / "responses"
    responses.mkdir(exist_ok=True)
    prompt_receipts = []
    for case in cases:
        order = list(ARMS)
        if int(hashlib.sha256(case["case_id"].encode()).hexdigest(), 16) % 2:
            order.reverse()
        for arm in order:
            output = responses / f"{case['case_id']}__{arm}.json"
            messages = messages_for(case, arm, facts)
            prompt_sha = hashlib.sha256(
                json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            prompt_receipts.append({"case_id": case["case_id"], "arm": arm, "prompt_sha256": prompt_sha})
            if output.exists():
                if not args.resume:
                    raise FileExistsError(output)
                continue
            try:
                body, latency = post_completion(args, messages)
                raw = body["choices"][0]["message"]["content"]
                parsed = json.loads(raw)
                payload = {
                    "case_id": case["case_id"], "arm": arm, "model": args.model,
                    "temperature": 0.0, "latency_s": latency, "parsed": parsed,
                    "raw_response": raw, "usage": body.get("usage"), "prompt_sha256": prompt_sha,
                }
                atomic_write_json(output, payload)
                print(json.dumps({"completed": str(output), "parsed": parsed, "latency_s": latency}), flush=True)
            except Exception as exc:
                error = {
                    "case_id": case["case_id"], "arm": arm,
                    "error_type": type(exc).__name__, "error": str(exc), "prompt_sha256": prompt_sha,
                }
                atomic_write_json(output.with_suffix(".ERROR.json"), error)
                raise
    receipt = {
        "model_health": health,
        "model": args.model,
        "base_url": args.base_url,
        "calls": len(prompt_receipts),
        "prompt_receipts": prompt_receipts,
        "ground_truth_fields_in_prompts": False,
    }
    atomic_write_json(args.output_dir / "EXECUTION_RECEIPT.json", receipt)
    result = evaluate(args, cases)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
