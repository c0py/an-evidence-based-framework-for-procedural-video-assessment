#!/usr/bin/env python3
"""Run frozen-Qwen CholecT50 absolute-performance development arms."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import socket
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from cholect50_absolute_common import (
    ARMS,
    BASIC_SYSTEM_PROMPT,
    PHASES,
    ROOT,
    SKILL_SYSTEM_PROMPT,
    WORKFLOW_ARBITRATION_GUIDANCE_V2,
    atomic_write_json,
    image_data_url,
    response_schema,
    sha256_file,
    sha256_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/cholect50_absolute_development_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default="qwen3-vl-32b-sop")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arms", nargs="*", choices=(*ARMS, "skill_long_graph_memory"), default=list(ARMS))
    parser.add_argument("--graph-memory-max-calls", type=int, default=4000)
    parser.add_argument("--graph-memory-max-tokens", type=int, default=4096)
    parser.add_argument("--graph-memory-batch-span-s", type=float, default=10.0)
    return parser.parse_args()


def health(url: str, model: str) -> dict[str, Any]:
    with urlopen(url.rstrip("/") + "/models", timeout=30) as response:
        payload = json.loads(response.read())
    if model not in [row.get("id") for row in payload.get("data", [])]:
        raise RuntimeError(f"{model} unavailable at {url}")
    return payload


def post(args: argparse.Namespace, url: str, messages: list[dict]) -> tuple[dict, float]:
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
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=args.timeout_seconds) as response:
            return json.loads(response.read()), time.monotonic() - started
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def _fact_prefix_and_events(row: dict) -> tuple[list[str], list[dict]]:
    lines = [
        "Phase-free specialist observations (fallible; no phase prediction was produced):",
        f"Center time {row['center_seconds']} s of {row['video_duration_seconds']} s "
        f"({100 * row['elapsed_fraction']:.1f}% elapsed).",
    ]
    current = row["current_atomic_observations"]
    if current:
        rendered = "; ".join(
            f"{item['component']}={item['name']} on {item['hit_count']}/{item['sampled_frames']} sampled frames "
            f"(precision estimate {item['calibrated_precision']:.2f})"
            for item in current
        )
        lines.append("Current ±10 s observations: " + rendered + ".")
    else:
        lines.append("Current ±10 s observations: no channel passed its reliability gate.")
    events = []
    for event in row["compound_event_history"]:
        name = event["event"]
        if name in {"clip_application_observed", "scissor_cut_observed", "specimen_bag_observed"}:
            events.append(event)
        elif name in {"coagulation_tool_action_observed", "hook_coagulation_observed"} and event["recent_120s"]:
            events.append(event)
        elif name == "irrigator_observed" and event["current_30s"]:
            events.append(event)
    return lines, events


def compact_facts_v1(row: dict) -> str:
    """Exact fact rendering used by the selected pilot-v1 hypothesis."""
    lines, events = _fact_prefix_and_events(row)
    if events:
        lines.append("Observed event history up to the center:")
        for event in events:
            recency = "current/recent" if event["current_30s"] else (
                "within 120 s" if event["recent_120s"] else "earlier"
            )
            lines.append(
                f"- {event['event']}: {event['hit_count']} sampled hits, first {event['first_hit_seconds']} s, "
                f"last {event['last_hit_seconds']} s ({recency})."
            )
    else:
        lines.append("Observed event history: none of the conservative compound events passed.")
    return "\n".join(lines)


def compact_facts_v2(row: dict) -> str:
    lines, events = _fact_prefix_and_events(row)
    if events:
        event_by_name = {event["event"]: event for event in events}
        if {
            "clip_application_observed",
            "scissor_cut_observed",
        }.issubset(event_by_name):
            lines.append(
                "Workflow milestone: reliable clip-application AND scissor-cut events were both "
                "observed before the center."
            )
        bag = event_by_name.get("specimen_bag_observed")
        if bag is not None:
            lines.append(
                f"Specimen-bag timing: first seen {row['center_seconds'] - bag['first_hit_seconds']} s "
                f"before center; last seen {row['center_seconds'] - bag['last_hit_seconds']} s before center."
            )
        lines.append("Observed event history up to the center:")
        for event in events:
            recency = "current/recent" if event["current_30s"] else (
                "within 120 s" if event["recent_120s"] else "earlier"
            )
            lines.append(
                f"- {event['event']}: {event['hit_count']} sampled hits, first {event['first_hit_seconds']} s, "
                f"last {event['last_hit_seconds']} s ({recency})."
            )
    else:
        lines.append("Observed event history: none of the conservative compound events passed.")
    return "\n".join(lines)


def messages(case: dict, arm: str, temporal: dict) -> list[dict]:
    if arm == "vision_only_local":
        system = BASIC_SYSTEM_PROMPT
        images = case["local_images"]
        frames = case["local_frames"]
    elif arm == "skill_local":
        system = SKILL_SYSTEM_PROMPT
        images = case["local_images"]
        frames = case["local_frames"]
    else:
        system = SKILL_SYSTEM_PROMPT
        images = case["long_images"]
        frames = case["long_frames"]
    content: list[dict] = []
    for position, (path, frame) in enumerate(zip(images, frames), start=1):
        marker = " CENTER" if int(frame) == int(case["anchor_frame"]) else ""
        content.append({"type": "text", "text": f"Chronological image {position}: t={frame} s{marker}."})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(Path(path))}})
    instruction = "Choose exactly one phase for the CENTER time."
    if arm in ("skill_long_visual", "skill_long_gated_facts", "skill_long_gated_facts_v2"):
        instruction += (
            f"\nTemporal metadata: center={case['anchor_frame']} s, video duration={case['video_frames'] - 1} s, "
            f"elapsed={100 * case['elapsed_fraction']:.1f}%."
        )
    if arm == "skill_long_gated_facts":
        instruction += "\n\n" + compact_facts_v1(temporal)
    if arm == "skill_long_gated_facts_v2":
        instruction += "\n\n" + compact_facts_v2(temporal)
    if arm == "skill_long_gated_facts_v2":
        instruction += "\n\n" + WORKFLOW_ARBITRATION_GUIDANCE_V2
    content.append({"type": "text", "text": instruction})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def select_cases(args: argparse.Namespace) -> list[dict]:
    name = "DEVELOPMENT_CASES.json" if args.cohort == "full" else "PILOT_CASES.json"
    rows = json.loads((args.run_dir / name).read_text())
    if args.cohort != "smoke":
        return rows
    by_phase = {phase: [] for phase in PHASES}
    for row in rows:
        by_phase[int(row["truth"])].append(row)
    return [row for phase in PHASES for row in by_phase[phase][:2]]


def main() -> None:
    args = parse_args()
    if "skill_long_graph_memory" in args.arms:
        if args.arms != ["skill_long_graph_memory"]:
            raise ValueError("Run graph memory as a separate arm/output to preserve frozen results")
        from run_cholec_graph_main import run
        run(args)
        return
    if args.output_dir.exists() and not args.resume:
        response_dir = args.output_dir / "responses"
        has_prior_results = (args.output_dir / "EXECUTION_RECEIPT.json").exists() or (
            response_dir.exists() and any(response_dir.glob("*.json"))
        )
        if has_prior_results:
            raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    responses = args.output_dir / "responses"
    responses.mkdir(exist_ok=True)
    cases = select_cases(args)
    temporal_payload = json.loads((args.run_dir / "TEMPORAL_FACTS_V2.json").read_text())
    temporal = {row["case_id"]: row for row in temporal_payload["cases"]}
    if any("truth" in temporal[row["case_id"]] for row in cases):
        raise RuntimeError("ground truth leaked into plugin facts")
    server_health = {url: health(url, args.model) for url in args.base_url}
    tasks = [(case, arm) for case in cases for arm in args.arms]
    tasks.sort(key=lambda item: hashlib.sha256(f"{item[0]['case_id']}:{item[1]}".encode()).hexdigest())
    work: queue.Queue = queue.Queue()
    for task in tasks:
        work.put(task)
    lock = threading.Lock()
    stop = threading.Event()
    completed = 0
    errors = []

    def worker(url: str) -> None:
        nonlocal completed
        while not stop.is_set():
            try:
                case, arm = work.get_nowait()
            except queue.Empty:
                return
            output = responses / f"{case['case_id']}__{arm}.json"
            try:
                prompt = messages(case, arm, temporal[case["case_id"]])
                prompt_sha = sha256_json(prompt)
                if output.exists():
                    existing = json.loads(output.read_text())
                    if not args.resume or existing.get("prompt_sha256") != prompt_sha:
                        raise RuntimeError(f"unsafe existing response {output}")
                else:
                    last_error = None
                    for attempt in range(1, 4):
                        try:
                            body, latency = post(args, url, prompt)
                            raw = body["choices"][0]["message"]["content"]
                            parsed = json.loads(raw)
                            atomic_write_json(
                                output,
                                {
                                    "case_id": case["case_id"],
                                    "video": case["video"],
                                    "anchor_frame": case["anchor_frame"],
                                    "arm": arm,
                                    "parsed": parsed,
                                    "raw_response": raw,
                                    "model": args.model,
                                    "server": url,
                                    "latency_seconds": latency,
                                    "attempt": attempt,
                                    "usage": body.get("usage"),
                                    "prompt_sha256": prompt_sha,
                                    "ground_truth_in_prompt": False,
                                    "plugin_phase_output_in_prompt": False,
                                },
                            )
                            last_error = None
                            break
                        except Exception as exc:
                            last_error = exc
                            time.sleep(min(10 * attempt, 30))
                    if last_error is not None:
                        raise last_error
                with lock:
                    completed += 1
                    if completed % 10 == 0 or completed == len(tasks):
                        print(json.dumps({"completed": completed, "total": len(tasks)}), flush=True)
            except Exception as exc:
                with lock:
                    errors.append({"case_id": case["case_id"], "arm": arm, "error": repr(exc)})
                stop.set()
            finally:
                work.task_done()

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(args.base_url)) as executor:
        futures = [executor.submit(worker, url) for url in args.base_url]
        for future in futures:
            future.result()
    if errors:
        atomic_write_json(args.output_dir / "ERRORS.json", errors)
        raise RuntimeError(errors)
    if completed != len(tasks):
        raise RuntimeError(f"incomplete {completed}/{len(tasks)}")
    receipt = {
        "status": "development_qwen_execution_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cohort": args.cohort,
        "cases": len(cases),
        "arms": args.arms,
        "calls": len(tasks),
        "temperature": 0.0,
        "model": args.model,
        "server_health": server_health,
        "ground_truth_in_prompts": False,
        "plugin_phase_output_in_prompts": False,
        "elapsed_seconds": time.monotonic() - started,
        "temporal_facts_sha256": sha256_file(args.run_dir / "TEMPORAL_FACTS_V2.json"),
        "script_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_json(args.output_dir / "EXECUTION_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
