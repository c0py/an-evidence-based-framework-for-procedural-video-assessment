"""Full-cohort, chronological graph-memory arm of the existing Cholec runner.

Called by run_cholect50_absolute_development_qwen.py --arms skill_long_graph_memory.
No truth enters the memory manager or phase prompts. Evaluation loads truth only
after all requested phase predictions are saved and hashed.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.graph_memory_transport import GraphMemoryChatClient, write_new_json
from cvs_assessment.tasks.cholec_graph_memory import (
    file_sha256, graph_judgment_messages, load_observations, observation_batches,
)
from cvs_assessment.temporal_graph_memory import EventGraphMemory, UPDATE_PROMPT, update_schema
from cholect50_absolute_common import (
    PHASES, SKILL_SYSTEM_PROMPT, WORKFLOW_ARBITRATION_GUIDANCE_V2,
    atomic_write_json, sha256_json, parse_prediction,
)


ARM = "skill_long_graph_memory"
CASE_FIELDS = ("case_id", "video", "anchor_frame", "video_frames", "elapsed_fraction",
               "long_frames", "long_images")


def _metrics(truth: list[int], predicted: list[int]) -> dict:
    if not truth or len(truth) != len(predicted):
        raise ValueError("Metrics require paired nonempty predictions")
    f1s = []
    for phase in PHASES:
        tp = sum(y == phase and p == phase for y, p in zip(truth, predicted))
        fp = sum(y != phase and p == phase for y, p in zip(truth, predicted))
        fn = sum(y == phase and p != phase for y, p in zip(truth, predicted))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"n": len(truth), "accuracy": sum(y == p for y, p in zip(truth, predicted)) / len(truth),
            "macro_f1_fixed_7_classes": sum(f1s) / len(f1s), "per_class_f1": f1s}


def _evaluate(args, cases: list[dict]) -> dict:
    labels = {r["case_id"]: int(r["truth"]) for r in
              json.loads((args.run_dir / "DEVELOPMENT_CASES.json").read_text())}
    baseline_dir = args.run_dir / "qwen_smoke_v1/responses"
    rows = []
    for case in cases:
        path = args.output_dir / "responses" / f"{case['case_id']}__{ARM}.json"
        prediction = parse_prediction(json.loads(path.read_text()))
        if prediction is None:
            raise ValueError(f"Invalid frozen prediction: {path}")
        baseline_path = baseline_dir / f"{case['case_id']}__skill_long_gated_facts_v2.json"
        baseline = parse_prediction(json.loads(baseline_path.read_text())) if baseline_path.exists() else None
        rows.append({"case_id": case["case_id"], "video": case["video"],
                     "truth": labels[case["case_id"]], "graph_prediction": prediction,
                     "existing_v2_prediction": baseline})
    result = {"status": "development_only_not_independent_confirmation",
              "graph": _metrics([r["truth"] for r in rows], [r["graph_prediction"] for r in rows]),
              "per_video": {}, "rows": rows,
              "interpretation": "New integrated arm includes extra memory-model calls; not an isolated graph ablation."}
    for video in sorted({r["video"] for r in rows}):
        group = [r for r in rows if r["video"] == video]
        result["per_video"][video] = _metrics([r["truth"] for r in group], [r["graph_prediction"] for r in group])
    if all(r["existing_v2_prediction"] is not None for r in rows):
        result["existing_v2"] = _metrics([r["truth"] for r in rows], [r["existing_v2_prediction"] for r in rows])
    return result


def run(args) -> None:
    from run_cholect50_absolute_development_qwen import health, post

    if args.cohort != "full" or args.resume:
        raise ValueError("Graph v1 main integration requires a fresh full-cohort run")
    if not args.base_url or args.graph_memory_max_calls < 1:
        raise ValueError("A backend and positive call budget are required")
    if (args.output_dir / "RUN_CONFIG.json").exists() or (args.output_dir / "responses").exists():
        raise FileExistsError(args.output_dir)
    manifest_path = args.run_dir / "DEVELOPMENT_CASES.json"
    # Existing case file contains truth; whitelist inputs immediately. No label-based selection.
    cases = [{k: r[k] for k in CASE_FIELDS} for r in json.loads(manifest_path.read_text())]
    if len({r["case_id"] for r in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if len(case["long_images"]) != len(case["long_frames"]):
            raise ValueError("Image alignment mismatch")
        if any(not Path(p).is_file() for p in case["long_images"]):
            raise FileNotFoundError(f"Missing images for {case['case_id']}")
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["video"]].append(case)
    for video in grouped:
        grouped[video].sort(key=lambda r: (r["anchor_frame"], r["case_id"]))
    server_health = {url: health(url, args.model) for url in args.base_url}
    sources = [Path(__file__).resolve(), ROOT / "scripts/run_cholect50_absolute_development_qwen.py",
               ROOT / "scripts/cholect50_absolute_common.py", ROOT / "cvs_assessment/temporal_graph_memory.py",
               ROOT / "cvs_assessment/graph_memory_transport.py", ROOT / "cvs_assessment/tasks/cholec_graph_memory.py",
               ROOT / "scripts/serve_qwen3_vl_openai.py",
               ROOT / "scripts/serve_qwen3_vl_openai_control_char_recovery.py"]
    hashes = {str(p): file_sha256(p) for p in sources}
    scores = args.run_dir / "DEVELOPMENT_SCORES.npz"
    calibration = args.run_dir / "RELIABILITY_CALIBRATION.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "responses").mkdir()
    (args.output_dir / "videos").mkdir()
    config = {"arm": ARM, "cohort": "full", "cases": len(cases), "videos": sorted(grouped),
              "model": args.model, "server_health": server_health, "code_sha256": hashes,
              "cases_sha256": file_sha256(manifest_path), "scores_sha256": file_sha256(scores),
              "calibration_sha256": file_sha256(calibration), "lookahead_s": 2,
              "batch_span_s": args.graph_memory_batch_span_s, "max_samples": 16,
              "candidate_limit": 12, "pending_limit": 24, "manager_max_tokens": args.graph_memory_max_tokens,
              "memory_call_budget": args.graph_memory_max_calls, "max_semantic_attempts": 2,
              "phase_max_tokens": args.max_tokens, "phase_prompt_uses_existing_v2_guidance": True,
              "prediction_guidance_visible_to_memory_manager": False,
              "extra_memory_calls": True, "started_utc": datetime.now(timezone.utc).isoformat()}
    write_new_json(args.output_dir / "RUN_CONFIG.json", config)
    write_new_json(args.output_dir / "LABEL_FREE_CASES.json", {"cases": cases})
    work = queue.Queue()
    for video in sorted(grouped):
        work.put(video)
    lock = threading.Lock()
    stop = threading.Event()
    completed = 0
    memory_calls = 0
    errors = []
    per_video_status = {}
    start = time.monotonic()

    def status() -> None:
        atomic_write_json(args.output_dir / "PROGRESS.json", {
            "status": "failed" if errors else "running", "completed": completed,
            "total": len(cases), "memory_calls": memory_calls,
            "elapsed_s": time.monotonic() - start, "videos": per_video_status, "errors": errors})

    def worker(url: str) -> None:
        nonlocal completed, memory_calls
        while not stop.is_set():
            try:
                video = work.get_nowait()
            except queue.Empty:
                return
            graph = None
            directory = args.output_dir / "videos" / video
            directory.mkdir()
            client = GraphMemoryChatClient(url, args.model, directory / "memory_audit",
                                           max_tokens=args.graph_memory_max_tokens,
                                           timeout_s=args.timeout_seconds)
            try:
                if any(file_sha256(Path(p)) != digest for p, digest in hashes.items()):
                    raise RuntimeError("Implementation changed after run configuration was frozen")
                video_cases = grouped[video]
                maximum = float(video_cases[-1]["anchor_frame"]) + 2
                observations = load_observations(scores, calibration, video, maximum)
                batches = list(observation_batches(observations, span_s=args.graph_memory_batch_span_s,
                    max_samples=16, boundaries=[float(c["anchor_frame"]) + 2 for c in video_cases]))
                graph = EventGraphMemory(video, {"run_config": str(args.output_dir / "RUN_CONFIG.json"),
                                                "model": args.model, "scores_sha256": config["scores_sha256"],
                                                "calibration_sha256": config["calibration_sha256"]})
                batch_index = 0
                (directory / "checkpoints").mkdir()
                (directory / "phase_inputs").mkdir()
                for case in video_cases:
                    if stop.is_set():
                        return
                    cutoff = float(case["anchor_frame"]) + 2
                    while batch_index < len(batches) and batches[batch_index][-1].time_s <= cutoff:
                        batch = batches[batch_index]
                        graph.ingest(batch, batch[-1].time_s)
                        ids = [o.observation_id for o in batch if o.atoms]
                        if ids:
                            payload, allowed, available = graph.manager_input(ids)
                            messages = [{"role": "system", "content": UPDATE_PROMPT},
                                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
                            for attempt in range(2):
                                if stop.is_set():
                                    return
                                with lock:
                                    if memory_calls >= args.graph_memory_max_calls:
                                        raise RuntimeError("Memory call budget exhausted")
                                    memory_calls += 1
                                    per_video_status[video] = {"stage": "memory_update", "cutoff_s": graph.cutoff_s,
                                                              "case_id": case["case_id"], "nodes": len(graph.nodes)}
                                    status()
                                proposal = client(messages, update_schema())
                                try:
                                    graph.apply(proposal, allowed, available)
                                    break
                                except ValueError as exc:
                                    write_new_json(directory / "memory_audit" / f"{client.calls:06d}.validation.json",
                                                   {"accepted": False, "error": str(exc)})
                                    if attempt == 1:
                                        raise
                                    messages += [{"role": "assistant", "content": json.dumps(proposal)},
                                                 {"role": "user", "content": "The entire update was rejected; nothing committed. "
                                                  + str(exc) + ". Return a corrected complete update using the same evidence. "
                                                  "If uncertain use DEFER; never invent supports or phases."}]
                        batch_index += 1
                    graph.ingest([], cutoff)
                    checkpoint = directory / "checkpoints" / f"{case['case_id']}.json"
                    write_new_json(checkpoint, graph.to_dict())
                    prompt = graph_judgment_messages(case, graph, skill_system_prompt=SKILL_SYSTEM_PROMPT,
                                                     lookahead_s=2)
                    prompt[-1]["content"][-1]["text"] += "\n\n" + WORKFLOW_ARBITRATION_GUIDANCE_V2
                    # Persist the exact graph evidence/guidance; image identity stays in the case manifest.
                    write_new_json(directory / "phase_inputs" / f"{case['case_id']}.json", {
                        "system": prompt[0]["content"], "instruction": prompt[-1]["content"][-1]["text"],
                        "image_paths": case["long_images"], "image_times": case["long_frames"],
                        "prompt_sha256": sha256_json(prompt)})
                    with lock:
                        per_video_status[video] = {"stage": "phase_judgment", "case_id": case["case_id"],
                                                  "cutoff_s": cutoff, "nodes": len(graph.nodes)}
                        status()
                    body, latency = post(args, url, prompt)
                    choice = body["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise ValueError("Phase output was truncated")
                    parsed = json.loads(choice["message"]["content"])
                    if parse_prediction({"parsed": parsed}) is None:
                        raise ValueError("Invalid phase ID/name pairing")
                    output = {"case_id": case["case_id"], "video": video, "anchor_frame": case["anchor_frame"],
                              "arm": ARM, "parsed": parsed, "raw_response": choice["message"]["content"],
                              "model": args.model, "server": url, "usage": body.get("usage"),
                              "latency_seconds": latency, "prompt_sha256": sha256_json(prompt),
                              "graph_checkpoint": str(checkpoint), "memory_calls_so_far": client.calls,
                              "ground_truth_in_prompt": False, "plugin_phase_output_in_prompt": False}
                    write_new_json(args.output_dir / "responses" / f"{case['case_id']}__{ARM}.json", output)
                    with lock:
                        completed += 1
                        per_video_status[video]["last_completed_case"] = case["case_id"]
                        status()
                        print(json.dumps({"completed": completed, "total": len(cases),
                                          "video": video, "memory_calls": memory_calls}), flush=True)
                write_new_json(directory / "EVENT_GRAPH.json", graph.to_dict())
                with lock:
                    per_video_status[video] = {"stage": "complete", "cases": len(video_cases),
                                              "memory_calls": client.calls, "nodes": len(graph.nodes)}
                    status()
            except Exception as exc:
                if graph is not None:
                    write_new_json(directory / "FAILED_GRAPH.json", graph.to_dict())
                with lock:
                    errors.append({"video": video, "error": repr(exc)})
                    stop.set()
                    status()
                raise
            finally:
                work.task_done()

    with ThreadPoolExecutor(max_workers=len(args.base_url)) as executor:
        futures = [executor.submit(worker, url) for url in args.base_url]
        for future in futures:
            future.result()
    if completed != len(cases):
        raise RuntimeError(f"Incomplete graph main run: {completed}/{len(cases)}")
    response_hashes = {p.name: file_sha256(p) for p in sorted((args.output_dir / "responses").glob("*.json"))}
    write_new_json(args.output_dir / "PREDICTION_FREEZE.json", {
        "response_sha256": response_hashes, "completed": completed,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "before_metrics": True})
    write_new_json(args.output_dir / "EVALUATION.json", _evaluate(args, cases))
    write_new_json(args.output_dir / "EXECUTION_RECEIPT.json", {
        "status": "graph_main_complete", "cases": completed, "memory_calls": memory_calls,
        "phase_calls": completed, "elapsed_s": time.monotonic() - start,
        "code_sha256": hashes, "server_health": server_health})
    atomic_write_json(args.output_dir / "PROGRESS.json", {
        "status": "complete", "completed": completed, "total": len(cases),
        "memory_calls": memory_calls, "videos": per_video_status})
