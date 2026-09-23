#!/usr/bin/env python3
"""Build one video's model-managed memory; does NOT run phase evaluation.

This script performs model calls ONLY when explicitly invoked with
--allow-model-calls. No existing predictions, facts, or manuscript are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.graph_memory_transport import GraphMemoryChatClient, write_new_json
from cvs_assessment.tasks.cholec_graph_memory import (
    file_sha256, graph_evidence_text, load_observations, observation_batches,
)
from cvs_assessment.temporal_graph_memory import EventGraphMemory, UPDATE_PROMPT, update_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--cutoff-s", type=float, required=True, help="Maximum ingested evidence time")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--base-url", required=True, help="Chat backend URL including /v1 if applicable")
    parser.add_argument("--model", default="qwen3-vl-32b-sop")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--seconds-per-frame-id", type=float, default=1.0)
    parser.add_argument("--batch-span-s", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--pending-limit", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-request-chars", type=int, default=120000)
    parser.add_argument("--max-calls", type=int, default=100, help="Hard call budget; no automatic continuation")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--query-s", type=float, action="append", default=[],
                        help="Optional cut points/evidence exports only, no phase judgment")
    parser.add_argument("--lookahead-s", type=float, choices=(0.0, 2.0), default=0.0,
                        help="0 causal memory; 2 matches existing offline evidence allowance")
    parser.add_argument("--allow-model-calls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_model_calls:
        raise SystemExit("No calls made. Building model-managed memory requires --allow-model-calls.")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not math.isfinite(args.cutoff_s) or args.cutoff_s < 0 or args.max_calls < 1 or
            args.candidate_limit < 1 or args.pending_limit < 0):
        raise ValueError("Invalid build configuration")
    query_cutoffs = [q + args.lookahead_s for q in args.query_s]
    if any(not math.isfinite(q) or q < 0 for q in args.query_s) or any(t > args.cutoff_s for t in query_cutoffs):
        raise ValueError("Query exceeds build cutoff")
    observations = load_observations(args.scores, args.calibration, args.video, args.cutoff_s,
                                     seconds_per_frame_id=args.seconds_per_frame_id)
    if not observations:
        raise ValueError("No sampled observations at/before cutoff")
    batches = list(observation_batches(observations, span_s=args.batch_span_s,
                                       max_samples=args.max_samples, boundaries=query_cutoffs))
    needed_calls = sum(any(o.atoms for o in batch) for batch in batches)
    if needed_calls > args.max_calls:
        raise ValueError(f"Build requires up to {needed_calls} calls; explicit budget is {args.max_calls}")
    metadata = {
        "scores_sha256": file_sha256(args.scores),
        "calibration_sha256": file_sha256(args.calibration),
        "prompt_sha256": hashlib.sha256(UPDATE_PROMPT.encode()).hexdigest(),
        "model": args.model, "seconds_per_frame_id": args.seconds_per_frame_id,
        "batch_span_s": args.batch_span_s, "max_samples": args.max_samples,
        "candidate_limit": args.candidate_limit, "pending_limit": args.pending_limit,
        "model_managed": True, "ground_truth_loaded": False,
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in (
            Path(__file__).resolve(), ROOT / "cvs_assessment/temporal_graph_memory.py",
            ROOT / "cvs_assessment/tasks/cholec_graph_memory.py",
            ROOT / "cvs_assessment/graph_memory_transport.py")},
    }
    client = GraphMemoryChatClient(args.base_url, args.model, args.output_dir / "model_audit",
                                   api_key_env=args.api_key_env, max_tokens=args.max_tokens,
                                   timeout_s=args.timeout_s, max_request_chars=args.max_request_chars)
    graph = EventGraphMemory(args.video, metadata)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_new_json(args.output_dir / "BUILD_CONFIG.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "planned_upper_bound_calls": needed_calls, "metadata": metadata})
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir()
    try:
        for index, batch in enumerate(batches):
            graph.ingest(batch, batch[-1].time_s)
            ids = [o.observation_id for o in batch if o.atoms]
            if ids:
                update_memory(graph, ids, client, candidate_limit=args.candidate_limit,
                              pending_limit=args.pending_limit)
            # Full append-only checkpoint, including raw evidence and replay journal.
            write_new_json(checkpoints / f"{index:06d}.json", graph.to_dict())
        graph.ingest([], args.cutoff_s)
        write_new_json(args.output_dir / "EVENT_GRAPH.json", graph.to_dict())
        for index, query_s in enumerate(args.query_s):
            text = graph_evidence_text(graph, query_s, lookahead_s=args.lookahead_s)
            write_new_json(args.output_dir / f"query_{index:04d}.json",
                           {"query_s": query_s, "evidence_text": text,
                            "phase_prediction_generated": False})
        write_new_json(args.output_dir / "BUILD_RECEIPT.json", {
            "status": "memory_built_not_evaluated", "calls": client.calls,
            "nodes": len(graph.nodes), "observations": len(graph.observations),
            "phase_evaluation_run": False})
    except Exception as exc:
        write_new_json(args.output_dir / "FAILED_GRAPH.json", graph.to_dict())
        write_new_json(args.output_dir / "BUILD_FAILURE.json", {
            "status": "failed_no_rule_fallback", "error_type": type(exc).__name__,
            "message": str(exc), "calls": client.calls, "cutoff_s": graph.cutoff_s})
        raise


if __name__ == "__main__":
    main()
