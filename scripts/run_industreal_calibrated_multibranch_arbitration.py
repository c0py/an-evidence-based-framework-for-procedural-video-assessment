#!/usr/bin/env python3
"""Run complete-disagreement calibrated final-Qwen arbitration on IndustReal."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.foundation_calibrated_multibranch_arbitration import (  # noqa: E402
    build_calibrated_coverage_manifests,
    merge_complete_disagreement_judgment,
    select_all_disagreement_frame_ids,
)
from cvs_assessment.foundation_multibranch_arbitration import (  # noqa: E402
    filter_plugin_for_multibranch_arbitration,
    slice_foundation_candidate,
)
from cvs_assessment.mllm_calibrated_multibranch_orchestration import CalibratedEvidencePromptMixin  # noqa: E402
from cvs_assessment.mllm_multibranch_orchestration import MultibranchJudgeRequest  # noqa: E402
from cvs_assessment.mllm_orchestration import MLLMAblation  # noqa: E402
from scripts.run_industreal_foundation_centered_ablation import (  # noqa: E402
    IndustRealFrozenMLLMJudge, encode_jpeg, read_video_frames, video_index,
)
from scripts.run_industreal_foundation_multibranch_arbitration import (  # noqa: E402
    CANDIDATE_ABLATIONS, cached_video_ids, criterion_contracts_from_candidates,
    load_cached_case, source_audit,
)


FINAL_ABLATION = MLLMAblation(
    "full_framework_calibrated_multibranch", True, ("visual", "temporal"),
    decision_protocol="ordinal_state", require_fact_only_plugins=True,
)


class IndustRealCalibratedJudge(CalibratedEvidencePromptMixin, IndustRealFrozenMLLMJudge):
    """Calibrated generic prompt plus the frozen IndustReal output normalizer."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--maximum-arbitration-frames", type=int, default=18)
    parser.add_argument("--base-url", default="http://127.0.0.1:18904/v1")
    parser.add_argument("--model", default="qwen3-vl-32b-sop")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    available = cached_video_ids(args.candidate_dir)
    video_ids = args.video_id or available
    if not video_ids or any(video_id not in available for video_id in video_ids):
        raise ValueError("IndustReal calibrated cohort is incomplete")
    if args.output_dir.exists() and not args.resume: raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    videos = video_index(args.video_root); skill_text = args.skill.read_text()
    judge = IndustRealCalibratedJudge(args.base_url, args.model, args.api_key, args.timeout_s, max_tokens=args.max_tokens)
    records = []
    for video_id in video_ids:
        path = args.output_dir / f"{video_id}__full_framework_calibrated_multibranch.json"
        if path.exists() and args.resume:
            row = json.loads(path.read_text()); record = {"video_id": video_id, "status": "complete", "resumed": True, "routed_frame_ids": row["arbitration_merge"]["routed_frame_ids"]}
            records.append(record); print(json.dumps(record), flush=True); continue
        try:
            candidates, facts, plugins = load_cached_case(args.candidate_dir, video_id)
            if candidates[0]["model"] != args.model: raise ValueError("Candidate model differs")
            source_indices = [int(value) for value in facts["sample_source_frame_indices"]]
            timestamps = [float(value) for value in facts["timestamps_s"]]
            frame_ids = list(range(len(source_indices)))
            manifests = build_calibrated_coverage_manifests(plugins, frame_ids, timestamps)
            routed = select_all_disagreement_frame_ids(candidates, maximum_frames=args.maximum_arbitration_frames)
            arbitration = None
            if routed:
                raw = read_video_frames(videos[video_id], [source_indices[index] for index in routed])
                routed_times = [timestamps[index] for index in routed]
                request = MultibranchJudgeRequest(
                    task_id=str(candidates[0]["task_id"]), sample_id=video_id,
                    criteria=criterion_contracts_from_candidates(candidates), frame_ids=routed,
                    timestamps_s=routed_times,
                    frame_jpegs=[encode_jpeg(raw[source_indices[index]]) for index in routed],
                    skill_text=skill_text,
                    plugin_evidence=[filter_plugin_for_multibranch_arbitration(plugin, routed, routed_times) for plugin in plugins],
                    foundation_candidate_judgments=[slice_foundation_candidate(candidate, routed) for candidate in candidates],
                    plugin_coverage_manifests=manifests,
                )
                arbitration = judge.judge(request, FINAL_ABLATION)
            judgment = merge_complete_disagreement_judgment(candidates, arbitration, routed)
            judgment["ablation"] = asdict(FINAL_ABLATION)
            judgment["inference_stages"] = {
                "stage_1": "three_cached_same_frozen_qwen_evidence_path_hypotheses",
                "stage_2": "same_frozen_qwen_calibrated_complete_disagreement_arbitration" if routed else "no_call_exact_consensus_merge",
                "candidate_order": list(CANDIDATE_ABLATIONS), "arbitrated_frame_ids": routed,
                "maximum_arbitration_frames": args.maximum_arbitration_frames,
                "final_latency_s": 0.0 if arbitration is None else arbitration["runtime"]["latency_s"],
                "coverage_manifests": manifests, "source_audit": source_audit(args.candidate_dir, video_id),
                "labels_loaded_by_runner": False, "official_test_annotations_accessed": False,
                "foundation_model_parameters_updated": False, "small_model_is_final_judge": False,
            }
            path.write_text(json.dumps(judgment, indent=2) + "\n")
            record = {"video_id": video_id, "status": "complete", "resumed": False, "routed_frame_ids": routed, "latency_s": judgment["inference_stages"]["final_latency_s"]}
        except Exception as exc:
            error = args.output_dir / f"{video_id}__full_framework_calibrated_multibranch__ERROR.json"
            error.write_text(json.dumps({"video_id": video_id, "error_type": type(exc).__name__, "error": str(exc), "safe_to_resume": True, "labels_loaded_by_runner": False, "foundation_model_parameters_updated": False}, indent=2) + "\n")
            record = {"video_id": video_id, "status": "error", "error": str(exc)}
        records.append(record); print(json.dumps(record), flush=True)
    summary = {"schema_version": "industreal_calibrated_multibranch_summary_v1", "created_at": datetime.now(timezone.utc).isoformat(), "model": args.model, "video_ids": video_ids, "candidate_order": list(CANDIDATE_ABLATIONS), "records": records, "labels_loaded_by_runner": False, "foundation_model_parameters_updated": False}
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__": main()
