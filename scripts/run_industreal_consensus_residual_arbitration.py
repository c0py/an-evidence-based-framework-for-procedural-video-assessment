#!/usr/bin/env python3
"""Run frozen-Qwen consensus-prior bounded-residual arbitration on IndustReal."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.foundation_calibrated_multibranch_arbitration import build_calibrated_coverage_manifests  # noqa: E402
from cvs_assessment.foundation_consensus_residual_arbitration import build_consensus_prior, merge_consensus_residual_judgment, select_consensus_residual_frame_ids  # noqa: E402
from cvs_assessment.foundation_multibranch_arbitration import filter_plugin_for_multibranch_arbitration, slice_foundation_candidate  # noqa: E402
from cvs_assessment.mllm_consensus_residual_orchestration import ConsensusResidualJudgeRequest, ConsensusResidualPromptMixin  # noqa: E402
from cvs_assessment.mllm_orchestration import MLLMAblation  # noqa: E402
from scripts.run_industreal_foundation_centered_ablation import IndustRealFrozenMLLMJudge, encode_jpeg, read_video_frames, sha256_file, video_index  # noqa: E402
from scripts.run_industreal_foundation_multibranch_arbitration import criterion_contracts_from_candidates, load_cached_case  # noqa: E402


FINAL_ABLATION = MLLMAblation(
    "full_framework_consensus_residual", True, ("visual", "temporal"),
    decision_protocol="ordinal_state", require_fact_only_plugins=True,
)


class IndustRealConsensusResidualJudge(ConsensusResidualPromptMixin, IndustRealFrozenMLLMJudge):
    """Generic residual contract plus existing optional-output normalization."""


def _ids(candidate_dir: Path, calibrated_dir: Path) -> list[str]:
    suffix = "__mllm_skill_visual.json"
    return sorted(
        path.name[:-len(suffix)] for path in candidate_dir.glob(f"*{suffix}")
        if (candidate_dir / f"{path.name[:-len(suffix)]}__grounded_facts.json").is_file()
        and (calibrated_dir / f"{path.name[:-len(suffix)]}__full_framework_calibrated_multibranch.json").is_file()
    )


def _load_calibrated(path: Path, video_id: str, model: str) -> dict:
    row = json.loads(path.read_text())
    if (row.get("sample_id") != video_id or row.get("model") != model
            or row.get("ablation", {}).get("name") != "full_framework_calibrated_multibranch"
            or row.get("foundation_model_parameters_updated") is not False):
        raise ValueError(f"Invalid calibrated frozen-Qwen candidate: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--calibrated-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skill", type=Path, default=ROOT / "specs/industreal_psr_assembly_skill.md")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--maximum-arbitration-frames", type=int, default=18)
    parser.add_argument("--base-url", default="http://127.0.0.1:18904/v1")
    parser.add_argument("--model", default="qwen3-vl-32b-sop")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    available = _ids(args.candidate_dir, args.calibrated_dir); video_ids = args.video_id or available
    if not video_ids or len(video_ids) != len(set(video_ids)) or any(x not in available for x in video_ids):
        raise ValueError("IndustReal consensus-residual cohort is incomplete or duplicated")
    if args.output_dir.exists() and not args.resume: raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    videos = video_index(args.video_root); skill_text = args.skill.read_text()
    judge = IndustRealConsensusResidualJudge(args.base_url, args.model, args.api_key, args.timeout_s, max_tokens=args.max_tokens)
    records = []
    for video_id in video_ids:
        path = args.output_dir / f"{video_id}__full_framework_consensus_residual.json"
        error = args.output_dir / f"{video_id}__full_framework_consensus_residual__ERROR.json"
        if path.exists() and args.resume:
            row = json.loads(path.read_text()); record = {"video_id": video_id, "status": "complete", "resumed": True,
                "routed_frame_ids": row["consensus_residual_merge"]["routed_frame_ids"]}
            records.append(record); print(json.dumps(record), flush=True); continue
        try:
            cached, facts, plugins = load_cached_case(args.candidate_dir, video_id)
            visual = cached[1]
            calibrated_path = args.calibrated_dir / f"{video_id}__full_framework_calibrated_multibranch.json"
            candidates = [visual, _load_calibrated(calibrated_path, video_id, args.model)]
            source_indices = [int(value) for value in facts["sample_source_frame_indices"]]
            timestamps = [float(value) for value in facts["timestamps_s"]]
            frame_ids = list(range(len(source_indices)))
            manifests = build_calibrated_coverage_manifests(plugins, frame_ids, timestamps)
            routed = select_consensus_residual_frame_ids(candidates, maximum_frames=args.maximum_arbitration_frames)
            arbitration = None
            if routed:
                raw = read_video_frames(videos[video_id], [source_indices[index] for index in routed])
                routed_times = [timestamps[index] for index in routed]
                sliced = [slice_foundation_candidate(candidate, routed) for candidate in candidates]
                request = ConsensusResidualJudgeRequest(
                    task_id=str(visual["task_id"]), sample_id=video_id,
                    criteria=criterion_contracts_from_candidates(candidates), frame_ids=routed,
                    timestamps_s=routed_times,
                    frame_jpegs=[encode_jpeg(raw[source_indices[index]]) for index in routed],
                    skill_text=skill_text,
                    plugin_evidence=[filter_plugin_for_multibranch_arbitration(plugin, routed, routed_times) for plugin in plugins],
                    foundation_candidate_judgments=sliced, plugin_coverage_manifests=manifests,
                    consensus_prior=build_consensus_prior(sliced),
                )
                arbitration = judge.judge(request, FINAL_ABLATION)
            judgment = merge_consensus_residual_judgment(candidates, arbitration, routed)
            judgment["ablation"] = asdict(FINAL_ABLATION)
            visual_path = args.candidate_dir / f"{video_id}__mllm_skill_visual.json"
            fact_path = args.candidate_dir / f"{video_id}__grounded_facts.json"
            sources = {"visual_qwen": visual_path, "calibrated_full_qwen": calibrated_path, "grounded_facts": fact_path,
                       "source_video": videos[video_id]}
            judgment["inference_stages"] = {
                "stage_1": "two_cached_complete_same_frozen_qwen_role_hypotheses",
                "stage_2": "label_free_qwen_consensus_prior",
                "stage_3": "same_frozen_qwen_bounded_residual_semantic_arbitration" if routed else "no_call_identical_qwen_consensus",
                "arbitrated_frame_ids": routed, "residual_bound": .1,
                "final_latency_s": 0.0 if arbitration is None else arbitration["runtime"]["latency_s"],
                "coverage_manifests": manifests,
                "source_audit": {name: {"path": str(p.resolve()), "sha256": sha256_file(p), "size_bytes": p.stat().st_size} for name, p in sources.items()},
                "labels_loaded_by_runner": False, "official_test_annotations_accessed": False,
                "foundation_model_parameters_updated": False, "small_model_is_final_judge": False,
            }
            path.write_text(json.dumps(judgment, indent=2) + "\n")
            if error.exists(): error.unlink()
            record = {"video_id": video_id, "status": "complete", "resumed": False,
                      "routed_frame_ids": routed, "latency_s": judgment["inference_stages"]["final_latency_s"]}
        except Exception as exc:
            error.write_text(json.dumps({"video_id": video_id, "error_type": type(exc).__name__, "error": str(exc),
                                         "safe_to_resume": True, "labels_loaded_by_runner": False,
                                         "foundation_model_parameters_updated": False}, indent=2) + "\n")
            record = {"video_id": video_id, "status": "error", "error": str(exc)}
        records.append(record); print(json.dumps(record), flush=True)
    summary = {"schema_version": "industreal_consensus_residual_summary_v1",
               "created_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
               "video_ids": video_ids, "records": records, "labels_loaded_by_runner": False,
               "foundation_model_parameters_updated": False}
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
