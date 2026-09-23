#!/usr/bin/env python3
"""Re-arbitrate cached IndustReal frozen-Qwen branches without reading labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.foundation_multibranch_arbitration import (  # noqa: E402
    build_plugin_coverage_manifests,
    filter_plugin_for_multibranch_arbitration,
    merge_multibranch_arbitrated_judgment,
    select_multibranch_arbitration_frame_ids,
    slice_foundation_candidate,
)
from cvs_assessment.mllm_multibranch_orchestration import (  # noqa: E402
    MultibranchJudgeRequest,
    SymmetricMultibranchPromptMixin,
)
from cvs_assessment.mllm_orchestration import (  # noqa: E402
    CriterionContract,
    MLLMAblation,
    PluginEvidence,
)
from scripts.run_industreal_foundation_centered_ablation import (  # noqa: E402
    IndustRealFrozenMLLMJudge,
    encode_jpeg,
    read_video_frames,
    sha256_file,
    video_index,
)


CANDIDATE_ABLATIONS = (
    "mllm_skill", "mllm_skill_visual", "mllm_skill_temporal",
)
FINAL_ABLATION = MLLMAblation(
    "full_framework_multibranch", True, ("visual", "temporal"),
    decision_protocol="ordinal_state", require_fact_only_plugins=True,
)


class IndustRealSymmetricMultibranchJudge(
    SymmetricMultibranchPromptMixin, IndustRealFrozenMLLMJudge,
):
    """Keep IndustReal's optional-output normalization in the symmetric judge."""


def criterion_contracts_from_candidates(
    candidates: list[dict[str, Any]],
) -> list[CriterionContract]:
    """Recover task definitions from branch identities, never from annotations."""
    if not candidates:
        raise ValueError("At least one candidate is required")
    frame_lists = [row.get("prediction", {}).get("frames", []) for row in candidates]
    if any(not frames for frames in frame_lists):
        raise ValueError("Candidate has no frames")
    reference = [
        str(row.get("criterion_id", ""))
        for row in frame_lists[0][0].get("criteria", [])
    ]
    if not reference or len(reference) != len(set(reference)):
        raise ValueError("Candidate criterion order is invalid")
    for frames in frame_lists:
        for frame in frames:
            observed = [str(row.get("criterion_id", "")) for row in frame.get("criteria", [])]
            if observed != reference:
                raise ValueError("Candidate criterion order differs")
    output = []
    pattern = re.compile(r"^action_(\d+)_([a-z0-9_]+)_(installed|removed)$")
    for criterion_id in reference:
        match = pattern.fullmatch(criterion_id)
        if not match:
            raise ValueError(f"Cannot derive label-free criterion contract: {criterion_id}")
        component = match.group(2).replace("_", " ")
        operation = match.group(3)
        output.append(CriterionContract(
            criterion_id=criterion_id,
            title=f"{operation.title()} {component}",
            minimal_description=(
                f"The {component} has been correctly {operation}; an incorrect part or "
                "orientation, or a merely attempted action, is not full completion."
            ),
        ))
    return output


def load_cached_case(
    candidate_dir: Path, video_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[PluginEvidence]]:
    candidates = []
    for name in CANDIDATE_ABLATIONS:
        path = candidate_dir / f"{video_id}__{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text())
        if row.get("sample_id") != video_id or row.get("ablation", {}).get("name") != name:
            raise ValueError(f"Cached candidate identity differs: {path}")
        if row.get("foundation_model_parameters_updated") is not False:
            raise ValueError(f"Cached candidate did not use a frozen foundation: {path}")
        candidates.append(row)
    task_ids = {str(row.get("task_id", "")) for row in candidates}
    models = {str(row.get("model", "")) for row in candidates}
    if len(task_ids) != 1 or "" in task_ids or len(models) != 1 or "" in models:
        raise ValueError("Cached candidates differ in task or foundation model")

    fact_path = candidate_dir / f"{video_id}__grounded_facts.json"
    if not fact_path.is_file():
        raise FileNotFoundError(fact_path)
    facts = json.loads(fact_path.read_text())
    # The v2 label-blind base runner uses the more explicit
    # ``official_test_psr_or_state_labels_accessed`` name, while the original
    # downstream runner used ``official_test_annotations_accessed``.  They are
    # equivalent negative safety assertions.  Require at least one and reject
    # the payload if any assertion that is present is not exactly False.
    safety_keys = (
        "official_test_annotations_accessed",
        "official_test_psr_or_state_labels_accessed",
    )
    present_safety = [key for key in safety_keys if key in facts]
    if (
        facts.get("recording_id") != video_id
        or not present_safety
        or any(facts.get(key) is not False for key in present_safety)
        or facts.get("labels_loaded_by_runner", False) is not False
        or facts.get("final_task_prediction_provided_by_plugins") is not False
    ):
        raise ValueError(f"Cached fact safety contract differs: {fact_path}")
    routed = facts.get("decision_routed_plugins")
    if not isinstance(routed, dict) or not routed:
        raise ValueError(f"Cached routed plugins are unavailable: {fact_path}")
    plugins = []
    for plugin_id, payload in routed.items():
        if not isinstance(payload, dict):
            raise ValueError("Cached plugin payload must be an object")
        kind = str(payload.get("modality", ""))
        if kind not in {"visual", "temporal"}:
            raise ValueError(f"Unexpected cached plugin modality: {kind}")
        plugins.append(PluginEvidence(
            str(plugin_id), kind,
            str(payload.get("source_description", f"Fallible {kind} observations")),
            payload,
        ))
    plugins.sort(key=lambda row: (row.plugin_kind != "visual", row.plugin_id))
    if {row.plugin_kind for row in plugins} != {"visual", "temporal"}:
        raise ValueError("Both visual and temporal fact-only plugins are required")
    return candidates, facts, plugins


def cached_video_ids(candidate_dir: Path) -> list[str]:
    suffix = "__mllm_skill.json"
    skill_ids = {
        path.name[:-len(suffix)] for path in candidate_dir.glob(f"*{suffix}")
    }
    return sorted(video_id for video_id in skill_ids if all(
        (candidate_dir / f"{video_id}__{name}.json").is_file()
        for name in CANDIDATE_ABLATIONS
    ) and (candidate_dir / f"{video_id}__grounded_facts.json").is_file())


def source_audit(candidate_dir: Path, video_id: str) -> dict[str, Any]:
    paths = {
        name: candidate_dir / f"{video_id}__{name}.json"
        for name in CANDIDATE_ABLATIONS
    }
    paths["grounded_facts"] = candidate_dir / f"{video_id}__grounded_facts.json"
    return {
        key: {
            "path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skill", type=Path, default=ROOT / "specs/industreal_psr_assembly_skill.md")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--maximum-arbitration-frames", type=int, default=6)
    parser.add_argument("--base-url", default="http://127.0.0.1:18904/v1")
    parser.add_argument("--model", default="qwen3-vl-32b-sop")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=float, default=1200)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.maximum_arbitration_frames < 1 or args.max_tokens < 1:
        raise ValueError("Arbitration frame/token limits must be positive")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    available_ids = cached_video_ids(args.candidate_dir)
    video_ids = list(args.video_id) if args.video_id else available_ids
    if not video_ids or len(video_ids) != len(set(video_ids)):
        raise ValueError("Video cohort must be nonempty and unique")
    if any(video_id not in available_ids for video_id in video_ids):
        raise FileNotFoundError("One or more requested videos lack complete cached branches/facts")
    videos = video_index(args.video_root)
    if any(video_id not in videos for video_id in video_ids):
        raise FileNotFoundError("One or more requested source videos are missing")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    skill_text = args.skill.read_text()
    judge = IndustRealSymmetricMultibranchJudge(
        args.base_url, args.model, args.api_key, args.timeout_s,
        max_tokens=args.max_tokens,
    )
    records = []
    for video_id in video_ids:
        output_path = args.output_dir / f"{video_id}__full_framework_multibranch.json"
        if output_path.exists() and args.resume:
            judgment = json.loads(output_path.read_text())
            records.append({
                "video_id": video_id, "status": "complete", "resumed": True,
                "output": str(output_path.resolve()),
                "routed_frame_ids": judgment.get("arbitration_merge", {}).get(
                    "routed_frame_ids", []
                ),
            })
            print(json.dumps(records[-1]), flush=True)
            continue
        try:
            candidates, facts, plugins = load_cached_case(args.candidate_dir, video_id)
            model = str(candidates[0]["model"])
            if model != args.model:
                raise ValueError(f"Cached model {model!r} differs from requested model {args.model!r}")
            task_id = str(candidates[0]["task_id"])
            criteria = criterion_contracts_from_candidates(candidates)
            source_indices = [int(value) for value in facts["sample_source_frame_indices"]]
            timestamps_s = [float(value) for value in facts["timestamps_s"]]
            frame_ids = list(range(len(source_indices)))
            if not source_indices or len(source_indices) != len(timestamps_s):
                raise ValueError("Cached sampled frame identities/timestamps differ")
            candidate_frame_ids = [
                int(frame["frame_index"])
                for frame in candidates[0]["prediction"]["frames"]
            ]
            if candidate_frame_ids != frame_ids:
                raise ValueError("Cached candidates do not match the sampled frame timeline")
            manifests = build_plugin_coverage_manifests(plugins, frame_ids, timestamps_s)
            routed_frame_ids = select_multibranch_arbitration_frame_ids(
                candidates, plugins,
                maximum_frames=args.maximum_arbitration_frames,
                coverage_manifests=manifests,
            )
            arbitration = None
            if routed_frame_ids:
                raw_map = read_video_frames(
                    videos[video_id], [source_indices[index] for index in routed_frame_ids],
                )
                request_timestamps = [timestamps_s[index] for index in routed_frame_ids]
                request_plugins = [
                    filter_plugin_for_multibranch_arbitration(
                        plugin, routed_frame_ids, request_timestamps,
                    ) for plugin in plugins
                ]
                request = MultibranchJudgeRequest(
                    task_id=task_id, sample_id=video_id, criteria=criteria,
                    frame_ids=routed_frame_ids, timestamps_s=request_timestamps,
                    frame_jpegs=[
                        encode_jpeg(raw_map[source_indices[index]])
                        for index in routed_frame_ids
                    ],
                    skill_text=skill_text, plugin_evidence=request_plugins,
                    foundation_candidate_judgments=[
                        slice_foundation_candidate(candidate, routed_frame_ids)
                        for candidate in candidates
                    ],
                    plugin_coverage_manifests=manifests,
                )
                arbitration = judge.judge(request, FINAL_ABLATION)
            judgment = merge_multibranch_arbitrated_judgment(
                candidates, arbitration, routed_frame_ids,
            )
            final_latency = float(
                arbitration.get("runtime", {}).get("latency_s", 0.0)
                if arbitration is not None else 0.0
            )
            judgment["inference_stages"] = {
                "stage_1": "three_cached_same_frozen_qwen_evidence_path_hypotheses",
                "stage_2": (
                    "same_frozen_qwen_symmetric_multibranch_arbitration"
                    if routed_frame_ids else "no_call_exact_consensus_merge"
                ),
                "candidate_order": list(CANDIDATE_ABLATIONS),
                "candidate_latencies_s": {
                    row["ablation"]["name"]: row.get("runtime", {}).get("latency_s")
                    for row in candidates
                },
                "final_latency_s": final_latency,
                "new_inference_latency_s": final_latency,
                "arbitrated_frame_ids": routed_frame_ids,
                "maximum_arbitration_frames": args.maximum_arbitration_frames,
                "coverage_manifests": manifests,
                "source_audit": source_audit(args.candidate_dir, video_id),
                "source_video_path": str(videos[video_id].resolve()),
                "labels_loaded_by_runner": False,
                "official_test_annotations_accessed": False,
                "foundation_model_parameters_updated": False,
                "small_model_is_final_judge": False,
            }
            output_path.write_text(json.dumps(judgment, indent=2) + "\n")
            error_path = args.output_dir / f"{video_id}__full_framework_multibranch__ERROR.json"
            if error_path.exists():
                error_path.rename(error_path.with_suffix(".superseded.json"))
            record = {
                "video_id": video_id, "status": "complete", "resumed": False,
                "output": str(output_path.resolve()),
                "routed_frame_ids": routed_frame_ids,
                "new_inference_latency_s": final_latency,
            }
        except Exception as exc:
            error_path = args.output_dir / f"{video_id}__full_framework_multibranch__ERROR.json"
            error_path.write_text(json.dumps({
                "schema_version": "industreal_multibranch_request_error_v1",
                "video_id": video_id, "error_type": type(exc).__name__,
                "error": str(exc), "safe_to_resume": True,
                "labels_loaded_by_runner": False,
                "official_test_annotations_accessed": False,
                "foundation_model_parameters_updated": False,
            }, indent=2) + "\n")
            record = {
                "video_id": video_id, "status": "error", "resumed": False,
                "error": str(exc), "error_path": str(error_path.resolve()),
            }
        records.append(record)
        print(json.dumps(record), flush=True)

    complete_ids = [row["video_id"] for row in records if row["status"] == "complete"]
    summary = {
        "schema_version": "industreal_symmetric_multibranch_run_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "video_ids": video_ids,
        "complete_video_ids": complete_ids,
        "error_video_ids": [row["video_id"] for row in records if row["status"] == "error"],
        "candidate_order": list(CANDIDATE_ABLATIONS),
        "final_ablation": FINAL_ABLATION.name,
        "maximum_arbitration_frames": args.maximum_arbitration_frames,
        "architecture_contract": {
            "foundation_mllm_is_common_base": True,
            "foundation_mllm_is_final_judge": True,
            "foundation_mllm_finetuned": False,
            "candidate_hypotheses_are_symmetric": True,
            "stable_primary_candidate": None,
            "visual_and_temporal_plugins_are_fact_only": True,
            "plugins_provide_final_task_verdict": False,
            "labels_loaded_by_runner": False,
            "small_model_is_final_judge": False,
        },
        "records": records,
    }
    summary_path = args.output_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
