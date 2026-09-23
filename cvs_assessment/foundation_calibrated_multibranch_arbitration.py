"""Complete-disagreement routing for calibrated-evidence-preserving arbitration."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .foundation_multibranch_arbitration import (
    _fact_support_tier,
    _row_signature,
    _validate_candidates,
    build_plugin_coverage_manifests,
    merge_multibranch_arbitrated_judgment,
)
from .mllm_orchestration import PluginEvidence


def build_calibrated_coverage_manifests(
    plugins: Iterable[PluginEvidence], frame_ids: Iterable[int],
    timestamps_s: Iterable[float],
) -> list[dict[str, Any]]:
    plugins = list(plugins)
    ids = [int(value) for value in frame_ids]
    times = [float(value) for value in timestamps_s]
    manifests = build_plugin_coverage_manifests(plugins, ids, times)
    time_by_frame = dict(zip(ids, times))
    for plugin, manifest in zip(plugins, manifests):
        support = {frame_id: [] for frame_id in ids}
        for fact in plugin.payload.get("facts", []):
            tier = _fact_support_tier(fact)
            applicable = set()
            if fact.get("frame_index") is not None and int(fact["frame_index"]) in support:
                applicable.add(int(fact["frame_index"]))
            if fact.get("timestamp_s") is not None:
                applicable.update(
                    frame_id for frame_id, timestamp in time_by_frame.items()
                    if abs(timestamp - float(fact["timestamp_s"])) <= 1e-3
                )
            if fact.get("start_s") is not None and fact.get("end_s") is not None:
                applicable.update(
                    frame_id for frame_id, timestamp in time_by_frame.items()
                    if float(fact["start_s"]) <= timestamp <= float(fact["end_s"])
                )
            for frame_id in applicable:
                support[frame_id].append({
                    "fact_id": str(fact.get("fact_id", "")), "reliability_tier": tier,
                    "localization": "direct" if fact.get("frame_index") is not None or fact.get("timestamp_s") is not None else "interval",
                })
        # Keep the validated v1 envelope; the additive contract version binds
        # the richer per-frame reliability register.
        manifest["calibrated_contract_schema_version"] = "foundation_plugin_coverage_manifest_v2"
        manifest["frame_support"] = [
            {"frame_index": frame_id, "applicable_facts": support[frame_id]}
            for frame_id in ids
        ]
        manifest["calibrated_high_is_reliable_observation_not_final_task_truth"] = True
    return manifests


def select_all_disagreement_frame_ids(
    candidates: Iterable[dict[str, Any]], *, maximum_frames: int = 18,
) -> list[int]:
    """Route every frame with any complete-Qwen row difference."""
    _, names, frame_ids, maps = _validate_candidates(candidates)
    selected = [
        frame_id for frame_id in frame_ids
        if len({_row_signature(maps[name][frame_id]) for name in names}) > 1
    ]
    if len(selected) > maximum_frames:
        raise ValueError(
            f"Complete disagreement coverage needs {len(selected)} frames, limit is {maximum_frames}"
        )
    return selected


def merge_complete_disagreement_judgment(
    candidates: Iterable[dict[str, Any]], arbitration: dict[str, Any] | None,
    frame_ids: Iterable[int],
) -> dict[str, Any]:
    candidates = list(candidates)
    output = merge_multibranch_arbitrated_judgment(candidates, arbitration, frame_ids)
    fallbacks = [
        row for row in output["arbitration_merge"]["row_sources"]
        if row["source"] == "same_frozen_qwen_skill_budget_fallback"
    ]
    if fallbacks:
        raise RuntimeError("Complete-disagreement arbitration cannot retain a primary fallback")
    output["arbitration_merge"]["complete_candidate_disagreement_coverage"] = True
    output["arbitration_merge"]["stable_primary_candidate"] = None
    return output
