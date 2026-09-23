"""Task-neutral symmetric arbitration across frozen-foundation hypotheses."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .mllm_orchestration import PluginEvidence


STATE_RANK = {"N": 0, "U": 1, "P": 2, "F": 3}


def _ablation_name(judgment: dict[str, Any]) -> str:
    name = str(judgment.get("ablation", {}).get("name", ""))
    if not name:
        raise ValueError("Foundation candidate has no ablation name")
    return name


def _frame_map(judgment: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = judgment.get("prediction", {}).get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Foundation candidate has no prediction frames")
    output: dict[int, dict[str, Any]] = {}
    for frame in frames:
        frame_id = int(frame.get("frame_index", -1))
        rows = frame.get("criteria")
        if frame_id in output or not isinstance(rows, list) or not rows:
            raise ValueError("Foundation candidate has invalid frame rows")
        states = [str(row.get("foundation_state", "")).upper() for row in rows]
        if any(state not in STATE_RANK for state in states):
            raise ValueError("Foundation candidate has invalid states")
        output[frame_id] = frame
    return output


def _criterion_ids(frame: dict[str, Any]) -> list[str]:
    rows = frame.get("criteria", [])
    ids = [str(row.get("criterion_id", "")) for row in rows]
    if not ids or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Foundation candidate has invalid criterion identities")
    return ids


def _fact_support_tier(fact: dict[str, Any]) -> str:
    annotation = fact.get("adjudication")
    if isinstance(annotation, dict) and annotation.get("reliability_tier") in {
        "calibrated_high", "supported", "context_only",
    }:
        return str(annotation["reliability_tier"])
    value = fact.get("value")
    if value == "high_reliability_candidate" or (
        isinstance(value, dict)
        and value.get("candidate_rank_band") == "high_reliability_candidate"
    ):
        return "calibrated_high"
    grounding = fact.get("grounding")
    grounding = grounding if isinstance(grounding, dict) else {}
    support = grounding.get(
        "supporting_observation_count", grounding.get("supporting_frame_count", 0),
    )
    try:
        if int(support or 0) >= 3:
            return "supported"
    except (TypeError, ValueError):
        pass
    return "context_only"


def build_plugin_coverage_manifests(
    plugins: Iterable[PluginEvidence], frame_ids: Iterable[int],
    timestamps_s: Iterable[float],
) -> list[dict[str, Any]]:
    """Describe label-free evidence applicability over the submitted timeline."""
    ids = [int(value) for value in frame_ids]
    times = [float(value) for value in timestamps_s]
    if not ids or len(ids) != len(times) or len(ids) != len(set(ids)):
        raise ValueError("Coverage manifests require aligned unique frames and timestamps")
    time_by_frame = dict(zip(ids, times))
    output = []
    for plugin in plugins:
        plugin.validate_fact_only()
        direct: set[int] = set()
        interval: set[int] = set()
        tiers = {"calibrated_high": 0, "supported": 0, "context_only": 0}
        facts = plugin.payload.get("facts", [])
        if not isinstance(facts, list):
            raise ValueError("Plugin facts must be a list")
        for fact in facts:
            if not isinstance(fact, dict):
                raise ValueError("Plugin facts must be objects")
            tiers[_fact_support_tier(fact)] += 1
            raw_frame = fact.get("frame_index")
            if raw_frame is not None and int(raw_frame) in time_by_frame:
                direct.add(int(raw_frame))
                continue
            timestamp = fact.get("timestamp_s")
            if timestamp is not None:
                for frame_id, frame_time in time_by_frame.items():
                    if abs(float(timestamp) - frame_time) <= 1e-3:
                        direct.add(frame_id)
            start, end = fact.get("start_s"), fact.get("end_s")
            if start is not None and end is not None:
                for frame_id, frame_time in time_by_frame.items():
                    if float(start) <= frame_time <= float(end):
                        interval.add(frame_id)
        covered = direct | interval
        output.append({
            "schema_version": "foundation_plugin_coverage_manifest_v1",
            "plugin_id": plugin.plugin_id,
            "plugin_kind": plugin.plugin_kind,
            "all_frame_ids": list(ids),
            "fact_count": len(facts),
            "tier_counts": tiers,
            "direct_localized_frame_ids": sorted(direct),
            "interval_covered_frame_ids": sorted(interval),
            "covered_frame_ids": sorted(covered),
            "uncovered_frame_ids": [value for value in ids if value not in covered],
            "coverage_is_evidence_applicability_not_task_truth": True,
            "final_task_prediction_provided": False,
            "labels_accessed": False,
        })
    return output


def _validate_candidates(
    candidates: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[int], dict[str, dict[int, dict[str, Any]]]]:
    rows = list(candidates)
    if len(rows) < 2:
        raise ValueError("Symmetric arbitration requires at least two foundation candidates")
    names = [_ablation_name(row) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Foundation candidates must use distinct evidence paths")
    maps = {name: _frame_map(row) for name, row in zip(names, rows)}
    frame_ids = list(maps[names[0]])
    reference_ids = _criterion_ids(maps[names[0]][frame_ids[0]])
    for name in names:
        if list(maps[name]) != frame_ids:
            raise ValueError("Foundation candidates have different frame identity/order")
        for frame_id in frame_ids:
            if _criterion_ids(maps[name][frame_id]) != reference_ids:
                raise ValueError("Foundation candidates have different criterion identity/order")
    return rows, names, frame_ids, maps


def select_multibranch_arbitration_frame_ids(
    candidates: Iterable[dict[str, Any]], plugins: Iterable[PluginEvidence],
    *, maximum_frames: int = 6,
    coverage_manifests: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Route three-way disagreements without labels or a stable-primary branch."""
    if maximum_frames < 1:
        raise ValueError("maximum_frames must be positive")
    _, names, frame_ids, maps = _validate_candidates(candidates)
    plugins = list(plugins)
    manifests = coverage_manifests
    if manifests is None:
        # Frame IDs are sufficient for exact-localized facts; interval facts need
        # timestamps and should be supplied through an explicit manifest.
        manifests = []
        for plugin in plugins:
            plugin.validate_fact_only()
    manifest_ids = {str(row.get("plugin_id")) for row in manifests}
    if manifests and manifest_ids != {plugin.plugin_id for plugin in plugins}:
        raise ValueError("Coverage manifests differ from the supplied plugins")
    covered = {
        frame_id: sum(frame_id in set(row.get("covered_frame_ids", [])) for row in manifests)
        for frame_id in frame_ids
    }
    direct = {
        frame_id: sum(
            frame_id in set(row.get("direct_localized_frame_ids", [])) for row in manifests
        ) for frame_id in frame_ids
    }
    disagreement_frames: set[int] = set()
    raw_priority: dict[int, tuple[int, int, int, int, int]] = {}
    for frame_id in frame_ids:
        state_rows = [
            [str(row.get("foundation_state", "")).upper()
             for row in maps[name][frame_id]["criteria"]]
            for name in names
        ]
        per_criterion = list(zip(*state_rows))
        three_way = sum(len(set(values)) >= 3 for values in per_criterion)
        disagreements = sum(len(set(values)) > 1 for values in per_criterion)
        full_disagreements = sum(
            any(value == "F" for value in values)
            and any(value != "F" for value in values)
            for values in per_criterion
        )
        maximum_gap = max(
            max(STATE_RANK[value] for value in values)
            - min(STATE_RANK[value] for value in values)
            for values in per_criterion
        )
        if disagreements:
            disagreement_frames.add(frame_id)
        raw_priority[frame_id] = (
            three_way, full_disagreements, disagreements,
            maximum_gap, direct[frame_id] * 2 + covered[frame_id],
        )
    priorities = []
    for frame_id in frame_ids:
        nearest = min(
            (abs(frame_id - other) for other in disagreement_frames), default=10**6,
        )
        neighbor_context = 2 if nearest == 1 else 1 if nearest == 2 else 0
        core = raw_priority[frame_id]
        priorities.append((
            (int(core[2] > 0), core[0], core[1], core[2], core[3], core[4], neighbor_context),
            frame_id,
        ))
    priorities.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [
        frame_id for priority, frame_id in priorities
        if priority[0] or priority[-1]
    ][:maximum_frames]
    return sorted(selected)


def slice_foundation_candidate(
    judgment: dict[str, Any], frame_ids: Iterable[int],
) -> dict[str, Any]:
    """Slice routed rows while preserving compact whole-timeline context."""
    selected = set(int(value) for value in frame_ids)
    output = deepcopy(judgment)
    frames = output.get("prediction", {}).get("frames", [])
    original_ids = [int(frame["frame_index"]) for frame in frames]
    if not selected.issubset(original_ids):
        raise ValueError("Cannot slice unavailable foundation frames")
    prediction = output["prediction"]
    output["multibranch_slice_context"] = {
        "source_ablation": _ablation_name(judgment),
        "original_frame_ids": original_ids,
        "original_frame_count": len(original_ids),
        "routed_frame_ids": sorted(selected),
        "whole_timeline_case_summary": str(prediction.get("case_summary", "")),
        "whole_timeline_plugin_assessment": deepcopy(
            prediction.get("plugin_assessment", [])
        ),
        "ground_truth_or_labels_included": False,
    }
    prediction["frames"] = [
        frame for frame in frames if int(frame["frame_index"]) in selected
    ]
    if {int(frame["frame_index"]) for frame in prediction["frames"]} != selected:
        raise ValueError("Sliced candidate does not cover every routed frame")
    return output


def filter_plugin_for_multibranch_arbitration(
    plugin: PluginEvidence, frame_ids: Iterable[int], timestamps_s: Iterable[float],
) -> PluginEvidence:
    """Retain only facts applicable to routed frames plus nonlocal summaries."""
    plugin.validate_fact_only()
    selected = set(int(value) for value in frame_ids)
    times = [float(value) for value in timestamps_s]
    output = deepcopy(plugin.payload)
    retained = []
    for fact in output.get("facts", []):
        frame_id = fact.get("frame_index")
        if frame_id is not None:
            if int(frame_id) in selected:
                retained.append(fact)
            continue
        timestamp = fact.get("timestamp_s")
        start, end = fact.get("start_s"), fact.get("end_s")
        if timestamp is not None and any(abs(float(timestamp) - value) <= 1e-3 for value in times):
            retained.append(fact)
        elif start is not None and end is not None and any(
            float(start) <= value <= float(end) for value in times
        ):
            retained.append(fact)
        elif timestamp is None and start is None and end is None:
            retained.append(fact)
    output["facts"] = retained
    output.setdefault("provenance", {})["multibranch_arbitration_filter"] = {
        "selected_frame_ids": sorted(selected),
        "selected_timestamps_s": times,
        "facts_retained": len(retained),
        "labels_accessed": False,
        "task_predictions_created": False,
    }
    return PluginEvidence(
        plugin.plugin_id, plugin.plugin_kind, plugin.description, output,
        plugin.foundation_model_parameters_updated,
    )


def _row_signature(frame: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple((
        str(row.get("criterion_id", "")),
        str(row.get("foundation_state", "")),
        str(row.get("foundation_confidence", "")),
        str(row.get("visibility", "")),
        float(row.get("probability_satisfied", -1.0)),
    ) for row in frame.get("criteria", []))


def merge_multibranch_arbitrated_judgment(
    candidates: Iterable[dict[str, Any]], arbitration: dict[str, Any] | None,
    frame_ids: Iterable[int], *, fallback_ablation: str = "mllm_skill",
) -> dict[str, Any]:
    """Merge final-Qwen rows with consensus or frozen-Qwen Skill fallback rows."""
    rows, names, all_frame_ids, maps = _validate_candidates(candidates)
    if fallback_ablation not in names:
        raise ValueError("Requested frozen-foundation fallback branch is unavailable")
    selected = set(int(value) for value in frame_ids)
    if not selected.issubset(all_frame_ids):
        raise ValueError("Routed frame IDs are unavailable in the candidates")
    final_by_id: dict[int, dict[str, Any]] = {}
    if selected:
        if arbitration is None:
            raise ValueError("Routed frames require a final foundation judgment")
        final_by_id = {
            int(frame["frame_index"]): frame
            for frame in arbitration.get("prediction", {}).get("frames", [])
        }
        if set(final_by_id) != selected:
            raise ValueError("Final judgment does not cover exactly the routed frames")
        output = deepcopy(arbitration)
    else:
        if arbitration is not None:
            raise ValueError("An arbitration judgment is invalid when no frame was routed")
        output = deepcopy(rows[names.index(fallback_ablation)])
    merged = []
    row_sources = []
    for frame_id in all_frame_ids:
        if frame_id in selected:
            merged.append(deepcopy(final_by_id[frame_id]))
            source = "same_frozen_qwen_final_multibranch_arbitration"
        else:
            signatures = {_row_signature(maps[name][frame_id]) for name in names}
            merged.append(deepcopy(maps[fallback_ablation][frame_id]))
            source = (
                "same_frozen_qwen_unanimous_branch_consensus"
                if len(signatures) == 1
                else "same_frozen_qwen_skill_budget_fallback"
            )
        row_sources.append({"frame_index": frame_id, "source": source})
    output.setdefault("prediction", {})["frames"] = merged
    copy_audit: dict[str, dict[str, Any]] = {}
    for name in names:
        state_equal = probability_equal = total = 0
        exact_rows = []
        for frame_id in sorted(selected):
            final_frame = final_by_id[frame_id]
            candidate_frame = maps[name][frame_id]
            exact_rows.append(_row_signature(final_frame) == _row_signature(candidate_frame))
            for final_row, candidate_row in zip(
                final_frame.get("criteria", []), candidate_frame.get("criteria", []),
            ):
                total += 1
                state_equal += (
                    final_row.get("foundation_state") == candidate_row.get("foundation_state")
                )
                probability_equal += abs(
                    float(final_row.get("probability_satisfied", -10.0))
                    - float(candidate_row.get("probability_satisfied", 10.0))
                ) <= 1e-12
        copy_audit[name] = {
            "state_equal_cells": int(state_equal),
            "probability_equal_cells": int(probability_equal),
            "routed_criterion_cells": int(total),
            "exact_row_equal_frame_ids": [
                frame_id for frame_id, equal in zip(sorted(selected), exact_rows) if equal
            ],
        }
    output["arbitration_merge"] = {
        "candidate_source_ablations": names,
        "candidates_are_symmetric_frozen_foundation_hypotheses": True,
        "routed_frame_ids": sorted(selected),
        "routed_frame_count": len(selected),
        "row_sources": row_sources,
        "nonrouted_disagreement_policy": "same_frozen_qwen_skill_budget_fallback",
        "routed_rows_source": "same_frozen_qwen_final_multibranch_arbitration",
        "branch_copy_audit": copy_audit,
        "small_model_rows_used_as_final_prediction": False,
        "labels_accessed": False,
    }
    return output
