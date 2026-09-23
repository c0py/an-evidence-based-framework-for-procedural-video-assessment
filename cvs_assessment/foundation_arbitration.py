"""Task-neutral routing and merge utilities for frozen-foundation branch arbitration."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .mllm_orchestration import PluginEvidence


STATE_RANK = {"N": 0, "U": 1, "P": 2, "F": 3}


def _frame_states(judgment: dict[str, Any]) -> dict[int, list[str]]:
    output: dict[int, list[str]] = {}
    for frame in judgment.get("prediction", {}).get("frames", []):
        frame_id = int(frame["frame_index"])
        output[frame_id] = [
            str(row.get("foundation_state", "")).upper()
            for row in frame.get("criteria", [])
        ]
        if not output[frame_id] or any(state not in STATE_RANK for state in output[frame_id]):
            raise ValueError("Foundation branch has invalid or empty states")
    return output


def select_arbitration_frame_ids(
    primary: dict[str, Any], auxiliary: dict[str, Any],
    plugins: Iterable[PluginEvidence], *, maximum_frames: int = 6,
) -> list[int]:
    """Select disagreement/evidence frames without consulting task labels."""
    if maximum_frames < 1:
        raise ValueError("maximum_frames must be positive")
    primary_states = _frame_states(primary)
    auxiliary_states = _frame_states(auxiliary)
    if set(primary_states) != set(auxiliary_states):
        raise ValueError("Foundation branches have different frame sets")

    evidence_priority = {frame_id: 0 for frame_id in primary_states}
    for plugin in plugins:
        plugin.validate_fact_only()
        for fact in plugin.payload.get("facts", []):
            if not isinstance(fact, dict) or fact.get("frame_index") is None:
                continue
            frame_id = int(fact["frame_index"])
            if frame_id not in evidence_priority:
                continue
            text = str(fact.get("value", ""))
            grounding = fact.get("grounding", {})
            support = grounding.get(
                "supporting_observation_count", grounding.get("supporting_frame_count", 0),
            ) if isinstance(grounding, dict) else 0
            if "high_reliability_candidate" in text:
                evidence_priority[frame_id] = max(evidence_priority[frame_id], 3)
            elif support is not None and int(support) >= 3:
                evidence_priority[frame_id] = max(evidence_priority[frame_id], 2)
            else:
                evidence_priority[frame_id] = max(evidence_priority[frame_id], 1)

    full_upgrade_frames = {
        frame_id for frame_id in primary_states
        if any(
            left != "F" and right == "F"
            for left, right in zip(primary_states[frame_id], auxiliary_states[frame_id])
        )
    }
    priorities: list[tuple[tuple[int, int, int, int, int], int]] = []
    for frame_id in primary_states:
        left, right = primary_states[frame_id], auxiliary_states[frame_id]
        if len(left) != len(right):
            raise ValueError("Foundation branches have different criterion counts")
        temporal_full_upgrade = int(any(a != "F" and b == "F" for a, b in zip(left, right)))
        nearest_full_upgrade = min(
            (abs(frame_id - candidate) for candidate in full_upgrade_frames),
            default=10**6,
        )
        full_upgrade_neighbor = (
            2 if nearest_full_upgrade == 1 else 1 if nearest_full_upgrade == 2 else 0
        )
        any_full_disagreement = int(any((a == "F") != (b == "F") for a, b in zip(left, right)))
        maximum_gap = max(abs(STATE_RANK[a] - STATE_RANK[b]) for a, b in zip(left, right))
        priorities.append(((
            temporal_full_upgrade, full_upgrade_neighbor,
            any_full_disagreement, maximum_gap,
            evidence_priority[frame_id],
        ), frame_id))
    priorities.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    selected = [frame_id for priority, frame_id in priorities if any(priority)][:maximum_frames]
    if not selected:
        selected = [min(primary_states)]
    return sorted(selected)


def slice_foundation_judgment(
    judgment: dict[str, Any], frame_ids: Iterable[int],
) -> dict[str, Any]:
    selected = set(int(value) for value in frame_ids)
    output = deepcopy(judgment)
    frames = output.get("prediction", {}).get("frames", [])
    output["prediction"]["frames"] = [
        frame for frame in frames if int(frame["frame_index"]) in selected
    ]
    if {int(frame["frame_index"]) for frame in output["prediction"]["frames"]} != selected:
        raise ValueError("Cannot slice unavailable foundation frames")
    return output


def filter_plugin_for_arbitration(
    plugin: PluginEvidence, frame_ids: Iterable[int], timestamps_s: Iterable[float],
) -> PluginEvidence:
    """Keep facts tied to selected frames/times plus nonlocal summaries."""
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
    output.setdefault("provenance", {})["arbitration_filter"] = {
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


def merge_arbitrated_judgment(
    primary: dict[str, Any], arbitration: dict[str, Any], frame_ids: Iterable[int],
) -> dict[str, Any]:
    """Use foundation-final rows only on routed frames; retain foundation-primary rows elsewhere."""
    selected = set(int(value) for value in frame_ids)
    output = deepcopy(arbitration)
    final_by_id = {
        int(frame["frame_index"]): frame
        for frame in arbitration.get("prediction", {}).get("frames", [])
    }
    if set(final_by_id) != selected:
        raise ValueError("Arbitration judgment does not cover exactly the routed frames")
    merged = []
    for frame in primary.get("prediction", {}).get("frames", []):
        frame_id = int(frame["frame_index"])
        merged.append(deepcopy(final_by_id.get(frame_id, frame)))
    output["prediction"]["frames"] = merged
    output["arbitration_merge"] = {
        "routed_frame_ids": sorted(selected),
        "routed_frame_count": len(selected),
        "nonrouted_rows_source": "same_frozen_qwen_skill_primary",
        "routed_rows_source": "same_frozen_qwen_final_arbitration",
        "small_model_rows_used_as_final_prediction": False,
        "labels_accessed": False,
    }
    return output
