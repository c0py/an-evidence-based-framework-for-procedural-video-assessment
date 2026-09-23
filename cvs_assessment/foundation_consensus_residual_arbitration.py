"""Consensus-prior and bounded-residual fusion of frozen-foundation judgments."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .foundation_multibranch_arbitration import _row_signature, _validate_candidates


ROLE_NAMES = (
    "mllm_skill_visual",
    "full_framework_calibrated_multibranch",
)


def _criterion_rows(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["criterion_id"]): row for row in frame["criteria"]}


def build_consensus_prior(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Average two complete frozen-Qwen hypotheses, never plugin predictions."""
    rows, names, frame_ids, maps = _validate_candidates(candidates)
    if tuple(names) != ROLE_NAMES or len(rows) != 2:
        raise ValueError(
            "Consensus prior requires visual-Qwen then calibrated-full-Qwen roles"
        )
    criterion_ids = [
        str(row["criterion_id"]) for row in maps[names[0]][frame_ids[0]]["criteria"]
    ]
    frames = []
    for frame_id in frame_ids:
        source_rows = {
            name: _criterion_rows(maps[name][frame_id]) for name in names
        }
        priors = []
        for criterion_id in criterion_ids:
            values = [
                float(source_rows[name][criterion_id]["probability_satisfied"])
                for name in names
            ]
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("Foundation candidate probability must be in [0,1]")
            priors.append({
                "criterion_id": criterion_id,
                "source_probabilities": dict(zip(names, values)),
                "consensus_probability": sum(values) / 2.0,
            })
        frames.append({"frame_index": frame_id, "criteria": priors})
    return {
        "schema_version": "frozen_qwen_consensus_prior_v1",
        "source_roles": list(names),
        "aggregation": "unweighted_arithmetic_mean_of_two_complete_frozen_qwen_probabilities",
        "frames": frames,
        "raw_small_model_rows_used": False,
        "labels_accessed": False,
        "foundation_model_parameters_updated": False,
    }


def select_consensus_residual_frame_ids(
    candidates: Iterable[dict[str, Any]], *, maximum_frames: int = 18,
) -> list[int]:
    """Route every frame on which the two complete Qwen hypotheses differ."""
    _, names, frame_ids, maps = _validate_candidates(candidates)
    if tuple(names) != ROLE_NAMES:
        raise ValueError("Unexpected consensus-residual candidate roles/order")
    selected = [
        frame_id for frame_id in frame_ids
        if _row_signature(maps[names[0]][frame_id])
        != _row_signature(maps[names[1]][frame_id])
    ]
    if len(selected) > maximum_frames:
        raise ValueError(
            f"Consensus residual requires {len(selected)} frames, limit is {maximum_frames}"
        )
    return selected


def _state_from_probability(probability: float) -> str:
    if probability >= 0.60:
        return "F"
    if probability >= 0.35:
        return "P"
    return "N"


def merge_consensus_residual_judgment(
    candidates: Iterable[dict[str, Any]],
    arbitration: dict[str, Any] | None,
    frame_ids: Iterable[int],
) -> dict[str, Any]:
    """Apply Qwen-authored +/-0.1 residuals to a frozen-Qwen consensus prior."""
    candidates = list(candidates)
    rows, names, all_frame_ids, maps = _validate_candidates(candidates)
    if tuple(names) != ROLE_NAMES or len(rows) != 2:
        raise ValueError("Unexpected consensus-residual candidate roles/order")
    selected = set(int(value) for value in frame_ids)
    if not selected.issubset(all_frame_ids):
        raise ValueError("Routed frame IDs are unavailable")
    prior = build_consensus_prior(candidates)
    prior_by_frame = {
        int(frame["frame_index"]): {
            str(row["criterion_id"]): row for row in frame["criteria"]
        } for frame in prior["frames"]
    }
    if selected:
        if arbitration is None:
            raise ValueError("Routed frames require a residual Qwen judgment")
        arbitration_frames = {
            int(frame["frame_index"]): frame
            for frame in arbitration.get("prediction", {}).get("frames", [])
        }
        if set(arbitration_frames) != selected:
            raise ValueError("Residual judgment does not exactly cover routed frames")
        dispositions = arbitration.get("prediction", {}).get("residual_dispositions")
        if not isinstance(dispositions, list):
            raise ValueError("Residual judgment lacks validated dispositions")
        disposition_map = {
            (int(row["frame_index"]), str(row["criterion_id"])): row
            for row in dispositions
        }
        output = deepcopy(arbitration)
    else:
        if arbitration is not None:
            raise ValueError("No residual judgment is allowed without routed frames")
        arbitration_frames, disposition_map = {}, {}
        output = deepcopy(rows[1])

    merged_frames, cell_audit = [], []
    for frame_id in all_frame_ids:
        reference = maps[names[1]][frame_id]
        qwen_rows = (
            _criterion_rows(arbitration_frames[frame_id]) if frame_id in selected else {}
        )
        merged_rows = []
        for reference_row in reference["criteria"]:
            criterion_id = str(reference_row["criterion_id"])
            prior_row = prior_by_frame[frame_id][criterion_id]
            consensus = float(prior_row["consensus_probability"])
            if frame_id in selected:
                disposition = disposition_map.get((frame_id, criterion_id))
                action = "hold" if disposition is None else str(disposition["action"])
                step = 0.0 if disposition is None else float(disposition["residual_step"])
                direction = 1.0 if action == "upgrade" else -1.0 if action == "downgrade" else 0.0
                probability = min(1.0, max(0.0, consensus + direction * step))
                result_row = deepcopy(qwen_rows[criterion_id])
                source = "same_frozen_qwen_bounded_residual"
            else:
                action, step, probability = "hold", 0.0, consensus
                result_row = deepcopy(reference_row)
                source = "identical_complete_frozen_qwen_consensus"
            result_row.update({
                "probability_satisfied": probability,
                "foundation_state": _state_from_probability(probability),
                "decision_action": action,
                "correction_magnitude": step,
                "prior_probability": consensus,
            })
            merged_rows.append(result_row)
            cell_audit.append({
                "frame_index": frame_id,
                "criterion_id": criterion_id,
                "source_probabilities": prior_row["source_probabilities"],
                "consensus_probability": consensus,
                "residual_action": action,
                "residual_step": step,
                "final_probability": probability,
                "final_row_source": source,
            })
        merged_frames.append({
            "frame_index": frame_id,
            "timestamp_s": float(reference.get("timestamp_s", 0.0)),
            "criteria": merged_rows,
        })
    output.setdefault("prediction", {})["frames"] = merged_frames
    output["consensus_residual_merge"] = {
        "schema_version": "frozen_qwen_consensus_residual_merge_v1",
        "source_roles": list(names),
        "routed_frame_ids": sorted(selected),
        "consensus_prior": prior,
        "cell_audit": cell_audit,
        "residual_bound": 0.10,
        "final_semantic_judge": "same_frozen_qwen",
        "raw_small_model_rows_used_as_final_prediction": False,
        "labels_accessed": False,
        "foundation_model_parameters_updated": False,
    }
    return output
