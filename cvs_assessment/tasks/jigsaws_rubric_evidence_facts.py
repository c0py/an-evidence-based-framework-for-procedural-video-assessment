"""Rubric-aligned, fact-only JIGSAWS evidence from motion primitives.

The adapter turns synchronized robot kinematics and a held-out-subject gesture
prediction into measurements that are easier to relate to the modified GRS.
It deliberately cannot emit a GRS score, skill class, or final verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..grounded_evidence import GroundedEvidenceFact, grounded_fact_payload
from ..mllm_orchestration import PluginEvidence
from .jigsaws_data import JIGSAWS_FPS


# Kept local so this lightweight post-confirmation adapter does not import the
# Torch gesture-model implementation merely to read its frozen vocabulary.
GESTURE_CLASSES = (
    "BG", "G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11",
)
SUTURING_GESTURE_DESCRIPTIONS = {
    "BG": "unannotated/background interval",
    "G1": "reaching for needle with right hand",
    "G2": "positioning needle",
    "G3": "pushing needle through tissue",
    "G4": "transferring needle from left to right",
    "G5": "moving to center with needle in grip",
    "G6": "pulling suture with left hand",
    "G8": "orienting needle",
    "G9": "using right hand to help tighten suture",
    "G10": "loosening more suture",
    "G11": "dropping suture at end and moving to end points",
}


SCALAR_DEFINITIONS = {
    "trial_duration_seconds": (
        "Elapsed trial time. A larger value means a longer trial, not automatically poorer skill."
    ),
    "combined_tool_path_rate_dataset_units_per_second": (
        "Sum of left and right tool-tip travel divided by elapsed time. Direction is descriptive only."
    ),
    "p95_smoothed_bimanual_jerk_dataset_units_per_second3": (
        "The 95th percentile of smoothed two-tool jerk. A larger value means more abrupt motion samples."
    ),
    "near_stationary_fraction": (
        "Fraction of frames where both tools are near stationary. Larger means more near-stationary time; lower means less, not more."
    ),
    "simultaneous_tool_motion_fraction": (
        "Fraction of frames where both tools move above their adaptive activity thresholds."
    ),
    "direction_reversal_events_per_minute": (
        "Rate of substantial tool-direction reversals after smoothing. Reversals may be necessary or avoidable."
    ),
    "absolute_gripper_change_per_minute": (
        "Accumulated absolute gripper-angle change per minute. This measures activity, not handling quality."
    ),
    "predicted_gesture_segments_per_minute": (
        "Rate of predicted motion-primitive segments. It has no validated good/bad direction and may only direct visual review."
    ),
    "predicted_aba_revisits_per_minute": (
        "Rate of A-B-A primitive patterns. It has no validated good/bad direction because normal repeated suturing can create revisits."
    ),
    "position_orient_episodes_per_predicted_push": (
        "Predicted G2/G8 positioning-orienting episodes per G3 needle-push episode; lower or higher values have no validated skill direction."
    ),
    "needle_transfer_episodes_per_predicted_push": (
        "Predicted G4 needle transfers per G3 needle-push episode; lower or higher values have no validated skill direction."
    ),
    "suture_adjustment_episodes_per_predicted_push": (
        "Predicted G9/G10 tighten-or-loosen episodes per G3 needle-push episode; lower or higher values have no validated skill direction."
    ),
    "predicted_gesture_confidence_mean": (
        "Mean selected-class probability of the held-out-subject gesture model; it may affect evidence adequacy only, never score direction."
    ),
    "predicted_gesture_low_confidence_fraction": (
        "Fraction of gesture samples with selected-class probability below 0.5; it may affect evidence adequacy only, never score direction."
    ),
}

SCALAR_ORDER = tuple(SCALAR_DEFINITIONS)
_HANDLING_ATTENTION_GESTURES = {"G2", "G4", "G8", "G10"}
_VISUAL_ATTENTION_GESTURES = {"G2", "G3", "G4", "G6", "G8", "G9", "G10"}
_TIME_MOTION_SCALARS = (
    "trial_duration_seconds",
    "combined_tool_path_rate_dataset_units_per_second",
    "p95_smoothed_bimanual_jerk_dataset_units_per_second3",
    "near_stationary_fraction",
    "simultaneous_tool_motion_fraction",
    "direction_reversal_events_per_minute",
    "absolute_gripper_change_per_minute",
)


@dataclass(frozen=True)
class RubricEvidenceDescriptor:
    """One complete-trial set of observable, non-verdict measurements."""

    scalar_values: dict[str, float | None]
    gesture_occurrence_count: dict[str, int]
    gesture_duration_seconds: dict[str, float]
    transition_count: dict[str, int]
    handling_attention_intervals: tuple[dict[str, Any], ...]

    def validate(self) -> None:
        if tuple(self.scalar_values) != SCALAR_ORDER:
            raise ValueError("JIGSAWS rubric evidence scalar order changed")
        finite = [value for value in self.scalar_values.values() if value is not None]
        if not np.isfinite(np.asarray(finite, dtype=np.float64)).all():
            raise ValueError("JIGSAWS rubric evidence contains a non-finite scalar")
        if set(self.gesture_occurrence_count) != set(GESTURE_CLASSES):
            raise ValueError("JIGSAWS rubric evidence gesture count vocabulary changed")
        if set(self.gesture_duration_seconds) != set(GESTURE_CLASSES):
            raise ValueError("JIGSAWS rubric evidence gesture duration vocabulary changed")
        if any(value < 0 for value in self.gesture_occurrence_count.values()):
            raise ValueError("JIGSAWS rubric evidence has a negative gesture count")
        if any(value < 0 for value in self.gesture_duration_seconds.values()):
            raise ValueError("JIGSAWS rubric evidence has a negative gesture duration")


def _rounded(value: float) -> float:
    return round(float(value), 4)


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as document:
        predicted = np.asarray(document["predicted_class"], dtype=np.int64)
        probability = np.asarray(document["class_probability"], dtype=np.float64)
        source_indices = np.asarray(document["source_frame_indices"], dtype=np.int64)
        classes = tuple(str(item) for item in document["gesture_classes"].tolist())
    if classes != GESTURE_CLASSES:
        raise ValueError("JIGSAWS rubric adapter gesture vocabulary changed")
    if predicted.ndim != 1 or probability.shape != (len(predicted), len(classes)):
        raise ValueError("JIGSAWS rubric adapter prediction shapes are inconsistent")
    if source_indices.shape != predicted.shape or len(predicted) < 1:
        raise ValueError("JIGSAWS rubric adapter prediction is empty or unaligned")
    if np.any(predicted < 0) or np.any(predicted >= len(classes)):
        raise ValueError("JIGSAWS rubric adapter prediction contains an invalid class")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("JIGSAWS rubric adapter prediction probabilities are invalid")
    if np.any(source_indices[1:] <= source_indices[:-1]):
        raise ValueError("JIGSAWS rubric adapter frame indices are not chronological")
    return predicted, probability, source_indices


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            rows.append((int(labels[start]), start, end))
            start = end
    return rows


def visual_attention_sampling_proposals(
    gesture_prediction_path: Path,
    *,
    source_frame_count: int,
    fps: float = JIGSAWS_FPS,
    model_downsample: int = 3,
    proposal_count: int = 6,
    minimum_selected_probability: float = 0.65,
) -> tuple[dict[str, Any], ...]:
    """Select diverse primitive midpoints for image review, never for scoring.

    The probability threshold and values are returned for the offline audit only.
    Callers must not place them in the foundation-model plugin payload.
    """
    if source_frame_count < 1 or fps <= 0 or model_downsample < 1:
        raise ValueError("Invalid JIGSAWS visual-attention sampling input")
    if proposal_count < 1 or not 0.0 <= minimum_selected_probability <= 1.0:
        raise ValueError("Invalid JIGSAWS visual-attention selection rule")
    predicted, probability, source_indices = _load_prediction(gesture_prediction_path)
    if int(source_indices[-1]) >= source_frame_count:
        raise ValueError("JIGSAWS visual-attention prediction exceeds source frames")
    selected_probability = probability[np.arange(len(predicted)), predicted]
    candidates: list[dict[str, Any]] = []
    for segment_index, (class_index, start, end) in enumerate(_segments(predicted)):
        gesture_id = GESTURE_CLASSES[class_index]
        if gesture_id not in _VISUAL_ATTENTION_GESTURES:
            continue
        start_frame = int(source_indices[start])
        end_frame = (
            int(source_indices[end])
            if end < len(source_indices)
            else min(source_frame_count, int(source_indices[-1]) + model_downsample)
        )
        end_frame = max(start_frame + 1, min(end_frame, source_frame_count))
        mean_probability = float(np.mean(selected_probability[start:end]))
        if mean_probability < minimum_selected_probability:
            continue
        midpoint_frame = min(
            source_frame_count - 1,
            int(round((start_frame + end_frame - 1) / 2.0)),
        )
        candidates.append({
            "gesture_id": gesture_id,
            "description": SUTURING_GESTURE_DESCRIPTIONS[gesture_id],
            "segment_index": segment_index,
            "start_seconds": _rounded(start_frame / fps),
            "end_seconds": _rounded(end_frame / fps),
            "midpoint_seconds": _rounded(midpoint_frame / fps),
            "source_frame_index": midpoint_frame,
            "selection_probability_mean": _rounded(mean_probability),
            "duration_seconds": _rounded((end_frame - start_frame) / fps),
        })
    if len(candidates) < proposal_count:
        raise ValueError(
            "Insufficient high-probability JIGSAWS intervals for visual attention"
        )

    def quality_key(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            -float(row["selection_probability_mean"]),
            -float(row["duration_seconds"]),
            float(row["start_seconds"]),
        )

    # First cover as many different handling primitives as possible.
    best_by_gesture: dict[str, dict[str, Any]] = {}
    for row in sorted(candidates, key=quality_key):
        best_by_gesture.setdefault(str(row["gesture_id"]), row)
    chosen = sorted(best_by_gesture.values(), key=quality_key)[:proposal_count]
    chosen_ids = {int(row["segment_index"]) for row in chosen}

    # If primitive diversity does not fill the budget, maximize temporal coverage.
    while len(chosen) < proposal_count:
        remaining = [
            row for row in candidates
            if int(row["segment_index"]) not in chosen_ids
        ]
        if not remaining:
            raise RuntimeError("JIGSAWS visual-attention selection exhausted candidates")

        def coverage_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
            timestamp = float(row["midpoint_seconds"])
            separation = min(
                abs(timestamp - float(selected["midpoint_seconds"]))
                for selected in chosen
            )
            return (
                -separation,
                -float(row["selection_probability_mean"]),
                -float(row["duration_seconds"]),
                float(row["start_seconds"]),
            )

        next_row = min(remaining, key=coverage_key)
        chosen.append(next_row)
        chosen_ids.add(int(next_row["segment_index"]))
    return tuple(sorted(chosen, key=lambda row: float(row["midpoint_seconds"])))


def _smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    if width < 1 or width % 2 == 0:
        raise ValueError("JIGSAWS rubric smoothing width must be positive and odd")
    radius = width // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
    cumulative = np.vstack([np.zeros((1, values.shape[1])), cumulative])
    return (cumulative[width:] - cumulative[:-width]) / width


def _arm_signals(values: np.ndarray, base: int, fps: float) -> dict[str, np.ndarray]:
    position = values[:, base:base + 3]
    velocity = _smooth(values[:, base + 12:base + 15])
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1]) * fps
    jerk = np.linalg.norm(
        np.diff(acceleration, axis=0, prepend=acceleration[:1]) * fps,
        axis=1,
    )
    gripper = values[:, base + 18]
    return {
        "position": position,
        "velocity": velocity,
        "speed": speed,
        "jerk": jerk,
        "gripper": gripper,
    }


def _event_run_count(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if len(values) == 0:
        return 0
    return int(values[0]) + int(np.sum(values[1:] & ~values[:-1]))


def _reversal_count(velocity: np.ndarray, speed: np.ndarray) -> int:
    threshold = max(1e-12, 0.10 * float(np.percentile(speed, 95)))
    left = velocity[:-1]
    right = velocity[1:]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.ones(len(left), dtype=np.float64)
    valid = denominator > 1e-12
    cosine[valid] = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    substantial = (speed[:-1] > threshold) & (speed[1:] > threshold)
    return _event_run_count(substantial & (cosine < -0.5))


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator > 0 else None


def rubric_evidence_descriptor(
    kinematics: np.ndarray,
    gesture_prediction_path: Path,
    *,
    fps: float = JIGSAWS_FPS,
    model_downsample: int = 3,
) -> RubricEvidenceDescriptor:
    """Build rubric-aligned observations without accessing expert ratings."""
    values = np.asarray(kinematics, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 76:
        raise ValueError("JIGSAWS rubric adapter expects [T,76] kinematics")
    if not np.isfinite(values).all() or fps <= 0 or model_downsample < 1:
        raise ValueError("Invalid JIGSAWS rubric adapter input")
    predicted, probability, source_indices = _load_prediction(gesture_prediction_path)
    if int(source_indices[-1]) >= len(values):
        raise ValueError("JIGSAWS rubric gesture prediction exceeds kinematics")

    left = _arm_signals(values, 38, fps)
    right = _arm_signals(values, 57, fps)
    duration = len(values) / fps
    combined_speed = (left["speed"] + right["speed"]) / 2.0
    combined_jerk = (left["jerk"] + right["jerk"]) / 2.0
    stationary_threshold = max(1e-12, 0.05 * float(np.percentile(combined_speed, 95)))
    left_active = left["speed"] > max(
        1e-12, 0.10 * float(np.percentile(left["speed"], 95))
    )
    right_active = right["speed"] > max(
        1e-12, 0.10 * float(np.percentile(right["speed"], 95))
    )
    path_length = sum(
        float(np.linalg.norm(np.diff(arm["position"], axis=0), axis=1).sum())
        for arm in (left, right)
    )
    reversal_count = sum(
        _reversal_count(arm["velocity"], arm["speed"]) for arm in (left, right)
    )
    gripper_change = sum(
        float(np.abs(np.diff(arm["gripper"])).sum()) for arm in (left, right)
    )

    segment_rows = _segments(predicted)
    selected_probability = probability[np.arange(len(predicted)), predicted]
    gesture_count = {name: 0 for name in GESTURE_CLASSES}
    gesture_duration = {name: 0.0 for name in GESTURE_CLASSES}
    transition_count: dict[str, int] = {}
    handling_intervals: list[dict[str, Any]] = []
    for segment_index, (class_index, start, end) in enumerate(segment_rows):
        name = GESTURE_CLASSES[class_index]
        start_frame = int(source_indices[start])
        end_frame = (
            int(source_indices[end])
            if end < len(source_indices)
            else min(len(values), int(source_indices[-1]) + model_downsample)
        )
        end_frame = max(start_frame + 1, min(end_frame, len(values)))
        segment_duration = (end_frame - start_frame) / fps
        gesture_count[name] += 1
        gesture_duration[name] += segment_duration
        if name in _HANDLING_ATTENTION_GESTURES:
            handling_intervals.append({
                "gesture_id": name,
                "description": SUTURING_GESTURE_DESCRIPTIONS[name],
                "segment_index": segment_index,
                "start_seconds": _rounded(start_frame / fps),
                "end_seconds": _rounded(end_frame / fps),
                "duration_seconds": _rounded(segment_duration),
                "gesture_confidence_mean": _rounded(
                    np.mean(selected_probability[start:end])
                ),
            })
    for left_segment, right_segment in zip(segment_rows, segment_rows[1:]):
        key = f"{GESTURE_CLASSES[left_segment[0]]}->{GESTURE_CLASSES[right_segment[0]]}"
        transition_count[key] = transition_count.get(key, 0) + 1
    aba_revisit_count = sum(
        left_row[0] == right_row[0] and left_row[0] != middle_row[0]
        for left_row, middle_row, right_row in zip(
            segment_rows, segment_rows[1:], segment_rows[2:]
        )
    )
    handling_intervals.sort(key=lambda row: (-row["duration_seconds"], row["start_seconds"]))

    push_count = gesture_count["G3"]
    scalar_values: dict[str, float | None] = {
        "trial_duration_seconds": float(duration),
        "combined_tool_path_rate_dataset_units_per_second": path_length / duration,
        "p95_smoothed_bimanual_jerk_dataset_units_per_second3": float(
            np.percentile(combined_jerk, 95)
        ),
        "near_stationary_fraction": float(np.mean(combined_speed <= stationary_threshold)),
        "simultaneous_tool_motion_fraction": float(np.mean(left_active & right_active)),
        "direction_reversal_events_per_minute": reversal_count * 60.0 / duration,
        "absolute_gripper_change_per_minute": gripper_change * 60.0 / duration,
        "predicted_gesture_segments_per_minute": len(segment_rows) * 60.0 / duration,
        "predicted_aba_revisits_per_minute": aba_revisit_count * 60.0 / duration,
        "position_orient_episodes_per_predicted_push": _ratio(
            gesture_count["G2"] + gesture_count["G8"], push_count,
        ),
        "needle_transfer_episodes_per_predicted_push": _ratio(
            gesture_count["G4"], push_count,
        ),
        "suture_adjustment_episodes_per_predicted_push": _ratio(
            gesture_count["G9"] + gesture_count["G10"], push_count,
        ),
        "predicted_gesture_confidence_mean": float(np.mean(selected_probability)),
        "predicted_gesture_low_confidence_fraction": float(
            np.mean(selected_probability < 0.5)
        ),
    }
    descriptor = RubricEvidenceDescriptor(
        scalar_values=scalar_values,
        gesture_occurrence_count=gesture_count,
        gesture_duration_seconds=gesture_duration,
        transition_count=transition_count,
        handling_attention_intervals=tuple(handling_intervals[:6]),
    )
    descriptor.validate()
    return descriptor


def _reference_row(
    name: str,
    observed: float | None,
    references: list[RubricEvidenceDescriptor],
) -> dict[str, Any]:
    if observed is None:
        return {
            "observed": "unavailable_no_predicted_needle_push",
            "definition": SCALAR_DEFINITIONS[name],
            "quality_verdict": "not_provided",
        }
    values = np.asarray([
        row.scalar_values[name]
        for row in references
        if row.scalar_values[name] is not None
    ], dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Insufficient JIGSAWS development references for {name}")
    less = int(np.sum(values < observed))
    equal = int(np.sum(values == observed))
    return {
        "observed": _rounded(observed),
        "other_consumed_development_median": _rounded(np.median(values)),
        "other_consumed_development_p20": _rounded(np.percentile(values, 20)),
        "other_consumed_development_p80": _rounded(np.percentile(values, 80)),
        "other_consumed_development_percentile": _rounded(
            (less + 0.5 * equal) / len(values)
        ),
        "definition": SCALAR_DEFINITIONS[name],
        "quality_verdict": "not_provided",
    }


def rubric_aligned_fact_payload(
    target: RubricEvidenceDescriptor,
    references: Iterable[RubricEvidenceDescriptor],
    *,
    sample_id: str,
    held_out_subject_excluded: bool,
    reference_scope: str = "other_consumed_development_samples",
) -> dict[str, Any]:
    """Create fact-only evidence with explicit measurement direction guards."""
    target.validate()
    reference_rows = list(references)
    if not held_out_subject_excluded:
        raise PermissionError("JIGSAWS rubric references must exclude the target subject")
    if len(reference_rows) < 2:
        raise ValueError("JIGSAWS rubric evidence requires at least two references")
    for row in reference_rows:
        row.validate()
    comparison = {
        name: _reference_row(name, target.scalar_values[name], reference_rows)
        for name in SCALAR_ORDER
    }
    nonzero_counts = {
        name: count for name, count in target.gesture_occurrence_count.items()
        if count > 0
    }
    nonzero_durations = {
        name: _rounded(duration)
        for name, duration in target.gesture_duration_seconds.items()
        if duration > 0
    }
    facts: list[GroundedEvidenceFact] = [
        GroundedEvidenceFact(
            fact_id="rubric_evidence_scope",
            fact_type="visibility",
            subject="complete_trial_quality_evidence",
            predicate="observable_dimension_scope",
            value={
                "directly_supported_by_this_adapter": [
                    "time_and_motion",
                ],
                "partially_supported_by_predicted_primitives": [
                    "suture_needle_handling", "flow_of_operation",
                ],
                "not_observable_from_motion_and_gesture_facts_alone": [
                    "respect_for_tissue", "quality_of_final_product",
                ],
                "overall_performance_requires_foundation_model_integration": True,
            },
            grounding={"not_a_skill_verdict": True},
        ),
        GroundedEvidenceFact(
            fact_id="time_motion_measurements",
            fact_type="attribute",
            subject="complete_trial_robot_motion",
            predicate="rubric_aligned_efficiency_and_smoothness_measurements",
            value={
                name: comparison[name]
                for name in (
                    "trial_duration_seconds",
                    "combined_tool_path_rate_dataset_units_per_second",
                    "p95_smoothed_bimanual_jerk_dataset_units_per_second3",
                    "near_stationary_fraction",
                    "simultaneous_tool_motion_fraction",
                    "direction_reversal_events_per_minute",
                    "absolute_gripper_change_per_minute",
                )
            },
            grounding={
                "source": "synchronized_robot_kinematics",
                "reference_scope": reference_scope,
                "reference_uses_expert_GRS": False,
                "measurement_direction_is_explicit": True,
                "permitted_score_target": "time_and_motion_only",
                "no_single_measurement_is_a_score": True,
                "not_a_skill_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="needle_suture_primitive_measurements",
            fact_type="attribute",
            subject="complete_trial_predicted_needle_suture_primitives",
            predicate="handling_related_counts_durations_and_normalized_ratios",
            value={
                "released_gesture_descriptions": SUTURING_GESTURE_DESCRIPTIONS,
                "occurrence_count": nonzero_counts,
                "duration_seconds": nonzero_durations,
                "normalized_measurements": {
                    name: comparison[name]
                    for name in (
                        "position_orient_episodes_per_predicted_push",
                        "needle_transfer_episodes_per_predicted_push",
                        "suture_adjustment_episodes_per_predicted_push",
                    )
                },
            },
            confidence=_rounded(
                target.scalar_values["predicted_gesture_confidence_mean"] or 0.0
            ),
            grounding={
                "source": "held_out_subject_TCN_BiGRU_prediction",
                "gesture_identity_is_not_handling_quality": True,
                "counts_ratios_and_durations_have_no_validated_score_direction": True,
                "permitted_use": (
                    "locate handling intervals and adjust evidence adequacy; "
                    "score direction requires corroborating video evidence"
                ),
                "normal_four_pass_suturing_structure_can_repeat_primitives": True,
                "ground_truth_transcript_used": False,
                "not_a_skill_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="flow_sequence_measurements",
            fact_type="relation",
            subject="predicted_motion_primitive_sequence",
            predicate="sequence_density_revisits_and_uncertainty",
            value={
                "adjacent_transition_count": target.transition_count,
                "reference_measurements": {
                    name: comparison[name]
                    for name in (
                        "predicted_gesture_segments_per_minute",
                        "predicted_aba_revisits_per_minute",
                        "predicted_gesture_confidence_mean",
                        "predicted_gesture_low_confidence_fraction",
                    )
                },
            },
            grounding={
                "source": "held_out_subject_TCN_BiGRU_prediction",
                "sequence_patterns_are_not_automatically_errors": True,
                "gesture_model_confidence_may_affect_evidence_adequacy_only": True,
                "sequence_counts_have_no_validated_score_direction": True,
                "score_direction_requires_corroborating_video_evidence": True,
                "not_a_skill_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="rubric_evidence_limitations",
            fact_type="visibility",
            subject="rubric_aligned_motion_adapter",
            predicate="interpretation_limits",
            value={
                "no_force_or_tissue_deformation_measurement": True,
                "no_needle_pose_or_suture_tension_measurement": True,
                "no_final_product_geometry_measurement": True,
                "absence_of_detected_problem_is_not_positive_evidence": True,
                "low_near_stationary_percentile_means_fewer_near_stationary_frames": True,
                "all_reference_percentiles_are_descriptive_not_quality_percentiles": True,
                "gesture_confidence_cannot_raise_or_lower_a_GRS_score": True,
                "primitive_ratios_cannot_raise_or_lower_a_GRS_score_without_video": True,
            },
            grounding={"not_a_skill_verdict": True},
        ),
    ]
    for index, interval in enumerate(target.handling_attention_intervals):
        facts.append(GroundedEvidenceFact(
            fact_id=f"handling_attention_interval_{index:02d}",
            fact_type="event",
            subject="predicted_needle_suture_activity",
            predicate="long_position_orient_transfer_or_loosen_interval",
            value={
                key: value for key, value in interval.items()
                if key not in {"start_seconds", "end_seconds"}
            },
            confidence=float(interval["gesture_confidence_mean"]),
            start_s=float(interval["start_seconds"]),
            end_s=float(interval["end_seconds"]),
            grounding={
                "source": "held_out_subject_TCN_BiGRU_prediction",
                "attention_interval_not_error_label": True,
                "permitted_use": "timestamp_for_visual_review_only",
                "not_a_skill_verdict": True,
            },
        ))
    return grounded_fact_payload(
        facts,
        modality="temporal",
        source_description=(
            "Rubric-aligned facts from synchronized robot kinematics and a "
            "held-out-subject gesture model. Measurements expose direction and "
            "limits but never emit an expert score or skill verdict."
        ),
        provenance={
            "adapter": "jigsaws_rubric_aligned_motion_facts_v2",
            "sample_id": sample_id,
            "reference_scope": reference_scope,
            "reference_trial_count": len(reference_rows),
            "held_out_subject_excluded_from_reference": True,
            "expert_GRS_used": False,
            "self_reported_experience_used": False,
            "ground_truth_gesture_transcript_used": False,
            "foundation_model_parameters_updated": False,
            "raw_task_verdict_exported": False,
        },
    )


def rubric_aligned_plugin_evidence(
    target: RubricEvidenceDescriptor,
    references: Iterable[RubricEvidenceDescriptor],
    *,
    sample_id: str,
    held_out_subject_excluded: bool = True,
) -> PluginEvidence:
    payload = rubric_aligned_fact_payload(
        target,
        references,
        sample_id=sample_id,
        held_out_subject_excluded=held_out_subject_excluded,
    )
    plugin = PluginEvidence(
        plugin_id="jigsaws_rubric_aligned_motion_facts_v2",
        plugin_kind="temporal",
        description=(
            "Fact-only rubric-aligned motion and predicted needle/suture primitive evidence."
        ),
        payload=payload,
        foundation_model_parameters_updated=False,
    )
    plugin.validate_fact_only()
    return plugin


def rubric_aligned_visual_attention_fact_payload(
    target: RubricEvidenceDescriptor,
    references: Iterable[RubricEvidenceDescriptor],
    sampling_proposals: Iterable[dict[str, Any]],
    *,
    sample_id: str,
    held_out_subject_excluded: bool,
    reference_scope: str = "other_consumed_development_samples",
) -> dict[str, Any]:
    """Create the v3 whitelist: motion facts plus score-neutral review pointers."""
    target.validate()
    reference_rows = list(references)
    proposal_rows = list(sampling_proposals)
    if not held_out_subject_excluded:
        raise PermissionError("JIGSAWS rubric references must exclude the target subject")
    if len(reference_rows) < 2:
        raise ValueError("JIGSAWS rubric evidence requires at least two references")
    if not proposal_rows:
        raise ValueError("JIGSAWS visual-attention evidence requires review pointers")
    for row in reference_rows:
        row.validate()
    comparison = {
        name: _reference_row(name, target.scalar_values[name], reference_rows)
        for name in _TIME_MOTION_SCALARS
    }
    facts: list[GroundedEvidenceFact] = [
        GroundedEvidenceFact(
            fact_id="rubric_evidence_scope",
            fact_type="visibility",
            subject="complete_trial_quality_evidence",
            predicate="observable_dimension_scope",
            value={
                "directly_supported_by_measurements": ["time_and_motion"],
                "requires_visual_evidence": [
                    "respect_for_tissue",
                    "suture_needle_handling",
                    "flow_of_operation",
                    "overall_performance",
                    "quality_of_final_product",
                ],
                "temporal_pointers_are_visual_attention_cues_only": True,
            },
            grounding={"not_a_skill_verdict": True},
        ),
        GroundedEvidenceFact(
            fact_id="time_motion_measurements",
            fact_type="attribute",
            subject="complete_trial_robot_motion",
            predicate="efficiency_and_smoothness_measurements",
            value={name: comparison[name] for name in _TIME_MOTION_SCALARS},
            grounding={
                "source": "synchronized_robot_kinematics",
                "reference_scope": reference_scope,
                "reference_uses_expert_GRS": False,
                "measurement_direction_is_explicit": True,
                "permitted_score_target": "time_and_motion_only",
                "no_single_measurement_is_a_score": True,
                "not_a_skill_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="rubric_evidence_limitations",
            fact_type="visibility",
            subject="rubric_aligned_visual_attention_adapter",
            predicate="interpretation_limits",
            value={
                "no_force_or_tissue_deformation_measurement": True,
                "no_needle_pose_or_suture_tension_measurement": True,
                "no_final_product_geometry_measurement": True,
                "absence_of_a_visible_problem_is_not_positive_evidence": True,
                "low_near_stationary_percentile_means_fewer_near_stationary_frames": True,
                "development_reference_percentiles_are_descriptive_not_quality_percentiles": True,
                "temporal_pointer_identity_is_not_correctness_or_error_evidence": True,
            },
            grounding={"not_a_skill_verdict": True},
        ),
    ]
    required = {
        "gesture_id", "description", "start_seconds", "end_seconds",
        "midpoint_seconds", "source_frame_index",
    }
    for index, proposal in enumerate(proposal_rows):
        if not required.issubset(proposal):
            raise ValueError("Incomplete JIGSAWS visual-attention proposal")
        facts.append(GroundedEvidenceFact(
            fact_id=f"visual_review_pointer_{index:02d}",
            fact_type="event",
            subject="needle_suture_activity",
            predicate="temporal_visual_review_pointer",
            value={
                "gesture_id": str(proposal["gesture_id"]),
                "description": str(proposal["description"]),
                "review_instruction": (
                    "Inspect the image itself; this pointer does not say whether "
                    "the action is correct, incorrect, proficient, or unskilled."
                ),
            },
            timestamp_s=float(proposal["midpoint_seconds"]),
            start_s=float(proposal["start_seconds"]),
            end_s=float(proposal["end_seconds"]),
            grounding={
                "source": "held_out_subject_temporal_locator",
                "permitted_use": "timestamp_for_visual_review_only",
                "not_a_skill_verdict": True,
            },
        ))
    payload = grounded_fact_payload(
        facts,
        modality="temporal",
        source_description=(
            "Complete-trial robot-motion measurements plus temporal pointers that "
            "select images for direct visual review. Pointers never emit quality."
        ),
        provenance={
            "adapter": "jigsaws_rubric_visual_attention_facts_v3",
            "sample_id": sample_id,
            "reference_scope": reference_scope,
            "held_out_subject_excluded_from_reference": True,
            "expert_GRS_used": False,
            "self_reported_experience_used": False,
            "ground_truth_gesture_transcript_used": False,
            "foundation_model_parameters_updated": False,
            "raw_task_verdict_exported": False,
        },
    )
    serialized = str(payload).lower()
    forbidden_proxy_terms = (
        "confidence", "selection_probability", "occurrence_count",
        "episodes_per_predicted_push", "aba_revisit", "transition_count",
        "gesture_segments_per_minute",
    )
    if any(term in serialized for term in forbidden_proxy_terms):
        raise RuntimeError("A score-ambiguous locator proxy leaked into the v3 payload")
    return payload


def rubric_aligned_visual_attention_plugin_evidence(
    target: RubricEvidenceDescriptor,
    references: Iterable[RubricEvidenceDescriptor],
    sampling_proposals: Iterable[dict[str, Any]],
    *,
    sample_id: str,
    held_out_subject_excluded: bool = True,
) -> PluginEvidence:
    payload = rubric_aligned_visual_attention_fact_payload(
        target,
        references,
        sampling_proposals,
        sample_id=sample_id,
        held_out_subject_excluded=held_out_subject_excluded,
    )
    plugin = PluginEvidence(
        plugin_id="jigsaws_rubric_visual_attention_facts_v3",
        plugin_kind="temporal",
        description=(
            "Fact-only motion measurements and score-neutral visual-review pointers."
        ),
        payload=payload,
        foundation_model_parameters_updated=False,
    )
    plugin.validate_fact_only()
    return plugin
