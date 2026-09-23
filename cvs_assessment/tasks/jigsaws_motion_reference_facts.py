"""Subject-out reference facts for JIGSAWS trial-level motion assessment.

The adapter combines raw synchronized kinematics with predictions from the
existing held-out-subject gesture model.  It exports observable, normalized
motion descriptors only.  It never reads expert GRS labels and never emits a
skill score or final task verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..grounded_evidence import GroundedEvidenceFact, grounded_fact_payload
from ..mllm_orchestration import PluginEvidence
from .jigsaws_data import JIGSAWS_FPS
from .jigsaws_gesture_model import GESTURE_CLASSES


DESCRIPTOR_ORDER = (
    "trial_duration_seconds",
    "left_path_rate_dataset_units_per_second",
    "right_path_rate_dataset_units_per_second",
    "mean_bimanual_speed_dataset_units_per_second",
    "p95_bimanual_jerk_dataset_units_per_second3",
    "low_motion_fraction",
    "p95_gripper_activity_per_second",
    "bimanual_speed_correlation",
    "predicted_gesture_segments_per_minute",
    "predicted_gesture_transition_rate_per_minute",
    "predicted_gesture_confidence_mean",
    "predicted_gesture_low_confidence_fraction",
)


@dataclass(frozen=True)
class MotionReferenceDescriptor:
    """One label-free complete-trial descriptor used by the reference adapter."""

    values: dict[str, float]
    gesture_duration_fraction: dict[str, float]
    gesture_segment_count: dict[str, int]
    transition_count: int

    def validate(self) -> None:
        if tuple(self.values) != DESCRIPTOR_ORDER:
            raise ValueError("JIGSAWS motion descriptor order changed")
        if not np.isfinite(np.asarray(list(self.values.values()), dtype=np.float64)).all():
            raise ValueError("JIGSAWS motion descriptors must be finite")
        if set(self.gesture_duration_fraction) != set(GESTURE_CLASSES):
            raise ValueError("JIGSAWS gesture duration vocabulary changed")
        if set(self.gesture_segment_count) != set(GESTURE_CLASSES):
            raise ValueError("JIGSAWS gesture segment vocabulary changed")
        if self.transition_count < 0:
            raise ValueError("JIGSAWS transition count must be nonnegative")


def _load_gesture_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as document:
        predicted = np.asarray(document["predicted_class"], dtype=np.int64)
        probability = np.asarray(document["class_probability"], dtype=np.float64)
        source_indices = np.asarray(document["source_frame_indices"], dtype=np.int64)
        classes = tuple(str(item) for item in document["gesture_classes"].tolist())
    if classes != GESTURE_CLASSES:
        raise ValueError("JIGSAWS gesture prediction vocabulary changed")
    if predicted.ndim != 1 or probability.shape != (len(predicted), len(classes)):
        raise ValueError("JIGSAWS gesture prediction has inconsistent shapes")
    if source_indices.shape != predicted.shape or len(predicted) < 1:
        raise ValueError("JIGSAWS gesture prediction has no aligned samples")
    if np.any(predicted < 0) or np.any(predicted >= len(classes)):
        raise ValueError("JIGSAWS gesture prediction contains an invalid class")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("JIGSAWS gesture probabilities are invalid")
    if np.any(source_indices[1:] <= source_indices[:-1]):
        raise ValueError("JIGSAWS gesture prediction is not chronological")
    return predicted, probability, source_indices


def _arm_motion(values: np.ndarray, base: int, fps: float) -> dict[str, np.ndarray]:
    position = values[:, base:base + 3]
    velocity = values[:, base + 12:base + 15]
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1]) * fps
    jerk = np.linalg.norm(
        np.diff(acceleration, axis=0, prepend=acceleration[:1]) * fps,
        axis=1,
    )
    gripper = values[:, base + 18]
    gripper_rate = np.abs(np.diff(gripper, prepend=gripper[:1])) * fps
    return {
        "position": position,
        "speed": speed,
        "jerk": jerk,
        "gripper_rate": gripper_rate,
    }


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            rows.append((int(labels[start]), start, end))
            start = end
    return rows


def motion_reference_descriptor(
    kinematics: np.ndarray,
    gesture_prediction_path: Path,
    *,
    fps: float = JIGSAWS_FPS,
    model_downsample: int = 3,
) -> MotionReferenceDescriptor:
    """Build one complete-trial label-free descriptor."""
    values = np.asarray(kinematics, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 76:
        raise ValueError("JIGSAWS reference adapter expects [T,76] kinematics")
    if not np.isfinite(values).all() or fps <= 0 or model_downsample < 1:
        raise ValueError("Invalid JIGSAWS motion reference input")
    predicted, probability, source_indices = _load_gesture_prediction(
        gesture_prediction_path,
    )
    if int(source_indices[-1]) >= len(values):
        raise ValueError("Gesture prediction exceeds the synchronized kinematics")

    left = _arm_motion(values, 38, fps)
    right = _arm_motion(values, 57, fps)
    duration = len(values) / fps
    left_path = float(np.linalg.norm(np.diff(left["position"], axis=0), axis=1).sum())
    right_path = float(np.linalg.norm(np.diff(right["position"], axis=0), axis=1).sum())
    combined_speed = (left["speed"] + right["speed"]) / 2.0
    combined_jerk = (left["jerk"] + right["jerk"]) / 2.0
    p95_speed = float(np.percentile(combined_speed, 95))
    low_motion_threshold = max(1e-12, 0.05 * p95_speed)
    correlation = 0.0
    if float(np.std(left["speed"])) > 0 and float(np.std(right["speed"])) > 0:
        correlation = float(np.corrcoef(left["speed"], right["speed"])[0, 1])

    selected_probability = probability[np.arange(len(predicted)), predicted]
    segment_rows = _segments(predicted)
    transition_count = max(0, len(segment_rows) - 1)
    gesture_duration = np.zeros(len(GESTURE_CLASSES), dtype=np.float64)
    gesture_count = np.zeros(len(GESTURE_CLASSES), dtype=np.int64)
    for class_index, start, end in segment_rows:
        start_frame = int(source_indices[start])
        end_frame = (
            int(source_indices[end])
            if end < len(source_indices)
            else min(len(values), int(source_indices[-1]) + model_downsample)
        )
        gesture_duration[class_index] += max(0, end_frame - start_frame) / fps
        gesture_count[class_index] += 1
    covered_duration = max(float(gesture_duration.sum()), 1e-12)

    descriptor = MotionReferenceDescriptor(
        values={
            "trial_duration_seconds": float(duration),
            "left_path_rate_dataset_units_per_second": left_path / duration,
            "right_path_rate_dataset_units_per_second": right_path / duration,
            "mean_bimanual_speed_dataset_units_per_second": float(np.mean(combined_speed)),
            "p95_bimanual_jerk_dataset_units_per_second3": float(np.percentile(combined_jerk, 95)),
            "low_motion_fraction": float(np.mean(combined_speed <= low_motion_threshold)),
            "p95_gripper_activity_per_second": float(np.percentile(
                np.maximum(left["gripper_rate"], right["gripper_rate"]), 95,
            )),
            "bimanual_speed_correlation": correlation,
            "predicted_gesture_segments_per_minute": len(segment_rows) * 60.0 / duration,
            "predicted_gesture_transition_rate_per_minute": transition_count * 60.0 / duration,
            "predicted_gesture_confidence_mean": float(np.mean(selected_probability)),
            "predicted_gesture_low_confidence_fraction": float(
                np.mean(selected_probability < 0.5)
            ),
        },
        gesture_duration_fraction={
            name: float(gesture_duration[index] / covered_duration)
            for index, name in enumerate(GESTURE_CLASSES)
        },
        gesture_segment_count={
            name: int(gesture_count[index])
            for index, name in enumerate(GESTURE_CLASSES)
        },
        transition_count=transition_count,
    )
    descriptor.validate()
    return descriptor


def _rounded(value: float) -> float:
    return round(float(value), 4)


def _reference_row(value: float, references: np.ndarray) -> dict[str, Any]:
    if references.ndim != 1 or len(references) < 2 or not np.isfinite(references).all():
        raise ValueError("JIGSAWS motion reference requires at least two finite trials")
    less = int(np.sum(references < value))
    equal = int(np.sum(references == value))
    percentile = (less + 0.5 * equal) / len(references)
    band = "lower_reference_range" if percentile < 0.20 else (
        "higher_reference_range" if percentile > 0.80 else "central_reference_range"
    )
    return {
        "observed": _rounded(value),
        "training_subject_reference_median": _rounded(np.median(references)),
        "training_subject_reference_p20": _rounded(np.percentile(references, 20)),
        "training_subject_reference_p80": _rounded(np.percentile(references, 80)),
        "training_subject_percentile": _rounded(percentile),
        "reference_band": band,
    }


def subject_out_motion_reference_fact_payload(
    target: MotionReferenceDescriptor,
    references: Iterable[MotionReferenceDescriptor],
    *,
    sample_id: str,
    held_out_subject_excluded: bool,
    gesture_prediction_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare a target with other-subject descriptors without GRS labels."""
    target.validate()
    reference_rows = list(references)
    if not held_out_subject_excluded:
        raise PermissionError("JIGSAWS reference facts require subject-out references")
    if len(reference_rows) < 2:
        raise ValueError("JIGSAWS reference facts need at least two training trials")
    for row in reference_rows:
        row.validate()
    comparison = {
        name: _reference_row(
            target.values[name],
            np.asarray([row.values[name] for row in reference_rows], dtype=np.float64),
        )
        for name in DESCRIPTOR_ORDER
    }
    nonzero_durations = {
        key: _rounded(value)
        for key, value in target.gesture_duration_fraction.items()
        if value > 0
    }
    nonzero_counts = {
        key: value for key, value in target.gesture_segment_count.items() if value > 0
    }
    facts = [
        GroundedEvidenceFact(
            fact_id="subject_out_motion_reference",
            fact_type="attribute",
            subject="complete_trial_robot_motion",
            predicate="descriptor_comparison_with_other_subject_trials",
            value=comparison,
            grounding={
                "source": "synchronized_robot_kinematics_and_training_subject_reference",
                "reference_trial_count": len(reference_rows),
                "held_out_subject_excluded": True,
                "reference_uses_expert_GRS": False,
                "percentile_direction_is_not_automatically_good_or_bad": True,
                "not_a_skill_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="predicted_motion_primitive_profile",
            fact_type="attribute",
            subject="complete_trial_predicted_motion_primitives",
            predicate="held_out_model_sequence_summary",
            value={
                "duration_fraction_by_predicted_gesture": nonzero_durations,
                "segment_count_by_predicted_gesture": nonzero_counts,
                "adjacent_transition_count": target.transition_count,
            },
            confidence=_rounded(target.values["predicted_gesture_confidence_mean"]),
            grounding={
                "source": "held_out_subject_TCN_BiGRU_prediction",
                "ground_truth_gesture_transcript_used_for_target_inference": False,
                "gesture_identity_is_not_execution_quality": True,
                "not_a_skill_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="motion_reference_limitations",
            fact_type="attribute",
            subject="motion_evidence_adapter",
            predicate="interpretation_limits",
            value={
                "descriptors_are_observations_not_ratings": True,
                "reference_percentiles_are_not_quality_percentiles": True,
                "video_is_required_for_semantic_and_final_product_judgment": True,
                "missing_or_uncertain_gesture_predictions_are_not_failure_evidence": True,
            },
            grounding={"not_a_skill_verdict": True},
        ),
    ]
    return grounded_fact_payload(
        facts,
        modality="temporal",
        source_description=(
            "A held-out-subject TCN-BiGRU supplies motion-primitive observations. "
            "Raw motion descriptors are normalized only against trials from other "
            "subjects, without expert ratings. The facts are advisory and cannot "
            "issue a skill score."
        ),
        provenance={
            "adapter": "jigsaws_subject_out_motion_reference_facts_v1",
            "sample_id": sample_id,
            "gesture_prediction_sha256": gesture_prediction_sha256,
            "held_out_subject_excluded_from_reference": True,
            "expert_GRS_used": False,
            "self_reported_experience_used": False,
            "ground_truth_gesture_transcript_used_for_target_inference": False,
            "foundation_model_parameters_updated": False,
            "raw_task_verdict_exported": False,
        },
    )


def subject_out_motion_reference_plugin(
    target: MotionReferenceDescriptor,
    references: Iterable[MotionReferenceDescriptor],
    *,
    sample_id: str,
    held_out_subject_excluded: bool = True,
    gesture_prediction_sha256: str | None = None,
) -> PluginEvidence:
    payload = subject_out_motion_reference_fact_payload(
        target,
        references,
        sample_id=sample_id,
        held_out_subject_excluded=held_out_subject_excluded,
        gesture_prediction_sha256=gesture_prediction_sha256,
    )
    plugin = PluginEvidence(
        plugin_id="jigsaws_subject_out_motion_reference_facts_v1",
        plugin_kind="temporal",
        description=(
            "Fact-only subject-out motion reference from synchronized kinematics "
            "and a held-out-subject temporal model."
        ),
        payload=payload,
        foundation_model_parameters_updated=False,
    )
    plugin.validate_fact_only()
    return plugin
