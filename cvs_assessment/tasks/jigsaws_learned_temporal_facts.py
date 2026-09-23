"""Fact-only adapter for subject-out JIGSAWS gesture-model predictions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..grounded_evidence import GroundedEvidenceFact, grounded_fact_payload
from ..mllm_orchestration import PluginEvidence
from .jigsaws_gesture_model import GESTURE_CLASSES


# Task adapters may explain their local action vocabulary.  These descriptions
# are the JIGSAWS release vocabulary; they do not encode execution quality.
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


def _rounded(value: float) -> float:
    return round(float(value), 4)


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as value:
        predicted = np.asarray(value["predicted_class"], dtype=np.int64)
        probability = np.asarray(value["class_probability"], dtype=np.float32)
        source_indices = np.asarray(value["source_frame_indices"], dtype=np.int64)
        classes = tuple(str(item) for item in value["gesture_classes"].tolist())
    if classes != GESTURE_CLASSES:
        raise ValueError("JIGSAWS learned fact vocabulary differs from the frozen model")
    if predicted.ndim != 1 or probability.shape != (len(predicted), len(classes)):
        raise ValueError("JIGSAWS learned prediction has inconsistent shapes")
    if source_indices.shape != predicted.shape or len(predicted) < 1:
        raise ValueError("JIGSAWS learned prediction has no valid frame alignment")
    if np.any(predicted < 0) or np.any(predicted >= len(classes)):
        raise ValueError("JIGSAWS learned prediction contains an invalid class")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("JIGSAWS learned prediction contains invalid probabilities")
    if np.any(source_indices[1:] <= source_indices[:-1]):
        raise ValueError("JIGSAWS learned prediction frame alignment is not increasing")
    return predicted, probability, source_indices


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    output = []
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            output.append((int(labels[start]), start, end))
            start = end
    return output


def learned_gesture_fact_payload(
    prediction_path: Path,
    *,
    sample_id: str,
    source_fps: float = 30.0,
    model_downsample: int = 3,
    prediction_sha256: str | None = None,
) -> dict[str, Any]:
    """Convert a blind OOF gesture sequence into observations, never a score."""
    if source_fps <= 0 or model_downsample < 1:
        raise ValueError("Invalid JIGSAWS learned temporal timing")
    predicted, probability, source_indices = _load_prediction(prediction_path)
    selected_probability = probability[np.arange(len(predicted)), predicted]
    segment_rows = _segments(predicted)
    facts: list[GroundedEvidenceFact] = [
        GroundedEvidenceFact(
            fact_id="gesture_vocabulary",
            fact_type="attribute",
            subject="suturing_motion_primitive_dictionary",
            predicate="released_gesture_descriptions",
            value=SUTURING_GESTURE_DESCRIPTIONS,
            grounding={
                "source": "JIGSAWS_release_gesture_vocabulary",
                "descriptions_encode_action_identity_not_quality": True,
                "not_a_task_verdict": True,
            },
        )
    ]
    duration_by_class = {name: 0.0 for name in GESTURE_CLASSES}
    count_by_class = {name: 0 for name in GESTURE_CLASSES}
    for segment_index, (class_index, start, end) in enumerate(segment_rows):
        name = GESTURE_CLASSES[class_index]
        start_s = source_indices[start] / source_fps
        end_frame = (
            source_indices[end]
            if end < len(source_indices)
            else source_indices[-1] + model_downsample
        )
        end_s = end_frame / source_fps
        duration_by_class[name] += end_s - start_s
        count_by_class[name] += 1
        facts.append(GroundedEvidenceFact(
            fact_id=f"predicted_motion_segment_{segment_index:03d}",
            fact_type="event",
            subject="robotic_suturing_activity",
            predicate="predicted_motion_primitive",
            value={
                "gesture_id": name,
                "description": SUTURING_GESTURE_DESCRIPTIONS[name],
                "segment_index": segment_index,
            },
            confidence=_rounded(np.mean(selected_probability[start:end])),
            start_s=_rounded(start_s),
            end_s=_rounded(end_s),
            grounding={
                "source": "held_out_subject_TCN_BiGRU_prediction",
                "fallible_model_observation": True,
                "ground_truth_transcript_used_for_this_inference": False,
                "not_a_task_verdict": True,
            },
        ))
    nonzero_counts = {
        name: count for name, count in count_by_class.items() if count > 0
    }
    nonzero_durations = {
        name: _rounded(duration) for name, duration in duration_by_class.items()
        if duration > 0
    }
    transitions: dict[str, int] = {}
    for left, right in zip(segment_rows, segment_rows[1:]):
        key = f"{GESTURE_CLASSES[left[0]]}->{GESTURE_CLASSES[right[0]]}"
        transitions[key] = transitions.get(key, 0) + 1
    facts.extend([
        GroundedEvidenceFact(
            fact_id="predicted_gesture_coverage",
            fact_type="attribute",
            subject="complete_trial",
            predicate="predicted_motion_primitive_coverage",
            value={
                "segment_count": len(segment_rows),
                "occurrence_count_by_gesture": nonzero_counts,
                "duration_seconds_by_gesture": nonzero_durations,
            },
            grounding={
                "source": "aggregation_of_OOF_predicted_segments",
                "absence_means_not_reliably_predicted_not_proven_absent": True,
                "not_a_task_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="predicted_transition_counts",
            fact_type="relation",
            subject="predicted_motion_primitive_sequence",
            predicate="adjacent_transition_counts",
            value=transitions,
            grounding={
                "source": "aggregation_of_OOF_predicted_segments",
                "sequence_relation_only_not_flow_quality": True,
                "not_a_task_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="gesture_prediction_uncertainty",
            fact_type="attribute",
            subject="small_temporal_model",
            predicate="selected_class_probability_summary",
            value={
                "mean": _rounded(np.mean(selected_probability)),
                "p10": _rounded(np.percentile(selected_probability, 10)),
                "fraction_below_0_5": _rounded(np.mean(selected_probability < 0.5)),
            },
            grounding={
                "source": "OOF_class_probabilities",
                "probability_is_gesture_confidence_not_skill_probability": True,
                "not_a_task_verdict": True,
            },
        ),
    ])
    return grounded_fact_payload(
        facts,
        modality="temporal",
        source_description=(
            "A compact TCN-BiGRU predicts a chronological sequence of observable "
            "JIGSAWS motion primitives for a held-out subject. The sequence and "
            "confidence are advisory facts; they contain no expert rating, operator "
            "identity, experience label, or skill verdict."
        ),
        provenance={
            "adapter": "jigsaws_oof_gesture_temporal_facts_v2",
            "sample_id": sample_id,
            "prediction_file_sha256": prediction_sha256,
            "source_fps": float(source_fps),
            "model_downsample": int(model_downsample),
            "held_out_subject_inference": True,
            "ground_truth_transcript_used_for_inference": False,
            "expert_grs_used": False,
            "self_reported_experience_used": False,
            "foundation_model_parameters_updated": False,
            "raw_task_verdict_exported": False,
        },
    )


def learned_gesture_plugin_evidence(
    prediction_path: Path,
    *,
    sample_id: str,
    source_fps: float = 30.0,
    model_downsample: int = 3,
    prediction_sha256: str | None = None,
) -> PluginEvidence:
    payload = learned_gesture_fact_payload(
        prediction_path,
        sample_id=sample_id,
        source_fps=source_fps,
        model_downsample=model_downsample,
        prediction_sha256=prediction_sha256,
    )
    plugin = PluginEvidence(
        plugin_id="jigsaws_oof_gesture_temporal_facts_v2",
        plugin_kind="temporal",
        description=(
            "Fact-only motion-primitive timeline from a subject-out small temporal model."
        ),
        payload=payload,
        foundation_model_parameters_updated=False,
    )
    plugin.validate_fact_only()
    return plugin
