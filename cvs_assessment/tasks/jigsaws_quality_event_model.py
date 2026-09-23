"""Weakly supervised temporal event localizer for JIGSAWS quality development.

The model is intentionally small. It consumes continuous robot kinematics plus
held-out-subject motion-primitive context and emits per-time risk attention for
needle handling and operational flow. Rule confirmation is separate from the
learned risk and remains an observable-event hypothesis, not a GRS verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .jigsaws_data import JIGSAWS_FPS
from .jigsaws_gesture_model import GESTURE_CLASSES


CONTINUOUS_FEATURE_NAMES = (
    "left_translation_speed",
    "right_translation_speed",
    "left_rotation_speed",
    "right_rotation_speed",
    "left_translation_acceleration",
    "right_translation_acceleration",
    "left_translation_jerk",
    "right_translation_jerk",
    "left_gripper_change_rate",
    "right_gripper_change_rate",
    "left_gripper_change_acceleration",
    "right_gripper_change_acceleration",
    "tool_tip_distance",
    "tool_tip_distance_change_rate",
    "tool_velocity_alignment",
    "left_direction_reversal",
    "right_direction_reversal",
    "both_tools_near_stationary",
    "both_tools_simultaneously_active",
)
FEATURE_NAMES = CONTINUOUS_FEATURE_NAMES + tuple(
    f"predicted_primitive_{name}" for name in GESTURE_CLASSES
)
FEATURE_DIMENSION = len(FEATURE_NAMES)
HANDLING_GESTURES = {"G2", "G4", "G8"}


@dataclass(frozen=True)
class QualityEventSequence:
    trial_id: str
    subject_id: str
    features: np.ndarray
    source_frame_indices: np.ndarray
    predicted_gesture_index: np.ndarray

    def validate(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != FEATURE_DIMENSION:
            raise ValueError("JIGSAWS quality-event features have an unexpected shape")
        if self.source_frame_indices.shape != (len(self.features),):
            raise ValueError("JIGSAWS quality-event frame indices are unaligned")
        if self.predicted_gesture_index.shape != (len(self.features),):
            raise ValueError("JIGSAWS quality-event gesture context is unaligned")
        if len(self.features) < 2 or not np.isfinite(self.features).all():
            raise ValueError("JIGSAWS quality-event sequence is empty or non-finite")
        if np.any(self.predicted_gesture_index < 0) or np.any(
            self.predicted_gesture_index >= len(GESTURE_CLASSES)
        ):
            raise ValueError("JIGSAWS quality-event gesture index is invalid")


def _smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    if width < 1 or width % 2 == 0:
        raise ValueError("JIGSAWS quality-event smoothing width must be positive and odd")
    array = np.asarray(values, dtype=np.float64)
    one_dimensional = array.ndim == 1
    if one_dimensional:
        array = array[:, None]
    radius = width // 2
    padded = np.pad(array, ((radius, radius), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
    cumulative = np.vstack([np.zeros((1, array.shape[1])), cumulative])
    result = (cumulative[width:] - cumulative[:-width]) / width
    return result[:, 0] if one_dimensional else result


def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1)


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = _norm(left) * _norm(right)
    result = np.zeros(len(left), dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as document:
        predicted = np.asarray(document["predicted_class"], dtype=np.int64)
        source_indices = np.asarray(document["source_frame_indices"], dtype=np.int64)
        classes = tuple(str(value) for value in document["gesture_classes"].tolist())
    if classes != GESTURE_CLASSES:
        raise ValueError("JIGSAWS quality-event primitive vocabulary changed")
    if predicted.ndim != 1 or source_indices.shape != predicted.shape:
        raise ValueError("JIGSAWS quality-event primitive prediction is unaligned")
    if np.any(source_indices[1:] <= source_indices[:-1]):
        raise ValueError("JIGSAWS quality-event primitive frames are not chronological")
    return predicted, source_indices


def quality_event_sequence(
    kinematics: np.ndarray,
    gesture_prediction_path: Path,
    *,
    trial_id: str,
    fps: float = JIGSAWS_FPS,
) -> QualityEventSequence:
    """Build observable 10-Hz features without reading GRS or transcripts."""
    values = np.asarray(kinematics, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 76:
        raise ValueError("JIGSAWS quality-event model expects [T,76] kinematics")
    if fps <= 0 or not np.isfinite(values).all():
        raise ValueError("Invalid JIGSAWS quality-event kinematics")
    predicted, source_indices = _load_prediction(gesture_prediction_path)
    if int(source_indices[-1]) >= len(values):
        raise ValueError("JIGSAWS quality-event prediction exceeds kinematics")

    left_position, right_position = values[:, 38:41], values[:, 57:60]
    left_velocity = _smooth(values[:, 50:53])
    right_velocity = _smooth(values[:, 69:72])
    left_rotation = _smooth(values[:, 53:56])
    right_rotation = _smooth(values[:, 72:75])
    left_speed, right_speed = _norm(left_velocity), _norm(right_velocity)
    left_acceleration = np.diff(
        left_velocity, axis=0, prepend=left_velocity[:1],
    ) * fps
    right_acceleration = np.diff(
        right_velocity, axis=0, prepend=right_velocity[:1],
    ) * fps
    left_jerk = np.diff(
        left_acceleration, axis=0, prepend=left_acceleration[:1],
    ) * fps
    right_jerk = np.diff(
        right_acceleration, axis=0, prepend=right_acceleration[:1],
    ) * fps
    left_gripper_rate = np.abs(np.diff(
        values[:, 56], prepend=values[0, 56],
    )) * fps
    right_gripper_rate = np.abs(np.diff(
        values[:, 75], prepend=values[0, 75],
    )) * fps
    left_gripper_acceleration = np.abs(np.diff(
        left_gripper_rate, prepend=left_gripper_rate[0],
    )) * fps
    right_gripper_acceleration = np.abs(np.diff(
        right_gripper_rate, prepend=right_gripper_rate[0],
    )) * fps
    tip_distance = _norm(left_position - right_position)
    tip_distance_rate = np.abs(np.diff(
        tip_distance, prepend=tip_distance[0],
    )) * fps
    velocity_alignment = _safe_cosine(left_velocity, right_velocity)

    def reversal(velocity: np.ndarray, speed: np.ndarray) -> np.ndarray:
        previous = np.vstack([velocity[:1], velocity[:-1]])
        cosine = _safe_cosine(previous, velocity)
        threshold = max(1e-12, 0.10 * float(np.percentile(speed, 95)))
        previous_speed = np.concatenate([[speed[0]], speed[:-1]])
        return ((cosine < -0.5) & (speed > threshold) & (previous_speed > threshold)).astype(float)

    left_reversal = reversal(left_velocity, left_speed)
    right_reversal = reversal(right_velocity, right_speed)
    combined_speed = (left_speed + right_speed) / 2.0
    stationary_threshold = max(
        1e-12, 0.08 * float(np.percentile(combined_speed, 95)),
    )
    left_active_threshold = max(1e-12, 0.10 * float(np.percentile(left_speed, 95)))
    right_active_threshold = max(1e-12, 0.10 * float(np.percentile(right_speed, 95)))
    stationary = ((left_speed <= stationary_threshold) & (right_speed <= stationary_threshold)).astype(float)
    simultaneous = ((left_speed > left_active_threshold) & (right_speed > right_active_threshold)).astype(float)
    continuous = np.column_stack([
        left_speed,
        right_speed,
        _norm(left_rotation),
        _norm(right_rotation),
        _norm(left_acceleration),
        _norm(right_acceleration),
        _norm(left_jerk),
        _norm(right_jerk),
        left_gripper_rate,
        right_gripper_rate,
        left_gripper_acceleration,
        right_gripper_acceleration,
        tip_distance,
        tip_distance_rate,
        velocity_alignment,
        left_reversal,
        right_reversal,
        stationary,
        simultaneous,
    ])[source_indices]
    one_hot = np.eye(len(GESTURE_CLASSES), dtype=np.float64)[predicted]
    parts = trial_id.rsplit("_", 1)
    if len(parts) != 2 or not parts[1] or not parts[1][0].isalpha():
        raise ValueError("Cannot derive JIGSAWS subject for quality-event sequence")
    sequence = QualityEventSequence(
        trial_id=trial_id,
        subject_id=parts[1][0],
        features=np.concatenate([continuous, one_hot], axis=1).astype(np.float32),
        source_frame_indices=source_indices.astype(np.int64),
        predicted_gesture_index=predicted.astype(np.int64),
    )
    sequence.validate()
    return sequence


def training_normalization(
    sequences: Iterable[QualityEventSequence],
) -> tuple[np.ndarray, np.ndarray]:
    rows = list(sequences)
    if not rows:
        raise ValueError("Cannot normalize empty JIGSAWS quality-event training data")
    values = np.concatenate([row.features.astype(np.float64) for row in rows], axis=0)
    mean = values.mean(axis=0)
    deviation = values.std(axis=0)
    deviation[deviation < 1e-6] = 1.0
    return mean.astype(np.float32), deviation.astype(np.float32)


class TemporalEventBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class JigsawsQualityEventTCN(nn.Module):
    """Small dense risk localizer; final GRS scoring is outside this model."""

    def __init__(
        self,
        *,
        input_dim: int = FEATURE_DIMENSION,
        hidden_dim: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.15,
        criterion_count: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("JIGSAWS quality-event hidden dimension must divide eight")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dilations = tuple(dilations)
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            TemporalEventBlock(hidden_dim, dilation, dropout)
            for dilation in self.dilations
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv1d(hidden_dim, criterion_count, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("JIGSAWS quality-event TCN expects [B,T,D]")
        hidden = self.input_projection(features.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return self.classifier(self.dropout(hidden)).transpose(1, 2)


def top_fraction_pool(logits: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
    if logits.ndim != 3 or not 0.0 < fraction <= 1.0:
        raise ValueError("Invalid JIGSAWS quality-event pooling input")
    count = max(1, int(round(logits.shape[1] * fraction)))
    return torch.topk(logits, count, dim=1).values.mean(dim=1)


def rolling_observables(
    sequence: QualityEventSequence,
    *,
    window_steps: int = 21,
) -> dict[str, np.ndarray]:
    if window_steps < 1 or window_steps % 2 == 0:
        raise ValueError("JIGSAWS event rule window must be positive and odd")
    by_name = {
        name: sequence.features[:, index].astype(np.float64)
        for index, name in enumerate(FEATURE_NAMES)
    }
    combined_speed = (
        by_name["left_translation_speed"] + by_name["right_translation_speed"]
    ) / 2.0
    return {
        "gripper_activity": _smooth(
            by_name["left_gripper_change_rate"]
            + by_name["right_gripper_change_rate"],
            window_steps,
        ),
        "rotation_activity": _smooth(
            by_name["left_rotation_speed"] + by_name["right_rotation_speed"],
            window_steps,
        ),
        "reversal_fraction": _smooth(
            np.maximum(
                by_name["left_direction_reversal"],
                by_name["right_direction_reversal"],
            ),
            window_steps,
        ),
        "stationary_fraction": _smooth(
            by_name["both_tools_near_stationary"], window_steps,
        ),
        "combined_speed": _smooth(combined_speed, window_steps),
        "tip_distance_change": _smooth(
            by_name["tool_tip_distance_change_rate"], window_steps,
        ),
    }


def training_rule_thresholds(
    sequences: Iterable[QualityEventSequence],
) -> dict[str, float]:
    rows = [rolling_observables(row) for row in sequences]
    if not rows:
        raise ValueError("Cannot derive JIGSAWS event rules from empty training data")
    return {
        "handling_gripper_activity_p80": float(np.percentile(
            np.concatenate([row["gripper_activity"] for row in rows]), 80,
        )),
        "handling_rotation_activity_p80": float(np.percentile(
            np.concatenate([row["rotation_activity"] for row in rows]), 80,
        )),
        "handling_reversal_fraction_p80": float(np.percentile(
            np.concatenate([row["reversal_fraction"] for row in rows]), 80,
        )),
        "flow_stationary_fraction_p80": float(np.percentile(
            np.concatenate([row["stationary_fraction"] for row in rows]), 80,
        )),
        "flow_reversal_fraction_p80": float(np.percentile(
            np.concatenate([row["reversal_fraction"] for row in rows]), 80,
        )),
    }


def localized_event_candidates(
    sequence: QualityEventSequence,
    probability: np.ndarray,
    thresholds: dict[str, float],
    *,
    criterion: str,
    maximum_events: int = 5,
    minimum_separation_seconds: float = 4.0,
    fps: float = JIGSAWS_FPS,
) -> list[dict[str, Any]]:
    """Return model-localized windows with separate observable rule checks."""
    risk = np.asarray(probability, dtype=np.float64)
    if risk.shape != (len(sequence.features),) or not np.isfinite(risk).all():
        raise ValueError("Invalid JIGSAWS localized risk sequence")
    if criterion not in {"handling", "flow"}:
        raise ValueError("Unknown JIGSAWS quality-event criterion")
    observables = rolling_observables(sequence)
    allowed = np.ones(len(risk), dtype=bool)
    if criterion == "handling":
        names = np.asarray([
            GESTURE_CLASSES[index] for index in sequence.predicted_gesture_index
        ])
        allowed = np.isin(names, sorted(HANDLING_GESTURES))
    ordered = np.argsort(-risk, kind="mergesort")
    minimum_steps = max(1, int(round(
        minimum_separation_seconds * fps
        / np.median(np.diff(sequence.source_frame_indices))
    )))
    selected: list[int] = []
    for index in ordered:
        if not allowed[index] or any(abs(int(index) - previous) < minimum_steps for previous in selected):
            continue
        selected.append(int(index))
        if len(selected) == maximum_events:
            break
    events: list[dict[str, Any]] = []
    for index in sorted(selected):
        frame = int(sequence.source_frame_indices[index])
        start_frame = max(0, frame - int(round(fps)))
        end_frame = frame + int(round(fps))
        if criterion == "handling":
            grip = float(observables["gripper_activity"][index])
            rotation = float(observables["rotation_activity"][index])
            reversal = float(observables["reversal_fraction"][index])
            grip_high = grip >= thresholds["handling_gripper_activity_p80"]
            rotation_high = rotation >= thresholds["handling_rotation_activity_p80"]
            reversal_high = (
                reversal > 0
                and reversal >= thresholds["handling_reversal_fraction_p80"]
            )
            confirmed = grip_high and (rotation_high or reversal_high)
            rule = "gripper_activity_plus_pose_correction"
            evidence = {
                "gripper_activity": round(grip, 5),
                "rotation_activity": round(rotation, 5),
                "reversal_fraction": round(reversal, 5),
                "gripper_activity_high": bool(grip_high),
                "rotation_activity_high": bool(rotation_high),
                "reversal_cluster": bool(reversal_high),
            }
        else:
            stationary = float(observables["stationary_fraction"][index])
            reversal = float(observables["reversal_fraction"][index])
            speed = float(observables["combined_speed"][index])
            after = float(np.mean(observables["combined_speed"][
                index + 1:min(len(risk), index + 11)
            ])) if index + 1 < len(risk) else speed
            stationary_high = (
                stationary >= max(0.15, thresholds["flow_stationary_fraction_p80"])
            )
            recovery = after >= max(1e-12, 1.5 * speed)
            reversal_high = (
                reversal > 0
                and reversal >= thresholds["flow_reversal_fraction_p80"]
            )
            confirmed = (stationary_high and recovery) or reversal_high
            rule = "stationary_then_recovery_or_reversal_cluster"
            evidence = {
                "stationary_fraction": round(stationary, 5),
                "reversal_fraction": round(reversal, 5),
                "combined_speed": round(speed, 6),
                "following_speed": round(after, 6),
                "stationary_high": bool(stationary_high),
                "recovery_burst": bool(recovery),
                "reversal_cluster": bool(reversal_high),
            }
        events.append({
            "criterion": criterion,
            "center_frame": frame,
            "center_seconds": round(frame / fps, 4),
            "start_seconds": round(start_frame / fps, 4),
            "end_seconds": round(end_frame / fps, 4),
            "model_risk_probability": round(float(risk[index]), 5),
            "predicted_primitive": GESTURE_CLASSES[
                int(sequence.predicted_gesture_index[index])
            ],
            "rule": rule,
            "rule_confirmed": bool(confirmed),
            "observable_evidence": evidence,
        })
    return events
