"""Small subject-out temporal gesture model for JIGSAWS kinematics.

The model predicts observable motion primitives. It is intentionally unable to
emit a GRS score, experience level, or final skill verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .jigsaws_data import GestureInterval, load_kinematics, load_transcription


GESTURE_CLASSES = (
    "BG", "G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11",
)
GESTURE_TO_INDEX = {value: index for index, value in enumerate(GESTURE_CLASSES)}
SLAVE_KINEMATIC_SLICE = slice(38, 76)


@dataclass(frozen=True)
class GestureSequence:
    trial_id: str
    subject_id: str
    features: np.ndarray
    labels: np.ndarray
    source_frame_indices: np.ndarray

    def validate(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != 38:
            raise ValueError("JIGSAWS gesture features must have shape [T,38]")
        if self.labels.shape != (len(self.features),):
            raise ValueError("JIGSAWS gesture labels do not align with features")
        if self.source_frame_indices.shape != (len(self.features),):
            raise ValueError("JIGSAWS source frame indices do not align")
        if not np.isfinite(self.features).all():
            raise ValueError("JIGSAWS gesture features contain non-finite values")
        if np.any(self.labels < 0) or np.any(self.labels >= len(GESTURE_CLASSES)):
            raise ValueError("JIGSAWS gesture label is outside the frozen vocabulary")


def frame_labels(
    frame_count: int, intervals: Iterable[GestureInterval],
) -> np.ndarray:
    """Convert one-based inclusive release intervals to dense class indices."""
    if frame_count < 1:
        raise ValueError("JIGSAWS frame count must be positive")
    labels = np.zeros(frame_count, dtype=np.int64)
    for interval in intervals:
        interval.validate()
        if interval.end_frame > frame_count:
            raise ValueError("JIGSAWS gesture interval exceeds the kinematic sequence")
        if interval.gesture_id not in GESTURE_TO_INDEX:
            raise ValueError(f"Gesture is outside the Suturing vocabulary: {interval.gesture_id}")
        labels[interval.start_frame - 1:interval.end_frame] = GESTURE_TO_INDEX[
            interval.gesture_id
        ]
    return labels


def load_gesture_sequence(
    kinematics_path: Path,
    transcription_path: Path,
    *,
    downsample: int = 3,
) -> GestureSequence:
    if downsample < 1:
        raise ValueError("JIGSAWS downsample must be positive")
    trial_id = Path(kinematics_path).stem
    parts = trial_id.rsplit("_", 1)
    if len(parts) != 2 or not parts[1] or not parts[1][0].isalpha():
        raise ValueError(f"Cannot infer JIGSAWS subject from {trial_id}")
    full = load_kinematics(kinematics_path)
    dense_labels = frame_labels(len(full), load_transcription(transcription_path))
    source_indices = np.arange(0, len(full), downsample, dtype=np.int64)
    sequence = GestureSequence(
        trial_id=trial_id,
        subject_id=parts[1][0],
        features=full[source_indices, SLAVE_KINEMATIC_SLICE].astype(np.float32),
        labels=dense_labels[source_indices],
        source_frame_indices=source_indices,
    )
    sequence.validate()
    return sequence


def discover_suturing_sequences(
    dataset_root: Path, *, downsample: int = 3,
) -> list[GestureSequence]:
    """Discover from transcripts, deliberately avoiding the GRS meta file."""
    task_root = Path(dataset_root) / "Suturing"
    transcription_root = task_root / "transcriptions"
    kinematics_root = task_root / "kinematics" / "AllGestures"
    output = []
    for transcription in sorted(transcription_root.glob("Suturing_*.txt")):
        kinematics = kinematics_root / transcription.name
        if not kinematics.is_file():
            raise FileNotFoundError(kinematics)
        output.append(load_gesture_sequence(
            kinematics, transcription, downsample=downsample,
        ))
    if not output or len({row.trial_id for row in output}) != len(output):
        raise ValueError("JIGSAWS Suturing gesture cohort is empty or duplicated")
    return output


def training_normalization(
    sequences: Iterable[GestureSequence],
) -> tuple[np.ndarray, np.ndarray]:
    rows = list(sequences)
    if not rows:
        raise ValueError("Cannot normalize an empty JIGSAWS training fold")
    values = np.concatenate([row.features.astype(np.float64) for row in rows], axis=0)
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return mean.astype(np.float32), standard_deviation.astype(np.float32)


def class_weights(
    sequences: Iterable[GestureSequence], *, maximum: float = 5.0,
) -> np.ndarray:
    counts = np.zeros(len(GESTURE_CLASSES), dtype=np.float64)
    for sequence in sequences:
        counts += np.bincount(sequence.labels, minlength=len(GESTURE_CLASSES))
    if np.any(counts == 0):
        raise ValueError(f"A JIGSAWS training fold lacks gesture classes: {counts.tolist()}")
    inverse_square_root = np.sqrt(counts.sum() / counts)
    weights = inverse_square_root / inverse_square_root.mean()
    return np.minimum(weights, maximum).astype(np.float32)


class GestureResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3,
                padding=dilation, dilation=dilation,
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class JigsawsGestureTCN(nn.Module):
    """Compact offline TCN producing dense motion-primitive logits."""

    def __init__(
        self,
        input_dim: int = 38,
        hidden_dim: int = 96,
        class_count: int = len(GESTURE_CLASSES),
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("JIGSAWS TCN hidden_dim must be divisible by eight")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_count = class_count
        self.dilations = tuple(dilations)
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            GestureResidualBlock(hidden_dim, dilation, dropout)
            for dilation in self.dilations
        ])
        self.classifier = nn.Conv1d(hidden_dim, class_count, kernel_size=1)

    @property
    def receptive_field_steps(self) -> int:
        return 1 + 2 * sum(self.dilations)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("JIGSAWS TCN expects features with shape [B,T,D]")
        hidden = self.input_projection(features.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return self.classifier(hidden).transpose(1, 2)


def stable_class_filter(labels: np.ndarray, width: int = 7) -> np.ndarray:
    """Deterministic centered majority filter for dense class predictions."""
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or width < 1 or width % 2 == 0:
        raise ValueError("Gesture majority filter needs a 1-D sequence and odd width")
    radius = width // 2
    output = values.copy()
    for index in range(len(values)):
        start, end = max(0, index - radius), min(len(values), index + radius + 1)
        counts = np.bincount(values[start:end], minlength=len(GESTURE_CLASSES))
        output[index] = int(np.argmax(counts))
    return output

