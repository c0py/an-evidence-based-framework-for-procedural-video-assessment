"""Second frozen JIGSAWS temporal fact extractor.

This module only predicts observable gesture primitives from robot motion.  It
cannot read or emit expert GRS scores, experience levels, or final skill labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .jigsaws_data import load_kinematics, load_transcription
from .jigsaws_gesture_model import (
    GESTURE_CLASSES,
    frame_labels,
    stable_class_filter,
)


RAW_KINEMATIC_DIMENSION = 76
MODEL_FEATURE_DIMENSION = 152


@dataclass(frozen=True)
class MotionGestureSequence:
    trial_id: str
    subject_id: str
    features: np.ndarray
    labels: np.ndarray
    source_frame_indices: np.ndarray

    def validate(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != MODEL_FEATURE_DIMENSION:
            raise ValueError("JIGSAWS v2 gesture features must have shape [T,152]")
        if self.labels.shape != (len(self.features),):
            raise ValueError("JIGSAWS v2 gesture labels do not align with features")
        if self.source_frame_indices.shape != (len(self.features),):
            raise ValueError("JIGSAWS v2 source frame indices do not align")
        if not np.isfinite(self.features).all():
            raise ValueError("JIGSAWS v2 gesture features contain non-finite values")
        if np.any(self.labels < 0) or np.any(self.labels >= len(GESTURE_CLASSES)):
            raise ValueError("JIGSAWS v2 gesture label is outside the frozen vocabulary")


def motion_features(values: np.ndarray, source_fps: float = 30.0) -> np.ndarray:
    """Combine the released master/slave states with first-order motion."""
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != RAW_KINEMATIC_DIMENSION:
        raise ValueError("JIGSAWS raw kinematics must have shape [T,76]")
    if source_fps <= 0:
        raise ValueError("JIGSAWS source FPS must be positive")
    delta = np.empty_like(raw)
    delta[0] = 0.0
    delta[1:] = (raw[1:] - raw[:-1]) * source_fps
    return np.concatenate([raw, delta], axis=1).astype(np.float32)


def load_motion_gesture_sequence(
    kinematics_path: Path,
    transcription_path: Path,
    *,
    downsample: int = 3,
    source_fps: float = 30.0,
) -> MotionGestureSequence:
    if downsample < 1:
        raise ValueError("JIGSAWS downsample must be positive")
    trial_id = Path(kinematics_path).stem
    parts = trial_id.rsplit("_", 1)
    if len(parts) != 2 or not parts[1] or not parts[1][0].isalpha():
        raise ValueError(f"Cannot infer JIGSAWS subject from {trial_id}")
    raw = load_kinematics(kinematics_path)
    labels = frame_labels(len(raw), load_transcription(transcription_path))
    source_indices = np.arange(0, len(raw), downsample, dtype=np.int64)
    # Derivatives are computed before downsampling so that their time scale is
    # fixed by the released 30-Hz signal rather than the model sampling rate.
    full_features = motion_features(raw, source_fps=source_fps)
    sequence = MotionGestureSequence(
        trial_id=trial_id,
        subject_id=parts[1][0],
        features=full_features[source_indices],
        labels=labels[source_indices],
        source_frame_indices=source_indices,
    )
    sequence.validate()
    return sequence


def discover_suturing_motion_sequences(
    dataset_root: Path,
    *,
    downsample: int = 3,
    source_fps: float = 30.0,
) -> list[MotionGestureSequence]:
    """Discover sequences without opening the task meta/GRS file."""
    task_root = Path(dataset_root) / "Suturing"
    transcription_root = task_root / "transcriptions"
    kinematics_root = task_root / "kinematics" / "AllGestures"
    output = []
    for transcription in sorted(transcription_root.glob("Suturing_*.txt")):
        kinematics = kinematics_root / transcription.name
        if not kinematics.is_file():
            raise FileNotFoundError(kinematics)
        output.append(load_motion_gesture_sequence(
            kinematics,
            transcription,
            downsample=downsample,
            source_fps=source_fps,
        ))
    if not output or len({row.trial_id for row in output}) != len(output):
        raise ValueError("JIGSAWS Suturing v2 cohort is empty or duplicated")
    return output


def training_normalization(
    sequences: Iterable[MotionGestureSequence],
) -> tuple[np.ndarray, np.ndarray]:
    rows = list(sequences)
    if not rows:
        raise ValueError("Cannot normalize an empty JIGSAWS v2 training fold")
    values = np.concatenate([row.features.astype(np.float64) for row in rows], axis=0)
    mean = values.mean(axis=0)
    deviation = values.std(axis=0)
    deviation[deviation < 1e-6] = 1.0
    return mean.astype(np.float32), deviation.astype(np.float32)


def balanced_class_weights(
    sequences: Iterable[MotionGestureSequence],
    *,
    exponent: float = 0.75,
    maximum: float = 8.0,
) -> np.ndarray:
    """Training-fold-only weights with a fixed, bounded imbalance correction."""
    counts = np.zeros(len(GESTURE_CLASSES), dtype=np.float64)
    for sequence in sequences:
        counts += np.bincount(sequence.labels, minlength=len(GESTURE_CLASSES))
    if np.any(counts == 0):
        raise ValueError(f"A JIGSAWS v2 training fold lacks classes: {counts.tolist()}")
    weights = np.power(counts.sum() / counts, exponent)
    weights /= weights.mean()
    weights = np.minimum(weights, maximum)
    weights /= weights.mean()
    return weights.astype(np.float32)


class TemporalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
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


class JigsawsGestureTCNBiGRU(nn.Module):
    """Compact offline TCN-BiGRU that emits dense primitive logits."""

    def __init__(
        self,
        input_dim: int = MODEL_FEATURE_DIMENSION,
        hidden_dim: int = 128,
        recurrent_dim: int = 64,
        class_count: int = len(GESTURE_CLASSES),
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("JIGSAWS v2 hidden_dim must be divisible by eight")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.recurrent_dim = recurrent_dim
        self.class_count = class_count
        self.dilations = tuple(dilations)
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(hidden_dim, dilation, dropout)
            for dilation in self.dilations
        ])
        self.recurrent = nn.GRU(
            input_size=hidden_dim,
            hidden_size=recurrent_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * recurrent_dim, class_count)

    @property
    def receptive_field_steps(self) -> int:
        return 1 + 2 * sum(self.dilations)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("JIGSAWS v2 model expects features with shape [B,T,152]")
        hidden = self.input_projection(features.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        hidden, _ = self.recurrent(hidden.transpose(1, 2))
        return self.classifier(self.output_dropout(hidden))


class FocalCrossEntropy(nn.Module):
    """Fixed weighted focal loss used only on training-subject labels."""

    def __init__(
        self,
        class_weight: torch.Tensor,
        *,
        gamma: float = 1.5,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weight", class_weight)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_target = target.reshape(-1)
        valid = flat_target != self.ignore_index
        if not torch.any(valid):
            return flat_logits.sum() * 0.0
        valid_logits = flat_logits[valid]
        valid_target = flat_target[valid]
        log_probability = torch.log_softmax(valid_logits, dim=-1)
        target_log_probability = log_probability.gather(1, valid_target[:, None]).squeeze(1)
        target_probability = target_log_probability.exp()
        weight = self.class_weight[valid_target]
        loss = -weight * torch.pow(1.0 - target_probability, self.gamma) * target_log_probability
        return loss.mean()


def filtered_prediction(probabilities: np.ndarray, width: int = 7) -> np.ndarray:
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != len(GESTURE_CLASSES):
        raise ValueError("JIGSAWS v2 probabilities must have shape [T,C]")
    return stable_class_filter(values.argmax(axis=-1), width=width)
