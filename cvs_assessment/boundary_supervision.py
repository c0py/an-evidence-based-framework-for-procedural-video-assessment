"""Task-neutral boundary-aware supervision for procedural state intervals.

The helpers in this module operate on timestamped state intervals and make no
assumptions about a particular procedure or criterion.  They are intentionally
separate from temporal inference: these targets are used only while training an
evidence model, whereas interval formation remains owned by the temporal
aggregator at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F


class BoundaryZone(IntEnum):
    """Mutually exclusive provenance for a generated target."""

    BACKGROUND = 0
    OUTER_BOUNDARY = 1
    INNER_BOUNDARY = 2
    INTERIOR_POSITIVE = 3
    PARTIAL = 4


@dataclass(frozen=True)
class BoundaryTargetConfig:
    """Cadence-relative target construction parameters.

    Margins are expressed as cadence multiples so an experiment can transfer
    across sampling rates without embedding task-specific seconds in the model.
    """

    cadence_s: float
    inner_margin_steps: float = 2.0
    outer_margin_steps: float = 2.0
    boundary_target: float = 0.5
    partial_target: float = 0.25

    def __post_init__(self) -> None:
        if self.cadence_s <= 0:
            raise ValueError("cadence_s must be positive")
        if self.inner_margin_steps <= 0 or self.outer_margin_steps <= 0:
            raise ValueError("Boundary margins must be positive")
        if not 0.0 < self.boundary_target < 1.0:
            raise ValueError("boundary_target must be strictly between zero and one")
        if not 0.0 <= self.partial_target < self.boundary_target:
            raise ValueError("partial_target must be below boundary_target")

    @property
    def inner_margin_s(self) -> float:
        return float(self.cadence_s * self.inner_margin_steps)

    @property
    def outer_margin_s(self) -> float:
        return float(self.cadence_s * self.outer_margin_steps)


def validate_train_only_scope(
    *,
    split_train_ids: Iterable[int],
    split_development_ids: Iterable[int],
    split_test_ids: Iterable[int],
    artifact_video_ids: Iterable[int],
    artifact_test_ids_loaded: Iterable[int] = (),
    artifact_test_labels_accessed: bool = False,
) -> None:
    """Reject any train artifact that contains development or sealed-test videos."""

    train = set(map(int, split_train_ids))
    development = set(map(int, split_development_ids))
    test = set(map(int, split_test_ids))
    artifact = set(map(int, artifact_video_ids))
    if train & development or train & test or development & test:
        raise ValueError("Formal split partitions overlap")
    if artifact != train:
        raise ValueError(
            f"Training artifact scope mismatch: missing={sorted(train-artifact)}, "
            f"extra={sorted(artifact-train)}"
        )
    if artifact & (development | test):
        raise ValueError("Training artifact contains development or sealed-test videos")
    if set(map(int, artifact_test_ids_loaded)) or artifact_test_labels_accessed:
        raise ValueError("Training artifact reports sealed-test access")


def _merged(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    output: list[list[float]] = []
    for start, end in ordered:
        if output and start <= output[-1][1]:
            output[-1][1] = max(output[-1][1], end)
        else:
            output.append([start, end])
    return [(start, end) for start, end in output]


def boundary_targets(
    timestamps_s: Sequence[float],
    state_intervals: Sequence[tuple[float, float, int]],
    config: BoundaryTargetConfig,
) -> dict[str, np.ndarray]:
    """Create soft full-state targets and target provenance.

    State ``2`` denotes a fully satisfied interval and state ``1`` denotes a
    partial/support interval.  Full intervals dominate partial intervals.  At a
    full-state boundary the target is ``boundary_target``; it rises toward one
    across the eroded inner band and falls toward zero across the outer band.
    """

    times = np.asarray(timestamps_s, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("timestamps_s must be one-dimensional")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps_s must be strictly increasing")

    full = _merged((start, end) for start, end, state in state_intervals if int(state) >= 2)
    partial = _merged((start, end) for start, end, state in state_intervals if int(state) == 1)
    targets = np.zeros(len(times), dtype=np.float32)
    zones = np.full(len(times), int(BoundaryZone.BACKGROUND), dtype=np.int64)
    nearest_boundary_distance_s = np.full(len(times), np.inf, dtype=np.float32)

    for index, time_s in enumerate(times):
        best_target = 0.0
        best_zone = BoundaryZone.BACKGROUND
        best_distance = np.inf
        for start, end in full:
            if start <= time_s <= end:
                distance = min(time_s - start, end - time_s)
                if distance >= config.inner_margin_s:
                    candidate = 1.0
                    zone = BoundaryZone.INTERIOR_POSITIVE
                else:
                    fraction = max(0.0, distance / config.inner_margin_s)
                    candidate = config.boundary_target + (1.0 - config.boundary_target) * fraction
                    zone = BoundaryZone.INNER_BOUNDARY
            else:
                distance = min(abs(time_s - start), abs(time_s - end))
                if distance <= config.outer_margin_s:
                    fraction = max(0.0, 1.0 - distance / config.outer_margin_s)
                    candidate = config.boundary_target * fraction
                    zone = BoundaryZone.OUTER_BOUNDARY
                else:
                    continue
            if candidate > best_target or (
                candidate > 0.0 and candidate == best_target and distance < best_distance
            ):
                best_target, best_zone, best_distance = candidate, zone, distance

        if best_zone == BoundaryZone.BACKGROUND and any(
            start <= time_s <= end for start, end in partial
        ):
            best_target = config.partial_target
            best_zone = BoundaryZone.PARTIAL

        targets[index] = float(best_target)
        zones[index] = int(best_zone)
        nearest_boundary_distance_s[index] = float(best_distance)

    return {
        "targets": targets,
        "zones": zones,
        "nearest_full_boundary_distance_s": nearest_boundary_distance_s,
    }


def adjacent_sequence_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    transition_delta: float = 0.05,
    monotonic_margin_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return sequence consistency and boundary monotonicity losses.

    Inputs have shape ``[batch, sequence]``.  Consistency matches adjacent
    probability changes to adjacent soft-target changes.  Monotonicity adds a
    directed logit-ranking constraint only where the target is genuinely rising
    or falling, so onset/offset transitions are not accidentally smoothed away.
    """

    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must have matching [batch, sequence] shape")
    if logits.shape[1] < 2:
        zero = logits.sum() * 0.0
        return {"consistency": zero, "monotonicity": zero, "adjacent_pair_count": zero}
    if valid_mask is None:
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
    if valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must match logits")

    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    target_delta_values = targets[:, 1:] - targets[:, :-1]
    probability_delta = torch.sigmoid(logits[:, 1:]) - torch.sigmoid(logits[:, :-1])
    consistency_terms = F.smooth_l1_loss(
        probability_delta, target_delta_values, reduction="none",
    )
    consistency = (
        consistency_terms[pair_mask].mean() if pair_mask.any() else logits.sum() * 0.0
    )

    transition_mask = pair_mask & (target_delta_values.abs() >= transition_delta)
    signed_logit_delta = (
        (logits[:, 1:] - logits[:, :-1]) * target_delta_values.sign()
    )
    required_margin = monotonic_margin_scale * target_delta_values.abs()
    monotonic_terms = torch.relu(required_margin - signed_logit_delta)
    monotonicity = (
        monotonic_terms[transition_mask].mean()
        if transition_mask.any() else logits.sum() * 0.0
    )
    return {
        "consistency": consistency,
        "monotonicity": monotonicity,
        "adjacent_pair_count": pair_mask.sum().to(logits.dtype),
    }
