"""Task-neutral local rescue of sparse events from auxiliary evidence streams.

The rescue operator is intentionally unable to remove a primary model decision.
It can only promote a small number of temporally local, corroborated peaks.  This
keeps the auxiliary semantic model subordinate to the SOP monitor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LocalRescuePolicy:
    visual_threshold: float
    base_support_ratio: float
    top_k: int
    smoothing_points: int
    radius_points: int
    min_peak_distance_points: int
    threshold_margin: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def centered_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Centered moving mean with edge-aware normalization."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if width < 1 or width % 2 == 0:
        raise ValueError("width must be a positive odd integer")
    kernel = np.ones(width, dtype=np.float64)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values), kernel, mode="same")
    return numerator / denominator


def apply_topk_local_rescue(
    base_scores: np.ndarray,
    visual_scores: np.ndarray,
    on_threshold: float,
    policy: LocalRescuePolicy,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Return a monotonic score update and an audit trail of selected peaks."""
    base = np.asarray(base_scores, dtype=np.float64)
    visual = np.asarray(visual_scores, dtype=np.float64)
    if base.ndim != 1 or visual.ndim != 1 or len(base) != len(visual):
        raise ValueError("base_scores and visual_scores must be aligned 1-D arrays")
    if not len(base) or policy.top_k <= 0:
        return base.copy(), []

    smooth = centered_mean(visual, policy.smoothing_points)
    support_radius = policy.smoothing_points // 2
    candidates: list[tuple[float, int, float]] = []
    for index, score in enumerate(smooth):
        left_value = smooth[index - 1] if index else -np.inf
        right_value = smooth[index + 1] if index + 1 < len(smooth) else -np.inf
        if score < left_value or score < right_value:
            continue
        if score < policy.visual_threshold or base[index] >= on_threshold:
            continue
        left = max(0, index - support_radius)
        right = min(len(base), index + support_radius + 1)
        base_support = float(np.max(base[left:right]) / max(on_threshold, 1e-8))
        if base_support < policy.base_support_ratio:
            continue
        candidates.append((float(score), index, base_support))

    selected: list[tuple[float, int, float]] = []
    for candidate in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if any(
            abs(candidate[1] - prior[1]) < policy.min_peak_distance_points
            for prior in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= policy.top_k:
            break

    fused = base.copy()
    promoted_score = min(1.0, on_threshold * (1.0 + policy.threshold_margin))
    actions = []
    for visual_peak, index, base_support in sorted(selected, key=lambda item: item[1]):
        left = max(0, index - policy.radius_points)
        right = min(len(fused), index + policy.radius_points + 1)
        fused[left:right] = np.maximum(fused[left:right], promoted_score)
        actions.append({
            "peak_index": index,
            "window_start_index": left,
            "window_end_index_exclusive": right,
            "visual_peak": visual_peak,
            "base_support_ratio": base_support,
        })
    return fused, actions
