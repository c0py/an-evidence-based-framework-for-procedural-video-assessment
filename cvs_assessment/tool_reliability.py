"""Task-neutral reliability policies learned outside the evaluated sample."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HighPrecisionToolPolicy:
    threshold: float | None
    estimated_precision: float | None
    estimated_recall: float | None
    accepted_count: int
    positive_count: int
    minimum_precision: float
    minimum_accepted_count: int

    @property
    def enabled(self) -> bool:
        return self.threshold is not None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {**asdict(self), "enabled": self.enabled}


def fit_high_precision_tool_policy(
    labels: np.ndarray, scores: np.ndarray, *, minimum_precision: float = 0.75,
    minimum_accepted_count: int = 25,
) -> HighPrecisionToolPolicy:
    """Choose the highest-recall threshold satisfying a precision floor."""
    truth = np.asarray(labels).reshape(-1) >= 0.5
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.shape != values.shape or not 0.0 < minimum_precision <= 1.0:
        raise ValueError("Reliability labels/scores or precision floor are invalid")
    if not np.isfinite(values).all():
        raise ValueError("Reliability scores must be finite")
    positive_count = int(truth.sum())
    order = np.argsort(-values, kind="stable")
    sorted_values, sorted_truth = values[order], truth[order]
    cumulative_tp = np.cumsum(sorted_truth)
    candidates = []
    for index in range(len(sorted_values)):
        if index + 1 < len(sorted_values) and sorted_values[index + 1] == sorted_values[index]:
            continue
        accepted = index + 1
        if accepted < minimum_accepted_count:
            continue
        tp = int(cumulative_tp[index])
        precision = tp / accepted
        recall = tp / positive_count if positive_count else 0.0
        if precision >= minimum_precision:
            candidates.append((recall, precision, float(sorted_values[index]), accepted))
    if not candidates:
        return HighPrecisionToolPolicy(
            None, None, None, 0, positive_count,
            float(minimum_precision), int(minimum_accepted_count),
        )
    recall, precision, threshold, accepted = max(candidates)
    return HighPrecisionToolPolicy(
        threshold, precision, recall, accepted, positive_count,
        float(minimum_precision), int(minimum_accepted_count),
    )
