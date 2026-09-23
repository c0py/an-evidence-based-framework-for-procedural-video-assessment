from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Dependency-free criterion metrics. `y_true` is binary at the chosen label granularity."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    pred = scores >= threshold
    tp, fp = np.sum(pred & y_true), np.sum(pred & ~y_true)
    fn, tn = np.sum(~pred & y_true), np.sum(~pred & ~y_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": float(precision), "sensitivity": float(recall), "specificity": float(specificity), "f1": float(f1), "accuracy": float((tp + tn) / max(len(y_true), 1))}


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    y_true, scores = np.asarray(y_true, dtype=float), np.asarray(scores, dtype=float)
    total = max(len(scores), 1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        mask = (scores >= lower) & (scores < upper if upper < 1 else scores <= upper)
        if mask.any():
            ece += mask.mean() * abs(scores[mask].mean() - y_true[mask].mean())
    return float(ece)


def isolated_high_score_rate(scores: np.ndarray, threshold: float = 0.68) -> float:
    scores = np.asarray(scores)
    high = scores >= threshold
    isolated = high & np.r_[True, ~high[:-1]] & np.r_[~high[1:], True]
    return float(isolated.sum() / max(high.sum(), 1))
