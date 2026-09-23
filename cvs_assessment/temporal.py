from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from .schema import CriterionEvidence, EvidenceInterval, ScorePoint


class StableEvidenceAggregator:
    """Causal median smoothing plus hysteresis and a minimum-duration rule."""

    def __init__(self, smoothing_seconds: float, on_threshold: float, off_threshold: float, min_stable_seconds: float, max_gap_seconds: float) -> None:
        if off_threshold > on_threshold:
            raise ValueError("off_threshold must not exceed on_threshold")
        self.smoothing_seconds = smoothing_seconds
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.min_stable_seconds = min_stable_seconds
        self.max_gap_seconds = max_gap_seconds

    def aggregate(self, points: list[ScorePoint]) -> tuple[list[ScorePoint], list[EvidenceInterval]]:
        smooth, summary = self.aggregate_structured(points)
        return smooth, summary.positive_intervals

    def aggregate_structured(self, points: list[ScorePoint]) -> tuple[list[ScorePoint], CriterionEvidence]:
        """Aggregate evidence without collapsing insufficient visibility into negative.

        Structured VLM observations retain their explicit state. Continuous
        checkpoint scores are deterministically converted to positive, negative,
        or unknown using the same hysteresis thresholds used for aggregation.
        """
        if not points:
            return [], CriterionEvidence()
        resolved = [self._resolve_state(point) for point in points]
        window: deque[ScorePoint] = deque()
        smooth: list[ScorePoint] = []
        for point in resolved:
            window.append(point)
            while window and point.time_s - window[0].time_s > self.smoothing_seconds:
                window.popleft()
            # Unknown observations carry no positive or negative claim. Excluding
            # them prevents occlusion from behaving like a confident zero.
            assessable = [p.score for p in window if p.evidence_state != "unknown"]
            median = float(np.median(assessable)) if assessable else 0.0
            state = point.evidence_state
            confidence = point.state_confidence
            if state == "unclassified":
                # Checkpoint tools emit continuous probabilities rather than an
                # explicit visibility state. Derive their auditable state only
                # after smoothing so the original hysteresis behavior is kept.
                state = "positive" if median >= 0.5 else "negative"
                confidence = median if state == "positive" else 1.0 - median
            smooth.append(ScorePoint(
                point.time_s, median, point.raw_score, point.quality,
                evidence_state=state,
                state_confidence=float(np.clip(confidence, 0.0, 1.0)),
            ))

        positive_intervals = self._extract_intervals(smooth, unknown_bridges=True)
        negative_signal = [
            ScorePoint(
                time_s=point.time_s,
                score=point.state_confidence if point.evidence_state == "negative" else 0.0,
                raw_score=point.state_confidence,
                quality=point.quality,
                evidence_state=point.evidence_state,
                state_confidence=point.state_confidence,
            )
            for point in smooth
        ]
        negative_intervals = self._extract_intervals(negative_signal, unknown_bridges=True)
        positive_count = sum(p.evidence_state == "positive" for p in smooth)
        negative_count = sum(p.evidence_state == "negative" for p in smooth)
        unknown_count = sum(p.evidence_state == "unknown" for p in smooth)
        negative_confidences = [p.state_confidence for p in smooth if p.evidence_state == "negative"]
        total = len(smooth)
        summary = CriterionEvidence(
            positive_intervals=positive_intervals,
            negative_intervals=negative_intervals,
            positive_point_count=positive_count,
            negative_point_count=negative_count,
            unknown_point_count=unknown_count,
            total_point_count=total,
            assessable_coverage=(positive_count + negative_count) / total if total else 0.0,
            explicit_negative_confidence=float(np.mean(negative_confidences)) if negative_confidences else 0.0,
        )
        return smooth, summary

    def _resolve_state(self, point: ScorePoint) -> ScorePoint:
        state = point.evidence_state
        confidence = point.state_confidence
        if state == "unclassified":
            if point.quality <= 0.25:
                state, confidence = "unknown", max(0.0, 1.0 - point.quality)
        return ScorePoint(
            point.time_s, point.score, point.raw_score, point.quality,
            evidence_state=state,
            state_confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    def _extract_intervals(self, smooth: list[ScorePoint], unknown_bridges: bool) -> list[EvidenceInterval]:
        candidates: list[list[ScorePoint]] = []
        active: list[ScorePoint] = []
        last_above_off: float | None = None
        for point in smooth:
            if point.evidence_state == "unknown":
                if active and unknown_bridges and last_above_off is not None and point.time_s - last_above_off <= self.max_gap_seconds:
                    active.append(point)
                elif active:
                    candidates.append(active)
                    active = []
                    last_above_off = None
                continue
            if not active:
                if point.score >= self.on_threshold:
                    active = [point]
                    last_above_off = point.time_s
            elif point.score >= self.off_threshold:
                active.append(point)
                last_above_off = point.time_s
            elif last_above_off is not None and point.time_s - last_above_off <= self.max_gap_seconds:
                # Preserve a short occlusion inside an otherwise stable interval.
                active.append(point)
            else:
                candidates.append(active)
                active = [point] if point.score >= self.on_threshold else []
                last_above_off = point.time_s if active else None
        if active:
            candidates.append(active)

        intervals = []
        for segment in candidates:
            duration = segment[-1].time_s - segment[0].time_s
            if duration < self.min_stable_seconds:
                continue
            assessable_segment = [p for p in segment if p.evidence_state != "unknown"]
            if not assessable_segment:
                continue
            supporting_segment = [p for p in assessable_segment if p.score >= self.off_threshold]
            # A single positive observation followed by unknown frames is not
            # temporal evidence, regardless of the wall-clock gap it spans.
            if len(supporting_segment) < 2:
                continue
            scores = np.array([p.score for p in assessable_segment])
            assessable_coverage = len(assessable_segment) / len(segment)
            # Prefer an interior representative frame, preventing a boundary spike from winning.
            center = len(assessable_segment) // 2
            interior = assessable_segment[
                max(0, center - len(assessable_segment) // 4):
                min(len(assessable_segment), center + len(assessable_segment) // 4 + 1)
            ]
            representative = max(interior, key=lambda p: p.score)
            intervals.append(EvidenceInterval(
                start_s=segment[0].time_s,
                end_s=segment[-1].time_s,
                confidence=float(
                    scores.mean()
                    * min(1.0, duration / (2 * self.min_stable_seconds))
                    * assessable_coverage
                ),
                mean_score=float(scores.mean()),
                duration_s=duration,
                representative_time_s=representative.time_s,
                supporting_point_count=len(supporting_segment),
                unknown_point_count=len(segment) - len(assessable_segment),
                assessable_coverage=assessable_coverage,
            ))
        return intervals


def stable_state_operator(
    points: list[ScorePoint], temporal_parameters: dict[str, float],
    temporal_policy: dict[str, Any] | None = None,
) -> tuple[list[ScorePoint], CriterionEvidence]:
    accepted = {
        key: float(temporal_parameters[key])
        for key in (
            "smoothing_seconds", "on_threshold", "off_threshold",
            "min_stable_seconds", "max_gap_seconds",
        )
    }
    return StableEvidenceAggregator(**accepted).aggregate_structured(points)


def persistent_state_transition_operator(
    points: list[ScorePoint], temporal_parameters: dict[str, float],
    temporal_policy: dict[str, Any] | None = None,
) -> tuple[list[ScorePoint], CriterionEvidence]:
    """Require an observed non-target state before a persistent target state."""
    smooth, evidence = stable_state_operator(
        points, temporal_parameters, temporal_policy,
    )
    resolved_negative_times = [
        point.time_s for point in smooth if point.evidence_state == "negative"
    ]
    if resolved_negative_times:
        first_source = min(resolved_negative_times)
        evidence.positive_intervals = [
            interval for interval in evidence.positive_intervals
            if first_source < interval.start_s
        ]
    elif (temporal_policy or {}).get("source_state"):
        evidence.positive_intervals = []
    return smooth, evidence


class TemporalOperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, Callable[..., tuple[list[ScorePoint], CriterionEvidence]]] = {}

    def register(self, name: str, operator: Callable[..., tuple[list[ScorePoint], CriterionEvidence]]) -> None:
        self._operators[name] = operator

    def get(self, name: str) -> Callable[..., tuple[list[ScorePoint], CriterionEvidence]]:
        if name not in self._operators:
            raise KeyError(f"Unknown temporal operator: {name}")
        return self._operators[name]

    def available(self) -> list[str]:
        return sorted(self._operators)


def default_temporal_operators() -> TemporalOperatorRegistry:
    registry = TemporalOperatorRegistry()
    registry.register("stable_state", stable_state_operator)
    registry.register("stable_state_before_anchor", stable_state_operator)
    registry.register("persistent_state_transition", persistent_state_transition_operator)
    return registry
