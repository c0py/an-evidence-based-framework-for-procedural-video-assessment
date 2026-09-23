"""Task-neutral mask-track evidence and temporal evidence-anchor selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import cv2
import numpy as np

from .object_observations import FrameObjectObservations, ObjectObservation


def _bbox_iou(left: ObjectObservation, right: ObjectObservation) -> float:
    lx0, ly0, lx1, ly1 = left.bbox_normalized_xyxy
    rx0, ry0, rx1, ry1 = right.bbox_normalized_xyxy
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0.0, min(ly1, ry1) - max(ly0, ry0),
    )
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return float(intersection / union) if union else 0.0


def _center_distance(left: ObjectObservation, right: ObjectObservation) -> float:
    lx0, ly0, lx1, ly1 = left.bbox_normalized_xyxy
    rx0, ry0, rx1, ry1 = right.bbox_normalized_xyxy
    return float(math.hypot(
        0.5 * (lx0 + lx1 - rx0 - rx1),
        0.5 * (ly0 + ly1 - ry0 - ry1),
    ) / math.sqrt(2.0))


def _raster(item: ObjectObservation, width: int = 96, height: int = 54) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for contour in item.mask_contours_normalized_xy:
        points = np.asarray([
            [round(x * (width - 1)), round(y * (height - 1))] for x, y in contour
        ], dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask


def _mask_iou(left: ObjectObservation, right: ObjectObservation) -> float:
    left_mask, right_mask = _raster(left), _raster(right)
    union = np.logical_or(left_mask, right_mask).sum()
    return float(np.logical_and(left_mask, right_mask).sum() / union) if union else 0.0


def _mask_contact(left: ObjectObservation, right: ObjectObservation) -> float:
    left_mask, right_mask = _raster(left), _raster(right)
    if not left_mask.any() or not right_mask.any():
        return 0.0
    dilated = cv2.dilate(left_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return float(np.logical_and(dilated, right_mask).sum() / max(1, left_mask.sum()))


def _best(frame: FrameObjectObservations, semantic_type: str) -> ObjectObservation | None:
    candidates = [item for item in frame.observations if item.semantic_type == semantic_type]
    return max(candidates, key=lambda item: item.confidence, default=None)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _longest_run(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def mask_track_features(
    frames: Sequence[FrameObjectObservations],
    semantic_types: Iterable[str],
    semantic_pairs: Iterable[tuple[str, str]] = (),
) -> dict[str, float]:
    """Summarize predicted mask persistence and geometry over a frame sequence.

    Semantic types and pairs are supplied by a task package/compiled skill; this
    module contains no procedure- or criterion-specific branches.
    """

    if not frames:
        raise ValueError("At least one frame is required")
    times = [float(frame.time_s) for frame in frames]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Frames must be strictly time ordered")
    output: dict[str, float] = {}
    selected_by_type: dict[str, list[ObjectObservation | None]] = {}
    for semantic_type in semantic_types:
        selected = [_best(frame, semantic_type) for frame in frames]
        selected_by_type[semantic_type] = selected
        present = [item is not None for item in selected]
        observed = [item for item in selected if item is not None]
        confidences = [float(item.confidence) for item in observed]
        areas = [float(item.area_fraction) for item in observed]
        adjacent = [
            (left, right) for left, right in zip(selected, selected[1:])
            if left is not None and right is not None
        ]
        prefix = f"track_{semantic_type}"
        output[f"{prefix}_presence_ratio"] = float(np.mean(present))
        output[f"{prefix}_longest_run_ratio"] = _longest_run(present) / len(frames)
        output[f"{prefix}_mean_confidence"] = _mean(confidences)
        output[f"{prefix}_maximum_confidence"] = max(confidences, default=0.0)
        output[f"{prefix}_mean_area_fraction"] = _mean(areas)
        output[f"{prefix}_confidence_stability"] = (
            float(np.clip(1.0 - np.std(confidences) / max(1e-6, np.mean(confidences)), 0.0, 1.0))
            if confidences else 0.0
        )
        output[f"{prefix}_adjacent_continuity"] = (
            len(adjacent) / max(1, len(frames) - 1)
        )
        output[f"{prefix}_adjacent_bbox_iou"] = _mean([
            _bbox_iou(left, right) for left, right in adjacent
        ])
        output[f"{prefix}_adjacent_mask_iou"] = _mean([
            _mask_iou(left, right) for left, right in adjacent
        ])
        output[f"{prefix}_adjacent_center_motion"] = _mean([
            _center_distance(left, right) for left, right in adjacent
        ])

    for left_name, right_name in semantic_pairs:
        if left_name not in selected_by_type or right_name not in selected_by_type:
            raise ValueError("Every semantic pair type must also be in semantic_types")
        pairs = [
            (left, right)
            for left, right in zip(selected_by_type[left_name], selected_by_type[right_name])
            if left is not None and right is not None
        ]
        prefix = f"relation_{left_name}__{right_name}"
        output[f"{prefix}_copresence_ratio"] = len(pairs) / len(frames)
        output[f"{prefix}_mean_bbox_iou"] = _mean([_bbox_iou(a, b) for a, b in pairs])
        output[f"{prefix}_mean_center_distance"] = _mean([
            _center_distance(a, b) for a, b in pairs
        ])
        output[f"{prefix}_left_mask_contact"] = _mean([_mask_contact(a, b) for a, b in pairs])
        output[f"{prefix}_right_mask_contact"] = _mean([_mask_contact(b, a) for a, b in pairs])
    return output


def localization_quality(
    features: dict[str, float], required_semantic_types: Sequence[str],
) -> float:
    """Conservatively summarize whether required predicted tracks are usable."""

    if not required_semantic_types:
        return 0.0
    values = []
    for semantic_type in required_semantic_types:
        prefix = f"track_{semantic_type}"
        presence = float(features.get(f"{prefix}_presence_ratio", 0.0))
        run = float(features.get(f"{prefix}_longest_run_ratio", 0.0))
        confidence = float(features.get(f"{prefix}_mean_confidence", 0.0))
        stability = float(features.get(f"{prefix}_confidence_stability", 0.0))
        values.append(np.clip(
            presence * confidence * (0.5 + 0.25 * run + 0.25 * stability), 0.0, 1.0,
        ))
    # Every required semantic type matters; a geometric mean prevents one
    # excellent but irrelevant track from hiding a missing required object.
    return float(np.prod(np.asarray(values, dtype=np.float64)) ** (1.0 / len(values)))


@dataclass(frozen=True)
class TemporalEvidenceAnchor:
    role: str
    index: int
    time_s: float
    small_model_score: float
    localization_quality: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def select_temporal_evidence_anchors(
    timestamps_s: Sequence[float],
    small_model_scores: Sequence[float],
    localization_quality: Sequence[float],
    interval_s: tuple[float, float],
) -> list[TemporalEvidenceAnchor]:
    """Select contrastive and stable evidence roles for a frozen MLLM.

    The selected roles are pre-context, onset, stable interior, offset, and
    post-context when those regions exist.  Selection uses only tool predictions
    and the candidate interval, never expert labels.
    """

    times = np.asarray(timestamps_s, dtype=np.float64)
    scores = np.asarray(small_model_scores, dtype=np.float64)
    quality = np.asarray(localization_quality, dtype=np.float64)
    if times.ndim != 1 or scores.shape != times.shape or quality.shape != times.shape:
        raise ValueError("timestamps, scores, and localization quality must align")
    if len(times) == 0 or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("timestamps must be nonempty and strictly increasing")
    start_s, end_s = map(float, interval_s)
    if end_s < start_s:
        raise ValueError("Invalid candidate interval")

    chosen: list[tuple[str, int]] = []
    pre = np.flatnonzero(times < start_s)
    inside = np.flatnonzero((times >= start_s) & (times <= end_s))
    post = np.flatnonzero(times > end_s)
    if len(pre):
        chosen.append(("pre_context", int(pre[-1])))
    if len(inside):
        onset = int(inside[np.argmin(np.abs(times[inside] - start_s))])
        chosen.append(("onset", onset))
        combined = 0.65 * np.clip(scores[inside], 0.0, 1.0) + 0.35 * np.clip(
            quality[inside], 0.0, 1.0,
        )
        stable_order = inside[np.argsort(-combined, kind="stable")]
        stable = next((int(index) for index in stable_order if int(index) != onset), onset)
        chosen.append(("stable_interior", stable))
        offset = int(inside[np.argmin(np.abs(times[inside] - end_s))])
        chosen.append(("offset", offset))
    if len(post):
        chosen.append(("post_context", int(post[0])))

    output, seen = [], set()
    for role, index in chosen:
        if index in seen:
            # Preserve unique pixels/timestamps; the earlier role already
            # explains why this evidence point was selected.
            continue
        seen.add(index)
        output.append(TemporalEvidenceAnchor(
            role=role, index=index, time_s=float(times[index]),
            small_model_score=float(scores[index]),
            localization_quality=float(quality[index]),
        ))
    return output


def select_temporal_evidence_grid(
    timestamps_s: Sequence[float],
    small_model_scores: Sequence[float],
    localization_quality: Sequence[float],
    interval_s: tuple[float, float],
    *,
    frame_count: int = 7,
) -> list[TemporalEvidenceAnchor]:
    """Select an exact 7/13-frame contrastive grid for frozen-MLLM review."""

    if frame_count not in {7, 13}:
        raise ValueError("Temporal evidence grids support exactly 7 or 13 frames")
    times = np.asarray(timestamps_s, dtype=np.float64)
    scores = np.asarray(small_model_scores, dtype=np.float64)
    quality = np.asarray(localization_quality, dtype=np.float64)
    if times.ndim != 1 or scores.shape != times.shape or quality.shape != times.shape:
        raise ValueError("timestamps, scores, and localization quality must align")
    if len(times) < frame_count or np.any(np.diff(times) <= 0):
        raise ValueError("Not enough strictly ordered timestamps for the evidence grid")
    start_s, end_s = map(float, interval_s)
    if end_s < start_s:
        raise ValueError("Invalid candidate interval")
    quotas = (
        {"pre_context": 1, "onset": 2, "stable_interior": 2, "offset": 1, "post_context": 1}
        if frame_count == 7 else
        {"pre_context": 2, "onset": 2, "stable_interior": 5, "offset": 2, "post_context": 2}
    )
    pre = np.flatnonzero(times < start_s).tolist()
    inside = np.flatnonzero((times >= start_s) & (times <= end_s)).tolist()
    post = np.flatnonzero(times > end_s).tolist()
    chosen: dict[int, str] = {}

    def add(indices: Sequence[int], role: str, count: int) -> None:
        for index in indices:
            if index not in chosen:
                chosen[int(index)] = role
            if sum(value == role for value in chosen.values()) >= count:
                break

    add(list(reversed(pre)), "pre_context", quotas["pre_context"])
    add(inside, "onset", quotas["onset"])
    combined = 0.65 * np.clip(scores, 0.0, 1.0) + 0.35 * np.clip(quality, 0.0, 1.0)
    stable_candidates = sorted(inside, key=lambda index: (-combined[index], index))
    add(stable_candidates, "stable_interior", quotas["stable_interior"])
    add(list(reversed(inside)), "offset", quotas["offset"])
    add(post, "post_context", quotas["post_context"])

    # Short intervals or window boundaries may make a role quota impossible.
    # Fill deterministically with the most informative remaining timestamps;
    # assign a truthful region role from the candidate interval.
    remaining = sorted(
        (index for index in range(len(times)) if index not in chosen),
        key=lambda index: (-combined[index], index),
    )
    for index in remaining:
        if len(chosen) == frame_count:
            break
        if times[index] < start_s:
            role = "pre_context"
        elif times[index] > end_s:
            role = "post_context"
        elif end_s == start_s or times[index] <= start_s + 0.25 * (end_s - start_s):
            role = "onset"
        elif times[index] >= start_s + 0.75 * (end_s - start_s):
            role = "offset"
        else:
            role = "stable_interior"
        chosen[index] = role
    if len(chosen) != frame_count:
        raise RuntimeError("Unable to construct the requested temporal evidence grid")
    return [
        TemporalEvidenceAnchor(
            role=chosen[index], index=index, time_s=float(times[index]),
            small_model_score=float(scores[index]),
            localization_quality=float(quality[index]),
        )
        for index in sorted(chosen)
    ]
