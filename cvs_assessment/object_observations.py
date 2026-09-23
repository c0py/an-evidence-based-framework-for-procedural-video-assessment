"""Task-neutral object observations and replaceable localization providers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectObservation:
    time_s: float
    semantic_type: str
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized_xyxy: tuple[float, float, float, float]
    area_fraction: float
    source: str
    mask_contours_normalized_xy: tuple[tuple[tuple[float, float], ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bbox_xyxy"] = list(self.bbox_xyxy)
        value["bbox_normalized_xyxy"] = list(self.bbox_normalized_xyxy)
        value["mask_contours_normalized_xy"] = [
            [list(point) for point in contour]
            for contour in self.mask_contours_normalized_xy
        ]
        return value


@dataclass(frozen=True)
class FrameObjectObservations:
    time_s: float
    frame_width: int
    frame_height: int
    observations: tuple[ObjectObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FrameObjectObservations":
        return cls(
            time_s=float(value["time_s"]),
            frame_width=int(value.get("frame_width", 1)),
            frame_height=int(value.get("frame_height", 1)),
            observations=tuple(
                ObjectObservation(
                    time_s=float(item.get("time_s", value["time_s"])),
                    semantic_type=str(item["semantic_type"]),
                    class_id=int(item.get("class_id", -1)),
                    confidence=float(item["confidence"]),
                    bbox_xyxy=tuple(float(x) for x in item["bbox_xyxy"]),
                    bbox_normalized_xyxy=tuple(
                        float(x) for x in item.get(
                            "bbox_normalized_xyxy", item["bbox_xyxy"],
                        )
                    ),
                    area_fraction=float(item["area_fraction"]),
                    source=str(item.get("source", "recorded_object_observation")),
                    mask_contours_normalized_xy=tuple(
                        tuple((float(point[0]), float(point[1])) for point in contour)
                        for contour in item.get("mask_contours_normalized_xy", [])
                    ),
                )
                for item in value.get("observations", [])
            ),
        )


class RecordedObjectObservationProvider:
    """Replay detector outputs, preserving the same provider contract as live models."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).resolve())
        payload = json.loads(Path(self.path).read_text(encoding="utf-8"))
        self.frames = {
            round(float(item["time_s"]), 6): FrameObjectObservations.from_dict(item)
            for item in payload["frames"]
        }
        self.checkpoint = self.path

    def observe(
        self, video_path: str | Path, timestamps: list[float],
    ) -> list[FrameObjectObservations]:
        missing = [time_s for time_s in timestamps if round(float(time_s), 6) not in self.frames]
        if missing:
            raise ValueError(
                f"Recorded observations {self.path} do not contain {len(missing)} requested "
                f"timestamps; first missing={missing[0]:.6f}s"
            )
        return [self.frames[round(float(time_s), 6)] for time_s in timestamps]


class ObjectLocalizerRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(self, provider_id: str, factory: Callable[[dict[str, Any]], Any]) -> None:
        if provider_id in self._factories:
            raise ValueError(f"Object localizer already registered: {provider_id}")
        self._factories[provider_id] = factory

    def build(self, provider_id: str, config: dict[str, Any]) -> Any:
        if provider_id not in self._factories:
            raise KeyError(
                f"Unknown object localizer {provider_id!r}; available={sorted(self._factories)}"
            )
        return self._factories[provider_id](config)

    def available(self) -> list[str]:
        return sorted(self._factories)


class YoloObjectObservationProvider:
    """Expose any Ultralytics detector through the generic observation schema."""

    def __init__(
        self, checkpoint: str, confidence: float = 0.15, iou: float = 0.45,
        image_size: int = 960, max_detections: int = 30, device: str = "0",
        batch_size: int = 32, source_name: str = "yolo_object_localizer",
    ) -> None:
        from ultralytics import YOLO

        self.checkpoint = str(Path(checkpoint).resolve())
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.max_detections = max_detections
        self.device = device
        self.batch_size = batch_size
        self.source_name = source_name
        self.model = YOLO(self.checkpoint)

    def observe(
        self, video_path: str | Path, timestamps: list[float],
    ) -> list[FrameObjectObservations]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video for object localization: {video_path}")
        output: list[FrameObjectObservations] = []
        try:
            for start in range(0, len(timestamps), self.batch_size):
                requested = timestamps[start:start + self.batch_size]
                frames, decoded_times = [], []
                for time_s in requested:
                    capture.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000)
                    ok, frame = capture.read()
                    if ok:
                        frames.append(frame)
                        decoded_times.append(float(time_s))
                if not frames:
                    continue
                results = self.model.predict(
                    frames, conf=self.confidence, iou=self.iou,
                    max_det=self.max_detections, imgsz=self.image_size,
                    device=self.device, verbose=False,
                )
                for time_s, frame, result in zip(decoded_times, frames, results):
                    height, width = frame.shape[:2]
                    observations = []
                    mask_values = None
                    if result.masks is not None and result.masks.data is not None:
                        mask_values = result.masks.data.detach().float().cpu().numpy()
                    if result.boxes is not None:
                        for box_index, box in enumerate(result.boxes):
                            score = float(box.conf[0])
                            class_id = int(box.cls[0])
                            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                            mask_contours: tuple[tuple[tuple[float, float], ...], ...] = ()
                            mask_area_fraction = None
                            if mask_values is not None and box_index < len(mask_values):
                                mask = cv2.resize(
                                    mask_values[box_index], (width, height),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                                binary = (mask >= 0.5).astype(np.uint8)
                                mask_area_fraction = float(binary.mean())
                                contours, _ = cv2.findContours(
                                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                )
                                normalized = []
                                for contour in contours:
                                    if cv2.contourArea(contour) < 4.0:
                                        continue
                                    perimeter = cv2.arcLength(contour, True)
                                    contour = cv2.approxPolyDP(
                                        contour, max(1.0, 0.0025 * perimeter), True,
                                    )
                                    normalized.append(tuple(
                                        (float(point[0][0]) / width, float(point[0][1]) / height)
                                        for point in contour
                                    ))
                                mask_contours = tuple(normalized)
                            observations.append(ObjectObservation(
                                time_s=time_s,
                                semantic_type=str(result.names[class_id]),
                                class_id=class_id,
                                confidence=score,
                                bbox_xyxy=(x1, y1, x2, y2),
                                bbox_normalized_xyxy=(
                                    x1 / width, y1 / height, x2 / width, y2 / height,
                                ),
                                area_fraction=(
                                    mask_area_fraction if mask_area_fraction is not None else
                                    max(0.0, x2 - x1) * max(0.0, y2 - y1)
                                    / max(1.0, float(width * height))
                                ),
                                source=self.source_name,
                                mask_contours_normalized_xy=mask_contours,
                            ))
                    output.append(FrameObjectObservations(
                        time_s=time_s, frame_width=width, frame_height=height,
                        observations=tuple(observations),
                    ))
        finally:
            capture.release()
        return output


def maximum_confidences(frame: FrameObjectObservations) -> dict[str, float]:
    output: dict[str, float] = {}
    for item in frame.observations:
        output[item.semantic_type] = max(
            output.get(item.semantic_type, 0.0), item.confidence,
        )
    return output


def bbox_iou(left: ObjectObservation, right: ObjectObservation) -> float:
    lx1, ly1, lx2, ly2 = left.bbox_xyxy
    rx1, ry1, rx2, ry2 = right.bbox_xyxy
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1),
    )
    union = (
        max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def maximum_tool_anatomy_iou(frame: FrameObjectObservations) -> float:
    tools = [item for item in frame.observations if item.semantic_type == "tool"]
    anatomy = [item for item in frame.observations if item.semantic_type != "tool"]
    return max((bbox_iou(tool, target) for tool in tools for target in anatomy), default=0.0)


def spatial_fusion_feature_names(
    criteria: list[str] | tuple[str, ...],
    object_classes: list[str] | tuple[str, ...],
) -> list[str]:
    """Canonical, checkpoint-stable schema for learned evidence fusion."""
    return (
        [f"small_score_{criterion}" for criterion in criteria]
        + [f"bbox_max_conf_{name}" for name in object_classes]
        + [f"bbox_count_scaled_{name}" for name in object_classes]
        + [f"bbox_area_fraction_{name}" for name in object_classes]
        + ["maximum_tool_anatomy_iou", "phase_progress"]
        + [f"bbox_candidate_{criterion}" for criterion in criteria]
    )


def _candidate_value(
    maxima: dict[str, float], rule: dict[str, Any],
) -> float:
    inputs = [maxima.get(str(name), 0.0) for name in rule.get("classes", [])]
    mode = str(rule.get("operator", "max"))
    if not inputs:
        return 0.0
    if mode == "geometric_mean":
        product = math.prod(inputs)
        return product ** (1.0 / len(inputs))
    if mode == "min":
        return min(inputs)
    if mode == "mean":
        return sum(inputs) / len(inputs)
    if mode == "max":
        return max(inputs)
    raise ValueError(f"Unsupported bbox candidate operator: {mode}")


def spatial_fusion_features(
    frame: FrameObjectObservations,
    small_scores: dict[str, float],
    criteria: list[str] | tuple[str, ...],
    object_classes: list[str] | tuple[str, ...],
    candidate_rules: dict[str, dict[str, Any]],
    phase_progress: float = 0.0,
) -> dict[str, float]:
    """Encode generic object observations into named fusion features."""
    grouped = {
        name: [item for item in frame.observations if item.semantic_type == name]
        for name in object_classes
    }
    maxima = {
        name: max((item.confidence for item in items), default=0.0)
        for name, items in grouped.items()
    }
    features = {f"small_score_{key}": float(small_scores[key]) for key in criteria}
    features.update({f"bbox_max_conf_{name}": maxima[name] for name in object_classes})
    features.update({
        f"bbox_count_scaled_{name}": min(1.0, len(grouped[name]) / 5.0)
        for name in object_classes
    })
    features.update({
        f"bbox_area_fraction_{name}": min(
            1.0, sum(item.area_fraction for item in grouped[name]),
        )
        for name in object_classes
    })
    features["maximum_tool_anatomy_iou"] = maximum_tool_anatomy_iou(frame)
    features["phase_progress"] = float(phase_progress)
    features.update({
        f"bbox_candidate_{criterion}": _candidate_value(
            maxima, candidate_rules.get(criterion, {}),
        )
        for criterion in criteria
    })
    return features


_DEFAULT_REGISTRY: ObjectLocalizerRegistry | None = None


def default_object_localizers() -> ObjectLocalizerRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = ObjectLocalizerRegistry()
        registry.register(
            "endoscapes_yolo",
            lambda cfg: YoloObjectObservationProvider(
                checkpoint=cfg["checkpoint"],
                confidence=float(cfg.get("confidence", 0.15)),
                iou=float(cfg.get("iou", 0.45)),
                image_size=int(cfg.get("image_size", 960)),
                max_detections=int(cfg.get("max_detections", 30)),
                device=str(cfg.get("device", "0")),
                batch_size=int(cfg.get("batch_size", 32)),
                source_name=str(cfg.get("source_name", "endoscapes_yolo_clean")),
            ),
        )
        registry.register(
            "recorded",
            lambda cfg: RecordedObjectObservationProvider(cfg["path"]),
        )
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
