"""Structured visual evidence packs for sparse MLLM verification.

This module deliberately has no dependency on the assessment pipeline.  It can
turn frozen object-localizer observations and real video frames into a sequence
of auditable triptychs (raw frame, predicted boxes, criterion-specific ROI),
build an OpenAI-compatible multimodal request, and strictly validate the four
state response used by Evidence Pack v1.

Predicted boxes are visual hints only.  Both the prompt and the rendered pack
retain the raw frame so that an MLLM can reject an incorrect localization.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .object_observations import FrameObjectObservations


EVIDENCE_STATES = frozenset({"full", "partial", "absent", "unobservable"})
VISIBILITY_STATES = frozenset({"good", "limited", "poor"})
SUPPORTED_FRAME_COUNTS = frozenset({7, 13, 17})


DEFAULT_CRITERION_ROI_CLASSES: dict[str, tuple[str, ...]] = {
    "two_structures": ("cystic_duct", "cystic_artery", "gallbladder"),
    "cystic_plate": ("cystic_plate", "gallbladder"),
    "hepatocystic_triangle": ("calot_triangle", "gallbladder"),
}


_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "tool": (40, 60, 240),
    "cystic_duct": (30, 210, 255),
    "cystic_artery": (220, 80, 220),
    "cystic_plate": (60, 210, 70),
    "calot_triangle": (255, 150, 40),
    "gallbladder": (220, 210, 60),
}


@dataclass(frozen=True)
class EvidencePack:
    """Chronological rendered evidence submitted in one MLLM request."""

    criterion: str
    center_s: float
    timestamps_s: tuple[float, ...]
    image_data_urls: tuple[str, ...]
    roi_classes: tuple[str, ...]
    view_layout: str = "triptych"
    time_scale_labels: tuple[str, ...] = ()
    temporal_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.timestamps_s) not in SUPPORTED_FRAME_COUNTS:
            raise ValueError("Evidence Pack v1 requires exactly 7 or 13 timestamps")
        if len(self.image_data_urls) != len(self.timestamps_s):
            raise ValueError("Each evidence timestamp must have exactly one triptych")
        if any(right <= left for left, right in zip(self.timestamps_s, self.timestamps_s[1:])):
            raise ValueError("Evidence pack timestamps must be strictly chronological")
        if any(not value.startswith("data:image/") for value in self.image_data_urls):
            raise ValueError("Evidence pack images must be data image URLs")
        if self.view_layout not in {"triptych", "raw_overlay", "raw_highres"}:
            raise ValueError("Unsupported evidence pack view layout")
        if self.time_scale_labels and len(self.time_scale_labels) != len(self.timestamps_s):
            raise ValueError("Each evidence timestamp must have exactly one time-scale label")
        if any(value not in {"long_context", "short_dense"} for value in self.time_scale_labels):
            raise ValueError("Unsupported evidence time-scale label")
        if self.temporal_roles and len(self.temporal_roles) != len(self.timestamps_s):
            raise ValueError("Each evidence timestamp must have exactly one temporal role")
        if any(not isinstance(value, str) or not value.strip() for value in self.temporal_roles):
            raise ValueError("Temporal evidence roles must be nonempty strings")

    @property
    def frame_count(self) -> int:
        return len(self.timestamps_s)

    def audit_dict(self, include_images: bool = False) -> dict[str, Any]:
        value = {
            "criterion": self.criterion,
            "center_s": self.center_s,
            "timestamps_s": list(self.timestamps_s),
            "roi_classes": list(self.roi_classes),
            "frame_count": self.frame_count,
            "view_layout": self.view_layout,
            "time_scale_labels": list(
                self.time_scale_labels or ("short_dense",) * self.frame_count
            ),
            "temporal_roles": list(self.temporal_roles),
        }
        if include_images:
            value["image_data_urls"] = list(self.image_data_urls)
        return value


@dataclass(frozen=True)
class EvidencePackObservation:
    """Strict four-state MLLM observation tied to visible pack frames."""

    criterion: str
    state: str
    confidence: float
    visibility: str
    supporting_frame_indices: tuple[int, ...]
    observed_facts: tuple[str, ...]
    rationale: str

    @property
    def fusion_state(self) -> str:
        """Conservative compatibility mapping for existing yes/no/unknown fusion."""
        return {"full": "yes", "absent": "no"}.get(self.state, "unknown")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supporting_frame_indices"] = list(self.supporting_frame_indices)
        value["observed_facts"] = list(self.observed_facts)
        return value


def symmetric_timestamps(
    center_s: float, frame_count: int = 7, spacing_s: float = 2.0,
    start_s: float = 0.0, end_s: float | None = None,
) -> list[float]:
    """Return an exact 7/13-point grid, shifting it inside available bounds."""
    if frame_count not in SUPPORTED_FRAME_COUNTS:
        raise ValueError("Evidence Pack v1 supports exactly 7 or 13 time points")
    if spacing_s <= 0:
        raise ValueError("spacing_s must be positive")
    span = spacing_s * (frame_count - 1)
    if end_s is not None and end_s < start_s + span:
        raise ValueError("Available time range is too short for the requested evidence pack")
    first = float(center_s) - span / 2.0
    first = max(float(start_s), first)
    if end_s is not None:
        first = min(first, float(end_s) - span)
    return [round(first + index * spacing_s, 3) for index in range(frame_count)]


def multiscale_timestamps(
    center_s: float, short_frame_count: int = 13, short_spacing_s: float = 2.0,
    long_lookback_s: float = 120.0, long_frame_count: int = 4,
    start_s: float = 0.0, end_s: float | None = None,
) -> tuple[list[float], list[str]]:
    """Return sparse historical context followed by dense current evidence."""
    if short_frame_count != 13 or long_frame_count != 4:
        raise ValueError("Evidence Pack multiscale v1 requires 4 long + 13 short frames")
    if long_lookback_s <= 0:
        raise ValueError("long_lookback_s must be positive")
    short = symmetric_timestamps(
        center_s, short_frame_count, short_spacing_s, start_s=start_s, end_s=end_s,
    )
    context_start = max(float(start_s), float(center_s) - float(long_lookback_s))
    context_end = short[0] - max(1.0, short_spacing_s)
    if context_end <= context_start:
        raise ValueError("Insufficient pre-candidate history for multiscale evidence")
    long_values = np.linspace(context_start, context_end, long_frame_count).tolist()
    timestamps = [round(float(value), 3) for value in long_values] + short
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Multiscale timestamps must be strictly chronological")
    return timestamps, ["long_context"] * long_frame_count + ["short_dense"] * short_frame_count


def _scaled_box(
    item: Any, image_width: int, image_height: int,
) -> tuple[int, int, int, int]:
    normalized = tuple(float(value) for value in item.bbox_normalized_xyxy)
    if len(normalized) == 4 and all(-0.01 <= value <= 1.01 for value in normalized):
        x1, y1, x2, y2 = (
            normalized[0] * image_width, normalized[1] * image_height,
            normalized[2] * image_width, normalized[3] * image_height,
        )
    else:
        x1, y1, x2, y2 = item.bbox_xyxy
    return (
        int(np.clip(round(x1), 0, max(0, image_width - 1))),
        int(np.clip(round(y1), 0, max(0, image_height - 1))),
        int(np.clip(round(x2), 1, image_width)),
        int(np.clip(round(y2), 1, image_height)),
    )


def _title(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 25), (15, 15, 15), -1)
    cv2.putText(
        image, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (245, 245, 245), 1, cv2.LINE_AA,
    )


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def render_evidence_triptych(
    frame: np.ndarray,
    observations: FrameObjectObservations,
    criterion: str,
    frame_index: int,
    roi_classes: Sequence[str] | None = None,
    panel_width: int = 320,
    panel_height: int = 180,
    roi_expansion: float = 0.18,
) -> np.ndarray:
    """Render raw, bbox and ROI panels without altering the source frame."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR HxWx3 image")
    if panel_width < 80 or panel_height < 60:
        raise ValueError("Evidence panels are too small")
    if roi_expansion < 0:
        raise ValueError("roi_expansion cannot be negative")
    classes = tuple(roi_classes or DEFAULT_CRITERION_ROI_CLASSES.get(criterion, ()))
    relevant = set(classes)
    height, width = frame.shape[:2]

    raw = _letterbox(frame, panel_width, panel_height)
    _title(raw, f"frame {frame_index} | raw | t={observations.time_s:.1f}s")

    boxed_source = frame.copy()
    mask_layer = np.zeros_like(frame)
    mask_visible = np.zeros((height, width), dtype=np.uint8)
    relevant_boxes: list[tuple[int, int, int, int]] = []
    for item in observations.observations:
        x1, y1, x2, y2 = _scaled_box(item, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        is_relevant = item.semantic_type in relevant
        if is_relevant:
            relevant_boxes.append((x1, y1, x2, y2))
        else:
            # The unmodified raw panel already preserves tool and context
            # pixels.  Drawing unrelated predictions over the evidence panel
            # obscures anatomy and encourages label-copying by the MLLM.
            continue
        color = _CLASS_COLORS.get(item.semantic_type, (210, 210, 210))
        for contour in item.mask_contours_normalized_xy:
            points = np.asarray(
                [[round(x * width), round(y * height)] for x, y in contour],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            if len(points) >= 3:
                cv2.fillPoly(mask_layer, [points], color)
                cv2.fillPoly(mask_visible, [points], 1)
        thickness = 3 if is_relevant else 1
        cv2.rectangle(boxed_source, (x1, y1), (x2, y2), color, thickness)
        label = f"{item.semantic_type} {item.confidence:.2f}"
        cv2.putText(
            boxed_source, label, (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA,
        )
    visible = mask_visible.astype(bool)
    if visible.any():
        blended = cv2.addWeighted(frame, 0.58, mask_layer, 0.42, 0.0)
        boxed_source[visible] = blended[visible]
        # Redraw boxes and labels after the mask blend so geometry remains legible.
        for item in observations.observations:
            if item.semantic_type not in relevant:
                continue
            x1, y1, x2, y2 = _scaled_box(item, width, height)
            color = _CLASS_COLORS.get(item.semantic_type, (210, 210, 210))
            cv2.rectangle(boxed_source, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                boxed_source, f"{item.semantic_type} {item.confidence:.2f}",
                (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                color, 1, cv2.LINE_AA,
            )
    boxed = _letterbox(boxed_source, panel_width, panel_height)
    _title(boxed, "criterion masks + boxes (may be wrong)")

    if relevant_boxes:
        x1 = min(box[0] for box in relevant_boxes)
        y1 = min(box[1] for box in relevant_boxes)
        x2 = max(box[2] for box in relevant_boxes)
        y2 = max(box[3] for box in relevant_boxes)
        pad_x = max(8, round((x2 - x1) * roi_expansion))
        pad_y = max(8, round((y2 - y1) * roi_expansion))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        roi_source = boxed_source[y1:y2, x1:x2]
        roi_label = f"criterion ROI | {criterion}"
    else:
        roi_source = frame
        roi_label = "ROI fallback: full raw frame"
    roi = _letterbox(roi_source, panel_width, panel_height)
    _title(roi, roi_label)
    return np.hstack((raw, boxed, roi))


class EvidencePackRenderer:
    """Decode real frames and join them with frozen object observations."""

    def __init__(
        self, video_path: str | Path, object_provider: Any,
        criterion_roi_classes: Mapping[str, Sequence[str]] | None = None,
        panel_width: int = 320, panel_height: int = 180, jpeg_quality: int = 88,
        view_layout: str = "triptych",
    ) -> None:
        self.video_path = str(video_path)
        self.object_provider = object_provider
        self.criterion_roi_classes = {
            key: tuple(value) for key, value in
            (criterion_roi_classes or DEFAULT_CRITERION_ROI_CLASSES).items()
        }
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.jpeg_quality = int(jpeg_quality)
        self.view_layout = view_layout
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.view_layout not in {"triptych", "raw_overlay", "raw_highres"}:
            raise ValueError("view_layout must be 'triptych', 'raw_overlay', or 'raw_highres'")

    def render(
        self, criterion: str, center_s: float, timestamps_s: Sequence[float],
        time_scale_labels: Sequence[str] | None = None,
        temporal_roles: Sequence[str] | None = None,
    ) -> EvidencePack:
        timestamps = tuple(float(value) for value in timestamps_s)
        if len(timestamps) not in SUPPORTED_FRAME_COUNTS:
            raise ValueError("Evidence Pack v1 requires exactly 7 or 13 timestamps")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("Evidence pack timestamps must be strictly chronological")
        observations = self.object_provider.observe(self.video_path, list(timestamps))
        by_time = {round(item.time_s, 6): item for item in observations}
        missing = [value for value in timestamps if round(value, 6) not in by_time]
        if missing:
            raise ValueError(f"Object provider omitted evidence timestamps: {missing}")

        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open evidence video: {self.video_path}")
        encoded_images = []
        try:
            for index, time_s in enumerate(timestamps):
                capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Unable to decode evidence frame at {time_s:.3f}s")
                if self.view_layout == "raw_highres":
                    rendered = _letterbox(
                        frame, 2 * self.panel_width, 2 * self.panel_height,
                    )
                    _title(rendered, f"frame {index} | raw high-resolution | t={time_s:.1f}s")
                else:
                    triptych = render_evidence_triptych(
                        frame, by_time[round(time_s, 6)], criterion, index,
                        self.criterion_roi_classes.get(criterion, ()),
                        self.panel_width, self.panel_height,
                    )
                    rendered = (
                        triptych[:, : 2 * self.panel_width]
                        if self.view_layout == "raw_overlay" else triptych
                    )
                ok, encoded = cv2.imencode(
                    ".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("Unable to JPEG-encode evidence triptych")
                encoded_images.append(
                    "data:image/jpeg;base64,"
                    + base64.b64encode(encoded.tobytes()).decode("ascii")
                )
        finally:
            capture.release()
        return EvidencePack(
            criterion=criterion, center_s=float(center_s), timestamps_s=timestamps,
            image_data_urls=tuple(encoded_images),
            roi_classes=self.criterion_roi_classes.get(criterion, ()),
            view_layout=self.view_layout,
            time_scale_labels=tuple(time_scale_labels or ("short_dense",) * len(timestamps)),
            temporal_roles=tuple(temporal_roles or ()),
        )


def build_evidence_pack_payload(
    pack: EvidencePack, requirement: str, model: str,
    max_tokens: int = 500,
) -> dict[str, Any]:
    """Build a deterministic OpenAI-compatible request for one evidence pack."""
    if not model:
        raise ValueError("model is required")
    timestamps = ", ".join(
        f"{index}:{time_s:.1f}s"
        + (f":{pack.temporal_roles[index]}" if pack.temporal_roles else "")
        for index, time_s in enumerate(pack.timestamps_s)
    )
    role_instruction = (
        " Temporal roles are tool-selected hypotheses, not labels; compare pre/post context "
        "with onset, stable-interior, and offset evidence and verify all roles from pixels."
        if pack.temporal_roles else ""
    )
    instruction = (
        "You are a visual evidence verifier, not the final SOP decision maker. "
        f"Assess only criterion '{pack.criterion}': {requirement}. "
        f"The {pack.frame_count} attached triptychs are chronological ({timestamps}). "
        f"{role_instruction} "
        "Each triptych contains the same moment as raw pixels, predicted boxes, and a "
        "criterion ROI. Predicted boxes are fallible hints, not facts; verify them against "
        "the raw pixels. Use full only when the complete requirement is visibly satisfied, "
        "partial when only part is satisfied, absent when adequate visible evidence shows it "
        "is not satisfied, and unobservable when occlusion or image quality prevents judgment. "
        "A full or absent judgment requires support in at least two adjacent frame indices. "
        "Return ONLY one JSON object with exactly: criterion (string), state "
        "('full','partial','absent','unobservable'), confidence (0..1), visibility "
        "('good','limited','poor'), supporting_frame_indices (array of zero-based integers), "
        "observed_facts (array of short strings), rationale (short string)."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in pack.image_data_urls
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }


def parse_evidence_pack_observation(
    text: str, criterion: str, frame_count: int,
) -> EvidencePackObservation:
    """Parse and semantically validate a four-state evidence response."""
    if frame_count not in SUPPORTED_FRAME_COUNTS:
        raise ValueError("Evidence Pack v1 supports exactly 7 or 13 frames")
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Evidence Pack JSON: {text[:400]}") from exc
    required = {
        "criterion", "state", "confidence", "visibility",
        "supporting_frame_indices", "observed_facts", "rationale",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"Evidence Pack response fields must be exactly {sorted(required)}")
    if value["criterion"] != criterion:
        raise ValueError("Evidence Pack response criterion does not match the request")
    state = value["state"]
    visibility = value["visibility"]
    if not isinstance(state, str) or state not in EVIDENCE_STATES:
        raise ValueError(f"Unsupported Evidence Pack state: {state}")
    if not isinstance(visibility, str) or visibility not in VISIBILITY_STATES:
        raise ValueError(f"Unsupported Evidence Pack visibility: {visibility}")
    if not isinstance(value["confidence"], (int, float)) or isinstance(value["confidence"], bool):
        raise ValueError("Evidence Pack confidence must be numeric")
    confidence = float(value["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Evidence Pack confidence must be in [0, 1]")
    indices = value["supporting_frame_indices"]
    if (
        not isinstance(indices, list)
        or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
    ):
        raise ValueError("supporting_frame_indices must be an integer array")
    if indices != sorted(set(indices)):
        raise ValueError("supporting_frame_indices must be unique and sorted")
    if any(index < 0 or index >= frame_count for index in indices):
        raise ValueError("supporting_frame_indices contains an out-of-range frame")
    if state in {"full", "absent"} and not any(
        right == left + 1 for left, right in zip(indices, indices[1:])
    ):
        raise ValueError(f"State {state!r} requires at least two adjacent support frames")
    if state == "partial" and not indices:
        raise ValueError("State 'partial' requires at least one support frame")
    if state == "unobservable" and visibility == "good":
        raise ValueError("State 'unobservable' cannot claim good visibility")
    if state in {"full", "absent"} and visibility == "poor":
        raise ValueError(f"State {state!r} requires adequate visibility")
    facts = value["observed_facts"]
    if not isinstance(facts, list) or not facts or not all(
        isinstance(item, str) and item.strip() for item in facts
    ):
        raise ValueError("observed_facts must be a non-empty string array")
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    return EvidencePackObservation(
        criterion=criterion, state=state, confidence=confidence,
        visibility=visibility, supporting_frame_indices=tuple(indices),
        observed_facts=tuple(item.strip() for item in facts),
        rationale=rationale.strip(),
    )
