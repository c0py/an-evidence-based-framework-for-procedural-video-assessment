from __future__ import annotations

import csv
import base64
import json
import math
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from .annotations import load_cvs_intervals, load_phase_starts
from .schema import ScorePoint, ToolCapability, VisualEvidencePoint


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._contracts: dict[str, ToolCapability] = {}
        self.calls: list[dict[str, Any]] = []

    def register(
        self, name: str, tool: Callable[..., Any],
        contract: ToolCapability | None = None,
    ) -> None:
        self._tools[name] = tool
        self._contracts[name] = contract or ToolCapability(tool_id=name)

    def contract(self, name: str) -> ToolCapability:
        if name not in self._contracts:
            raise KeyError(f"Unknown tool contract: {name}")
        return self._contracts[name]

    def find_by_capability(self, capability: str, task_id: str = "*") -> list[str]:
        return sorted(
            name for name, contract in self._contracts.items()
            if capability in contract.capabilities
            and ("*" in contract.supported_tasks or task_id in contract.supported_tasks)
        )

    def manifest(self) -> dict[str, dict[str, Any]]:
        return {name: asdict(contract) for name, contract in self._contracts.items()}

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        result = self._tools[name](**kwargs)
        summary = result if isinstance(result, dict) else {"type": type(result).__name__, "count": len(result) if hasattr(result, "__len__") else None}
        self.calls.append({
            "tool": name,
            "contract": _jsonable(self._contracts.get(name)),
            "arguments": _audit_arguments(kwargs),
            "result": _jsonable(summary),
        })
        return result


class DatasetReplayScorer:
    """Task-neutral development oracle backed by a DatasetAdapter."""

    def __init__(self, adapter, video_id: str | int, **sources: Any) -> None:
        self.intervals = adapter.load_intervals(video_id, **sources)

    def score(self, criterion: str, timestamps: list[float]) -> list[ScorePoint]:
        points = []
        intervals = self.intervals.get(criterion, [])
        for time_s in timestamps:
            label = max(
                (value for start, end, value in intervals if start <= time_s <= end),
                default=0,
            )
            raw = (0.08, 0.58, 0.92)[min(max(int(label), 0), 2)]
            points.append(ScorePoint(
                time_s=float(time_s), score=raw, raw_score=raw, quality=1.0,
            ))
        return points


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _audit_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep tool traces inspectable without duplicating long sampled timelines."""
    audit = _jsonable(arguments)
    timestamps = audit.get("timestamps")
    if isinstance(timestamps, list) and timestamps and all(isinstance(item, (int, float)) for item in timestamps):
        audit["timestamps"] = {
            "count": len(timestamps), "first_s": timestamps[0], "last_s": timestamps[-1],
        }
    points = audit.get("points")
    if isinstance(points, list):
        audit["points"] = {
            "count": len(points),
            "first_time_s": points[0].get("time_s") if points and isinstance(points[0], dict) else None,
            "last_time_s": points[-1].get("time_s") if points and isinstance(points[-1], dict) else None,
        }
    return audit


def locate_evaluation_window(phase_annotation_path: str, candidate_phase: str, anchor_phase: str) -> dict[str, float]:
    phases = load_phase_starts(phase_annotation_path)
    if candidate_phase not in phases or anchor_phase not in phases:
        raise ValueError(f"Phase annotations must contain {candidate_phase} and {anchor_phase}")
    start, end = phases[candidate_phase], phases[anchor_phase]
    if end <= start:
        raise ValueError("Anchor phase begins before candidate phase")
    return {"start_s": start, "end_s": end, "anchor_s": end}


def sample_timestamps(start_s: float, end_s: float, sampling_fps: float) -> list[float]:
    step = 1.0 / sampling_fps
    count = int(math.floor((end_s - start_s) / step)) + 1
    return [round(start_s + i * step, 3) for i in range(count) if start_s + i * step < end_s]


class AnnotationReplayScorer:
    """Development oracle: replay CVS intervals as noisy scores. Never use for test metrics."""

    def __init__(self, annotation_path: str, video_id: int) -> None:
        self.intervals = load_cvs_intervals(annotation_path, video_id)

    def score(self, criterion: str, timestamps: list[float]) -> list[ScorePoint]:
        points = []
        for time_s in timestamps:
            label = 0
            for start, end, value in self.intervals[criterion]:
                if start <= time_s <= end:
                    label = max(label, value)
            # A deterministic, low-amplitude nuisance term makes the temporal test nontrivial.
            nuisance = 0.08 * math.sin(time_s * 0.73 + len(criterion))
            base = (0.10, 0.58, 0.90)[min(label, 2)]
            raw = float(np.clip(base + nuisance, 0.01, 0.99))
            quality = float(np.clip(0.85 + 0.12 * math.cos(time_s * 0.11), 0.45, 0.98))
            points.append(ScorePoint(time_s=time_s, score=raw * quality, raw_score=raw, quality=quality))
        return points


class AppearanceProxyScorer:
    """Non-clinical, checkpoint-free fallback that exposes frame quality only."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path

    def score(self, criterion: str, timestamps: list[float]) -> list[ScorePoint]:
        cap = cv2.VideoCapture(self.video_path)
        points: list[ScorePoint] = []
        for time_s in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 180.0, 1.0)
            brightness = gray.mean() / 255.0
            quality = float(np.clip(0.65 * sharpness + 0.35 * (1 - abs(brightness - 0.5)), 0, 1))
            # No anatomical claim: keep scores below pass threshold.
            score = float(np.clip(0.15 + 0.20 * quality, 0, 0.45))
            points.append(ScorePoint(time_s=time_s, score=score, raw_score=score, quality=quality))
        cap.release()
        return points


class QwenVLEvidenceScorer:
    """Ground task-defined evidence in clips through an OpenAI-compatible MLLM.

    Each query contains several frames surrounding one timestamp.  The model's JSON
    observation is retained verbatim as a structured audit artifact, while its
    bounded support confidence is converted to the existing temporal score interface.
    This class never reads annotations and never falls back to a proxy score.
    """

    _VISIBILITY_TO_QUALITY = {"good": 1.0, "limited": 0.55, "poor": 0.2}

    def __init__(
        self,
        video_path: str,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 120.0,
        clip_radius_s: float = 3.0,
        clip_frames: int = 3,
        max_queries: int | None = None,
        frame_width: int = 640,
        frame_height: int = 360,
    ) -> None:
        if not model:
            raise ValueError("qwen_vl.model is required")
        if clip_frames < 1:
            raise ValueError("qwen_vl.clip_frames must be at least 1")
        self.video_path = video_path
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.clip_radius_s = clip_radius_s
        self.clip_frames = clip_frames
        self.max_queries = max_queries
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.observations: dict[str, list[VisualEvidencePoint]] = {}
        self.request_log: list[dict[str, Any]] = []

    def score(self, criterion: str, timestamps: list[float], visual_requirement: str = "") -> list[ScorePoint]:
        if self.max_queries is not None and len(timestamps) > self.max_queries:
            # A bounded run should still cover the whole candidate phase rather
            # than biasing evidence toward its first few seconds.
            selected = [timestamps[index] for index in np.linspace(0, len(timestamps) - 1, self.max_queries, dtype=int)]
        else:
            selected = timestamps
        evidence: list[VisualEvidencePoint] = []
        points: list[ScorePoint] = []
        for time_s in selected:
            frame_times, images = self._clip_images(time_s)
            payload = self._request_payload(criterion, visual_requirement, time_s, frame_times, images)
            answer = self._post(payload)
            observation = self._parse_observation(answer, criterion, time_s, frame_times)
            evidence.append(observation)
            support = observation.confidence if observation.supports_criterion == "yes" else 0.0
            state = {
                "yes": "positive", "no": "negative", "unknown": "unknown",
            }[observation.supports_criterion]
            points.append(ScorePoint(
                time_s=time_s,
                score=support * self._VISIBILITY_TO_QUALITY[observation.visibility],
                raw_score=support,
                quality=self._VISIBILITY_TO_QUALITY[observation.visibility],
                evidence_state=state,
                state_confidence=observation.confidence,
            ))
        self.observations[criterion] = evidence
        return points

    def _clip_images(self, center_s: float) -> tuple[list[float], list[str]]:
        offsets = np.linspace(-self.clip_radius_s, self.clip_radius_s, self.clip_frames).tolist()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video for Qwen-VL evidence: {self.video_path}")
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1.0)
        times, images = [], []
        for offset in offsets:
            frame_time = round(float(np.clip(center_s + offset, 0.0, duration)), 3)
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Unable to decode Qwen-VL evidence frame at {frame_time:.2f}s")
            frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                cap.release()
                raise RuntimeError("Unable to JPEG-encode Qwen-VL evidence frame")
            times.append(frame_time)
            images.append("data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"))
        cap.release()
        return times, images

    def _request_payload(self, criterion: str, requirement: str, time_s: float, frame_times: list[float], images: list[str]) -> dict[str, Any]:
        instruction = (
            "You are a procedural-video evidence tool, not a final decision maker. "
            f"Assess only this task criterion: {criterion}. Requirement: {requirement}. "
            f"The attached images are a chronological short clip centered at {time_s:.3f} seconds; "
            f"their timestamps are {frame_times}. Do not infer facts that are not visibly supported. "
            "Label semantics are strict: criterion_met means the visible facts satisfy the criterion; "
            "criterion_not_met means the visible facts clearly contradict or fail the criterion; "
            "insufficient_visibility means the "
            "view is insufficient, obscured, or ambiguous. If your observed facts and rationale say "
            "the requirement is satisfied, evidence_state MUST be criterion_met. Check that the label and "
            "rationale agree before responding. "
            "Return ONLY a JSON object with exactly these fields: criterion (string), "
            "evidence_state ('criterion_met', 'criterion_not_met', or 'insufficient_visibility'), "
            "confidence (number 0 to 1), "
            "visibility ('good', 'limited', or 'poor'), observed_facts (array of short strings), "
            "and rationale (short string). Use evidence_state='insufficient_visibility' when anatomy is obscured or insufficiently visible."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        content.extend({"type": "image_url", "image_url": {"url": image}} for image in images)
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 350,
            "response_format": {"type": "json_object"},
        }

    def _post(self, payload: dict[str, Any]) -> str:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen-VL request failed with HTTP {exc.code}: {detail[:500]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach Qwen-VL server at {self.base_url}: {exc.reason}") from exc
        self.request_log.append({
            "latency_s": time.monotonic() - started,
            "usage": body.get("usage", {}),
        })
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Qwen-VL returned no chat completion: {body}") from exc

    def _parse_observation(self, text: str, criterion: str, time_s: float, frame_times: list[float]) -> VisualEvidencePoint:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(candidate)
            if "evidence_state" in value:
                support = {
                    "criterion_met": "yes",
                    "criterion_not_met": "no",
                    "insufficient_visibility": "unknown",
                }.get(value["evidence_state"])
            else:
                # Backward-compatible parser for earlier recorded runs.
                support = value["supports_criterion"]
            visibility = value["visibility"]
            confidence = float(value["confidence"])
            facts = value["observed_facts"]
            rationale = value["rationale"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Qwen-VL returned invalid structured evidence: {text[:800]}") from exc
        if support not in {"yes", "no", "unknown"} or visibility not in self._VISIBILITY_TO_QUALITY:
            raise RuntimeError(f"Qwen-VL returned unsupported evidence labels: {value}")
        if not 0.0 <= confidence <= 1.0 or not isinstance(facts, list) or not all(isinstance(x, str) for x in facts):
            raise RuntimeError(f"Qwen-VL returned invalid evidence values: {value}")
        return VisualEvidencePoint(criterion=criterion, time_s=time_s, supports_criterion=support,
                                   confidence=confidence, visibility=visibility, observed_facts=facts,
                                   rationale=str(rationale), frame_times_s=frame_times)


class SpatialFusionCheckpointScorer:
    """Fuse a dense visual scorer with replaceable object observations.

    The checkpoint declares the exact named feature schema and selected feature
    indices. Domain mappings are supplied by backend configuration, leaving the
    pipeline and model head task-neutral.
    """

    def __init__(
        self, video_path: str, small_scorer: Any, object_provider: Any,
        checkpoint_path: str, criteria: list[str], object_classes: list[str],
        candidate_rules: dict[str, dict[str, Any]], device: str | None = None,
    ) -> None:
        from .models import FeatureFusionCalibrator

        self.video_path = video_path
        self.small_scorer = small_scorer
        self.object_provider = object_provider
        self.criteria = list(criteria)
        self.object_classes = list(object_classes)
        self.candidate_rules = candidate_rules
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_metadata = checkpoint["metadata"]
        self.feature_names = list(raw_metadata["feature_names"])
        self.feature_indices = [int(value) for value in raw_metadata["feature_indices"]]
        self.feature_mean = torch.as_tensor(raw_metadata["feature_mean"], dtype=torch.float32)
        self.feature_std = torch.as_tensor(raw_metadata["feature_std"], dtype=torch.float32)
        first_weight = checkpoint["model_state"]["network.0.weight"]
        final_weight = checkpoint["model_state"]["network.3.weight"]
        self.model = FeatureFusionCalibrator(
            input_dim=int(first_weight.shape[1]), output_dim=int(final_weight.shape[0]),
            hidden_dim=int(first_weight.shape[0]), dropout=0.0,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.metadata = {
            **raw_metadata,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "small_scorer": getattr(small_scorer, "metadata", {}),
            "object_observation_source": getattr(object_provider, "checkpoint", None),
        }
        self.ordinal = False
        self.object_observations: list[Any] = []
        self._cached_timestamps: tuple[float, ...] | None = None
        self._cached_points: dict[str, list[ScorePoint]] = {}

    @torch.inference_mode()
    def _prepare(self, timestamps: list[float]) -> None:
        from .object_observations import spatial_fusion_features

        timestamp_key = tuple(float(value) for value in timestamps)
        if self._cached_timestamps == timestamp_key:
            return
        small_points = {
            criterion: self.small_scorer.score(criterion, timestamps)
            for criterion in self.criteria
        }
        decoded_times = [point.time_s for point in small_points[self.criteria[0]]]
        for criterion in self.criteria[1:]:
            criterion_times = [point.time_s for point in small_points[criterion]]
            if criterion_times != decoded_times:
                raise RuntimeError("Small scorer returned inconsistent timestamps across criteria")
        frames = self.object_provider.observe(self.video_path, decoded_times)
        frame_by_time = {round(frame.time_s, 6): frame for frame in frames}
        rows = []
        for index, time_s in enumerate(decoded_times):
            frame = frame_by_time.get(round(float(time_s), 6))
            if frame is None:
                raise RuntimeError(f"Object provider omitted requested timestamp {time_s:.6f}s")
            named = spatial_fusion_features(
                frame=frame,
                small_scores={key: small_points[key][index].score for key in self.criteria},
                criteria=self.criteria,
                object_classes=self.object_classes,
                candidate_rules=self.candidate_rules,
            )
            try:
                rows.append([named[name] for name in self.feature_names])
            except KeyError as exc:
                raise RuntimeError(f"Fusion checkpoint requests unavailable feature {exc}") from exc
        features = torch.tensor(rows, dtype=torch.float32)[:, self.feature_indices]
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        fused = torch.sigmoid(self.model(normalized.to(self.device))).cpu()
        self._cached_points = {}
        for criterion_index, criterion in enumerate(self.criteria):
            self._cached_points[criterion] = [
                ScorePoint(
                    time_s=float(time_s), score=float(fused[index, criterion_index]),
                    raw_score=float(small_points[criterion][index].score), quality=1.0,
                )
                for index, time_s in enumerate(decoded_times)
            ]
        self.object_observations = frames
        self._cached_timestamps = timestamp_key

    def score(self, criterion: str, timestamps: list[float]) -> list[ScorePoint]:
        if criterion not in self.criteria:
            raise KeyError(f"Criterion {criterion!r} is not supported by fusion checkpoint")
        self._prepare(timestamps)
        return self._cached_points[criterion]


def fuse_small_mllm_observation(
    point: ScorePoint, observation: VisualEvidencePoint,
    maximum_fusion_weight: float, explicit_state_threshold: float,
    unknown_policy: str = "explicit_unknown", update_policy: str = "symmetric",
) -> tuple[ScorePoint, dict[str, Any]]:
    """Apply one MLLM observation without coupling fusion to query transport."""
    if unknown_policy not in {"explicit_unknown", "passthrough"}:
        raise ValueError(f"Unsupported unknown_policy: {unknown_policy}")
    if update_policy not in {"symmetric", "negative_only"}:
        raise ValueError(f"Unsupported update_policy: {update_policy}")
    visibility_quality = QwenVLEvidenceScorer._VISIBILITY_TO_QUALITY[observation.visibility]
    reliability = observation.confidence * visibility_quality
    ignored = (
        observation.supports_criterion == "unknown" and unknown_policy == "passthrough"
    ) or (
        observation.supports_criterion == "yes" and update_policy == "negative_only"
    )
    if ignored:
        fused_score = point.score
        fusion_weight = 0.0
        evidence_state = "unclassified"
        state_confidence = 0.0
        output_quality = point.quality
    elif observation.supports_criterion == "unknown":
        fused_score = point.score
        fusion_weight = 0.0
        evidence_state = "unknown"
        state_confidence = reliability
        output_quality = visibility_quality
    else:
        mllm_probability = (
            observation.confidence
            if observation.supports_criterion == "yes"
            else 1.0 - observation.confidence
        )
        fusion_weight = maximum_fusion_weight * reliability
        fused_score = (1.0 - fusion_weight) * point.score + fusion_weight * mllm_probability
        evidence_state = "unclassified"
        state_confidence = reliability
        output_quality = visibility_quality
        if reliability >= explicit_state_threshold:
            evidence_state = (
                "positive" if observation.supports_criterion == "yes" else "negative"
            )
    fused = ScorePoint(
        point.time_s, float(fused_score), point.raw_score, output_quality,
        evidence_state=evidence_state, state_confidence=float(state_confidence),
    )
    return fused, {
        "fusion_weight": float(fusion_weight),
        "fused_score": float(fused_score),
        "fused_evidence_state": evidence_state,
        "fused_state_confidence": float(state_confidence),
        "mllm_reliability": float(reliability),
        "observation_ignored": ignored,
        "unknown_policy": unknown_policy,
        "update_policy": update_policy,
    }


class SmallMllmFusionScorer:
    """Run dense small-tool scoring and sparse MLLM verification in one task.

    Triggering and fusion are task-neutral: criterion semantics arrive through
    the generated skill's visual requirement.  Unknown/poorly visible MLLM
    observations never erase small-model evidence, while confident visible
    observations softly update it instead of acting as a brittle hard gate.
    """

    def __init__(
        self, small_scorer: Any, mllm_scorer: QwenVLEvidenceScorer,
        top_k_candidates: int = 2, uncertainty_queries: int = 1,
        transition_queries: int = 1, max_queries_per_criterion: int = 4,
        candidate_threshold: float = 0.5,
        uncertainty_low: float = 0.35, uncertainty_high: float = 0.65,
        minimum_separation_s: float = 20.0, maximum_fusion_weight: float = 0.55,
        explicit_state_threshold: float = 0.60,
        unknown_policy: str = "explicit_unknown",
        update_policy: str = "symmetric",
    ) -> None:
        if not 0 <= uncertainty_low <= uncertainty_high <= 1:
            raise ValueError("Fusion uncertainty bounds must satisfy 0 <= low <= high <= 1")
        if not 0 <= maximum_fusion_weight <= 1:
            raise ValueError("maximum_fusion_weight must be in [0, 1]")
        if unknown_policy not in {"explicit_unknown", "passthrough"}:
            raise ValueError("unknown_policy must be explicit_unknown or passthrough")
        if update_policy not in {"symmetric", "negative_only"}:
            raise ValueError("update_policy must be symmetric or negative_only")
        self.small_scorer = small_scorer
        self.mllm_scorer = mllm_scorer
        self.top_k_candidates = top_k_candidates
        self.uncertainty_queries = uncertainty_queries
        self.transition_queries = transition_queries
        self.max_queries_per_criterion = max_queries_per_criterion
        self.candidate_threshold = candidate_threshold
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high
        self.minimum_separation_s = minimum_separation_s
        self.maximum_fusion_weight = maximum_fusion_weight
        self.explicit_state_threshold = explicit_state_threshold
        self.unknown_policy = unknown_policy
        self.update_policy = update_policy
        self.metadata = getattr(small_scorer, "metadata", {})
        self.ordinal = False
        self.observations = mllm_scorer.observations
        self.request_log = mllm_scorer.request_log
        self.model = mllm_scorer.model
        self.frame_width = mllm_scorer.frame_width
        self.frame_height = mllm_scorer.frame_height
        self.clip_frames = mllm_scorer.clip_frames
        self.max_queries = max_queries_per_criterion
        self.fusion_audit: dict[str, dict[str, Any]] = {}

    def _select_queries(self, points: list[ScorePoint]) -> list[tuple[int, str]]:
        selected: list[tuple[int, str]] = []

        def add_ranked(indices: list[int], reason: str, limit: int) -> None:
            for index in indices:
                if len([item for item in selected if item[1] == reason]) >= limit:
                    break
                if any(
                    abs(points[index].time_s - points[other].time_s) < self.minimum_separation_s
                    for other, _ in selected
                ):
                    continue
                selected.append((index, reason))

        candidates = sorted(
            (index for index, point in enumerate(points) if point.score >= self.candidate_threshold),
            key=lambda index: points[index].score, reverse=True,
        )
        if not candidates and points:
            candidates = [max(range(len(points)), key=lambda index: points[index].score)]
        add_ranked(candidates, "high_candidate", self.top_k_candidates)

        uncertain = sorted(
            (
                index for index, point in enumerate(points)
                if self.uncertainty_low <= point.score <= self.uncertainty_high
            ),
            key=lambda index: abs(points[index].score - 0.5),
        )
        add_ranked(uncertain, "uncertainty", self.uncertainty_queries)

        transitions = sorted(
            range(1, len(points)),
            key=lambda index: abs(points[index].score - points[index - 1].score),
            reverse=True,
        )
        add_ranked(transitions, "transition", self.transition_queries)
        return selected[:self.max_queries_per_criterion]

    def score(
        self, criterion: str, timestamps: list[float], visual_requirement: str = "",
    ) -> list[ScorePoint]:
        small_points = self.small_scorer.score(criterion, timestamps)
        selected = self._select_queries(small_points)
        query_times = [small_points[index].time_s for index, _ in selected]
        if query_times:
            self.mllm_scorer.score(criterion, query_times, visual_requirement)
        observations = {
            observation.time_s: observation
            for observation in self.mllm_scorer.observations.get(criterion, [])
        }
        reason_by_time = {
            small_points[index].time_s: reason for index, reason in selected
        }
        fused_points: list[ScorePoint] = []
        query_audit = []
        for point in small_points:
            observation = observations.get(point.time_s)
            if observation is None:
                fused_points.append(ScorePoint(
                    point.time_s, point.score, point.score, point.quality,
                    evidence_state="unclassified", state_confidence=0.0,
                ))
                continue
            fused_point, fusion = fuse_small_mllm_observation(
                point, observation, self.maximum_fusion_weight,
                self.explicit_state_threshold, self.unknown_policy, self.update_policy,
            )
            fused_points.append(fused_point)
            query_audit.append({
                "criterion": criterion,
                "time_s": point.time_s,
                "trigger_reason": reason_by_time[point.time_s],
                "small_score": point.score,
                "mllm_state": observation.supports_criterion,
                "mllm_confidence": observation.confidence,
                "visibility": observation.visibility,
                "observed_facts": observation.observed_facts,
                "rationale": observation.rationale,
                **fusion,
            })
        self.fusion_audit[criterion] = {
            "dense_small_point_count": len(small_points),
            "mllm_query_count": len(query_audit),
            "queries": query_audit,
        }
        return fused_points


class CheckpointCvsScorer:
    """Adapter for a trained criterion-conditioned frame model checkpoint."""

    def __init__(self, video_path: str, checkpoint_path: str, device: str | None = None) -> None:
        from .models import CriterionConditionedCvsModel

        self.video_path = video_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
        self.model = CriterionConditionedCvsModel()
        self.model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
        self.model.to(self.device).eval()
        self.transform = v2.Compose([v2.ToImage(), v2.Resize((224, 224)), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    @torch.inference_mode()
    def score(self, criterion: str, timestamps: list[float]) -> list[ScorePoint]:
        cap = cv2.VideoCapture(self.video_path)
        points: list[ScorePoint] = []
        for time_s in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            tensor = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
            logit = self.model(tensor)[criterion].item()
            score = float(torch.sigmoid(torch.tensor(logit)))
            points.append(ScorePoint(time_s=time_s, score=score, raw_score=score, quality=1.0))
        cap.release()
        return points


class PeskaVLPCheckpointScorer:
    """Frozen PeskaVLP visual encoder with a trained lightweight CVS head.

    Features are cached by timestamp so all criterion tools share one visual
    pass over the candidate window.  The checkpoint is self-contained and its
    provenance/development status is exposed through ``metadata``.
    """

    def __init__(
        self, video_path: str, checkpoint_path: str, device: str | None = None,
        inference_batch_size: int = 64,
    ) -> None:
        from .models import PeskaVLPCvsModel, PeskaVLPOrdinalCvsModel

        self.video_path = video_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_batch_size = inference_batch_size
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "encoder_state" not in checkpoint or "head_state" not in checkpoint:
            raise ValueError("PeskaVLP CVS checkpoint must contain encoder_state and head_state")
        self.metadata = checkpoint.get("metadata", {})
        self.ordinal = self.metadata.get("head_type") == "ordinal"
        monotonic = self.metadata.get("ordinal_parameterization") == "conditional_product"
        self.model = (
            PeskaVLPOrdinalCvsModel(monotonic=monotonic)
            if self.ordinal else PeskaVLPCvsModel()
        )
        self.model.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
        self.model.head.load_state_dict(checkpoint["head_state"], strict=True)
        self.model.encoder.freeze()
        self.model.to(self.device).eval()
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self._cached_timestamps: tuple[float, ...] | None = None
        self._cached_logits: dict[str, torch.Tensor] = {}
        self._cached_support_logits: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def _infer_window(self, timestamps: list[float]) -> None:
        cache_key = tuple(timestamps)
        if cache_key == self._cached_timestamps:
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video for PeskaVLP evidence: {self.video_path}")
        frames, decoded_times = [], []
        for time_s in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            resized = cv2.resize(frame, (640, 360))
            top, left = (360 - 224) // 2, (640 - 224) // 2
            crop = resized[top:top + 224, left:left + 224]
            frames.append(self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            decoded_times.append(time_s)
        cap.release()
        if not frames:
            raise RuntimeError("PeskaVLP evidence tool could not decode any candidate frames")
        outputs: dict[str, list[torch.Tensor]] = {}
        support_outputs: dict[str, list[torch.Tensor]] = {}
        for start in range(0, len(frames), self.inference_batch_size):
            images = torch.stack(frames[start:start + self.inference_batch_size]).to(self.device)
            batch_output = self.model(images)
            for criterion, logits in batch_output.items():
                if self.ordinal:
                    support_outputs.setdefault(criterion, []).append(logits["support_or_full"].cpu())
                    outputs.setdefault(criterion, []).append(logits["full_only"].cpu())
                else:
                    outputs.setdefault(criterion, []).append(logits.cpu())
        self._cached_timestamps = cache_key
        self._cached_decoded_times = decoded_times
        self._cached_logits = {key: torch.cat(value) for key, value in outputs.items()}
        self._cached_support_logits = {key: torch.cat(value) for key, value in support_outputs.items()}

    def score(self, criterion: str, timestamps: list[float]) -> list[ScorePoint]:
        self._infer_window(timestamps)
        if criterion not in self._cached_logits:
            raise KeyError(f"PeskaVLP CVS head has no criterion: {criterion}")
        scores = torch.sigmoid(self._cached_logits[criterion]).tolist()
        return [
            ScorePoint(time_s=time_s, score=float(score), raw_score=float(score), quality=1.0)
            for time_s, score in zip(self._cached_decoded_times, scores)
        ]

    def structured_ordinal_evidence(self, timestamps: list[float]) -> dict[str, list[dict]]:
        """Return auditable absent/partial/full evidence without changing verifier input.

        StableEvidenceAggregator intentionally continues to consume the
        stricter ``full_only`` score.  This additional artifact exposes both
        ordinal outputs so downstream analyses can inspect how evidence state
        changes over time.
        """
        if not self.ordinal:
            return {}
        self._infer_window(timestamps)
        calibrated = self.metadata.get("ordinal_thresholds_calibrated_on_validation", {})
        support_thresholds = calibrated.get("support_or_full", {})
        full_thresholds = calibrated.get("full_only", {})
        output: dict[str, list[dict]] = {}
        for criterion, full_logits in self._cached_logits.items():
            support_scores = torch.sigmoid(self._cached_support_logits[criterion]).tolist()
            full_scores = torch.sigmoid(full_logits).tolist()
            support_threshold = float(support_thresholds.get(criterion, 0.5))
            full_threshold = float(full_thresholds.get(criterion, 0.5))
            observations = []
            for time_s, raw_support_score, full_score in zip(
                self._cached_decoded_times, support_scores, full_scores,
            ):
                # The training loss uses a soft consistency penalty.  Apply a
                # transparent monotonic projection for the exported ordinal
                # evidence so that full is always a subset of support while
                # retaining the unprojected value for audit.
                support_score = max(raw_support_score, full_score)
                if full_score >= full_threshold:
                    state = "full"
                elif support_score >= support_threshold:
                    state = "partial"
                else:
                    state = "absent"
                observations.append({
                    "time_s": float(time_s),
                    "support_or_full_score": float(support_score),
                    "raw_support_or_full_score": float(raw_support_score),
                    "full_only_score": float(full_score),
                    "predicted_state": state,
                    "support_threshold": support_threshold,
                    "full_threshold": full_threshold,
                    "monotonic_projection_applied": bool(full_score > raw_support_score),
                    "ordinal_consistent": True,
                })
            output[criterion] = observations
        return output


class PeskaVLPTemporalCheckpointScorer:
    """Frozen PeskaVLP encoder plus learned frame and temporal CVS heads.

    All requested candidate-phase frames are decoded once. The learned TCN
    then observes the complete chronological feature sequence and emits an
    ordinal CVS trajectory plus auxiliary onset/offset probabilities.
    """

    def __init__(
        self, video_path: str, checkpoint_path: str, device: str | None = None,
        inference_batch_size: int = 96, temporal_enabled: bool = True,
    ) -> None:
        from .models import (
            OrdinalCriterionHead,
            PeskaVLPVisualEncoder,
            TemporalOrdinalBoundaryHead,
        )

        self.video_path = video_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_batch_size = inference_batch_size
        self.temporal_enabled = temporal_enabled
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {"encoder_state", "frame_head_state", "temporal_head_state"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError(
                "Temporal PeskaVLP checkpoint must contain encoder_state, "
                "frame_head_state, and temporal_head_state"
            )
        self.metadata = dict(checkpoint.get("metadata", {}))
        self.metadata["runtime_inference_mode"] = (
            "temporal_head" if temporal_enabled else "matched_frame_head_ablation"
        )
        parameters = self.metadata.get("temporal_parameters", {})
        self.criteria = tuple(self.metadata.get("criteria", (
            "two_structures", "cystic_plate", "hepatocystic_triangle",
        )))
        self.encoder = PeskaVLPVisualEncoder()
        self.frame_head = OrdinalCriterionHead(
            feature_dim=self.encoder.output_dim, criteria=self.criteria,
        )
        self.temporal_head = TemporalOrdinalBoundaryHead(
            feature_dim=self.encoder.output_dim,
            hidden_dim=int(parameters.get("hidden_dim", 128)),
            dilations=tuple(parameters.get("dilations", (1, 2, 4, 8, 16, 32))),
            dropout=float(parameters.get("dropout", 0.15)),
            criteria=self.criteria,
        )
        self.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
        self.frame_head.load_state_dict(checkpoint["frame_head_state"], strict=True)
        self.temporal_head.load_state_dict(checkpoint["temporal_head_state"], strict=True)
        self.encoder.freeze().to(self.device)
        self.frame_head.requires_grad_(False).to(self.device).eval()
        self.temporal_head.requires_grad_(False).to(self.device).eval()
        self.ordinal = True
        self.temporal = True
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
            ),
        ])
        self._cached_timestamps: tuple[float, ...] | None = None
        self._cached_decoded_times: list[float] = []
        self._cached_support: dict[str, torch.Tensor] = {}
        self._cached_full: dict[str, torch.Tensor] = {}
        self._cached_boundary: dict[str, torch.Tensor] = {}

    def _frame_logits(self, head, features: torch.Tensor) -> torch.Tensor:
        output = head(features)
        criteria = self.criteria
        support = torch.stack(
            [output[key]["support_or_full"] for key in criteria], dim=-1,
        )
        full = torch.stack([output[key]["full_only"] for key in criteria], dim=-1)
        return torch.cat([support, full], dim=-1)

    @torch.inference_mode()
    def _infer_window(self, timestamps: list[float]) -> None:
        cache_key = tuple(timestamps)
        if cache_key == self._cached_timestamps:
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video for temporal PeskaVLP evidence: {self.video_path}")
        frames, decoded_times = [], []
        for time_s in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            top, left = (360 - 224) // 2, (640 - 224) // 2
            crop = cv2.cvtColor(
                resized[top:top + 224, left:left + 224], cv2.COLOR_BGR2RGB,
            )
            frames.append(self.transform(crop.copy()))
            decoded_times.append(float(time_s))
        cap.release()
        if not frames:
            raise RuntimeError("Temporal PeskaVLP evidence tool decoded no candidate frames")

        feature_batches = []
        for start in range(0, len(frames), self.inference_batch_size):
            images = torch.stack(frames[start:start + self.inference_batch_size]).to(self.device)
            feature_batches.append(self.encoder(images).cpu())
        features = torch.cat(feature_batches)
        frame_logit_batches = []
        for start in range(0, len(features), self.inference_batch_size * 4):
            frame_logit_batches.append(self._frame_logits(
                self.frame_head, features[start:start + self.inference_batch_size * 4].to(self.device),
            ).cpu())
        frame_logits = torch.cat(frame_logit_batches)
        times = torch.tensor(decoded_times, dtype=torch.float32)
        progress = ((times - times[0]) / max(float(times[-1] - times[0]), 1.0)).clamp(0, 1)
        if self.temporal_enabled:
            prediction = self.temporal_head(
                features.unsqueeze(0).to(self.device),
                frame_logits.unsqueeze(0).to(self.device),
                progress.unsqueeze(0).to(self.device),
            )
            support = torch.sigmoid(prediction["support_or_full"][0]).cpu()
            full = torch.sigmoid(prediction["full_only"][0]).cpu()
            boundary = torch.sigmoid(prediction["boundary"][0]).cpu()
        else:
            support = torch.sigmoid(frame_logits[:, :3])
            full = torch.sigmoid(frame_logits[:, 3:])
            boundary = torch.full((len(frame_logits), 3, 2), float("nan"))
        self._cached_timestamps = cache_key
        self._cached_decoded_times = decoded_times
        self._cached_support = {
            criterion: support[:, index]
            for index, criterion in enumerate(self.criteria)
        }
        self._cached_full = {
            criterion: full[:, index]
            for index, criterion in enumerate(self.criteria)
        }
        self._cached_boundary = {
            criterion: boundary[:, index]
            for index, criterion in enumerate(self.criteria)
        }

    def score(self, criterion: str, timestamps: list[float]) -> list[ScorePoint]:
        self._infer_window(timestamps)
        if criterion not in self._cached_full:
            raise KeyError(f"Temporal PeskaVLP CVS head has no criterion: {criterion}")
        return [
            ScorePoint(time_s=time_s, score=score.item(), raw_score=score.item(), quality=1.0)
            for time_s, score in zip(self._cached_decoded_times, self._cached_full[criterion])
        ]

    def structured_ordinal_evidence(self, timestamps: list[float]) -> dict[str, list[dict]]:
        self._infer_window(timestamps)
        calibrated = self.metadata.get("ordinal_thresholds_calibrated_on_validation", {})
        support_thresholds = calibrated.get("support_or_full", {})
        full_thresholds = calibrated.get("full_only", {})
        output = {}
        for criterion in self.criteria:
            observations = []
            support_threshold = float(support_thresholds.get(criterion, 0.5))
            full_threshold = float(full_thresholds.get(criterion, 0.5))
            for time_s, raw_support, full, boundary in zip(
                self._cached_decoded_times,
                self._cached_support[criterion],
                self._cached_full[criterion],
                self._cached_boundary[criterion],
            ):
                support = max(raw_support.item(), full.item())
                state = (
                    "full" if full.item() >= full_threshold else
                    "partial" if support >= support_threshold else "absent"
                )
                observations.append({
                    "time_s": time_s,
                    "support_or_full_score": support,
                    "raw_support_or_full_score": raw_support.item(),
                    "full_only_score": full.item(),
                    "onset_probability": (
                        boundary[0].item() if self.temporal_enabled else None
                    ),
                    "offset_probability": (
                        boundary[1].item() if self.temporal_enabled else None
                    ),
                    "predicted_state": state,
                    "support_threshold": support_threshold,
                    "full_threshold": full_threshold,
                    "monotonic_projection_applied": full.item() > raw_support.item(),
                    "ordinal_consistent": True,
                })
            output[criterion] = observations
        return output


def export_representative_frame(video_path: str, time_s: float, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read representative frame at {time_s:.2f}s")
    label = f"t={time_s:.1f}s"
    cv2.putText(frame, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), frame)
    return str(output_path)


def export_monitoring_hit_frames(
    video_path: str, output_dir: str | Path, records: list[dict[str, Any]],
    context_label: str = "procedural-video monitoring",
    object_observations: list[Any] | None = None,
) -> dict[str, Any]:
    """Export real decoded frames with model/MLLM/fusion evidence overlays."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cards, frames = [], []
    colors = {"yes": (45, 190, 70), "no": (45, 55, 220), "unknown": (20, 190, 230)}
    object_colors = {
        "tool": (40, 60, 240), "cystic_duct": (30, 210, 255),
        "cystic_artery": (220, 80, 220), "cystic_plate": (60, 210, 70),
        "calot_triangle": (255, 150, 40), "gallbladder": (220, 210, 60),
    }
    objects_by_time = {
        round(float(item.time_s), 6): item for item in (object_observations or [])
    }
    for index, record in enumerate(records):
        time_s = float(record["time_s"])
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        object_frame = objects_by_time.get(round(time_s, 6))
        if object_frame is not None:
            for observation in object_frame.observations:
                x1, y1, x2, y2 = [int(round(value)) for value in observation.bbox_xyxy]
                box_color = object_colors.get(observation.semantic_type, (230, 230, 230))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                label = f"{observation.semantic_type} {observation.confidence:.2f}"
                cv2.putText(
                    frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, box_color, 2, cv2.LINE_AA,
                )
        frame = cv2.resize(frame, (720, 405), interpolation=cv2.INTER_AREA)
        state = str(record.get("mllm_state", "unknown"))
        color = colors.get(state, colors["unknown"])
        cv2.rectangle(frame, (2, 2), (717, 402), color, 5)
        footer = np.full((145, 720, 3), 22, dtype=np.uint8)
        criterion = str(record.get("criterion", "criterion"))
        lines = [
            f"{context_label} | {criterion} | t={time_s:.1f}s",
            f"trigger={record.get('trigger_reason')}  small={float(record.get('small_score', 0)):.3f}  fused={float(record.get('fused_score', 0)):.3f}  state={record.get('fused_evidence_state', 'n/a')}",
            f"Qwen-VL={state}  confidence={float(record.get('mllm_confidence', 0)):.2f}  visibility={record.get('visibility', 'unknown')}",
        ]
        facts = record.get("observed_facts", [])
        evaluation = ""
        if record.get("ground_truth_state") is not None:
            evaluation = (
                f"GT={record['ground_truth_state']}  result={record.get('monitoring_outcome', 'n/a')}  "
            )
        if facts or evaluation:
            lines.append(evaluation + ("observed: " + str(facts[0])[:45] if facts else ""))
        for line_index, line in enumerate(lines):
            base_scale = 0.58
            width = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, base_scale, 1,
            )[0][0]
            font_scale = max(0.38, min(base_scale, base_scale * 690 / max(width, 1)))
            cv2.putText(
                footer, line, (14, 27 + 31 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (235, 235, 235), 1, cv2.LINE_AA,
            )
        card = np.vstack([frame, footer])
        safe_criterion = re.sub(r"[^a-zA-Z0-9_-]+", "_", criterion)
        target = output_dir / f"{index+1:02d}_{safe_criterion}_{time_s:.1f}s_{state}.jpg"
        cv2.imwrite(str(target), card)
        frames.append(str(target))
        cards.append(card)
    cap.release()
    sheet_path = None
    if cards:
        columns = 2
        rows = math.ceil(len(cards) / columns)
        sheet = np.full((rows * 550, columns * 720, 3), 12, dtype=np.uint8)
        for index, card in enumerate(cards):
            row, column = divmod(index, columns)
            sheet[row * 550:(row + 1) * 550, column * 720:(column + 1) * 720] = card
        sheet_path = output_dir / "monitoring_hits_contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet)
    return {"frames": frames, "contact_sheet": str(sheet_path) if sheet_path else None}


def write_scores_csv(path: str | Path, scores: dict[str, list[ScorePoint]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "criterion", "time_s", "score", "raw_score", "quality",
            "evidence_state", "state_confidence",
        ])
        for criterion, points in scores.items():
            for point in points:
                writer.writerow([
                    criterion, point.time_s, point.score, point.raw_score, point.quality,
                    point.evidence_state, point.state_confidence,
                ])


def write_score_plot(path: str | Path, scores: dict[str, list[ScorePoint]], intervals: dict[str, list]) -> None:
    """Dependency-free evidence plot; x axis is shared video time in seconds."""
    width, height, margin = 1500, 820, 80
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    colors = [(33, 102, 225), (29, 148, 83), (196, 75, 42)]
    all_times = [p.time_s for series in scores.values() for p in series]
    if not all_times:
        return
    start, end = min(all_times), max(all_times)
    cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (220, 220, 220), 1)
    for threshold, label in ((0.68, "pass threshold"), (0.52, "off threshold")):
        y = int(height - margin - threshold * (height - 2 * margin))
        cv2.line(canvas, (margin, y), (width - margin, y), (205, 205, 205), 1)
        cv2.putText(canvas, label, (width - margin - 150, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    for index, (criterion, points) in enumerate(scores.items()):
        color = colors[index % len(colors)]
        for interval in intervals.get(criterion, []):
            x1 = int(margin + (interval.start_s - start) / (end - start) * (width - 2 * margin))
            x2 = int(margin + (interval.end_s - start) / (end - start) * (width - 2 * margin))
            cv2.rectangle(canvas, (x1, margin + index * 10), (x2, margin + index * 10 + 7), color, -1)
        previous = None
        for point in points:
            x = int(margin + (point.time_s - start) / (end - start) * (width - 2 * margin))
            y = int(height - margin - point.score * (height - 2 * margin))
            if previous:
                cv2.line(canvas, previous, (x, y), color, 2)
            previous = (x, y)
        cv2.putText(canvas, criterion, (margin + 8, margin + 35 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(canvas, f"Video time: {start:.0f}s to {end:.0f}s (pre-clipping window)", (margin, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    cv2.imwrite(str(path), canvas)
