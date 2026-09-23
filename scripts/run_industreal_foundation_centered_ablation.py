#!/usr/bin/env python3
"""Run the SAGES-matched frozen-MLLM ablation on real IndustReal videos."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.mllm_orchestration import (  # noqa: E402
    FOUNDATION_CENTERED_ABLATIONS,
    FrozenMLLMJudge,
    MLLMJudgeRequest,
    PluginEvidence,
)
from cvs_assessment.evidence_adjudication import (  # noqa: E402
    detach_evidence_adjudication_audit,
    route_grounded_payload_for_decision,
)
from cvs_assessment.foundation_arbitration import (  # noqa: E402
    filter_plugin_for_arbitration,
    merge_arbitrated_judgment,
    select_arbitration_frame_ids,
    slice_foundation_judgment,
)
from cvs_assessment.tasks.industreal_mllm_facts import (  # noqa: E402
    ASD_STATE_PATTERNS,
    assembly_criterion_contracts,
    labels_at_timestamps,
    temporal_fact_payload,
    visual_fact_payload,
)
from cvs_assessment.tasks.industreal import create_industreal_real_psr_package  # noqa: E402


class IndustRealFrozenMLLMJudge(FrozenMLLMJudge):
    """Auditably bound optional explanations without changing task decisions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_contract_normalization: dict[str, Any] | None = None

    def validate_response(
        self, value: Any, request: MLLMJudgeRequest,
        available_plugin_ids: set[str] | None = None,
        probability_prior: list[list[float]] | None = None,
        log_odds_step: float = 0.5,
        decision_protocol: str = "direct_probability",
    ) -> dict[str, Any]:
        self._last_contract_normalization = None
        findings = value.get("key_findings") if isinstance(value, dict) else None
        if (
            isinstance(findings, list) and 8 < len(findings) <= len(request.criteria)
        ):
            original_count = len(findings)
            value = deepcopy(value)
            value["key_findings"] = findings[:8]
            self._last_contract_normalization = {
                "method": "deterministic_optional_key_findings_prefix_bound",
                "original_count": original_count,
                "retained_count": 8,
                "frame_predictions_changed": False,
                "criterion_states_or_probabilities_changed": False,
                "plugin_assessment_changed": False,
                "raw_completion_preserved": True,
            }
        return super().validate_response(
            value, request, available_plugin_ids, probability_prior,
            log_odds_step, decision_protocol,
        )

    def judge(self, request: MLLMJudgeRequest, ablation: Any) -> dict[str, Any]:
        self._last_contract_normalization = None
        output = super().judge(request, ablation)
        output["response_contract_normalization"] = (
            self._last_contract_normalization or {
                "method": "none", "frame_predictions_changed": False,
                "criterion_states_or_probabilities_changed": False,
            }
        )
        return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels).reshape(-1) >= 0.5
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-values, kind="stable")
    ranked = truth[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def metric_summary(
    labels: np.ndarray, scores: np.ndarray, criterion_ids: list[str],
    timestamps_by_video: list[list[float]],
) -> dict[str, Any]:
    aps: dict[str, float] = {}
    binary_rows: dict[str, dict[str, Any]] = {}
    delays: list[float] = []
    for index, criterion_id in enumerate(criterion_ids):
        truth = labels[:, :, index].reshape(-1) >= 0.5
        predicted = scores[:, :, index].reshape(-1) >= 0.60
        aps[criterion_id] = average_precision(truth, scores[:, :, index].reshape(-1))
        tp = int(np.logical_and(truth, predicted).sum())
        tn = int(np.logical_and(~truth, ~predicted).sum())
        fp = int(np.logical_and(~truth, predicted).sum())
        fn = int(np.logical_and(truth, ~predicted).sum())
        sensitivity = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        balanced = (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None else None
        )
        binary_rows[criterion_id] = {
            "true_positive": tp, "true_negative": tn,
            "false_positive": fp, "false_negative": fn,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": balanced, "f1": f1,
        }
        for video_index, timestamps_s in enumerate(timestamps_by_video):
            y = labels[video_index, :, index] >= 0.5
            p = scores[video_index, :, index] >= 0.60
            if y.any() and p.any():
                delays.append(
                    float(timestamps_s[int(np.flatnonzero(p)[0])])
                    - float(timestamps_s[int(np.flatnonzero(y)[0])])
                )
    finite_aps = [value for value in aps.values() if np.isfinite(value)]
    defined_ba = [
        row["balanced_accuracy"] for row in binary_rows.values()
        if row["balanced_accuracy"] is not None
    ]
    defined_f1 = [
        row["f1"] for row in binary_rows.values()
        if row["sensitivity"] is not None and row["f1"] is not None
    ]
    return {
        "average_precision_by_criterion": aps,
        "macro_average_precision_over_defined_criteria": (
            float(np.mean(finite_aps)) if finite_aps else None
        ),
        "full_state_binary_by_criterion": binary_rows,
        "macro_balanced_accuracy_over_defined_criteria": (
            float(np.mean(defined_ba)) if defined_ba else None
        ),
        "macro_f1_over_positive_defined_criteria": (
            float(np.mean(defined_f1)) if defined_f1 else None
        ),
        "mean_first_full_transition_delay_s": (
            float(np.mean(delays)) if delays else None
        ),
    }


def video_index(video_root: Path) -> dict[str, Path]:
    output = {path.stem: path for path in video_root.rglob("*.mp4")}
    if not output:
        raise FileNotFoundError(f"No IndustReal MP4 files under {video_root}")
    return output


def video_metadata(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid IndustReal video metadata: {path}")
    return count, fps, width, height


def read_video_frames(path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    requested = sorted(set(int(value) for value in frame_indices))
    capture = cv2.VideoCapture(str(path))
    output: dict[int, np.ndarray] = {}
    try:
        for frame_index in requested:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
            output[frame_index] = frame
    finally:
        capture.release()
    return output


def _official_state_index(class_id: int, names: dict[int, str] | list[str]) -> int:
    name = str(names[class_id] if isinstance(names, (dict, list)) else class_id).lower()
    if name in ASD_STATE_PATTERNS:
        return ASD_STATE_PATTERNS.index(name)
    if "error" in name:
        return ASD_STATE_PATTERNS.index("error_state")
    match = re.search(r"(?:state[_ -]?)?(\d+)$", name)
    if match and 1 <= int(match.group(1)) <= 22:
        return int(match.group(1))
    count = len(names)
    if count == 23 and 0 <= class_id < 23:
        return class_id + 1
    if count == 24 and 0 <= class_id < 24:
        return class_id
    raise ValueError(f"Cannot map ASD model class {class_id}:{name!r} to official state order")


def detector_rows(
    detector: Any, frames: dict[int, np.ndarray], device: str,
) -> list[dict[str, Any]]:
    indices = sorted(frames)
    results = detector.predict(
        [frames[index] for index in indices], imgsz=640, conf=0.15,
        iou=0.7, max_det=4, device=device, verbose=False,
    )
    output: list[dict[str, Any]] = []
    for frame_index, result in zip(indices, results):
        for box in result.boxes:
            class_id = int(box.cls[0])
            output.append({
                "frame_index": frame_index,
                "state_class_index": _official_state_index(class_id, result.names),
                "confidence": float(box.conf[0]),
                "bbox_xywh": [float(value) for value in box.xywh[0].tolist()],
            })
    return output


def detector_rows_from_video(
    detector: Any, path: Path, frame_indices: list[int], device: str,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Run the dense tool in bounded-memory batches while decoding once."""
    requested = sorted(set(int(value) for value in frame_indices))
    if not requested or batch_size < 1:
        raise ValueError("Dense detector indices/batch size are invalid")
    wanted = set(requested)
    maximum = requested[-1]
    capture = cv2.VideoCapture(str(path))
    batch: dict[int, np.ndarray] = {}
    output: list[dict[str, Any]] = []
    try:
        for frame_index in range(maximum + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot sequentially decode frame {frame_index} from {path}")
            if frame_index not in wanted:
                continue
            batch[frame_index] = frame
            if len(batch) >= batch_size:
                output.extend(detector_rows(detector, batch, device))
                batch.clear()
        if batch:
            output.extend(detector_rows(detector, batch, device))
    finally:
        capture.release()
    return output


def encode_jpeg(frame: np.ndarray, quality: int = 88) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Cannot encode IndustReal evidence frame")
    return encoded.tobytes()


def render_overlay(
    raw: np.ndarray, rows: list[dict[str, Any]], source_frame_index: int,
) -> np.ndarray:
    output = raw.copy()
    for row in rows:
        if int(row["frame_index"]) != source_frame_index:
            continue
        x, y, width, height = row["bbox_xywh"]
        left, top = round(x - width / 2), round(y - height / 2)
        right, bottom = round(x + width / 2), round(y + height / 2)
        cv2.rectangle(output, (left, top), (right, bottom), (30, 220, 255), 3)
        label = (
            f"ASD {ASD_STATE_PATTERNS[int(row['state_class_index'])]} "
            f"{float(row['confidence']):.2f}"
        )
        cv2.putText(
            output, label, (max(0, left), max(24, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 220, 255), 2, cv2.LINE_AA,
        )
    return output


def paired_jpeg(raw: np.ndarray, overlay: np.ndarray) -> bytes:
    height, width = raw.shape[:2]
    left, right = raw.copy(), overlay.copy()
    cv2.rectangle(left, (0, 0), (width, 26), (0, 0, 0), -1)
    cv2.rectangle(right, (0, 0), (width, 26), (0, 0, 0), -1)
    cv2.putText(left, "RAW FRAME", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(right, "FALLIBLE ASD TOOL", (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return encode_jpeg(np.concatenate([left, right], axis=1), quality=90)


def prediction_array(judgment: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        [criterion["probability_satisfied"] for criterion in frame["criteria"]]
        for frame in judgment["prediction"]["frames"]
    ], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--skill", type=Path, default=ROOT / "specs/industreal_psr_assembly_skill.md")
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-device", default="5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--video-limit", type=int, default=1)
    parser.add_argument("--frame-count", type=int, default=12)
    parser.add_argument(
        "--tool-fps", type=float, default=10.0,
        help="Dense tool cadence; 10 FPS matches the official IndustReal PSR release.",
    )
    parser.add_argument("--detector-batch-size", type=int, default=64)
    parser.add_argument("--selection-seed", default="industreal-transfer-v1:")
    parser.add_argument("--base-url", default="http://127.0.0.1:18903/v1")
    parser.add_argument("--model", default="qwen3-vl-32b-sop")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=float, default=1200)
    parser.add_argument("--maximum-context-facts", type=int, default=8)
    parser.add_argument("--maximum-arbitration-frames", type=int, default=6)
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(FOUNDATION_CENTERED_ABLATIONS),
        default=list(FOUNDATION_CENTERED_ABLATIONS),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--facts-only", action="store_true",
        help="Cache label-blind dense tool facts without calling the foundation MLLM.",
    )
    args = parser.parse_args()
    if (
        args.frame_count < 2 or args.tool_fps <= 0
        or args.detector_batch_size < 1 or args.maximum_context_facts < 0
        or args.maximum_arbitration_frames < 1
    ):
        raise ValueError("IndustReal frame count/tool FPS are invalid")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    annotations = json.loads(args.normalized_annotations.read_text())
    if annotations.get("official_test_annotations_accessed") is not False:
        raise PermissionError("Development runner refuses unlocked IndustReal test annotations")
    recordings = annotations["recordings"]
    eligible = [
        recording_id for recording_id, row in recordings.items()
        if row["split"] == args.split and row["procedure_kind"] == "assy"
    ]
    if args.video_id:
        video_ids = args.video_id
    else:
        video_ids = sorted(
            eligible,
            key=lambda value: hashlib.sha256(
                (args.selection_seed + value).encode()
            ).hexdigest(),
        )[:args.video_limit]
    if any(value not in eligible for value in video_ids):
        raise KeyError("Requested recording is unavailable in the permitted split/assembly task")
    videos = video_index(args.video_root)
    if any(value not in videos for value in video_ids):
        raise FileNotFoundError("One or more selected IndustReal MP4 files are missing")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)

    from ultralytics import YOLO
    detector = YOLO(str(args.detector_checkpoint))
    detector_sha256 = sha256_file(args.detector_checkpoint)
    judge = IndustRealFrozenMLLMJudge(
        args.base_url, args.model, args.api_key, args.timeout_s, max_tokens=4096,
    )
    skill_text = args.skill.read_text()
    task_package = create_industreal_real_psr_package(
        annotations["procedure_info"], "assy",
    )
    criteria = [
        next(
            criterion for criterion in assembly_criterion_contracts(
                annotations["procedure_info"], "assy",
            )
            if criterion.criterion_id == criterion_id
        )
        for criterion_id in task_package.criterion_catalog
    ]
    criterion_ids = [criterion.criterion_id for criterion in criteria]
    records: list[dict[str, Any]] = []

    for video_id in video_ids:
        path = videos[video_id]
        total_frames, fps, width, height = video_metadata(path)
        sample_indices = sorted(set(
            int(round(value)) for value in np.linspace(0, total_frames - 1, args.frame_count)
        ))
        dense_step = max(1, int(round(fps / args.tool_fps)))
        dense_indices = sorted(set(range(0, total_frames, dense_step)) | set(sample_indices))
        raw_frame_map = read_video_frames(path, sample_indices)
        cache_path = args.output_dir / f"{video_id}__asd_rows.json"
        if cache_path.exists() and args.resume:
            cached = json.loads(cache_path.read_text())
            expected_cache = {
                "video_id": video_id,
                "total_frames": total_frames,
                "fps": fps,
                "dense_step_frames": dense_step,
                "detector_sha256": detector_sha256,
            }
            if any(cached.get(key) != value for key, value in expected_cache.items()):
                raise ValueError(f"Incompatible dense tool cache: {cache_path}")
            rows = cached["rows"]
            tool_cache_resumed = True
        else:
            rows = detector_rows_from_video(
                detector, path, dense_indices, args.detector_device,
                batch_size=args.detector_batch_size,
            )
            cache_path.write_text(json.dumps({
                "schema_version": "industreal_dense_asd_rows_v1",
                "video_id": video_id,
                "total_frames": total_frames,
                "fps": fps,
                "dense_step_frames": dense_step,
                "detector_sha256": detector_sha256,
                "source_video_path": str(path.resolve()),
                "source_video_sha256": sha256_file(path),
                "rows": rows,
                "official_test_annotations_accessed": False,
                "final_task_prediction_provided": False,
            }, indent=2) + "\n")
            tool_cache_resumed = False
        timestamps_s = [index / fps for index in sample_indices]
        raw_frames = [raw_frame_map[index] for index in sample_indices]
        raw_jpegs = [encode_jpeg(frame) for frame in raw_frames]
        visual_facts = visual_fact_payload(
            rows, timestamps_s, sample_indices, frame_width=width, frame_height=height,
        )
        temporal_facts = temporal_fact_payload(
            rows, fps=fps, expected_cadence_frames=dense_step,
        )
        upstream_plugins = [
            PluginEvidence(
                "industreal_official_asd_visual_v1", "visual",
                "A frozen assembly-state detector supplies fallible, box-grounded component "
                "state observations; it never decides procedure-step completion.",
                visual_facts,
            ),
            PluginEvidence(
                "industreal_dense_state_transition_v1", "temporal",
                "A dense temporal tool reports fallible component-state transitions from the "
                "same frozen detector; Qwen verifies correctness, persistence, and order.",
                temporal_facts,
            ),
        ]
        plugins: list[PluginEvidence] = []
        routing_audits: dict[str, dict[str, Any]] = {}
        for plugin in upstream_plugins:
            routed = route_grounded_payload_for_decision(
                plugin.payload,
                maximum_context_facts=(
                    max(args.maximum_context_facts, len(plugin.payload.get("facts", [])))
                    if plugin.plugin_kind == "temporal"
                    else args.maximum_context_facts
                ),
                retain_context_without_supported_anchor=(plugin.plugin_kind == "temporal"),
            )
            prompt_payload, routing_audits[plugin.plugin_id] = (
                detach_evidence_adjudication_audit(routed)
            )
            plugins.append(PluginEvidence(
                plugin.plugin_id, plugin.plugin_kind, plugin.description,
                prompt_payload, plugin.foundation_model_parameters_updated,
            ))
        fact_path = args.output_dir / f"{video_id}__grounded_facts.json"
        fact_path.write_text(json.dumps({
            "recording_id": video_id,
            "sample_source_frame_indices": sample_indices,
            "timestamps_s": timestamps_s,
            "visual_plugin": visual_facts,
            "temporal_plugin": temporal_facts,
            "decision_routed_plugins": {
                plugin.plugin_id: plugin.payload for plugin in plugins
            },
            "evidence_adjudication_audits": routing_audits,
            "final_task_prediction_provided_by_plugins": False,
            "official_test_annotations_accessed": False,
        }, indent=2) + "\n")
        print(json.dumps({
            "video_id": video_id,
            "dense_tool_frames": len(dense_indices),
            "dense_detection_rows": len(rows),
            "tool_cache": str(cache_path),
            "tool_cache_resumed": tool_cache_resumed,
        }), flush=True)
        if args.facts_only:
            continue
        label_matrix = labels_at_timestamps(recordings[video_id], criteria, timestamps_s)

        for variant_name in args.variants:
            ablation = FOUNDATION_CENTERED_ABLATIONS[variant_name]
            output_path = args.output_dir / f"{video_id}__{variant_name}.json"
            if output_path.exists() and args.resume:
                judgment = json.loads(output_path.read_text())
                resumed = True
            else:
                try:
                    preliminary = None
                    auxiliary: list[dict[str, Any]] = []
                    request_plugins = plugins
                    request_frame_ids = list(range(len(sample_indices)))
                    request_timestamps_s = timestamps_s
                    request_jpegs = raw_jpegs
                    arbitration_frame_ids: list[int] = []
                    if variant_name == "full_framework":
                        preliminary_path = (
                            args.output_dir / f"{video_id}__mllm_skill.json"
                        )
                        if preliminary_path.exists():
                            preliminary = json.loads(preliminary_path.read_text())
                        else:
                            preliminary_request = MLLMJudgeRequest(
                                task_id=task_package.task_id, sample_id=video_id,
                                criteria=list(criteria),
                                frame_ids=list(range(len(sample_indices))),
                                timestamps_s=timestamps_s, frame_jpegs=raw_jpegs,
                                skill_text=skill_text, plugin_evidence=plugins,
                            )
                            preliminary = judge.judge(
                                preliminary_request,
                                FOUNDATION_CENTERED_ABLATIONS["mllm_skill"],
                            )
                            preliminary_path.write_text(
                                json.dumps(preliminary, indent=2) + "\n"
                            )
                        auxiliary_path = (
                            args.output_dir / f"{video_id}__mllm_skill_temporal.json"
                        )
                        if auxiliary_path.exists():
                            temporal_hypothesis = json.loads(auxiliary_path.read_text())
                        else:
                            auxiliary_request = MLLMJudgeRequest(
                                task_id=task_package.task_id, sample_id=video_id,
                                criteria=list(criteria),
                                frame_ids=list(range(len(sample_indices))),
                                timestamps_s=timestamps_s, frame_jpegs=raw_jpegs,
                                skill_text=skill_text, plugin_evidence=plugins,
                            )
                            temporal_hypothesis = judge.judge(
                                auxiliary_request,
                                FOUNDATION_CENTERED_ABLATIONS["mllm_skill_temporal"],
                            )
                            auxiliary_path.write_text(
                                json.dumps(temporal_hypothesis, indent=2) + "\n"
                            )
                        auxiliary = [temporal_hypothesis]
                        arbitration_frame_ids = select_arbitration_frame_ids(
                            preliminary, temporal_hypothesis, plugins,
                            maximum_frames=args.maximum_arbitration_frames,
                        )
                        request_frame_ids = arbitration_frame_ids
                        request_timestamps_s = [
                            timestamps_s[frame_id] for frame_id in arbitration_frame_ids
                        ]
                        request_jpegs = [
                            raw_jpegs[frame_id] for frame_id in arbitration_frame_ids
                        ]
                        request_plugins = [
                            filter_plugin_for_arbitration(
                                plugin, arbitration_frame_ids, request_timestamps_s,
                            )
                            for plugin in plugins
                        ]
                        preliminary = slice_foundation_judgment(
                            preliminary, arbitration_frame_ids,
                        )
                        auxiliary = [slice_foundation_judgment(
                            temporal_hypothesis, arbitration_frame_ids,
                        )]
                    request = MLLMJudgeRequest(
                        task_id=task_package.task_id, sample_id=video_id,
                        criteria=list(criteria),
                        frame_ids=request_frame_ids,
                        timestamps_s=request_timestamps_s, frame_jpegs=request_jpegs,
                        skill_text=skill_text, plugin_evidence=request_plugins,
                        foundation_preliminary_judgment=preliminary,
                        foundation_auxiliary_judgments=auxiliary,
                    )
                    judgment = judge.judge(request, ablation)
                    if preliminary is not None:
                        judgment = merge_arbitrated_judgment(
                            json.loads(preliminary_path.read_text()), judgment,
                            arbitration_frame_ids,
                        )
                        judgment["inference_stages"] = {
                            "stage_1": "same_frozen_qwen_skill_preliminary",
                            "stage_1_auxiliary": "same_frozen_qwen_skill_temporal_hypothesis",
                            "stage_2": "same_frozen_qwen_branch_and_plugin_arbitration_final",
                            "preliminary_path": str(preliminary_path.resolve()),
                            "auxiliary_path": str(auxiliary_path.resolve()),
                            "preliminary_latency_s": preliminary["runtime"]["latency_s"],
                            "auxiliary_latency_s": temporal_hypothesis["runtime"]["latency_s"],
                            "final_latency_s": judgment["runtime"]["latency_s"],
                            "total_latency_s": (
                                json.loads(preliminary_path.read_text())["runtime"]["latency_s"]
                                + temporal_hypothesis["runtime"]["latency_s"]
                                + judgment["runtime"]["latency_s"]
                            ),
                            "arbitrated_frame_ids": arbitration_frame_ids,
                            "maximum_arbitration_frames": args.maximum_arbitration_frames,
                            "small_model_is_final_judge": False,
                            "foundation_model_parameters_updated": False,
                        }
                except Exception as exc:
                    error_path = args.output_dir / f"{video_id}__{variant_name}__ERROR.json"
                    error_path.write_text(json.dumps({
                        "schema_version": "industreal_foundation_request_error_v1",
                        "video_id": video_id, "variant": variant_name,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "safe_to_resume": True,
                        "foundation_model_parameters_updated": False,
                        "official_test_annotations_accessed": False,
                    }, indent=2) + "\n")
                    print(json.dumps({
                        "video_id": video_id, "variant": variant_name,
                        "error_path": str(error_path), "error": str(exc)[:500],
                    }), flush=True)
                    continue
                output_path.write_text(json.dumps(judgment, indent=2) + "\n")
                resumed = False
            records.append({
                "video_id": video_id, "variant": variant_name,
                "labels": label_matrix, "prediction": prediction_array(judgment),
                "timestamps_s": timestamps_s,
                "latency_s": judgment.get("inference_stages", {}).get(
                    "total_latency_s", judgment["runtime"]["latency_s"],
                ),
            })
            print(json.dumps({
                "video_id": video_id, "variant": variant_name,
                "output": str(output_path), "resumed": resumed,
                "latency_s": judgment["runtime"]["latency_s"],
            }), flush=True)

    summary: dict[str, Any] = {
        "schema_version": "industreal_foundation_centered_fact_ablation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "video_ids": video_ids,
        "criterion_order": criterion_ids, "variants": {},
        "task_package": {
            "task_id": task_package.task_id,
            "procedure_kind": task_package.metadata["procedure_kind"],
            "ordering_constraints": task_package.ordering_constraints,
        },
        "architecture_contract": {
            "foundation_mllm_is_common_base": True,
            "foundation_mllm_is_final_judge": True,
            "skill_is_replaceable_task_package": True,
            "plugins_provide_final_task_verdict": False,
            "dense_temporal_tool_analyzes_more_frames_than_mllm": True,
            "all_arms_receive_identical_raw_frames": True,
            "full_framework_fusion": "same_frozen_qwen_skill_and_temporal_branches_then_plugin_grounded_arbitration",
            "full_framework_arbitration_scope": "label_blind_branch_disagreements_plus_two_frame_full_candidate_context",
        },
        "sampling": {
            "mllm_frame_count": args.frame_count,
            "dense_tool_fps": args.tool_fps,
            "detector_batch_size": args.detector_batch_size,
            "selection_uses_labels": False,
        },
        "evidence_adjudication": {
            "policy": "retain_calibrated_supported_and_bounded_direct_grounding_v1",
            "maximum_context_facts_per_plugin": args.maximum_context_facts,
            "maximum_arbitration_frames": args.maximum_arbitration_frames,
            "task_labels_accessed_by_router": False,
        },
        "sources": {
            "normalized_annotations": {
                "path": str(args.normalized_annotations.resolve()),
                "sha256": sha256_file(args.normalized_annotations),
            },
            "skill": {"path": str(args.skill.resolve()), "sha256": sha256_file(args.skill)},
            "detector": {
                "path": str(args.detector_checkpoint.resolve()),
                "sha256": detector_sha256,
            },
        },
        "facts_only": args.facts_only,
        "foundation_model_parameters_updated": False,
        "labels_in_mllm_prompt": False,
        "official_test_annotations_accessed": False,
        "note": "Train/validation transfer development; not an untouched final estimate.",
    }
    for variant_name in args.variants:
        subset = [row for row in records if row["variant"] == variant_name]
        if not subset:
            summary["variants"][variant_name] = {
                "completed_video_count": 0,
                "requested_video_count": len(video_ids), "incomplete": True,
            }
            continue
        labels = np.stack([row["labels"] for row in subset])
        predictions = np.stack([row["prediction"] for row in subset])
        result = metric_summary(
            labels, predictions, criterion_ids,
            [row["timestamps_s"] for row in subset],
        )
        result.update({
            "completed_video_count": len(subset),
            "requested_video_count": len(video_ids),
            "incomplete": len(subset) != len(video_ids),
            "mean_latency_s": float(np.mean([row["latency_s"] for row in subset])),
        })
        summary["variants"][variant_name] = result
    summary_path = args.output_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
