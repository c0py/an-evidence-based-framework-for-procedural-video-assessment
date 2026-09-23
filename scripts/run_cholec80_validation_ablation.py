"""Run the frozen Cholec80 validation M0/M1 ablation without touching test labels.

M0 thresholds independent frame scores and converts positive samples into
sampling cells.  M1 applies the framework's causal smoothing, hysteresis,
minimum-duration, and gap rules to the exact same scores.  The paired design
therefore isolates temporal aggregation from the visual model and sampling.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvs_assessment.annotations import load_cvs_intervals, load_phase_starts
from cvs_assessment.schema import ScorePoint
from cvs_assessment.temporal import StableEvidenceAggregator
from cvs_assessment.tools import PeskaVLPCheckpointScorer
from train_peskavlp_cvs_head import binary_metrics


CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def sampling_cell_intervals(
    positive_times: list[float], cadence_s: float, start_s: float, end_s: float,
) -> list[tuple[float, float]]:
    half = cadence_s / 2.0
    return merge_intervals([
        (max(start_s, time_s - half), min(end_s, time_s + half))
        for time_s in positive_times
    ])


def expand_aggregated_intervals(
    intervals: list[Any], cadence_s: float, start_s: float, end_s: float,
) -> list[tuple[float, float]]:
    half = cadence_s / 2.0
    return merge_intervals([
        (max(start_s, float(item.start_s) - half), min(end_s, float(item.end_s) + half))
        for item in intervals
    ])


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = (left[1] - left[0]) + (right[1] - right[0]) - intersection
    return intersection / union if union > 0 else 0.0


def segment_metrics(
    predicted: list[tuple[float, float]], truth: list[tuple[float, float]],
    iou_threshold: float = 0.3,
) -> dict[str, float | int]:
    candidates = sorted(
        (
            (interval_iou(predicted[pred_index], truth[truth_index]), pred_index, truth_index)
            for pred_index in range(len(predicted))
            for truth_index in range(len(truth))
        ),
        reverse=True,
    )
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    matches = 0
    for iou, pred_index, truth_index in candidates:
        if iou < iou_threshold:
            break
        if pred_index in used_predicted or truth_index in used_truth:
            continue
        used_predicted.add(pred_index)
        used_truth.add(truth_index)
        matches += 1
    precision = matches / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = matches / len(truth) if truth else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou_threshold": iou_threshold,
        "tp": matches,
        "fp": len(predicted) - matches,
        "fn": len(truth) - matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def temporal_metrics(
    predicted: list[tuple[float, float]], truth: list[tuple[float, float]],
) -> dict[str, Any]:
    predicted, truth = merge_intervals(predicted), merge_intervals(truth)
    intersection = sum(
        max(0.0, min(pred_end, truth_end) - max(pred_start, truth_start))
        for pred_start, pred_end in predicted
        for truth_start, truth_end in truth
    )
    predicted_duration = duration(predicted)
    truth_duration = duration(truth)
    union = predicted_duration + truth_duration - intersection
    onset_error = (
        predicted[0][0] - truth[0][0] if predicted and truth else None
    )
    return {
        "predicted_intervals": predicted,
        "truth_intervals": truth,
        "predicted_duration_s": predicted_duration,
        "truth_duration_s": truth_duration,
        "intersection_s": intersection,
        "temporal_iou": intersection / union if union else 1.0,
        "temporal_precision": (
            intersection / predicted_duration
            if predicted_duration else (1.0 if not truth else 0.0)
        ),
        "temporal_recall": (
            intersection / truth_duration
            if truth_duration else (1.0 if not predicted else 0.0)
        ),
        "presence_correct": bool(predicted) == bool(truth),
        "onset_error_s": onset_error,
        "absolute_onset_error_s": abs(onset_error) if onset_error is not None else None,
        "segment_f1_at_iou_0_3": segment_metrics(predicted, truth, 0.3),
    }


def truth_intervals(
    annotations: dict[str, list[tuple[float, float, int]]], criterion: str,
    start_s: float, end_s: float,
) -> list[tuple[float, float]]:
    return merge_intervals([
        (max(start_s, start), min(end_s, end))
        for start, end, state in annotations[criterion]
        if state >= 2 and min(end_s, end) > max(start_s, start)
    ])


def label_at(intervals: list[tuple[float, float]], time_s: float) -> bool:
    return any(start <= time_s <= end for start, end in intervals)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_rows = [row for row in rows if row["truth_intervals"]]
    negative_rows = [row for row in rows if not row["truth_intervals"]]
    onset = [float(row["absolute_onset_error_s"]) for row in rows if row["absolute_onset_error_s"] is not None]
    return {
        "n_criterion_video_pairs": len(rows),
        "macro_temporal_iou": mean([float(row["temporal_iou"]) for row in rows]),
        "macro_temporal_precision": mean([float(row["temporal_precision"]) for row in rows]),
        "macro_temporal_recall": mean([float(row["temporal_recall"]) for row in rows]),
        "macro_segment_f1_at_iou_0_3": mean([
            float(row["segment_f1_at_iou_0_3"]["f1"]) for row in rows
        ]),
        "presence_accuracy": mean([float(row["presence_correct"]) for row in rows]),
        "n_truth_positive_pairs": len(positive_rows),
        "macro_positive_pair_temporal_iou": mean([
            float(row["temporal_iou"]) for row in positive_rows
        ]),
        "macro_positive_pair_temporal_precision": mean([
            float(row["temporal_precision"]) for row in positive_rows
        ]),
        "macro_positive_pair_temporal_recall": mean([
            float(row["temporal_recall"]) for row in positive_rows
        ]),
        "macro_positive_pair_segment_f1_at_iou_0_3": mean([
            float(row["segment_f1_at_iou_0_3"]["f1"]) for row in positive_rows
        ]),
        "positive_pair_detection_rate": mean([
            float(bool(row["predicted_intervals"])) for row in positive_rows
        ]),
        "n_truth_negative_pairs": len(negative_rows),
        "negative_pair_rejection_rate": mean([
            float(not row["predicted_intervals"]) for row in negative_rows
        ]),
        "onset_mae_s_detected_pairs": mean(onset) if onset else None,
        "n_detected_positive_pairs_for_onset": len(onset),
    }


def bootstrap_by_video(
    rows: list[dict[str, Any]], seed: int, repetitions: int,
) -> dict[str, dict[str, float]]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[int(row["video_id"])].append(row)
    video_ids = sorted(by_video)
    rng = np.random.default_rng(seed)
    metric_names = (
        "temporal_iou", "positive_pair_temporal_iou", "segment_f1",
        "positive_pair_segment_f1", "presence_accuracy",
    )
    samples = {name: [] for name in metric_names}
    for _ in range(repetitions):
        selected = rng.choice(video_ids, size=len(video_ids), replace=True)
        sampled_rows = [row for video_id in selected for row in by_video[int(video_id)]]
        positive_rows = [row for row in sampled_rows if row["truth_intervals"]]
        samples["temporal_iou"].append(mean([float(row["temporal_iou"]) for row in sampled_rows]))
        samples["positive_pair_temporal_iou"].append(mean([
            float(row["temporal_iou"]) for row in positive_rows
        ]))
        samples["segment_f1"].append(mean([
            float(row["segment_f1_at_iou_0_3"]["f1"]) for row in sampled_rows
        ]))
        samples["positive_pair_segment_f1"].append(mean([
            float(row["segment_f1_at_iou_0_3"]["f1"]) for row in positive_rows
        ]))
        samples["presence_accuracy"].append(mean([
            float(row["presence_correct"]) for row in sampled_rows
        ]))
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
    }


def method_summary(
    rows: list[dict[str, Any]], seed: int, bootstrap_repetitions: int,
) -> dict[str, Any]:
    by_criterion = {
        criterion: summarize_rows([row for row in rows if row["criterion"] == criterion])
        for criterion in CRITERIA
    }
    return {
        **summarize_rows(rows),
        "by_criterion": by_criterion,
        "video_bootstrap_95_ci": bootstrap_by_video(rows, seed, bootstrap_repetitions),
    }


def paired_bootstrap_comparison(
    baseline_rows: list[dict[str, Any]], temporal_rows: list[dict[str, Any]],
    seed: int, repetitions: int, baseline_label: str = "M0",
    comparison_label: str = "M1",
) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[int, str]:
        return int(row["video_id"]), str(row["criterion"])

    baseline = {key(row): row for row in baseline_rows}
    temporal = {key(row): row for row in temporal_rows}
    if baseline.keys() != temporal.keys():
        raise ValueError("M0/M1 rows are not paired on identical video-criterion keys")
    video_ids = sorted({video_id for video_id, _ in baseline})

    metric_map = {
        "macro_temporal_iou": "macro_temporal_iou",
        "macro_positive_pair_temporal_iou": "macro_positive_pair_temporal_iou",
        "macro_segment_f1_at_iou_0_3": "macro_segment_f1_at_iou_0_3",
        "macro_positive_pair_segment_f1_at_iou_0_3": "macro_positive_pair_segment_f1_at_iou_0_3",
        "presence_accuracy": "presence_accuracy",
    }
    base_summary = summarize_rows(list(baseline.values()))
    temporal_summary = summarize_rows(list(temporal.values()))
    output = {
        name: {
            baseline_label: float(base_summary[source]),
            comparison_label: float(temporal_summary[source]),
            f"absolute_delta_{comparison_label}_minus_{baseline_label}": float(
                temporal_summary[source] - base_summary[source]
            ),
        }
        for name, source in metric_map.items()
    }

    rng = np.random.default_rng(seed)
    deltas = {name: [] for name in metric_map}
    for _ in range(repetitions):
        selected = rng.choice(video_ids, size=len(video_ids), replace=True)
        sampled_keys = [
            (int(video_id), criterion)
            for video_id in selected for criterion in CRITERIA
        ]
        sampled_base = [baseline[item] for item in sampled_keys]
        sampled_temporal = [temporal[item] for item in sampled_keys]
        sampled_base_summary = summarize_rows(sampled_base)
        sampled_temporal_summary = summarize_rows(sampled_temporal)
        for name, source in metric_map.items():
            deltas[name].append(
                float(sampled_temporal_summary[source] - sampled_base_summary[source])
            )
    for name, values in deltas.items():
        output[name].update({
            "paired_video_bootstrap_lower_95": float(np.quantile(values, 0.025)),
            "paired_video_bootstrap_upper_95": float(np.quantile(values, 0.975)),
            "bootstrap_probability_delta_gt_0": float(np.mean(np.asarray(values) > 0)),
        })
    output["by_criterion"] = {}
    for criterion in CRITERIA:
        criterion_base = summarize_rows([
            row for row in baseline.values() if row["criterion"] == criterion
        ])
        criterion_temporal = summarize_rows([
            row for row in temporal.values() if row["criterion"] == criterion
        ])
        output["by_criterion"][criterion] = {
            name: float(criterion_temporal[source] - criterion_base[source])
            for name, source in metric_map.items()
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    split_path = Path(args.split_json).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation_ids = [int(value) for value in split["validation_video_ids"]]
    test_ids = {int(value) for value in split["test_video_ids"]}
    if set(validation_ids) & test_ids:
        raise ValueError("Validation/test overlap in frozen split")
    if len(validation_ids) != 15:
        raise ValueError(f"Expected 15 frozen validation videos, got {len(validation_ids)}")

    small_cfg = cfg.get("small_mllm_fusion", {}).get("small_model", {})
    checkpoint_path = Path(
        small_cfg.get("checkpoint", cfg.get("peskavlp_checkpoint", ""))
    ).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    checkpoint_split = metadata.get("split_manifest", {})
    if checkpoint_split.get("validation_video_ids") != split["validation_video_ids"]:
        raise ValueError("Checkpoint and requested frozen validation split do not match")
    if metadata.get("test_metrics") is not None or metadata.get("test_per_video_metrics") is not None:
        raise ValueError("Checkpoint metadata indicates test labels were accessed")

    cadence_s = 1.0 / float(small_cfg.get("sampling_fps", 0.2))
    batch_size = int(small_cfg.get("inference_batch_size", 64))
    dataset_root = Path(split["dataset_root"])
    video_root = dataset_root / "videos"
    phase_root = dataset_root / "phase_annotations"
    annotation_xlsx = dataset_root / "annotations" / "cholec80-CVS.xlsx"
    temporal_defaults = {key: float(value) for key, value in cfg["temporal"].items()}
    criterion_temporal = cfg.get("criterion_temporal", {})

    raw_scores: dict[str, list[float]] = defaultdict(list)
    raw_truth: dict[str, list[bool]] = defaultdict(list)
    rows: dict[str, list[dict[str, Any]]] = {"M0_frame_only": [], "M1_framework_temporal": []}
    per_video: list[dict[str, Any]] = []

    for video_id in validation_ids:
        phase_path = phase_root / f"video{video_id:02d}-phase.txt"
        phases = load_phase_starts(phase_path)
        start_s = float(phases["CalotTriangleDissection"])
        end_s = float(phases["ClippingCutting"])
        timestamps = np.arange(start_s, end_s, cadence_s).astype(float).tolist()
        scorer = PeskaVLPCheckpointScorer(
            str(video_root / f"video{video_id:02d}.mp4"), str(checkpoint_path),
            inference_batch_size=batch_size,
        )
        annotations = load_cvs_intervals(annotation_xlsx, video_id)
        video_record = {
            "video_id": video_id,
            "window": {"start_s": start_s, "end_s": end_s},
            "requested_samples": len(timestamps),
            "criteria": {},
        }
        for criterion in CRITERIA:
            points = scorer.score(criterion, timestamps)
            truth = truth_intervals(annotations, criterion, start_s, end_s)
            effective_temporal = {
                **temporal_defaults,
                **{
                    key: float(value)
                    for key, value in criterion_temporal.get(criterion, {}).items()
                },
            }
            threshold = float(effective_temporal["on_threshold"])
            raw_scores[criterion].extend(float(point.score) for point in points)
            raw_truth[criterion].extend(label_at(truth, point.time_s) for point in points)

            m0_intervals = sampling_cell_intervals(
                [point.time_s for point in points if point.score >= threshold],
                cadence_s, start_s, end_s,
            )
            _, aggregated = StableEvidenceAggregator(**effective_temporal).aggregate_structured(points)
            m1_intervals = expand_aggregated_intervals(
                aggregated.positive_intervals, cadence_s, start_s, end_s,
            )
            for method, predicted in (
                ("M0_frame_only", m0_intervals),
                ("M1_framework_temporal", m1_intervals),
            ):
                row = {
                    "video_id": video_id,
                    "criterion": criterion,
                    "threshold": threshold,
                    **temporal_metrics(predicted, truth),
                }
                rows[method].append(row)
                video_record["criteria"].setdefault(criterion, {})[method] = row
            video_record["criteria"][criterion]["n_decoded_samples"] = len(points)
        per_video.append(video_record)
        print(
            f"validation_video_done={video_id:02d} samples={video_record['requested_samples']}",
            flush=True,
        )
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame_metrics = {}
    for criterion in CRITERIA:
        effective = {
            **temporal_defaults,
            **{
                key: float(value)
                for key, value in criterion_temporal.get(criterion, {}).items()
            },
        }
        frame_metrics[criterion] = binary_metrics(
            torch.tensor(raw_scores[criterion]), torch.tensor(raw_truth[criterion]),
            float(effective["on_threshold"]),
        )
        balanced_accuracy = 0.5 * (
            frame_metrics[criterion]["recall"]
            + frame_metrics[criterion]["tn"]
            / max(1, frame_metrics[criterion]["tn"] + frame_metrics[criterion]["fp"])
        )
        frame_metrics[criterion]["balanced_accuracy"] = balanced_accuracy

    summaries = {
        method: method_summary(values, args.bootstrap_seed, args.bootstrap_repetitions)
        for method, values in rows.items()
    }
    paired_comparison = paired_bootstrap_comparison(
        rows["M0_frame_only"], rows["M1_framework_temporal"],
        args.bootstrap_seed, args.bootstrap_repetitions,
    )
    output = {
        "experiment": "cholec80_cvs_frozen_validation_m0_m1_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "split_role": "validation_only",
            "test_video_ids_loaded": [],
            "test_labels_accessed": False,
            "paired_scores": True,
            "M0_frame_only": "Threshold independent samples and evaluate their sampling-cell coverage.",
            "M1_framework_temporal": "Apply causal median smoothing, hysteresis, minimum duration, and gap filling to the identical samples.",
            "truth_definition": "CVS ordinal state >= 2 (full-only) inside CalotTriangleDissection-to-ClippingCutting.",
            "sampling_cadence_s": cadence_s,
            "segment_iou_threshold": 0.3,
            "bootstrap_unit": "video",
            "bootstrap_repetitions": args.bootstrap_repetitions,
        },
        "config_path": str(config_path),
        "split_path": str(split_path),
        "split_name": split["name"],
        "validation_video_ids": validation_ids,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "frame_metrics_full_only": frame_metrics,
        "method_summaries": summaries,
        "paired_comparison_M1_minus_M0": paired_comparison,
        "method_rows": rows,
        "per_video": per_video,
    }
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(destination, flush=True)
    for method, summary in summaries.items():
        print(
            f"{method} macro_temporal_iou={summary['macro_temporal_iou']:.6f} "
            f"segment_f1={summary['macro_segment_f1_at_iou_0_3']:.6f} "
            f"presence_accuracy={summary['presence_accuracy']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
