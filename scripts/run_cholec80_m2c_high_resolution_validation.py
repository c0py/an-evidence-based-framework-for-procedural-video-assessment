"""Evaluate train-motivated short-event temporal rules and 2 s sampling on validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.models import FeatureFusionCalibrator
from cvs_assessment.object_observations import default_object_localizers
from cvs_assessment.schema import ScorePoint
from cvs_assessment.temporal import StableEvidenceAggregator
from cvs_assessment.tools import PeskaVLPCheckpointScorer
from run_cholec80_validation_ablation import (
    CRITERIA, expand_aggregated_intervals, method_summary,
    paired_bootstrap_comparison, temporal_metrics, truth_intervals,
)
from train_cholec80_m2b_spatial_calibrator import cache_video


TEMPORAL_VARIANTS = {
    "M2b_5s_current_main": {
        "smoothing_seconds": 9.0, "min_stable_seconds": 12.0,
        "max_gap_seconds": 5.0,
    },
    "M2c_5s_short_event": {
        "smoothing_seconds": 5.0, "min_stable_seconds": 5.0,
        "max_gap_seconds": 5.0,
    },
    "M2c_2s_short_event": {
        "smoothing_seconds": 5.0, "min_stable_seconds": 5.0,
        "max_gap_seconds": 5.0,
    },
}


def load_calibrator(path: Path) -> tuple[FeatureFusionCalibrator, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    model = FeatureFusionCalibrator(
        int(state["network.0.weight"].shape[1]),
        int(state["network.3.weight"].shape[0]),
        int(state["network.0.weight"].shape[0]), 0.0,
    )
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint["metadata"]


def predict(
    value: dict[str, Any], model: FeatureFusionCalibrator, metadata: dict[str, Any],
) -> torch.Tensor:
    features = value["features"][:, metadata["feature_indices"]]
    mean = torch.as_tensor(metadata["feature_mean"], dtype=torch.float32)
    std = torch.as_tensor(metadata["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    with torch.inference_mode():
        return torch.sigmoid(model((features - mean) / std))


def evaluate(
    videos: list[dict[str, Any]], model: FeatureFusionCalibrator,
    metadata: dict[str, Any], temporal: dict[str, float], dataset_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    thresholds = metadata["thresholds_calibrated_on_validation"]
    annotation_path = dataset_root / "annotations" / "cholec80-CVS.xlsx"
    for value in videos:
        video_id = int(value["video_id"])
        scores = predict(value, model, metadata)
        timestamps = [float(item) for item in value["timestamps_s"].tolist()]
        start_s = float(value["window"]["start_s"])
        end_s = float(value["window"]["end_s"])
        cadence_s = float(value["cadence_s"])
        annotations = load_cvs_intervals(annotation_path, video_id)
        for index, criterion in enumerate(CRITERIA):
            on_threshold = float(thresholds[criterion])
            parameters = {
                **temporal, "on_threshold": on_threshold,
                "off_threshold": max(0.05, 0.8 * on_threshold),
            }
            points = [
                ScorePoint(time_s, float(score), float(score), 1.0)
                for time_s, score in zip(timestamps, scores[:, index].tolist())
            ]
            _, evidence = StableEvidenceAggregator(**parameters).aggregate_structured(points)
            predicted = expand_aggregated_intervals(
                evidence.positive_intervals, cadence_s, start_s, end_s,
            )
            truth = truth_intervals(annotations, criterion, start_s, end_s)
            rows.append({
                "video_id": video_id, "criterion": criterion,
                "threshold": on_threshold, **temporal_metrics(predicted, truth),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--m2b-checkpoint", required=True)
    parser.add_argument("--cache-5s", required=True)
    parser.add_argument("--cache-2s", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="5")
    args = parser.parse_args()

    experiment_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(experiment_cfg["base_config"]).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    validation_ids = [int(value) for value in split["validation_video_ids"]]
    test_ids = set(map(int, split["test_video_ids"]))
    if set(validation_ids) & test_ids:
        raise ValueError("Validation/test overlap")
    dataset_root = Path(split["dataset_root"])
    cache_5s = Path(args.cache_5s)
    cache_2s = Path(args.cache_2s)
    observation_dir = cache_2s / "object_observations"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    object_cfg = dict(experiment_cfg["object_localizer"])
    object_cfg["device"] = str(args.device)
    provider = default_object_localizers().build(object_cfg["provider"], object_cfg)
    small_cfg = base_cfg["small_mllm_fusion"]["small_model"]
    scorer = PeskaVLPCheckpointScorer(
        str(dataset_root / "videos" / f"video{validation_ids[0]:02d}.mp4"),
        small_cfg["checkpoint"], device=f"cuda:{args.device}",
        inference_batch_size=int(small_cfg["inference_batch_size"]),
    )
    high_resolution = [
        cache_video(
            video_id, dataset_root, 2.0, scorer, provider, cache_2s,
            observation_dir,
        )
        for video_id in validation_ids
    ]
    low_resolution = [
        torch.load(
            cache_5s / f"video{video_id:02d}.pt", map_location="cpu", weights_only=False,
        )
        for video_id in validation_ids
    ]
    model, metadata = load_calibrator(Path(args.m2b_checkpoint))
    rows = {
        "M2b_5s_current_main": evaluate(
            low_resolution, model, metadata,
            TEMPORAL_VARIANTS["M2b_5s_current_main"], dataset_root,
        ),
        "M2c_5s_short_event": evaluate(
            low_resolution, model, metadata,
            TEMPORAL_VARIANTS["M2c_5s_short_event"], dataset_root,
        ),
        "M2c_2s_short_event": evaluate(
            high_resolution, model, metadata,
            TEMPORAL_VARIANTS["M2c_2s_short_event"], dataset_root,
        ),
    }
    summaries = {
        name: method_summary(value, 20260728, 5000) for name, value in rows.items()
    }
    comparisons = {
        "short_5s_minus_current": paired_bootstrap_comparison(
            rows["M2b_5s_current_main"], rows["M2c_5s_short_event"],
            20260728, 5000, "current", "short_5s",
        ),
        "highres_2s_minus_short_5s": paired_bootstrap_comparison(
            rows["M2c_5s_short_event"], rows["M2c_2s_short_event"],
            20260728, 5000, "short_5s", "highres_2s",
        ),
        "highres_2s_minus_current": paired_bootstrap_comparison(
            rows["M2b_5s_current_main"], rows["M2c_2s_short_event"],
            20260728, 5000, "current", "highres_2s",
        ),
    }
    output = {
        "experiment": "cholec80_m2c_short_event_high_resolution_validation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "train_motivated_temporal_rule": TEMPORAL_VARIANTS["M2c_5s_short_event"],
            "validation_video_ids": validation_ids,
            "test_video_ids_loaded": [], "test_labels_accessed": False,
            "same_frozen_m2b_checkpoint": str(Path(args.m2b_checkpoint).resolve()),
            "same_thresholds": metadata["thresholds_calibrated_on_validation"],
            "changed_variables": ["temporal duration rule", "sampling cadence"],
        },
        "method_summaries": summaries,
        "comparisons": comparisons,
        "method_rows": rows,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(result_path.resolve(), flush=True)
    for name, summary in summaries.items():
        print(
            f"{name} temporal_iou={summary['macro_temporal_iou']:.6f} "
            f"positive_iou={summary['macro_positive_pair_temporal_iou']:.6f} "
            f"positive_detection={summary['positive_pair_detection_rate']:.6f} "
            f"positive_segment_f1={summary['macro_positive_pair_segment_f1_at_iou_0_3']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
