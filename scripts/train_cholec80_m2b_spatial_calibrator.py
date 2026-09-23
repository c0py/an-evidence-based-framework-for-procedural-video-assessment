"""Train matched small-only and small+bbox calibrators without test access."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvs_assessment.annotations import load_cvs_intervals, load_phase_starts
from cvs_assessment.models import FeatureFusionCalibrator
from cvs_assessment.object_observations import (
    FrameObjectObservations,
    default_object_localizers,
    spatial_fusion_feature_names,
    spatial_fusion_features,
)
from cvs_assessment.schema import ScorePoint
from cvs_assessment.temporal import StableEvidenceAggregator
from cvs_assessment.tools import PeskaVLPCheckpointScorer
from run_cholec80_validation_ablation import (
    CRITERIA,
    expand_aggregated_intervals,
    method_summary,
    paired_bootstrap_comparison,
    sha256,
    temporal_metrics,
    truth_intervals,
)
from train_peskavlp_cvs_head import binary_metrics


OBJECT_CLASSES = (
    "cystic_plate", "calot_triangle", "cystic_artery",
    "cystic_duct", "gallbladder", "tool",
)
CANDIDATE_RULES = {
    "two_structures": {
        "operator": "geometric_mean", "classes": ["cystic_duct", "cystic_artery"],
    },
    "cystic_plate": {"operator": "max", "classes": ["cystic_plate"]},
    "hepatocystic_triangle": {"operator": "max", "classes": ["calot_triangle"]},
}


def feature_names() -> list[str]:
    return spatial_fusion_feature_names(CRITERIA, OBJECT_CLASSES)


def bbox_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1),
    )
    union = (
        max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def spatial_features(frame: dict[str, Any]) -> list[float]:
    value = {"time_s": float(frame.get("time_s", 0.0)), **frame}
    encoded = spatial_fusion_features(
        FrameObjectObservations.from_dict(value),
        {criterion: 0.0 for criterion in CRITERIA},
        CRITERIA, OBJECT_CLASSES, CANDIDATE_RULES,
    )
    names = feature_names()[len(CRITERIA):]
    names.remove("phase_progress")
    return [encoded[name] for name in names]


def labels_at(annotations: dict, time_s: float) -> list[float]:
    return [
        float(any(start <= time_s <= end and state >= 2 for start, end, state in annotations[key]))
        for key in CRITERIA
    ]


def load_observation_frames(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["frames"]


def reset_small_scorer(scorer: PeskaVLPCheckpointScorer, video_path: Path) -> None:
    scorer.video_path = str(video_path)
    scorer._cached_timestamps = None
    scorer._cached_logits = {}
    scorer._cached_support_logits = {}


def cache_video(
    video_id: int, dataset_root: Path, cadence_s: float,
    scorer: PeskaVLPCheckpointScorer, provider: Any,
    cache_dir: Path, observation_dir: Path,
) -> dict[str, Any]:
    destination = cache_dir / f"video{video_id:02d}.pt"
    if destination.exists():
        value = torch.load(destination, map_location="cpu", weights_only=False)
        if value.get("feature_names") != feature_names() or float(value["cadence_s"]) != cadence_s:
            raise RuntimeError(f"Stale M2b cache schema: {destination}")
        print(f"m2b_cache_hit video={video_id:02d} samples={len(value['timestamps_s'])}", flush=True)
        return value

    phases = load_phase_starts(
        dataset_root / "phase_annotations" / f"video{video_id:02d}-phase.txt"
    )
    start_s = float(phases["CalotTriangleDissection"])
    end_s = float(phases["ClippingCutting"])
    timestamps = np.arange(start_s, end_s, cadence_s).astype(float).tolist()
    video_path = dataset_root / "videos" / f"video{video_id:02d}.mp4"
    observation_path = observation_dir / f"video{video_id:02d}.json"
    if observation_path.exists():
        frames = load_observation_frames(observation_path)
    else:
        observed = provider.observe(video_path, timestamps)
        frames = [frame.to_dict() for frame in observed]
        observation_path.parent.mkdir(parents=True, exist_ok=True)
        observation_path.write_text(json.dumps({
            "video_id": video_id,
            "video_path": str(video_path.resolve()),
            "meaning": "Predicted object observations; not CVS ground truth.",
            "frames": frames,
        }, indent=2))
    frame_by_time = {round(float(frame["time_s"]), 6): frame for frame in frames}

    reset_small_scorer(scorer, video_path)
    small_points = {criterion: scorer.score(criterion, timestamps) for criterion in CRITERIA}
    decoded_times = [point.time_s for point in small_points[CRITERIA[0]]]
    annotations = load_cvs_intervals(
        dataset_root / "annotations" / "cholec80-CVS.xlsx", video_id,
    )
    rows, targets = [], []
    for index, time_s in enumerate(decoded_times):
        small = [float(small_points[key][index].score) for key in CRITERIA]
        frame = frame_by_time.get(round(float(time_s), 6), {"observations": []})
        spatial = spatial_features(frame)
        progress = (time_s - start_s) / max(1.0, end_s - start_s)
        # phase_progress is placed before the three candidate features.
        rows.append(small + spatial[:-3] + [progress] + spatial[-3:])
        targets.append(labels_at(annotations, time_s))
    value = {
        "video_id": video_id,
        "timestamps_s": torch.tensor(decoded_times, dtype=torch.float32),
        "features": torch.tensor(rows, dtype=torch.float32),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "window": {"start_s": start_s, "end_s": end_s},
        "cadence_s": cadence_s,
        "feature_names": feature_names(),
        "observation_path": str(observation_path.resolve()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)
    print(
        f"m2b_cache_saved video={video_id:02d} samples={len(decoded_times)} "
        f"boxes={sum(len(frame.get('observations', [])) for frame in frames)}",
        flush=True,
    )
    return value


def macro_auprc(scores: torch.Tensor, truth: torch.Tensor) -> float:
    values = []
    for index in range(len(CRITERIA)):
        value = binary_metrics(scores[:, index], truth[:, index].bool(), 0.5)["auprc"]
        if value is not None:
            values.append(float(value))
    return sum(values) / max(1, len(values))


def train_calibrator(
    name: str, train_features: torch.Tensor, train_targets: torch.Tensor,
    validation_features: torch.Tensor, validation_targets: torch.Tensor,
    indices: list[int], cfg: dict[str, Any], device: str,
) -> dict[str, Any]:
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    selected_train = train_features[:, indices]
    selected_validation = validation_features[:, indices]
    mean = selected_train.mean(dim=0)
    std = selected_train.std(dim=0).clamp_min(1e-6)
    normalized_train = (selected_train - mean) / std
    normalized_validation = (selected_validation - mean) / std
    model = FeatureFusionCalibrator(
        len(indices), len(CRITERIA), int(cfg["hidden_dim"]), float(cfg["dropout"]),
    ).to(device)
    positives = train_targets.sum(dim=0)
    pos_weight = ((len(train_targets) - positives) / positives.clamp_min(1)).clamp(
        max=float(cfg["positive_weight_cap"]),
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    batch_size = int(cfg["batch_size"])
    best_score, best_epoch, best_state = -1.0, 0, None
    history = []
    stale = 0
    for epoch in range(int(cfg["epochs"])):
        model.train()
        permutation = torch.randperm(len(normalized_train))
        total_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start:start + batch_size]
            logits = model(normalized_train[batch].to(device))
            loss = loss_fn(logits, train_targets[batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(batch)
        model.eval()
        with torch.inference_mode():
            scores = torch.sigmoid(model(normalized_validation.to(device))).cpu()
        selection = macro_auprc(scores, validation_targets)
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(normalized_train),
            "validation_macro_auprc": selection,
        })
        if selection > best_score + 1e-5:
            best_score, best_epoch = selection, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {name}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        validation_scores = torch.sigmoid(
            model(normalized_validation.to(device))
        ).cpu()
    return {
        "name": name,
        "model": model.cpu(),
        "feature_indices": indices,
        "feature_mean": mean,
        "feature_std": std,
        "validation_scores": validation_scores,
        "best_epoch": best_epoch,
        "best_validation_macro_auprc": best_score,
        "history": history,
        "pos_weight": pos_weight.cpu(),
    }


def calibrate_thresholds(scores: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    output = {}
    for index, criterion in enumerate(CRITERIA):
        truth = targets[:, index].bool()
        best_threshold, best_f1 = 0.5, -1.0
        for candidate in torch.linspace(0.05, 0.95, 91):
            threshold = float(candidate)
            f1 = binary_metrics(scores[:, index], truth, threshold)["f1"]
            if f1 > best_f1:
                best_threshold, best_f1 = threshold, f1
        output[criterion] = best_threshold
    return output


def frame_metrics(scores: torch.Tensor, targets: torch.Tensor, thresholds: dict[str, float]) -> dict:
    return {
        criterion: binary_metrics(
            scores[:, index], targets[:, index].bool(), thresholds[criterion],
        )
        for index, criterion in enumerate(CRITERIA)
    }


def temporal_rows(
    validation_videos: list[dict[str, Any]], score_chunks: list[torch.Tensor],
    thresholds: dict[str, float], base_cfg: dict[str, Any], dataset_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    temporal_defaults = {key: float(value) for key, value in base_cfg["temporal"].items()}
    annotation_path = dataset_root / "annotations" / "cholec80-CVS.xlsx"
    for video, scores in zip(validation_videos, score_chunks):
        video_id = int(video["video_id"])
        timestamps = video["timestamps_s"].tolist()
        start_s, end_s = float(video["window"]["start_s"]), float(video["window"]["end_s"])
        annotations = load_cvs_intervals(annotation_path, video_id)
        for index, criterion in enumerate(CRITERIA):
            on_threshold = float(thresholds[criterion])
            parameters = {
                **temporal_defaults,
                "on_threshold": on_threshold,
                "off_threshold": max(0.05, 0.8 * on_threshold),
            }
            points = [
                ScorePoint(float(time_s), float(score), float(score), 1.0)
                for time_s, score in zip(timestamps, scores[:, index].tolist())
            ]
            _, evidence = StableEvidenceAggregator(**parameters).aggregate_structured(points)
            predicted = expand_aggregated_intervals(
                evidence.positive_intervals, float(video["cadence_s"]), start_s, end_s,
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
    parser.add_argument("--m1-results", required=True)
    parser.add_argument("--validation-observation-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    experiment_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(experiment_cfg["base_config"]).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    train_ids = [int(value) for value in split["train_video_ids"]]
    validation_ids = [int(value) for value in split["validation_video_ids"]]
    test_ids = {int(value) for value in split["test_video_ids"]}
    if (set(train_ids) | set(validation_ids)) & test_ids or set(train_ids) & set(validation_ids):
        raise ValueError("Frozen train/validation/test roles overlap")

    m2b_cfg = dict(experiment_cfg["m2b"])
    if args.seed is not None:
        m2b_cfg["seed"] = args.seed
    cache_dir = Path(m2b_cfg["feature_cache_dir"])
    observation_dir = cache_dir / "object_observations"
    output_dir = Path(args.output_dir or m2b_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    object_cfg = experiment_cfg["object_localizer"]
    provider = default_object_localizers().build(object_cfg["provider"], object_cfg)
    small_cfg = base_cfg["small_mllm_fusion"]["small_model"]
    checkpoint_path = Path(small_cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("split_manifest", {}).get("train_video_ids") != split["train_video_ids"]:
        raise ValueError("Small checkpoint and frozen split differ")
    if metadata.get("test_metrics") is not None or metadata.get("test_per_video_metrics") is not None:
        raise ValueError("Small checkpoint indicates test access")
    scorer = PeskaVLPCheckpointScorer(
        str(Path(split["dataset_root"]) / "videos" / f"video{train_ids[0]:02d}.mp4"),
        str(checkpoint_path), inference_batch_size=int(small_cfg["inference_batch_size"]),
    )
    dataset_root = Path(split["dataset_root"])
    cadence_s = 1.0 / float(small_cfg["sampling_fps"])

    videos: dict[int, dict[str, Any]] = {}
    validation_source_dir = Path(args.validation_observation_dir)
    for video_id in train_ids + validation_ids:
        validation_source = validation_source_dir / f"video{video_id:02d}.json"
        if video_id in validation_ids and validation_source.exists():
            target = observation_dir / validation_source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(validation_source.resolve())
        videos[video_id] = cache_video(
            video_id, dataset_root, cadence_s, scorer, provider, cache_dir,
            observation_dir,
        )

    train_videos = [videos[video_id] for video_id in train_ids]
    validation_videos = [videos[video_id] for video_id in validation_ids]
    train_features = torch.cat([value["features"] for value in train_videos])
    train_targets = torch.cat([value["targets"] for value in train_videos])
    validation_features = torch.cat([value["features"] for value in validation_videos])
    validation_targets = torch.cat([value["targets"] for value in validation_videos])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phase_index = feature_names().index("phase_progress")
    variants = {
        "learned_small_only": list(range(3)),
        "learned_small_phase": [0, 1, 2, phase_index],
        "learned_small_bbox_no_phase": [
            index for index in range(len(feature_names())) if index != phase_index
        ],
        "M2b_learned_small_bbox": list(range(len(feature_names()))),
    }
    trained = {
        name: train_calibrator(
            name, train_features, train_targets, validation_features,
            validation_targets, indices, m2b_cfg, device,
        )
        for name, indices in variants.items()
    }
    offsets, cursor = [], 0
    for value in validation_videos:
        length = len(value["timestamps_s"])
        offsets.append((cursor, cursor + length))
        cursor += length

    frozen_m1 = json.loads(Path(args.m1_results).read_text(encoding="utf-8"))
    rows = {"M1_framework_temporal": frozen_m1["method_rows"]["M1_framework_temporal"]}
    summaries = {
        "M1_framework_temporal": frozen_m1["method_summaries"]["M1_framework_temporal"]
    }
    model_artifacts = {}
    frame_results = {}
    for name, value in trained.items():
        scores = value["validation_scores"]
        thresholds = calibrate_thresholds(scores, validation_targets)
        chunks = [scores[start:end] for start, end in offsets]
        rows[name] = temporal_rows(
            validation_videos, chunks, thresholds, base_cfg, dataset_root,
        )
        summaries[name] = method_summary(rows[name], 20260728, 2000)
        frame_results[name] = frame_metrics(scores, validation_targets, thresholds)
        checkpoint_output = output_dir / f"{name}_seed{m2b_cfg['seed']}.pt"
        torch.save({
            "model_state": value["model"].state_dict(),
            "metadata": {
                "model_type": "feature_fusion_calibrator",
                "variant": name,
                "feature_names": feature_names(),
                "feature_indices": value["feature_indices"],
                "feature_mean": value["feature_mean"],
                "feature_std": value["feature_std"],
                "thresholds_calibrated_on_validation": thresholds,
                "best_epoch": value["best_epoch"],
                "best_validation_macro_auprc": value["best_validation_macro_auprc"],
                "train_video_ids": train_ids,
                "validation_video_ids": validation_ids,
                "test_video_ids_held_out": sorted(test_ids),
                "test_labels_accessed": False,
                "history": value["history"],
            },
        }, checkpoint_output)
        model_artifacts[name] = str(checkpoint_output.resolve())

    comparisons = {
        "learned_small_only_minus_M1": paired_bootstrap_comparison(
            rows["M1_framework_temporal"], rows["learned_small_only"],
            20260728, 2000, "M1", "learned_small_only",
        ),
        "M2b_minus_learned_small_only": paired_bootstrap_comparison(
            rows["learned_small_only"], rows["M2b_learned_small_bbox"],
            20260728, 2000, "learned_small_only", "M2b",
        ),
        "learned_small_phase_minus_small_only": paired_bootstrap_comparison(
            rows["learned_small_only"], rows["learned_small_phase"],
            20260728, 2000, "learned_small_only", "learned_small_phase",
        ),
        "learned_bbox_no_phase_minus_small_only": paired_bootstrap_comparison(
            rows["learned_small_only"], rows["learned_small_bbox_no_phase"],
            20260728, 2000, "learned_small_only", "learned_bbox_no_phase",
        ),
        "M2b_minus_learned_small_phase": paired_bootstrap_comparison(
            rows["learned_small_phase"], rows["M2b_learned_small_bbox"],
            20260728, 2000, "learned_small_phase", "M2b",
        ),
        "M2b_minus_learned_bbox_no_phase": paired_bootstrap_comparison(
            rows["learned_small_bbox_no_phase"], rows["M2b_learned_small_bbox"],
            20260728, 2000, "learned_bbox_no_phase", "M2b",
        ),
        "M2b_minus_M1": paired_bootstrap_comparison(
            rows["M1_framework_temporal"], rows["M2b_learned_small_bbox"],
            20260728, 2000, "M1", "M2b",
        ),
    }
    output = {
        "experiment": "cholec80_cvs_m2b_train_only_spatial_calibrator_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "train_video_ids": train_ids,
            "validation_video_ids": validation_ids,
            "test_video_ids_loaded": [],
            "test_labels_accessed": False,
            "model_selection": "validation macro frame AUPRC",
            "threshold_calibration": "validation frame F1",
            "matched_ablation": "same MLP/training; small-only uses first three features, M2b uses all spatial features",
            "factorial_ablation": "small-only, small+phase, small+bbox without phase, and small+bbox+phase",
        },
        "feature_names": feature_names(),
        "n_train_samples": len(train_features),
        "n_validation_samples": len(validation_features),
        "train_positive_counts": train_targets.sum(dim=0).tolist(),
        "validation_positive_counts": validation_targets.sum(dim=0).tolist(),
        "small_checkpoint": str(checkpoint_path.resolve()),
        "small_checkpoint_sha256": sha256(checkpoint_path),
        "localizer_checkpoint": provider.checkpoint,
        "localizer_checkpoint_sha256": sha256(Path(provider.checkpoint)),
        "model_artifacts": model_artifacts,
        "frame_metrics": frame_results,
        "method_summaries": summaries,
        "comparisons": comparisons,
        "method_rows": rows,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(output, indent=2))
    print(result_path, flush=True)
    for name, summary in summaries.items():
        print(
            f"{name} temporal_iou={summary['macro_temporal_iou']:.6f} "
            f"positive_iou={summary['macro_positive_pair_temporal_iou']:.6f} "
            f"segment_f1={summary['macro_segment_f1_at_iou_0_3']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
