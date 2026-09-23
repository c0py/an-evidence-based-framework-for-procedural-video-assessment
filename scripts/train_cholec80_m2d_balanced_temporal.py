"""Train a cadence-matched spatial/temporal calibrator with train-only tuning.

The sealed validation split is evaluated exactly once after all model, threshold,
temporal-policy, and video-gate choices have been made from grouped OOF predictions
on the 50-video training split.  The test split is never loaded.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.models import FeatureFusionCalibrator
from cvs_assessment.schema import ScorePoint
from cvs_assessment.temporal import StableEvidenceAggregator
from run_cholec80_validation_ablation import (
    CRITERIA, expand_aggregated_intervals, method_summary,
    paired_bootstrap_comparison, temporal_metrics, truth_intervals,
)


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    output = np.empty_like(values)
    half = width // 2
    for row in range(len(values)):
        left, right = max(0, row - half), min(len(values), row + half + 1)
        output[row] = values[left:right].mean(axis=0)
    return output


def rolling_max(values: np.ndarray, width: int) -> np.ndarray:
    output = np.empty_like(values)
    half = width // 2
    for row in range(len(values)):
        left, right = max(0, row - half), min(len(values), row + half + 1)
        output[row] = values[left:right].max(axis=0)
    return output


def augment_video(value: dict[str, Any]) -> torch.Tensor:
    """Add task-neutral 6 s/10 s context while excluding phase identity."""
    phase_index = value["feature_names"].index("phase_progress")
    indices = [index for index in range(value["features"].shape[1]) if index != phase_index]
    raw = value["features"][:, indices].numpy()
    return torch.from_numpy(np.concatenate([
        raw, rolling_mean(raw, 3), rolling_mean(raw, 5), rolling_max(raw, 3),
    ], axis=1)).float()


def augmented_names(value: dict[str, Any]) -> list[str]:
    base = [name for name in value["feature_names"] if name != "phase_progress"]
    return base + [f"mean3_{name}" for name in base] + [
        f"mean5_{name}" for name in base
    ] + [f"max3_{name}" for name in base]


def grouped_folds(videos: list[dict[str, Any]], folds: int, seed: int) -> list[list[int]]:
    """Greedily balance rare criterion-positive videos across group folds."""
    rng = np.random.default_rng(seed)
    records = []
    for index, value in enumerate(videos):
        labels = (value["targets"].sum(dim=0) > 0).int().numpy()
        records.append((index, labels, float(rng.random())))
    records.sort(key=lambda item: (-int(item[1].sum()), item[2]))
    assignments = [[] for _ in range(folds)]
    label_counts = np.zeros((folds, len(CRITERIA)), dtype=int)
    maximum_size = int(np.ceil(len(videos) / folds))
    for index, labels, _ in records:
        costs = [
            # First spread the labels carried by this video, then balance fold
            # sizes.  The hard capacity prevents all-negative videos from
            # accumulating in whichever fold happens to have fewer positives.
            (float(np.dot(label_counts[fold], labels)), len(assignments[fold]), fold)
            for fold in range(folds)
            if len(assignments[fold]) < maximum_size
        ]
        selected = min(costs)[-1]
        assignments[selected].append(index)
        label_counts[selected] += labels
    return assignments


def macro_auprc(scores: torch.Tensor, truth: torch.Tensor) -> float:
    from train_peskavlp_cvs_head import binary_metrics
    values = []
    for index in range(len(CRITERIA)):
        metric = binary_metrics(scores[:, index], truth[:, index].bool(), 0.5)["auprc"]
        if metric is not None:
            values.append(float(metric))
    return float(np.mean(values)) if values else 0.0


def fit_model(
    train_x: torch.Tensor, train_y: torch.Tensor, valid_x: torch.Tensor,
    valid_y: torch.Tensor, device: str, seed: int, epochs: int = 160,
    fixed_epochs: int | None = None,
) -> tuple[FeatureFusionCalibrator, torch.Tensor, torch.Tensor, int, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-6)
    model = FeatureFusionCalibrator(train_x.shape[1], len(CRITERIA), 64, 0.15).to(device)
    positives = train_y.sum(0)
    pos_weight = ((len(train_y) - positives) / positives.clamp_min(1)).clamp(max=30).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=8e-4)
    normalized = (train_x - mean) / std
    normalized_valid = (valid_x - mean) / std
    best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
    limit = fixed_epochs or epochs
    for epoch in range(limit):
        model.train()
        order = torch.randperm(len(train_x))
        for start in range(0, len(order), 384):
            batch = order[start:start + 384]
            logits = model(normalized[batch].to(device))
            loss = loss_fn(logits, train_y[batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.inference_mode():
            scores = torch.sigmoid(model(normalized_valid.to(device))).cpu()
        selection = macro_auprc(scores, valid_y)
        if selection > best_score + 1e-5:
            best_state, best_score = copy.deepcopy(model.state_dict()), selection
            best_epoch, stale = epoch + 1, 0
        else:
            stale += 1
        if stale >= 22:
            break
    if fixed_epochs is not None:
        best_state, best_epoch = copy.deepcopy(model.state_dict()), fixed_epochs
        model.eval()
        with torch.inference_mode():
            best_score = macro_auprc(
                torch.sigmoid(model(normalized_valid.to(device))).cpu(), valid_y,
            )
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.cpu(), mean, std, best_epoch, best_score


def predict(model: nn.Module, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model((x - mean) / std)).cpu()


def top3_score(scores: torch.Tensor) -> float:
    count = min(3, len(scores))
    return float(torch.topk(scores, count).values.mean()) if count else 0.0


def aggregate_intervals(
    value: dict[str, Any], scores: torch.Tensor, criterion_index: int,
    parameters: dict[str, float], gate_threshold: float,
) -> list[list[float]]:
    criterion_scores = scores[:, criterion_index]
    if top3_score(criterion_scores) < gate_threshold:
        return []
    on = parameters["on_threshold"]
    points = [
        ScorePoint(float(time_s), float(score), float(score), 1.0)
        for time_s, score in zip(value["timestamps_s"].tolist(), criterion_scores.tolist())
    ]
    _, evidence = StableEvidenceAggregator(
        smoothing_seconds=parameters["smoothing_seconds"],
        on_threshold=on, off_threshold=max(0.05, 0.8 * on),
        min_stable_seconds=parameters["min_stable_seconds"],
        max_gap_seconds=parameters["max_gap_seconds"],
    ).aggregate_structured(points)
    return expand_aggregated_intervals(
        evidence.positive_intervals, float(value["cadence_s"]),
        float(value["window"]["start_s"]), float(value["window"]["end_s"]),
    )


def tune_policy(
    videos: list[dict[str, Any]], scores: list[torch.Tensor], dataset_root: Path,
) -> dict[str, dict[str, float]]:
    annotations = {
        int(value["video_id"]): load_cvs_intervals(
            dataset_root / "annotations" / "cholec80-CVS.xlsx", int(value["video_id"]),
        ) for value in videos
    }
    policies = {}
    for criterion_index, criterion in enumerate(CRITERIA):
        gate_values = sorted({round(top3_score(item[:, criterion_index]), 3) for item in scores})
        if len(gate_values) > 15:
            gate_values = [float(np.quantile(gate_values, q)) for q in np.linspace(0, 0.9, 15)]
        gate_values = sorted({0.0, *gate_values})
        best = None
        for on in np.arange(0.20, 0.86, 0.05):
            for smoothing in (2.0, 4.0, 6.0):
                for stable in (2.0, 4.0, 6.0):
                    for gap in (2.0, 4.0, 6.0):
                        base = {
                            "on_threshold": float(on), "smoothing_seconds": smoothing,
                            "min_stable_seconds": stable, "max_gap_seconds": gap,
                        }
                        cached = []
                        for value, prediction in zip(videos, scores):
                            truth = truth_intervals(
                                annotations[int(value["video_id"])], criterion,
                                float(value["window"]["start_s"]),
                                float(value["window"]["end_s"]),
                            )
                            cached.append((value, prediction, truth))
                        for gate in gate_values:
                            rows = []
                            for value, prediction, truth in cached:
                                predicted = aggregate_intervals(
                                    value, prediction, criterion_index, base, gate,
                                )
                                rows.append(temporal_metrics(predicted, truth))
                            positives = [row for row in rows if row["truth_duration_s"] > 0]
                            negatives = [row for row in rows if row["truth_duration_s"] == 0]
                            detection = np.mean([row["predicted_duration_s"] > 0 for row in positives])
                            rejection = np.mean([row["predicted_duration_s"] == 0 for row in negatives])
                            positive_iou = np.mean([row["temporal_iou"] for row in positives])
                            harmonic = 2 * detection * rejection / max(1e-9, detection + rejection)
                            objective = 0.60 * harmonic + 0.40 * positive_iou
                            candidate = (objective, positive_iou, harmonic, detection, rejection, -gate)
                            if best is None or candidate > best[0]:
                                best = (candidate, {**base, "gate_threshold": float(gate),
                                    "oof_objective": float(objective),
                                    "oof_positive_iou": float(positive_iou),
                                    "oof_detection_rate": float(detection),
                                    "oof_negative_rejection_rate": float(rejection)})
        assert best is not None
        policies[criterion] = best[1]
        print(f"policy {criterion}: {json.dumps(best[1])}", flush=True)
    return policies


def evaluate(
    videos: list[dict[str, Any]], scores: list[torch.Tensor], policies: dict[str, Any],
    dataset_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    annotation_path = dataset_root / "annotations" / "cholec80-CVS.xlsx"
    for value, prediction in zip(videos, scores):
        video_id = int(value["video_id"])
        labels = load_cvs_intervals(annotation_path, video_id)
        for index, criterion in enumerate(CRITERIA):
            policy = policies[criterion]
            predicted = aggregate_intervals(
                value, prediction, index, policy, policy["gate_threshold"],
            )
            truth = truth_intervals(
                labels, criterion, float(value["window"]["start_s"]),
                float(value["window"]["end_s"]),
            )
            rows.append({
                "video_id": video_id, "criterion": criterion,
                "threshold": policy["on_threshold"], "gate_threshold": policy["gate_threshold"],
                **temporal_metrics(predicted, truth),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    split = json.loads(Path(args.split_json).read_text())
    train_ids = list(map(int, split["train_video_ids"]))
    validation_ids = list(map(int, split["validation_video_ids"]))
    test_ids = set(map(int, split["test_video_ids"]))
    if (set(train_ids) | set(validation_ids)) & test_ids or set(train_ids) & set(validation_ids):
        raise ValueError("Frozen split roles overlap")
    train_videos = [
        torch.load(Path(args.train_cache) / f"video{video_id:02d}.pt", map_location="cpu", weights_only=False)
        for video_id in train_ids
    ]
    validation_videos = [
        torch.load(Path(args.validation_cache) / f"video{video_id:02d}.pt", map_location="cpu", weights_only=False)
        for video_id in validation_ids
    ]
    if any(float(value["cadence_s"]) != 2.0 for value in train_videos + validation_videos):
        raise ValueError("M2d requires cadence-matched 2 s caches")
    train_x = [augment_video(value) for value in train_videos]
    folds = grouped_folds(train_videos, 5, args.seed)
    oof_scores: list[torch.Tensor | None] = [None] * len(train_videos)
    selected_epochs = []
    all_indices = set(range(len(train_videos)))
    for fold_index, heldout in enumerate(folds):
        fit = sorted(all_indices - set(heldout))
        x_fit = torch.cat([train_x[index] for index in fit])
        y_fit = torch.cat([train_videos[index]["targets"] for index in fit])
        x_hold = torch.cat([train_x[index] for index in heldout])
        y_hold = torch.cat([train_videos[index]["targets"] for index in heldout])
        model, mean, std, epoch, score = fit_model(
            x_fit, y_fit, x_hold, y_hold, args.device, args.seed + fold_index,
        )
        selected_epochs.append(epoch)
        cursor = 0
        predictions = predict(model, x_hold, mean, std)
        for index in heldout:
            length = len(train_x[index])
            oof_scores[index] = predictions[cursor:cursor + length]
            cursor += length
        print(f"fold={fold_index} heldout={len(heldout)} epoch={epoch} auprc={score:.6f}", flush=True)
    assert all(item is not None for item in oof_scores)
    typed_oof = [item for item in oof_scores if item is not None]
    dataset_root = Path(split["dataset_root"])
    policies = tune_policy(train_videos, typed_oof, dataset_root)

    final_epochs = max(8, int(round(float(np.median(selected_epochs)))))
    full_x, full_y = torch.cat(train_x), torch.cat([value["targets"] for value in train_videos])
    # The validation tensors passed here are used only to satisfy the fixed-epoch
    # API and report a diagnostic; no value can affect training or selection.
    model, mean, std, _, _ = fit_model(
        full_x, full_y, full_x[:512], full_y[:512], args.device,
        args.seed + 100, fixed_epochs=final_epochs,
    )
    validation_scores = [
        predict(model, augment_video(value), mean, std) for value in validation_videos
    ]
    rows = evaluate(validation_videos, validation_scores, policies, dataset_root)
    summary = method_summary(rows, args.seed, 5000)
    baseline = json.loads(Path(args.baseline_results).read_text())
    baseline_rows = baseline["method_rows"]["M2c_5s_short_event"]
    comparison = paired_bootstrap_comparison(
        baseline_rows, rows, args.seed, 5000, "M2c_5s_short_event", "M2d_balanced_2s",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "m2d_balanced_2s_seed29.pt"
    torch.save({
        "model_state": model.state_dict(),
        "metadata": {
            "model_type": "cadence_matched_context_fusion_calibrator",
            "cadence_s": 2.0, "feature_names": augmented_names(train_videos[0]),
            "feature_mean": mean, "feature_std": std, "policies": policies,
            "grouped_oof_folds": [[train_ids[index] for index in fold] for fold in folds],
            "selected_epochs": selected_epochs, "final_epochs": final_epochs,
            "train_video_ids": train_ids, "validation_video_ids": validation_ids,
            "test_video_ids_held_out": sorted(test_ids), "test_labels_accessed": False,
        },
    }, checkpoint)
    result = {
        "experiment": "cholec80_m2d_balanced_cadence_matched_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model_selection": "5-fold grouped train-only OOF macro AUPRC",
            "policy_selection": "train-only OOF balanced positive-IoU/detection/negative-rejection",
            "validation_used_once_after_freeze": True,
            "test_video_ids_loaded": [], "test_labels_accessed": False,
        },
        "checkpoint": str(checkpoint.resolve()), "policies": policies,
        "method_summaries": {"M2d_balanced_2s": summary},
        "method_rows": {"M2d_balanced_2s": rows},
        "comparison_vs_M2c_5s_short_event": comparison,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(result_path.resolve(), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
