"""Train a 2 s/5 s TCN with dense CVS and video-presence supervision.

All choices are made with grouped out-of-fold predictions from the training
split. Validation is evaluated once after freezing; test files are not loaded.
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
from cvs_assessment.models import DualScaleTemporalFusionHead
from run_cholec80_validation_ablation import (
    CRITERIA, method_summary, paired_bootstrap_comparison,
    temporal_metrics, truth_intervals,
)
from train_cholec80_m2d_balanced_temporal import grouped_folds
from train_peskavlp_cvs_head import binary_metrics


def load_pair(video_id: int, fine_dir: Path, coarse_dir: Path) -> dict[str, Any]:
    fine = torch.load(fine_dir / f"video{video_id:02d}.pt", map_location="cpu", weights_only=False)
    coarse = torch.load(coarse_dir / f"video{video_id:02d}.pt", map_location="cpu", weights_only=False)
    if float(fine["cadence_s"]) != 2.0 or float(coarse["cadence_s"]) != 5.0:
        raise ValueError("Dual-scale inputs must be 2 s and 5 s")
    if fine["feature_names"] != coarse["feature_names"]:
        raise ValueError("Fine/coarse feature schemas differ")
    phase = fine["feature_names"].index("phase_progress")
    selected = [index for index in range(len(fine["feature_names"])) if index != phase]
    fine_features = fine["features"][:, selected]
    coarse_features = coarse["features"][:, selected]
    fine_times, coarse_times = fine["timestamps_s"], coarse["timestamps_s"]
    right = torch.searchsorted(coarse_times, fine_times).clamp(max=len(coarse_times) - 1)
    left = (right - 1).clamp(min=0)
    choose_left = (fine_times - coarse_times[left]).abs() <= (coarse_times[right] - fine_times).abs()
    nearest = torch.where(choose_left, left, right)
    aligned_coarse = coarse_features[nearest]
    output = dict(fine)
    output["dual_features"] = torch.cat(
        [fine_features, aligned_coarse, fine_features - aligned_coarse], dim=-1,
    ).float()
    output["presence_targets"] = (fine["targets"].sum(0) > 0).float()
    output["dual_feature_names"] = (
        [f"fine_{fine['feature_names'][i]}" for i in selected]
        + [f"coarse_{fine['feature_names'][i]}" for i in selected]
        + [f"fine_minus_coarse_{fine['feature_names'][i]}" for i in selected]
    )
    return output


def macro_auprc(scores: torch.Tensor, truth: torch.Tensor) -> float:
    values = []
    for index in range(len(CRITERIA)):
        metric = binary_metrics(scores[:, index], truth[:, index].bool(), 0.5)["auprc"]
        if metric is not None:
            values.append(float(metric))
    return float(np.mean(values)) if values else 0.0


def dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probability * target).sum(0)
    denominator = (probability + target).sum(0)
    return (1.0 - (2 * intersection + 1.0) / (denominator + 1.0)).mean()


@torch.inference_mode()
def infer(
    model: DualScaleTemporalFusionHead, videos: list[dict[str, Any]],
    mean: torch.Tensor, std: torch.Tensor, device: str,
) -> list[dict[str, torch.Tensor]]:
    model.to(device).eval()
    output = []
    for value in videos:
        features = ((value["dual_features"] - mean) / std).unsqueeze(0).to(device)
        prediction = model(features)
        output.append({
            "frame": torch.sigmoid(prediction["frame_logits"][0]).cpu(),
            "presence": torch.sigmoid(prediction["presence_logits"][0]).cpu(),
        })
    return output


def fit(
    train_videos: list[dict[str, Any]], valid_videos: list[dict[str, Any]],
    device: str, seed: int, fixed_epochs: int | None = None,
) -> tuple[DualScaleTemporalFusionHead, torch.Tensor, torch.Tensor, int, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    all_x = torch.cat([value["dual_features"] for value in train_videos])
    all_y = torch.cat([value["targets"] for value in train_videos])
    video_y = torch.stack([value["presence_targets"] for value in train_videos])
    mean, std = all_x.mean(0), all_x.std(0).clamp_min(1e-6)
    frame_positive = all_y.sum(0)
    frame_weight = ((len(all_y) - frame_positive) / frame_positive.clamp_min(1)).clamp(max=25).to(device)
    video_positive = video_y.sum(0)
    video_weight = ((len(video_y) - video_positive) / video_positive.clamp_min(1)).clamp(max=8).to(device)
    model = DualScaleTemporalFusionHead(
        input_dim=all_x.shape[1], hidden_dim=96, dropout=0.18,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
    limit = fixed_epochs or 70
    for epoch in range(1, limit + 1):
        model.train()
        order = np.random.permutation(len(train_videos))
        total_loss = 0.0
        for video_index in order:
            value = train_videos[int(video_index)]
            features = ((value["dual_features"] - mean) / std).unsqueeze(0).to(device)
            target = value["targets"].unsqueeze(0).to(device)
            presence_target = value["presence_targets"].unsqueeze(0).to(device)
            prediction = model(features)
            frame_loss = nn.functional.binary_cross_entropy_with_logits(
                prediction["frame_logits"], target, pos_weight=frame_weight,
            )
            probabilities = torch.sigmoid(prediction["frame_logits"][0])
            dense_dice = dice_loss(probabilities, target[0])
            presence_loss = nn.functional.binary_cross_entropy_with_logits(
                prediction["presence_logits"], presence_target, pos_weight=video_weight,
            )
            hard_negative_terms = []
            for criterion in range(len(CRITERIA)):
                if not bool(presence_target[0, criterion]):
                    count = min(8, len(probabilities))
                    hard_negative_terms.append(
                        torch.topk(probabilities[:, criterion], count).values.square().mean()
                    )
            hard_negative = (
                torch.stack(hard_negative_terms).mean()
                if hard_negative_terms else probabilities.sum() * 0.0
            )
            smoothness = (probabilities[1:] - probabilities[:-1]).abs().mean()
            loss = frame_loss + 0.35 * dense_dice + 0.80 * presence_loss
            loss = loss + 0.80 * hard_negative + 0.02 * smoothness
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss)
        if fixed_epochs is not None:
            continue
        predictions = infer(model, valid_videos, mean, std, device)
        frame_score = macro_auprc(
            torch.cat([item["frame"] for item in predictions]),
            torch.cat([value["targets"] for value in valid_videos]),
        )
        presence_score = macro_auprc(
            torch.stack([item["presence"] for item in predictions]),
            torch.stack([value["presence_targets"] for value in valid_videos]),
        )
        selection = 0.65 * frame_score + 0.35 * presence_score
        print(
            f"seed={seed} epoch={epoch} loss={total_loss/len(train_videos):.5f} "
            f"frame_auprc={frame_score:.5f} presence_auprc={presence_score:.5f} "
            f"selection={selection:.5f}", flush=True,
        )
        if selection > best_score + 1e-4:
            best_state, best_score, best_epoch, stale = (
                copy.deepcopy(model.state_dict()), selection, epoch, 0,
            )
        else:
            stale += 1
        if stale >= 12:
            break
    if fixed_epochs is not None:
        best_state, best_epoch, best_score = copy.deepcopy(model.state_dict()), fixed_epochs, 0.0
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.cpu(), mean, std, best_epoch, best_score


def intervals_from_scores(
    value: dict[str, Any], scores: torch.Tensor, frame_threshold: float,
    min_duration_s: float, max_gap_s: float,
) -> list[list[float]]:
    times = [float(item) for item in value["timestamps_s"].tolist()]
    active = [times[index] for index in torch.where(scores >= frame_threshold)[0].tolist()]
    if not active:
        return []
    cadence = float(value["cadence_s"])
    groups = [[active[0]]]
    for time_s in active[1:]:
        if time_s - groups[-1][-1] <= cadence + max_gap_s + 1e-6:
            groups[-1].append(time_s)
        else:
            groups.append([time_s])
    start_limit, end_limit = float(value["window"]["start_s"]), float(value["window"]["end_s"])
    return [
        [max(start_limit, group[0] - cadence / 2), min(end_limit, group[-1] + cadence / 2)]
        for group in groups if group[-1] - group[0] + cadence >= min_duration_s
    ]


def row_for(
    value: dict[str, Any], prediction: dict[str, torch.Tensor], criterion_index: int,
    policy: dict[str, float], annotations: dict[str, Any],
) -> dict[str, Any]:
    criterion = CRITERIA[criterion_index]
    predicted = []
    if float(prediction["presence"][criterion_index]) >= policy["presence_threshold"]:
        predicted = intervals_from_scores(
            value, prediction["frame"][:, criterion_index], policy["frame_threshold"],
            policy["min_duration_s"], policy["max_gap_s"],
        )
    truth = truth_intervals(
        annotations, criterion, float(value["window"]["start_s"]),
        float(value["window"]["end_s"]),
    )
    return {
        "video_id": int(value["video_id"]), "criterion": criterion,
        "threshold": policy["frame_threshold"],
        "presence_threshold": policy["presence_threshold"],
        **temporal_metrics(predicted, truth),
    }


def tune_policies(
    videos: list[dict[str, Any]], predictions: list[dict[str, torch.Tensor]],
    dataset_root: Path,
) -> dict[str, dict[str, float]]:
    annotation_map = {
        int(value["video_id"]): load_cvs_intervals(
            dataset_root / "annotations" / "cholec80-CVS.xlsx", int(value["video_id"]),
        ) for value in videos
    }
    output = {}
    for criterion_index, criterion in enumerate(CRITERIA):
        best = None
        for frame_threshold in np.arange(0.20, 0.86, 0.05):
            for presence_threshold in np.arange(0.15, 0.86, 0.05):
                for min_duration_s in (2.0, 4.0, 6.0):
                    for max_gap_s in (0.0, 2.0, 4.0):
                        policy = {
                            "frame_threshold": float(frame_threshold),
                            "presence_threshold": float(presence_threshold),
                            "min_duration_s": min_duration_s, "max_gap_s": max_gap_s,
                        }
                        rows = [
                            row_for(value, prediction, criterion_index, policy,
                                    annotation_map[int(value["video_id"])])
                            for value, prediction in zip(videos, predictions)
                        ]
                        positive = [row for row in rows if row["truth_duration_s"] > 0]
                        negative = [row for row in rows if row["truth_duration_s"] == 0]
                        detection = float(np.mean([row["predicted_duration_s"] > 0 for row in positive]))
                        rejection = float(np.mean([row["predicted_duration_s"] == 0 for row in negative]))
                        positive_iou = float(np.mean([row["temporal_iou"] for row in positive]))
                        harmonic = 2 * detection * rejection / max(1e-9, detection + rejection)
                        objective = 0.55 * harmonic + 0.45 * positive_iou
                        candidate = (objective, min(detection, rejection), positive_iou, detection, rejection)
                        if best is None or candidate > best[0]:
                            best = (candidate, {**policy, "oof_objective": objective,
                                "oof_positive_iou": positive_iou,
                                "oof_detection_rate": detection,
                                "oof_negative_rejection_rate": rejection})
        assert best is not None
        output[criterion] = best[1]
        print(f"policy {criterion}: {json.dumps(best[1])}", flush=True)
    return output


def evaluate(
    videos: list[dict[str, Any]], predictions: list[dict[str, torch.Tensor]],
    policies: dict[str, dict[str, float]], dataset_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for value, prediction in zip(videos, predictions):
        annotations = load_cvs_intervals(
            dataset_root / "annotations" / "cholec80-CVS.xlsx", int(value["video_id"]),
        )
        for criterion_index, criterion in enumerate(CRITERIA):
            rows.append(row_for(
                value, prediction, criterion_index, policies[criterion], annotations,
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--fine-train-cache", required=True)
    parser.add_argument("--fine-validation-cache", required=True)
    parser.add_argument("--coarse-cache", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    split = json.loads(Path(args.split_json).read_text())
    train_ids = list(map(int, split["train_video_ids"]))
    validation_ids = list(map(int, split["validation_video_ids"]))
    test_ids = set(map(int, split["test_video_ids"]))
    if (set(train_ids) | set(validation_ids)) & test_ids or set(train_ids) & set(validation_ids):
        raise ValueError("Frozen split roles overlap")
    train = [load_pair(i, Path(args.fine_train_cache), Path(args.coarse_cache)) for i in train_ids]
    validation = [load_pair(i, Path(args.fine_validation_cache), Path(args.coarse_cache)) for i in validation_ids]
    folds = grouped_folds(train, 5, args.seed)
    oof: list[dict[str, torch.Tensor] | None] = [None] * len(train)
    epochs = []
    all_indices = set(range(len(train)))
    for fold_index, heldout in enumerate(folds):
        fitting = sorted(all_indices - set(heldout))
        model, mean, std, epoch, selection = fit(
            [train[index] for index in fitting], [train[index] for index in heldout],
            args.device, args.seed + fold_index,
        )
        predictions = infer(model, [train[index] for index in heldout], mean, std, args.device)
        for index, prediction in zip(heldout, predictions):
            oof[index] = prediction
        epochs.append(epoch)
        print(f"fold={fold_index} epoch={epoch} selection={selection:.6f}", flush=True)
    assert all(item is not None for item in oof)
    oof_predictions = [item for item in oof if item is not None]
    dataset_root = Path(split["dataset_root"])
    policies = tune_policies(train, oof_predictions, dataset_root)
    final_epochs = max(6, int(round(float(np.median(epochs)))))
    model, mean, std, _, _ = fit(
        train, train[:5], args.device, args.seed + 100, fixed_epochs=final_epochs,
    )
    validation_predictions = infer(model, validation, mean, std, args.device)
    rows = evaluate(validation, validation_predictions, policies, dataset_root)
    summary = method_summary(rows, args.seed, 5000)
    baseline = json.loads(Path(args.baseline_results).read_text())
    comparisons = {}
    for name in ("M2c_5s_short_event", "M2c_2s_short_event"):
        comparisons[f"M2e_minus_{name}"] = paired_bootstrap_comparison(
            baseline["method_rows"][name], rows, args.seed, 5000, name, "M2e_dual_scale_tcn",
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"m2e_dual_scale_tcn_seed{args.seed}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "metadata": {
            "model_type": "dual_scale_temporal_fusion_presence_head",
            "fine_cadence_s": 2.0, "coarse_cadence_s": 5.0,
            "feature_names": train[0]["dual_feature_names"],
            "feature_mean": mean, "feature_std": std, "policies": policies,
            "grouped_oof_folds": [[train_ids[index] for index in fold] for fold in folds],
            "selected_epochs": epochs, "final_epochs": final_epochs,
            "train_video_ids": train_ids, "validation_video_ids": validation_ids,
            "test_video_ids_held_out": sorted(test_ids), "test_labels_accessed": False,
        },
    }, checkpoint)
    result = {
        "experiment": "cholec80_m2e_dual_scale_tcn_presence_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model_and_policy_selection": "5-fold grouped train-only OOF",
            "validation_used_once_after_freeze": True,
            "test_video_ids_loaded": [], "test_labels_accessed": False,
        },
        "checkpoint": str(checkpoint.resolve()), "policies": policies,
        "method_summaries": {"M2e_dual_scale_tcn": summary},
        "method_rows": {"M2e_dual_scale_tcn": rows}, "comparisons": comparisons,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(result_path.resolve(), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
