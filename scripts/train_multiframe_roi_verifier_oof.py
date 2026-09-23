#!/usr/bin/env python3
"""Train and evaluate a shared multi-frame ROI verifier with grouped OOF folds."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.visual_adapters import MultiFrameCriterionVerifier


CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")


def balanced_video_folds(
    records: list[dict[str, Any]], fold_count: int, seed: int,
) -> list[list[int]]:
    """Greedily balance rare full/partial samples while keeping videos intact."""
    videos = sorted({int(record["video_id"]) for record in records})
    signatures = {}
    for video_id in videos:
        current = [record for record in records if int(record["video_id"]) == video_id]
        signatures[video_id] = np.asarray([
            sum(record["criterion"] == criterion and int(record["expert_state"]) == state
                for record in current)
            for criterion in CRITERIA for state in (2, 1)
        ], dtype=np.float64)
    total = np.sum(list(signatures.values()), axis=0)
    target = np.maximum(1.0, total / fold_count)
    rng = random.Random(seed)
    tie = {video_id: rng.random() for video_id in videos}
    videos.sort(key=lambda video_id: (
        -float(np.max(signatures[video_id] / target)),
        -float(np.sum(signatures[video_id] / target)), tie[video_id],
    ))
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    counts = np.zeros((fold_count, len(target)), dtype=np.float64)
    capacity = int(np.ceil(len(videos) / fold_count))
    for video_id in videos:
        candidates = [index for index in range(fold_count) if len(folds[index]) < capacity]
        selected = min(candidates, key=lambda index: (
            float(np.sum(((counts[index] + signatures[video_id]) / target) ** 2)),
            len(folds[index]), index,
        ))
        folds[selected].append(video_id)
        counts[selected] += signatures[video_id]
    # Repair greedy end effects (for example, the final nine-video fold receiving
    # only one rare-positive sample) through deterministic pairwise swaps.
    def imbalance() -> float:
        normalized = (counts - target[None]) / target[None]
        return float(np.sum(normalized ** 2))

    for _ in range(100):
        current_loss = imbalance()
        best: tuple[float, int, int, int, int] | None = None
        for left_fold in range(fold_count):
            for right_fold in range(left_fold + 1, fold_count):
                for left_video in folds[left_fold]:
                    for right_video in folds[right_fold]:
                        left_new = counts[left_fold] - signatures[left_video] + signatures[right_video]
                        right_new = counts[right_fold] - signatures[right_video] + signatures[left_video]
                        loss = current_loss
                        loss -= float(np.sum(((counts[left_fold] - target) / target) ** 2))
                        loss -= float(np.sum(((counts[right_fold] - target) / target) ** 2))
                        loss += float(np.sum(((left_new - target) / target) ** 2))
                        loss += float(np.sum(((right_new - target) / target) ** 2))
                        if loss + 1e-12 < current_loss and (best is None or loss < best[0]):
                            best = (loss, left_fold, right_fold, left_video, right_video)
        if best is None:
            break
        _, left_fold, right_fold, left_video, right_video = best
        folds[left_fold].remove(left_video)
        folds[right_fold].remove(right_video)
        folds[left_fold].append(right_video)
        folds[right_fold].append(left_video)
        counts[left_fold] += signatures[right_video] - signatures[left_video]
        counts[right_fold] += signatures[left_video] - signatures[right_video]
    return [sorted(fold) for fold in folds]


def binary_report(labels: list[int], scores: list[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    auc = None
    if positives and negatives:
        order = np.argsort(s, kind="stable")
        ranks = np.empty(len(s), dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1)
        # Average tied ranks.
        for value in np.unique(s):
            tied = np.flatnonzero(s == value)
            ranks[tied] = ranks[tied].mean()
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    order = np.argsort(-s, kind="stable")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y)
    average_precision = float(np.sum(
        cumulative[sorted_y == 1] / (np.flatnonzero(sorted_y == 1) + 1)
    ) / positives) if positives else None
    predicted = s >= 0.5
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum((~predicted) & (y == 1)))
    tn = int(np.sum((~predicted) & (y == 0)))
    return {
        "n": len(y), "positives": positives, "negatives": negatives,
        "roc_auc": auc, "average_precision": average_precision,
        "precision_at_0_5": tp / (tp + fp) if tp + fp else 0.0,
        "recall_at_0_5": tp / positives if positives else 0.0,
        "specificity_at_0_5": tn / negatives if negatives else 0.0,
        "false_positive_count_at_0_5": fp, "false_negative_count_at_0_5": fn,
    }


def reports(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    output = {}
    for criterion in (*CRITERIA, "overall"):
        selected = [row for row in rows if criterion == "overall" or row["criterion"] == criterion]
        output[criterion] = binary_report(
            [int(row["expert_state"] == 2) for row in selected],
            [float(row[score_key]) for row in selected],
        )
    hard = [row for row in rows if row["sample_category"] == "hard_negative"]
    output["hard_negative"] = {
        "n": len(hard),
        "mean_full_probability": float(np.mean([row[score_key] for row in hard])) if hard else None,
        "false_positive_rate_at_0_5": float(np.mean([
            row[score_key] >= 0.5 for row in hard
        ])) if hard else None,
        "p95_full_probability": float(np.quantile([
            row[score_key] for row in hard
        ], 0.95)) if hard else None,
    }
    return output


def high_precision_operating_point(rows: list[dict[str, Any]], criterion: str) -> dict[str, Any]:
    selected = [row for row in rows if row["criterion"] == criterion]
    labels = np.asarray([int(row["expert_state"] == 2) for row in selected])
    scores = np.asarray([float(row["verifier_full_probability"]) for row in selected])
    candidates = []
    for threshold in sorted(set(scores.tolist()), reverse=True):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / int(labels.sum()) if labels.sum() else 0.0
        if precision >= 0.70:
            candidates.append((recall, precision, threshold, tp, fp))
    if not candidates:
        return {"available": False}
    recall, precision, threshold, tp, fp = max(candidates)
    return {
        "available": True, "target_precision": 0.70,
        "threshold": threshold, "precision": precision, "recall": recall,
        "true_positives": tp, "false_positives": fp,
    }


def gather_batch(
    indices: list[int], feature_indices: torch.Tensor, features: torch.Tensor,
    criterion_tensor: torch.Tensor, targets: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.long)
    panels = features[feature_indices[index]].to(device, dtype=torch.float32, non_blocking=True)
    criteria = criterion_tensor[index].to(device, non_blocking=True)
    labels = targets[index].to(device, non_blocking=True)
    return panels, criteria, labels


@torch.inference_mode()
def evaluate(
    model: MultiFrameCriterionVerifier, indices: list[int], records: list[dict[str, Any]],
    feature_indices: torch.Tensor, features: torch.Tensor, criterion_tensor: torch.Tensor,
    targets: torch.Tensor, device: torch.device, batch_size: int,
    verifier_fold_by_video: dict[int, str],
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for start in range(0, len(indices), batch_size):
        current = indices[start:start + batch_size]
        panels, criteria, _ = gather_batch(
            current, feature_indices, features, criterion_tensor, targets, device,
        )
        probabilities = torch.softmax(model(panels, criteria), dim=-1).cpu().tolist()
        for index, probability in zip(current, probabilities):
            record = records[index]
            rows.append({
                "reference_id": record["reference_id"],
                "video_id": int(record["video_id"]), "criterion": record["criterion"],
                "center_time_s": float(record["center_time_s"]),
                "expert_state": int(record["expert_state"]),
                "sample_category": record["sample_category"],
                "source_oof_fold": record["source_oof_fold"],
                "verifier_oof_fold": verifier_fold_by_video[int(record["video_id"])],
                "baseline_visual_full_probability": float(record["visual_full_probability"]),
                "m2c_probability": float(record["m2c_probability"]),
                "verifier_state_probabilities": probability,
                "verifier_full_probability": float(probability[2]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    if cache["manifest_payload_sha256"] != manifest["payload_sha256"]:
        raise ValueError("Feature cache and manifest do not match")
    if cache.get("test_video_or_annotation_accessed") is not False:
        raise ValueError("Feature cache is not test-sealed")
    records = list(manifest["records"])
    key_to_index = {
        (int(key[0]), float(key[1]), str(key[2])): index
        for index, key in enumerate(cache["keys"])
    }
    feature_indices = torch.tensor([[
        key_to_index[(int(record["video_id"]), float(time_s), record["criterion"])]
        for time_s in record["frame_timestamps_s"]
    ] for record in records], dtype=torch.long)
    features = cache["features"]
    criterion_tensor = torch.stack([
        cache["criterion_embeddings"][record["criterion"]].float() for record in records
    ])
    targets = torch.tensor([int(record["expert_state"]) for record in records], dtype=torch.long)
    fold_video_ids = balanced_video_folds(records, 5, args.seed)
    folds = [f"fold{index}" for index in range(len(fold_video_ids))]
    verifier_fold_by_video = {
        video_id: fold for fold, video_ids in zip(folds, fold_video_ids)
        for video_id in video_ids
    }
    print(json.dumps({
        fold: {
            "video_ids": video_ids,
            "full_counts": {
                criterion: sum(
                    int(record["expert_state"] == 2)
                    for record in records
                    if int(record["video_id"]) in video_ids and record["criterion"] == criterion
                ) for criterion in CRITERIA
            },
        } for fold, video_ids in zip(folds, fold_video_ids)
    }, indent=2), flush=True)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_rows, fold_results = [], []
    for fold_index, fold in enumerate(folds):
        torch.manual_seed(args.seed + fold_index)
        np.random.seed(args.seed + fold_index)
        random.seed(args.seed + fold_index)
        heldout_videos = fold_video_ids[fold_index]
        train_indices = [
            index for index, record in enumerate(records)
            if int(record["video_id"]) not in heldout_videos
        ]
        valid_indices = [
            index for index, record in enumerate(records)
            if int(record["video_id"]) in heldout_videos
        ]
        model = MultiFrameCriterionVerifier(
            hidden_dim=args.hidden_dim, max_frames=int(manifest["frame_count"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=2e-3,
        )
        group_counts = Counter(
            (records[index]["criterion"], int(records[index]["expert_state"]))
            for index in train_indices
        )
        weights = torch.tensor([
            (1.8 if records[index]["sample_category"] == "hard_negative" else 1.0)
            / group_counts[(records[index]["criterion"], int(records[index]["expert_state"]))]
            for index in train_indices
        ], dtype=torch.float64)
        best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            generator = torch.Generator().manual_seed(args.seed * 1000 + fold_index * 100 + epoch)
            sampled = torch.multinomial(
                weights, len(train_indices), replacement=True, generator=generator,
            ).tolist()
            order = [train_indices[index] for index in sampled]
            losses = []
            for start in range(0, len(order), args.batch_size):
                current = order[start:start + args.batch_size]
                panels, criteria, labels = gather_batch(
                    current, feature_indices, features, criterion_tensor, targets, device,
                )
                logits = model(panels, criteria).float()
                state_loss = nn.functional.cross_entropy(logits, labels)
                full_logit = logits[:, 2] - torch.logsumexp(logits[:, :2], dim=1)
                binary_weights = torch.tensor([
                    3.0 if records[index]["sample_category"] == "hard_negative"
                    else 1.8 if int(records[index]["expert_state"]) == 2
                    else 1.4 if int(records[index]["expert_state"]) == 1 else 1.0
                    for index in current
                ], device=device)
                full_loss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        full_logit, (labels == 2).float(), reduction="none",
                    ) * binary_weights
                ).mean()
                rank_terms = []
                for criterion in CRITERIA:
                    criterion_mask = torch.tensor([
                        records[index]["criterion"] == criterion for index in current
                    ], device=device)
                    positive = full_logit[criterion_mask & (labels == 2)]
                    hard_mask = torch.tensor([
                        records[index]["sample_category"] == "hard_negative" for index in current
                    ], device=device)
                    negative = full_logit[criterion_mask & hard_mask]
                    if len(positive) and len(negative):
                        rank_terms.append(torch.relu(
                            1.0 - positive[:, None] + negative[None, :]
                        ).mean())
                rank_loss = torch.stack(rank_terms).mean() if rank_terms else logits.sum() * 0.0
                expected = (torch.softmax(logits, dim=1) * torch.arange(
                    3, device=device, dtype=torch.float32,
                )).sum(1)
                ordinal_loss = nn.functional.smooth_l1_loss(expected, labels.float())
                loss = state_loss + 0.70 * full_loss + 0.30 * rank_loss + 0.10 * ordinal_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            valid_rows = evaluate(
                model, valid_indices, records, feature_indices, features,
                criterion_tensor, targets, device, args.batch_size, verifier_fold_by_video,
            )
            current_reports = reports(valid_rows, "verifier_full_probability")
            valid_values = [
                value for criterion in CRITERIA
                for value in (
                    current_reports[criterion]["roc_auc"],
                    current_reports[criterion]["average_precision"],
                ) if value is not None
            ]
            score = float(np.mean(valid_values)) if valid_values else 0.0
            history.append({
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                "selection_score_mean_auc_ap": score,
                "hard_negative_fpr_at_0_5": current_reports["hard_negative"]["false_positive_rate_at_0_5"],
            })
            print(json.dumps({"fold": fold, **history[-1]}), flush=True)
            if score > best_score + 1e-5:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= args.patience:
                break
        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {fold}")
        model.load_state_dict(best_state)
        fold_rows = evaluate(
            model, valid_indices, records, feature_indices, features,
            criterion_tensor, targets, device, args.batch_size, verifier_fold_by_video,
        )
        oof_rows.extend(fold_rows)
        checkpoint = args.output_dir / f"{fold}_best.pt"
        torch.save({
            "schema_version": "multiframe_roi_criterion_verifier_v1",
            "model_state": best_state, "hidden_dim": args.hidden_dim,
            "foundation_model": cache["foundation_model"],
            "feature_layout": cache["feature_layout"],
            "criterion_interface": "free_text_shared_absent_partial_full_verifier",
            "heldout_video_ids": heldout_videos, "best_epoch": best_epoch,
            "test_accessed": False,
        }, checkpoint)
        fold_results.append({
            "fold": fold, "heldout_video_ids": heldout_videos,
            "train_records": len(train_indices), "valid_records": len(valid_indices),
            "best_epoch": best_epoch, "best_selection_score": best_score,
            "reports": reports(fold_rows, "verifier_full_probability"),
            "history": history, "checkpoint": str(checkpoint.resolve()),
        })

    oof_rows.sort(key=lambda row: row["reference_id"])
    verifier_reports = reports(oof_rows, "verifier_full_probability")
    baseline_reports = reports(oof_rows, "baseline_visual_full_probability")
    operating_points = {
        criterion: high_precision_operating_point(oof_rows, criterion)
        for criterion in CRITERIA
    }
    result = {
        "schema_version": "multiframe_roi_verifier_grouped_oof_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "scope": "training_split_only_grouped_oof",
            "folds": folds, "fold_video_ids": fold_video_ids,
            "fold_stratification": "video-grouped greedy balance of full/partial sample counts",
            "test_video_ids_loaded": [], "test_labels_accessed": False,
            "features_frozen_before_verifier_training": True,
            "hard_negative_source": "prior grouped-OOF visual and M2c predictions",
        },
        "manifest": str(args.manifest.resolve()), "features": str(args.features.resolve()),
        "record_count": len(records), "oof_prediction_count": len(oof_rows),
        "baseline_single_frame_reports": baseline_reports,
        "multiframe_roi_verifier_reports": verifier_reports,
        "oof_high_precision_operating_points_diagnostic": operating_points,
        "fold_results": fold_results, "oof_predictions": oof_rows,
    }
    output = args.output_dir / "oof_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "baseline": baseline_reports, "verifier": verifier_reports,
        "operating_points": operating_points,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
