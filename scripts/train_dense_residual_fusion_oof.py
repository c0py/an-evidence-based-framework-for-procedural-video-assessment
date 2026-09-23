#!/usr/bin/env python3
"""Train dense text-conditioned visual residual corrections over M2c, video OOF."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.visual_adapters import MultiFrameCriterionResidualFusion
from train_multiframe_roi_verifier_oof import CRITERIA, binary_report


def state_at(intervals, time_s: float) -> int:
    return max((int(state) for start, end, state in intervals if start <= time_s <= end), default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fold-definition-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--epoch-samples", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=131)
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    train_ids = set(map(int, split["train_video_ids"]))
    test_ids = set(map(int, split["test_video_ids"]))
    if train_ids & test_ids:
        raise ValueError("Training/test overlap")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    if cache["manifest_payload_sha256"] != manifest["payload_sha256"]:
        raise ValueError("Feature cache/manifest mismatch")
    records = list(manifest["records"])
    key_to_index = {
        (int(key[0]), float(key[1]), str(key[2])): index
        for index, key in enumerate(cache["keys"])
    }
    feature_indices = torch.tensor([[
        key_to_index[(int(record["video_id"]), float(time_s), record["criterion"])]
        for time_s in record["frame_timestamps_s"]
    ] for record in records], dtype=torch.long)
    pair_score = {
        (int(record["video_id"]), record["criterion"], round(float(record["center_time_s"]), 4)):
        float(record["m2c_probability"])
        for record in records
    }
    base_sequences = torch.tensor([[
        pair_score[(int(record["video_id"]), record["criterion"], round(float(time_s), 4))]
        for time_s in record["frame_timestamps_s"]
    ] for record in records], dtype=torch.float32)
    base_centers = torch.tensor([
        float(record["m2c_probability"]) for record in records
    ], dtype=torch.float32)
    thresholds = torch.tensor([
        float(record["m2c_threshold"]) for record in records
    ], dtype=torch.float32)
    criterion_embeddings = {
        name: value.float().cpu() for name, value in cache["criterion_embeddings"].items()
    }

    annotation_path = Path(split["dataset_root"]) / "annotations" / "cholec80-CVS.xlsx"
    annotations = {video_id: load_cvs_intervals(annotation_path, video_id) for video_id in train_ids}
    labeled_ids = {
        video_id for video_id, current in annotations.items()
        if any(current[criterion] for criterion in CRITERIA)
    }
    if labeled_ids != train_ids - {1}:
        raise ValueError("Unexpected training annotation coverage")
    states = torch.tensor([
        state_at(
            annotations[int(record["video_id"])][record["criterion"]],
            float(record["center_time_s"]),
        ) for record in records
    ], dtype=torch.long)
    targets = (states == 2).float()

    definition_paths = sorted(args.fold_definition_dir.glob("fold*_best.pt"))
    fold_video_ids = [
        list(map(int, torch.load(path, map_location="cpu", weights_only=False)["heldout_video_ids"]))
        for path in definition_paths
    ]
    if len(fold_video_ids) != 5 or {v for fold in fold_video_ids for v in fold} != labeled_ids:
        raise ValueError("Invalid fold definition")

    features = cache["features"]
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_probabilities = torch.zeros(len(records))
    all_residuals = torch.zeros(len(records))
    fold_by_video = {}
    fold_results = []

    def gather(indices: list[int]):
        index = torch.tensor(indices, dtype=torch.long)
        return (
            features[feature_indices[index]].to(device, dtype=torch.float32, non_blocking=True),
            torch.stack([criterion_embeddings[records[i]["criterion"]] for i in indices]).to(device),
            base_sequences[index].to(device), base_centers[index].to(device),
            thresholds[index].to(device), targets[index].to(device),
        )

    @torch.inference_mode()
    def evaluate(model, indices: list[int]):
        model.eval()
        probabilities, residuals = torch.empty(len(indices)), torch.empty(len(indices))
        for start in range(0, len(indices), args.batch_size):
            current = indices[start:start + args.batch_size]
            panels, criteria, sequences, centers, current_thresholds, _ = gather(current)
            logits, residual = model(
                panels, criteria, sequences, centers, current_thresholds,
            )
            probabilities[start:start + len(current)] = torch.sigmoid(logits).cpu()
            residuals[start:start + len(current)] = residual.cpu()
        reports = {}
        for criterion in CRITERIA:
            positions = [p for p, i in enumerate(indices) if records[i]["criterion"] == criterion]
            reports[criterion] = binary_report(
                [bool(targets[indices[p]]) for p in positions],
                [float(probabilities[p]) for p in positions],
            )
        return probabilities, residuals, reports

    for fold_index, heldout in enumerate(fold_video_ids):
        fold_name = f"fold{fold_index}"
        for video_id in heldout:
            fold_by_video[video_id] = fold_name
        train_indices = [
            i for i, record in enumerate(records)
            if int(record["video_id"]) in labeled_ids and int(record["video_id"]) not in heldout
        ]
        valid_indices = [i for i, record in enumerate(records) if int(record["video_id"]) in heldout]
        torch.manual_seed(args.seed + fold_index)
        np.random.seed(args.seed + fold_index)
        random.seed(args.seed + fold_index)
        model = MultiFrameCriterionResidualFusion(
            hidden_dim=args.hidden_dim, max_frames=int(manifest["frame_count"]),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-3)
        group_counts = Counter(
            (records[i]["criterion"], bool(targets[i])) for i in train_indices
        )
        sample_weights = torch.tensor([
            (2.0 if not bool(targets[i]) and base_centers[i] >= thresholds[i] else 1.0)
            / group_counts[(records[i]["criterion"], bool(targets[i]))]
            for i in train_indices
        ], dtype=torch.float64)

        baseline_reports = {}
        for criterion in CRITERIA:
            current = [i for i in valid_indices if records[i]["criterion"] == criterion]
            baseline_reports[criterion] = binary_report(
                [bool(targets[i]) for i in current], [float(base_centers[i]) for i in current],
            )
        def selection_score(reports):
            return float(np.mean([
                value for criterion in CRITERIA
                for value in (reports[criterion]["roc_auc"], reports[criterion]["average_precision"])
                if value is not None
            ]))
        best_state = copy.deepcopy(model.state_dict())
        best_score = selection_score(baseline_reports)
        best_epoch, stale, history = 0, 0, []
        print(json.dumps({
            "fold": fold_name, "epoch": 0, "selection_score_mean_auc_ap": best_score,
            "role": "exact_M2c_identity_initialization",
        }), flush=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            generator = torch.Generator().manual_seed(args.seed * 10000 + fold_index * 100 + epoch)
            sampled = torch.multinomial(
                sample_weights, args.epoch_samples, replacement=True, generator=generator,
            ).tolist()
            order = [train_indices[position] for position in sampled]
            losses = []
            for start in range(0, len(order), args.batch_size):
                current = order[start:start + args.batch_size]
                panels, criteria, sequences, centers, current_thresholds, labels = gather(current)
                logits, residual = model(
                    panels, criteria, sequences, centers, current_thresholds,
                )
                weights = torch.tensor([
                    2.0 if not bool(targets[i]) and base_centers[i] >= thresholds[i]
                    else 1.5 if bool(targets[i]) else 1.0 for i in current
                ], device=device)
                binary_loss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, reduction="none",
                    ) * weights
                ).mean()
                rank_terms = []
                for criterion in CRITERIA:
                    mask = torch.tensor([records[i]["criterion"] == criterion for i in current], device=device)
                    positive = logits[mask & (labels > 0.5)]
                    negative = logits[mask & (labels <= 0.5)]
                    if len(positive) and len(negative):
                        rank_terms.append(torch.relu(0.5 - positive[:, None] + negative[None, :]).mean())
                rank_loss = torch.stack(rank_terms).mean() if rank_terms else logits.sum() * 0
                loss = binary_loss + 0.20 * rank_loss + 0.01 * residual.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            _, _, reports = evaluate(model, valid_indices)
            score = selection_score(reports)
            item = {
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                "selection_score_mean_auc_ap": score,
                "by_criterion": {c: {"auc": reports[c]["roc_auc"], "ap": reports[c]["average_precision"]} for c in CRITERIA},
            }
            history.append(item)
            print(json.dumps({"fold": fold_name, **item}), flush=True)
            if score > best_score + 1e-5:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= args.patience:
                break

        model.load_state_dict(best_state)
        probabilities, residuals, reports = evaluate(model, valid_indices)
        all_probabilities[torch.tensor(valid_indices)] = probabilities
        all_residuals[torch.tensor(valid_indices)] = residuals
        checkpoint_path = args.output_dir / f"{fold_name}_best.pt"
        torch.save({
            "schema_version": "dense_multiframe_m2c_residual_fusion_v1",
            "model_state": best_state, "hidden_dim": args.hidden_dim,
            "heldout_video_ids": heldout, "best_epoch": best_epoch, "seed": args.seed,
            "foundation_model": cache["foundation_model"], "test_accessed": False,
        }, checkpoint_path)
        fold_results.append({
            "fold": fold_name, "heldout_video_ids": heldout, "best_epoch": best_epoch,
            "baseline_reports": baseline_reports, "corrected_reports": reports,
            "history": history, "checkpoint": str(checkpoint_path.resolve()),
        })

    # Unlabeled video01 was unseen by every model; fold0 provides a clean score.
    video01 = [i for i, record in enumerate(records) if int(record["video_id"]) == 1]
    checkpoint = torch.load(args.output_dir / "fold0_best.pt", map_location="cpu", weights_only=False)
    model = MultiFrameCriterionResidualFusion(
        hidden_dim=int(checkpoint["hidden_dim"]), max_frames=int(manifest["frame_count"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probabilities, residuals, _ = evaluate(model, video01)
    all_probabilities[torch.tensor(video01)] = probabilities
    all_residuals[torch.tensor(video01)] = residuals
    fold_by_video[1] = "fold0"

    rows = []
    for i, record in enumerate(records):
        video_id = int(record["video_id"])
        probability = float(all_probabilities[i])
        rows.append({
            "reference_id": record["reference_id"], "video_id": video_id,
            "criterion": record["criterion"], "center_time_s": float(record["center_time_s"]),
            "expert_state_train_only": int(states[i]) if video_id in labeled_ids else None,
            "baseline_visual_full_probability": float(record["baseline_visual_full_probability"]),
            "m2c_probability": float(base_centers[i]), "m2c_threshold": float(thresholds[i]),
            "verifier_oof_fold": fold_by_video[video_id], "verifier_never_saw_video": True,
            "visual_residual_logit": float(all_residuals[i]),
            "verifier_full_probability": probability,
            "verifier_state_probabilities": [1.0 - probability, 0.0, probability],
        })
    output = {
        "schema_version": "dense_multiframe_m2c_residual_fusion_oof_predictions_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": str(args.split_json.resolve()), "fold_results": fold_results,
        "prediction_role": "training_strict_video_oof_residual_fusion",
        "annotation_files_loaded_during_prediction": False,
        "training_annotations_loaded_for_training_only": True,
        "test_video_or_annotation_accessed": False, "rows": rows,
    }
    output_path = args.output_dir / "predictions_oof.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path.resolve()), "rows": len(rows), "test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
