#!/usr/bin/env python3
"""Train the text-conditioned multi-frame verifier on dense train-only labels, video OOF."""
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
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.visual_adapters import MultiFrameCriterionVerifier
from train_multiframe_roi_verifier_oof import CRITERIA, binary_report


def expert_state_at(intervals: list[tuple[float, float, int]], time_s: float) -> int:
    return max((int(state) for start, end, state in intervals if start <= time_s <= end), default=0)


def gather(
    indices: list[int], feature_indices: torch.Tensor, features: torch.Tensor,
    records: list[dict[str, Any]], criterion_embeddings: dict[str, torch.Tensor],
    labels: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.long)
    panels = features[feature_indices[index]].to(device, dtype=torch.float32, non_blocking=True)
    criteria = torch.stack([
        criterion_embeddings[records[item]["criterion"]] for item in indices
    ]).to(device, non_blocking=True)
    return panels, criteria, labels[index].to(device, non_blocking=True)


@torch.inference_mode()
def score_indices(
    model: MultiFrameCriterionVerifier, indices: list[int], feature_indices: torch.Tensor,
    features: torch.Tensor, records: list[dict[str, Any]],
    criterion_embeddings: dict[str, torch.Tensor], labels: torch.Tensor,
    device: torch.device, batch_size: int,
) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    model.eval()
    probabilities = torch.empty(len(indices), 3)
    for start in range(0, len(indices), batch_size):
        current = indices[start:start + batch_size]
        panels, criteria, _ = gather(
            current, feature_indices, features, records, criterion_embeddings, labels, device,
        )
        probabilities[start:start + len(current)] = torch.softmax(
            model(panels, criteria), dim=-1,
        ).cpu()
    reports = {}
    for criterion in CRITERIA:
        selected = [position for position, index in enumerate(indices)
                    if records[index]["criterion"] == criterion]
        reports[criterion] = binary_report(
            [int(labels[indices[position]]) == 2 for position in selected],
            [float(probabilities[position, 2]) for position in selected],
        )
    return probabilities, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fold-definition-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epoch-samples", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=107)
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    train_ids = set(map(int, split["train_video_ids"]))
    test_ids = set(map(int, split["test_video_ids"]))
    if train_ids & test_ids:
        raise ValueError("Training/test overlap")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("scope") != "training_prediction_only":
        raise ValueError("Expected the annotation-free dense training manifest")
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    if cache["manifest_payload_sha256"] != manifest["payload_sha256"]:
        raise ValueError("Feature cache/manifest mismatch")
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
    criterion_embeddings = {
        name: value.float().cpu() for name, value in cache["criterion_embeddings"].items()
    }

    annotation_path = Path(split["dataset_root"]) / "annotations" / "cholec80-CVS.xlsx"
    annotations = {
        video_id: load_cvs_intervals(annotation_path, video_id) for video_id in sorted(train_ids)
    }
    labeled_ids = {
        video_id for video_id, current in annotations.items()
        if any(current[criterion] for criterion in CRITERIA)
    }
    if labeled_ids != train_ids - {1}:
        raise ValueError(f"Unexpected labeled training videos: {sorted(labeled_ids)}")
    labels = torch.tensor([
        expert_state_at(
            annotations[int(record["video_id"])][record["criterion"]],
            float(record["center_time_s"]),
        ) for record in records
    ], dtype=torch.long)

    checkpoints = sorted(args.fold_definition_dir.glob("fold*_best.pt"))
    if len(checkpoints) != 5:
        raise ValueError("Expected five fold-definition checkpoints")
    fold_video_ids = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        fold_video_ids.append(list(map(int, checkpoint["heldout_video_ids"])))
    flattened = [video_id for fold in fold_video_ids for video_id in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != labeled_ids:
        raise ValueError("Fold definition does not partition labeled training videos")

    print(json.dumps({
        "dense_records": len(records),
        "labeled_training_videos": len(labeled_ids),
        "state_counts": dict(Counter(map(int, labels.tolist()))),
        "fold_video_ids": fold_video_ids,
        "test_accessed": False,
    }, indent=2), flush=True)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_probabilities = torch.zeros(len(records), 3)
    fold_by_video: dict[int, str] = {}
    fold_results = []
    for fold_index, heldout in enumerate(fold_video_ids):
        fold_name = f"fold{fold_index}"
        for video_id in heldout:
            fold_by_video[video_id] = fold_name
        train_indices = [
            index for index, record in enumerate(records)
            if int(record["video_id"]) in labeled_ids
            and int(record["video_id"]) not in heldout
        ]
        valid_indices = [
            index for index, record in enumerate(records)
            if int(record["video_id"]) in heldout
        ]
        torch.manual_seed(args.seed + fold_index)
        np.random.seed(args.seed + fold_index)
        random.seed(args.seed + fold_index)
        model = MultiFrameCriterionVerifier(
            hidden_dim=args.hidden_dim, max_frames=int(manifest["frame_count"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=2e-3,
        )
        group_counts = Counter(
            (records[index]["criterion"], int(labels[index])) for index in train_indices
        )
        weights = torch.tensor([
            (
                2.5 if int(labels[index]) < 2
                and float(records[index]["m2c_probability"]) >= float(records[index]["m2c_threshold"])
                else 1.0
            ) / group_counts[(records[index]["criterion"], int(labels[index]))]
            for index in train_indices
        ], dtype=torch.float64)
        best_state, best_epoch, best_score, stale = None, 0, -1.0, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            generator = torch.Generator().manual_seed(
                args.seed * 10000 + fold_index * 100 + epoch
            )
            sampled_positions = torch.multinomial(
                weights, args.epoch_samples, replacement=True, generator=generator,
            ).tolist()
            order = [train_indices[position] for position in sampled_positions]
            losses = []
            for start in range(0, len(order), args.batch_size):
                current = order[start:start + args.batch_size]
                panels, criteria, targets = gather(
                    current, feature_indices, features, records, criterion_embeddings,
                    labels, device,
                )
                logits = model(panels, criteria).float()
                state_loss = nn.functional.cross_entropy(logits, targets)
                full_logit = logits[:, 2] - torch.logsumexp(logits[:, :2], dim=1)
                binary_weight = torch.tensor([
                    2.5 if int(labels[index]) < 2
                    and float(records[index]["m2c_probability"])
                    >= float(records[index]["m2c_threshold"])
                    else 1.8 if int(labels[index]) == 2 else 1.0
                    for index in current
                ], device=device)
                binary_loss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        full_logit, (targets == 2).float(), reduction="none",
                    ) * binary_weight
                ).mean()
                rank_terms = []
                for criterion in CRITERIA:
                    criterion_mask = torch.tensor([
                        records[index]["criterion"] == criterion for index in current
                    ], device=device)
                    positive = full_logit[criterion_mask & (targets == 2)]
                    hard = full_logit[criterion_mask & (targets < 2)]
                    if len(positive) and len(hard):
                        rank_terms.append(torch.relu(
                            1.0 - positive[:, None] + hard[None, :]
                        ).mean())
                rank_loss = torch.stack(rank_terms).mean() if rank_terms else logits.sum() * 0
                loss = state_loss + 1.1 * binary_loss + 0.30 * rank_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach()))

            _, reports = score_indices(
                model, valid_indices, feature_indices, features, records,
                criterion_embeddings, labels, device, args.batch_size,
            )
            values = [
                value for criterion in CRITERIA
                for value in (reports[criterion]["roc_auc"], reports[criterion]["average_precision"])
                if value is not None
            ]
            selection_score = float(np.mean(values))
            current = {
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                "selection_score_mean_auc_ap": selection_score,
                "by_criterion": {
                    criterion: {
                        "auc": reports[criterion]["roc_auc"],
                        "ap": reports[criterion]["average_precision"],
                    } for criterion in CRITERIA
                },
            }
            history.append(current)
            print(json.dumps({"fold": fold_name, **current}), flush=True)
            if selection_score > best_score + 1e-5:
                best_score, best_epoch, stale = selection_score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= args.patience:
                break
        if best_state is None:
            raise RuntimeError(f"No model selected for {fold_name}")
        model.load_state_dict(best_state)
        probabilities, reports = score_indices(
            model, valid_indices, feature_indices, features, records,
            criterion_embeddings, labels, device, args.batch_size,
        )
        all_probabilities[torch.tensor(valid_indices)] = probabilities
        checkpoint_path = args.output_dir / f"{fold_name}_best.pt"
        torch.save({
            "schema_version": "dense_multiframe_roi_criterion_verifier_v1",
            "model_state": best_state, "hidden_dim": args.hidden_dim,
            "foundation_model": cache["foundation_model"],
            "feature_layout": cache["feature_layout"],
            "criterion_interface": "free_text_shared_absent_partial_full_verifier",
            "training_distribution": "dense_5s_train_only",
            "heldout_video_ids": heldout, "best_epoch": best_epoch,
            "seed": args.seed, "test_accessed": False,
        }, checkpoint_path)
        fold_results.append({
            "fold": fold_name, "heldout_video_ids": heldout,
            "best_epoch": best_epoch, "best_selection_score": best_score,
            "reports": reports, "history": history,
            "checkpoint": str(checkpoint_path.resolve()),
        })

    # Video01 has no usable expert annotations and was never used by any fold model.
    video01_indices = [
        index for index, record in enumerate(records) if int(record["video_id"]) == 1
    ]
    checkpoint = torch.load(
        args.output_dir / "fold0_best.pt", map_location="cpu", weights_only=False,
    )
    model = MultiFrameCriterionVerifier(
        hidden_dim=int(checkpoint["hidden_dim"]), max_frames=int(manifest["frame_count"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    probabilities, _ = score_indices(
        model, video01_indices, feature_indices, features, records,
        criterion_embeddings, labels, device, args.batch_size,
    )
    all_probabilities[torch.tensor(video01_indices)] = probabilities
    fold_by_video[1] = "fold0"

    output_rows = []
    for record, state, probability in zip(records, labels.tolist(), all_probabilities.tolist()):
        video_id = int(record["video_id"])
        output_rows.append({
            "reference_id": record["reference_id"], "video_id": video_id,
            "criterion": record["criterion"],
            "center_time_s": float(record["center_time_s"]),
            "expert_state_train_only": int(state) if video_id in labeled_ids else None,
            "baseline_visual_full_probability": float(
                record["baseline_visual_full_probability"]
            ),
            "m2c_probability": float(record["m2c_probability"]),
            "m2c_threshold": float(record["m2c_threshold"]),
            "verifier_oof_fold": fold_by_video[video_id],
            "verifier_never_saw_video": True,
            "verifier_state_probabilities": probability,
            "verifier_full_probability": float(probability[2]),
        })
    output = {
        "schema_version": "dense_trained_multiframe_roi_verifier_oof_predictions_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": str(args.split_json.resolve()),
        "training_scope": "dense_5s_train_only_grouped_video_oof",
        "fold_results": fold_results,
        "annotation_files_loaded_during_prediction": False,
        "training_annotations_loaded_for_training_only": True,
        "test_video_or_annotation_accessed": False,
        "rows": output_rows,
    }
    output_path = args.output_dir / "predictions_oof.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path.resolve()), "rows": len(output_rows),
        "strict_video_oof": True, "test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
