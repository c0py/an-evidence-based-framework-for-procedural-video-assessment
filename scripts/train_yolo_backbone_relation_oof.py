#!/usr/bin/env python3
"""Train the frozen-YOLO text-conditioned relation verifier with strict video OOF."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.backbone_relation import TextConditionedBackboneRelationVerifier
from cvs_assessment.temporal_proposals import centered_windows
from train_dense_residual_fusion_oof import state_at
from train_multiframe_roi_verifier_oof import binary_report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def criterion_report(
    rows: list[dict[str, Any]], states: np.ndarray, scores: np.ndarray, criterion: str,
) -> dict[str, Any]:
    indices = [
        index for index, row in enumerate(rows)
        if row["criterion"] == criterion and int(states[index]) in {0, 2}
    ]
    labels = [int(states[index] == 2) for index in indices]
    values = [float(scores[index]) for index in indices]
    report = binary_report(labels, values)
    missed = [
        index for index in indices if states[index] == 2
        and float(rows[index]["m2c_probability"]) < float(rows[index]["m2c_threshold"])
    ]
    hard_negative = [
        index for index in indices if states[index] == 0
        and float(rows[index]["m2c_probability"]) >= float(rows[index]["m2c_threshold"])
    ]
    report.update({
        "state2_missed_by_m2c_count": len(missed),
        "visual_score_mean_on_state2_missed_by_m2c": float(np.mean(scores[missed])) if missed else None,
        "visual_recall_at_0_5_on_state2_missed_by_m2c": float(np.mean(scores[missed] >= 0.5)) if missed else None,
        "m2c_hard_negative_count": len(hard_negative),
        "visual_fpr_at_0_5_on_m2c_hard_negatives": float(np.mean(scores[hard_negative] >= 0.5)) if hard_negative else None,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_policy.read_text(encoding="utf-8"))
    if frozen.get("new_oof_labels_loaded_during_freeze") is not False:
        raise ValueError("Policy was not frozen before new OOF labels")
    if frozen.get("development_or_test_accessed") is not False:
        raise ValueError("Frozen policy reports development/test access")
    sequence_path = Path(frozen["sequences"])
    if sha256_file(sequence_path) != frozen["sequences_sha256"]:
        raise ValueError("Frozen backbone sequences changed")
    payload = torch.load(sequence_path, map_location="cpu", weights_only=False)
    if payload.get("annotation_files_loaded") is not False:
        raise ValueError("Backbone input was not annotation-free")
    split_path = Path(frozen["split_json"])
    if sha256_file(split_path) != frozen["split_json_sha256"]:
        raise ValueError("Frozen split changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(map(int, split["train_video_ids"]))
    development_ids = set(map(int, split["validation_video_ids"]))
    test_ids = set(map(int, split["test_video_ids"]))
    if train_ids & (development_ids | test_ids):
        raise ValueError("Train overlaps development/test")
    folds = []
    for fold_row in frozen["fold_definition_files"]:
        path = Path(fold_row["path"])
        if sha256_file(path) != fold_row["sha256"]:
            raise ValueError("Frozen fold definition changed")
        fold = torch.load(path, map_location="cpu", weights_only=False)
        folds.append(sorted(map(int, fold["outer_heldout_video_ids"])))
    labeled_ids = {video_id for fold in folds for video_id in fold}
    if len(labeled_ids) != sum(map(len, folds)):
        raise ValueError("Frozen folds overlap")
    if set(payload["video_ids"]) != labeled_ids:
        raise ValueError("Backbone tracks are not exact labeled OOF scope")
    expected_fold = {
        video_id: f"fold{fold_index}"
        for fold_index, videos in enumerate(folds) for video_id in videos
    }
    sequences = list(payload["sequences"])
    for row in sequences:
        if row["outer_oof_fold"] != expected_fold[int(row["video_id"] )]:
            raise ValueError("Backbone track has wrong frozen OOF fold")

    # Labels are opened only after scope, hashes, and the complete policy are fixed.
    annotation_path = Path(split["dataset_root"]) / "annotations" / "cholec80-CVS.xlsx"
    annotations = {video_id: load_cvs_intervals(annotation_path, video_id) for video_id in labeled_ids}
    model_cfg = frozen["policy_family"]["model"]
    train_cfg = frozen["policy_family"]["training"]
    criteria = frozen["policy_family"]["criteria"]
    criterion_to_index = {name: index for index, name in enumerate(criteria)}
    embedding_table = torch.stack([
        payload["criterion_embeddings"][name].float() for name in criteria
    ])
    radius = int(model_cfg["window_radius_points"])
    flat_rows, all_windows, all_criterion_indices, all_states = [], [], [], []
    for row in sequences:
        video_id, criterion = int(row["video_id"]), str(row["criterion"])
        windows = centered_windows(row["backbone_features"], radius)
        states = torch.tensor([
            state_at(annotations[video_id][criterion], float(time_s))
            for time_s in row["center_times_s"]
        ], dtype=torch.long)
        for index, time_s in enumerate(row["center_times_s"]):
            flat_rows.append({
                "reference_id": row["reference_ids"][index],
                "video_id": video_id,
                "criterion": criterion,
                "center_time_s": float(time_s),
                "outer_oof_fold": row["outer_oof_fold"],
                "m2c_probability": float(row["m2c_probabilities"][index]),
                "m2c_threshold": float(row["m2c_threshold"]),
            })
        all_windows.append(windows)
        all_criterion_indices.append(torch.full((len(windows),), criterion_to_index[criterion], dtype=torch.long))
        all_states.append(states)
    windows = torch.cat(all_windows)
    criterion_indices = torch.cat(all_criterion_indices)
    states = torch.cat(all_states)
    if len(flat_rows) != len(windows):
        raise ValueError("Flattened backbone tracks are misaligned")

    device = torch.device(args.device)
    predictions = np.zeros(len(flat_rows), dtype=np.float64)
    fold_audits = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_target = {int(key): float(value) for key, value in train_cfg["state_targets"].items()}
    state_mass = {int(key): float(value) for key, value in train_cfg["state_sampling_mass"].items()}
    for fold_index, heldout_list in enumerate(folds):
        heldout = set(heldout_list)
        train_indices = [index for index, row in enumerate(flat_rows) if row["video_id"] not in heldout]
        valid_indices = [index for index, row in enumerate(flat_rows) if row["video_id"] in heldout]
        weights = torch.zeros(len(train_indices), dtype=torch.float64)
        for state in (0, 1, 2):
            state_criteria = sorted({flat_rows[index]["criterion"] for index in train_indices if int(states[index]) == state})
            for criterion in state_criteria:
                videos = sorted({flat_rows[index]["video_id"] for index in train_indices if int(states[index]) == state and flat_rows[index]["criterion"] == criterion})
                for video_id in videos:
                    positions = [position for position, index in enumerate(train_indices) if int(states[index]) == state and flat_rows[index]["criterion"] == criterion and flat_rows[index]["video_id"] == video_id]
                    index_tensor = torch.tensor(positions, dtype=torch.long)
                    raw = torch.ones(len(positions), dtype=torch.float64)
                    for local, position in enumerate(positions):
                        row = flat_rows[train_indices[position]]
                        if state == 2 and row["m2c_probability"] < row["m2c_threshold"]:
                            raw[local] *= float(train_cfg["m2c_missed_state2_multiplier_within_group"])
                        if state == 0 and row["m2c_probability"] >= row["m2c_threshold"]:
                            raw[local] *= float(train_cfg["m2c_hard_negative_multiplier_within_group"])
                    group_mass = state_mass[state] / len(state_criteria) / len(videos)
                    weights[index_tensor] = group_mass * raw / raw.sum()
        seed = int(train_cfg["seed"]) + int(args.seed_offset) + fold_index
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = TextConditionedBackboneRelationVerifier(
            visual_dim=windows.shape[-1], text_dim=embedding_table.shape[-1],
            hidden_dim=int(model_cfg["hidden_dim"]), kernel_size=int(model_cfg["kernel_size"]),
            dropout=float(model_cfg["dropout"]),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
        generator = torch.Generator().manual_seed(seed * 1000)
        losses = []
        model.train()
        for _ in range(int(train_cfg["steps"])):
            sampled_position = torch.multinomial(weights, int(train_cfg["batch_size"]), replacement=True, generator=generator)
            sampled = torch.tensor([train_indices[position] for position in sampled_position.tolist()], dtype=torch.long)
            logits = model(
                windows[sampled].to(device),
                embedding_table[criterion_indices[sampled]].to(device),
            )
            labels = torch.tensor([state_target[int(states[index])] for index in sampled], dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(valid_indices), 512):
                current = torch.tensor(valid_indices[start:start + 512], dtype=torch.long)
                probability = torch.sigmoid(model(
                    windows[current].to(device),
                    embedding_table[criterion_indices[current]].to(device),
                ))
                predictions[current.numpy()] = probability.cpu().numpy()
        checkpoint_path = args.output_dir / f"fold{fold_index}_yolo_backbone_relation.pt"
        torch.save({
            "schema_version": "yolo_backbone_relation_checkpoint_v1",
            "model_state": model.state_dict(),
            "model_config": model_cfg,
            "visual_feature_dim": int(windows.shape[-1]),
            "seed": seed, "seed_offset": int(args.seed_offset),
            "heldout_video_ids": heldout_list,
            "visual_backbone_parameters_updated": False,
            "foundation_model_parameters_updated": False,
            "development_or_test_accessed": False,
        }, checkpoint_path)
        fold_audits.append({
            "fold": f"fold{fold_index}", "heldout_video_ids": heldout_list,
            "train_points": len(train_indices), "valid_points": len(valid_indices),
            "mean_train_loss": float(np.mean(losses)), "checkpoint": str(checkpoint_path.resolve()),
        })
        print(json.dumps({"fold": fold_index, "train": len(train_indices), "valid": len(valid_indices), "mean_loss": fold_audits[-1]["mean_train_loss"]}), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    state_values = states.numpy()
    reports = {criterion: criterion_report(flat_rows, state_values, predictions, criterion) for criterion in criteria}
    gate_cfg = frozen["policy_family"]["frame_gate"]
    gates = {}
    for criterion in criteria:
        current, prior = reports[criterion], gate_cfg["prior_metrics"][criterion]
        checks = {
            "auc_preserved": current["roc_auc"] + 1e-12 >= prior["roc_auc"] - float(gate_cfg["auc_allowed_drop"]),
            "ap_or_miss_recall_improved": bool(
                current["average_precision"] + 1e-12 >= prior["average_precision"] * float(gate_cfg["ap_relative_improvement"])
                or current["visual_recall_at_0_5_on_state2_missed_by_m2c"] + 1e-12 >= prior["visual_recall_at_0_5_on_state2_missed_by_m2c"] + float(gate_cfg["miss_recall_absolute_improvement_alternative"])
            ),
            "hard_negative_fpr_below_ceiling": current["visual_fpr_at_0_5_on_m2c_hard_negatives"] <= float(gate_cfg["hard_negative_fpr_ceiling"]) + 1e-12,
        }
        gates[criterion] = {"passed": all(checks.values()), "checks": checks, "prior": prior}
    passed = sorted(name for name, gate in gates.items() if gate["passed"])
    proceed = len(passed) >= int(gate_cfg["minimum_passing_criteria_for_interval_proposals"])
    result = {
        "schema_version": "yolo_backbone_relation_oof_result_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_policy": str(args.frozen_policy.resolve()),
        "frozen_policy_sha256": sha256_file(args.frozen_policy),
        "protocol": "fixed_model_strict_video_oof_train_only",
        "frame_reports": reports,
        "criterion_gates": gates,
        "passed_criteria": passed,
        "proceed_to_nested_interval_proposals": proceed,
        "fold_audits": fold_audits,
        "visual_backbone_parameters_updated": False,
        "foundation_model_parameters_updated": False,
        "development_reused": False,
        "test_accessed": False,
    }
    result_path = args.output_dir / "oof_audit.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    prediction_rows = [{
        "reference_id": row["reference_id"], "video_id": row["video_id"],
        "criterion": row["criterion"], "center_time_s": row["center_time_s"],
        "outer_oof_fold": row["outer_oof_fold"],
        "yolo_backbone_relation_probability": float(predictions[index]),
    } for index, row in enumerate(flat_rows)]
    (args.output_dir / "predictions_oof.json").write_text(json.dumps({
        "schema_version": "yolo_backbone_relation_predictions_oof_v1",
        "frozen_policy": str(args.frozen_policy.resolve()),
        "annotation_files_loaded_during_prediction": False,
        "training_annotations_loaded_for_training_only": True,
        "visual_backbone_parameters_updated": False,
        "foundation_model_parameters_updated": False,
        "development_or_test_accessed": False,
        "seed_offset": int(args.seed_offset),
        "rows": prediction_rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(result_path.resolve()), "passed_criteria": passed,
        "proceed_to_nested_interval_proposals": proceed, "reports": reports,
        "development_reused": False, "test_accessed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
