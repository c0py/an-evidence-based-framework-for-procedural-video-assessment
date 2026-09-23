#!/usr/bin/env python3
"""Train the fixed spatial-ROI Skill-conditioned verifier with strict video OOF."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.boundary_supervision import (
    BoundaryTargetConfig, adjacent_sequence_losses, boundary_targets,
)
from cvs_assessment.spatial_roi_relation import TextConditionedSpatialRoiRelationVerifier
from train_dense_residual_fusion_oof import state_at
from train_yolo_backbone_relation_oof import criterion_report, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_policy.read_text(encoding="utf-8"))
    if frozen.get("new_oof_labels_loaded_during_freeze") is not False or frozen.get("development_or_test_accessed") is not False:
        raise ValueError("Policy was not safely frozen")
    sequence_path = Path(frozen["sequences"])
    if sha256_file(sequence_path) != frozen["sequences_sha256"]:
        raise ValueError("Frozen spatial sequences changed")
    payload = torch.load(sequence_path, map_location="cpu", weights_only=False)
    if payload.get("annotation_files_loaded") is not False or payload.get("development_or_test_accessed") is not False:
        raise ValueError("Spatial sequence input is not annotation-free train-only")
    split_path = Path(frozen["split_json"])
    if sha256_file(split_path) != frozen["split_json_sha256"]:
        raise ValueError("Frozen split changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(map(int, split["train_video_ids"]))
    if train_ids & (set(map(int, split["validation_video_ids"])) | set(map(int, split["test_video_ids"]))):
        raise ValueError("Train overlaps development/test")
    folds = []
    for fold_row in frozen["fold_definition_files"]:
        path = Path(fold_row["path"])
        if sha256_file(path) != fold_row["sha256"]:
            raise ValueError("Frozen fold definition changed")
        row = torch.load(path, map_location="cpu", weights_only=False)
        folds.append(sorted(map(int, row["outer_heldout_video_ids"])))
    labeled_ids = {video_id for fold in folds for video_id in fold}
    if len(labeled_ids) != sum(map(len, folds)) or set(payload["video_ids"]) != labeled_ids:
        raise ValueError("Spatial sequences are not exact nonoverlapping OOF scope")
    expected_fold = {video_id: f"fold{fold_index}" for fold_index, videos in enumerate(folds) for video_id in videos}
    sequences = list(payload["sequences"])
    for row in sequences:
        if row["outer_oof_fold"] != expected_fold[int(row["video_id"])]:
            raise ValueError("Spatial sequence has wrong OOF fold")

    # Open train labels only after the complete artifact/policy/fold audit above.
    annotation_path = Path(split["dataset_root"]) / "annotations" / "cholec80-CVS.xlsx"
    annotations = {video_id: load_cvs_intervals(annotation_path, video_id) for video_id in labeled_ids}
    model_cfg, train_cfg = frozen["policy_family"]["model"], frozen["policy_family"]["training"]
    criteria = frozen["policy_family"]["criteria"]
    criterion_to_index = {name: index for index, name in enumerate(criteria)}
    embedding_table = torch.stack([payload["criterion_embeddings"][name].float() for name in criteria])
    global_count = int(payload["token_layout"]["global_grid_tokens"])
    semantic_types = list(payload["semantic_types"])
    roi_classes = payload["task_visual_evidence_schema"]["criterion_roi_classes"]
    relevance = {}
    for criterion in criteria:
        relevant = set(roi_classes[criterion])
        relevance[criterion] = torch.tensor(
            [1.0] * global_count + [float(name in relevant) for name in semantic_types],
            dtype=torch.float16,
        )
    radius = int(model_cfg["window_radius_points"])
    flat_rows, states, soft_targets, boundary_zones = [], [], [], []
    boundary_cfg = train_cfg.get("boundary_supervision")
    for sequence_index, row in enumerate(sequences):
        video_id, criterion = int(row["video_id"]), str(row["criterion"])
        current_states = [state_at(annotations[video_id][criterion], float(time_s)) for time_s in row["center_times_s"]]
        if boundary_cfg:
            generated = boundary_targets(
                list(map(float, row["center_times_s"])), annotations[video_id][criterion],
                BoundaryTargetConfig(
                    cadence_s=float(boundary_cfg["cadence_s"]),
                    inner_margin_steps=float(boundary_cfg["inner_margin_steps"]),
                    outer_margin_steps=float(boundary_cfg["outer_margin_steps"]),
                    boundary_target=float(boundary_cfg["boundary_target"]),
                    partial_target=float(boundary_cfg["partial_target"]),
                ),
            )
            soft_targets.extend(map(float, generated["targets"]))
            boundary_zones.extend(map(int, generated["zones"]))
        for point_index, time_s in enumerate(row["center_times_s"]):
            flat_rows.append({
                "sequence_index": sequence_index, "point_index": point_index,
                "reference_id": row["reference_ids"][point_index],
                "video_id": video_id, "criterion": criterion,
                "criterion_index": criterion_to_index[criterion],
                "center_time_s": float(time_s), "outer_oof_fold": row["outer_oof_fold"],
                "m2c_probability": float(row["m2c_probabilities"][point_index]),
                "m2c_threshold": float(row["m2c_threshold"]),
            })
        states.extend(current_states)
    states = torch.tensor(states, dtype=torch.long)
    soft_targets_tensor = (
        torch.tensor(soft_targets, dtype=torch.float32) if boundary_cfg else None
    )
    boundary_zones_tensor = (
        torch.tensor(boundary_zones, dtype=torch.long) if boundary_cfg else None
    )
    flat_index_by_sequence_point = {
        (int(row["sequence_index"]), int(row["point_index"])): index
        for index, row in enumerate(flat_rows)
    }

    # Concatenate the annotation-free sequence tensors once and gather complete temporal
    # windows with one indexed tensor operation. This is numerically equivalent to the former
    # per-sample Python loop but prevents CPU window assembly from starving the GPU.
    token_sources, geometry_sources, window_rows = [], [], []
    sequence_offset = 0
    for sequence in sequences:
        length = len(sequence["tokens"])
        token_sources.append(sequence["tokens"])
        current_relevance = relevance[str(sequence["criterion"])][None, :, None].expand(
            length, -1, -1,
        )
        geometry_sources.append(torch.cat([sequence["geometry"], current_relevance], dim=-1))
        for center in range(length):
            window_rows.append([
                sequence_offset + min(length - 1, max(0, center + offset))
                for offset in range(-radius, radius + 1)
            ])
        sequence_offset += length
    token_source = torch.cat(token_sources, dim=0)
    geometry_source = torch.cat(geometry_sources, dim=0)
    window_index = torch.tensor(window_rows, dtype=torch.long)
    if len(window_index) != len(flat_rows):
        raise AssertionError("Vectorized temporal window index is misaligned")

    def gather(flat_indices: torch.Tensor):
        indices = window_index[flat_indices]
        return token_source[indices], geometry_source[indices]

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
                    position_tensor = torch.tensor(positions, dtype=torch.long)
                    raw = torch.ones(len(positions), dtype=torch.float64)
                    for local, position in enumerate(positions):
                        row = flat_rows[train_indices[position]]
                        if state == 2 and row["m2c_probability"] < row["m2c_threshold"]:
                            raw[local] *= float(train_cfg["m2c_missed_state2_multiplier_within_group"])
                        if state == 0 and row["m2c_probability"] >= row["m2c_threshold"]:
                            raw[local] *= float(train_cfg["m2c_hard_negative_multiplier_within_group"])
                        if boundary_cfg and int(boundary_zones_tensor[train_indices[position]]) in (1, 2):
                            raw[local] *= float(boundary_cfg["boundary_sampling_multiplier"])
                    group_mass = state_mass[state] / len(state_criteria) / len(videos)
                    weights[position_tensor] = group_mass * raw / raw.sum()
        seed = int(train_cfg["seed"]) + int(args.seed_offset) + fold_index
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = TextConditionedSpatialRoiRelationVerifier(
            token_dim=int(payload["token_dim"]), geometry_dim=len(payload["geometry_layout"]) + 1,
            text_dim=embedding_table.shape[-1], hidden_dim=int(model_cfg["hidden_dim"]),
            attention_heads=int(model_cfg["attention_heads"]),
            temporal_kernel_size=int(model_cfg["temporal_kernel_size"]), dropout=float(model_cfg["dropout"]),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
        generator = torch.Generator().manual_seed(seed * 1000)
        losses = []
        model.train()
        for _ in range(int(train_cfg["steps"])):
            sample_count = int(train_cfg["batch_size"])
            if boundary_cfg:
                sample_count = max(1, sample_count // 2)
            sampled_position = torch.multinomial(weights, sample_count, replacement=True, generator=generator)
            sampled = torch.tensor([train_indices[position] for position in sampled_position.tolist()], dtype=torch.long)
            if boundary_cfg:
                paired = []
                for index in sampled.tolist():
                    meta = flat_rows[index]
                    sequence_length = len(sequences[int(meta["sequence_index"])]["center_times_s"])
                    neighbor_point = int(meta["point_index"]) + 1
                    if neighbor_point >= sequence_length:
                        neighbor_point = max(0, int(meta["point_index"]) - 1)
                    neighbor = flat_index_by_sequence_point[(int(meta["sequence_index"]), neighbor_point)]
                    if flat_rows[neighbor]["video_id"] in heldout:
                        raise AssertionError("Adjacent training pair crossed into the held-out fold")
                    paired.extend([index, neighbor])
                sampled = torch.tensor(paired, dtype=torch.long)
            token_batch, geometry_batch = gather(sampled)
            criterion_batch = torch.tensor([flat_rows[index]["criterion_index"] for index in sampled], dtype=torch.long)
            logits = model(token_batch.to(device), geometry_batch.to(device), embedding_table[criterion_batch].to(device))
            if boundary_cfg:
                labels = soft_targets_tensor[sampled].to(device)
                point_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                adjacent = adjacent_sequence_losses(
                    logits.reshape(-1, 2), labels.reshape(-1, 2),
                    transition_delta=float(boundary_cfg["transition_delta"]),
                    monotonic_margin_scale=float(boundary_cfg["monotonic_margin_scale"]),
                )
                loss = (
                    point_loss
                    + float(boundary_cfg["consistency_loss_weight"]) * adjacent["consistency"]
                    + float(boundary_cfg["monotonicity_loss_weight"]) * adjacent["monotonicity"]
                )
            else:
                labels = torch.tensor([state_target[int(states[index])] for index in sampled], dtype=torch.float32, device=device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(valid_indices), 64):
                current = torch.tensor(valid_indices[start:start + 64], dtype=torch.long)
                token_batch, geometry_batch = gather(current)
                criterion_batch = torch.tensor([flat_rows[index]["criterion_index"] for index in current], dtype=torch.long)
                probability = torch.sigmoid(model(token_batch.to(device), geometry_batch.to(device), embedding_table[criterion_batch].to(device)))
                predictions[current.numpy()] = probability.cpu().numpy()
        checkpoint_path = args.output_dir / f"fold{fold_index}_spatial_roi_relation.pt"
        torch.save({
            "schema_version": "spatial_roi_relation_checkpoint_v1", "model_state": model.state_dict(),
            "model_config": model_cfg, "token_dim": int(payload["token_dim"]),
            "geometry_dim": len(payload["geometry_layout"]) + 1, "heldout_video_ids": heldout_list,
            "seed": seed, "seed_offset": int(args.seed_offset),
            "visual_backbone_parameters_updated": False, "foundation_model_parameters_updated": False,
            "development_or_test_accessed": False,
        }, checkpoint_path)
        fold_audits.append({"fold": f"fold{fold_index}", "heldout_video_ids": heldout_list, "train_points": len(train_indices), "valid_points": len(valid_indices), "mean_train_loss": float(np.mean(losses)), "boundary_supervision_enabled": bool(boundary_cfg), "checkpoint": str(checkpoint_path.resolve())})
        print(json.dumps({"fold": fold_index, "train": len(train_indices), "valid": len(valid_indices), "mean_loss": fold_audits[-1]["mean_train_loss"]}), flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    state_values = states.numpy()
    reports = {criterion: criterion_report(flat_rows, state_values, predictions, criterion) for criterion in criteria}
    gate_cfg, gates = frozen["policy_family"]["frame_gate"], {}
    for criterion in criteria:
        current, prior = reports[criterion], gate_cfg["prior_metrics"][criterion]
        checks = {
            "auc_preserved": current["roc_auc"] + 1e-12 >= prior["roc_auc"] - float(gate_cfg["auc_allowed_drop"]),
            "ap_or_miss_recall_improved": bool(current["average_precision"] + 1e-12 >= prior["average_precision"] * float(gate_cfg["ap_relative_improvement"]) or current["visual_recall_at_0_5_on_state2_missed_by_m2c"] + 1e-12 >= prior["visual_recall_at_0_5_on_state2_missed_by_m2c"] + float(gate_cfg["miss_recall_absolute_improvement_alternative"])),
            "hard_negative_fpr_below_ceiling": current["visual_fpr_at_0_5_on_m2c_hard_negatives"] <= float(gate_cfg["hard_negative_fpr_ceiling"]) + 1e-12,
        }
        gates[criterion] = {"passed": all(checks.values()), "checks": checks, "prior": prior}
    passed = sorted(name for name, gate in gates.items() if gate["passed"])
    proceed = len(passed) >= int(gate_cfg["minimum_passing_criteria_for_interval_proposals"])
    result = {
        "schema_version": "spatial_roi_relation_oof_result_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_policy": str(args.frozen_policy.resolve()), "frozen_policy_sha256": sha256_file(args.frozen_policy),
        "protocol": "fixed_model_strict_video_oof_train_only", "frame_reports": reports,
        "boundary_supervision_enabled": bool(boundary_cfg),
        "criterion_gates": gates, "passed_criteria": passed, "proceed_to_nested_interval_proposals": proceed,
        "fold_audits": fold_audits, "visual_backbone_parameters_updated": False,
        "foundation_model_parameters_updated": False, "development_reused": False, "test_accessed": False,
    }
    result_path = args.output_dir / "oof_audit.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows = [{
        "reference_id": row["reference_id"], "video_id": row["video_id"], "criterion": row["criterion"],
        "center_time_s": row["center_time_s"], "outer_oof_fold": row["outer_oof_fold"],
        "spatial_roi_relation_probability": float(predictions[index]),
    } for index, row in enumerate(flat_rows)]
    (args.output_dir / "predictions_oof.json").write_text(json.dumps({
        "schema_version": "spatial_roi_relation_predictions_oof_v1", "frozen_policy": str(args.frozen_policy.resolve()),
        "annotation_files_loaded_during_prediction": False, "training_annotations_loaded_for_training_only": True,
        "visual_backbone_parameters_updated": False, "foundation_model_parameters_updated": False,
        "development_or_test_accessed": False, "seed_offset": int(args.seed_offset), "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(result_path.resolve()), "passed_criteria": passed, "proceed_to_nested_interval_proposals": proceed, "reports": reports, "development_reused": False, "test_accessed": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
