#!/usr/bin/env python3
"""Run one DINOv2 visual-upgrade outer fold with strict inner OOF selection."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_sages_nested_oof_outer_fold as base
from train_direct_interval_nested_oof import sha256_file
from train_endoscapes_text_spatial_temporal import average_precision

CRITERIA = base.CRITERIA


def feature_view(existing: torch.Tensor, dino: torch.Tensor, candidate: dict) -> torch.Tensor:
    name = candidate["feature_set"]
    if name == "dinov2_only": result = dino
    elif name == "dinov2_plus_detector": result = torch.cat([existing[..., :1298], dino], dim=-1)
    elif name == "all_visual": result = torch.cat([existing, dino], dim=-1)
    else: raise ValueError(name)
    if result.shape[-1] != int(candidate["input_dim"]): raise ValueError("Feature dimension mismatch")
    return result


def pairwise_rank_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    values = []
    flat_logits = logits.reshape(-1, 3); flat_targets = targets.reshape(-1, 3)
    for criterion in range(3):
        positive = flat_logits[flat_targets[:, criterion] >= .5, criterion]; negative = flat_logits[flat_targets[:, criterion] < .5, criterion]
        if len(positive) and len(negative): values.append(torch.nn.functional.softplus(negative[:, None] - positive[None, :]).mean())
    return torch.stack(values).mean() if values else logits.sum() * 0


def fit(candidate: dict, train_x: torch.Tensor, train_y: torch.Tensor, train_soft_y: torch.Tensor, evaluation_x: torch.Tensor, skills: torch.Tensor, config: dict, device: torch.device, seed: int, checkpoint_epochs: list[int]) -> dict[int, np.ndarray]:
    base.seed_all(seed); model = base.build(candidate).to(device); targets = train_soft_y if candidate["target_mode"] == "mean_rater" else train_y; flat_targets = targets.reshape(-1, 3)
    if candidate["positive_weight_mode"] == "none": positive_weight = torch.ones(3, device=device)
    elif candidate["positive_weight_mode"] == "sqrt_cap_2": positive_weight = (((len(flat_targets) - flat_targets.sum(0)) / flat_targets.sum(0).clamp_min(1)).sqrt().clamp(1, 2)).to(device)
    else: raise ValueError(candidate["positive_weight_mode"])
    dataset = TensorDataset(train_x.reshape(-1, train_x.shape[-1]), flat_targets); loader = DataLoader(dataset, batch_size=int(config["frame_batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]), eta_min=float(config["minimum_learning_rate"])); requested = set(map(int, checkpoint_epochs)); output = {}
    for epoch in range(1, max(requested) + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True); logits = model(batch_x.to(device), skills); target = batch_y.to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
            if candidate["loss_mode"] == "bce_plus_pairwise_rank": loss = loss + float(candidate["rank_weight"]) * pairwise_rank_loss(logits, target)
            elif candidate["loss_mode"] != "bce": raise ValueError(candidate["loss_mode"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"])); optimizer.step()
        scheduler.step()
        if epoch in requested: output[epoch] = base.predict(model, evaluation_x, skills, device, False)
    return output


def load_development(protocol: dict) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = sorted(protocol["development_video_ids"]); confirmation = set(protocol["locked_confirmation_video_ids"])
    if set(ids) & confirmation: raise ValueError("Confirmation leakage")
    visual_audit = json.loads(Path(protocol["sources"]["visual_feature_cache_audit"]["path"]).read_text()); dino_audit = json.loads(Path(protocol["sources"]["dinov2_feature_cache_audit"]["path"]).read_text())
    visual_rows = {row["video_id"]: row for row in visual_audit["videos"]}; dino_rows = {row["video_id"]: row for row in dino_audit["videos"]}; label_root = Path(protocol["sources"]["train_label_download_audit"]["path"]).parent / "train" / "labels"
    existing_values = []; dino_values = []; bases = []; ys = []; soft_ys = []
    for video_id in ids:
        visual = torch.load(visual_rows[video_id]["path"], map_location="cpu", weights_only=False); dino = torch.load(dino_rows[video_id]["path"], map_location="cpu", weights_only=False)
        if visual.get("SAGES_test_labels_accessed") is not False or dino.get("SAGES_test_labels_accessed") is not False: raise ValueError("Unsafe cache")
        labels = {}
        with (label_root / video_id / "frame.csv").open(newline="") as handle:
            for item in csv.DictReader(handle): labels[int(item["frame_id"])] = ([base.majority(item, key) for key in ("c1", "c3", "c2")], [sum(int(item[f"{key}_rater{i}"]) for i in (1, 2, 3)) / 3 for key in ("c1", "c3", "c2")])
        frame_ids = list(map(int, visual["frame_ids"].tolist()))
        if frame_ids != list(map(int, dino["frame_ids"].tolist())) or set(frame_ids) != set(labels): raise ValueError(f"Alignment: {video_id}")
        existing_values.append(visual["detector_and_dual_moco_features"].float()); dino_values.append(dino["dual_view_DINOv2S_features"].float()); bases.append(visual["baseline_probability"].float()); ys.append(torch.tensor([labels[index][0] for index in frame_ids])); soft_ys.append(torch.tensor([labels[index][1] for index in frame_ids]))
    return ids, torch.stack(existing_values), torch.stack(dino_values), torch.stack(bases), torch.stack(ys).float(), torch.stack(soft_ys).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--outer-fold-index", type=int, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda:2"); args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    if protocol["training_code_sha256"] != sha256_file(Path(__file__)) or protocol["base_training_code_sha256"] != sha256_file(ROOT / "scripts/run_sages_nested_oof_outer_fold.py") or protocol["model_code_sha256"] != sha256_file(ROOT / "cvs_assessment/detector_skill_cvs.py"): raise ValueError("Code differs from frozen protocol")
    fold = protocol["outer_folds"][args.outer_fold_index]; ids, existing, dino, baseline_scores, y, soft_y = load_development(protocol); index = {video_id: position for position, video_id in enumerate(ids)}; take = lambda selected: torch.tensor([index[v] for v in selected], dtype=torch.long)
    skill_payload = torch.load(protocol["sources"]["frozen_skill_embeddings"]["path"], map_location="cpu", weights_only=False); device = torch.device(args.device); skills = torch.stack([skill_payload["criterion_embeddings"][key] for key in CRITERIA]).to(device); config = protocol["training"]; epochs = list(map(int, config["checkpoint_epochs"])); outer_train_positions = take(fold["training_video_ids"]); outer_train_labels = y[outer_train_positions].numpy(); position_within_outer = {video_id: position for position, video_id in enumerate(fold["training_video_ids"])}
    candidate_records = []; stored = {}
    for candidate_index, candidate in enumerate(protocol["candidates"]):
        features = feature_view(existing, dino, candidate); oof = {epoch: np.zeros((480, 18, 3), dtype=np.float32) for epoch in epochs}
        for inner in fold["inner_folds"]:
            train_positions = take(inner["training_video_ids"]); validation_positions = take(inner["validation_video_ids"]); predictions = fit(candidate, features[train_positions], y[train_positions], soft_y[train_positions], features[validation_positions], skills, config, device, int(config["seed"]) + args.outer_fold_index * 10000 + candidate_index * 100 + inner["inner_fold_index"], epochs); destinations = [position_within_outer[v] for v in inner["validation_video_ids"]]
            for epoch in epochs: oof[epoch][destinations] = predictions[epoch]
        scores = {str(epoch): base.macro_ap(outer_train_labels, oof[epoch]) for epoch in epochs}; selected_epoch = max(epochs, key=lambda epoch: (scores[str(epoch)], -epoch)); candidate_records.append({"candidate_index": candidate_index, "candidate": candidate, "inner_OOF_mAP_by_epoch": scores, "selected_epoch": selected_epoch, "selected_inner_OOF_mAP": scores[str(selected_epoch)]}); stored[candidate_index] = oof; print(json.dumps({"outer_fold": args.outer_fold_index, "candidate": candidate_index + 1, "of": len(protocol["candidates"]), "selected_epoch": selected_epoch, "inner_OOF_mAP": scores[str(selected_epoch)]}), flush=True)
    selected_index = max(range(len(candidate_records)), key=lambda value: (candidate_records[value]["selected_inner_OOF_mAP"], -value)); selected = candidate_records[selected_index]; inner_scores = stored[selected_index][selected["selected_epoch"]]; outer_train_base = baseline_scores[outer_train_positions].numpy(); fusion_weights = {}; fused_inner = np.empty_like(inner_scores)
    for criterion_index, criterion in enumerate(CRITERIA):
        options = [(average_precision(outer_train_labels[:, :, criterion_index].reshape(-1), (weight * inner_scores[:, :, criterion_index] + (1 - weight) * outer_train_base[:, :, criterion_index]).reshape(-1)), float(weight)) for weight in config["fusion_weights"]]; _, weight = max(options, key=lambda item: (item[0], -item[1])); fusion_weights[criterion] = weight; fused_inner[:, :, criterion_index] = weight * inner_scores[:, :, criterion_index] + (1 - weight) * outer_train_base[:, :, criterion_index]
    outer_test_positions = take(fold["test_video_ids"]); selected_features = feature_view(existing, dino, selected["candidate"]); final_predictions = fit(selected["candidate"], selected_features[outer_train_positions], y[outer_train_positions], soft_y[outer_train_positions], selected_features[outer_test_positions], skills, config, device, int(config["seed"]) + args.outer_fold_index * 10000 + 9999, [selected["selected_epoch"]])[selected["selected_epoch"]]; outer_base = baseline_scores[outer_test_positions].numpy(); framework = np.empty_like(final_predictions)
    for criterion_index, criterion in enumerate(CRITERIA): framework[:, :, criterion_index] = fusion_weights[criterion] * final_predictions[:, :, criterion_index] + (1 - fusion_weights[criterion]) * outer_base[:, :, criterion_index]
    args.output_dir.mkdir(parents=True, exist_ok=False); prediction_path = args.output_dir / "outer_test_predictions.pt"; torch.save({"schema_version": "sages_cvs_2024_dinov2_nested_oof_outer_predictions_v1", "outer_fold_index": args.outer_fold_index, "video_ids": fold["test_video_ids"], "labels": y[outer_test_positions], "baseline_probability": baseline_scores[outer_test_positions], "selected_skill_probability": torch.from_numpy(final_predictions), "framework_probability": torch.from_numpy(framework), "criterion_order": CRITERIA, "SAGES_confirmation_labels_used": False, "SAGES_test_labels_accessed": False, "LLM_or_MLLM_parameters_updated": False}, prediction_path)
    result = {"schema_version": "sages_cvs_2024_dinov2_nested_oof_outer_result_v1", "created_at": datetime.now(timezone.utc).isoformat(), "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)}, "outer_fold_index": args.outer_fold_index, "inner_fold_count": 4, "training_video_count": 480, "test_video_count": 120, "candidate_records": candidate_records, "selected": selected, "fusion_weights": fusion_weights, "inner_selected_framework_mAP": base.macro_ap(outer_train_labels, fused_inner), "outer_test_baseline_mAP": base.macro_ap(y[outer_test_positions].numpy(), outer_base), "outer_test_framework_mAP": base.macro_ap(y[outer_test_positions].numpy(), framework), "predictions": {"path": str(prediction_path.resolve()), "sha256": sha256_file(prediction_path)}, "SAGES_confirmation_labels_used": False, "SAGES_test_labels_accessed": False, "LLM_or_MLLM_parameters_updated": False}
    path = args.output_dir / "outer_result.json"; path.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"result": str(path.resolve()), "selected": selected, "fusion_weights": fusion_weights, "baseline_mAP": result["outer_test_baseline_mAP"], "framework_mAP": result["outer_test_framework_mAP"]}, indent=2))


if __name__ == "__main__": main()
