#!/usr/bin/env python3
"""Run one strict SAGES outer fold with inner-fold model/epoch selection."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from cvs_assessment.detector_skill_cvs import (
    DetectorMultiLabelProbe, SkillConditionedDetectorHead, SkillConditionedTemporalHead,
)
from train_direct_interval_nested_oof import sha256_file
from train_endoscapes_text_spatial_temporal import average_precision

CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")


def majority(row: dict, key: str) -> float:
    return float(sum(int(row[f"{key}_rater{i}"]) for i in (1, 2, 3)) >= 2)


def build(candidate: dict) -> torch.nn.Module:
    family = candidate["family"]; dim = int(candidate["input_dim"])
    if family == "skill_shared_frame":
        return SkillConditionedDetectorHead(dim, 4096, int(candidate["hidden_dim"]), float(candidate["dropout"]))
    if family == "skill_shared_temporal":
        return SkillConditionedTemporalHead(dim, 4096, int(candidate["hidden_dim"]), float(candidate["dropout"]), tuple(map(int, candidate["dilations"])))
    if family == "multilabel_probe":
        return DetectorMultiLabelProbe(dim, 3, int(candidate["hidden_dim"]), float(candidate["dropout"]))
    raise ValueError(f"Unknown family: {family}")


def temporal(candidate: dict) -> bool:
    return candidate["family"] == "skill_shared_temporal"


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def predict(model: torch.nn.Module, features: torch.Tensor, skills: torch.Tensor, device: torch.device, is_temporal: bool) -> np.ndarray:
    model.eval(); values = features if is_temporal else features.reshape(-1, features.shape[-1]); outputs = []
    step = 32 if is_temporal else 512
    for start in range(0, len(values), step):
        outputs.append(torch.sigmoid(model(values[start:start + step].to(device), skills)).cpu())
    return torch.cat(outputs).reshape(len(features), features.shape[1], 3).numpy()


def fit(
    candidate: dict, train_x: torch.Tensor, train_y: torch.Tensor, train_soft_y: torch.Tensor,
    evaluation_x: torch.Tensor, skills: torch.Tensor, config: dict, device: torch.device,
    seed: int, checkpoint_epochs: list[int],
) -> dict[int, np.ndarray]:
    seed_all(seed); model = build(candidate).to(device); is_temporal = temporal(candidate)
    targets = train_soft_y if candidate["target_mode"] == "mean_rater" else train_y
    flat_targets = targets.reshape(-1, 3)
    positive_weight = ((len(flat_targets) - flat_targets.sum(0)) / flat_targets.sum(0).clamp_min(1)).clamp(1, float(config["positive_weight_cap"])).to(device)
    if is_temporal:
        dataset = TensorDataset(train_x, targets); batch_size = int(config["video_batch_size"])
    else:
        dataset = TensorDataset(train_x.reshape(-1, train_x.shape[-1]), flat_targets); batch_size = int(config["frame_batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]), eta_min=float(config["minimum_learning_rate"]))
    requested = set(map(int, checkpoint_epochs)); output = {}
    for epoch in range(1, max(requested) + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device), skills)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y.to(device), pos_weight=positive_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"])); optimizer.step()
        scheduler.step()
        if epoch in requested:
            output[epoch] = predict(model, evaluation_x, skills, device, is_temporal)
    return output


def load_development(protocol: dict) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    confirmation = set(protocol["locked_confirmation_video_ids"])
    ids = sorted(protocol["development_video_ids"])
    if set(ids) & confirmation: raise ValueError("Confirmation leakage")
    audit = json.loads(Path(protocol["sources"]["feature_cache_audit"]["path"]).read_text())
    rows = {row["video_id"]: row for row in audit["videos"]}
    label_root = Path(protocol["sources"]["train_label_download_audit"]["path"]).parent / "train" / "labels"
    xs = []; bases = []; ys = []; soft_ys = []
    for video_id in ids:
        value = torch.load(rows[video_id]["path"], map_location="cpu", weights_only=False)
        if value.get("SAGES_test_labels_accessed") is not False: raise ValueError("Unsafe cache")
        labels = {}
        with (label_root / video_id / "frame.csv").open(newline="") as handle:
            for item in csv.DictReader(handle):
                # Official C1,C2,C3 -> internal two_structures,cystic_plate,hepatocystic_triangle.
                labels[int(item["frame_id"])] = (
                    [majority(item, key) for key in ("c1", "c3", "c2")],
                    [sum(int(item[f"{key}_rater{i}"]) for i in (1, 2, 3)) / 3 for key in ("c1", "c3", "c2")],
                )
        frame_ids = list(map(int, value["frame_ids"].tolist()))
        if len(frame_ids) != 18 or set(frame_ids) != set(labels): raise ValueError(f"Alignment: {video_id}")
        xs.append(value["detector_and_dual_moco_features"].float()); bases.append(value["baseline_probability"].float())
        ys.append(torch.tensor([labels[index][0] for index in frame_ids], dtype=torch.float32))
        soft_ys.append(torch.tensor([labels[index][1] for index in frame_ids], dtype=torch.float32))
    return ids, torch.stack(xs), torch.stack(bases), torch.stack(ys), torch.stack(soft_ys)


def macro_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.reshape(-1, 3); scores = scores.reshape(-1, 3)
    return float(np.mean([average_precision(labels[:, index], scores[:, index]) for index in range(3)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--outer-fold-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    if protocol["training_code_sha256"] != sha256_file(Path(__file__)) or protocol["model_code_sha256"] != sha256_file(ROOT / "cvs_assessment/detector_skill_cvs.py"):
        raise ValueError("Code differs from frozen protocol")
    fold = protocol["outer_folds"][args.outer_fold_index]
    if fold["outer_fold_index"] != args.outer_fold_index: raise ValueError("Fold index mismatch")
    ids, x, base, y, soft_y = load_development(protocol); index = {video_id: position for position, video_id in enumerate(ids)}
    take = lambda selected: torch.tensor([index[video_id] for video_id in selected], dtype=torch.long)
    skills_payload = torch.load(protocol["sources"]["frozen_skill_embeddings"]["path"], map_location="cpu", weights_only=False)
    skills = torch.stack([skills_payload["criterion_embeddings"][key] for key in CRITERIA]).to(args.device)
    device = torch.device(args.device); config = protocol["training"]; epochs = list(map(int, config["checkpoint_epochs"]))
    candidate_records = []; stored_predictions = {}
    outer_train_positions = take(fold["training_video_ids"]); outer_train_labels = y[outer_train_positions].numpy()
    position_within_outer = {video_id: position for position, video_id in enumerate(fold["training_video_ids"])}
    for candidate_index, candidate in enumerate(protocol["candidates"]):
        oof_by_epoch = {epoch: np.zeros((len(fold["training_video_ids"]), 18, 3), dtype=np.float32) for epoch in epochs}
        for inner in fold["inner_folds"]:
            train_positions = take(inner["training_video_ids"]); validation_positions = take(inner["validation_video_ids"])
            predictions = fit(
                candidate, x[train_positions], y[train_positions], soft_y[train_positions], x[validation_positions],
                skills, config, device, int(config["seed"]) + args.outer_fold_index * 10000 + candidate_index * 100 + inner["inner_fold_index"], epochs,
            )
            destinations = [position_within_outer[video_id] for video_id in inner["validation_video_ids"]]
            for epoch in epochs: oof_by_epoch[epoch][destinations] = predictions[epoch]
        epoch_scores = {str(epoch): macro_ap(outer_train_labels, oof_by_epoch[epoch]) for epoch in epochs}
        best_epoch = max(epochs, key=lambda epoch: (epoch_scores[str(epoch)], -epoch))
        candidate_records.append({"candidate_index": candidate_index, "candidate": candidate, "inner_OOF_mAP_by_epoch": epoch_scores, "selected_epoch": best_epoch, "selected_inner_OOF_mAP": epoch_scores[str(best_epoch)]})
        stored_predictions[candidate_index] = oof_by_epoch
        print(json.dumps({"outer_fold": args.outer_fold_index, "candidate": candidate_index + 1, "of": len(protocol["candidates"]), "selected_epoch": best_epoch, "inner_OOF_mAP": epoch_scores[str(best_epoch)]}), flush=True)
    primary_indices = [record["candidate_index"] for record in candidate_records if record["candidate"]["family"] in protocol["primary_families"]]
    selected_index = max(primary_indices, key=lambda value: (candidate_records[value]["selected_inner_OOF_mAP"], -value))
    selected = candidate_records[selected_index]; inner_scores = stored_predictions[selected_index][selected["selected_epoch"]]
    outer_train_base = base[outer_train_positions].numpy(); fusion_weights = {}; fused_inner = np.empty_like(inner_scores)
    for criterion_index, criterion in enumerate(CRITERIA):
        options = []
        for weight in config["fusion_weights"]:
            score = weight * inner_scores[:, :, criterion_index] + (1 - weight) * outer_train_base[:, :, criterion_index]
            options.append((average_precision(outer_train_labels[:, :, criterion_index].reshape(-1), score.reshape(-1)), float(weight)))
        _, weight = max(options, key=lambda item: (item[0], -item[1])); fusion_weights[criterion] = weight
        fused_inner[:, :, criterion_index] = weight * inner_scores[:, :, criterion_index] + (1 - weight) * outer_train_base[:, :, criterion_index]
    outer_test_positions = take(fold["test_video_ids"])
    final_predictions = fit(
        selected["candidate"], x[outer_train_positions], y[outer_train_positions], soft_y[outer_train_positions], x[outer_test_positions],
        skills, config, device, int(config["seed"]) + args.outer_fold_index * 10000 + 9999, [selected["selected_epoch"]],
    )[selected["selected_epoch"]]
    outer_test_base = base[outer_test_positions].numpy(); framework = np.empty_like(final_predictions)
    for criterion_index, criterion in enumerate(CRITERIA):
        weight = fusion_weights[criterion]
        framework[:, :, criterion_index] = weight * final_predictions[:, :, criterion_index] + (1 - weight) * outer_test_base[:, :, criterion_index]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output_dir / "outer_test_predictions.pt"
    torch.save({
        "schema_version": "sages_cvs_2024_strict_nested_oof_outer_predictions_v1",
        "outer_fold_index": args.outer_fold_index,
        "video_ids": fold["test_video_ids"],
        "labels": y[outer_test_positions], "baseline_probability": base[outer_test_positions],
        "selected_skill_probability": torch.from_numpy(final_predictions), "framework_probability": torch.from_numpy(framework),
        "criterion_order": CRITERIA, "SAGES_confirmation_labels_used": False, "SAGES_test_labels_accessed": False,
        "LLM_or_MLLM_parameters_updated": False,
    }, prediction_path)
    result = {
        "schema_version": "sages_cvs_2024_strict_nested_oof_outer_result_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "outer_fold_index": args.outer_fold_index, "inner_fold_count": len(fold["inner_folds"]),
        "training_video_count": len(fold["training_video_ids"]), "test_video_count": len(fold["test_video_ids"]),
        "candidate_records": candidate_records, "selected": selected, "fusion_weights": fusion_weights,
        "inner_selected_framework_mAP": macro_ap(outer_train_labels, fused_inner),
        "outer_test_baseline_mAP": macro_ap(y[outer_test_positions].numpy(), outer_test_base),
        "outer_test_framework_mAP": macro_ap(y[outer_test_positions].numpy(), framework),
        "predictions": {"path": str(prediction_path.resolve()), "sha256": sha256_file(prediction_path)},
        "SAGES_confirmation_labels_used": False, "SAGES_test_labels_accessed": False, "LLM_or_MLLM_parameters_updated": False,
    }
    result_path = args.output_dir / "outer_result.json"; result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(result_path.resolve()), "selected": selected, "fusion_weights": fusion_weights, "baseline_mAP": result["outer_test_baseline_mAP"], "framework_mAP": result["outer_test_framework_mAP"]}, indent=2))


if __name__ == "__main__":
    main()
