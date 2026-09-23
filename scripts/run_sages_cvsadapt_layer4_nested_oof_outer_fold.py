#!/usr/bin/env python3
"""Run one strict SAGES outer fold with controlled CVS-AdaptNet layer4 adaptation."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from cache_sages_cvsadapt_visual_features import load_visual
from cvs_assessment.cvsadapt_layer4_cvs import (
    CVSAdaptLayer4Fusion, SkillConditionedCVSAdaptLayer4Fusion,
)
from train_direct_interval_nested_oof import sha256_file
from train_endoscapes_text_spatial_temporal import average_precision
import run_sages_nested_oof_outer_fold as base

CRITERIA = base.CRITERIA
LAYER4_TEMPLATE = None
PROJECTION_TEMPLATE = None


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build(candidate: dict) -> torch.nn.Module:
    common = {
        "layer4": copy.deepcopy(LAYER4_TEMPLATE),
        "projection": copy.deepcopy(PROJECTION_TEMPLATE),
        "context_dim": int(candidate["context_dim"]),
        "hidden_dim": int(candidate["hidden_dim"]),
        "dropout": float(candidate["dropout"]),
        "layer4_trainable": bool(candidate["layer4_trainable"]),
        "projection_trainable": bool(candidate["projection_trainable"]),
    }
    if candidate["family"] == "layer4_probe":
        return CVSAdaptLayer4Fusion(criterion_count=3, **common)
    if candidate["family"] == "layer4_skill_shared":
        return SkillConditionedCVSAdaptLayer4Fusion(text_dim=4096, **common)
    raise ValueError(candidate["family"])


@torch.inference_mode()
def predict(
    model: torch.nn.Module, layer3: torch.Tensor, context: torch.Tensor,
    skills: torch.Tensor, device: torch.device, batch_size: int,
) -> np.ndarray:
    model.eval(); outputs = []
    for start in range(0, len(layer3), batch_size):
        maps = layer3[start:start + batch_size].to(device, non_blocking=True)
        frozen = context[start:start + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(maps, frozen, skills)
        outputs.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(outputs).numpy()


def fit(
    candidate: dict, train_layer3: torch.Tensor, train_context: torch.Tensor,
    train_y: torch.Tensor, train_soft_y: torch.Tensor,
    evaluation_layer3: torch.Tensor, evaluation_context: torch.Tensor,
    skills: torch.Tensor, config: dict, device: torch.device, seed: int,
    checkpoint_epochs: list[int],
) -> dict[int, np.ndarray]:
    seed_all(seed); model = build(candidate).to(device)
    # A full inner-fold training split is about 9.7 GiB and fits safely on an A40.
    # Keeping it resident avoids repeatedly collating 128 independent 19x19 maps
    # (roughly 190 MiB per batch) on CPU. The scientific batching rule remains a
    # seeded random permutation without replacement for every epoch.
    resident_layer3 = train_layer3.to(device)
    resident_context = train_context.to(device)
    targets = train_soft_y if candidate["target_mode"] == "mean_rater" else train_y
    resident_targets = targets.to(device)
    batch_size = int(config["frame_batch_size"])
    permutation_generator = torch.Generator().manual_seed(seed)
    groups = [{"params": model.head_parameters(), "lr": float(config["head_learning_rate"])}]
    upper = model.upper_visual_parameters()
    if upper:
        groups.append({"params": upper, "lr": float(candidate["upper_visual_learning_rate"])})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=float(config["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    requested = set(map(int, checkpoint_epochs)); output = {}
    for epoch in range(1, max(requested) + 1):
        model.train()
        permutation = torch.randperm(len(resident_layer3), generator=permutation_generator)
        for start in range(0, len(permutation), batch_size):
            positions = permutation[start:start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            maps = resident_layer3.index_select(0, positions)
            frozen = resident_context.index_select(0, positions)
            target = resident_targets.index_select(0, positions)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(maps, frozen, skills)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            scaler.step(optimizer); scaler.update()
        scheduler.step()
        if epoch in requested:
            output[epoch] = predict(
                model, evaluation_layer3, evaluation_context, skills, device,
                int(config["prediction_frame_batch_size"]),
            )
    return output


def load_development(protocol: dict):
    ids = sorted(protocol["development_video_ids"])
    if set(ids) & set(protocol["locked_confirmation_video_ids"]):
        raise ValueError("Confirmation leakage")
    layer3_audit = json.loads(Path(protocol["sources"]["layer3_feature_cache_audit"]["path"]).read_text())
    visual_audit = json.loads(Path(protocol["sources"]["visual_feature_cache_audit"]["path"]).read_text())
    dino_audit = json.loads(Path(protocol["sources"]["dinov2_feature_cache_audit"]["path"]).read_text())
    layer3_rows = {row["video_id"]: row for row in layer3_audit["videos"]}
    visual_rows = {row["video_id"]: row for row in visual_audit["videos"]}
    dino_rows = {row["video_id"]: row for row in dino_audit["videos"]}
    label_root = Path(protocol["sources"]["train_label_download_audit"]["path"]).parent / "train" / "labels"
    maps = []; contexts = []; baselines = []; ys = []; soft_ys = []
    for video_id in ids:
        layer3 = torch.load(layer3_rows[video_id]["path"], map_location="cpu", weights_only=False)
        visual = torch.load(visual_rows[video_id]["path"], map_location="cpu", weights_only=False)
        dino = torch.load(dino_rows[video_id]["path"], map_location="cpu", weights_only=False)
        if any(value.get("SAGES_test_labels_accessed") is not False for value in (layer3, visual, dino)):
            raise ValueError("Unsafe cache")
        labels = {}
        with (label_root / video_id / "frame.csv").open(newline="") as handle:
            for item in csv.DictReader(handle):
                labels[int(item["frame_id"])] = (
                    [base.majority(item, key) for key in ("c1", "c3", "c2")],
                    [sum(int(item[f"{key}_rater{i}"]) for i in (1, 2, 3)) / 3 for key in ("c1", "c3", "c2")],
                )
        frame_ids = list(map(int, visual["frame_ids"].tolist()))
        if frame_ids != list(map(int, layer3["frame_ids"].tolist())) or frame_ids != list(map(int, dino["frame_ids"].tolist())) or set(frame_ids) != set(labels):
            raise ValueError(f"Alignment: {video_id}")
        maps.append(layer3["dual_view_layer3_float16"])
        contexts.append(torch.cat([
            visual["detector_and_dual_moco_features"].half(),
            dino["dual_view_DINOv2S_features"].half(),
        ], dim=-1))
        baselines.append(visual["baseline_probability"].float())
        ys.append(torch.tensor([labels[index][0] for index in frame_ids], dtype=torch.float32))
        soft_ys.append(torch.tensor([labels[index][1] for index in frame_ids], dtype=torch.float32))
    return (
        ids, torch.stack(maps), torch.stack(contexts), torch.stack(baselines),
        torch.stack(ys), torch.stack(soft_ys),
    )


def macro_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.reshape(-1, 3); scores = scores.reshape(-1, 3)
    return float(np.mean([average_precision(labels[:, index], scores[:, index]) for index in range(3)]))


def main() -> None:
    global LAYER4_TEMPLATE, PROJECTION_TEMPLATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--outer-fold-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    if protocol["training_code_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Training code differs from frozen protocol")
    if protocol["model_code_sha256"] != sha256_file(ROOT / "cvs_assessment/cvsadapt_layer4_cvs.py"):
        raise ValueError("Model code differs from frozen protocol")
    checkpoint = Path(protocol["sources"]["official_cvsadapt_checkpoint"]["path"])
    if sha256_file(checkpoint) != protocol["sources"]["official_cvsadapt_checkpoint"]["sha256"]:
        raise ValueError("Official checkpoint changed")
    official = load_visual(checkpoint)
    LAYER4_TEMPLATE = copy.deepcopy(official.backbone.layer4)
    PROJECTION_TEMPLATE = copy.deepcopy(official.projection)
    del official

    fold = protocol["outer_folds"][args.outer_fold_index]
    ids, layer3, context, baseline, y, soft_y = load_development(protocol)
    index = {video_id: position for position, video_id in enumerate(ids)}
    take = lambda selected: torch.tensor([index[video_id] for video_id in selected], dtype=torch.long)
    # Flatten frames only after video-safe positions have been chosen.
    flatten = lambda values, positions: values[positions].reshape(-1, *values.shape[2:])
    skill_payload = torch.load(protocol["sources"]["frozen_skill_embeddings"]["path"], map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    skills = torch.stack([skill_payload["criterion_embeddings"][key] for key in CRITERIA]).to(device)
    config = protocol["training"]; epochs = list(map(int, config["checkpoint_epochs"]))
    outer_train_positions = take(fold["training_video_ids"])
    outer_train_labels = y[outer_train_positions].numpy()
    position_within_outer = {video_id: position for position, video_id in enumerate(fold["training_video_ids"])}
    records = []; stored = {}
    for candidate_index, candidate in enumerate(protocol["candidates"]):
        oof = {epoch: np.zeros((480, 18, 3), dtype=np.float32) for epoch in epochs}
        for inner in fold["inner_folds"]:
            train_positions = take(inner["training_video_ids"]); validation_positions = take(inner["validation_video_ids"])
            predictions = fit(
                candidate,
                flatten(layer3, train_positions), flatten(context, train_positions),
                flatten(y, train_positions), flatten(soft_y, train_positions),
                flatten(layer3, validation_positions), flatten(context, validation_positions),
                skills, config, device,
                int(config["seed"]) + args.outer_fold_index * 10000 + candidate_index * 100 + inner["inner_fold_index"],
                epochs,
            )
            destinations = [position_within_outer[video_id] for video_id in inner["validation_video_ids"]]
            for epoch in epochs:
                oof[epoch][destinations] = predictions[epoch].reshape(120, 18, 3)
        scores = {str(epoch): macro_ap(outer_train_labels, oof[epoch]) for epoch in epochs}
        selected_epoch = max(epochs, key=lambda epoch: (scores[str(epoch)], -epoch))
        records.append({
            "candidate_index": candidate_index, "candidate": candidate,
            "inner_OOF_mAP_by_epoch": scores, "selected_epoch": selected_epoch,
            "selected_inner_OOF_mAP": scores[str(selected_epoch)],
        })
        stored[candidate_index] = oof
        print(json.dumps({
            "outer_fold": args.outer_fold_index, "candidate": candidate_index + 1,
            "of": len(protocol["candidates"]), "selected_epoch": selected_epoch,
            "inner_OOF_mAP": scores[str(selected_epoch)],
        }), flush=True)
    selected_index = max(range(len(records)), key=lambda value: (records[value]["selected_inner_OOF_mAP"], -value))
    selected = records[selected_index]; inner_scores = stored[selected_index][selected["selected_epoch"]]
    outer_train_base = baseline[outer_train_positions].numpy(); fusion_weights = {}; fused_inner = np.empty_like(inner_scores)
    for criterion_index, criterion in enumerate(CRITERIA):
        options = [(
            average_precision(
                outer_train_labels[:, :, criterion_index].reshape(-1),
                (weight * inner_scores[:, :, criterion_index] + (1 - weight) * outer_train_base[:, :, criterion_index]).reshape(-1),
            ), float(weight),
        ) for weight in config["fusion_weights"]]
        _, weight = max(options, key=lambda item: (item[0], -item[1]))
        fusion_weights[criterion] = weight
        fused_inner[:, :, criterion_index] = weight * inner_scores[:, :, criterion_index] + (1 - weight) * outer_train_base[:, :, criterion_index]
    outer_test_positions = take(fold["test_video_ids"])
    final = fit(
        selected["candidate"],
        flatten(layer3, outer_train_positions), flatten(context, outer_train_positions),
        flatten(y, outer_train_positions), flatten(soft_y, outer_train_positions),
        flatten(layer3, outer_test_positions), flatten(context, outer_test_positions),
        skills, config, device, int(config["seed"]) + args.outer_fold_index * 10000 + 9999,
        [selected["selected_epoch"]],
    )[selected["selected_epoch"]].reshape(120, 18, 3)
    outer_base = baseline[outer_test_positions].numpy(); framework = np.empty_like(final)
    for criterion_index, criterion in enumerate(CRITERIA):
        framework[:, :, criterion_index] = fusion_weights[criterion] * final[:, :, criterion_index] + (1 - fusion_weights[criterion]) * outer_base[:, :, criterion_index]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output_dir / "outer_test_predictions.pt"
    torch.save({
        "schema_version": "sages_cvs_2024_layer4_nested_oof_outer_predictions_v1",
        "outer_fold_index": args.outer_fold_index, "video_ids": fold["test_video_ids"],
        "labels": y[outer_test_positions], "baseline_probability": baseline[outer_test_positions],
        "selected_skill_probability": torch.from_numpy(final),
        "framework_probability": torch.from_numpy(framework), "criterion_order": CRITERIA,
        "SAGES_confirmation_labels_used": False, "SAGES_test_labels_accessed": False,
        "LLM_or_MLLM_parameters_updated": False,
    }, prediction_path)
    result = {
        "schema_version": "sages_cvs_2024_layer4_nested_oof_outer_result_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "outer_fold_index": args.outer_fold_index, "inner_fold_count": 4,
        "training_video_count": 480, "test_video_count": 120,
        "candidate_records": records, "selected": selected, "fusion_weights": fusion_weights,
        "inner_selected_framework_mAP": macro_ap(outer_train_labels, fused_inner),
        "outer_test_baseline_mAP": macro_ap(y[outer_test_positions].numpy(), outer_base),
        "outer_test_framework_mAP": macro_ap(y[outer_test_positions].numpy(), framework),
        "predictions": {"path": str(prediction_path.resolve()), "sha256": sha256_file(prediction_path)},
        "SAGES_confirmation_labels_used": False, "SAGES_test_labels_accessed": False,
        "LLM_or_MLLM_parameters_updated": False,
    }
    result_path = args.output_dir / "outer_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "result": str(result_path.resolve()), "selected": selected,
        "fusion_weights": fusion_weights, "baseline_mAP": result["outer_test_baseline_mAP"],
        "framework_mAP": result["outer_test_framework_mAP"],
    }, indent=2))


if __name__ == "__main__":
    main()
