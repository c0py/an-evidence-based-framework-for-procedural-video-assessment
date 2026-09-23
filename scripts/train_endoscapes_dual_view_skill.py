#!/usr/bin/env python3
"""Train dual-view PeskaVLP with frozen Skill prototypes and no LLM updates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as VF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.aligned_skill_visual_cvs import DualViewSkillInitializedCvs
from run_cholec80_validation_ablation import CRITERIA
from train_direct_interval_nested_oof import sha256_file
from train_endoscapes_text_spatial_temporal import binary_summary, choose_thresholds
from train_endoscapes_visual_skill_adaptation import TrainData, seed_everything, seed_worker


class DualViewFrames(Dataset):
    def __init__(self, data: TrainData, videos: set[int], training: bool) -> None:
        self.data = data; self.training = training
        self.rows = [
            (video, frame)
            for video in sorted(videos)
            for frame in range(len(data.cache[video]["frame_indices"]))
        ]
        self.video_counts = {
            video: len(data.cache[video]["frame_indices"]) for video in videos
        }
        self.color = v2.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.12, hue=0.015,
        )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def normalize(image: torch.Tensor) -> torch.Tensor:
        image = VF.to_dtype(image, torch.float32, scale=True)
        return VF.normalize(
            image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        )

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        video, frame = self.rows[item]; value = self.data.cache[video]
        path = self.data.image_root / value["image_names"][frame]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(path)
        image = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        if self.training:
            if torch.rand(()) < 0.5:
                image = VF.horizontal_flip(image)
            if torch.rand(()) < 0.8:
                image = self.color(image)
        global_image = VF.resize(image, (224, 384), antialias=True)
        center_image = VF.center_crop(
            VF.resize(image, (360, 640), antialias=True), (224, 224),
        )
        return {
            "center": self.normalize(center_image),
            "global": self.normalize(global_image),
            "label": (value["labels_soft_C1_C3_C2"][frame] >= 0.5).float(),
            "video": torch.tensor(video, dtype=torch.long),
        }


def make_loader(
    data: TrainData, videos: set[int], training: bool,
    cfg: dict[str, Any], seed: int,
) -> DataLoader:
    dataset = DualViewFrames(data, videos, training)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if training:
        weights = torch.tensor([
            1.0 / dataset.video_counts[video] for video, _ in dataset.rows
        ], dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights, num_samples=int(cfg["samples_per_epoch"]),
            replacement=True, generator=generator,
        )
    return DataLoader(
        dataset, batch_size=int(cfg["batch_size"]),
        shuffle=False, sampler=sampler, num_workers=int(cfg["data_workers"]),
        pin_memory=True, persistent_workers=int(cfg["data_workers"]) > 0,
        worker_init_fn=seed_worker, generator=generator, drop_last=False,
    )


def build_model(data: TrainData, protocol: dict[str, Any], device: torch.device) -> nn.Module:
    source = protocol["sources"]["frozen_peskavlp_skill_prototypes"]
    path = Path(source["path"])
    if sha256_file(path) != source["sha256"]:
        raise ValueError("Frozen Skill prototypes changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("language_parameters_updated") is not False:
        raise ValueError("Skill prototype provenance is unsafe")
    if list(payload["criterion_order"]) != list(CRITERIA):
        raise ValueError("Criterion order mismatch")
    positive_prompts = torch.stack([
        payload["prompt_embeddings"][criterion]["positive"] for criterion in CRITERIA
    ])
    negative_prompts = torch.stack([
        payload["prompt_embeddings"][criterion]["negative"] for criterion in CRITERIA
    ])
    model = DualViewSkillInitializedCvs(
        payload["positive_prototypes"], payload["negative_prototypes"],
        positive_prompts, negative_prompts,
        dropout=float(protocol["model"]["dropout"]),
    )
    visual = protocol["sources"]["official_peskavlp_checkpoint"]
    visual_path = Path(visual["path"])
    if sha256_file(visual_path) != visual["sha256"]:
        raise ValueError("PeskaVLP checkpoint changed")
    model.encoder.load_official_checkpoint(visual_path)
    model.set_visual_trainability("head_only")
    return model.to(device)


def make_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    layer4_ids = {id(value) for value in model.encoder.backbone.layer4.parameters()}
    projection_ids = {id(value) for value in model.encoder.projection.parameters()}
    lower = [
        value for value in model.encoder.parameters()
        if id(value) not in layer4_ids | projection_ids
    ]
    head = [value for name, value in model.named_parameters() if not name.startswith("encoder.")]
    return torch.optim.AdamW([
        {"params": head, "lr": float(cfg["head_learning_rate"])},
        {"params": model.encoder.projection.parameters(), "lr": float(cfg["projection_learning_rate"])},
        {"params": model.encoder.backbone.layer4.parameters(), "lr": float(cfg["layer4_learning_rate"])},
        {"params": lower, "lr": float(cfg["lower_visual_learning_rate"])},
    ], weight_decay=float(cfg["weight_decay"]))


def visual_stage(epoch: int, cfg: dict[str, Any]) -> str:
    if epoch <= int(cfg["head_only_epochs"]):
        return "head_only"
    if epoch <= int(cfg["layer4_only_until_epoch"]):
        return "layer4_projection"
    return "all_visual"


def train_epoch(
    model: DualViewSkillInitializedCvs, loader: DataLoader,
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, pos_weight: torch.Tensor, cfg: dict[str, Any],
) -> dict[str, float]:
    model.train(); totals = {key: 0.0 for key in ("loss", "classification", "contrastive", "anchor")}; count = 0
    for batch in loader:
        center = batch["center"].to(device, non_blocking=True)
        global_image = batch["global"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(center, global_image)
            main = nn.functional.binary_cross_entropy_with_logits(
                output["logits"].float(), labels, pos_weight=pos_weight[None, :],
            )
            center_loss = nn.functional.binary_cross_entropy_with_logits(
                output["center_logits"].float(), labels, pos_weight=pos_weight[None, :],
            )
            global_loss = nn.functional.binary_cross_entropy_with_logits(
                output["global_logits"].float(), labels, pos_weight=pos_weight[None, :],
            )
            classification = main + float(cfg["single_view_auxiliary_weight"]) * 0.5 * (center_loss + global_loss)
            contrastive = model.prompt_contrastive_loss(output["normalized_features"], labels)
            anchor = model.skill_anchor_loss()
            loss = classification + float(cfg["prompt_contrastive_weight"]) * contrastive + float(cfg["skill_anchor_weight"]) * anchor
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer); scaler.update()
        size = len(labels); count += size
        for key, value in {
            "loss": loss, "classification": classification,
            "contrastive": contrastive, "anchor": anchor,
        }.items():
            totals[key] += float(value.detach()) * size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); labels = []; scores = []; videos = []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(
                batch["center"].to(device, non_blocking=True),
                batch["global"].to(device, non_blocking=True),
            )
        labels.append(batch["label"]); scores.append(torch.sigmoid(output["logits"].float()).cpu()); videos.append(batch["video"])
    return torch.cat(labels).numpy(), torch.cat(scores).numpy(), torch.cat(videos).numpy()


def train_run(
    data: TrainData, videos: set[int], protocol: dict[str, Any],
    device: torch.device, seed: int, epochs: int,
    development: DataLoader | None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any] | None]:
    cfg = protocol["training"]; seed_everything(seed)
    model = build_model(data, protocol, device); optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(cfg["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda")
    loader = make_loader(data, videos, True, cfg, seed)
    pos_weight = data.class_balance(videos, float(cfg["positive_weight_cap"])).to(device)
    selection_epochs = set(map(int, cfg["selection_epochs"]))
    history = []; best = None
    for epoch in range(1, epochs + 1):
        stage = visual_stage(epoch, cfg); model.set_visual_trainability(stage)
        losses = train_epoch(model, loader, optimizer, scaler, device, pos_weight, cfg)
        scheduler.step(); row: dict[str, Any] = {"epoch": epoch, "visual_stage": stage, **losses}
        if development is not None and epoch in selection_epochs:
            labels, scores, _ = predict(model, development, device)
            summary = binary_summary(labels, scores, {criterion: 0.5 for criterion in CRITERIA})
            row["development_at_0_5"] = summary
            key = (summary["macro_average_precision"], summary["macro_balanced_accuracy"], summary["macro_f1"])
            if best is None or key > best["key"]:
                best = {"key": key, "epoch": epoch, "state": deepcopy(model.state_dict()), "development_at_0_5": summary}
        history.append(row)
        print(json.dumps({
            "stage": "internal_selection" if development is not None else "full_train",
            "epoch": epoch, "visual_stage": stage, **losses,
            "development_mAP": row.get("development_at_0_5", {}).get("macro_average_precision"),
            "development_BA": row.get("development_at_0_5", {}).get("macro_balanced_accuracy"),
        }), flush=True)
    return model, history, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["training_code_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Training code changed after protocol freeze")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = TrainData(args.protocol); cfg = protocol["training"]; device = torch.device(args.device)
    development = make_loader(data, data.dev_ids, False, cfg, int(cfg["seed"]))
    model, history, best = train_run(
        data, data.train_ids, protocol, device, int(cfg["seed"]), int(cfg["epochs"]), development,
    )
    if best is None:
        raise RuntimeError("No model selected")
    model.load_state_dict(best["state"])
    labels, scores, videos = predict(model, development, device)
    thresholds = choose_thresholds(labels, scores, list(map(float, cfg["threshold_choices"])))
    selected = binary_summary(labels, scores, thresholds)
    gate = protocol["internal_development_gate"]
    passed = selected["macro_average_precision"] > float(gate["target_mAP_strictly_above"]) and selected["macro_balanced_accuracy"] >= float(gate["target_balanced_accuracy_not_lower"])
    del model; torch.cuda.empty_cache()
    full, full_history, _ = train_run(
        data, set(data.cache), protocol, device, int(cfg["full_retrain_seed"]), int(best["epoch"]), None,
    )
    checkpoint = args.output_dir / "full_train_deployment.pt"
    torch.save({
        "schema_version": "endoscapes_dual_view_skill_visual_v1",
        "model_state": full.cpu().state_dict(), "criterion_order": list(CRITERIA),
        "selected_epoch": int(best["epoch"]), "thresholds": thresholds,
        "model_config": protocol["model"],
        "language_parameters_updated": False, "LLM_or_MLLM_parameters_updated": False,
        "official_val_or_test_labels_used": False,
    }, checkpoint)
    result = {
        "schema_version": "endoscapes_dual_view_skill_train_selection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "selected_epoch": int(best["epoch"]), "development_at_0_5": best["development_at_0_5"],
        "thresholds": thresholds, "selected_development": selected,
        "internal_gate": {**gate, "passed": passed},
        "history": history, "full_train_history": full_history,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "internal_development_video_ids": sorted(map(int, np.unique(videos))),
        "official_val_metrics_computed": False, "official_test_reused": False,
        "language_parameters_updated": False, "LLM_or_MLLM_parameters_updated": False,
    }
    path = args.output_dir / "train_selection_result.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(path.resolve()), "selected_epoch": best["epoch"],
        "development": selected, "internal_gate_passed": passed,
        "checkpoint": result["checkpoint"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
