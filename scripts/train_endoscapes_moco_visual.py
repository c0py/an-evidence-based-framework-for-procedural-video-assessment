#!/usr/bin/env python3
"""Train an Endoscapes train-only MoCo ResNet-50 visual anchor."""
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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.moco_visual_cvs import MocoResNet50Cvs, load_pretrained_backbone
from run_cholec80_validation_ablation import CRITERIA
from train_direct_interval_nested_oof import sha256_file
from train_endoscapes_text_spatial_temporal import binary_summary, choose_thresholds


class TrainData:
    def __init__(self, protocol_path: Path) -> None:
        self.protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if self.protocol.get("official_val_CVS_metrics_computed_during_freeze") is not False:
            raise ValueError("Unsafe protocol: official validation was opened")
        audit_source = self.protocol["sources"]["train_only_feature_cache_audit"]
        audit_path = Path(audit_source["path"])
        if sha256_file(audit_path) != audit_source["sha256"]:
            raise ValueError("Feature-cache audit changed after protocol freeze")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("complete") or audit.get("official_val_or_test_labels_loaded") is not False:
            raise ValueError("Unsafe feature cache")
        self.cache: dict[int, dict[str, Any]] = {}
        for row in audit["videos"]:
            path = Path(row["path"])
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"Feature cache changed: {path}")
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value.get("official_val_or_test_labels_loaded") is not False:
                raise ValueError(f"Non-training labels found in {path}")
            self.cache[int(row["video_id"])] = value
        split = self.protocol["internal_train_only_split"]
        self.train_ids = set(map(int, split["training_video_ids"]))
        self.dev_ids = set(map(int, split["development_video_ids"]))
        if set(self.cache) != self.train_ids | self.dev_ids:
            raise ValueError("Cache does not exactly cover official training videos")
        self.image_root = Path(self.protocol["dataset"]["root"]) / "train"

    def class_balance(self, videos: set[int], cap: float) -> torch.Tensor:
        labels = torch.cat([
            self.cache[video]["labels_soft_C1_C3_C2"] >= 0.5
            for video in sorted(videos)
        ])
        positives = labels.sum(0).float()
        negatives = len(labels) - positives
        return (negatives / positives.clamp_min(1)).clamp(1.0, cap)


class EndoscapesFrames(Dataset):
    def __init__(self, data: TrainData, videos: set[int], training: bool) -> None:
        self.data = data
        self.rows = [
            (video, frame)
            for video in sorted(videos)
            for frame in range(len(data.cache[video]["frame_indices"]))
        ]
        transforms: list[nn.Module] = [
            v2.Resize((360, 640), antialias=True),
            v2.CenterCrop((224, 224)),
        ]
        if training:
            transforms.extend([
                v2.RandomHorizontalFlip(0.5),
                v2.RandomAffine(degrees=8.0, translate=(0.03, 0.03)),
                v2.RandomApply([
                    v2.ColorJitter(
                        brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02,
                    ),
                ], p=0.8),
            ])
        transforms.extend([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        self.transform = v2.Compose(transforms)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        video, frame = self.rows[index]
        value = self.data.cache[video]
        image_path = self.data.image_root / value["image_names"][frame]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read {image_path}")
        image = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        return {
            "image": self.transform(image),
            "label": (value["labels_soft_C1_C3_C2"][frame] >= 0.5).float(),
            "video": torch.tensor(video, dtype=torch.long),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    data: TrainData,
    videos: set[int],
    training: bool,
    cfg: dict[str, Any],
    seed: int,
) -> DataLoader:
    return DataLoader(
        EndoscapesFrames(data, videos, training),
        batch_size=int(cfg["batch_size"]),
        shuffle=training,
        num_workers=int(cfg["data_workers"]),
        pin_memory=True,
        persistent_workers=int(cfg["data_workers"]) > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def build_model(protocol: dict[str, Any], device: torch.device) -> tuple[MocoResNet50Cvs, dict[str, Any]]:
    model = MocoResNet50Cvs(
        num_classes=len(CRITERIA), dropout=float(protocol["model"]["dropout"]),
    )
    source = protocol["sources"]["visual_initialization"]
    path = Path(source["path"])
    if sha256_file(path) != source["sha256"]:
        raise ValueError("Visual initialization changed after protocol freeze")
    audit = load_pretrained_backbone(model, path, str(source["format"]))
    return model.to(device), audit


def train_epoch(
    model: MocoResNet50Cvs,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> float:
    model.train()
    total = count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)["logits"]
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.float(), labels.float(), pos_weight=pos_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(images)
        count += len(images)
    return total / max(count, 1)


@torch.inference_mode()
def predict(
    model: MocoResNet50Cvs,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, scores, videos = [], [], []
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch["image"].to(device, non_blocking=True))["logits"]
        labels.append(batch["label"])
        scores.append(torch.sigmoid(logits.float()).cpu())
        videos.append(batch["video"])
    return (
        torch.cat(labels).numpy(),
        torch.cat(scores).numpy(),
        torch.cat(videos).numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("training_code_sha256") != sha256_file(Path(__file__)):
        raise ValueError("Training code differs from frozen protocol")
    model_path = ROOT / "cvs_assessment" / "moco_visual_cvs.py"
    if protocol.get("model_code_sha256") != sha256_file(model_path):
        raise ValueError("Model code differs from frozen protocol")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    data = TrainData(args.protocol)
    cfg = protocol["training"]
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    model, load_audit = build_model(protocol, device)
    train_loader = make_loader(data, data.train_ids, True, cfg, seed)
    dev_loader = make_loader(data, data.dev_ids, False, cfg, seed)
    pos_weight = data.class_balance(
        data.train_ids, float(cfg["positive_weight_cap"]),
    ).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": float(cfg["backbone_learning_rate"])},
        {"params": model.classifier.parameters(), "lr": float(cfg["classifier_learning_rate"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["epochs"]), eta_min=float(cfg["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda")
    selection_epochs = set(map(int, cfg["selection_epochs"]))
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for epoch in range(1, int(cfg["epochs"]) + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, device, pos_weight)
        scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, "loss": loss}
        if epoch in selection_epochs:
            labels, scores, _ = predict(model, dev_loader, device)
            summary = binary_summary(
                labels, scores, {criterion: 0.5 for criterion in CRITERIA},
            )
            row["development_at_0_5"] = summary
            key = (
                summary["macro_average_precision"],
                summary["macro_balanced_accuracy"],
                summary["macro_f1"],
            )
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "epoch": epoch,
                    "state": deepcopy(model.state_dict()),
                    "development_at_0_5": summary,
                }
        history.append(row)
        print(json.dumps({
            "epoch": epoch,
            "loss": loss,
            "development_mAP": row.get("development_at_0_5", {}).get("macro_average_precision"),
            "development_BA": row.get("development_at_0_5", {}).get("macro_balanced_accuracy"),
        }), flush=True)
    if best is None:
        raise RuntimeError("No selection epoch was evaluated")

    model.load_state_dict(best["state"])
    labels, scores, videos = predict(model, dev_loader, device)
    thresholds = choose_thresholds(
        labels, scores, list(map(float, cfg["threshold_choices"])),
    )
    selected = binary_summary(labels, scores, thresholds)
    gate = protocol["internal_development_gate"]
    gate_passed = (
        selected["macro_average_precision"] > float(gate["target_mAP_strictly_above"])
        and selected["macro_balanced_accuracy"] >= float(gate["target_balanced_accuracy_not_lower"])
    )
    checkpoint_path = args.output_dir / "selected_internal_model.pt"
    torch.save({
        "schema_version": "endoscapes_moco_visual_internal_selection_v1",
        "model_state": model.cpu().state_dict(),
        "criterion_order": list(CRITERIA),
        "selected_epoch": int(best["epoch"]),
        "thresholds_internal_development_only": thresholds,
        "initialization_load_audit": load_audit,
        "official_val_or_test_labels_used": False,
        "LLM_or_MLLM_parameters_updated": False,
    }, checkpoint_path)
    result = {
        "schema_version": "endoscapes_moco_visual_train_selection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "initialization_load_audit": load_audit,
        "selected_epoch": int(best["epoch"]),
        "development_at_0_5": best["development_at_0_5"],
        "thresholds": thresholds,
        "selected_development": selected,
        "internal_gate": {**gate, "passed": gate_passed},
        "history": history,
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
        "internal_development_video_ids": sorted(map(int, np.unique(videos))),
        "official_val_metrics_computed": False,
        "official_test_reused": False,
        "LLM_or_MLLM_parameters_updated": False,
    }
    result_path = args.output_dir / "train_selection_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(result_path.resolve()),
        "selected_epoch": best["epoch"],
        "development": selected,
        "internal_gate_passed": gate_passed,
        "checkpoint": result["checkpoint"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
