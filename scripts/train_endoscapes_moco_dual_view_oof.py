#!/usr/bin/env python3
"""Generate one fixed-epoch fold of strict dual-view MoCo OOF predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import train_endoscapes_moco_visual as base
from train_endoscapes_moco_visual_full_frame import FullFrameEndoscapesFrames
from train_endoscapes_text_spatial_temporal import binary_summary


class FoldData:
    def __init__(self, protocol: dict, allowed: set[int]) -> None:
        audit_source = protocol["sources"]["train_only_feature_cache_audit"]
        audit_path = Path(audit_source["path"])
        if base.sha256_file(audit_path) != audit_source["sha256"]:
            raise ValueError("Feature audit changed after freeze")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("complete") or audit.get("official_val_or_test_labels_loaded") is not False:
            raise ValueError("Unsafe feature audit")
        self.cache = {}
        for row in audit["videos"]:
            video = int(row["video_id"])
            if video not in allowed:
                continue
            path = Path(row["path"])
            if base.sha256_file(path) != row["sha256"]:
                raise ValueError(f"Cached video changed: {path}")
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value.get("official_val_or_test_labels_loaded") is not False:
                raise ValueError("Unsafe cached video")
            self.cache[video] = value
        if set(self.cache) != allowed:
            raise ValueError("Fold cache coverage mismatch")
        self.image_root = Path(protocol["dataset_root"]) / "train"

    def class_balance(self, videos: set[int], cap: float) -> torch.Tensor:
        labels = torch.cat([self.cache[video]["labels_soft_C1_C3_C2"] >= 0.5 for video in sorted(videos)])
        positives = labels.sum(0).float(); negatives = len(labels) - positives
        return (negatives / positives.clamp_min(1)).clamp(1.0, cap)


def loader(data: FoldData, videos: set[int], view: str, training: bool, cfg: dict, seed: int) -> DataLoader:
    dataset_type = base.EndoscapesFrames if view == "center" else FullFrameEndoscapesFrames
    return DataLoader(
        dataset_type(data, videos, training), batch_size=int(cfg["batch_size"]),
        shuffle=training, num_workers=int(cfg["data_workers"]), pin_memory=True,
        persistent_workers=int(cfg["data_workers"]) > 0, drop_last=False,
        worker_init_fn=base.seed_worker, generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def predict(model, data: FoldData, videos: set[int], view: str, cfg: dict, device: torch.device):
    model.eval(); output = {}
    for video in sorted(videos):
        logits, features = [], []
        for batch in loader(data, {video}, view, False, cfg, 0):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                value = model(batch["image"].to(device, non_blocking=True))
            logits.append(value["logits"].float().cpu()); features.append(value["features"].float().cpu())
        output[video] = {"logits": torch.cat(logits), "features": torch.cat(features)}
    return output


def train_view(protocol: dict, data: FoldData, train_ids: set[int], heldout: set[int], view: str, device: torch.device, seed: int):
    cfg = protocol["training"]; epochs = int(cfg[f"{view}_epochs"])
    base.seed_everything(seed)
    model, load_audit = base.build_model(protocol, device)
    train_loader = loader(data, train_ids, view, True, cfg, seed)
    pos_weight = data.class_balance(train_ids, float(cfg["positive_weight_cap"])).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": float(cfg["backbone_learning_rate"])},
        {"params": model.classifier.parameters(), "lr": float(cfg["classifier_learning_rate"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(cfg["minimum_learning_rate"]))
    scaler = torch.amp.GradScaler("cuda"); history = []
    for epoch in range(1, epochs + 1):
        loss = base.train_epoch(model, train_loader, optimizer, scaler, device, pos_weight)
        scheduler.step(); history.append({"epoch": epoch, "loss": loss})
        print(json.dumps({"view": view, "epoch": epoch, "epochs": epochs, "loss": loss}), flush=True)
    return predict(model, data, heldout, view, cfg, device), history, load_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("training_code_sha256") != base.sha256_file(Path(__file__)):
        raise ValueError("Training code differs from frozen protocol")
    if protocol.get("model_code_sha256") != base.sha256_file(ROOT / "cvs_assessment" / "moco_visual_cvs.py"):
        raise ValueError("Model code differs from frozen protocol")
    folds = {int(row["fold"]): set(map(int, row["heldout_video_ids"])) for row in protocol["folds"]}
    if args.fold not in folds:
        raise ValueError("Unknown fold")
    outer = set(map(int, protocol["outer_internal_training_video_ids"])); forbidden = set(map(int, protocol["forbidden_internal_development_video_ids"]))
    heldout = folds[args.fold]; train_ids = outer - heldout
    if train_ids & heldout or outer & forbidden or len(train_ids) != 80 or len(heldout) != 20:
        raise ValueError("Unsafe OOF split")
    data = FoldData(protocol, outer); device = torch.device(args.device)
    seed = int(protocol["training"]["seed"]) + args.fold * 100
    center, center_history, center_load = train_view(protocol, data, train_ids, heldout, "center", device, seed)
    full, full_history, full_load = train_view(protocol, data, train_ids, heldout, "full", device, seed + 1)
    args.output_dir.mkdir(parents=True, exist_ok=False); rows = []
    for video in sorted(heldout):
        source = data.cache[video]; path = args.output_dir / f"video{video:03d}.pt"
        torch.save({
            "schema_version": "endoscapes_moco_dual_view_strict_oof_v1", "fold": args.fold,
            "video_id": video, "frame_indices": source["frame_indices"],
            "timestamps_s": source["timestamps_s"],
            "labels_C1_C3_C2": source["labels_soft_C1_C3_C2"].ge(0.5),
            "center_logits": center[video]["logits"].half(), "center_features": center[video]["features"].half(),
            "full_logits": full[video]["logits"].half(), "full_features": full[video]["features"].half(),
            "training_video_ids": sorted(train_ids), "heldout_video_ids": sorted(heldout),
            "heldout_labels_used_for_model_selection": False, "forbidden_internal_development_used": False,
            "official_val_or_test_used": False, "LLM_or_MLLM_parameters_updated": False,
        }, path)
        rows.append({"video_id": video, "path": str(path.resolve()), "sha256": base.sha256_file(path), "frame_count": len(source["frame_indices"])})
    labels = torch.cat([data.cache[v]["labels_soft_C1_C3_C2"].ge(0.5) for v in sorted(heldout)]).numpy()
    scores = torch.cat([(torch.sigmoid(center[v]["logits"]) + torch.sigmoid(full[v]["logits"])) / 2 for v in sorted(heldout)]).numpy()
    summary = binary_summary(labels, scores, {key: 0.5 for key in base.CRITERIA})
    audit = {
        "schema_version": "endoscapes_moco_dual_view_strict_oof_fold_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "fold": args.fold,
        "protocol": {"path": str(args.protocol.resolve()), "sha256": base.sha256_file(args.protocol)},
        "training_video_ids": sorted(train_ids), "heldout_video_ids": sorted(heldout),
        "videos": rows, "frame_count": sum(row["frame_count"] for row in rows),
        "raw_equal_probability_OOF_metrics_at_0_5": summary,
        "center_history": center_history, "full_history": full_history,
        "center_initialization_load_audit": center_load, "full_initialization_load_audit": full_load,
        "heldout_labels_used_for_model_selection": False, "forbidden_internal_development_used": False,
        "official_val_or_test_used": False, "LLM_or_MLLM_parameters_updated": False,
    }
    (args.output_dir / "FOLD_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fold": args.fold, "output": str(args.output_dir.resolve()), "frames": audit["frame_count"], "OOF_mAP": summary["macro_average_precision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
