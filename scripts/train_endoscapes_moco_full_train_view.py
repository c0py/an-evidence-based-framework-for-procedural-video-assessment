#!/usr/bin/env python3
"""Train one selected MoCo view on all official Endoscapes train videos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import train_endoscapes_moco_visual as base
from train_endoscapes_moco_dual_view_oof import loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--view", choices=("center", "full"), required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["training_code_sha256"] != base.sha256_file(Path(__file__)) or protocol["model_code_sha256"] != base.sha256_file(ROOT / "cvs_assessment" / "moco_visual_cvs.py"): raise ValueError("Code differs from frozen protocol")
    data = base.TrainData(args.protocol); videos = set(map(int, protocol["official_train_video_ids"]))
    if set(data.cache) != videos: raise ValueError("Training cache does not exactly cover official train")
    cfg = protocol["training"]; seed = int(cfg["seed"]) + (0 if args.view == "center" else 1); epochs = int(cfg[f"{args.view}_epochs"]); base.seed_everything(seed); device = torch.device(args.device)
    model, load_audit = base.build_model(protocol, device); train_loader = loader(data, videos, args.view, True, cfg, seed); pos_weight = data.class_balance(videos, float(cfg["positive_weight_cap"])).to(device)
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": float(cfg["backbone_learning_rate"])}, {"params": model.classifier.parameters(), "lr": float(cfg["classifier_learning_rate"])}], weight_decay=float(cfg["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(cfg["minimum_learning_rate"])); scaler = torch.amp.GradScaler("cuda"); history = []
    for epoch in range(1, epochs + 1):
        loss = base.train_epoch(model, train_loader, optimizer, scaler, device, pos_weight); scheduler.step(); history.append({"epoch": epoch, "loss": loss}); print(json.dumps({"view": args.view, "epoch": epoch, "epochs": epochs, "loss": loss}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False); checkpoint = args.output_dir / "full_train_model.pt"
    torch.save({"schema_version": "endoscapes_moco_full_train_view_v1", "view": args.view, "model_state": model.cpu().state_dict(), "criterion_order": list(base.CRITERIA), "fixed_epoch": epochs, "initialization_load_audit": load_audit, "official_train_video_ids": sorted(videos), "official_val_or_test_labels_used": False, "LLM_or_MLLM_parameters_updated": False}, checkpoint)
    audit = {"schema_version": "endoscapes_moco_full_train_view_audit_v1", "created_at": datetime.now(timezone.utc).isoformat(), "protocol": {"path": str(args.protocol.resolve()), "sha256": base.sha256_file(args.protocol)}, "view": args.view, "fixed_epoch": epochs, "history": history, "checkpoint": {"path": str(checkpoint.resolve()), "sha256": base.sha256_file(checkpoint)}, "official_train_video_ids": sorted(videos), "official_val_or_test_used": False, "LLM_or_MLLM_parameters_updated": False}
    (args.output_dir / "TRAINING_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8"); print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__": main()
