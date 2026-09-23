#!/usr/bin/env python3
"""Train the selected frozen-Skill frame or temporal head on all SAGES train videos."""
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
from cvs_assessment.detector_skill_cvs import SkillConditionedDetectorHead, SkillConditionedTemporalHead
from train_direct_interval_nested_oof import sha256_file

CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")


def majority(row: dict, key: str) -> bool:
    return sum(int(row[f"{key}_rater{i}"]) for i in (1, 2, 3)) >= 2


def build(candidate: dict) -> torch.nn.Module:
    input_dim = int(candidate["input_dim"])
    if candidate["family"] == "skill_shared_frame":
        return SkillConditionedDetectorHead(input_dim, 4096, int(candidate["hidden_dim"]), float(candidate["dropout"]))
    if candidate["family"] == "skill_shared_temporal":
        return SkillConditionedTemporalHead(input_dim, 4096, int(candidate["hidden_dim"]), float(candidate["dropout"]), tuple(map(int, candidate["dilations"])))
    raise ValueError("Only a Skill-conditioned primary model can be deployed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda:6"); args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    if protocol["training_code_sha256"] != sha256_file(Path(__file__)) or protocol["model_code_sha256"] != sha256_file(ROOT / "cvs_assessment/detector_skill_cvs.py"):
        raise ValueError("Code mismatch")
    feature_audit = json.loads(Path(protocol["sources"]["feature_cache_audit"]["path"]).read_text())
    label_root = Path(protocol["sources"]["train_label_download_audit"]["path"]).parent; features = []; labels = []
    for row in feature_audit["videos"]:
        value = torch.load(row["path"], map_location="cpu", weights_only=False); annotations = {}
        with (label_root / "train" / "labels" / value["video_id"] / "frame.csv").open(newline="") as handle:
            for item in csv.DictReader(handle):
                if protocol["model"].get("target_mode", "majority_vote") == "mean_rater": annotations[int(item["frame_id"])] = [sum(int(item[f"{key}_rater{i}"]) for i in (1, 2, 3)) / 3 for key in ("c1", "c3", "c2")]
                else: annotations[int(item["frame_id"])] = [majority(item, "c1"), majority(item, "c3"), majority(item, "c2")]
        frame_ids = list(map(int, value["frame_ids"].tolist()))
        features.append(value["detector_and_dual_moco_features"].float()); labels.append(torch.tensor([annotations[index] for index in frame_ids], dtype=torch.float32))
    x = torch.stack(features); y = torch.stack(labels); device = torch.device(args.device)
    skill_payload = torch.load(protocol["sources"]["frozen_skill_embeddings"]["path"], map_location="cpu", weights_only=False)
    skills = torch.stack([skill_payload["criterion_embeddings"][key] for key in CRITERIA]).to(device)
    config = protocol["training"]; candidate = protocol["model"]; seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); model = build(candidate).to(device)
    flat_y = y.reshape(-1, 3); positive_weight = ((len(flat_y) - flat_y.sum(0)) / flat_y.sum(0).clamp_min(1)).clamp(1, float(config["positive_weight_cap"])).to(device)
    if candidate["family"] == "skill_shared_temporal": dataset = TensorDataset(x, y); batch_size = int(config["video_batch_size"])
    else: dataset = TensorDataset(x.reshape(-1, int(candidate["input_dim"])), flat_y); batch_size = int(config["frame_batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]), eta_min=float(config["minimum_learning_rate"])); history = []
    for epoch in range(1, int(protocol["fixed_epoch"]) + 1):
        model.train(); total = count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True); logits = model(batch_x.to(device), skills)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y.to(device), pos_weight=positive_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"])); optimizer.step()
            total += loss.item() * batch_y.numel(); count += batch_y.numel()
        scheduler.step(); history.append({"epoch": epoch, "loss": total / count}); print(json.dumps(history[-1]), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False); checkpoint = args.output_dir / "full_train_skill_head.pt"
    torch.save({"schema_version": "sages_cvs_2024_small_head_full_train_v2", "model_state": model.cpu().state_dict(), "candidate": candidate, "fixed_epoch": protocol["fixed_epoch"], "fusion_weights": protocol["fusion_weights"], "criterion_order": CRITERIA, "SAGES_test_labels_used": False, "LLM_or_MLLM_parameters_updated": False}, checkpoint)
    output = {"schema_version": "sages_cvs_2024_small_head_full_train_audit_v2", "created_at": datetime.now(timezone.utc).isoformat(), "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)}, "history": history, "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}, "SAGES_test_labels_used": False, "LLM_or_MLLM_parameters_updated": False}
    audit_path = args.output_dir / "TRAINING_AUDIT.json"; audit_path.write_text(json.dumps(output, indent=2) + "\n"); print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
