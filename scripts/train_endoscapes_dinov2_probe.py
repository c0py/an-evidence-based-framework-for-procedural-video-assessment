#!/usr/bin/env python3
"""Select and retrain frozen DINOv2 dual-view CVS probes on train only."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_cholec80_validation_ablation import CRITERIA
from train_direct_interval_nested_oof import sha256_file
from train_endoscapes_text_spatial_temporal import binary_summary, choose_thresholds


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__(); self.network = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 3))
    def forward(self, value: torch.Tensor) -> torch.Tensor: return self.network(value.float())


class ResidualMlpProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 3))
        self.residual = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Dropout(dropout), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
    def forward(self, value: torch.Tensor) -> torch.Tensor: return self.linear(value.float()) + self.residual(value.float())


class Data:
    def __init__(self, protocol_path: Path) -> None:
        self.protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        source = self.protocol["sources"]["dinov2_train_feature_cache_audit"]
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError("Feature audit changed")
        audit = json.loads(path.read_text(encoding="utf-8"))
        if not audit.get("complete") or audit.get("official_val_or_test_labels_loaded") is not False:
            raise ValueError("Unsafe DINO feature cache")
        self.data = {}
        for row in audit["videos"]:
            item = Path(row["path"])
            if sha256_file(item) != row["sha256"]: raise ValueError(item)
            value = torch.load(item, map_location="cpu", weights_only=False)
            if value.get("official_val_or_test_labels_loaded") is not False: raise ValueError(item)
            self.data[int(row["video_id"])] = value
        split = self.protocol["internal_train_only_split"]
        self.train_ids = set(map(int, split["training_video_ids"])); self.dev_ids = set(map(int, split["development_video_ids"]))
        if set(self.data) != self.train_ids | self.dev_ids: raise ValueError("Coverage mismatch")

    def flatten(self, videos: set[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = []; labels = []; ids = []
        for video in sorted(videos):
            value = self.data[video]
            center = value["center_features"].float(); global_value = value["global_features"].float()
            features.append(torch.cat([center, global_value], dim=-1))
            labels.append((value["labels_soft_C1_C3_C2"] >= 0.5).float())
            ids.append(torch.full((len(center),), video, dtype=torch.long))
        return torch.cat(features), torch.cat(labels), torch.cat(ids)


def make_model(name: str, cfg: dict[str, Any], input_dim: int, device: torch.device, seed: int) -> nn.Module:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if name == "linear": model = LinearProbe(input_dim)
    elif name == "residual_mlp": model = ResidualMlpProbe(input_dim, int(cfg["hidden_dim"]), float(cfg["dropout"]))
    else: raise ValueError(name)
    return model.to(device)


@torch.inference_mode()
def score(model: nn.Module, features: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval(); output = []
    for start in range(0, len(features), batch_size):
        output.append(torch.sigmoid(model(features[start:start + batch_size].to(device))).cpu())
    return torch.cat(output).numpy()


def train_candidate(
    name: str, cfg: dict[str, Any], train_x: torch.Tensor, train_y: torch.Tensor,
    dev_x: torch.Tensor | None, dev_y: torch.Tensor | None,
    device: torch.device, seed: int, epochs: int,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any] | None]:
    model = make_model(name, cfg, train_x.shape[1], device, seed)
    positives = train_y.sum(0); pos_weight = ((len(train_y) - positives) / positives.clamp_min(1)).clamp(1, float(cfg["positive_weight_cap"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=float(cfg["minimum_learning_rate"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator)
    selection = set(map(int, cfg["selection_epochs"])); history = []; best = None
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for features, labels in loader:
            logits = model(features.to(device)); loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device), pos_weight=pos_weight[None, :])
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step(); total += float(loss.detach()) * len(labels)
        scheduler.step(); row: dict[str, Any] = {"epoch": epoch, "loss": total / len(train_y)}
        if dev_x is not None and epoch in selection:
            values = score(model, dev_x, int(cfg["batch_size"]), device)
            summary = binary_summary(dev_y.numpy(), values, {criterion: 0.5 for criterion in CRITERIA}); row["development"] = summary
            key = (summary["macro_average_precision"], summary["macro_balanced_accuracy"], summary["macro_f1"])
            if best is None or key > best["key"]: best = {"key": key, "epoch": epoch, "state": deepcopy(model.state_dict()), "development": summary}
        history.append(row)
        print(json.dumps({"candidate": name, "epoch": epoch, "loss": row["loss"], "development_mAP": row.get("development", {}).get("macro_average_precision"), "development_BA": row.get("development", {}).get("macro_balanced_accuracy")}), flush=True)
    return model, history, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["training_code_sha256"] != sha256_file(Path(__file__)): raise ValueError("Training code changed")
    args.output_dir.mkdir(parents=True, exist_ok=False); data = Data(args.protocol); cfg = protocol["training"]; device = torch.device(args.device)
    train_x, train_y, _ = data.flatten(data.train_ids); dev_x, dev_y, dev_videos = data.flatten(data.dev_ids)
    candidates = []; all_history = {}
    for offset, name in enumerate(cfg["candidates"]):
        model, history, best = train_candidate(name, cfg, train_x, train_y, dev_x, dev_y, device, int(cfg["seed"]) + offset, int(cfg["epochs"]))
        if best is None: raise RuntimeError(name)
        candidates.append({"name": name, **best}); all_history[name] = history
    selected = max(candidates, key=lambda item: item["key"]); name = selected["name"]
    selected_model = make_model(name, cfg, train_x.shape[1], device, int(cfg["seed"]) + list(cfg["candidates"]).index(name)); selected_model.load_state_dict(selected["state"])
    dev_scores = score(selected_model, dev_x, int(cfg["batch_size"]), device); thresholds = choose_thresholds(dev_y.numpy(), dev_scores, list(map(float, cfg["threshold_choices"]))); development = binary_summary(dev_y.numpy(), dev_scores, thresholds)
    gate = protocol["internal_development_gate"]; passed = development["macro_average_precision"] > float(gate["target_mAP_strictly_above"]) and development["macro_balanced_accuracy"] >= float(gate["target_balanced_accuracy_not_lower"])
    all_x, all_y, _ = data.flatten(set(data.data)); full, full_history, _ = train_candidate(name, cfg, all_x, all_y, None, None, device, int(cfg["full_retrain_seed"]), int(selected["epoch"]))
    checkpoint = args.output_dir / "full_train_deployment.pt"; torch.save({"schema_version": "endoscapes_dinov2_dual_view_probe_v1", "model_state": full.cpu().state_dict(), "candidate": name, "input_dim": int(all_x.shape[1]), "selected_epoch": int(selected["epoch"]), "thresholds": thresholds, "DINO_parameters_updated": False, "LLM_or_MLLM_parameters_updated": False, "official_val_or_test_labels_used": False}, checkpoint)
    result = {"schema_version": "endoscapes_dinov2_probe_selection_v1", "created_at": datetime.now(timezone.utc).isoformat(), "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)}, "candidate_results": [{"name": item["name"], "selected_epoch": item["epoch"], "development_at_0_5": item["development"]} for item in candidates], "selected_candidate": name, "selected_epoch": int(selected["epoch"]), "thresholds": thresholds, "selected_development": development, "internal_gate": {**gate, "passed": passed}, "history": all_history, "full_train_history": full_history, "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}, "internal_development_video_ids": sorted(map(int, np.unique(dev_videos.numpy()))), "official_val_metrics_computed": False, "official_test_reused": False, "DINO_parameters_updated": False, "LLM_or_MLLM_parameters_updated": False}
    path = args.output_dir / "train_selection_result.json"; path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(path.resolve()), "selected_candidate": name, "selected_epoch": selected["epoch"], "development": development, "internal_gate_passed": passed, "checkpoint": result["checkpoint"]}, indent=2), flush=True)


if __name__ == "__main__": main()
