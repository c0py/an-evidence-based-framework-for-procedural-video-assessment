#!/usr/bin/env python3
"""Train a strong dual-view visual baseline with a safe frozen-Skill residual."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.safe_residual_dual_view_cvs import SafeResidualDualViewCvs
from run_cholec80_validation_ablation import CRITERIA
from train_direct_interval_nested_oof import sha256_file
from train_endoscapes_dual_view_skill import DualViewFrames
from train_endoscapes_text_spatial_temporal import binary_summary, choose_thresholds
from train_endoscapes_visual_skill_adaptation import TrainData, seed_everything, seed_worker


def make_loader(
    data: TrainData, videos: set[int], training: bool,
    cfg: dict[str, Any], seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        DualViewFrames(data, videos, training),
        batch_size=int(cfg["batch_size"]), shuffle=training,
        num_workers=int(cfg["data_workers"]), pin_memory=True,
        persistent_workers=int(cfg["data_workers"]) > 0,
        worker_init_fn=seed_worker, generator=generator, drop_last=False,
    )


def build_model(data: TrainData, protocol: dict[str, Any], device: torch.device) -> SafeResidualDualViewCvs:
    source = protocol["sources"]["frozen_peskavlp_skill_prototypes"]
    path = Path(source["path"])
    if sha256_file(path) != source["sha256"]:
        raise ValueError("Skill prototypes changed after freeze")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("language_parameters_updated") is not False:
        raise ValueError("Unsafe Skill provenance")
    model = SafeResidualDualViewCvs(
        payload["positive_prototypes"], payload["negative_prototypes"],
        dropout=float(protocol["model"]["dropout"]),
    )
    visual = protocol["sources"]["official_peskavlp_checkpoint"]
    visual_path = Path(visual["path"])
    if sha256_file(visual_path) != visual["sha256"]:
        raise ValueError("PeskaVLP checkpoint changed after freeze")
    model.encoder.load_official_checkpoint(visual_path)
    model.set_visual_trainability("head_only")
    return model.to(device)


def visual_stage(epoch: int, cfg: dict[str, Any]) -> str:
    if epoch <= int(cfg["head_only_epochs"]):
        return "head_only"
    if epoch <= int(cfg["layer4_only_until_epoch"]):
        return "layer4_projection"
    return "all_visual"


def make_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    layer4_ids = {id(value) for value in model.encoder.backbone.layer4.parameters()}
    projection_ids = {id(value) for value in model.encoder.projection.parameters()}
    lower = [value for value in model.encoder.parameters() if id(value) not in layer4_ids | projection_ids]
    head = [value for name, value in model.named_parameters() if not name.startswith("encoder.")]
    return torch.optim.AdamW([
        {"params": head, "lr": float(cfg["head_learning_rate"])},
        {"params": model.encoder.projection.parameters(), "lr": float(cfg["projection_learning_rate"])},
        {"params": model.encoder.backbone.layer4.parameters(), "lr": float(cfg["layer4_learning_rate"])},
        {"params": lower, "lr": float(cfg["lower_visual_learning_rate"])},
    ], weight_decay=float(cfg["weight_decay"]))


def train_epoch(
    model: SafeResidualDualViewCvs, loader: DataLoader,
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    device: torch.device, pos_weight: torch.Tensor, cfg: dict[str, Any],
) -> dict[str, float]:
    model.train(); totals = {"loss": 0.0, "framework": 0.0, "visual": 0.0}; count = 0
    for batch in loader:
        center = batch["center"].to(device, non_blocking=True)
        global_image = batch["global"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(center, global_image)
            framework = nn.functional.binary_cross_entropy_with_logits(
                output["logits"].float(), labels, pos_weight=pos_weight[None, :],
            )
            visual = nn.functional.binary_cross_entropy_with_logits(
                output["visual_logits"].float(), labels, pos_weight=pos_weight[None, :],
            )
            center_loss = nn.functional.binary_cross_entropy_with_logits(
                output["center_visual_logits"].float(), labels, pos_weight=pos_weight[None, :],
            )
            global_loss = nn.functional.binary_cross_entropy_with_logits(
                output["global_visual_logits"].float(), labels, pos_weight=pos_weight[None, :],
            )
            loss = framework + float(cfg["visual_primary_weight"]) * visual
            loss = loss + float(cfg["single_view_auxiliary_weight"]) * 0.5 * (center_loss + global_loss)
            loss = loss + float(cfg["skill_gate_l1_weight"]) * output["skill_gate"].abs().mean()
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer); scaler.update()
        size = len(labels); count += size
        for key, value in {"loss": loss, "framework": framework, "visual": visual}.items():
            totals[key] += float(value.detach()) * size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); labels = []; framework = []; visual = []; videos = []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(
                batch["center"].to(device, non_blocking=True),
                batch["global"].to(device, non_blocking=True),
            )
        labels.append(batch["label"]); videos.append(batch["video"])
        framework.append(torch.sigmoid(output["logits"].float()).cpu())
        visual.append(torch.sigmoid(output["visual_logits"].float()).cpu())
    return torch.cat(labels).numpy(), torch.cat(framework).numpy(), torch.cat(visual).numpy(), torch.cat(videos).numpy()


def safe_summary(labels: np.ndarray, framework: np.ndarray, visual: np.ndarray) -> dict[str, Any]:
    threshold = {criterion: 0.5 for criterion in CRITERIA}
    framework_summary = binary_summary(labels, framework, threshold)
    visual_summary = binary_summary(labels, visual, threshold)
    skill_safe = (
        framework_summary["macro_average_precision"] >= visual_summary["macro_average_precision"]
        and framework_summary["macro_balanced_accuracy"] >= visual_summary["macro_balanced_accuracy"]
    )
    return {
        "framework": framework_summary,
        "visual_baseline": visual_summary,
        "skill_residual_safe": skill_safe,
        "selected_mode": "framework" if skill_safe else "visual_baseline",
        "selected": framework_summary if skill_safe else visual_summary,
    }


def train_run(
    data: TrainData, videos: set[int], protocol: dict[str, Any], device: torch.device,
    seed: int, epochs: int, development: DataLoader | None,
) -> tuple[SafeResidualDualViewCvs, list[dict[str, Any]], dict[str, Any] | None]:
    cfg = protocol["training"]; seed_everything(seed)
    model = build_model(data, protocol, device); optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(cfg["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda"); loader = make_loader(data, videos, True, cfg, seed)
    pos_weight = data.class_balance(videos, float(cfg["positive_weight_cap"])).to(device)
    selection_epochs = set(map(int, cfg["selection_epochs"])); history = []; best = None
    for epoch in range(1, epochs + 1):
        stage = visual_stage(epoch, cfg); model.set_visual_trainability(stage)
        losses = train_epoch(model, loader, optimizer, scaler, device, pos_weight, cfg); scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, "visual_stage": stage, **losses}
        if development is not None and epoch in selection_epochs:
            labels, framework, visual, _ = predict(model, development, device)
            summary = safe_summary(labels, framework, visual); row["development"] = summary
            selected = summary["selected"]
            key = (selected["macro_average_precision"], selected["macro_balanced_accuracy"], selected["macro_f1"])
            if best is None or key > best["key"]:
                best = {"key": key, "epoch": epoch, "state": deepcopy(model.state_dict()), "development": summary}
        history.append(row)
        selected = row.get("development", {}).get("selected", {})
        print(json.dumps({
            "stage": "internal_selection" if development is not None else "full_train",
            "epoch": epoch, "visual_stage": stage, **losses,
            "development_mAP": selected.get("macro_average_precision"),
            "development_BA": selected.get("macro_balanced_accuracy"),
            "selected_mode": row.get("development", {}).get("selected_mode"),
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
    model, history, best = train_run(data, data.train_ids, protocol, device, int(cfg["seed"]), int(cfg["epochs"]), development)
    if best is None:
        raise RuntimeError("No model selected")
    model.load_state_dict(best["state"])
    labels, framework, visual, videos = predict(model, development, device)
    mode = best["development"]["selected_mode"]
    if mode == "visual_baseline":
        model.skill_gate.data.zero_(); selected_scores = visual
    else:
        selected_scores = framework
    thresholds = choose_thresholds(labels, selected_scores, list(map(float, cfg["threshold_choices"])))
    selected = binary_summary(labels, selected_scores, thresholds)
    gate = protocol["internal_development_gate"]
    passed = selected["macro_average_precision"] > float(gate["target_mAP_strictly_above"]) and selected["macro_balanced_accuracy"] >= float(gate["target_balanced_accuracy_not_lower"])
    del model; torch.cuda.empty_cache()
    full, full_history, _ = train_run(data, set(data.cache), protocol, device, int(cfg["full_retrain_seed"]), int(best["epoch"]), None)
    if mode == "visual_baseline":
        full.skill_gate.data.zero_()
    checkpoint = args.output_dir / "full_train_deployment.pt"
    torch.save({
        "schema_version": "endoscapes_safe_residual_dual_view_v1",
        "model_state": full.cpu().state_dict(), "criterion_order": list(CRITERIA),
        "selected_epoch": int(best["epoch"]), "selected_mode": mode, "thresholds": thresholds,
        "model_config": protocol["model"], "language_parameters_updated": False,
        "LLM_or_MLLM_parameters_updated": False, "official_val_or_test_labels_used": False,
    }, checkpoint)
    result = {
        "schema_version": "endoscapes_safe_residual_dual_view_train_selection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "selected_epoch": int(best["epoch"]), "selected_mode": mode,
        "development_at_0_5": best["development"], "thresholds": thresholds,
        "selected_development": selected, "internal_gate": {**gate, "passed": passed},
        "history": history, "full_train_history": full_history,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "internal_development_video_ids": sorted(map(int, np.unique(videos))),
        "official_val_metrics_computed": False, "official_test_reused": False,
        "language_parameters_updated": False, "LLM_or_MLLM_parameters_updated": False,
    }
    path = args.output_dir / "train_selection_result.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(path.resolve()), "selected_epoch": best["epoch"], "selected_mode": mode,
        "development": selected, "internal_gate_passed": passed, "checkpoint": result["checkpoint"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
