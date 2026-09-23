#!/usr/bin/env python3
"""Train a frozen-Skill-conditioned temporal residual on internal train videos."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.dual_view_skill_temporal_residual import (
    DualViewSkillTemporalResidual, centered_probability_mean,
)
from train_endoscapes_text_spatial_temporal import average_precision, binary_summary, choose_thresholds

CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_data(protocol: dict) -> dict[int, dict]:
    source = protocol["sources"]["dual_view_cache_audit"]
    path = Path(source["path"])
    if sha256_file(path) != source["sha256"]:
        raise ValueError("Cache audit changed after protocol freeze")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not audit.get("complete") or audit.get("official_val_or_test_loaded") is not False:
        raise ValueError("Unsafe visual cache")
    output = {}
    for row in audit["videos"]:
        item_path = Path(row["path"])
        if sha256_file(item_path) != row["sha256"]:
            raise ValueError(f"Cached video changed: {item_path}")
        value = torch.load(item_path, map_location="cpu", weights_only=False)
        if value.get("official_val_or_test_loaded") is not False:
            raise ValueError("Unsafe cached video")
        output[int(value["video_id"])] = value
    expected = set(map(int, protocol["internal_train_only_split"]["training_video_ids"] + protocol["internal_train_only_split"]["development_video_ids"]))
    if set(output) != expected:
        raise ValueError("Cache does not exactly cover the frozen internal split")
    return output


def baseline_logits(value: dict) -> torch.Tensor:
    probabilities = (torch.sigmoid(value["center_logits"].float()) + torch.sigmoid(value["full_logits"].float())) / 2
    return centered_probability_mean(torch.logit(probabilities.clamp(1e-5, 1 - 1e-5)), radius=4)


@torch.inference_mode()
def predict(
    model: DualViewSkillTemporalResidual, data: dict[int, dict], video_ids: list[int],
    skills: torch.Tensor, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); labels, baseline, learned, videos = [], [], [], []
    for video in sorted(video_ids):
        value = data[video]
        base = baseline_logits(value).to(device)
        output = model(
            value["center_features"].to(device), value["full_features"].to(device),
            base, skills,
        )["logits"]
        count = len(base)
        labels.append(value["labels_C1_C3_C2"].float())
        baseline.append(torch.sigmoid(base).cpu())
        learned.append(torch.sigmoid(output).cpu())
        videos.append(torch.full((count,), video, dtype=torch.long))
    return (
        torch.cat(labels).numpy(), torch.cat(baseline).numpy(),
        torch.cat(learned).numpy(), torch.cat(videos).numpy(),
    )


def macro_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean([average_precision(labels[:, index], scores[:, index]) for index in range(labels.shape[1])]))


def paired_video_bootstrap(
    labels: np.ndarray, baseline: np.ndarray, learned: np.ndarray, videos: np.ndarray,
    seed: int, repetitions: int = 5000,
) -> dict:
    unique = np.unique(videos); rng = np.random.default_rng(seed); deltas = []
    indices = {video: np.flatnonzero(videos == video) for video in unique}
    for _ in range(repetitions):
        sampled = rng.choice(unique, len(unique), replace=True)
        chosen = np.concatenate([indices[video] for video in sampled])
        deltas.append(macro_ap(labels[chosen], learned[chosen]) - macro_ap(labels[chosen], baseline[chosen]))
    values = np.asarray(deltas)
    return {
        "unit": "video", "repetitions": repetitions, "seed": seed,
        "absolute_delta": macro_ap(labels, learned) - macro_ap(labels, baseline),
        "percentile_95_CI": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "bootstrap_probability_delta_positive": float(np.mean(values > 0)),
    }


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
    model_path = ROOT / "cvs_assessment" / "dual_view_skill_temporal_residual.py"
    if protocol.get("model_code_sha256") != sha256_file(model_path):
        raise ValueError("Model code differs from frozen protocol")
    if protocol.get("official_val_or_test_used") is not False:
        raise ValueError("Unsafe protocol")
    data = load_data(protocol)
    split = protocol["internal_train_only_split"]
    train_ids = list(map(int, split["training_video_ids"])); dev_ids = list(map(int, split["development_video_ids"]))
    skill_source = protocol["sources"]["frozen_skill_embeddings"]
    skill_path = Path(skill_source["path"])
    if sha256_file(skill_path) != skill_source["sha256"]:
        raise ValueError("Skill artifact changed after protocol freeze")
    skill_payload = torch.load(skill_path, map_location="cpu", weights_only=False)
    if skill_payload.get("foundation_model_parameters_updated") is not False:
        raise ValueError("Skill source updated foundation parameters")
    if skill_payload["criterion_order"] != list(CRITERIA):
        raise ValueError("Skill criterion order mismatch")
    device = torch.device(args.device)
    skills = torch.stack([skill_payload["criterion_embeddings"][key] for key in CRITERIA]).to(device)
    cfg = protocol["training"]; model_cfg = protocol["model"]
    seed = int(cfg["seed"]); seed_everything(seed)
    model = DualViewSkillTemporalResidual(
        visual_dim=int(model_cfg["visual_dim"]), text_dim=int(model_cfg["text_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dilations=tuple(map(int, model_cfg["dilations"])), dropout=float(model_cfg["dropout"]),
    ).to(device)
    all_train_labels = torch.cat([data[video]["labels_C1_C3_C2"] for video in train_ids]).float()
    positives = all_train_labels.sum(0); negatives = len(all_train_labels) - positives
    pos_weight = (negatives / positives.clamp_min(1)).clamp(1, float(cfg["positive_weight_cap"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]), eta_min=1e-6)
    labels, base_scores, initial_scores, videos = predict(model, data, dev_ids, skills, device)
    threshold_choices = [value / 100 for value in range(5, 96, 5)]
    baseline_thresholds = choose_thresholds(labels, base_scores, threshold_choices)
    baseline_summary = binary_summary(labels, base_scores, baseline_thresholds)
    best = {
        "epoch": 0, "key": (macro_ap(labels, initial_scores), baseline_summary["macro_balanced_accuracy"], baseline_summary["macro_f1"]),
        "state": deepcopy(model.state_dict()), "scores": initial_scores,
    }
    history = [{"epoch": 0, "loss": None, "development_macro_average_precision": best["key"][0], "fallback_baseline": True}]
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train(); shuffled = train_ids.copy(); random.shuffle(shuffled); total_loss = total_frames = 0
        for video in shuffled:
            value = data[video]; base = baseline_logits(value).to(device); targets = value["labels_C1_C3_C2"].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                value["center_features"].to(device), value["full_features"].to(device), base, skills,
            )
            classification = torch.nn.functional.binary_cross_entropy_with_logits(
                output["logits"], targets, pos_weight=pos_weight,
            )
            residual = output["residual"]
            smoothness = (residual[1:] - residual[:-1]).abs().mean() if len(residual) > 1 else residual.new_zeros(())
            loss = classification + float(cfg["residual_first_difference_weight"]) * smoothness
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"])); optimizer.step()
            total_loss += float(loss.detach()) * len(targets); total_frames += len(targets)
        scheduler.step()
        row = {"epoch": epoch, "loss": total_loss / total_frames}
        if epoch % int(cfg["selection_every_epochs"]) == 0:
            labels, _, learned_scores, videos = predict(model, data, dev_ids, skills, device)
            thresholds = choose_thresholds(labels, learned_scores, threshold_choices)
            summary = binary_summary(labels, learned_scores, thresholds)
            key = (summary["macro_average_precision"], summary["macro_balanced_accuracy"], summary["macro_f1"])
            row["development_macro_average_precision"] = key[0]
            row["development_macro_balanced_accuracy"] = key[1]
            row["development_macro_f1"] = key[2]
            if key > best["key"]:
                best = {"epoch": epoch, "key": key, "state": deepcopy(model.state_dict()), "scores": learned_scores}
        history.append(row); print(json.dumps(row), flush=True)
    model.load_state_dict(best["state"])
    labels, base_scores, learned_scores, videos = predict(model, data, dev_ids, skills, device)
    learned_thresholds = choose_thresholds(labels, learned_scores, threshold_choices)
    learned_summary = binary_summary(labels, learned_scores, learned_thresholds)
    bootstrap = paired_video_bootstrap(labels, base_scores, learned_scores, videos, seed + 1000)
    sensitivity_deltas = {
        key: learned_summary["by_criterion"][key]["sensitivity"] - baseline_summary["by_criterion"][key]["sensitivity"]
        for key in CRITERIA
    }
    specificity_deltas = {
        key: learned_summary["by_criterion"][key]["specificity"] - baseline_summary["by_criterion"][key]["specificity"]
        for key in CRITERIA
    }
    official_gate = protocol["internal_development_gate"]
    gate = {
        "mAP_strictly_above_reference": learned_summary["macro_average_precision"] > float(official_gate["target_mAP_strictly_above"]),
        "balanced_accuracy_not_lower_reference": learned_summary["macro_balanced_accuracy"] >= float(official_gate["target_balanced_accuracy_not_lower"]),
        "all_criterion_sensitivity_not_lower_than_fallback": min(sensitivity_deltas.values()) >= -1e-12,
        "all_criterion_specificity_not_lower_than_fallback": min(specificity_deltas.values()) >= -1e-12,
        "paired_mAP_CI_lower_above_zero": bootstrap["percentile_95_CI"][0] > 0,
    }
    gate["passed_all"] = all(gate.values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir / "selected_internal_model.pt"
    torch.save({
        "schema_version": "endoscapes_skill_temporal_residual_internal_selection_v1",
        "model_state": model.cpu().state_dict(), "selected_epoch": int(best["epoch"]),
        "criterion_order": list(CRITERIA), "thresholds_internal_development_only": learned_thresholds,
        "visual_backbone_parameters_updated": False, "skill_text_embeddings_updated": False,
        "LLM_or_MLLM_parameters_updated": False, "official_val_or_test_labels_used": False,
    }, checkpoint)
    result = {
        "schema_version": "endoscapes_skill_temporal_residual_train_selection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "selected_epoch": int(best["epoch"]), "baseline_centered9": baseline_summary,
        "selected_development": learned_summary, "paired_video_bootstrap_mAP": bootstrap,
        "sensitivity_delta_selected_minus_baseline": sensitivity_deltas,
        "specificity_delta_selected_minus_baseline": specificity_deltas,
        "paper_internal_gate": gate, "history": history,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "internal_development_video_ids": sorted(dev_ids),
        "visual_backbone_parameters_updated": False, "skill_text_embeddings_updated": False,
        "LLM_or_MLLM_parameters_updated": False, "official_val_metrics_computed": False,
        "official_test_reused": False,
    }
    result_path = args.output_dir / "train_selection_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path.resolve()), "selected_epoch": best["epoch"], "baseline": baseline_summary, "selected": learned_summary, "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
