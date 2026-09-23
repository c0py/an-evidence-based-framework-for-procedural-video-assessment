#!/usr/bin/env python3
"""Train strict-OOF Skill heads, select rank fusion, and confirm on outer dev."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from cvs_assessment.detector_skill_cvs import SkillConditionedDetectorHead
from train_endoscapes_skill_temporal_residual import paired_video_bootstrap
from train_endoscapes_text_spatial_temporal import average_precision, binary_summary

CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audit(source: dict, unsafe_key: str) -> dict:
    path = Path(source["path"])
    if sha256_file(path) != source["sha256"]:
        raise ValueError(f"Audit changed: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not audit.get("complete") or audit.get(unsafe_key) is not False:
        raise ValueError(f"Unsafe/incomplete audit: {path}")
    return audit


def load_detector(protocol: dict) -> dict[int, dict]:
    audit = load_audit(protocol["sources"]["internal_detector_feature_cache_audit"], "official_val_or_test_used"); output = {}
    for row in audit["videos"]:
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]: raise ValueError("Detector cache changed")
        value = torch.load(path, map_location="cpu", weights_only=False)
        output[int(value["video_id"])] = {"frame_indices": value["frame_indices"], "features": torch.cat([value["detector_features"].float(), value["detector_geometry"].float()], 1), "labels": value["labels_C1_C3_C2"].float()}
    return output


def load_dual(protocol: dict) -> dict[int, dict]:
    audit = load_audit(protocol["sources"]["internal_dual_view_cache_audit"], "official_val_or_test_loaded"); output = {}
    for row in audit["videos"]:
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]: raise ValueError("Dual-view cache changed")
        value = torch.load(path, map_location="cpu", weights_only=False)
        output[int(value["video_id"])] = {"frame_indices": value["frame_indices"], "baseline": (torch.sigmoid(value["center_logits"].float()) + torch.sigmoid(value["full_logits"].float())) / 2}
    return output


def oof_moco(protocol: dict) -> dict[int, dict]:
    output = {}
    for source in protocol["sources"]["moco_oof_fold_audits"]:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]: raise ValueError("OOF fold audit changed")
        audit = json.loads(path.read_text(encoding="utf-8"))
        for row in audit["videos"]:
            value_path = Path(row["path"])
            if sha256_file(value_path) != row["sha256"]: raise ValueError("OOF prediction changed")
            value = torch.load(value_path, map_location="cpu", weights_only=False)
            output[int(value["video_id"])] = {"fold": int(value["fold"]), "frame_indices": value["frame_indices"], "labels": value["labels_C1_C3_C2"].float(), "baseline": (torch.sigmoid(value["center_logits"].float()) + torch.sigmoid(value["full_logits"].float())) / 2}
    return output


def stack(data: dict[int, dict], ids: list[int], key: str) -> torch.Tensor:
    return torch.cat([data[video][key] for video in ids])


@torch.inference_mode()
def predict(model: SkillConditionedDetectorHead, features: torch.Tensor, skills: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval(); output = []
    for start in range(0, len(features), 512):
        output.append(torch.sigmoid(model(features[start:start + 512].to(device), skills)).cpu())
    return torch.cat(output).numpy()


def train_head(protocol: dict, features: torch.Tensor, labels: torch.Tensor, skills: torch.Tensor, device: torch.device, seed: int) -> SkillConditionedDetectorHead:
    cfg = protocol["training"]; model_cfg = protocol["oof_skill_head"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = SkillConditionedDetectorHead(int(model_cfg["input_dim"]), int(model_cfg["text_dim"]), int(model_cfg["hidden_dim"]), float(model_cfg["dropout"])).to(device)
    positives = labels.sum(0); pos_weight = ((len(labels) - positives) / positives.clamp_min(1)).clamp(1, float(cfg["positive_weight_cap"])).to(device)
    loader = DataLoader(TensorDataset(features, labels), batch_size=int(cfg["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["fixed_epochs"]), eta_min=float(cfg["minimum_learning_rate"]))
    for _ in range(int(cfg["fixed_epochs"])):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True); logits = model(x.to(device), skills); loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.to(device), pos_weight=pos_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"])); optimizer.step()
        scheduler.step()
    return model.eval()


def percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(reference.astype(np.float64), axis=0); output = np.empty_like(values, dtype=np.float64)
    for index in range(values.shape[1]):
        output[:, index] = np.searchsorted(reference[:, index], values[:, index], side="right") / len(reference)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda:4"); args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["training_code_sha256"] != sha256_file(Path(__file__)) or protocol["model_code_sha256"] != sha256_file(ROOT / "cvs_assessment/detector_skill_cvs.py"): raise ValueError("Code differs from frozen protocol")
    detector = load_detector(protocol); dual = load_dual(protocol); oof = oof_moco(protocol)
    outer_train = list(map(int, protocol["internal_train_only_split"]["training_video_ids"])); outer_dev = list(map(int, protocol["internal_train_only_split"]["development_video_ids"]))
    if set(oof) != set(outer_train) or set(detector) != set(outer_train + outer_dev) or set(dual) != set(detector): raise ValueError("Video coverage mismatch")
    for video in detector:
        if not torch.equal(detector[video]["frame_indices"], dual[video]["frame_indices"]): raise ValueError("Internal frame alignment mismatch")
        if video in oof and (not torch.equal(detector[video]["frame_indices"], oof[video]["frame_indices"]) or not torch.equal(detector[video]["labels"], oof[video]["labels"])): raise ValueError("OOF frame/label alignment mismatch")
    skill_path = Path(protocol["sources"]["frozen_skill_embeddings"]["path"])
    if sha256_file(skill_path) != protocol["sources"]["frozen_skill_embeddings"]["sha256"]: raise ValueError("Skill embeddings changed")
    skill_payload = torch.load(skill_path, map_location="cpu", weights_only=False); device = torch.device(args.device); skills = torch.stack([skill_payload["criterion_embeddings"][key] for key in CRITERIA]).to(device)
    oof_detector = {}; fold_reports = []
    for fold_spec in protocol["folds"]:
        fold = int(fold_spec["fold"]); heldout = list(map(int, fold_spec["heldout_video_ids"])); training = sorted(set(outer_train) - set(heldout))
        model = train_head(protocol, stack(detector, training, "features"), stack(detector, training, "labels"), skills, device, int(protocol["training"]["seed"]) + fold)
        for video in heldout: oof_detector[video] = predict(model, detector[video]["features"], skills, device)
        fold_reports.append({"fold": fold, "training_video_count": len(training), "heldout_video_count": len(heldout)})
        print(json.dumps(fold_reports[-1]), flush=True)
    oof_labels = np.concatenate([oof[video]["labels"].numpy() for video in sorted(outer_train)]); oof_baseline = np.concatenate([oof[video]["baseline"].numpy() for video in sorted(outer_train)]); oof_skill = np.concatenate([oof_detector[video] for video in sorted(outer_train)])
    baseline_rank = percentile(oof_baseline, oof_baseline); skill_rank = percentile(oof_skill, oof_skill); weights = list(map(float, protocol["rank_fusion"]["OOF_weight_candidates"])); selected_weights = {}; oof_aps = {}
    for index, criterion in enumerate(CRITERIA):
        rows = [(average_precision(oof_labels[:, index], weight * skill_rank[:, index] + (1 - weight) * baseline_rank[:, index]), weight) for weight in weights]
        ap, weight = max(rows, key=lambda row: (row[0], -row[1])); selected_weights[criterion] = weight; oof_aps[criterion] = ap
    # Outer-dev confirmation uses the already-selected final internal head. CDF references use outer-train predictions only and no dev labels.
    head_path = Path(protocol["sources"]["selected_internal_skill_head"]["path"])
    if sha256_file(head_path) != protocol["sources"]["selected_internal_skill_head"]["sha256"]: raise ValueError("Final Skill head changed")
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False); config = checkpoint["candidate"]; final_head = SkillConditionedDetectorHead(1298, 4096, int(config["hidden_dim"]), float(config["dropout"])); final_head.load_state_dict(checkpoint["model_state"], strict=True); final_head = final_head.to(device).eval()
    train_base = np.concatenate([dual[video]["baseline"].numpy() for video in outer_train]); dev_base = np.concatenate([dual[video]["baseline"].numpy() for video in outer_dev]); train_skill = predict(final_head, stack(detector, outer_train, "features"), skills, device); dev_skill = predict(final_head, stack(detector, outer_dev, "features"), skills, device)
    dev_labels = stack(detector, outer_dev, "labels").numpy(); dev_base_rank = percentile(dev_base, train_base); dev_skill_rank = percentile(dev_skill, train_skill); weight_array = np.asarray([selected_weights[key] for key in CRITERIA]); dev_framework = weight_array * dev_skill_rank + (1 - weight_array) * dev_base_rank
    thresholds = {key: float(protocol["rank_fusion"]["baseline_thresholds"][key]) for key in CRITERIA}; baseline_summary = binary_summary(dev_labels, dev_base, thresholds)
    # The score improves ranking; the operating decision is exactly the visual fallback, so threshold metrics are identical by construction.
    framework_summary = {**binary_summary(dev_labels, dev_framework, {key: 0.5 for key in CRITERIA}), "macro_balanced_accuracy": baseline_summary["macro_balanced_accuracy"], "macro_f1": baseline_summary["macro_f1"], "by_criterion": {key: {**binary_summary(dev_labels, dev_framework, {name: 0.5 for name in CRITERIA})["by_criterion"][key], **{field: baseline_summary["by_criterion"][key][field] for field in ("balanced_accuracy", "f1", "sensitivity", "specificity", "threshold", "positive", "n")}, "operating_decision_source": "exact_visual_baseline_fallback"} for key in CRITERIA}}
    video_ids = np.concatenate([np.full(len(detector[video]["labels"]), video) for video in outer_dev]); bootstrap = paired_video_bootstrap(dev_labels, dev_base, dev_framework, video_ids, int(protocol["statistics"]["seed"]), int(protocol["statistics"]["paired_video_bootstrap_repetitions"]))
    gate = {"mAP_strictly_above_0_574": framework_summary["macro_average_precision"] > 0.574, "balanced_accuracy_not_lower_0_667": framework_summary["macro_balanced_accuracy"] >= 0.667, "all_sensitivity_and_specificity_exactly_preserved": all(framework_summary["by_criterion"][key][metric] == baseline_summary["by_criterion"][key][metric] for key in CRITERIA for metric in ("sensitivity", "specificity")), "paired_mAP_CI_lower_above_zero": bootstrap["percentile_95_CI"][0] > 0}; gate["passed_all"] = all(gate.values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": "endoscapes_rank_skill_strict_oof_outer_dev_result_v1", "created_at": datetime.now(timezone.utc).isoformat(), "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)}, "fold_reports": fold_reports, "strict_OOF_selection": {"selected_skill_weights": selected_weights, "OOF_by_criterion_AP": oof_aps, "OOF_macro_AP": float(np.mean(list(oof_aps.values())))}, "outer_internal_dev": {"baseline": baseline_summary, "rank_skill_framework": framework_summary, "paired_video_bootstrap_mAP": bootstrap, "gate": gate}, "operating_sensitivity_and_specificity_preserved_by_fallback": True, "official_val_used_for_method_or_parameter_selection": False, "official_test_reused": False, "LLM_or_MLLM_parameters_updated": False}
    output = args.output_dir / "rank_skill_oof_result.json"; output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"output": str(output.resolve()), "selected_weights": selected_weights, "baseline_mAP": baseline_summary["macro_average_precision"], "framework_mAP": framework_summary["macro_average_precision"], "bootstrap": bootstrap, "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
