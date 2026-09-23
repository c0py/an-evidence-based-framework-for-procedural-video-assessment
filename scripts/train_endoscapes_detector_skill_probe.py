#!/usr/bin/env python3
"""Select frozen-detector probes and safe dual-view fusion on internal dev."""
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.detector_skill_cvs import DetectorMultiLabelProbe, SkillConditionedDetectorHead
from train_endoscapes_skill_temporal_residual import paired_video_bootstrap
from train_endoscapes_text_spatial_temporal import average_precision, binary_summary, choose_thresholds

CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean([average_precision(labels[:, index], scores[:, index]) for index in range(3)]))


def load_audit(source: dict, unsafe_key: str) -> dict:
    path = Path(source["path"])
    if sha256_file(path) != source["sha256"]: raise ValueError("Cache audit changed after freeze")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not audit.get("complete") or audit.get(unsafe_key) is not False: raise ValueError("Unsafe/incomplete cache")
    return audit


def load_data(protocol: dict) -> dict[int, dict]:
    detector_audit = load_audit(protocol["sources"]["detector_feature_cache_audit"], "official_val_or_test_used")
    dual_audit = load_audit(protocol["sources"]["dual_view_cache_audit"], "official_val_or_test_loaded")
    detector_rows = {int(row["video_id"]): row for row in detector_audit["videos"]}; dual_rows = {int(row["video_id"]): row for row in dual_audit["videos"]}
    if set(detector_rows) != set(dual_rows): raise ValueError("Cache video coverage mismatch")
    output = {}
    for video in sorted(detector_rows):
        dr, mr = detector_rows[video], dual_rows[video]; dp, mp = Path(dr["path"]), Path(mr["path"])
        if sha256_file(dp) != dr["sha256"] or sha256_file(mp) != mr["sha256"]: raise ValueError("Cached video changed")
        detector = torch.load(dp, map_location="cpu", weights_only=False); dual = torch.load(mp, map_location="cpu", weights_only=False)
        if detector["image_names"] != dual["image_names"]: raise ValueError("Detector/MoCo frame alignment mismatch")
        output[video] = {
            "features": torch.cat([detector["detector_features"].float(), detector["detector_geometry"].float()], dim=1),
            "labels": detector["labels_C1_C3_C2"].float(),
            "baseline": (torch.sigmoid(dual["center_logits"].float()) + torch.sigmoid(dual["full_logits"].float())) / 2,
        }
    return output


def stack(data: dict[int, dict], videos: list[int]):
    return tuple(torch.cat([data[video][key] for video in videos]) for key in ("features", "labels", "baseline"))


@torch.inference_mode()
def predict(model, features: torch.Tensor, skills: torch.Tensor, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval(); scores = []
    for start in range(0, len(features), batch_size):
        logits = model(features[start:start + batch_size].to(device), skills)
        scores.append(torch.sigmoid(logits).cpu())
    return torch.cat(scores).numpy()


def build_model(candidate: dict, input_dim: int):
    if candidate["family"] == "skill_shared":
        return SkillConditionedDetectorHead(input_dim, 4096, int(candidate["hidden_dim"]), float(candidate["dropout"]))
    if candidate["family"] == "multilabel_probe":
        return DetectorMultiLabelProbe(input_dim, len(CRITERIA), int(candidate["hidden_dim"]), float(candidate["dropout"]))
    raise ValueError("Unknown candidate family")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["training_code_sha256"] != sha256_file(Path(__file__)) or protocol["model_code_sha256"] != sha256_file(ROOT / "cvs_assessment" / "detector_skill_cvs.py"): raise ValueError("Code differs from frozen protocol")
    data = load_data(protocol); split = protocol["internal_train_only_split"]; train_ids = list(map(int, split["training_video_ids"])); dev_ids = list(map(int, split["development_video_ids"]))
    train_x, train_y, train_base = stack(data, train_ids); dev_x, dev_y, dev_base = stack(data, dev_ids); labels = dev_y.numpy(); baseline = dev_base.numpy()
    video_ids = np.concatenate([np.full(len(data[video]["labels"]), video) for video in dev_ids])
    skill_source = protocol["sources"]["frozen_skill_embeddings"]; skill_path = Path(skill_source["path"])
    if sha256_file(skill_path) != skill_source["sha256"]: raise ValueError("Skill embeddings changed")
    skill_payload = torch.load(skill_path, map_location="cpu", weights_only=False); device = torch.device(args.device)
    skills = torch.stack([skill_payload["criterion_embeddings"][key] for key in CRITERIA]).to(device)
    cfg = protocol["training"]; seed = int(cfg["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    positives = train_y.sum(0); pos_weight = ((len(train_y) - positives) / positives.clamp_min(1)).clamp(1, float(cfg["positive_weight_cap"])).to(device)
    candidate_results = []; best_by_family = {}
    for candidate_index, candidate in enumerate(protocol["candidates"]):
        torch.manual_seed(seed + candidate_index); model = build_model(candidate, train_x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]), eta_min=float(cfg["minimum_learning_rate"]))
        dataset = TensorDataset(train_x, train_y); loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed + candidate_index))
        best = None; history = []
        for epoch in range(1, int(cfg["epochs"]) + 1):
            model.train(); total = count = 0
            for features, targets in loader:
                optimizer.zero_grad(set_to_none=True); logits = model(features.to(device), skills); loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.to(device), pos_weight=pos_weight)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"])); optimizer.step(); total += float(loss.detach()) * len(features); count += len(features)
            scheduler.step(); row = {"epoch": epoch, "loss": total / count}
            if epoch % int(cfg["selection_every_epochs"]) == 0:
                scores = predict(model, dev_x, skills, device); row["development_mAP"] = macro_ap(labels, scores)
                key = row["development_mAP"]
                if best is None or key > best["mAP"]: best = {"epoch": epoch, "mAP": key, "state": deepcopy(model.state_dict()), "scores": scores}
            history.append(row)
        assert best is not None
        result = {"candidate_index": candidate_index, "candidate": candidate, "selected_epoch": best["epoch"], "development_mAP": best["mAP"], "history": history, "state": best["state"], "scores": best["scores"]}
        candidate_results.append(result); family = candidate["family"]
        if family not in best_by_family or best["mAP"] > best_by_family[family]["development_mAP"]: best_by_family[family] = result
        print(json.dumps({"candidate": candidate, "selected_epoch": best["epoch"], "development_mAP": best["mAP"]}), flush=True)
    overall = max(candidate_results, key=lambda row: row["development_mAP"]); skill_best = best_by_family["skill_shared"]
    weights = list(map(float, cfg["fusion_weights"])); fusion_reports = {}
    for name, selected in [("best_absolute", overall), ("best_skill_shared", skill_best)]:
        scores = selected["scores"]; common_rows = [(macro_ap(labels, weight * scores + (1 - weight) * baseline), weight) for weight in weights]; common_map, common_weight = max(common_rows)
        per_weights, columns, per_aps = {}, [], {}
        for index, criterion in enumerate(CRITERIA):
            rows = [(average_precision(labels[:, index], weight * scores[:, index] + (1 - weight) * baseline[:, index]), weight) for weight in weights]; ap, weight = max(rows); per_weights[criterion] = weight; per_aps[criterion] = ap; columns.append(weight * scores[:, index] + (1 - weight) * baseline[:, index])
        fused = np.stack(columns, axis=1); fusion_reports[name] = {"selected_candidate_index": selected["candidate_index"], "candidate": selected["candidate"], "standalone_mAP": selected["development_mAP"], "common_fusion_weight_detector": common_weight, "common_fusion_mAP": common_map, "criterion_specific_detector_weights": per_weights, "criterion_specific_by_AP": per_aps, "criterion_specific_fusion_mAP": macro_ap(labels, fused), "scores": fused}
    primary = fusion_reports["best_skill_shared"]; primary_scores = primary.pop("scores"); absolute_scores = fusion_reports["best_absolute"].pop("scores")
    choices = [value / 100 for value in range(5, 96, 5)]; baseline_summary = binary_summary(labels, baseline, choose_thresholds(labels, baseline, choices)); primary_summary = binary_summary(labels, primary_scores, choose_thresholds(labels, primary_scores, choices)); absolute_summary = binary_summary(labels, absolute_scores, choose_thresholds(labels, absolute_scores, choices))
    bootstrap = paired_video_bootstrap(labels, baseline, primary_scores, video_ids, seed + 1000, 5000); gate_ref = protocol["internal_development_gate"]
    gate = {"mAP_strictly_above_reference": primary_summary["macro_average_precision"] > float(gate_ref["target_mAP_strictly_above"]), "balanced_accuracy_not_lower_reference": primary_summary["macro_balanced_accuracy"] >= float(gate_ref["target_balanced_accuracy_not_lower"]), "paired_mAP_CI_lower_above_zero": bootstrap["percentile_95_CI"][0] > 0}; gate["passed_all"] = all(gate.values())
    args.output_dir.mkdir(parents=True, exist_ok=False); checkpoint = args.output_dir / "selected_skill_shared_model.pt"; selected_model = build_model(skill_best["candidate"], train_x.shape[1]); selected_model.load_state_dict(skill_best["state"])
    torch.save({"schema_version": "endoscapes_detector_skill_probe_internal_selection_v1", "model_state": selected_model.state_dict(), "candidate": skill_best["candidate"], "selected_epoch": skill_best["selected_epoch"], "criterion_order": list(CRITERIA), "detector_parameters_updated": False, "skill_text_embeddings_updated": False, "official_val_or_test_labels_used": False, "LLM_or_MLLM_parameters_updated": False}, checkpoint)
    serial_candidates = [{k: v for k, v in row.items() if k not in {"state", "scores"}} for row in candidate_results]
    result = {"schema_version": "endoscapes_detector_skill_probe_train_selection_v1", "created_at": datetime.now(timezone.utc).isoformat(), "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)}, "candidate_results": serial_candidates, "fusion_reports": fusion_reports, "baseline_raw_dual_view": baseline_summary, "primary_skill_shared_fusion": primary_summary, "best_absolute_fusion": absolute_summary, "paired_video_bootstrap_primary_minus_baseline_mAP": bootstrap, "internal_gate": gate, "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}, "internal_development_video_ids": sorted(dev_ids), "detector_parameters_updated": False, "skill_text_embeddings_updated": False, "official_val_metrics_computed": False, "official_test_reused": False, "LLM_or_MLLM_parameters_updated": False}
    path = args.output_dir / "train_selection_result.json"; path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"result": str(path.resolve()), "primary": primary_summary, "absolute": absolute_summary, "gate": gate}, indent=2), flush=True)


if __name__ == "__main__": main()
