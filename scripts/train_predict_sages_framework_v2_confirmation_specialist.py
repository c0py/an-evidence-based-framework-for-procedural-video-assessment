#!/usr/bin/env python3
"""Train the frozen SAGES small specialist on 700 train videos and predict 60 test videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.tool_reliability import fit_high_precision_tool_policy  # noqa: E402
from scripts.run_sages_nested_oof_outer_fold import CRITERIA, fit  # noqa: E402
from scripts.train_predict_sages_locked_confirmation_component import label_values  # noqa: E402
from train_direct_interval_nested_oof import sha256_file  # noqa: E402


def audit_index(path: Path, expected_count: int) -> dict[str, dict]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (
        not audit.get("complete") or int(audit.get("video_count", -1)) != expected_count
        or audit.get("SAGES_test_labels_accessed") is not False
    ):
        raise ValueError(f"Unsafe or incomplete feature audit: {path}")
    rows = {str(row["video_id"]): row for row in audit["videos"]}
    if len(rows) != expected_count:
        raise ValueError("Duplicate feature-cache video ids")
    return rows


def load_training(
    rows: dict[str, dict], label_root: Path,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = sorted(rows); features = []; labels = []; soft = []
    for video_id in ids:
        payload = torch.load(rows[video_id]["path"], map_location="cpu", weights_only=False)
        if payload.get("SAGES_test_labels_accessed") is not False:
            raise ValueError("Unsafe SAGES train feature")
        frame_ids = list(map(int, payload["frame_ids"].tolist()))
        y, soft_y = label_values(label_root, video_id, frame_ids)
        features.append(payload["detector_and_dual_moco_features"].float())
        labels.append(y); soft.append(soft_y)
    return ids, torch.stack(features), torch.stack(labels), torch.stack(soft)


def load_test(rows: dict[str, dict]) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    ids = sorted(rows); features = []; baseline = []
    for video_id in ids:
        payload = torch.load(rows[video_id]["path"], map_location="cpu", weights_only=False)
        if payload.get("SAGES_test_labels_accessed") is not False:
            raise ValueError("Unsafe SAGES confirmation feature")
        features.append(payload["detector_and_dual_moco_features"].float())
        baseline.append(payload["baseline_probability"].float())
    return ids, torch.stack(features), torch.stack(baseline)


def reliability_from_oof(
    oof_root: Path, locked_prediction: Path, label_root: Path,
) -> dict[str, dict]:
    all_labels = []; all_scores = []
    for path in sorted(oof_root.glob("outer_fold_*/outer_test_predictions.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("SAGES_test_labels_accessed") is not False:
            raise ValueError("Unsafe OOF source")
        all_labels.append(payload["labels"].float())
        all_scores.append(payload["framework_probability"].float())
    locked = torch.load(locked_prediction, map_location="cpu", weights_only=False)
    if locked.get("SAGES_test_labels_accessed") is not False:
        raise ValueError("Unsafe locked-confirmation prediction source")
    locked_labels = []
    for video_id in locked["video_ids"]:
        # All official-train videos have the same 18 fixed frame ids.
        y, _ = label_values(label_root, str(video_id), list(range(0, 2700, 150)))
        locked_labels.append(y)
    all_labels.append(torch.stack(locked_labels))
    all_scores.append(locked["framework_probability"].float())
    labels = torch.cat(all_labels).numpy(); scores = torch.cat(all_scores).numpy()
    if labels.shape != (700, 18, 3) or scores.shape != labels.shape:
        raise ValueError("Reliability calibration must contain all 700 train videos")
    return {
        criterion: fit_high_precision_tool_policy(
            labels[:, :, index], scores[:, :, index],
            minimum_precision=0.80, minimum_accepted_count=25,
        ).to_dict()
        for index, criterion in enumerate(CRITERIA)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--deployment-selection", type=Path, required=True)
    parser.add_argument("--train-feature-audit", type=Path, required=True)
    parser.add_argument("--test-feature-audit", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--locked-old-prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    training_protocol = json.loads(args.training_protocol.read_text(encoding="utf-8"))
    deployment = json.loads(args.deployment_selection.read_text(encoding="utf-8"))
    if (
        deployment.get("selected_without_confirmation_labels") is not True
        or deployment.get("SAGES_test_labels_accessed") is not False
    ):
        raise ValueError("Unsafe deployment selection")
    train_rows = audit_index(args.train_feature_audit, 700)
    test_rows = audit_index(args.test_feature_audit, 60)
    label_root = Path(training_protocol["sources"]["train_label_download_audit"]["path"]).parent / "train" / "labels"
    train_ids, train_x, train_y, train_soft = load_training(train_rows, label_root)
    test_ids, test_x, baseline = load_test(test_rows)
    skills_payload = torch.load(
        training_protocol["sources"]["frozen_skill_embeddings"]["path"],
        map_location="cpu", weights_only=False,
    )
    device = torch.device(args.device)
    skills = torch.stack([skills_payload["criterion_embeddings"][key] for key in CRITERIA]).to(device)
    candidate = deployment["candidate"]
    fixed_epoch = int(deployment["fixed_epoch"])
    if candidate.get("family") != "skill_shared_frame" or fixed_epoch != 40:
        raise ValueError("Unexpected frozen SAGES deployment recipe")
    selected = fit(
        candidate, train_x, train_y, train_soft, test_x, skills,
        training_protocol["training"], device, 20321300, [fixed_epoch],
    )[fixed_epoch]
    framework = np.empty_like(selected)
    for index, criterion in enumerate(CRITERIA):
        weight = float(deployment["fusion_weights"][criterion])
        framework[:, :, index] = weight * selected[:, :, index] + (1 - weight) * baseline.numpy()[:, :, index]
    reliability = reliability_from_oof(
        args.oof_root, args.locked_old_prediction, label_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "sages_framework_v2_confirmation_specialist_predictions_v1",
        "video_ids": test_ids, "criterion_order": CRITERIA,
        "baseline_probability": baseline,
        "selected_skill_probability": torch.from_numpy(selected),
        "framework_probability": torch.from_numpy(framework),
        "reliability_policies_from_700_labelled_train_videos": reliability,
        "training_video_ids": train_ids, "training_video_count": 700,
        "fixed_recipe": {"candidate": candidate, "fixed_epoch": fixed_epoch,
                         "fusion_weights": deployment["fusion_weights"], "seed": 20321300},
        "sources": {
            "training_protocol_sha256": sha256_file(args.training_protocol),
            "deployment_selection_sha256": sha256_file(args.deployment_selection),
            "train_feature_audit_sha256": sha256_file(args.train_feature_audit),
            "test_feature_audit_sha256": sha256_file(args.test_feature_audit),
            "locked_old_prediction_sha256": sha256_file(args.locked_old_prediction),
        },
        "SAGES_test_labels_accessed": False,
        "LLM_or_MLLM_parameters_updated": False,
    }, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()), "sha256": sha256_file(args.output),
        "training_videos": 700, "confirmation_videos": 60,
        "SAGES_test_labels_accessed": False,
        "LLM_or_MLLM_parameters_updated": False,
    }, indent=2))


if __name__ == "__main__":
    main()

