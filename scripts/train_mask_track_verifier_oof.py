#!/usr/bin/env python3
"""Fixed-protocol video-OOF audit of text-conditioned mask-track evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.boundary_supervision import validate_train_only_scope
from cvs_assessment.visual_adapters import TextConditionedMaskTrackVerifier
from train_dense_residual_fusion_oof import state_at
from train_multiframe_roi_verifier_oof import CRITERIA, binary_report


STATE_TARGET = {0: 0.0, 1: 0.25, 2: 1.0}
STATE_MASS = {0: 0.30, 1: 0.20, 2: 0.50}


def report(
    records: list[dict], indices: list[int], states: torch.Tensor,
    scores: np.ndarray,
) -> dict:
    output = {}
    for criterion in CRITERIA:
        current = [
            index for index in indices
            if records[index]["criterion"] == criterion and int(states[index]) in {0, 2}
        ]
        labels = [int(states[index] == 2) for index in current]
        values = [float(scores[index]) for index in current]
        current_report = binary_report(labels, values)
        positives_missed_by_m2c = [
            index for index in current
            if states[index] == 2
            and float(records[index]["m2c_probability"]) < float(records[index]["m2c_threshold"])
        ]
        hard_negatives = [
            index for index in current
            if states[index] == 0
            and float(records[index]["m2c_probability"]) >= float(records[index]["m2c_threshold"])
        ]
        current_report.update({
            "state2_missed_by_m2c_count": len(positives_missed_by_m2c),
            "mask_score_mean_on_state2_missed_by_m2c": (
                float(np.mean([scores[index] for index in positives_missed_by_m2c]))
                if positives_missed_by_m2c else None
            ),
            "mask_recall_at_0_5_on_state2_missed_by_m2c": (
                float(np.mean([scores[index] >= 0.5 for index in positives_missed_by_m2c]))
                if positives_missed_by_m2c else None
            ),
            "m2c_hard_negative_count": len(hard_negatives),
            "mask_fpr_at_0_5_on_m2c_hard_negatives": (
                float(np.mean([scores[index] >= 0.5 for index in hard_negatives]))
                if hard_negatives else None
            ),
        })
        output[criterion] = current_report
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-features", type=Path, required=True)
    parser.add_argument("--qwen-feature-cache", type=Path, required=True)
    parser.add_argument("--fold-definition-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=419)
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_train_only_scope(
        split_train_ids=split["train_video_ids"],
        split_development_ids=split["validation_video_ids"],
        split_test_ids=split["test_video_ids"],
        artifact_video_ids=manifest["train_video_ids"],
        artifact_test_ids_loaded=manifest.get("test_video_ids_loaded", []),
        artifact_test_labels_accessed=bool(manifest.get("test_labels_accessed", False)),
    )
    mask_cache = torch.load(args.mask_features, map_location="cpu", weights_only=False)
    qwen_cache = torch.load(args.qwen_feature_cache, map_location="cpu", weights_only=False)
    if mask_cache["manifest_payload_sha256"] != manifest["payload_sha256"]:
        raise ValueError("Mask feature/manifest mismatch")
    if qwen_cache["manifest_payload_sha256"] != manifest["payload_sha256"]:
        raise ValueError("Qwen feature/manifest mismatch")
    if mask_cache.get("annotation_files_loaded") is not False:
        raise ValueError("Mask features are not annotation-free")
    records = list(manifest["records"])
    if mask_cache["reference_ids"] != [row["reference_id"] for row in records]:
        raise ValueError("Mask feature rows are misaligned")
    features = mask_cache["features"].float()
    embeddings = {
        name: value.float() for name, value in qwen_cache["criterion_embeddings"].items()
    }

    train_ids = set(map(int, split["train_video_ids"]))
    annotation_path = Path(split["dataset_root"]) / "annotations" / "cholec80-CVS.xlsx"
    annotations = {video_id: load_cvs_intervals(annotation_path, video_id) for video_id in train_ids}
    labeled_ids = {
        video_id for video_id in train_ids
        if any(annotations[video_id][criterion] for criterion in CRITERIA)
    }
    states = torch.tensor([
        state_at(
            annotations[int(row["video_id"])][row["criterion"]],
            float(row["center_time_s"]),
        ) for row in records
    ], dtype=torch.long)
    definition_paths = sorted(args.fold_definition_dir.glob("fold*_best.pt"))
    folds = [
        sorted(map(int, torch.load(path, map_location="cpu", weights_only=False)["heldout_video_ids"]))
        for path in definition_paths
    ]
    if len(folds) != 5 or {video for fold in folds for video in fold} != labeled_ids:
        raise ValueError("Invalid video-fold definition")

    device = torch.device(args.device)
    probabilities = np.zeros(len(records), dtype=np.float64)
    fold_by_video, fold_audits = {}, []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for fold_index, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_indices = [
            index for index, row in enumerate(records)
            if int(row["video_id"]) in labeled_ids and int(row["video_id"]) not in heldout_set
        ]
        valid_indices = [
            index for index, row in enumerate(records) if int(row["video_id"]) in heldout_set
        ]
        seed = args.seed + fold_index
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = TextConditionedMaskTrackVerifier(
            mask_feature_dim=features.shape[1], hidden_dim=args.hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=3e-3,
        )
        groups = Counter((
            int(states[index]), int(records[index]["video_id"]), records[index]["criterion"],
        ) for index in train_indices)
        state_videos = {
            state: len({
                int(records[index]["video_id"]) for index in train_indices if int(states[index]) == state
            }) for state in (0, 1, 2)
        }
        weights = torch.tensor([
            STATE_MASS[int(states[index])]
            / max(1, state_videos[int(states[index])])
            / groups[(int(states[index]), int(records[index]["video_id"]), records[index]["criterion"])]
            * (
                2.0 if states[index] == 0
                and float(records[index]["m2c_probability"]) >= float(records[index]["m2c_threshold"])
                else 1.0
            )
            for index in train_indices
        ], dtype=torch.float64)
        generator = torch.Generator().manual_seed(seed * 1000)
        losses = []
        model.train()
        for step in range(args.steps):
            sampled = torch.multinomial(
                weights, args.batch_size, replacement=True, generator=generator,
            ).tolist()
            current = [train_indices[position] for position in sampled]
            index = torch.tensor(current, dtype=torch.long)
            text = torch.stack([embeddings[records[i]["criterion"]] for i in current])
            logits = model(features[index].to(device), text.to(device))
            labels = torch.tensor(
                [STATE_TARGET[int(states[i])] for i in current], device=device,
            )
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(valid_indices), args.batch_size):
                current = valid_indices[start:start + args.batch_size]
                index = torch.tensor(current, dtype=torch.long)
                text = torch.stack([embeddings[records[i]["criterion"]] for i in current])
                values = torch.sigmoid(model(features[index].to(device), text.to(device)))
                probabilities[index.numpy()] = values.cpu().numpy()
        checkpoint_path = args.output_dir / f"fold{fold_index}_fixed.pt"
        torch.save({
            "schema_version": "text_conditioned_mask_track_verifier_v1",
            "model_state": model.state_dict(), "mask_feature_dim": int(features.shape[1]),
            "hidden_dim": args.hidden_dim, "heldout_video_ids": heldout,
            "fixed_steps": args.steps, "seed": seed, "base_seed": int(args.seed),
            "foundation_model_parameters_updated": False,
            "development_or_test_accessed": False,
        }, checkpoint_path)
        for video_id in heldout:
            fold_by_video[video_id] = f"fold{fold_index}"
        fold_audits.append({
            "fold": f"fold{fold_index}", "heldout_video_ids": heldout,
            "mean_train_loss": float(np.mean(losses)),
            "checkpoint": str(checkpoint_path.resolve()),
        })
        print(
            f"mask_track_oof fold={fold_index} train={len(train_indices)} valid={len(valid_indices)}",
            flush=True,
        )

    labeled_indices = [
        index for index, row in enumerate(records) if int(row["video_id"]) in labeled_ids
    ]
    mask_report = report(records, labeled_indices, states, probabilities)
    m2c_scores = np.asarray([float(row["m2c_probability"]) for row in records])
    m2c_report = report(records, labeled_indices, states, m2c_scores)
    gate_by_criterion = {}
    for criterion in CRITERIA:
        current = mask_report[criterion]
        prevalence = current["positives"] / max(1, current["n"])
        passed = bool(
            current["roc_auc"] is not None and current["roc_auc"] >= 0.60
            and current["average_precision"] is not None
            and current["average_precision"] >= 1.5 * prevalence
            and current["mask_recall_at_0_5_on_state2_missed_by_m2c"] is not None
            and current["mask_recall_at_0_5_on_state2_missed_by_m2c"] >= 0.10
        )
        gate_by_criterion[criterion] = {
            "passed": passed, "predeclared_auc_floor": 0.60,
            "predeclared_ap_to_prevalence_ratio": 1.5,
            "predeclared_m2c_miss_recall_floor": 0.10,
        }
    output = {
        "schema_version": "mask_track_evidence_train_oof_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "fixed_step_grouped_video_oof_training_only",
        "foundation_model_parameters_updated": False,
        "mask_track_report": mask_report, "m2c_frame_report": m2c_report,
        "gate_by_criterion": gate_by_criterion,
        "proceed_to_frozen_mllm_evidence_pack": sum(
            item["passed"] for item in gate_by_criterion.values()
        ) >= 2,
        "fold_audits": fold_audits,
        "base_seed": int(args.seed),
        "development_or_test_accessed": False,
    }
    output_path = args.output_dir / "oof_audit.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    rows = [{
        "reference_id": row["reference_id"], "video_id": int(row["video_id"]),
        "criterion": row["criterion"], "center_time_s": float(row["center_time_s"]),
        "outer_oof_fold": fold_by_video[int(row["video_id"])],
        "mask_track_probability": float(probabilities[index]),
    } for index, row in enumerate(records) if int(row["video_id"]) in labeled_ids]
    (args.output_dir / "predictions_oof.json").write_text(
        json.dumps({
            "schema_version": "mask_track_evidence_oof_predictions_v1",
            "annotation_files_loaded_during_prediction": False,
            "training_annotations_loaded_for_training_only": True,
            "foundation_model_parameters_updated": False,
            "development_or_test_accessed": False, "base_seed": int(args.seed), "rows": rows,
        }, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_path.resolve()),
        "gate": gate_by_criterion,
        "proceed": output["proceed_to_frozen_mllm_evidence_pack"],
        "development_or_test_accessed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
