#!/usr/bin/env python3
"""Train a recipe-conditioned coarse-step/action fact head on official features.

Run this script with ``.venv-qwen3vl/bin/python``.  It reads only the separated
fact train/val manifests and labels; it never reads test data or final error
targets.  ``--check-only`` performs dependency and input checks without
training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/captaincook4d_adaptation_v1"
FEATURE_ROOT = ROOT / "incoming/captaincook4d_features/3dresnet_1s_selected"
CHECKPOINT = ROOT / "models/captaincook4d_fact_head_v1.pt"
TRAINING_RECORD = RUN / "FACT_HEAD_TRAINING.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="The fact head is small; CPU is the deterministic default.",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--training-record", type=Path, default=TRAINING_RECORD)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_joined(paths: list[Path]) -> str:
    return hashlib.sha256(
        "\n".join(f"{path.name}:{sha256_file(path)}" for path in paths).encode()
    ).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_aligned(inputs_path: Path, labels_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = json.loads(inputs_path.read_text())
    labels = json.loads(labels_path.read_text())
    if [row["sample_key"] for row in inputs] != [row["sample_key"] for row in labels]:
        raise RuntimeError(f"input/label order mismatch for {inputs_path.name}")
    forbidden = {"has_errors", "is_error", "error_probability", "verdict", "final_answer"}
    if any(forbidden & set(row) for row in inputs):
        raise RuntimeError(f"target field leaked into {inputs_path.name}")
    return inputs, labels


def feature_path(row: dict[str, Any]) -> Path:
    return FEATURE_ROOT / row["split"] / f"{row['recording_id']}.npz"


def pooled_feature(row: dict[str, Any]) -> tuple[np.ndarray, float]:
    path = feature_path(row)
    with np.load(path) as archive:
        if "arr_0" not in archive:
            raise RuntimeError(f"official feature key arr_0 missing from {path.name}")
        values = np.asarray(archive["arr_0"], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise RuntimeError(f"unexpected feature shape {values.shape} for {path.name}")
    requested_start = int(row["feature_start_index_inclusive"])
    requested_end = int(row["feature_end_index_exclusive"])
    actual_start = min(max(requested_start, 0), values.shape[0])
    actual_end = min(max(requested_end, actual_start), values.shape[0])
    if actual_end <= actual_start:
        raise RuntimeError(f"empty feature window for {row['sample_key']}")
    clip = values[actual_start:actual_end]
    norms = np.linalg.norm(clip, axis=1, keepdims=True)
    clip = clip / np.maximum(norms, 1e-8)
    pooled = np.concatenate([clip.mean(axis=0), clip.max(axis=0)]).astype(np.float32)
    requested_count = max(1, requested_end - requested_start)
    coverage = (actual_end - actual_start) / requested_count
    return pooled, float(min(1.0, coverage))


def class_weights(labels: np.ndarray, class_count: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    weights = np.zeros_like(counts)
    weights[counts > 0] = 1.0 / np.sqrt(counts[counts > 0])
    weights *= class_count / max(weights.sum(), 1e-8)
    return torch.from_numpy(weights)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> float:
    values = []
    for index in range(class_count):
        tp = int(np.sum((y_true == index) & (y_pred == index)))
        fp = int(np.sum((y_true != index) & (y_pred == index)))
        fn = int(np.sum((y_true == index) & (y_pred != index)))
        denominator = 2 * tp + fp + fn
        if denominator:
            values.append(2 * tp / denominator)
    return float(np.mean(values)) if values else 0.0


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    total = len(labels)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index + 1 < bins else confidence <= high)
        if mask.any():
            accuracy = np.mean(prediction[mask] == labels[mask])
            value += float(mask.sum() / total * abs(accuracy - confidence[mask].mean()))
    return value


def recipe_mask(activity_indices: torch.Tensor, step_activity_indices: torch.Tensor) -> torch.Tensor:
    return activity_indices[:, None] == step_activity_indices[None, :]


class FactHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, step_count: int, verb_count: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.step_head = nn.Linear(hidden_dim, step_count)
        self.verb_head = nn.Linear(hidden_dim, verb_count)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.step_head(hidden), self.verb_head(hidden)


def apply_step_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e4)


def best_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    candidates = torch.linspace(0.5, 3.0, 101)
    losses = torch.stack([nn.functional.cross_entropy(logits / value, labels) for value in candidates])
    return float(candidates[int(torch.argmin(losses))])


def emission_threshold(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    candidates = np.linspace(0.30, 0.90, 61)
    eligible = []
    for threshold in candidates:
        mask = confidence >= threshold
        if not mask.any():
            continue
        precision = float(np.mean(prediction[mask] == labels[mask]))
        coverage = float(mask.mean())
        if precision >= 0.75 and coverage >= 0.40:
            eligible.append((coverage, -threshold, precision, threshold))
    if not eligible:
        return {
            "threshold": None,
            "precision": None,
            "coverage": None,
            "automatic_gate_passed": False,
        }
    coverage, _, precision, threshold = max(eligible)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision), 6),
        "coverage": round(float(coverage), 6),
        "automatic_gate_passed": True,
    }


def main() -> None:
    args = parse_args()
    train_inputs_path = RUN / "FACT_TRAIN_INPUTS.json"
    train_labels_path = RUN / "FACT_TRAIN_LABELS.json"
    val_inputs_path = RUN / "FACT_VAL_INPUTS.json"
    val_labels_path = RUN / "FACT_VAL_LABELS.json"
    feature_receipt_path = RUN / "FEATURE_EXTRACTION_RECEIPT.json"
    protocol_path = RUN / "FACT_PLUGIN_DEVELOPMENT_PROTOCOL.json"
    train_inputs, train_labels = load_aligned(train_inputs_path, train_labels_path)
    val_inputs, val_labels = load_aligned(val_inputs_path, val_labels_path)
    all_inputs = train_inputs + val_inputs
    missing = sorted({str(feature_path(row)) for row in all_inputs if not feature_path(row).exists()})
    check = {
        "environment": {
            "python": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "requested_device": args.device,
        },
        "train_samples": len(train_inputs),
        "val_samples": len(val_inputs),
        "expected_feature_files": len({feature_path(row) for row in all_inputs}),
        "missing_feature_files": len(missing),
        "test_samples": 0,
        "training_started": False,
    }
    if args.check_only:
        print(json.dumps(check, indent=2))
        return
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected feature files are missing; run the guarded sparse downloader first"
        )
    if not feature_receipt_path.exists():
        raise RuntimeError("FEATURE_EXTRACTION_RECEIPT.json is required before training")
    feature_receipt = json.loads(feature_receipt_path.read_text())
    if feature_receipt.get("status") != "complete" or feature_receipt.get("test_feature_payloads_opened") is not False:
        raise RuntimeError("feature receipt is incomplete or violates the test firewall")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    train_pooled = [pooled_feature(row) for row in train_inputs]
    val_pooled = [pooled_feature(row) for row in val_inputs]
    train_x_raw = np.stack([value for value, _ in train_pooled])
    val_x_raw = np.stack([value for value, _ in val_pooled])
    mean = train_x_raw.mean(axis=0)
    std = train_x_raw.std(axis=0)
    std[std < 1e-6] = 1.0
    train_x = (train_x_raw - mean) / std
    val_x = (val_x_raw - mean) / std

    activity_ids = sorted({int(row["activity_id"]) for row in all_inputs})
    activity_to_index = {value: index for index, value in enumerate(activity_ids)}
    train_activity = np.asarray([activity_to_index[int(row["activity_id"])] for row in train_inputs])
    val_activity = np.asarray([activity_to_index[int(row["activity_id"])] for row in val_inputs])
    train_one_hot = np.eye(len(activity_ids), dtype=np.float32)[train_activity]
    val_one_hot = np.eye(len(activity_ids), dtype=np.float32)[val_activity]
    train_x = np.concatenate([train_x, train_one_hot], axis=1).astype(np.float32)
    val_x = np.concatenate([val_x, val_one_hot], axis=1).astype(np.float32)

    step_ids = sorted({int(row["canonical_step_id"]) for row in train_labels + val_labels})
    verbs = sorted({str(row["action_verb"]) for row in train_labels + val_labels})
    step_to_index = {value: index for index, value in enumerate(step_ids)}
    verb_to_index = {value: index for index, value in enumerate(verbs)}
    descriptions: dict[int, str] = {}
    step_activity: dict[int, int] = {}
    for inputs, labels in ((train_inputs, train_labels), (val_inputs, val_labels)):
        for input_item, label_item in zip(inputs, labels):
            step_id = int(label_item["canonical_step_id"])
            activity_id = int(input_item["activity_id"])
            if step_id in step_activity and step_activity[step_id] != activity_id:
                raise RuntimeError(f"canonical step {step_id} appears in multiple recipes")
            step_activity[step_id] = activity_id
            descriptions[step_id] = str(label_item["canonical_step_description"])
    if set(step_activity) != set(step_ids):
        raise RuntimeError("canonical step/activity mapping is incomplete")

    train_step_y = np.asarray([step_to_index[int(row["canonical_step_id"])] for row in train_labels])
    val_step_y = np.asarray([step_to_index[int(row["canonical_step_id"])] for row in val_labels])
    train_verb_y = np.asarray([verb_to_index[str(row["action_verb"])] for row in train_labels])
    val_verb_y = np.asarray([verb_to_index[str(row["action_verb"])] for row in val_labels])
    step_activity_indices = torch.tensor(
        [activity_to_index[step_activity[step_id]] for step_id in step_ids], dtype=torch.long
    )

    train_x_t = torch.from_numpy(train_x).to(device)
    val_x_t = torch.from_numpy(val_x).to(device)
    train_step_t = torch.from_numpy(train_step_y).long().to(device)
    val_step_t = torch.from_numpy(val_step_y).long().to(device)
    train_verb_t = torch.from_numpy(train_verb_y).long().to(device)
    val_verb_t = torch.from_numpy(val_verb_y).long().to(device)
    train_activity_t = torch.from_numpy(train_activity).long().to(device)
    val_activity_t = torch.from_numpy(val_activity).long().to(device)
    step_activity_t = step_activity_indices.to(device)
    train_mask = recipe_mask(train_activity_t, step_activity_t)
    val_mask = recipe_mask(val_activity_t, step_activity_t)

    model = FactHead(train_x.shape[1], args.hidden_dim, len(step_ids), len(verbs)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    step_weight = class_weights(train_step_y, len(step_ids)).to(device)
    verb_weight = class_weights(train_verb_y, len(verbs)).to(device)
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_logits, verb_logits = model(train_x_t)
        step_logits = apply_step_mask(step_logits, train_mask)
        step_loss = nn.functional.cross_entropy(step_logits, train_step_t, weight=step_weight)
        verb_loss = nn.functional.cross_entropy(verb_logits, train_verb_t, weight=verb_weight)
        loss = step_loss + 0.5 * verb_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_step_logits, val_verb_logits = model(val_x_t)
            val_step_logits = apply_step_mask(val_step_logits, val_mask)
            val_step_pred = val_step_logits.argmax(dim=1).cpu().numpy()
            val_verb_pred = val_verb_logits.argmax(dim=1).cpu().numpy()
            step_f1 = macro_f1(val_step_y, val_step_pred, len(step_ids))
            verb_f1 = macro_f1(val_verb_y, val_verb_pred, len(verbs))
            top3 = float(
                np.mean(
                    [
                        target in row
                        for target, row in zip(
                            val_step_y,
                            torch.topk(val_step_logits, k=min(3, len(step_ids)), dim=1).indices.cpu().numpy(),
                        )
                    ]
                )
            )
            score = top3 + step_f1 + 0.5 * verb_f1
        if epoch == 1 or epoch % 10 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss.detach().cpu()), 6),
                    "val_step_macro_f1": round(step_f1, 6),
                    "val_action_verb_macro_f1": round(verb_f1, 6),
                    "val_step_top3_recall": round(top3, 6),
                }
            )
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("fact-head training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_step_logits, val_verb_logits = model(val_x_t)
        val_step_logits = apply_step_mask(val_step_logits, val_mask)
        step_temperature = best_temperature(val_step_logits, val_step_t)
        verb_temperature = best_temperature(val_verb_logits, val_verb_t)
        step_prob = torch.softmax(val_step_logits / step_temperature, dim=1).cpu().numpy()
        verb_prob = torch.softmax(val_verb_logits / verb_temperature, dim=1).cpu().numpy()
    step_prediction = step_prob.argmax(axis=1)
    verb_prediction = verb_prob.argmax(axis=1)
    top3_indices = np.argsort(-step_prob, axis=1)[:, : min(3, len(step_ids))]
    validation_metrics = {
        "coarse_step_top1_accuracy": round(float(np.mean(step_prediction == val_step_y)), 6),
        "coarse_step_top3_recall": round(
            float(np.mean([target in row for target, row in zip(val_step_y, top3_indices)])), 6
        ),
        "coarse_step_macro_f1": round(macro_f1(val_step_y, step_prediction, len(step_ids)), 6),
        "action_verb_accuracy": round(float(np.mean(verb_prediction == val_verb_y)), 6),
        "action_verb_macro_f1": round(macro_f1(val_verb_y, verb_prediction, len(verbs)), 6),
        "coarse_step_ece": round(expected_calibration_error(step_prob, val_step_y), 6),
        "action_verb_ece": round(expected_calibration_error(verb_prob, val_verb_y), 6),
    }
    threshold = emission_threshold(step_prob, val_step_y)
    verb_threshold = emission_threshold(verb_prob, val_verb_y)
    training_hash = sha256_joined([train_inputs_path, train_labels_path])
    feature_bundle_hash = str(feature_receipt["selected_feature_bundle_sha256"])
    checkpoint_payload = {
        "schema_version": "captaincook4d_fact_head_checkpoint_v1",
        "model_state": best_state,
        "model_config": {
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "step_count": len(step_ids),
            "verb_count": len(verbs),
        },
        "normalization": {
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "activity_ids": activity_ids,
        },
        "label_maps": {
            "step_ids": step_ids,
            "verbs": verbs,
            "step_descriptions": descriptions,
            "step_activity": step_activity,
        },
        "calibration": {
            "step_temperature": step_temperature,
            "verb_temperature": verb_temperature,
            "step_emission": threshold,
            "verb_emission": verb_threshold,
        },
        "provenance": {
            "feature_extractor": "official_3dresnet_slow_r50_1s_features",
            "feature_asset_sha256": feature_bundle_hash,
            "training_split_sha256": training_hash,
            "training_label_kind": "canonical_step_or_text_derived_fact_only",
            "test_target_labels_accessed": False,
            "official_error_model_used": False,
        },
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, args.checkpoint)
    checkpoint_hash = sha256_file(args.checkpoint)
    record = {
        "schema_version": "captaincook4d_fact_head_training_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_consumed_development",
        "device": str(device),
        "seed": args.seed,
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "patience": args.patience,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "requested_device": args.device,
        },
        "data": {
            "train_samples": len(train_inputs),
            "val_samples": len(val_inputs),
            "train_recordings": len({row["recording_id"] for row in train_inputs}),
            "val_recordings": len({row["recording_id"] for row in val_inputs}),
            "train_val_recording_overlap": len(
                {row["recording_id"] for row in train_inputs}
                & {row["recording_id"] for row in val_inputs}
            ),
            "minimum_feature_coverage": round(
                min([coverage for _, coverage in train_pooled + val_pooled]), 6
            ),
        },
        "class_support": {
            "step_classes": len(step_ids),
            "verb_classes": len(verbs),
            "train_step_counts": dict(sorted(Counter(map(str, [row["canonical_step_id"] for row in train_labels])).items())),
            "train_verb_counts": dict(sorted(Counter(row["action_verb"] for row in train_labels).items())),
        },
        "validation_metrics": validation_metrics,
        "emission_threshold": threshold,
        "verb_emission_threshold": verb_threshold,
        "history_every_10_epochs": history,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_hash},
        "provenance": checkpoint_payload["provenance"],
        "frozen_protocol": {"path": str(protocol_path.resolve()), "sha256": sha256_file(protocol_path)},
        "safety": {
            "test_samples": 0,
            "test_target_labels_accessed": False,
            "official_error_labels_used_as_training_targets": False,
            "official_error_model_used": False,
            "final_task_verdict_trained": False,
            "qwen_called": False,
        },
    }
    atomic_write_json(args.training_record, record)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                **validation_metrics,
                "step_emission": threshold,
                "verb_emission": verb_threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
