#!/usr/bin/env python3
"""Build the bounded CaptainCook4D visual/action/temporal fact head v2.

V2 preserves the failed v1 visual checkpoint and adds only two fact-side
signals: compatibility with the predicted action verb and a train-derived
within-recording temporal prior.  It reads no final compliance/error targets
and never touches the official test split.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_captaincook4d_fact_head import (
    FactHead,
    apply_step_mask,
    best_temperature,
    emission_threshold,
    expected_calibration_error,
    load_aligned,
    macro_f1,
    recipe_mask,
    sha256_file,
    sha256_joined,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/captaincook4d_adaptation_v1"
FEATURE_ROOT = ROOT / "incoming/captaincook4d_features/3dresnet_1s_selected"
BASE_CHECKPOINT = ROOT / "models/captaincook4d_fact_head_v1.pt"
CHECKPOINT = ROOT / "models/captaincook4d_fact_head_v2.pt"
TRAINING_RECORD = RUN / "FACT_HEAD_TRAINING_V2.json"
PROTOCOL = RUN / "FACT_PLUGIN_DEVELOPMENT_PROTOCOL_V2.json"
V1_VALIDATION = RUN / "FACT_PLUGIN_VALIDATION.json"
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--training-record", type=Path, default=TRAINING_RECORD)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def feature_path(row: dict[str, Any], feature_root: Path) -> Path:
    return feature_root / str(row["split"]) / f"{row['recording_id']}.npz"


def pooled_feature(row: dict[str, Any], feature_root: Path) -> tuple[np.ndarray, float, int]:
    path = feature_path(row, feature_root)
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
    coverage = min(1.0, (actual_end - actual_start) / requested_count)
    return pooled, float(coverage), int(values.shape[0])


def normalized_midpoints(rows: list[dict[str, Any]], feature_root: Path) -> np.ndarray:
    positions = []
    for row in rows:
        _, _, feature_count = pooled_feature(row, feature_root)
        midpoint = (
            int(row["feature_start_index_inclusive"])
            + int(row["feature_end_index_exclusive"])
        ) / 2.0
        positions.append(min(1.0, max(0.0, midpoint / max(feature_count, 1))))
    return np.asarray(positions, dtype=np.float64)


def base_probabilities(
    rows: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    feature_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    pooled = [pooled_feature(row, feature_root) for row in rows]
    raw = np.stack([value for value, _, _ in pooled])
    mean = checkpoint["normalization"]["mean"].numpy()
    std = checkpoint["normalization"]["std"].numpy()
    normalized = (raw - mean) / std
    activity_ids = [int(value) for value in checkpoint["normalization"]["activity_ids"]]
    activity_to_index = {value: index for index, value in enumerate(activity_ids)}
    try:
        activity_indices = np.asarray(
            [activity_to_index[int(row["activity_id"])] for row in rows], dtype=np.int64
        )
    except KeyError as exception:
        raise RuntimeError(f"unsupported recipe/activity id: {exception}") from exception
    one_hot = np.eye(len(activity_ids), dtype=np.float32)[activity_indices]
    features = torch.from_numpy(
        np.concatenate([normalized, one_hot], axis=1).astype(np.float32)
    )
    config = checkpoint["model_config"]
    model = FactHead(
        int(config["input_dim"]),
        int(config["hidden_dim"]),
        int(config["step_count"]),
        int(config["verb_count"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    label_maps = checkpoint["label_maps"]
    step_ids = [int(value) for value in label_maps["step_ids"]]
    step_activity = {
        int(key): int(value) for key, value in label_maps["step_activity"].items()
    }
    step_activity_indices = torch.tensor(
        [activity_to_index[step_activity[step_id]] for step_id in step_ids],
        dtype=torch.long,
    )
    with torch.no_grad():
        step_logits, verb_logits = model(features)
        mask = recipe_mask(torch.from_numpy(activity_indices), step_activity_indices)
        step_logits = apply_step_mask(step_logits, mask)
        step_probability = torch.softmax(
            step_logits / float(checkpoint["calibration"]["step_temperature"]), dim=1
        ).numpy()
        verb_probability = torch.softmax(
            verb_logits / float(checkpoint["calibration"]["verb_temperature"]), dim=1
        ).numpy()
    return step_probability, verb_probability, [float(value[1]) for value in pooled]


def step_verb_mapping(
    train_labels: list[dict[str, Any]], step_ids: list[int], verbs: list[str]
) -> dict[int, str]:
    values: dict[int, set[str]] = {}
    for row in train_labels:
        values.setdefault(int(row["canonical_step_id"]), set()).add(str(row["action_verb"]))
    missing = sorted(set(step_ids) - set(values))
    ambiguous = {step: sorted(items) for step, items in values.items() if len(items) != 1}
    if missing or ambiguous:
        raise RuntimeError(f"incomplete/ambiguous train step-to-verb map: missing={missing}, ambiguous={ambiguous}")
    mapping = {step: next(iter(values[step])) for step in step_ids}
    unsupported = sorted(set(mapping.values()) - set(verbs))
    if unsupported:
        raise RuntimeError(f"step-to-verb map contains unsupported verbs: {unsupported}")
    return mapping


def temporal_statistics(
    train_inputs: list[dict[str, Any]],
    train_labels: list[dict[str, Any]],
    step_ids: list[int],
    feature_root: Path,
    standard_deviation_floor: float,
) -> dict[int, dict[str, float | int]]:
    positions = normalized_midpoints(train_inputs, feature_root)
    by_step: dict[int, list[float]] = {step: [] for step in step_ids}
    for label, position in zip(train_labels, positions):
        by_step[int(label["canonical_step_id"])].append(float(position))
    if any(not values for values in by_step.values()):
        raise RuntimeError("every supported canonical step needs at least one train temporal sample")
    result = {}
    for step, values in by_step.items():
        deviation = float(np.std(values)) if len(values) > 1 else 0.12
        result[step] = {
            "median": float(np.median(values)),
            "standard_deviation": max(float(standard_deviation_floor), deviation),
            "support": len(values),
        }
    return result


def fusion_logits(
    rows: list[dict[str, Any]],
    step_probability: np.ndarray,
    verb_probability: np.ndarray,
    base_checkpoint: dict[str, Any],
    feature_root: Path,
    step_to_verb: dict[int, str],
    temporal: dict[int, dict[str, float | int]],
    weights: dict[str, float],
) -> np.ndarray:
    label_maps = base_checkpoint["label_maps"]
    step_ids = [int(value) for value in label_maps["step_ids"]]
    verbs = [str(value) for value in label_maps["verbs"]]
    verb_to_index = {value: index for index, value in enumerate(verbs)}
    step_activity = {
        int(key): int(value) for key, value in label_maps["step_activity"].items()
    }
    allowed = np.asarray(
        [
            [step_activity[step] == int(row["activity_id"]) for step in step_ids]
            for row in rows
        ],
        dtype=bool,
    )
    positions = normalized_midpoints(rows, feature_root)
    medians = np.asarray([float(temporal[step]["median"]) for step in step_ids])
    deviations = np.asarray(
        [float(temporal[step]["standard_deviation"]) for step in step_ids]
    )
    temporal_probability = np.exp(
        -0.5 * ((positions[:, None] - medians[None, :]) / deviations[None, :]) ** 2
    ) / (deviations[None, :] + EPSILON)
    temporal_probability = np.where(allowed, temporal_probability, EPSILON)
    temporal_probability /= temporal_probability.sum(axis=1, keepdims=True)
    verb_compatibility = np.asarray(
        [
            [
                verb_probability[index, verb_to_index[step_to_verb[step]]]
                if allowed[index, step_index]
                else EPSILON
                for step_index, step in enumerate(step_ids)
            ]
            for index in range(len(rows))
        ],
        dtype=np.float64,
    )
    verb_compatibility /= verb_compatibility.sum(axis=1, keepdims=True)
    logits = (
        float(weights["visual"]) * np.log(step_probability + EPSILON)
        + float(weights["action_verb_compatibility"])
        * np.log(verb_compatibility + EPSILON)
        + float(weights["train_temporal_position"])
        * np.log(temporal_probability + EPSILON)
    )
    return np.where(allowed, logits, -1e4)


def probabilities_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits / float(temperature)
    shifted -= shifted.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    required = [
        args.base_checkpoint,
        args.protocol,
        RUN / "FACT_TRAIN_INPUTS.json",
        RUN / "FACT_TRAIN_LABELS.json",
        RUN / "FACT_VAL_INPUTS.json",
        RUN / "FACT_VAL_LABELS.json",
        V1_VALIDATION,
    ]
    missing = [str(path) for path in required if not path.exists()]
    check = {
        "required_files": len(required),
        "missing_files": missing,
        "test_samples": 0,
        "training_started": False,
    }
    if args.check_only:
        print(json.dumps(check, indent=2))
        return
    if missing:
        raise RuntimeError(f"missing v2 prerequisite files: {missing}")

    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "frozen_after_v1_failure_and_consumed_val_diagnostic_before_v2_checkpoint_generation":
        raise RuntimeError("v2 protocol is not frozen at the required pre-checkpoint state")
    frozen = protocol["frozen_inputs"]
    path_hashes = {
        "fact_train_inputs_sha256": sha256_file(RUN / "FACT_TRAIN_INPUTS.json"),
        "fact_train_labels_sha256": sha256_file(RUN / "FACT_TRAIN_LABELS.json"),
        "fact_val_inputs_sha256": sha256_file(RUN / "FACT_VAL_INPUTS.json"),
        "fact_val_labels_sha256": sha256_file(RUN / "FACT_VAL_LABELS.json"),
        "task_package_sha256": sha256_file(RUN / "TASK_PACKAGE.json"),
    }
    if path_hashes != frozen:
        raise RuntimeError("current v2 inputs do not match the frozen protocol hashes")
    if sha256_file(args.base_checkpoint) != protocol["frozen_v2_method"]["base_checkpoint_sha256"]:
        raise RuntimeError("base checkpoint does not match the frozen v2 protocol")
    if sha256_file(V1_VALIDATION) != protocol["reason_for_revision"]["v1_validation_sha256"]:
        raise RuntimeError("v1 failure artifact changed after the v2 protocol freeze")

    train_inputs, train_labels = load_aligned(
        RUN / "FACT_TRAIN_INPUTS.json", RUN / "FACT_TRAIN_LABELS.json"
    )
    val_inputs, val_labels = load_aligned(
        RUN / "FACT_VAL_INPUTS.json", RUN / "FACT_VAL_LABELS.json"
    )
    if any(row.get("split") == "test" for row in train_inputs + val_inputs):
        raise RuntimeError("v2 training refuses official test inputs")
    missing_features = sorted(
        {
            str(feature_path(row, args.feature_root))
            for row in train_inputs + val_inputs
            if not feature_path(row, args.feature_root).exists()
        }
    )
    if missing_features:
        raise RuntimeError(f"{len(missing_features)} selected feature files are missing")

    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    if base.get("schema_version") != "captaincook4d_fact_head_checkpoint_v1":
        raise RuntimeError("unexpected base fact-head checkpoint schema")
    if base["provenance"].get("test_target_labels_accessed") is not False:
        raise RuntimeError("base checkpoint provenance violates the test firewall")
    step_ids = [int(value) for value in base["label_maps"]["step_ids"]]
    verbs = [str(value) for value in base["label_maps"]["verbs"]]
    step_to_verb = step_verb_mapping(train_labels, step_ids, verbs)
    method = protocol["frozen_v2_method"]
    weights = {key: float(value) for key, value in method["weights"].items()}
    temporal = temporal_statistics(
        train_inputs,
        train_labels,
        step_ids,
        args.feature_root,
        float(method["temporal_standard_deviation_floor"]),
    )
    step_probability, verb_probability, coverages = base_probabilities(
        val_inputs, base, args.feature_root
    )
    logits = fusion_logits(
        val_inputs,
        step_probability,
        verb_probability,
        base,
        args.feature_root,
        step_to_verb,
        temporal,
        weights,
    )
    step_to_index = {step: index for index, step in enumerate(step_ids)}
    verb_to_index = {verb: index for index, verb in enumerate(verbs)}
    step_y = np.asarray(
        [step_to_index[int(row["canonical_step_id"])] for row in val_labels], dtype=np.int64
    )
    verb_y = np.asarray(
        [verb_to_index[str(row["action_verb"])] for row in val_labels], dtype=np.int64
    )
    temperature = best_temperature(
        torch.from_numpy(logits.astype(np.float32)), torch.from_numpy(step_y)
    )
    fused_probability = probabilities_from_logits(logits, temperature)
    step_prediction = fused_probability.argmax(axis=1)
    verb_prediction = verb_probability.argmax(axis=1)
    top3 = np.argsort(-fused_probability, axis=1)[:, : min(3, len(step_ids))]
    emission = emission_threshold(fused_probability, step_y)
    validation_metrics = {
        "coarse_step_top1_accuracy": round(float(np.mean(step_prediction == step_y)), 6),
        "coarse_step_top3_recall": round(
            float(np.mean([target in row for target, row in zip(step_y, top3)])), 6
        ),
        "coarse_step_macro_f1": round(
            macro_f1(step_y, step_prediction, len(step_ids)), 6
        ),
        "action_verb_accuracy": round(float(np.mean(verb_prediction == verb_y)), 6),
        "action_verb_macro_f1": round(
            macro_f1(verb_y, verb_prediction, len(verbs)), 6
        ),
        "coarse_step_ece": round(
            expected_calibration_error(fused_probability, step_y), 6
        ),
    }
    automatic_gate = bool(
        validation_metrics["coarse_step_top3_recall"]
        >= float(protocol["automatic_gate"]["coarse_step_top3_recall_minimum"])
        and emission["automatic_gate_passed"]
    )
    checkpoint_payload = {
        "schema_version": "captaincook4d_fact_head_checkpoint_v2",
        "base_checkpoint": base,
        "fusion": {
            "weights": weights,
            "temperature": float(temperature),
            "epsilon": EPSILON,
            "step_to_verb": step_to_verb,
            "temporal": temporal,
            "temporal_position_definition": method["temporal_position"],
        },
        "calibration": {
            "step_emission": emission,
            "verb_emission": base["calibration"]["verb_emission"],
        },
        "provenance": {
            "feature_extractor": base["provenance"]["feature_extractor"],
            "feature_asset_sha256": base["provenance"]["feature_asset_sha256"],
            "training_split_sha256": sha256_joined(
                [RUN / "FACT_TRAIN_INPUTS.json", RUN / "FACT_TRAIN_LABELS.json"]
            ),
            "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
            "protocol_sha256": sha256_file(args.protocol),
            "training_label_kind": "canonical_step_or_text_derived_fact_only",
            "consumed_val_used_for_calibration": True,
            "test_target_labels_accessed": False,
            "official_error_model_used": False,
        },
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, args.checkpoint)
    output = {
        "schema_version": "captaincook4d_fact_head_training_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "automatic_gate_passed_pending_label_free_inference_evaluation"
            if automatic_gate
            else "automatic_gate_failed_stop_before_qwen"
        ),
        "method": {
            "weights": weights,
            "fusion_temperature": round(float(temperature), 6),
            "temporal_standard_deviation_floor": method[
                "temporal_standard_deviation_floor"
            ],
            "train_only_temporal_priors": True,
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
            "minimum_feature_coverage": round(min(coverages), 6),
        },
        "validation_metrics": validation_metrics,
        "emission_threshold": emission,
        "automatic_p2_gate_candidate": automatic_gate,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
        "base_checkpoint": {
            "path": str(args.base_checkpoint.resolve()),
            "sha256": sha256_file(args.base_checkpoint),
        },
        "frozen_protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol),
        },
        "v1_failure_preserved": {
            "path": str(V1_VALIDATION.resolve()),
            "sha256": sha256_file(V1_VALIDATION),
        },
        "development_selection_disclosure": protocol[
            "development_selection_disclosure"
        ],
        "safety": {
            "test_samples": 0,
            "test_target_labels_accessed": False,
            "official_error_labels_used_as_training_targets": False,
            "official_error_model_used": False,
            "final_task_verdict_trained": False,
            "qwen_called": False,
            "rgb_used": False,
        },
    }
    atomic_write_json(args.training_record, output)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                **validation_metrics,
                "step_emission": emission,
                "automatic_p2_gate_candidate": automatic_gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
