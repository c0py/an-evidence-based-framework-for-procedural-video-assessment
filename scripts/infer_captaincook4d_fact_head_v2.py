#!/usr/bin/env python3
"""Run label-free CaptainCook4D v2 visual/action/temporal fact inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import torch

from train_captaincook4d_fact_head import sha256_file
from train_captaincook4d_fact_head_v2 import (
    FEATURE_ROOT,
    RUN,
    base_probabilities,
    fusion_logits,
    probabilities_from_logits,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = RUN / "FACT_VAL_INPUTS.json"
DEFAULT_CHECKPOINT = ROOT / "models/captaincook4d_fact_head_v2.pt"
DEFAULT_OUTPUT = RUN / "FACT_VAL_PREDICTIONS_V2.json"
FACT_SCHEMA = RUN / "FACT_SCHEMA.json"
FORBIDDEN_KEYS = {
    "has_errors",
    "is_error",
    "error_probability",
    "state",
    "verdict",
    "final_answer",
    "error_description",
    "modified_description",
    "official_error_logit",
    "official_error_category",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--schema", type=Path, default=FACT_SCHEMA)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def keys_at_any_depth(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(
            *(keys_at_any_depth(item) for item in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(keys_at_any_depth(item) for item in value), set())
    return set()


def threshold_value(calibration: dict[str, Any], name: str) -> float | None:
    value = calibration[name].get("threshold")
    return float(value) if value is not None else None


def polarity(confidence: float, threshold: float | None) -> str:
    return "observed" if threshold is not None and confidence >= threshold else "uncertain"


def main() -> None:
    args = parse_args()
    if any(token in args.inputs.name.upper() for token in ("LABEL", "SEALED", "EVALUATION")):
        raise RuntimeError("prediction refuses label/evaluation input files")
    inputs = json.loads(args.inputs.read_text())
    leaked = keys_at_any_depth(inputs) & FORBIDDEN_KEYS
    if leaked:
        raise RuntimeError(f"prediction input contains forbidden target fields: {sorted(leaked)}")
    if any(row.get("split") == "test" for row in inputs):
        raise RuntimeError("development fact inference refuses official test inputs")
    missing_features = sorted(
        {
            str(args.feature_root / row["split"] / f"{row['recording_id']}.npz")
            for row in inputs
            if not (args.feature_root / row["split"] / f"{row['recording_id']}.npz").exists()
        }
    )
    check = {
        "input_manifest": str(args.inputs.resolve()),
        "input_cases": len(inputs),
        "checkpoint_exists": args.checkpoint.exists(),
        "missing_feature_files": len(missing_features),
        "forbidden_input_keys": sorted(leaked),
        "test_inputs": 0,
        "inference_started": False,
    }
    if args.check_only:
        print(json.dumps(check, indent=2))
        return
    if not args.checkpoint.exists():
        raise RuntimeError("v2 fact-head checkpoint is missing")
    if missing_features:
        raise RuntimeError(f"{len(missing_features)} selected feature files are missing")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "captaincook4d_fact_head_checkpoint_v2":
        raise RuntimeError("unexpected v2 fact-head checkpoint schema")
    if checkpoint["provenance"].get("test_target_labels_accessed") is not False:
        raise RuntimeError("v2 checkpoint provenance violates the test firewall")
    base = checkpoint["base_checkpoint"]
    step_probability, verb_probability, coverages = base_probabilities(
        inputs, base, args.feature_root
    )
    fusion = checkpoint["fusion"]
    step_to_verb = {int(key): str(value) for key, value in fusion["step_to_verb"].items()}
    temporal = {int(key): value for key, value in fusion["temporal"].items()}
    logits = fusion_logits(
        inputs,
        step_probability,
        verb_probability,
        base,
        args.feature_root,
        step_to_verb,
        temporal,
        {key: float(value) for key, value in fusion["weights"].items()},
    )
    fused_probability = probabilities_from_logits(logits, float(fusion["temperature"]))
    label_maps = base["label_maps"]
    step_ids = [int(value) for value in label_maps["step_ids"]]
    verbs = [str(value) for value in label_maps["verbs"]]
    descriptions = {
        int(key): str(value) for key, value in label_maps["step_descriptions"].items()
    }
    step_threshold = threshold_value(checkpoint["calibration"], "step_emission")
    verb_threshold = threshold_value(checkpoint["calibration"], "verb_emission")
    checkpoint_hash = sha256_file(args.checkpoint)
    source = {
        "model_name": "captaincook4d_visual_action_temporal_fact_head_v2",
        "model_role": "lightweight_fact_head",
        "checkpoint_sha256": checkpoint_hash,
        "feature_extractor": str(checkpoint["provenance"]["feature_extractor"]),
        "feature_asset_sha256": str(checkpoint["provenance"]["feature_asset_sha256"]),
        "training_split_sha256": str(checkpoint["provenance"]["training_split_sha256"]),
        "training_label_kind": "canonical_step_or_text_derived_fact_only",
    }
    schema = json.loads(args.schema.read_text())
    validator = jsonschema.Draft7Validator(schema)
    bundles = []
    for index, row in enumerate(inputs):
        top_steps = np.argsort(-fused_probability[index])[: min(3, len(step_ids))]
        best_verb = int(np.argmax(verb_probability[index]))
        facts = []
        for rank_index, class_index in enumerate(top_steps, start=1):
            confidence = float(fused_probability[index, class_index])
            step_id = step_ids[int(class_index)]
            facts.append(
                {
                    "fact_id": f"{row['sample_key']}:coarse_step_v2:{rank_index}",
                    "fact_type": "coarse_step_hypothesis",
                    "value": descriptions[step_id],
                    "polarity": (
                        polarity(confidence, step_threshold)
                        if rank_index == 1
                        else "uncertain"
                    ),
                    "confidence": round(confidence, 8),
                    "start_s": float(row["start_time"]),
                    "end_s": float(row["end_time"]),
                    "coverage_fraction": round(coverages[index], 8),
                    "rank": rank_index,
                    "derivation": "temporal_aggregation_of_predictions",
                }
            )
        verb_confidence = float(verb_probability[index, best_verb])
        facts.append(
            {
                "fact_id": f"{row['sample_key']}:action_verb_v2:1",
                "fact_type": "action_verb",
                "value": verbs[best_verb],
                "polarity": polarity(verb_confidence, verb_threshold),
                "confidence": round(verb_confidence, 8),
                "start_s": float(row["start_time"]),
                "end_s": float(row["end_time"]),
                "coverage_fraction": round(coverages[index], 8),
                "rank": 1,
                "derivation": "direct_classifier_prediction",
            }
        )
        bundle = {
            "schema_version": "captaincook4d_fact_bundle_v1",
            "fact_bundle_id": hashlib.sha256(
                f"{row['sample_key']}:{checkpoint_hash}".encode()
            ).hexdigest()[:24],
            "sample_key": row["sample_key"],
            "task_id": "captaincook4d_step_conditioned_compliance",
            "observation_window": {
                "requested_start_s": float(row["start_time"]),
                "requested_end_s": float(row["end_time"]),
                "observed_start_s": float(row["start_time"]),
                "observed_end_s": float(row["end_time"]),
                "coverage_fraction": round(coverages[index], 8),
                "fully_observed": coverages[index] >= 0.999,
            },
            "facts": facts,
            "source": source,
            "final_task_verdict": False,
        }
        errors = sorted(validator.iter_errors(bundle), key=lambda error: list(error.path))
        if errors:
            raise RuntimeError(f"fact schema violation for {row['sample_key']}: {errors[0].message}")
        if keys_at_any_depth(bundle) & (FORBIDDEN_KEYS - {"final_task_verdict"}):
            raise RuntimeError(f"forbidden verdict key emitted for {row['sample_key']}")
        bundles.append(bundle)

    output = {
        "schema_version": "captaincook4d_fact_prediction_collection_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_fact_predictions_consumed_development",
        "inputs": {"path": str(args.inputs.resolve()), "sha256": sha256_file(args.inputs)},
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_hash},
        "fact_schema": {"path": str(args.schema.resolve()), "sha256": sha256_file(args.schema)},
        "bundle_count": len(bundles),
        "bundles": bundles,
        "safety": {
            "input_labels_read": False,
            "test_inputs": 0,
            "test_target_labels_accessed": False,
            "official_error_model_used": False,
            "final_task_verdict_emitted_by_plugin": False,
            "qwen_called": False,
            "rgb_used": False,
        },
    }
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "bundles": len(bundles), **output["safety"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
