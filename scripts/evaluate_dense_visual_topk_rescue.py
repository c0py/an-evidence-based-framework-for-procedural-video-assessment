#!/usr/bin/env python3
"""Calibrate dense visual top-k rescue on train OOF, then evaluate frozen on dev."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_qwen_visual_adapter_fusion_oof import TEMPORAL
from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.local_rescue import (
    LocalRescuePolicy, apply_topk_local_rescue, centered_mean,
)
from run_cholec80_m2c_high_resolution_validation import load_calibrator, predict
from run_cholec80_validation_ablation import (
    CRITERIA, merge_intervals, method_summary,
    paired_bootstrap_comparison, summarize_rows,
    temporal_metrics, truth_intervals,
)


SMOOTHING_POINTS = (1, 3, 5, 7)
VISUAL_QUANTILES = (0.80, 0.90, 0.95, 0.975, 0.99)
BASE_SUPPORT_RATIOS = (0.25, 0.50, 0.75, 0.90)
TOP_K_VALUES = (1, 2)
RADIUS_POINTS = (0, 1, 2, 3)


def intervals_from_scores(
    timestamps: list[float], scores: np.ndarray, threshold: float,
    cadence_s: float, start_s: float, end_s: float,
) -> list[tuple[float, float]]:
    """Fast numerical equivalent of StableEvidenceAggregator for dense scores."""
    times = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    smooth = np.empty_like(values)
    left = 0
    smoothing_seconds = float(TEMPORAL["smoothing_seconds"])
    for index, time_s in enumerate(times):
        while time_s - times[left] > smoothing_seconds:
            left += 1
        # Cholec caches use a 5 s cadence, so the causal 5 s window normally
        # contains one or two points.  Avoid the high overhead of np.median for
        # these overwhelmingly common scalar/two-scalar cases.
        count = index + 1 - left
        if count == 1:
            smooth[index] = values[index]
        elif count == 2:
            smooth[index] = 0.5 * (values[index - 1] + values[index])
        else:
            smooth[index] = np.median(values[left:index + 1])

    off_threshold = max(0.05, 0.8 * threshold)
    max_gap_seconds = float(TEMPORAL["max_gap_seconds"])
    min_stable_seconds = float(TEMPORAL["min_stable_seconds"])
    candidates: list[tuple[int, int]] = []
    active_start: int | None = None
    active_end: int | None = None
    last_above_off: float | None = None
    for index, (time_s, score) in enumerate(zip(times, smooth)):
        if active_start is None:
            if score >= threshold:
                active_start = active_end = index
                last_above_off = float(time_s)
        elif score >= off_threshold:
            active_end = index
            last_above_off = float(time_s)
        elif last_above_off is not None and time_s - last_above_off <= max_gap_seconds:
            active_end = index
        else:
            candidates.append((active_start, int(active_end)))
            if score >= threshold:
                active_start = active_end = index
                last_above_off = float(time_s)
            else:
                active_start = active_end = None
                last_above_off = None
    if active_start is not None:
        candidates.append((active_start, int(active_end)))

    half = cadence_s / 2.0
    intervals = []
    for first, last in candidates:
        if times[last] - times[first] < min_stable_seconds:
            continue
        if int(np.sum(smooth[first:last + 1] >= off_threshold)) < 2:
            continue
        intervals.append((
            max(start_s, float(times[first]) - half),
            min(end_s, float(times[last]) + half),
        ))
    return merge_intervals(intervals)


@dataclass(frozen=True)
class SearchConfig:
    smoothing_points: int
    visual_quantile: float
    base_support_ratio: float
    top_k: int
    radius_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "smoothing_points": self.smoothing_points,
            "visual_quantile": self.visual_quantile,
            "base_support_ratio": self.base_support_ratio,
            "top_k": self.top_k,
            "radius_points": self.radius_points,
        }


def search_configs() -> list[SearchConfig]:
    return [
        SearchConfig(smoothing, quantile, support, top_k, radius)
        for smoothing in SMOOTHING_POINTS
        for quantile in VISUAL_QUANTILES
        for support in BASE_SUPPORT_RATIOS
        for top_k in TOP_K_VALUES
        for radius in RADIUS_POINTS
    ]


def load_prediction_files(
    paths: list[Path], expected_ids: set[int], require_oof: bool,
) -> tuple[dict[tuple[int, str], dict[str, np.ndarray]], dict[int, str]]:
    grouped: dict[tuple[int, str], list[tuple[float, float]]] = {}
    fold_by_video: dict[int, str] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("annotation_files_loaded_during_prediction") is not False:
            raise ValueError(f"Prediction artifact is not annotation-free: {path}")
        if payload.get("test_video_or_annotation_accessed") is not False:
            raise ValueError(f"Prediction artifact accessed sealed test data: {path}")
        source_ids = set(map(int, payload["validation_video_ids"]))
        for video_id in source_ids:
            if video_id in fold_by_video:
                raise ValueError(f"Duplicate visual prediction for video {video_id}")
            fold_by_video[video_id] = path.parent.name
        for row in payload["rows"]:
            key = (int(row["video_id"]), row["criterion"])
            grouped.setdefault(key, []).append(
                (float(row["time_s"]), float(row["full_probability"]))
            )
    if set(fold_by_video) != expected_ids:
        raise ValueError(
            f"Prediction videos mismatch: missing={sorted(expected_ids-set(fold_by_video))} "
            f"extra={sorted(set(fold_by_video)-expected_ids)}"
        )
    if require_oof and any(not name.startswith(("fold", "video")) for name in fold_by_video.values()):
        raise ValueError("Training auxiliary predictions must be grouped OOF artifacts")
    output = {}
    for key, rows in grouped.items():
        rows.sort()
        output[key] = {
            "timestamps": np.asarray([row[0] for row in rows], dtype=np.float64),
            "scores": np.asarray([row[1] for row in rows], dtype=np.float64),
        }
    return output, fold_by_video


def align_visual(
    timestamps: np.ndarray, visual: dict[str, np.ndarray], video_id: int, criterion: str,
) -> np.ndarray:
    source_times, source_scores = visual["timestamps"], visual["scores"]
    if len(source_times) == len(timestamps) and np.allclose(source_times, timestamps, atol=1e-4):
        return source_scores.copy()
    source = {round(float(time_s), 4): float(score) for time_s, score in zip(source_times, source_scores)}
    missing = [float(time_s) for time_s in timestamps if round(float(time_s), 4) not in source]
    if missing:
        raise ValueError(
            f"Visual/M2c timestamp mismatch for video={video_id} criterion={criterion}: "
            f"{missing[:5]}"
        )
    return np.asarray([source[round(float(time_s), 4)] for time_s in timestamps])


def build_pairs(
    video_ids: list[int], visual_predictions: dict[tuple[int, str], dict[str, np.ndarray]],
    cache_dir: Path, annotation_path: Path, model: torch.nn.Module,
    metadata: dict[str, Any], label_role: str,
) -> tuple[dict[tuple[int, str], dict[str, Any]], list[int]]:
    pairs, labeled_ids = {}, []
    thresholds = metadata["thresholds_calibrated_on_validation"]
    for progress, video_id in enumerate(video_ids, start=1):
        cache = torch.load(
            cache_dir / f"video{video_id:02d}.pt", map_location="cpu", weights_only=False,
        )
        base_matrix = predict(cache, model, metadata).numpy()
        timestamps = cache["timestamps_s"].numpy().astype(np.float64)
        start_s, end_s = float(cache["window"]["start_s"]), float(cache["window"]["end_s"])
        cadence_s = float(cache["cadence_s"])
        annotations = load_cvs_intervals(annotation_path, video_id)
        if any(annotations[criterion] for criterion in CRITERIA):
            labeled_ids.append(video_id)
        for criterion_index, criterion in enumerate(CRITERIA):
            visual = align_visual(
                timestamps, visual_predictions[(video_id, criterion)], video_id, criterion,
            )
            truth = truth_intervals(annotations, criterion, start_s, end_s)
            base_scores = base_matrix[:, criterion_index].astype(np.float64)
            baseline = intervals_from_scores(
                timestamps.tolist(), base_scores, float(thresholds[criterion]), cadence_s,
                start_s, end_s,
            )
            pairs[(video_id, criterion)] = {
                "timestamps": timestamps, "base_scores": base_scores,
                "visual_scores": visual, "truth": truth,
                "baseline": baseline, "on_threshold": float(thresholds[criterion]),
                "cadence_s": cadence_s, "start_s": start_s, "end_s": end_s,
            }
        print(f"{label_role}_alignment {progress}/{len(video_ids)} video={video_id:02d}", flush=True)
    return pairs, labeled_ids


def visual_threshold(
    criterion: str, video_ids: list[int], pairs: dict[tuple[int, str], dict[str, Any]],
    smoothing_points: int, quantile: float,
) -> float:
    values = np.concatenate([
        centered_mean(pairs[(video_id, criterion)]["visual_scores"], smoothing_points)
        for video_id in video_ids
    ])
    return float(np.quantile(values, quantile))


def evaluate_config(
    criterion: str, video_ids: list[int], pairs: dict[tuple[int, str], dict[str, Any]],
    config: SearchConfig, threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, actions = [], []
    min_distance = max(config.smoothing_points, 2 * config.radius_points + 1, 6)
    policy = LocalRescuePolicy(
        visual_threshold=threshold, base_support_ratio=config.base_support_ratio,
        top_k=config.top_k, smoothing_points=config.smoothing_points,
        radius_points=config.radius_points, min_peak_distance_points=min_distance,
    )
    for video_id in video_ids:
        pair = pairs[(video_id, criterion)]
        fused_scores, current_actions = apply_topk_local_rescue(
            pair["base_scores"], pair["visual_scores"], pair["on_threshold"], policy,
        )
        if current_actions:
            fused = intervals_from_scores(
                pair["timestamps"].tolist(), fused_scores, pair["on_threshold"],
                pair["cadence_s"], pair["start_s"], pair["end_s"],
            )
        else:
            fused = pair["baseline"]
        rows.append({
            "video_id": video_id, "criterion": criterion,
            **temporal_metrics(fused, pair["truth"]),
        })
        for action in current_actions:
            actions.append({
                "video_id": video_id, "criterion": criterion,
                "peak_time_s": float(pair["timestamps"][int(action["peak_index"])]),
                **action,
            })
    return rows, actions


def baseline_rows(
    criterion: str, video_ids: list[int], pairs: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [{
        "video_id": video_id, "criterion": criterion,
        **temporal_metrics(
            pairs[(video_id, criterion)]["baseline"],
            pairs[(video_id, criterion)]["truth"],
        ),
    } for video_id in video_ids]


def select_config(
    criterion: str, video_ids: list[int], pairs: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    base_summary = summarize_rows(baseline_rows(criterion, video_ids, pairs))
    negative_tolerance = 1.0 / max(1, base_summary["n_truth_negative_pairs"])
    candidates = []
    threshold_cache = {
        (smoothing, quantile): visual_threshold(
            criterion, video_ids, pairs, smoothing, quantile,
        )
        for smoothing in SMOOTHING_POINTS for quantile in VISUAL_QUANTILES
    }
    for config in search_configs():
        threshold = threshold_cache[(config.smoothing_points, config.visual_quantile)]
        rows, actions = evaluate_config(criterion, video_ids, pairs, config, threshold)
        summary = summarize_rows(rows)
        eligible = (
            summary["positive_pair_detection_rate"] + 1e-12
            >= base_summary["positive_pair_detection_rate"]
            and summary["negative_pair_rejection_rate"] + negative_tolerance + 1e-12
            >= base_summary["negative_pair_rejection_rate"]
        )
        objective = (
            summary["macro_positive_pair_temporal_iou"]
            + summary["positive_pair_detection_rate"]
            + summary["negative_pair_rejection_rate"]
        ) / 3.0
        candidates.append({
            "config": config, "visual_threshold": threshold, "summary": summary,
            "objective": objective, "eligible": eligible, "action_count": len(actions),
        })
    candidates.append({
        "config": None, "visual_threshold": None, "summary": base_summary,
        "objective": (
            base_summary["macro_positive_pair_temporal_iou"]
            + base_summary["positive_pair_detection_rate"]
            + base_summary["negative_pair_rejection_rate"]
        ) / 3.0,
        "eligible": True, "action_count": 0,
    })
    selected = max(
        (item for item in candidates if item["eligible"]),
        key=lambda item: (
            item["objective"], item["summary"]["macro_positive_pair_temporal_iou"],
            item["summary"]["negative_pair_rejection_rate"], -item["action_count"],
        ),
    )
    return {
        "enabled": selected["config"] is not None,
        "config": selected["config"].to_dict() if selected["config"] else None,
        "visual_threshold": selected["visual_threshold"],
        "objective": selected["objective"], "summary": selected["summary"],
        "baseline_summary": base_summary, "action_count": selected["action_count"],
        "negative_rejection_tolerance": negative_tolerance,
    }


def rows_with_frozen(
    video_ids: list[int], pairs: dict[tuple[int, str], dict[str, Any]],
    policies: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, actions = [], []
    for criterion in CRITERIA:
        selected = policies[criterion]
        if not selected["enabled"]:
            rows.extend(baseline_rows(criterion, video_ids, pairs))
            continue
        config = SearchConfig(**selected["config"])
        criterion_rows, criterion_actions = evaluate_config(
            criterion, video_ids, pairs, config, float(selected["visual_threshold"]),
        )
        rows.extend(criterion_rows)
        actions.extend(criterion_actions)
    rows.sort(key=lambda row: (row["video_id"], CRITERIA.index(row["criterion"])))
    return rows, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-5s", type=Path, required=True)
    parser.add_argument("--train-predictions-dir", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    train_ids = list(map(int, split["train_video_ids"]))
    validation_ids = list(map(int, split["validation_video_ids"]))
    test_ids = set(map(int, split["test_video_ids"]))
    if (set(train_ids) | set(validation_ids)) & test_ids:
        raise ValueError("Train/development overlap with sealed test")
    annotation_path = Path(split["dataset_root"]) / "annotations" / "cholec80-CVS.xlsx"

    train_paths = sorted(args.train_predictions_dir.glob("*/predictions.json"))
    train_visual, fold_by_video = load_prediction_files(
        train_paths, set(train_ids), require_oof=True,
    )
    validation_visual, _ = load_prediction_files(
        [args.validation_predictions], set(validation_ids), require_oof=False,
    )
    model, metadata = load_calibrator(args.checkpoint)
    train_pairs, labeled_train_ids = build_pairs(
        train_ids, train_visual, args.cache_5s, annotation_path, model, metadata, "train_oof",
    )

    fold_names = sorted({fold_by_video[video_id] for video_id in labeled_train_ids})
    nested_by_key = {}
    fold_policies = []
    for fold_name in fold_names:
        heldout = [video_id for video_id in labeled_train_ids if fold_by_video[video_id] == fold_name]
        inner = [video_id for video_id in labeled_train_ids if video_id not in heldout]
        for criterion in CRITERIA:
            selected = select_config(criterion, inner, train_pairs)
            heldout_rows, heldout_actions = rows_with_frozen(
                heldout, train_pairs, {name: (
                    selected if name == criterion else {"enabled": False}
                ) for name in CRITERIA},
            )
            for row in heldout_rows:
                if row["criterion"] == criterion:
                    nested_by_key[(row["video_id"], criterion)] = row
            fold_policies.append({
                "fold": fold_name, "heldout_video_ids": heldout, "criterion": criterion,
                **selected, "heldout_action_count": sum(
                    action["criterion"] == criterion for action in heldout_actions
                ),
            })
        print(f"nested_selection fold={fold_name} heldout={heldout}", flush=True)

    train_baseline = [
        row for criterion in CRITERIA
        for row in baseline_rows(criterion, labeled_train_ids, train_pairs)
    ]
    train_baseline.sort(key=lambda row: (row["video_id"], CRITERIA.index(row["criterion"])))
    nested_rows = [
        nested_by_key[(video_id, criterion)]
        for video_id in labeled_train_ids for criterion in CRITERIA
    ]
    proposed_frozen_policies = {
        criterion: select_config(criterion, labeled_train_ids, train_pairs)
        for criterion in CRITERIA
    }
    # A full-train optimum is allowed into development only when its policy
    # family first improves under grouped nested OOF.  This prevents rare-event
    # train-fit gains from silently becoming a frozen deployment rule.
    stability_gate = {}
    frozen_policies = {}
    for criterion in CRITERIA:
        base = summarize_rows([
            row for row in train_baseline if row["criterion"] == criterion
        ])
        nested = summarize_rows([
            row for row in nested_rows if row["criterion"] == criterion
        ])
        base_objective = (
            base["macro_positive_pair_temporal_iou"]
            + base["positive_pair_detection_rate"]
            + base["negative_pair_rejection_rate"]
        ) / 3.0
        nested_objective = (
            nested["macro_positive_pair_temporal_iou"]
            + nested["positive_pair_detection_rate"]
            + nested["negative_pair_rejection_rate"]
        ) / 3.0
        passed = (
            proposed_frozen_policies[criterion]["enabled"]
            and nested_objective > base_objective + 1e-12
            and nested["macro_temporal_iou"] >= base["macro_temporal_iou"] - 1e-12
            and nested["negative_pair_rejection_rate"]
            >= base["negative_pair_rejection_rate"] - 1e-12
        )
        stability_gate[criterion] = {
            "passed": passed, "baseline_nested_objective": base_objective,
            "rescue_nested_objective": nested_objective,
            "baseline_macro_temporal_iou": base["macro_temporal_iou"],
            "rescue_macro_temporal_iou": nested["macro_temporal_iou"],
            "baseline_negative_pair_rejection_rate": base["negative_pair_rejection_rate"],
            "rescue_negative_pair_rejection_rate": nested["negative_pair_rejection_rate"],
        }
        frozen_policies[criterion] = (
            proposed_frozen_policies[criterion] if passed else {
                "enabled": False, "config": None, "visual_threshold": None,
                "disabled_by_nested_oof_stability_gate": True,
                "proposed_train_fit_policy": proposed_frozen_policies[criterion],
            }
        )
    train_fit_rows, train_fit_actions = rows_with_frozen(
        labeled_train_ids, train_pairs, frozen_policies,
    )

    # Development annotations are loaded only after all rescue parameters freeze.
    validation_pairs, labeled_validation_ids = build_pairs(
        validation_ids, validation_visual, args.cache_5s, annotation_path,
        model, metadata, "development_frozen",
    )
    if labeled_validation_ids != validation_ids:
        raise ValueError("Unexpected development video without CVS annotation rows")
    validation_baseline = [
        row for criterion in CRITERIA
        for row in baseline_rows(criterion, validation_ids, validation_pairs)
    ]
    validation_baseline.sort(
        key=lambda row: (row["video_id"], CRITERIA.index(row["criterion"]))
    )
    validation_fused, validation_actions = rows_with_frozen(
        validation_ids, validation_pairs, frozen_policies,
    )

    train_summaries = {
        "M2c_train_baseline": method_summary(train_baseline, 20260731, 2000),
        "M2c_dense_topk_rescue_nested_oof": method_summary(nested_rows, 20260731, 2000),
        "M2c_dense_topk_rescue_train_fit_diagnostic": method_summary(
            train_fit_rows, 20260731, 2000,
        ),
    }
    validation_summaries = {
        "M2c_5s_short_event": method_summary(validation_baseline, 20260731, 5000),
        "M2c_dense_visual_topk_local_rescue": method_summary(
            validation_fused, 20260731, 5000,
        ),
    }
    comparison = paired_bootstrap_comparison(
        validation_baseline, validation_fused, 20260731, 5000,
        "M2c_5s_short_event", "M2c_dense_visual_topk_local_rescue",
    )
    output = {
        "schema_version": "dense_visual_topk_local_rescue_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "train_auxiliary_predictions": "five-fold video-grouped OOF",
            "fusion_selection": "nested grouped-fold OOF on labeled train videos",
            "development_evaluation": "one frozen pass after parameter selection",
            "test_video_ids_loaded": [], "test_labels_accessed": False,
            "operator": "monotonic top-k local promotion with M2c corroboration",
            "search_grid_size_per_criterion": len(search_configs()) + 1,
        },
        "frozen_policies": frozen_policies,
        "proposed_train_fit_policies": proposed_frozen_policies,
        "nested_oof_stability_gate": stability_gate,
        "fold_policy_counts": {
            criterion: dict(Counter(
                json.dumps(item["config"], sort_keys=True)
                for item in fold_policies
                if item["criterion"] == criterion and item["enabled"]
            )) for criterion in CRITERIA
        },
        "train_summaries": train_summaries,
        "development_summaries": validation_summaries,
        "development_paired_comparison": comparison,
        "action_audit": {
            "train_fit_action_count": len(train_fit_actions),
            "development_action_count": len(validation_actions),
            "development_actions": validation_actions,
        },
        "rows": {
            "train_baseline": train_baseline, "train_nested_oof": nested_rows,
            "train_fit_diagnostic": train_fit_rows,
            "development_baseline": validation_baseline,
            "development_frozen_rescue": validation_fused,
        },
        "fold_policies": fold_policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "frozen_policies": frozen_policies,
        "train_summaries": train_summaries,
        "development_summaries": validation_summaries,
        "development_paired_comparison": comparison,
        "development_action_count": len(validation_actions),
        "output": str(args.output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
