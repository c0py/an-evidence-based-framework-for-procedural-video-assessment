#!/usr/bin/env python3
"""Nested video-OOF calibration of Qwen visual-state evidence over M2c intervals."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.schema import ScorePoint
from cvs_assessment.temporal import StableEvidenceAggregator
from run_cholec80_m2c_high_resolution_validation import load_calibrator, predict
from run_cholec80_validation_ablation import (
    CRITERIA, expand_aggregated_intervals, merge_intervals, method_summary,
    summarize_rows, temporal_metrics, truth_intervals,
)


STATE_CRITERIA = ("cystic_plate", "hepatocystic_triangle")
TEMPORAL = {"smoothing_seconds": 5.0, "min_stable_seconds": 5.0, "max_gap_seconds": 5.0}
# A candidates already passed the frozen temporal model.  The visual adapter is
# allowed to veto them only under extremely strong negative evidence; a wider
# grid overfit the few positive state videos and reduced held-out HCT recall.
A_THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
B_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01)


def subtract_holes(
    intervals: list[tuple[float, float]], holes: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    output = list(intervals)
    for hole_start, hole_end in sorted(holes):
        updated = []
        for start, end in output:
            if hole_end <= start or hole_start >= end:
                updated.append((start, end))
                continue
            if start < hole_start:
                updated.append((start, min(end, hole_start)))
            if hole_end < end:
                updated.append((max(start, hole_end), end))
        output = [(start, end) for start, end in updated if end > start]
    return merge_intervals(output)


def evidence_score(row: dict[str, Any]) -> float:
    """Equal-weight semantic confidence and 13-frame temporal persistence."""
    return 0.5 * float(row["mean_probability"]) + 0.5 * float(row["positive_frame_count"]) / 13.0


def apply_policy(
    base_intervals: list[tuple[float, float]], candidates: list[dict[str, Any]],
    adapter_rows: dict[str, dict[str, Any]], a_threshold: float, b_threshold: float,
    b_peak_threshold: float = 0.0,
    action_window: str = "candidate_interval",
) -> tuple[list[tuple[float, float]], dict[str, list[str]]]:
    holes, additions, rejected_a, accepted_b = [], [], [], []
    for candidate in candidates:
        row = adapter_rows[candidate["candidate_id"]]
        score = evidence_score(row)
        window_key = "pack_window_s" if action_window == "pack_window" else "interval_s"
        interval = tuple(map(float, candidate[window_key]))
        if candidate["candidate_class"] == "A_base_positive" and score < a_threshold:
            holes.append(interval)
            rejected_a.append(candidate["candidate_id"])
        elif (
            candidate["candidate_class"] == "B_rescue"
            and score >= b_threshold
            and float(candidate["score_summary"]["peak_to_threshold"]) >= b_peak_threshold
        ):
            additions.append(interval)
            accepted_b.append(candidate["candidate_id"])
    retained = subtract_holes(base_intervals, holes)
    return merge_intervals(retained + additions), {
        "rejected_A": rejected_a, "accepted_B": accepted_b,
    }


def policy_rows(
    criterion: str, video_ids: list[int], pair_data: dict[tuple[int, str], dict[str, Any]],
    by_pair_candidates: dict[tuple[int, str], list[dict[str, Any]]],
    adapter_rows: dict[str, dict[str, Any]], a_threshold: float, b_threshold: float,
    action_window: str = "candidate_interval",
) -> list[dict[str, Any]]:
    rows = []
    for video_id in video_ids:
        pair = pair_data[(video_id, criterion)]
        predicted, _ = apply_policy(
            pair["base_intervals"], by_pair_candidates.get((video_id, criterion), []),
            adapter_rows, a_threshold, b_threshold,
            action_window=action_window,
        )
        rows.append({
            "video_id": video_id, "criterion": criterion,
            "a_threshold": a_threshold, "b_threshold": b_threshold,
            **temporal_metrics(predicted, pair["truth_intervals"]),
        })
    return rows


def select_policy(
    criterion: str, video_ids: list[int], pair_data: dict[tuple[int, str], dict[str, Any]],
    by_pair_candidates: dict[tuple[int, str], list[dict[str, Any]]],
    adapter_rows: dict[str, dict[str, Any]], action_window: str = "candidate_interval",
) -> dict[str, Any]:
    baseline = summarize_rows([
        {"video_id": video_id, "criterion": criterion, **temporal_metrics(
            pair_data[(video_id, criterion)]["base_intervals"],
            pair_data[(video_id, criterion)]["truth_intervals"],
        )} for video_id in video_ids
    ])
    candidates = []
    for a_threshold in A_THRESHOLDS:
        for b_threshold in B_THRESHOLDS:
            rows = policy_rows(
                criterion, video_ids, pair_data, by_pair_candidates, adapter_rows,
                a_threshold, b_threshold,
                action_window,
            )
            summary = summarize_rows(rows)
            # Preserve baseline positive-pair detection during calibration.  This makes
            # the semantic adapter a precision enhancer unless a rescue compensates.
            eligible = (
                summary["positive_pair_detection_rate"] + 1e-12
                >= baseline["positive_pair_detection_rate"]
            )
            balanced_objective = (
                summary["macro_positive_pair_temporal_iou"]
                + summary["positive_pair_detection_rate"]
                + summary["negative_pair_rejection_rate"]
            ) / 3.0
            candidates.append({
                "a_threshold": a_threshold, "b_threshold": b_threshold,
                "eligible": eligible, "objective": balanced_objective,
                "summary": summary,
            })
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        raise RuntimeError(f"No recall-preserving policy for {criterion}")
    selected = max(
        eligible,
        key=lambda item: (
            item["objective"], item["summary"]["macro_temporal_iou"],
            -item["a_threshold"], item["b_threshold"],
        ),
    )
    return {**selected, "baseline_summary": baseline}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-5s", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--adapter-oof", type=Path, required=True)
    parser.add_argument(
        "--state-criteria", nargs="+", choices=CRITERIA, default=list(STATE_CRITERIA),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action-window", choices=("candidate_interval", "pack_window"),
        default="candidate_interval",
    )
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    train_ids = [int(value) for value in split["train_video_ids"]]
    dataset_root = Path(split["dataset_root"])
    annotation_path = dataset_root / "annotations" / "cholec80-CVS.xlsx"
    manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter_oof.read_text(encoding="utf-8"))
    adapter_rows = {item["candidate_id"]: item for item in adapter["candidate_rows"]}
    state_candidates = [
        item for item in manifest["candidates"] if item["criterion"] in args.state_criteria
    ]
    if set(adapter_rows) != {item["candidate_id"] for item in state_candidates}:
        raise ValueError("Adapter rows and state candidate manifest do not match")
    by_pair_candidates: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in state_candidates:
        by_pair_candidates[(int(candidate["video_id"]), candidate["criterion"])].append(candidate)

    model, metadata = load_calibrator(args.checkpoint)
    thresholds = metadata["thresholds_calibrated_on_validation"]
    pair_data: dict[tuple[int, str], dict[str, Any]] = {}
    annotation_presence = {}
    for progress, video_id in enumerate(train_ids, start=1):
        value = torch.load(
            args.cache_5s / f"video{video_id:02d}.pt", map_location="cpu", weights_only=False,
        )
        scores = predict(value, model, metadata)
        timestamps = [float(item) for item in value["timestamps_s"].tolist()]
        start_s, end_s = float(value["window"]["start_s"]), float(value["window"]["end_s"])
        cadence_s = float(value["cadence_s"])
        annotations = load_cvs_intervals(annotation_path, video_id)
        annotation_presence[video_id] = any(annotations[criterion] for criterion in CRITERIA)
        for criterion_index, criterion in enumerate(CRITERIA):
            threshold = float(thresholds[criterion])
            points = [
                ScorePoint(time_s, float(score), float(score), 1.0)
                for time_s, score in zip(timestamps, scores[:, criterion_index].tolist())
            ]
            _, evidence = StableEvidenceAggregator(
                **TEMPORAL, on_threshold=threshold,
                off_threshold=max(0.05, 0.8 * threshold),
            ).aggregate_structured(points)
            base = expand_aggregated_intervals(
                evidence.positive_intervals, cadence_s, start_s, end_s,
            )
            pair_data[(video_id, criterion)] = {
                "base_intervals": base,
                "truth_intervals": truth_intervals(
                    annotations, criterion, start_s, end_s,
                ),
            }
        print(f"baseline_cache {progress}/{len(train_ids)} video={video_id:02d}", flush=True)

    labeled_ids = [video_id for video_id in train_ids if annotation_presence[video_id]]
    unlabeled_ids = sorted(set(train_ids) - set(labeled_ids))
    baseline_rows = [
        {
            "video_id": video_id, "criterion": criterion,
            **temporal_metrics(
                pair_data[(video_id, criterion)]["base_intervals"],
                pair_data[(video_id, criterion)]["truth_intervals"],
            ),
        }
        for video_id in labeled_ids for criterion in CRITERIA
    ]

    nested_rows_by_key = {
        (row["video_id"], row["criterion"]): dict(row) for row in baseline_rows
    }
    fold_policies = []
    for progress, heldout_id in enumerate(labeled_ids, start=1):
        inner_ids = [video_id for video_id in labeled_ids if video_id != heldout_id]
        for criterion in args.state_criteria:
            selected = select_policy(
                criterion, inner_ids, pair_data, by_pair_candidates, adapter_rows,
                args.action_window,
            )
            heldout_row = policy_rows(
                criterion, [heldout_id], pair_data, by_pair_candidates, adapter_rows,
                selected["a_threshold"], selected["b_threshold"],
                args.action_window,
            )[0]
            nested_rows_by_key[(heldout_id, criterion)] = heldout_row
            fold_policies.append({
                "heldout_video_id": heldout_id, "criterion": criterion,
                "a_threshold": selected["a_threshold"],
                "b_threshold": selected["b_threshold"],
                "inner_objective": selected["objective"],
            })
        print(f"nested_calibration {progress}/{len(labeled_ids)} video={heldout_id:02d}", flush=True)
    nested_rows = [
        nested_rows_by_key[(video_id, criterion)]
        for video_id in labeled_ids for criterion in CRITERIA
    ]

    final_policies = {
        criterion: select_policy(
            criterion, labeled_ids, pair_data, by_pair_candidates, adapter_rows,
            args.action_window,
        ) for criterion in args.state_criteria
    }
    final_rows_by_key = {
        (row["video_id"], row["criterion"]): dict(row) for row in baseline_rows
    }
    policy_audit = {}
    for criterion, selected in final_policies.items():
        rows = policy_rows(
            criterion, labeled_ids, pair_data, by_pair_candidates, adapter_rows,
            selected["a_threshold"], selected["b_threshold"],
            args.action_window,
        )
        for row in rows:
            final_rows_by_key[(row["video_id"], criterion)] = row
        rejected_a, accepted_b = [], []
        for video_id in labeled_ids:
            pair = pair_data[(video_id, criterion)]
            _, audit = apply_policy(
                pair["base_intervals"], by_pair_candidates.get((video_id, criterion), []),
                adapter_rows, selected["a_threshold"], selected["b_threshold"],
                action_window=args.action_window,
            )
            rejected_a.extend(audit["rejected_A"])
            accepted_b.extend(audit["accepted_B"])
        policy_audit[criterion] = {
            "a_threshold": selected["a_threshold"],
            "b_threshold": selected["b_threshold"],
            "training_objective": selected["objective"],
            "rejected_A_count": len(rejected_a), "accepted_B_count": len(accepted_b),
            "rejected_A_ids": rejected_a, "accepted_B_ids": accepted_b,
        }
    final_rows = [
        final_rows_by_key[(video_id, criterion)]
        for video_id in labeled_ids for criterion in CRITERIA
    ]

    # Conservative agreement policy motivated only by OOF training distributions:
    # preserve temporal A unless the visual state is an extreme contradiction;
    # rescue B only when both temporal peak and visual persistence are strong.
    fixed_safe_policy = {
        "two_structures": {
            # OOF mask geometry shows a clean extreme-negative tail: the first
            # six A candidates below 0.18 are all non-full, while the lowest
            # full A candidate is 0.239.  Keep B rescue disabled.
            "a_threshold": 0.18, "b_threshold": 1.01, "b_peak_threshold": 0.0,
        },
        "cystic_plate": {
            "a_threshold": 0.0, "b_threshold": 0.70, "b_peak_threshold": 1.25,
        },
        "hepatocystic_triangle": {
            "a_threshold": 0.02, "b_threshold": 1.01, "b_peak_threshold": 0.0,
        },
    }
    fixed_rows_by_key = {
        (row["video_id"], row["criterion"]): dict(row) for row in baseline_rows
    }
    fixed_action_audit = {}
    for criterion, policy in fixed_safe_policy.items():
        if criterion not in args.state_criteria:
            continue
        criterion_rows = []
        rejected_a, accepted_b = [], []
        for video_id in labeled_ids:
            pair = pair_data[(video_id, criterion)]
            predicted, audit = apply_policy(
                pair["base_intervals"], by_pair_candidates.get((video_id, criterion), []),
                adapter_rows, policy["a_threshold"], policy["b_threshold"],
                policy["b_peak_threshold"], action_window=args.action_window,
            )
            row = {
                "video_id": video_id, "criterion": criterion,
                **temporal_metrics(predicted, pair["truth_intervals"]),
            }
            fixed_rows_by_key[(video_id, criterion)] = row
            rejected_a.extend(audit["rejected_A"])
            accepted_b.extend(audit["accepted_B"])
        fixed_action_audit[criterion] = {
            **policy, "rejected_A_count": len(rejected_a),
            "accepted_B_count": len(accepted_b),
            "rejected_A_ids": rejected_a, "accepted_B_ids": accepted_b,
        }
    fixed_rows = [
        fixed_rows_by_key[(video_id, criterion)]
        for video_id in labeled_ids for criterion in CRITERIA
    ]

    summaries = {
        "M2c_labeled_train_baseline": method_summary(baseline_rows, 20260731, 2000),
        "M2c_qwen_adapter_nested_video_oof": method_summary(nested_rows, 20260731, 2000),
        "M2c_qwen_adapter_global_train_fit_diagnostic": method_summary(final_rows, 20260731, 2000),
        "M2c_qwen_adapter_fixed_safe_agreement": method_summary(fixed_rows, 20260731, 2000),
    }
    output = {
        "schema_version": "m2c_qwen_visual_adapter_fusion_calibration_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "training_split_nested_video_oof",
        "action_window": args.action_window,
        "test_labels_accessed": False, "test_video_ids_loaded": [],
        "labeled_train_video_ids": labeled_ids,
        "excluded_unannotated_video_ids": unlabeled_ids,
        "evidence_score": "0.5*mean_frame_probability + 0.5*positive_frame_count/13",
        "calibration_constraint": "inner positive-pair detection >= inner M2c baseline",
        "conservative_A_veto_cap": 0.10,
        "calibration_objective": "mean(positive temporal IoU, positive detection, negative rejection)",
        "threshold_grid": {"A": list(A_THRESHOLDS), "B": list(B_THRESHOLDS)},
        "final_frozen_policy_candidates": policy_audit,
        "fixed_safe_agreement_policy_candidate": fixed_action_audit,
        "fold_policy_counts": {
            criterion: dict(Counter(
                f"A={item['a_threshold']:.2f},B={item['b_threshold']:.2f}"
                for item in fold_policies if item["criterion"] == criterion
            )) for criterion in args.state_criteria
        },
        "summaries": summaries,
        "rows": {
            "baseline": baseline_rows,
            "nested_video_oof": nested_rows,
            "global_train_fit_diagnostic": final_rows,
            "fixed_safe_agreement": fixed_rows,
        },
        "fold_policies": fold_policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summaries": summaries,
        "final_frozen_policy_candidates": policy_audit,
        "fixed_safe_agreement_policy_candidate": fixed_action_audit,
        "fold_policy_counts": output["fold_policy_counts"],
    }, indent=2), flush=True)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
