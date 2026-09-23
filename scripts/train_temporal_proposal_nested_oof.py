#!/usr/bin/env python3
"""Train and audit a shared dense temporal rescue proposal model with nested video OOF."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.temporal_proposals import (
    TemporalProposalWindowVerifier,
    centered_windows,
)
from evaluate_dense_visual_topk_rescue import intervals_from_scores
from run_cholec80_validation_ablation import (
    CRITERIA,
    method_summary,
    paired_bootstrap_comparison,
    summarize_rows,
    temporal_metrics,
    truth_intervals,
)
from train_dense_residual_fusion_oof import state_at


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start_s, end_s in sorted(intervals):
        if end_s <= start_s:
            continue
        if merged and start_s <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end_s)
        else:
            merged.append([float(start_s), float(end_s)])
    return [(start_s, end_s) for start_s, end_s in merged]


def point_is_covered(time_s: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start_s - 1e-9 <= time_s <= end_s + 1e-9 for start_s, end_s in intervals)


@dataclass(frozen=True)
class ProposalDecoder:
    enabled: bool
    absolute_probability: float = 1.0
    within_video_percentile: float = 1.0
    minimum_consecutive_points: int = 999

    @property
    def is_baseline(self) -> bool:
        return not self.enabled


def decoder_space(frozen: dict[str, Any]) -> list[ProposalDecoder]:
    family = frozen["policy_family"]["decoder"]
    result = [ProposalDecoder(False)]
    result.extend(
        ProposalDecoder(True, float(probability), float(percentile), int(points))
        for probability in family["absolute_probability_choices"]
        for percentile in family["within_video_percentile_choices"]
        for points in family["minimum_consecutive_points_choices"]
    )
    return result


class Dataset:
    def __init__(self, frozen: dict[str, Any], cache_dir: Path) -> None:
        self.frozen = frozen
        sequence_path = Path(frozen["sequences"])
        if sha256_file(sequence_path) != frozen["sequences_sha256"]:
            raise ValueError("Frozen temporal proposal sequences changed")
        payload = torch.load(sequence_path, map_location="cpu", weights_only=False)
        if payload.get("annotation_files_loaded") is not False:
            raise ValueError("Proposal sequence input was not annotation-free")
        if payload.get("development_or_test_accessed") is not False:
            raise ValueError("Proposal sequence input reports development/test access")
        if payload["feature_names"] != frozen["policy_family"]["feature_names"]:
            raise ValueError("Frozen proposal feature order changed")
        split_path = Path(frozen["split_json"])
        if sha256_file(split_path) != frozen["split_json_sha256"]:
            raise ValueError("Frozen split changed")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = set(map(int, split["train_video_ids"]))
        development_ids = set(map(int, split["validation_video_ids"]))
        test_ids = set(map(int, split["test_video_ids"]))
        if train_ids & (development_ids | test_ids):
            raise ValueError("Train split overlaps development/test")

        self.folds: list[list[int]] = []
        for fold_row in frozen["fold_definition_files"]:
            path = Path(fold_row["path"])
            if sha256_file(path) != fold_row["sha256"]:
                raise ValueError("Frozen fold definition changed")
            fold = torch.load(path, map_location="cpu", weights_only=False)
            self.folds.append(sorted(map(int, fold["outer_heldout_video_ids"])))
        self.labeled_ids = sorted({video_id for fold in self.folds for video_id in fold})
        if len(self.labeled_ids) != sum(map(len, self.folds)):
            raise ValueError("Frozen outer folds overlap")
        if set(payload["video_ids"]) != set(self.labeled_ids):
            raise ValueError("Proposal inputs are not exact labeled OOF video scope")
        expected_folds = {
            video_id: f"fold{fold_index}"
            for fold_index, fold in enumerate(self.folds) for video_id in fold
        }

        radius = int(frozen["policy_family"]["model"]["window_radius_points"])
        self.sequences: dict[tuple[int, str], dict[str, Any]] = {}
        self.meta: dict[int, dict[str, float]] = {}
        self.baseline: dict[tuple[int, str], list[tuple[float, float]]] = {}
        for video_id in self.labeled_ids:
            cache = torch.load(
                cache_dir / f"video{video_id:02d}.pt", map_location="cpu", weights_only=False,
            )
            self.meta[video_id] = {
                "start_s": float(cache["window"]["start_s"]),
                "end_s": float(cache["window"]["end_s"]),
                "cadence_s": float(cache["cadence_s"]),
            }
        for row in payload["sequences"]:
            video_id, criterion = int(row["video_id"]), str(row["criterion"])
            if row["outer_oof_fold"] != expected_folds[video_id]:
                raise ValueError("Sequence evidence does not match frozen OOF fold")
            row = dict(row)
            row["windows"] = centered_windows(row["features"].float(), radius)
            key = (video_id, criterion)
            self.sequences[key] = row
            meta = self.meta[video_id]
            self.baseline[key] = intervals_from_scores(
                row["center_times_s"].numpy().astype(float).tolist(),
                row["m2c_probabilities"].numpy().astype(float),
                float(row["m2c_threshold"]), meta["cadence_s"],
                meta["start_s"], meta["end_s"],
            )

        # Temporal labels are opened only after scope/hash/fold/OOF integrity checks.
        annotation_path = Path(split["dataset_root"]) / "annotations" / "cholec80-CVS.xlsx"
        self.annotations = {
            video_id: load_cvs_intervals(annotation_path, video_id)
            for video_id in self.labeled_ids
        }
        self.truth = {
            (video_id, criterion): truth_intervals(
                self.annotations[video_id], criterion,
                self.meta[video_id]["start_s"], self.meta[video_id]["end_s"],
            )
            for video_id in self.labeled_ids for criterion in CRITERIA
        }
        for (video_id, criterion), row in self.sequences.items():
            states = torch.tensor([
                state_at(
                    self.annotations[video_id][criterion], float(time_s),
                ) for time_s in row["center_times_s"]
            ], dtype=torch.long)
            covered = torch.tensor([
                point_is_covered(float(time_s), self.baseline[(video_id, criterion)])
                for time_s in row["center_times_s"]
            ], dtype=torch.bool)
            row["states"] = states
            row["proposal_target"] = (states == 2) & ~covered
            row["proposal_negative"] = (states == 0) & ~covered

    def training_tensors(
        self, video_ids: set[int], criteria: set[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[str, int]]]:
        windows, targets, negative_hardness, groups = [], [], [], []
        for (video_id, criterion), row in self.sequences.items():
            if video_id not in video_ids or criterion not in criteria:
                continue
            eligible = row["proposal_target"] | row["proposal_negative"]
            if not eligible.any():
                continue
            windows.append(row["windows"][eligible])
            targets.append(row["proposal_target"][eligible].float())
            feature = row["features"][eligible]
            mask_rank = feature[:, 2]
            m2c_support = torch.sigmoid(feature[:, 0])
            negative_hardness.append(1.0 + 2.0 * mask_rank + 2.0 * m2c_support)
            groups.extend([(criterion, video_id)] * int(eligible.sum()))
        return torch.cat(windows), torch.cat(targets), torch.cat(negative_hardness), groups

    def predict(
        self, model: TemporalProposalWindowVerifier, video_ids: set[int],
        device: torch.device, batch_size: int = 1024,
    ) -> dict[tuple[int, str], np.ndarray]:
        result = {}
        model.eval()
        with torch.inference_mode():
            for key, row in self.sequences.items():
                if key[0] not in video_ids:
                    continue
                values = []
                for start in range(0, len(row["windows"]), batch_size):
                    logits = model(row["windows"][start:start + batch_size].to(device))
                    values.append(torch.sigmoid(logits).cpu())
                result[key] = torch.cat(values).numpy().astype(np.float64)
        return result

    def decode(
        self, key: tuple[int, str], scores: np.ndarray, decoder: ProposalDecoder,
    ) -> list[dict[str, Any]]:
        if decoder.is_baseline:
            return []
        row = self.sequences[key]
        times = row["center_times_s"].numpy().astype(float)
        meta = self.meta[key[0]]
        cadence = meta["cadence_s"]
        rank_threshold = float(np.quantile(scores, decoder.within_video_percentile))
        threshold = max(decoder.absolute_probability, rank_threshold)
        active = [
            bool(score >= threshold and not point_is_covered(time_s, self.baseline[key]))
            for time_s, score in zip(times, scores)
        ]
        runs: list[list[int]] = []
        current: list[int] = []
        max_gap = cadence * float(
            self.frozen["policy_family"]["decoder"][
                "maximum_contiguous_gap_cadence_multiplier"
            ]
        )
        for index, is_active in enumerate(active):
            contiguous = bool(
                current and times[index] - times[current[-1]] <= max_gap + 1e-9
            )
            if is_active:
                if current and not contiguous:
                    runs.append(current)
                    current = []
                current.append(index)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        half = cadence * float(
            self.frozen["policy_family"]["decoder"][
                "cell_half_width_cadence_multiplier"
            ]
        )
        proposals = []
        for run in runs:
            if len(run) < decoder.minimum_consecutive_points:
                continue
            interval = (
                max(meta["start_s"], float(times[run[0]]) - half),
                min(meta["end_s"], float(times[run[-1]]) + half),
            )
            if interval[1] <= interval[0]:
                continue
            proposals.append({
                "interval_s": [interval[0], interval[1]],
                "point_indices": run,
                "point_timestamps_s": [float(times[index]) for index in run],
                "proposal_probability_max": float(np.max(scores[run])),
                "proposal_probability_mean": float(np.mean(scores[run])),
                "effective_threshold": threshold,
            })
        return proposals

    def metric_rows(
        self, video_ids: set[int], proposals: dict[tuple[int, str], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        rows = []
        for video_id in sorted(video_ids):
            for criterion in CRITERIA:
                key = (video_id, criterion)
                predicted = [
                    *self.baseline[key],
                    *(tuple(item["interval_s"]) for item in proposals.get(key, [])),
                ]
                rows.append({
                    "video_id": video_id, "criterion": criterion,
                    **temporal_metrics(merge_intervals(predicted), self.truth[key]),
                })
        return rows

    def proposal_audit(
        self, video_ids: set[int], criterion: str,
        proposals: dict[tuple[int, str], list[dict[str, Any]]],
    ) -> dict[str, Any]:
        total = beneficial = harmful = neutral = 0
        beneficial_pairs: set[tuple[int, str]] = set()
        recoverable_pairs: set[tuple[int, str]] = set()
        labeled = []
        for video_id in sorted(video_ids):
            key = (video_id, criterion)
            base_metric = temporal_metrics(self.baseline[key], self.truth[key])["temporal_iou"]
            if self.truth[key] and base_metric < 1.0 - 1e-12:
                recoverable_pairs.add(key)
            for proposal in proposals.get(key, []):
                total += 1
                added = merge_intervals([
                    *self.baseline[key], tuple(proposal["interval_s"]),
                ])
                added_metric = temporal_metrics(added, self.truth[key])["temporal_iou"]
                utility = added_metric - base_metric
                label = "beneficial" if utility > 1e-12 else (
                    "harmful" if utility < -1e-12 else "neutral"
                )
                beneficial += label == "beneficial"
                harmful += label == "harmful"
                neutral += label == "neutral"
                if label == "beneficial":
                    beneficial_pairs.add(key)
                labeled.append({
                    "video_id": video_id,
                    "criterion": criterion,
                    "interval_s": proposal["interval_s"],
                    "utility_label": label,
                    "temporal_iou_delta_if_added_alone": utility,
                })
        precision = beneficial / total if total else 0.0
        coverage = len(beneficial_pairs) / len(recoverable_pairs) if recoverable_pairs else 0.0
        return {
            "candidate_count": total,
            "beneficial_candidate_count": beneficial,
            "harmful_candidate_count": harmful,
            "neutral_candidate_count": neutral,
            "beneficial_candidate_precision": precision,
            "recoverable_positive_pair_count": len(recoverable_pairs),
            "beneficial_positive_pair_count": len(beneficial_pairs),
            "beneficial_positive_pair_coverage_rate": coverage,
            "selection_objective": 0.5 * (precision + coverage),
            "beneficial_video_ids": sorted({video_id for video_id, _ in beneficial_pairs}),
            "labeled_candidate_audit": labeled,
        }


def train_model(
    dataset: Dataset, video_ids: set[int], device: torch.device, seed: int,
) -> tuple[TemporalProposalWindowVerifier, dict[str, Any]]:
    family = dataset.frozen["policy_family"]
    model_cfg, train_cfg = family["model"], family["training"]
    criteria = set(family["enabled_criteria"])
    windows, targets, negative_hardness, groups = dataset.training_tensors(video_ids, criteria)
    positives = targets > 0.5
    negatives = ~positives
    if not positives.any() or not negatives.any():
        raise ValueError("Temporal proposal training needs positive and negative samples")
    positive_mass = float(train_cfg["positive_sampling_mass"])
    weights = torch.zeros(len(targets), dtype=torch.float64)
    grouped_sampling = train_cfg.get("positive_sampling_strategy") == (
        "equal_mass_per_criterion_then_video_then_point"
    )
    if grouped_sampling:
        for target_value, total_mass in ((True, positive_mass), (False, 1.0 - positive_mass)):
            target_mask = positives if target_value else negatives
            present_criteria = sorted({
                groups[index][0] for index in range(len(groups)) if bool(target_mask[index])
            })
            for criterion in present_criteria:
                present_videos = sorted({
                    groups[index][1] for index in range(len(groups))
                    if bool(target_mask[index]) and groups[index][0] == criterion
                })
                for video_id in present_videos:
                    index = torch.tensor([
                        position for position, group in enumerate(groups)
                        if bool(target_mask[position]) and group == (criterion, video_id)
                    ], dtype=torch.long)
                    group_mass = total_mass / len(present_criteria) / len(present_videos)
                    if target_value:
                        weights[index] = group_mass / len(index)
                    else:
                        hardness = negative_hardness[index].double()
                        weights[index] = group_mass * hardness / hardness.sum()
    else:
        weights[positives] = positive_mass / int(positives.sum())
        raw_negative = negative_hardness[negatives].double()
        weights[negatives] = (1.0 - positive_mass) * raw_negative / raw_negative.sum()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = TemporalProposalWindowVerifier(
        feature_dim=windows.shape[-1], hidden_dim=int(model_cfg["hidden_dim"]),
        kernel_size=int(model_cfg["kernel_size"]), dropout=float(model_cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed * 1000)
    losses = []
    model.train()
    for _ in range(int(train_cfg["steps"])):
        index = torch.multinomial(
            weights, int(train_cfg["batch_size"]), replacement=True, generator=generator,
        )
        logits = model(windows[index].to(device))
        labels = targets[index].to(device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, {
        "train_video_ids": sorted(video_ids),
        "eligible_points": len(targets),
        "positive_points": int(positives.sum()),
        "negative_points": int(negatives.sum()),
        "mean_train_loss": float(np.mean(losses)),
        "final_train_loss": losses[-1],
    }


def decode_scope(
    dataset: Dataset, video_ids: set[int], scores: dict[tuple[int, str], np.ndarray],
    decoders: dict[str, ProposalDecoder],
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    proposals = {}
    for video_id in video_ids:
        for criterion, decoder in decoders.items():
            key = (video_id, criterion)
            proposals[key] = dataset.decode(key, scores[key], decoder)
    return proposals


def select_decoder(
    dataset: Dataset, video_ids: set[int], criterion: str,
    scores: dict[tuple[int, str], np.ndarray], choices: list[ProposalDecoder],
) -> dict[str, Any]:
    rows = []
    for decoder in choices:
        proposals = decode_scope(dataset, video_ids, scores, {criterion: decoder})
        audit = dataset.proposal_audit(video_ids, criterion, proposals)
        rows.append({"decoder": asdict(decoder), "audit": audit})
    selected = max(rows, key=lambda row: (
        row["audit"]["selection_objective"],
        row["audit"]["beneficial_candidate_count"],
        -row["audit"]["candidate_count"],
        row["decoder"]["absolute_probability"],
        row["decoder"]["within_video_percentile"],
        row["decoder"]["minimum_consecutive_points"],
    ))
    return {**selected, "all_decoder_audits": rows}


def objective(summary: dict[str, Any]) -> float:
    return float((
        summary["macro_positive_pair_temporal_iou"]
        + summary["positive_pair_detection_rate"]
        + summary["negative_pair_rejection_rate"]
    ) / 3.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--cache-5s", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    frozen = json.loads(args.frozen_policy.read_text(encoding="utf-8"))
    if frozen.get("temporal_labels_loaded_during_freeze") is not False:
        raise ValueError("Temporal proposal family was not frozen before labels")
    if frozen.get("development_or_test_accessed") is not False:
        raise ValueError("Frozen proposal policy reports development/test access")
    dataset = Dataset(frozen, args.cache_5s)
    device = torch.device(args.device)
    choices = decoder_space(frozen)
    enabled = frozen["policy_family"]["enabled_criteria"]
    labeled = set(dataset.labeled_ids)
    base_rows = dataset.metric_rows(labeled, {})
    nested_proposals: dict[tuple[int, str], list[dict[str, Any]]] = {}
    fold_audits = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for outer_index, outer_list in enumerate(dataset.folds):
        outer = set(outer_list)
        outer_train = labeled - outer
        seed = int(frozen["policy_family"]["training"]["seed"]) + outer_index * 10
        crossfit_inner = frozen["policy_family"]["nested_protocol"].get(
            "inner_validation"
        ) == "cross_fitted_predictions_over_all_four_outer_training_folds"
        selection_train_audits = []
        if crossfit_inner:
            inner_validation = set(outer_train)
            inner_scores: dict[tuple[int, str], np.ndarray] = {}
            for inner_fold_index, inner_fold_list in enumerate(dataset.folds):
                current_validation = outer_train & set(inner_fold_list)
                if not current_validation:
                    continue
                current_train = outer_train - current_validation
                selection_model, current_audit = train_model(
                    dataset, current_train, device, seed + inner_fold_index,
                )
                current_scores = dataset.predict(
                    selection_model, current_validation, device,
                )
                overlap = set(inner_scores) & set(current_scores)
                if overlap:
                    raise ValueError("Inner cross-fit predictions overlap")
                inner_scores.update(current_scores)
                selection_train_audits.append({
                    "inner_fold": f"fold{inner_fold_index}",
                    "inner_validation_video_ids": sorted(current_validation),
                    **current_audit,
                })
                del selection_model
            expected_keys = {
                key for key in dataset.sequences if key[0] in outer_train
            }
            if set(inner_scores) != expected_keys:
                raise ValueError("Inner cross-fitting did not score every outer-training sequence")
        else:
            inner_validation = set(dataset.folds[(outer_index + 1) % len(dataset.folds)])
            inner_train = labeled - outer - inner_validation
            selection_model, selection_train_audit = train_model(
                dataset, inner_train, device, seed,
            )
            inner_scores = dataset.predict(selection_model, inner_validation, device)
            selection_train_audits.append(selection_train_audit)
            del selection_model
        selections = {
            criterion: select_decoder(
                dataset, inner_validation, criterion, inner_scores, choices,
            ) for criterion in enabled
        }
        final_model, outer_train_audit = train_model(
            dataset, outer_train, device, seed + 9,
        )
        outer_scores = dataset.predict(final_model, outer, device)
        decoders = {
            criterion: ProposalDecoder(**selections[criterion]["decoder"])
            for criterion in enabled
        }
        current = decode_scope(dataset, outer, outer_scores, decoders)
        nested_proposals.update(current)
        checkpoint_path = args.output_dir / f"fold{outer_index}_temporal_proposal.pt"
        torch.save({
            "schema_version": "temporal_rescue_proposal_checkpoint_v1",
            "model_state": final_model.state_dict(),
            "model_config": frozen["policy_family"]["model"],
            "feature_names": frozen["policy_family"]["feature_names"],
            "outer_heldout_video_ids": outer_list,
            "selected_decoders": {name: asdict(value) for name, value in decoders.items()},
            "foundation_model_parameters_updated": False,
            "development_or_test_accessed": False,
        }, checkpoint_path)
        fold_audits.append({
            "fold": f"fold{outer_index}",
            "outer_heldout_video_ids": outer_list,
            "inner_validation_video_ids": sorted(inner_validation),
            "selection_train_audits": selection_train_audits,
            "outer_train_audit": outer_train_audit,
            "decoder_selections": selections,
            "outer_candidate_counts": {
                criterion: sum(
                    len(current.get((video_id, criterion), [])) for video_id in outer
                ) for criterion in enabled
            },
            "checkpoint": str(checkpoint_path.resolve()),
        })
        print(json.dumps({
            "outer_fold": outer_index,
            "heldout": len(outer),
            "selected_decoders": {name: asdict(value) for name, value in decoders.items()},
            "candidate_counts": fold_audits[-1]["outer_candidate_counts"],
        }), flush=True)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    nested_rows = dataset.metric_rows(labeled, nested_proposals)
    baseline_summary = method_summary(
        base_rows, int(frozen["policy_family"]["bootstrap_seed"]),
        int(frozen["policy_family"]["bootstrap_repetitions"]),
    )
    proposal_summary = method_summary(
        nested_rows, int(frozen["policy_family"]["bootstrap_seed"]),
        int(frozen["policy_family"]["bootstrap_repetitions"]),
    )
    comparison = paired_bootstrap_comparison(
        base_rows, nested_rows, int(frozen["policy_family"]["bootstrap_seed"]),
        int(frozen["policy_family"]["bootstrap_repetitions"]),
        "M2c_train_baseline", "temporal_proposal_nested_OOF_diagnostic",
    )
    criterion_audits = {
        criterion: dataset.proposal_audit(labeled, criterion, nested_proposals)
        for criterion in enabled
    }
    all_labeled_candidates = [
        row for audit in criterion_audits.values() for row in audit["labeled_candidate_audit"]
    ]
    beneficial = [row for row in all_labeled_candidates if row["utility_label"] == "beneficial"]
    quality_cfg = frozen["policy_family"]["proposal_quality_gate_for_qwen_review"]
    overall_precision = len(beneficial) / len(all_labeled_candidates) if all_labeled_candidates else 0.0
    quality_checks = {
        "beneficial_candidate_precision": overall_precision >= float(
            quality_cfg["beneficial_candidate_precision_floor"]
        ),
        "beneficial_candidate_count": len(beneficial) >= int(
            quality_cfg["beneficial_candidate_count_floor"]
        ),
        "beneficial_video_count": len({row["video_id"] for row in beneficial}) >= int(
            quality_cfg["beneficial_video_count_floor"]
        ),
        "beneficial_criterion_count": len({row["criterion"] for row in beneficial}) >= int(
            quality_cfg["beneficial_criterion_count_floor"]
        ),
    }
    quality_gate = {
        "passed": all(quality_checks.values()),
        "checks": quality_checks,
        "candidate_count": len(all_labeled_candidates),
        "beneficial_candidate_count": len(beneficial),
        "beneficial_candidate_precision": overall_precision,
        "beneficial_video_ids": sorted({row["video_id"] for row in beneficial}),
        "beneficial_criteria": sorted({row["criterion"] for row in beneficial}),
    }
    base_by_key = {(row["video_id"], row["criterion"]): row for row in base_rows}
    improved_pairs = [
        row for row in nested_rows
        if row["temporal_iou"] > base_by_key[(row["video_id"], row["criterion"])]["temporal_iou"] + 1e-12
    ]
    automatic_checks = {
        "objective_improved": objective(proposal_summary) > objective(baseline_summary) + 1e-12,
        "macro_temporal_iou_non_decreasing": proposal_summary["macro_temporal_iou"] + 1e-12 >= baseline_summary["macro_temporal_iou"],
        "positive_iou_improved": proposal_summary["macro_positive_pair_temporal_iou"] > baseline_summary["macro_positive_pair_temporal_iou"] + 1e-12,
        "positive_detection_non_decreasing": proposal_summary["positive_pair_detection_rate"] + 1e-12 >= baseline_summary["positive_pair_detection_rate"],
        "negative_rejection_non_decreasing": proposal_summary["negative_pair_rejection_rate"] + 1e-12 >= baseline_summary["negative_pair_rejection_rate"],
        "improvement_spans_two_videos": len({row["video_id"] for row in improved_pairs}) >= 2,
    }

    blind_candidates, audit_index = [], {}
    sequence_counter: dict[tuple[int, str], int] = {}
    for key, proposals in sorted(nested_proposals.items()):
        video_id, criterion = key
        for proposal in proposals:
            sequence_counter[key] = sequence_counter.get(key, 0) + 1
            candidate_id = f"v{video_id:02d}-{criterion}-P{sequence_counter[key]:02d}"
            blind_candidates.append({
                "candidate_id": candidate_id,
                "video_id": video_id,
                "criterion": criterion,
                "candidate_class": "temporal_proposal_rescue",
                "interval_s": proposal["interval_s"],
                "point_timestamps_s": proposal["point_timestamps_s"],
                "proposal_probability_max": proposal["proposal_probability_max"],
                "proposal_probability_mean": proposal["proposal_probability_mean"],
                "outer_oof_fold": dataset.sequences[key]["outer_oof_fold"],
            })
            match = next(
                row for row in criterion_audits[criterion]["labeled_candidate_audit"]
                if row["video_id"] == video_id and row["interval_s"] == proposal["interval_s"]
            )
            audit_index[candidate_id] = match
    proposal_artifact = {
        "schema_version": "temporal_rescue_proposals_nested_oof_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_policy": str(args.frozen_policy.resolve()),
        "frozen_policy_sha256": sha256_file(args.frozen_policy),
        "selection_uses_outer_fold_labels": False,
        "training_annotations_used_on_outer_training_videos_only": True,
        "foundation_model_parameters_updated": False,
        "development_or_test_accessed": False,
        "candidates": blind_candidates,
    }
    proposal_path = args.output_dir / "frozen_nested_oof_proposals.json"
    proposal_path.write_text(json.dumps(proposal_artifact, indent=2) + "\n", encoding="utf-8")

    result = {
        "schema_version": "temporal_rescue_proposal_nested_oof_result_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "post_development_failure_train_only_method_development",
        "frozen_policy": str(args.frozen_policy.resolve()),
        "frozen_policy_sha256": sha256_file(args.frozen_policy),
        "protocol": {
            "outer_video_OOF": True,
            "inner_decoder_selection": "next cyclic disjoint video fold",
            "criterion_id_used_as_model_feature": False,
            "foundation_model_parameters_updated": False,
            "development_reused": False,
            "test_accessed": False,
        },
        "baseline_summary": baseline_summary,
        "proposal_only_nested_oof_diagnostic": proposal_summary,
        "paired_comparison": comparison,
        "proposal_quality_gate_for_qwen_review": quality_gate,
        "automatic_performance_diagnostic": {
            "passed": all(automatic_checks.values()),
            "checks": automatic_checks,
            "improved_video_ids": sorted({row["video_id"] for row in improved_pairs}),
        },
        "criterion_proposal_audits": criterion_audits,
        "candidate_label_audit_by_id": audit_index,
        "fold_audits": fold_audits,
        "baseline_rows": base_rows,
        "proposal_nested_oof_rows": nested_rows,
        "frozen_proposal_artifact": str(proposal_path.resolve()),
        "proceed_to_frozen_qwen_review": quality_gate["passed"],
        "development_reused": False,
        "test_accessed": False,
    }
    result_path = args.output_dir / "train_nested_oof_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(result_path.resolve()),
        "proposal_quality_gate": quality_gate,
        "automatic_performance_passed": result["automatic_performance_diagnostic"]["passed"],
        "baseline_macro_iou": baseline_summary["macro_temporal_iou"],
        "proposal_macro_iou": proposal_summary["macro_temporal_iou"],
        "baseline_positive_iou": baseline_summary["macro_positive_pair_temporal_iou"],
        "proposal_positive_iou": proposal_summary["macro_positive_pair_temporal_iou"],
        "proceed_to_frozen_qwen_review": result["proceed_to_frozen_qwen_review"],
        "development_reused": False,
        "test_accessed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
