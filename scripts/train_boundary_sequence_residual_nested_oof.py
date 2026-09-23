#!/usr/bin/env python3
"""Strict nested-OOF boundary-aware sequence residual training (train only).

For every outer video fold, an existing disjoint fold is used as inner
validation.  Boundary margins, loss weights, epoch, and residual strength are
selected using only that inner validation fold.  The model is then retrained on
all outer-training videos and the untouched outer fold is scored once.

This script deliberately has no development-prediction argument.  It writes an
immutable train-OOF gate; a separate command may train/apply a final model only
if that gate passes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.boundary_supervision import (
    BoundaryTargetConfig,
    BoundaryZone,
    adjacent_sequence_losses,
    boundary_targets,
    validate_train_only_scope,
)
from cvs_assessment.visual_adapters import MultiFrameCriterionResidualFusion
from evaluate_dense_visual_topk_rescue import intervals_from_scores
from run_cholec80_validation_ablation import (
    CRITERIA,
    summarize_rows,
    temporal_metrics,
    truth_intervals,
)


RESIDUAL_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Candidate:
    name: str
    inner_margin_steps: float
    outer_margin_steps: float
    boundary_target: float
    partial_target: float
    boundary_weight: float
    partial_weight: float
    hard_negative_weight: float
    consistency_weight: float
    monotonicity_weight: float
    residual_regularization: float


DEFAULT_CANDIDATES = (
    Candidate("balanced_2x2", 2.0, 2.0, 0.50, 0.25, 1.5, 0.6, 2.0, 0.15, 0.10, 0.010),
    Candidate("wide_inner_3x2", 3.0, 2.0, 0.50, 0.25, 1.5, 0.6, 2.0, 0.20, 0.15, 0.015),
    Candidate("wide_outer_2x3_hard", 2.0, 3.0, 0.50, 0.20, 1.8, 0.5, 3.0, 0.20, 0.15, 0.020),
)


def sigmoid_from_residual(base: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    clipped = np.clip(np.asarray(base, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) + float(alpha) * residual
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def criterion_objective(summary: dict[str, Any]) -> float:
    return float((
        summary["macro_positive_pair_temporal_iou"]
        + summary["positive_pair_detection_rate"]
        + summary["negative_pair_rejection_rate"]
    ) / 3.0)


def mean_objective(by_criterion: dict[str, dict[str, Any]]) -> float:
    return float(np.mean([criterion_objective(by_criterion[name]) for name in CRITERIA]))


class Dataset:
    def __init__(
        self, split_path: Path, manifest_path: Path, feature_path: Path,
        cache_dir: Path,
    ) -> None:
        self.split_path = split_path
        self.manifest_path = manifest_path
        self.feature_path = feature_path
        self.split = json.loads(split_path.read_text(encoding="utf-8"))
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.cache = torch.load(feature_path, map_location="cpu", weights_only=False)
        if self.cache["manifest_payload_sha256"] != self.manifest["payload_sha256"]:
            raise ValueError("Feature cache/manifest mismatch")
        validate_train_only_scope(
            split_train_ids=self.split["train_video_ids"],
            split_development_ids=self.split["validation_video_ids"],
            split_test_ids=self.split["test_video_ids"],
            artifact_video_ids=self.manifest["train_video_ids"],
            artifact_test_ids_loaded=self.manifest.get("test_video_ids_loaded", []),
            artifact_test_labels_accessed=bool(self.manifest.get("test_labels_accessed", False)),
        )
        if self.manifest.get("annotation_files_loaded") is not False:
            raise ValueError("Dense visual manifest is not annotation-free")
        if self.cache.get("test_video_or_annotation_accessed") is not False:
            raise ValueError("Feature cache reports sealed-test access")

        self.train_ids = list(map(int, self.split["train_video_ids"]))
        self.records = list(self.manifest["records"])
        if {int(row["video_id"]) for row in self.records} != set(self.train_ids):
            raise ValueError("Dense records do not exactly cover training videos")
        self.features = self.cache["features"]
        self.criterion_embeddings = {
            name: value.float().cpu() for name, value in self.cache["criterion_embeddings"].items()
        }
        key_to_index = {
            (int(key[0]), round(float(key[1]), 4), str(key[2])): index
            for index, key in enumerate(self.cache["keys"])
        }
        self.feature_indices = torch.tensor([[
            key_to_index[(int(row["video_id"]), round(float(time_s), 4), row["criterion"])]
            for time_s in row["frame_timestamps_s"]
        ] for row in self.records], dtype=torch.long)
        score_by_key = {
            (int(row["video_id"]), row["criterion"], round(float(row["center_time_s"]), 4)):
            float(row["m2c_probability"])
            for row in self.records
        }
        self.base_sequences = torch.tensor([[
            score_by_key[(int(row["video_id"]), row["criterion"], round(float(time_s), 4))]
            for time_s in row["frame_timestamps_s"]
        ] for row in self.records], dtype=torch.float32)
        self.base_centers = torch.tensor(
            [float(row["m2c_probability"]) for row in self.records], dtype=torch.float32,
        )
        self.thresholds = torch.tensor(
            [float(row["m2c_threshold"]) for row in self.records], dtype=torch.float32,
        )
        self.groups: dict[tuple[int, str], list[int]] = {}
        for index, row in enumerate(self.records):
            self.groups.setdefault((int(row["video_id"]), row["criterion"]), []).append(index)
        for indices in self.groups.values():
            indices.sort(key=lambda index: float(self.records[index]["center_time_s"]))

        annotation_path = Path(self.split["dataset_root"]) / "annotations" / "cholec80-CVS.xlsx"
        self.annotations = {
            video_id: load_cvs_intervals(annotation_path, video_id) for video_id in self.train_ids
        }
        self.labeled_ids = sorted(
            video_id for video_id, current in self.annotations.items()
            if any(current[criterion] for criterion in CRITERIA)
        )
        if set(self.labeled_ids) != set(self.train_ids) - {1}:
            raise ValueError("Unexpected training annotation coverage")

        self.video_meta: dict[int, dict[str, float]] = {}
        for video_id in self.train_ids:
            payload = torch.load(
                cache_dir / f"video{video_id:02d}.pt", map_location="cpu", weights_only=False,
            )
            self.video_meta[video_id] = {
                "start_s": float(payload["window"]["start_s"]),
                "end_s": float(payload["window"]["end_s"]),
                "cadence_s": float(payload["cadence_s"]),
            }
            for criterion in CRITERIA:
                indices = self.groups[(video_id, criterion)]
                times = np.asarray(
                    [self.records[index]["center_time_s"] for index in indices], dtype=np.float64,
                )
                if not np.allclose(times, payload["timestamps_s"].numpy(), atol=1e-4):
                    raise ValueError(f"Timestamp mismatch video={video_id} criterion={criterion}")

    def indices_for_videos(self, video_ids: set[int]) -> list[int]:
        return [
            index for index, row in enumerate(self.records)
            if int(row["video_id"]) in video_ids
        ]

    def targets(self, candidate: Candidate) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.zeros(len(self.records), dtype=torch.float32)
        zones = torch.full(
            (len(self.records),), int(BoundaryZone.BACKGROUND), dtype=torch.long,
        )
        for (video_id, criterion), indices in self.groups.items():
            cadence = self.video_meta[video_id]["cadence_s"]
            config = BoundaryTargetConfig(
                cadence_s=cadence,
                inner_margin_steps=candidate.inner_margin_steps,
                outer_margin_steps=candidate.outer_margin_steps,
                boundary_target=candidate.boundary_target,
                partial_target=candidate.partial_target,
            )
            result = boundary_targets(
                [float(self.records[index]["center_time_s"]) for index in indices],
                self.annotations[video_id][criterion], config,
            )
            target[indices] = torch.from_numpy(result["targets"])
            zones[indices] = torch.from_numpy(result["zones"])
        return target, zones

    def baseline_rows(self, video_ids: set[int]) -> list[dict[str, Any]]:
        zeros = np.zeros(len(self.records), dtype=np.float64)
        return self.temporal_rows(video_ids, zeros, alpha=0.0)

    def temporal_rows(
        self, video_ids: set[int], residuals: np.ndarray, alpha: float,
    ) -> list[dict[str, Any]]:
        rows = []
        for video_id in sorted(video_ids):
            meta = self.video_meta[video_id]
            for criterion in CRITERIA:
                indices = self.groups[(video_id, criterion)]
                times = [float(self.records[index]["center_time_s"]) for index in indices]
                base = self.base_centers[indices].numpy().astype(np.float64)
                scores = sigmoid_from_residual(base, residuals[indices], alpha)
                threshold_values = {round(float(self.thresholds[index]), 8) for index in indices}
                if len(threshold_values) != 1:
                    raise ValueError("Multiple thresholds within video/criterion pair")
                predicted = intervals_from_scores(
                    times, scores, float(self.thresholds[indices[0]]), meta["cadence_s"],
                    meta["start_s"], meta["end_s"],
                )
                truth = truth_intervals(
                    self.annotations[video_id], criterion, meta["start_s"], meta["end_s"],
                )
                rows.append({
                    "video_id": video_id, "criterion": criterion,
                    **temporal_metrics(predicted, truth),
                })
        return rows


def summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        criterion: summarize_rows([row for row in rows if row["criterion"] == criterion])
        for criterion in CRITERIA
    }


def eligible_against_baseline(
    current: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]],
) -> bool:
    return all(
        current[name]["positive_pair_detection_rate"] + 1e-12
        >= baseline[name]["positive_pair_detection_rate"]
        and current[name]["negative_pair_rejection_rate"] + 1e-12
        >= baseline[name]["negative_pair_rejection_rate"]
        for name in CRITERIA
    )


def choose_strength(
    dataset: Dataset, video_ids: set[int], residuals: np.ndarray,
) -> dict[str, Any]:
    baseline_rows = dataset.baseline_rows(video_ids)
    baseline = summaries(baseline_rows)
    choices = []
    for alpha in RESIDUAL_STRENGTHS:
        current_rows = dataset.temporal_rows(video_ids, residuals, alpha)
        current = summaries(current_rows)
        choices.append({
            "alpha": alpha,
            "eligible": eligible_against_baseline(current, baseline),
            "objective": mean_objective(current),
            "positive_iou_mean": float(np.mean([
                current[name]["macro_positive_pair_temporal_iou"] for name in CRITERIA
            ])),
            "by_criterion": current,
        })
    selected = max(
        (choice for choice in choices if choice["eligible"]),
        key=lambda choice: (
            choice["objective"], choice["positive_iou_mean"], -choice["alpha"],
        ),
    )
    return {**selected, "baseline_by_criterion": baseline, "all_strengths": choices}


class Trainer:
    def __init__(self, dataset: Dataset, args: argparse.Namespace) -> None:
        self.dataset = dataset
        self.args = args
        self.device = torch.device(args.device)

    def model(self, seed: int) -> MultiFrameCriterionResidualFusion:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return MultiFrameCriterionResidualFusion(
            hidden_dim=self.args.hidden_dim,
            max_frames=int(self.dataset.manifest["frame_count"]),
        ).to(self.device)

    def gather(self, indices: list[int], targets: torch.Tensor):
        index = torch.tensor(indices, dtype=torch.long)
        return (
            self.dataset.features[self.dataset.feature_indices[index]].to(
                self.device, dtype=torch.float32, non_blocking=True,
            ),
            torch.stack([
                self.dataset.criterion_embeddings[self.dataset.records[i]["criterion"]]
                for i in indices
            ]).to(self.device),
            self.dataset.base_sequences[index].to(self.device),
            self.dataset.base_centers[index].to(self.device),
            self.dataset.thresholds[index].to(self.device),
            targets[index].to(self.device),
        )

    @torch.inference_mode()
    def predict(self, model: nn.Module, indices: list[int]) -> np.ndarray:
        model.eval()
        residuals = np.zeros(len(self.dataset.records), dtype=np.float64)
        dummy = torch.zeros(len(self.dataset.records))
        for start in range(0, len(indices), self.args.batch_size):
            current = indices[start:start + self.args.batch_size]
            panels, criteria, sequences, centers, thresholds, _ = self.gather(current, dummy)
            _, residual = model(panels, criteria, sequences, centers, thresholds)
            residuals[np.asarray(current, dtype=np.int64)] = residual.cpu().numpy()
        return residuals

    def windows(
        self, video_ids: set[int], targets: torch.Tensor, zones: torch.Tensor,
        candidate: Candidate,
    ) -> tuple[list[list[int]], torch.Tensor]:
        length = self.args.sequence_length
        stride = max(1, length // 2)
        windows, weights = [], []
        criterion_counts = {name: 0 for name in CRITERIA}
        staged: list[tuple[list[int], str, float]] = []
        for (video_id, criterion), indices in self.dataset.groups.items():
            if video_id not in video_ids or len(indices) < length:
                continue
            starts = list(range(0, len(indices) - length + 1, stride))
            if starts[-1] != len(indices) - length:
                starts.append(len(indices) - length)
            for start in starts:
                current = indices[start:start + length]
                current_targets = targets[current]
                current_zones = zones[current]
                base = self.dataset.base_centers[current]
                threshold = self.dataset.thresholds[current]
                has_transition = bool((current_targets[1:] - current_targets[:-1]).abs().max() >= 0.05)
                has_hard_negative = bool(((current_targets == 0) & (base >= threshold)).any())
                has_interior = bool((current_zones == int(BoundaryZone.INTERIOR_POSITIVE)).any())
                weight = 1.0 + 2.0 * has_transition + 1.5 * has_hard_negative + 0.5 * has_interior
                staged.append((current, criterion, weight))
                criterion_counts[criterion] += 1
        for current, criterion, weight in staged:
            windows.append(current)
            weights.append(weight / max(1, criterion_counts[criterion]))
        if not windows:
            raise ValueError("No training sequences were constructed")
        return windows, torch.tensor(weights, dtype=torch.float64)

    def train_epoch(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
        video_ids: set[int], targets: torch.Tensor, zones: torch.Tensor,
        candidate: Candidate, epoch: int, seed: int,
    ) -> dict[str, float]:
        model.train()
        windows, window_weights = self.windows(video_ids, targets, zones, candidate)
        points_per_batch = max(self.args.sequence_length, self.args.batch_size)
        windows_per_batch = max(1, points_per_batch // self.args.sequence_length)
        window_samples = max(windows_per_batch, math.ceil(self.args.epoch_samples / self.args.sequence_length))
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        sampled = torch.multinomial(
            window_weights, window_samples, replacement=True, generator=generator,
        ).tolist()
        loss_values: dict[str, list[float]] = {
            "total": [], "binary": [], "consistency": [], "monotonicity": [], "residual": [],
        }
        for start in range(0, len(sampled), windows_per_batch):
            chosen = [windows[position] for position in sampled[start:start + windows_per_batch]]
            flat = [index for window in chosen for index in window]
            panels, criteria, sequences, centers, thresholds, labels = self.gather(flat, targets)
            logits, residual = model(panels, criteria, sequences, centers, thresholds)
            point_weights = torch.ones_like(labels)
            current_zones = zones[flat].to(self.device)
            boundary = (
                (current_zones == int(BoundaryZone.INNER_BOUNDARY))
                | (current_zones == int(BoundaryZone.OUTER_BOUNDARY))
            )
            point_weights[boundary] *= candidate.boundary_weight
            point_weights[current_zones == int(BoundaryZone.PARTIAL)] *= candidate.partial_weight
            hard_negative = (labels == 0) & (centers >= thresholds)
            point_weights[hard_negative] *= candidate.hard_negative_weight
            binary = (
                nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none",
                ) * point_weights
            ).sum() / point_weights.sum().clamp_min(1.0)
            batch_sequences = len(chosen)
            sequence_losses = adjacent_sequence_losses(
                logits.reshape(batch_sequences, self.args.sequence_length),
                labels.reshape(batch_sequences, self.args.sequence_length),
            )
            residual_penalty = residual.square().mean()
            loss = (
                binary
                + candidate.consistency_weight * sequence_losses["consistency"]
                + candidate.monotonicity_weight * sequence_losses["monotonicity"]
                + candidate.residual_regularization * residual_penalty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for name, value in (
                ("total", loss), ("binary", binary),
                ("consistency", sequence_losses["consistency"]),
                ("monotonicity", sequence_losses["monotonicity"]),
                ("residual", residual_penalty),
            ):
                loss_values[name].append(float(value.detach()))
        return {name: float(np.mean(values)) for name, values in loss_values.items()}

    def select_candidate(
        self, candidate: Candidate, inner_train: set[int], inner_valid: set[int], seed: int,
    ) -> dict[str, Any]:
        targets, zones = self.dataset.targets(candidate)
        model = self.model(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.args.learning_rate, weight_decay=2e-3,
        )
        valid_indices = self.dataset.indices_for_videos(inner_valid)
        zero_residuals = np.zeros(len(self.dataset.records), dtype=np.float64)
        initial = choose_strength(self.dataset, inner_valid, zero_residuals)
        best = {
            "epoch": 0, "alpha": 0.0, "objective": initial["objective"],
            "positive_iou_mean": initial["positive_iou_mean"],
            "by_criterion": initial["by_criterion"],
        }
        history, stale = [], 0
        for epoch in range(1, self.args.max_epochs + 1):
            losses = self.train_epoch(
                model, optimizer, inner_train, targets, zones, candidate, epoch, seed,
            )
            residuals = self.predict(model, valid_indices)
            selected = choose_strength(self.dataset, inner_valid, residuals)
            item = {
                "epoch": epoch, "losses": losses,
                "selected_alpha": selected["alpha"],
                "eligible": selected["eligible"],
                "objective": selected["objective"],
                "positive_iou_mean": selected["positive_iou_mean"],
                "by_criterion": selected["by_criterion"],
            }
            history.append(item)
            print(json.dumps({"candidate": candidate.name, **item}), flush=True)
            key = (item["objective"], item["positive_iou_mean"], -item["selected_alpha"])
            best_key = (best["objective"], best["positive_iou_mean"], -best["alpha"])
            if item["eligible"] and key > best_key:
                best = {
                    "epoch": epoch, "alpha": float(item["selected_alpha"]),
                    "objective": float(item["objective"]),
                    "positive_iou_mean": float(item["positive_iou_mean"]),
                    "by_criterion": item["by_criterion"],
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.args.patience:
                break
        return {
            "candidate": asdict(candidate), "selection": best,
            "baseline_by_criterion": initial["baseline_by_criterion"], "history": history,
        }

    def retrain_outer(
        self, candidate: Candidate, epochs: int, outer_train: set[int], seed: int,
    ) -> tuple[nn.Module, list[dict[str, float]]]:
        targets, zones = self.dataset.targets(candidate)
        model = self.model(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.args.learning_rate, weight_decay=2e-3,
        )
        history = []
        for epoch in range(1, epochs + 1):
            losses = self.train_epoch(
                model, optimizer, outer_train, targets, zones, candidate, epoch, seed,
            )
            history.append({"epoch": epoch, **losses})
            print(json.dumps({"role": "outer_retrain", "epoch": epoch, **losses}), flush=True)
        return model, history


def load_candidates(path: Path | None) -> list[Candidate]:
    if path is None:
        return list(DEFAULT_CANDIDATES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Candidate(**row) for row in payload]


def gate_results(
    dataset: Dataset, baseline_rows: list[dict[str, Any]], corrected_rows: list[dict[str, Any]],
    fold_by_video: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_by_key = {(row["video_id"], row["criterion"]): row for row in baseline_rows}
    corrected_by_key = {(row["video_id"], row["criterion"]): row for row in corrected_rows}
    gates, frozen_rows = {}, []
    for criterion in CRITERIA:
        base_rows = [row for row in baseline_rows if row["criterion"] == criterion]
        current_rows = [row for row in corrected_rows if row["criterion"] == criterion]
        base, current = summarize_rows(base_rows), summarize_rows(current_rows)
        positive_improvements = [
            row for row in current_rows
            if row["truth_intervals"]
            and row["temporal_iou"] > baseline_by_key[(row["video_id"], criterion)]["temporal_iou"] + 1e-12
        ]
        supporting_videos = sorted({int(row["video_id"]) for row in positive_improvements})
        supporting_folds = sorted({fold_by_video[int(row["video_id"])] for row in positive_improvements})
        passed = bool(
            current["macro_positive_pair_temporal_iou"]
            > base["macro_positive_pair_temporal_iou"] + 1e-12
            and current["positive_pair_detection_rate"] + 1e-12
            >= base["positive_pair_detection_rate"]
            and current["negative_pair_rejection_rate"] + 1e-12
            >= base["negative_pair_rejection_rate"]
            and criterion_objective(current) > criterion_objective(base) + 1e-12
            and len(supporting_videos) >= 2
            and len(supporting_folds) >= 2
        )
        gates[criterion] = {
            "passed": passed, "baseline": base, "nested_outer_oof": current,
            "objective_delta": criterion_objective(current) - criterion_objective(base),
            "positive_iou_supporting_videos": supporting_videos,
            "positive_iou_supporting_folds": supporting_folds,
        }
        frozen_rows.extend(current_rows if passed else base_rows)
    frozen_rows.sort(key=lambda row: (int(row["video_id"]), CRITERIA.index(row["criterion"])))
    baseline_summary = summarize_rows(baseline_rows)
    framework_summary = summarize_rows(frozen_rows)
    enabled = [name for name in CRITERIA if gates[name]["passed"]]
    overall_passed = bool(
        enabled
        and framework_summary["macro_positive_pair_temporal_iou"]
        > baseline_summary["macro_positive_pair_temporal_iou"] + 1e-12
        and framework_summary["positive_pair_detection_rate"] + 1e-12
        >= baseline_summary["positive_pair_detection_rate"]
        and framework_summary["negative_pair_rejection_rate"] + 1e-12
        >= baseline_summary["negative_pair_rejection_rate"]
    )
    return {
        "passed": overall_passed,
        "enabled_criteria": enabled if overall_passed else [],
        "by_criterion": gates,
        "baseline_summary": baseline_summary,
        "framework_with_failed_criteria_fallback_summary": framework_summary,
    }, frozen_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--cache-5s", type=Path, required=True)
    parser.add_argument("--fold-definition-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-grid-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=14)
    parser.add_argument("--epoch-samples", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=241)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--outer-fold-limit", type=int)
    parser.add_argument("--candidate-limit", type=int)
    args = parser.parse_args()
    if args.sequence_length < 2:
        raise ValueError("sequence-length must be at least two")
    if not args.smoke and (args.outer_fold_limit is not None or args.candidate_limit is not None):
        raise ValueError("Fold/candidate limits are allowed only with --smoke")

    dataset = Dataset(args.split_json, args.manifest, args.features, args.cache_5s)
    candidates = load_candidates(args.candidate_grid_json)
    if args.candidate_limit is not None:
        candidates = candidates[:args.candidate_limit]
    definition_paths = sorted(args.fold_definition_dir.glob("fold*_best.pt"))
    fold_video_ids = [
        sorted(map(int, torch.load(path, map_location="cpu", weights_only=False)["heldout_video_ids"]))
        for path in definition_paths
    ]
    if len(fold_video_ids) != 5 or {video for fold in fold_video_ids for video in fold} != set(dataset.labeled_ids):
        raise ValueError("Invalid outer fold definition")
    fold_count = args.outer_fold_limit or len(fold_video_ids)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(dataset, args)
    all_residuals = np.zeros(len(dataset.records), dtype=np.float64)
    all_corrected = np.asarray(dataset.base_centers, dtype=np.float64).copy()
    fold_by_video: dict[int, str] = {}
    fold_audits = []

    for fold_index in range(fold_count):
        fold_name = f"fold{fold_index}"
        outer_heldout = set(fold_video_ids[fold_index])
        inner_fold_index = (fold_index + 1) % len(fold_video_ids)
        inner_valid = set(fold_video_ids[inner_fold_index])
        outer_train = set(dataset.labeled_ids) - outer_heldout
        inner_train = outer_train - inner_valid
        print(json.dumps({
            "outer_fold": fold_name, "outer_heldout": sorted(outer_heldout),
            "inner_validation_fold": f"fold{inner_fold_index}",
            "inner_validation": sorted(inner_valid), "inner_train_count": len(inner_train),
        }), flush=True)
        candidate_audits = []
        for candidate_index, candidate in enumerate(candidates):
            current = trainer.select_candidate(
                candidate, inner_train, inner_valid,
                args.seed + fold_index * 100 + candidate_index * 10,
            )
            candidate_audits.append(current)
        selected = max(candidate_audits, key=lambda item: (
            item["selection"]["objective"], item["selection"]["positive_iou_mean"],
            -item["selection"]["alpha"], -item["selection"]["epoch"],
        ))
        selected_candidate = Candidate(**selected["candidate"])
        selected_epoch = int(selected["selection"]["epoch"])
        selected_alpha = float(selected["selection"]["alpha"])
        model, retrain_history = trainer.retrain_outer(
            selected_candidate, selected_epoch, outer_train,
            args.seed + 10000 + fold_index,
        )
        checkpoint_path = args.output_dir / f"{fold_name}_outer_frozen.pt"
        torch.save({
            "schema_version": "boundary_sequence_residual_outer_model_v1",
            "model_state": model.state_dict(), "hidden_dim": args.hidden_dim,
            "outer_heldout_video_ids": sorted(outer_heldout),
            "inner_validation_video_ids": sorted(inner_valid),
            "selected_candidate": asdict(selected_candidate),
            "selected_epoch": selected_epoch, "selected_residual_strength": selected_alpha,
            "outer_labels_used_for_selection": False,
            "development_or_test_accessed": False,
        }, checkpoint_path)
        heldout_indices = dataset.indices_for_videos(outer_heldout)
        residuals = trainer.predict(model, heldout_indices)
        all_residuals[heldout_indices] = residuals[heldout_indices]
        base = dataset.base_centers[heldout_indices].numpy()
        all_corrected[heldout_indices] = sigmoid_from_residual(
            base, residuals[heldout_indices], selected_alpha,
        )
        for video_id in outer_heldout:
            fold_by_video[video_id] = fold_name
        heldout_rows = dataset.temporal_rows(outer_heldout, residuals, selected_alpha)
        audit = {
            "fold": fold_name, "outer_heldout_video_ids": sorted(outer_heldout),
            "inner_validation_fold": f"fold{inner_fold_index}",
            "inner_validation_video_ids": sorted(inner_valid),
            "candidate_audits": candidate_audits,
            "selected": selected, "outer_retrain_history": retrain_history,
            "checkpoint": str(checkpoint_path.resolve()),
            "outer_score_loaded_after_checkpoint_freeze": True,
            "outer_summary_by_criterion": summaries(heldout_rows),
        }
        fold_audits.append(audit)
        (args.output_dir / "nested_selection_partial.json").write_text(
            json.dumps({"completed_folds": fold_audits}, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps({
            "outer_fold_complete": fold_name, "selected_candidate": selected_candidate.name,
            "selected_epoch": selected_epoch, "selected_alpha": selected_alpha,
        }), flush=True)

    if args.smoke:
        output = {
            "schema_version": "boundary_sequence_residual_nested_smoke_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_outer_folds": fold_count, "fold_audits": fold_audits,
            "gate_valid": False, "reason": "smoke_run_not_all_outer_folds",
            "development_or_test_accessed": False,
        }
        path = args.output_dir / "smoke_result.json"
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(path.resolve()), "gate_valid": False}, indent=2))
        return

    labeled = set(dataset.labeled_ids)
    baseline_rows = dataset.baseline_rows(labeled)
    corrected_rows = []
    for fold_index, heldout in enumerate(fold_video_ids):
        fold_name = f"fold{fold_index}"
        audit = fold_audits[fold_index]
        alpha = float(audit["selected"]["selection"]["alpha"])
        corrected_rows.extend(dataset.temporal_rows(set(heldout), all_residuals, alpha))
    corrected_rows.sort(key=lambda row: (int(row["video_id"]), CRITERIA.index(row["criterion"])))
    gate, frozen_rows = gate_results(
        dataset, baseline_rows, corrected_rows, fold_by_video,
    )

    prediction_rows = []
    for index, row in enumerate(dataset.records):
        video_id = int(row["video_id"])
        if video_id not in labeled:
            continue
        audit = fold_audits[int(fold_by_video[video_id].replace("fold", ""))]
        alpha = float(audit["selected"]["selection"]["alpha"])
        prediction_rows.append({
            "reference_id": row["reference_id"], "video_id": video_id,
            "criterion": row["criterion"], "center_time_s": float(row["center_time_s"]),
            "m2c_probability": float(dataset.base_centers[index]),
            "m2c_threshold": float(dataset.thresholds[index]),
            "outer_oof_fold": fold_by_video[video_id],
            "model_never_saw_video": True,
            "outer_labels_used_for_model_or_policy_selection": False,
            "raw_visual_residual_logit": float(all_residuals[index]),
            "selected_residual_strength": alpha,
            "boundary_sequence_probability": float(all_corrected[index]),
        })
    predictions = {
        "schema_version": "boundary_sequence_residual_nested_outer_oof_predictions_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction_role": "training_strict_nested_outer_video_oof",
        "annotation_files_loaded_during_prediction": False,
        "training_annotations_loaded_for_training_only": True,
        "development_video_or_annotation_accessed": False,
        "test_video_or_annotation_accessed": False,
        "rows": prediction_rows,
    }
    predictions_path = args.output_dir / "predictions_outer_oof.json"
    predictions_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")

    core = {
        "schema_version": "boundary_sequence_residual_train_gate_v1",
        "split": str(args.split_json.resolve()),
        "manifest": str(args.manifest.resolve()),
        "features": str(args.features.resolve()),
        "protocol": {
            "outer_folds": 5,
            "inner_validation_rule": "next_existing_disjoint_outer_fold_rotated_by_outer_index",
            "selected_on_inner_only": [
                "boundary_margin", "loss_weights", "epoch", "residual_strength",
            ],
            "outer_labels_loaded_only_after_checkpoint_freeze": True,
            "development_labels_loaded": False,
            "test_set": "sealed_not_accessed",
        },
        "gate": gate,
        "fold_audits": fold_audits,
        "baseline_rows": baseline_rows,
        "nested_outer_oof_rows": corrected_rows,
        "frozen_fallback_rows": frozen_rows,
        "development_evaluation_authorized_by_gate": bool(gate["passed"]),
        "development_or_test_accessed": False,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"))
    frozen = {
        **core, "created_at": datetime.now(timezone.utc).isoformat(),
        "payload_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    frozen_path = args.output_dir / "frozen_train_gate.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "frozen_train_gate": str(frozen_path.resolve()),
        "passed": gate["passed"], "enabled_criteria": gate["enabled_criteria"],
        "baseline_positive_iou": gate["baseline_summary"]["macro_positive_pair_temporal_iou"],
        "framework_positive_iou": gate["framework_with_failed_criteria_fallback_summary"]["macro_positive_pair_temporal_iou"],
        "development_accessed": False, "test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
