#!/usr/bin/env python3
"""Train and evaluate the frozen direct interval model with strict nested OOF."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
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
    BoundaryTargetConfig, BoundaryZone, adjacent_sequence_losses, boundary_targets,
)
from cvs_assessment.spatial_roi_relation import TextConditionedDirectIntervalModel
from evaluate_dense_visual_topk_rescue import intervals_from_scores
from run_cholec80_validation_ablation import (
    CRITERIA, method_summary, paired_bootstrap_comparison, summarize_rows,
    temporal_metrics, truth_intervals,
)
from train_dense_residual_fusion_oof import state_at


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid_residual(base: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    base = np.clip(np.asarray(base, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(base / (1.0 - base)) + float(alpha) * np.asarray(residual)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def jsonable_config(config: tuple[int, float, float, float | None]) -> dict[str, Any]:
    epoch, alpha, offset, presence = config
    return {
        "epoch": int(epoch), "correction_strength": float(alpha),
        "threshold_offset": float(offset),
        "presence_veto_threshold": None if presence is None else float(presence),
        "is_baseline": bool(epoch == 0 and alpha == 0 and offset == 0 and presence is None),
    }


class DirectIntervalDataset:
    """Audited annotation-free inputs plus training-only supervision."""

    def __init__(self, frozen_path: Path) -> None:
        self.frozen_path = frozen_path
        self.frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if self.frozen.get("new_labels_loaded_during_freeze") is not False:
            raise ValueError("Direct interval policy was not safely frozen")
        if self.frozen.get("development_or_consumed_test_accessed") is not False:
            raise ValueError("Frozen policy accessed development/test")
        sequence_path = Path(self.frozen["sequences"])
        if sha256_file(sequence_path) != self.frozen["sequences_sha256"]:
            raise ValueError("Frozen sequence artifact changed")
        self.payload = torch.load(sequence_path, map_location="cpu", weights_only=False)
        if self.payload.get("annotation_files_loaded") is not False:
            raise ValueError("Sequence artifact is not annotation-free")
        if self.payload.get("development_or_test_accessed") is not False:
            raise ValueError("Sequence artifact accessed development/test")
        split_path = Path(self.frozen["split_json"])
        if sha256_file(split_path) != self.frozen["split_json_sha256"]:
            raise ValueError("Frozen split changed")
        self.split = json.loads(split_path.read_text(encoding="utf-8"))
        train = set(map(int, self.split["train_video_ids"]))
        development = set(map(int, self.split["validation_video_ids"]))
        test = set(map(int, self.split["test_video_ids"]))
        if train & development or train & test or development & test:
            raise ValueError("Formal split roles overlap")
        self.folds: list[list[int]] = []
        for row in self.frozen["fold_definition_files"]:
            path = Path(row["path"])
            if sha256_file(path) != row["sha256"]:
                raise ValueError("Frozen outer-fold definition changed")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.folds.append(sorted(map(int, payload["outer_heldout_video_ids"])))
        self.labeled_ids = sorted({video for fold in self.folds for video in fold})
        if len(self.folds) != 5 or len(self.labeled_ids) != sum(map(len, self.folds)):
            raise ValueError("Outer folds must be five nonoverlapping groups")
        if set(self.labeled_ids) != train - {1}:
            raise ValueError("Unexpected labeled train coverage")
        if set(map(int, self.payload["video_ids"])) != set(self.labeled_ids):
            raise ValueError("Spatial sequences do not exactly cover labeled train")
        expected_fold = {
            video: f"fold{index}" for index, fold in enumerate(self.folds) for video in fold
        }
        self.sequences = list(self.payload["sequences"])
        for row in self.sequences:
            video = int(row["video_id"])
            if row["outer_oof_fold"] != expected_fold[video]:
                raise ValueError("Sequence OOF provenance disagrees with frozen folds")
        keys = [(int(row["video_id"]), str(row["criterion"])) for row in self.sequences]
        if len(keys) != len(set(keys)) or set(name for _, name in keys) != set(CRITERIA):
            raise ValueError("Expected one sequence per labeled video and criterion")

        self.criteria = list(self.frozen["criteria"])
        if set(self.criteria) != set(CRITERIA):
            raise ValueError("Frozen criterion set changed")
        self.embedding = {
            name: self.payload["criterion_embeddings"][name].float() for name in self.criteria
        }
        global_count = int(self.payload["token_layout"]["global_grid_tokens"])
        semantic = list(self.payload["semantic_types"])
        roi_classes = self.payload["task_visual_evidence_schema"]["criterion_roi_classes"]
        self.relevance = {}
        for name in self.criteria:
            relevant = set(roi_classes[name])
            self.relevance[name] = torch.tensor(
                [1.0] * global_count + [float(item in relevant) for item in semantic],
                dtype=torch.float16,
            )

        # Labels are opened only after every feature, split, hash, scope, and
        # OOF-provenance assertion above has succeeded.
        annotation_path = Path(self.split["dataset_root"]) / "annotations" / "cholec80-CVS.xlsx"
        self.annotations = {
            video: load_cvs_intervals(annotation_path, video) for video in self.labeled_ids
        }
        metadata_cache = ROOT / "cache" / "cholec80_m2b_spatial_5s"
        video_metadata = {}
        for video in self.labeled_ids:
            cached = torch.load(
                metadata_cache / f"video{video:02d}.pt",
                map_location="cpu", weights_only=False,
            )
            video_metadata[video] = cached
        target_cfg = self.frozen["training"]["boundary_targets"]
        self.supervision: list[dict[str, torch.Tensor | bool]] = []
        self.meta: list[dict[str, float]] = []
        for row in self.sequences:
            video, criterion = int(row["video_id"]), str(row["criterion"])
            times = np.asarray(row["center_times_s"], dtype=np.float64)
            cached = video_metadata[video]
            cached_times = cached["timestamps_s"].numpy().astype(np.float64)
            if len(times) != len(cached_times) or not np.allclose(times, cached_times, atol=1e-4):
                raise ValueError(f"M2c window metadata mismatch for video {video}")
            cadence = float(cached["cadence_s"])
            meta = {
                "cadence_s": cadence,
                "start_s": float(cached["window"]["start_s"]),
                "end_s": float(cached["window"]["end_s"]),
            }
            self.meta.append(meta)
            generated = boundary_targets(
                times.tolist(), self.annotations[video][criterion],
                BoundaryTargetConfig(
                    cadence_s=cadence,
                    inner_margin_steps=float(target_cfg["inner_margin_steps"]),
                    outer_margin_steps=float(target_cfg["outer_margin_steps"]),
                    boundary_target=float(target_cfg["boundary_target"]),
                    partial_target=float(target_cfg["partial_target"]),
                ),
            )
            hard_state = torch.tensor([
                state_at(self.annotations[video][criterion], float(value)) >= 2
                for value in times
            ], dtype=torch.bool)
            edge = torch.zeros(len(times), 2, dtype=torch.float32)
            starts = hard_state & ~torch.cat([torch.zeros(1, dtype=torch.bool), hard_state[:-1]])
            ends = hard_state & ~torch.cat([hard_state[1:], torch.zeros(1, dtype=torch.bool)])
            edge[:, 0] = starts.float()
            edge[:, 1] = ends.float()
            self.supervision.append({
                "target": torch.from_numpy(generated["targets"]),
                "zone": torch.from_numpy(generated["zones"]),
                "boundary": edge,
                "presence": bool(hard_state.any()),
            })

    def indices(self, videos: set[int], criterion: str | None = None) -> list[int]:
        return [
            index for index, row in enumerate(self.sequences)
            if int(row["video_id"]) in videos
            and (criterion is None or str(row["criterion"]) == criterion)
        ]

    def geometry(self, index: int, start: int, end: int) -> torch.Tensor:
        row = self.sequences[index]
        relevance = self.relevance[str(row["criterion"])][None, :, None].expand(
            end - start, -1, -1,
        )
        return torch.cat([row["geometry"][start:end], relevance], dim=-1)

    def truth(self, index: int) -> list[tuple[float, float]]:
        row, meta = self.sequences[index], self.meta[index]
        return truth_intervals(
            self.annotations[int(row["video_id"])], str(row["criterion"]),
            meta["start_s"], meta["end_s"],
        )

    def baseline_rows(self, videos: set[int]) -> list[dict[str, Any]]:
        rows = []
        for index in self.indices(videos):
            row, meta = self.sequences[index], self.meta[index]
            predicted = intervals_from_scores(
                list(map(float, row["center_times_s"])),
                row["m2c_probabilities"].numpy().astype(np.float64),
                float(row["m2c_threshold"]), meta["cadence_s"],
                meta["start_s"], meta["end_s"],
            )
            rows.append({
                "video_id": int(row["video_id"]), "criterion": str(row["criterion"]),
                **temporal_metrics(predicted, self.truth(index)),
            })
        return rows


class Trainer:
    def __init__(self, dataset: DirectIntervalDataset, device: str) -> None:
        self.dataset = dataset
        self.frozen = dataset.frozen
        self.device = torch.device(device)
        self.model_cfg = self.frozen["model"]
        self.train_cfg = self.frozen["training"]

    def new_model(self, seed: int) -> TextConditionedDirectIntervalModel:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        return TextConditionedDirectIntervalModel(
            token_dim=int(self.dataset.payload["token_dim"]),
            geometry_dim=len(self.dataset.payload["geometry_layout"]) + 1,
            text_dim=next(iter(self.dataset.embedding.values())).numel(),
            hidden_dim=int(self.model_cfg["hidden_dim"]),
            attention_heads=int(self.model_cfg["attention_heads"]),
            temporal_dilations=tuple(self.model_cfg["temporal_dilations"]),
            dropout=float(self.model_cfg["dropout"]),
        ).to(self.device)

    def chunks(self, indices: list[int]) -> list[tuple[int, int, int]]:
        length = int(self.train_cfg["chunk_length"])
        stride = int(self.train_cfg["chunk_stride"])
        output = []
        for index in indices:
            total = len(self.dataset.sequences[index]["center_times_s"])
            if total <= length:
                output.append((index, 0, total))
                continue
            starts = list(range(0, total - length + 1, stride))
            if starts[-1] != total - length:
                starts.append(total - length)
            output.extend((index, start, start + length) for start in starts)
        return output

    def collate(self, chunks: list[tuple[int, int, int]]) -> dict[str, torch.Tensor]:
        batch, maximum = len(chunks), max(end - start for _, start, end in chunks)
        example = self.dataset.sequences[chunks[0][0]]
        token_count, token_dim = example["tokens"].shape[1:]
        geometry_dim = example["geometry"].shape[-1] + 1
        tokens = torch.zeros(batch, maximum, token_count, token_dim, dtype=torch.float16)
        geometry = torch.zeros(batch, maximum, token_count, geometry_dim, dtype=torch.float16)
        base = torch.full((batch, maximum), 0.5, dtype=torch.float32)
        target = torch.zeros(batch, maximum, dtype=torch.float32)
        zones = torch.zeros(batch, maximum, dtype=torch.long)
        boundary = torch.zeros(batch, maximum, 2, dtype=torch.float32)
        progress = torch.zeros(batch, maximum, dtype=torch.float32)
        valid = torch.zeros(batch, maximum, dtype=torch.bool)
        embeddings, thresholds, presences = [], [], []
        for batch_index, (index, start, end) in enumerate(chunks):
            row = self.dataset.sequences[index]
            length, total = end - start, len(row["center_times_s"])
            tokens[batch_index, :length] = row["tokens"][start:end]
            geometry[batch_index, :length] = self.dataset.geometry(index, start, end)
            base[batch_index, :length] = row["m2c_probabilities"][start:end].float()
            supervision = self.dataset.supervision[index]
            target[batch_index, :length] = supervision["target"][start:end]
            zones[batch_index, :length] = supervision["zone"][start:end]
            boundary[batch_index, :length] = supervision["boundary"][start:end]
            progress[batch_index, :length] = torch.arange(start, end) / max(total - 1, 1)
            valid[batch_index, :length] = True
            embeddings.append(self.dataset.embedding[str(row["criterion"])])
            thresholds.append(float(row["m2c_threshold"]))
            presences.append(float(supervision["presence"]))
        return {
            "tokens": tokens.to(self.device, non_blocking=True),
            "geometry": geometry.to(self.device, non_blocking=True),
            "base": base.to(self.device, non_blocking=True),
            "target": target.to(self.device, non_blocking=True),
            "zones": zones.to(self.device, non_blocking=True),
            "boundary": boundary.to(self.device, non_blocking=True),
            "progress": progress.to(self.device, non_blocking=True),
            "valid": valid.to(self.device, non_blocking=True),
            "embedding": torch.stack(embeddings).to(self.device, non_blocking=True),
            "threshold": torch.tensor(thresholds, device=self.device),
            "presence": torch.tensor(presences, device=self.device),
        }

    def balance(self, indices: list[int]) -> dict[str, float]:
        targets = torch.cat([
            self.dataset.supervision[index]["target"] for index in indices
        ])
        positive = float((targets >= 0.5).sum())
        negative = float((targets < 0.5).sum())
        presence = [bool(self.dataset.supervision[index]["presence"]) for index in indices]
        boundary = torch.cat([
            self.dataset.supervision[index]["boundary"] for index in indices
        ])
        return {
            "point": min(8.0, max(1.0, negative / max(positive, 1.0))),
            "presence": min(8.0, max(1.0, (len(presence) - sum(presence)) / max(sum(presence), 1))),
            "boundary": min(20.0, max(1.0, (boundary.numel() - float(boundary.sum())) / max(float(boundary.sum()), 1.0))),
        }

    def train_epoch(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
        indices: list[int], epoch: int, seed: int,
    ) -> dict[str, float]:
        model.train()
        chunks = self.chunks(indices)
        rng = random.Random(seed * 1000 + epoch)
        rng.shuffle(chunks)
        balance = self.balance(indices)
        weights = self.train_cfg["loss_weights"]
        values: dict[str, list[float]] = defaultdict(list)
        batch_size = int(self.train_cfg["batch_size"])
        for start in range(0, len(chunks), batch_size):
            batch = self.collate(chunks[start:start + batch_size])
            output = model(
                batch["tokens"], batch["geometry"], batch["embedding"], batch["base"],
                batch["threshold"], batch["progress"], batch["valid"],
            )
            point_weight = torch.ones_like(batch["target"])
            point_weight[batch["target"] >= 0.5] *= balance["point"]
            boundary_zone = (
                (batch["zones"] == int(BoundaryZone.INNER_BOUNDARY))
                | (batch["zones"] == int(BoundaryZone.OUTER_BOUNDARY))
            )
            point_weight[boundary_zone] *= float(weights["boundary_zone_point"])
            point_weight[batch["zones"] == int(BoundaryZone.PARTIAL)] *= float(weights["partial_point"])
            hard_negative = (batch["target"] == 0) & (batch["base"] >= batch["threshold"][:, None])
            point_weight[hard_negative] *= float(weights["m2c_hard_negative_point"])
            point_terms = nn.functional.binary_cross_entropy_with_logits(
                output["frame_logits"], batch["target"], reduction="none",
            )
            point = (point_terms * point_weight * batch["valid"]).sum() / (
                point_weight * batch["valid"]
            ).sum().clamp_min(1)
            adjacent = adjacent_sequence_losses(
                output["frame_logits"], batch["target"], batch["valid"],
            )
            boundary_terms = nn.functional.binary_cross_entropy_with_logits(
                output["boundary_logits"], batch["boundary"], reduction="none",
                pos_weight=torch.tensor(balance["boundary"], device=self.device),
            )
            boundary_loss = (
                boundary_terms * batch["valid"][..., None]
            ).sum() / (batch["valid"].sum() * 2).clamp_min(1)
            presence_loss = nn.functional.binary_cross_entropy_with_logits(
                output["presence_logits"], batch["presence"],
                pos_weight=torch.tensor(balance["presence"], device=self.device),
            )
            residual = (
                output["frame_residual"].square() * batch["valid"]
            ).sum() / batch["valid"].sum().clamp_min(1)
            loss = (
                point
                + float(weights["adjacent_consistency"]) * adjacent["consistency"]
                + float(weights["adjacent_monotonicity"]) * adjacent["monotonicity"]
                + float(weights["onset_offset"]) * boundary_loss
                + float(weights["pair_presence"]) * presence_loss
                + float(weights["residual_l2"]) * residual
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(self.train_cfg["gradient_clip_norm"]))
            optimizer.step()
            for name, item in (
                ("total", loss), ("point", point), ("consistency", adjacent["consistency"]),
                ("monotonicity", adjacent["monotonicity"]), ("boundary", boundary_loss),
                ("presence", presence_loss), ("residual", residual),
            ):
                values[name].append(float(item.detach()))
        return {name: float(np.mean(current)) for name, current in values.items()}

    @torch.inference_mode()
    def predict(self, model: nn.Module, indices: list[int]) -> dict[int, dict[str, Any]]:
        model.eval()
        output: dict[int, dict[str, Any]] = {}
        for index in indices:
            row = self.dataset.sequences[index]
            batch = self.collate([(index, 0, len(row["center_times_s"]))])
            current = model(
                batch["tokens"], batch["geometry"], batch["embedding"], batch["base"],
                batch["threshold"], batch["progress"], batch["valid"],
            )
            length = len(row["center_times_s"])
            output[index] = {
                "residual": current["frame_residual"][0, :length].cpu().numpy(),
                "presence_probability": float(torch.sigmoid(current["presence_logits"][0]).cpu()),
                "onset_probability": torch.sigmoid(current["boundary_logits"][0, :length, 0]).cpu().numpy(),
                "offset_probability": torch.sigmoid(current["boundary_logits"][0, :length, 1]).cpu().numpy(),
            }
        return output


def decoded_rows(
    dataset: DirectIntervalDataset, videos: set[int],
    predictions: dict[int, dict[str, Any]],
    configs: dict[str, tuple[int, float, float, float | None]],
    risk_config: tuple[float, int] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index in dataset.indices(videos):
        source, meta = dataset.sequences[index], dataset.meta[index]
        criterion = str(source["criterion"])
        _, alpha, offset, presence_threshold = configs[criterion]
        probability = sigmoid_residual(
            source["m2c_probabilities"].numpy(), predictions[index]["residual"], alpha,
        )
        threshold = float(np.clip(float(source["m2c_threshold"]) + offset, 0.05, 0.95))
        accepted = (
            presence_threshold is None
            or predictions[index]["presence_probability"] >= presence_threshold
        )
        baseline = intervals_from_scores(
            list(map(float, source["center_times_s"])),
            source["m2c_probabilities"].numpy().astype(np.float64),
            float(source["m2c_threshold"]), meta["cadence_s"],
            meta["start_s"], meta["end_s"],
        )
        predicted = intervals_from_scores(
            list(map(float, source["center_times_s"])), probability, threshold,
            meta["cadence_s"], meta["start_s"], meta["end_s"],
        ) if accepted else []
        if risk_config is not None and bool(predicted) != bool(baseline):
            threshold_cap, duration_points_cap = risk_config
            baseline_duration_points = sum(
                end - start for start, end in baseline
            ) / meta["cadence_s"]
            removal_permitted = bool(
                baseline and not predicted
                and float(source["m2c_threshold"]) <= threshold_cap + 1e-12
                and baseline_duration_points < duration_points_cap
            )
            if not removal_permitted:
                predicted = baseline
        rows.append({
            "video_id": int(source["video_id"]), "criterion": criterion,
            **temporal_metrics(predicted, dataset.truth(index)),
        })
    return rows


def choose_risk_config(
    dataset: DirectIntervalDataset, videos: set[int],
    predictions: dict[int, dict[str, Any]],
    configs: dict[str, tuple[int, float, float, float | None]],
) -> dict[str, Any]:
    policy = dataset.frozen["presence_change_policy"]
    baseline_rows = dataset.baseline_rows(videos)
    baseline = summarize_rows(baseline_rows)
    rules = [(0.0, 0)] + [
        (float(threshold), int(duration))
        for threshold in policy["m2c_threshold_cap_choices"]
        for duration in policy["maximum_duration_points_choices"]
    ]
    choices = []
    for rule in rules:
        rows = decoded_rows(dataset, videos, predictions, configs, rule)
        summary = summarize_rows(rows)
        is_eligible = bool(
            summary["positive_pair_detection_rate"] + 1e-12 >= baseline["positive_pair_detection_rate"]
            and summary["negative_pair_rejection_rate"] + 1e-12 >= baseline["negative_pair_rejection_rate"]
            and summary["macro_positive_pair_temporal_iou"] + 1e-12 >= baseline["macro_positive_pair_temporal_iou"]
            and summary["macro_segment_f1_at_iou_0_3"] + 1e-12 >= baseline["macro_segment_f1_at_iou_0_3"]
            and summary["macro_positive_pair_segment_f1_at_iou_0_3"] + 1e-12
            >= baseline["macro_positive_pair_segment_f1_at_iou_0_3"]
        )
        choices.append({"rule": rule, "summary": summary, "eligible": is_eligible})
    selected = max(
        (row for row in choices if row["eligible"]),
        key=lambda row: (
            row["summary"]["macro_temporal_iou"],
            row["summary"]["macro_positive_pair_temporal_iou"],
            row["summary"]["macro_segment_f1_at_iou_0_3"],
            -row["rule"][0], -row["rule"][1],
        ),
    )
    return {
        "rule": selected["rule"], "summary": selected["summary"],
        "baseline": baseline,
        "rule_json": {
            "m2c_threshold_cap": selected["rule"][0],
            "maximum_duration_points": selected["rule"][1],
            "presence_locked": selected["rule"] == (0.0, 0),
        },
        "all_rules": [{
            "m2c_threshold_cap": row["rule"][0],
            "maximum_duration_points": row["rule"][1],
            "eligible": row["eligible"],
            "macro_temporal_iou": row["summary"]["macro_temporal_iou"],
            "macro_positive_pair_temporal_iou": row["summary"]["macro_positive_pair_temporal_iou"],
        } for row in choices],
    }


def eligible(current: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        current["positive_pair_detection_rate"] + 1e-12 >= baseline["positive_pair_detection_rate"]
        and current["negative_pair_rejection_rate"] + 1e-12 >= baseline["negative_pair_rejection_rate"]
        and current["macro_segment_f1_at_iou_0_3"] + 1e-12 >= baseline["macro_segment_f1_at_iou_0_3"]
        and current["macro_positive_pair_segment_f1_at_iou_0_3"] + 1e-12
        >= baseline["macro_positive_pair_segment_f1_at_iou_0_3"]
    )


def selection_key(summary: dict[str, Any], config: tuple[int, float, float, float | None]) -> tuple:
    epoch, alpha, offset, presence = config
    complexity = float(alpha) + abs(float(offset)) + (0 if presence is None else float(presence))
    return (
        float(summary["macro_temporal_iou"]),
        float(summary["macro_positive_pair_temporal_iou"]),
        float(summary["macro_segment_f1_at_iou_0_3"]),
        -complexity, -int(epoch),
    )


def choose_configs(
    dataset: DirectIntervalDataset, videos: set[int],
    predictions: dict[int, dict[str, Any]], epoch: int,
) -> dict[str, dict[str, Any]]:
    search = dataset.frozen["nested_selection"]
    output = {}
    for criterion in CRITERIA:
        baseline_rows = [
            row for row in dataset.baseline_rows(videos) if row["criterion"] == criterion
        ]
        baseline = summarize_rows(baseline_rows)
        baseline_config = (0, 0.0, 0.0, None)
        best = {
            "config": baseline_config, "summary": baseline,
            "eligible": True, "strict_improvement": False,
        }
        for alpha in search["correction_strengths"]:
            for offset in search["threshold_offsets"]:
                for presence in search["presence_veto_thresholds"]:
                    # A threshold-only calibration does not depend on the
                    # trained model and therefore correctly freezes at epoch 0.
                    effective_epoch = 0 if alpha == 0 and presence is None else epoch
                    config = (effective_epoch, float(alpha), float(offset), presence)
                    if alpha == 0 and offset == 0 and presence is None:
                        continue
                    rows = decoded_rows(
                        dataset, videos, predictions,
                        {name: config if name == criterion else baseline_config for name in CRITERIA},
                    )
                    current = summarize_rows([row for row in rows if row["criterion"] == criterion])
                    if not eligible(current, baseline):
                        continue
                    if selection_key(current, config) > selection_key(best["summary"], best["config"]):
                        best = {
                            "config": config, "summary": current, "eligible": True,
                            "strict_improvement": current["macro_temporal_iou"] > baseline["macro_temporal_iou"] + 1e-12,
                        }
        output[criterion] = {
            **best, "baseline": baseline,
            "config_json": jsonable_config(best["config"]),
        }
    return output


def update_best(
    best: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    for criterion in CRITERIA:
        if selection_key(current[criterion]["summary"], current[criterion]["config"]) > selection_key(
            best[criterion]["summary"], best[criterion]["config"],
        ):
            best[criterion] = deepcopy(current[criterion])
    return best


def paper_gate(
    baseline_rows: list[dict[str, Any]], framework_rows: list[dict[str, Any]],
    fold_by_video: dict[int, str], bootstrap: dict[str, Any], frozen: dict[str, Any],
) -> dict[str, Any]:
    baseline, framework = summarize_rows(baseline_rows), summarize_rows(framework_rows)
    base_by_key = {(int(row["video_id"]), row["criterion"]): row for row in baseline_rows}
    improved = [
        row for row in framework_rows
        if row["temporal_iou"] > base_by_key[(int(row["video_id"]), row["criterion"])]["temporal_iou"] + 1e-12
    ]
    videos = sorted({int(row["video_id"]) for row in improved})
    folds = sorted({fold_by_video[video] for video in videos})
    cfg = frozen["paper_gate"]
    ci = bootstrap["macro_temporal_iou"]
    checks = {
        "macro_temporal_iou_strictly_improves": framework["macro_temporal_iou"] > baseline["macro_temporal_iou"] + 1e-12,
        "macro_positive_pair_temporal_iou_not_lower": framework["macro_positive_pair_temporal_iou"] + 1e-12 >= baseline["macro_positive_pair_temporal_iou"],
        "macro_segment_f1_not_lower": framework["macro_segment_f1_at_iou_0_3"] + 1e-12 >= baseline["macro_segment_f1_at_iou_0_3"],
        "positive_segment_f1_not_lower": framework["macro_positive_pair_segment_f1_at_iou_0_3"] + 1e-12 >= baseline["macro_positive_pair_segment_f1_at_iou_0_3"],
        "positive_detection_not_lower": framework["positive_pair_detection_rate"] + 1e-12 >= baseline["positive_pair_detection_rate"],
        "negative_rejection_not_lower": framework["negative_pair_rejection_rate"] + 1e-12 >= baseline["negative_pair_rejection_rate"],
        "supporting_videos": len(videos) >= int(cfg["supporting_videos_min"]),
        "supporting_outer_folds": len(folds) >= int(cfg["supporting_outer_folds_min"]),
        "paired_macro_iou_lower_95_above_zero": ci["paired_video_bootstrap_lower_95"] > 0,
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "baseline": baseline, "framework": framework,
        "supporting_videos": videos, "supporting_outer_folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--outer-fold-limit", type=int)
    args = parser.parse_args()
    if not args.smoke and args.outer_fold_limit is not None:
        raise ValueError("outer-fold-limit is smoke-only")
    dataset = DirectIntervalDataset(args.frozen_policy)
    trainer = Trainer(dataset, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fold_count = args.outer_fold_limit or len(dataset.folds)
    prediction_by_sequence: dict[int, dict[str, Any]] = {}
    selected_by_fold: list[dict[str, Any]] = []
    fold_by_video: dict[int, str] = {}
    train_cfg = dataset.frozen["training"]
    selection_epochs = set(map(int, train_cfg["selection_epochs"]))
    seed_base = int(train_cfg["seed"])

    for outer_index in range(fold_count):
        outer_name = f"fold{outer_index}"
        heldout = set(dataset.folds[outer_index])
        inner_index = (outer_index + 1) % len(dataset.folds)
        inner_valid = set(dataset.folds[inner_index])
        outer_train = set(dataset.labeled_ids) - heldout
        inner_train = outer_train - inner_valid
        print(json.dumps({
            "outer_fold": outer_name, "heldout": sorted(heldout),
            "inner_validation_fold": f"fold{inner_index}",
            "inner_train_count": len(inner_train),
        }), flush=True)

        seed = seed_base + outer_index * 100
        model = trainer.new_model(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        base_selection = {}
        for criterion in CRITERIA:
            rows = [
                row for row in dataset.baseline_rows(inner_valid)
                if row["criterion"] == criterion
            ]
            summary = summarize_rows(rows)
            base_selection[criterion] = {
                "config": (0, 0.0, 0.0, None), "summary": summary,
                "baseline": summary, "eligible": True, "strict_improvement": False,
                "config_json": jsonable_config((0, 0.0, 0.0, None)),
            }
        best = deepcopy(base_selection)
        best_predictions: dict[int, dict[str, Any]] = {}
        inner_history = []
        for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
            losses = trainer.train_epoch(
                model, optimizer, dataset.indices(inner_train), epoch, seed,
            )
            item: dict[str, Any] = {"epoch": epoch, "losses": losses}
            if epoch in selection_epochs:
                predictions = trainer.predict(model, dataset.indices(inner_valid))
                choices = choose_configs(dataset, inner_valid, predictions, epoch)
                for criterion in CRITERIA:
                    if selection_key(
                        choices[criterion]["summary"], choices[criterion]["config"],
                    ) > selection_key(best[criterion]["summary"], best[criterion]["config"]):
                        best[criterion] = deepcopy(choices[criterion])
                        for index in dataset.indices(inner_valid, criterion):
                            best_predictions[index] = deepcopy(predictions[index])
                item["choices"] = {
                    name: {
                        "config": row["config_json"],
                        "macro_iou": row["summary"]["macro_temporal_iou"],
                        "positive_iou": row["summary"]["macro_positive_pair_temporal_iou"],
                        "segment_f1": row["summary"]["macro_segment_f1_at_iou_0_3"],
                    } for name, row in choices.items()
                }
            inner_history.append(item)
            print(json.dumps({
                "outer_fold": outer_name, "inner_epoch": epoch, "losses": losses,
                "best": {name: jsonable_config(best[name]["config"]) for name in CRITERIA},
            }), flush=True)
        selected_configs = {name: best[name]["config"] for name in CRITERIA}
        for index in dataset.indices(inner_valid):
            if index not in best_predictions:
                length = len(dataset.sequences[index]["center_times_s"])
                best_predictions[index] = {
                    "residual": np.zeros(length, dtype=np.float64),
                    "presence_probability": 1.0,
                    "onset_probability": np.zeros(length, dtype=np.float64),
                    "offset_probability": np.zeros(length, dtype=np.float64),
                }
        selected_risk = (
            choose_risk_config(
                dataset, inner_valid, best_predictions, selected_configs,
            ) if "presence_change_policy" in dataset.frozen else None
        )
        if selected_risk is not None:
            print(json.dumps({
                "outer_fold": outer_name,
                "selected_presence_risk_rule": selected_risk["rule_json"],
                "inner_macro_iou": selected_risk["summary"]["macro_temporal_iou"],
            }), flush=True)
        del model, optimizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        maximum_epoch = max(config[0] for config in selected_configs.values())
        outer_model = trainer.new_model(seed + 10000)
        outer_optimizer = torch.optim.AdamW(
            outer_model.parameters(), lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        needed = defaultdict(list)
        for criterion, config in selected_configs.items():
            needed[int(config[0])].append(criterion)
        outer_history = []
        if 0 in needed:
            initial = trainer.predict(
                outer_model, [index for criterion in needed[0] for index in dataset.indices(heldout, criterion)],
            )
            prediction_by_sequence.update(initial)
        for epoch in range(1, maximum_epoch + 1):
            losses = trainer.train_epoch(
                outer_model, outer_optimizer, dataset.indices(outer_train), epoch, seed + 10000,
            )
            outer_history.append({"epoch": epoch, **losses})
            if epoch in needed:
                current_indices = [
                    index for criterion in needed[epoch]
                    for index in dataset.indices(heldout, criterion)
                ]
                prediction_by_sequence.update(trainer.predict(outer_model, current_indices))
            print(json.dumps({
                "outer_fold": outer_name, "outer_retrain_epoch": epoch,
                "losses": losses,
            }), flush=True)
        checkpoint = args.output_dir / f"{outer_name}_direct_interval.pt"
        torch.save({
            "schema_version": "text_conditioned_direct_interval_outer_model_v1",
            "model_state": outer_model.state_dict(),
            "outer_heldout_video_ids": sorted(heldout),
            "inner_validation_video_ids": sorted(inner_valid),
            "selected_configs": {name: jsonable_config(value) for name, value in selected_configs.items()},
            "selected_presence_risk_rule": (
                selected_risk["rule_json"] if selected_risk is not None else None
            ),
            "outer_labels_used_for_model_or_policy_selection": False,
            "LLM_or_MLLM_parameters_updated": False,
            "development_or_consumed_test_accessed": False,
        }, checkpoint)
        for video in heldout:
            fold_by_video[video] = outer_name
        selected_by_fold.append({
            "fold": outer_name, "outer_heldout_video_ids": sorted(heldout),
            "inner_validation_fold": f"fold{inner_index}",
            "inner_validation_video_ids": sorted(inner_valid),
            "selected_configs": {name: jsonable_config(value) for name, value in selected_configs.items()},
            "selected_presence_risk_rule": (
                selected_risk["rule_json"] if selected_risk is not None else None
            ),
            "presence_risk_selection": selected_risk,
            "selection_summaries": {name: best[name]["summary"] for name in CRITERIA},
            "inner_history": inner_history, "outer_retrain_history": outer_history,
            "checkpoint": str(checkpoint.resolve()),
        })
        (args.output_dir / "partial_audit.json").write_text(json.dumps({
            "completed_folds": selected_by_fold,
            "development_or_consumed_test_accessed": False,
        }, indent=2) + "\n", encoding="utf-8")
        del outer_model, outer_optimizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if args.smoke:
        result = {
            "schema_version": "direct_interval_nested_oof_smoke_v1",
            "completed_outer_folds": fold_count, "gate_valid": False,
            "reason": "smoke_run_not_all_outer_folds", "fold_audits": selected_by_fold,
            "development_or_consumed_test_accessed": False,
        }
        (args.output_dir / "smoke_result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps({"smoke_complete": True, "gate_valid": False}, indent=2))
        return

    all_videos = set(dataset.labeled_ids)
    baseline_rows = dataset.baseline_rows(all_videos)
    framework_rows = []
    prediction_rows = []
    for outer_index, fold in enumerate(dataset.folds):
        videos = set(fold)
        configs = {
            name: (
                int(selected_by_fold[outer_index]["selected_configs"][name]["epoch"]),
                float(selected_by_fold[outer_index]["selected_configs"][name]["correction_strength"]),
                float(selected_by_fold[outer_index]["selected_configs"][name]["threshold_offset"]),
                selected_by_fold[outer_index]["selected_configs"][name]["presence_veto_threshold"],
            ) for name in CRITERIA
        }
        risk_row = selected_by_fold[outer_index].get("selected_presence_risk_rule")
        risk_config = (
            (float(risk_row["m2c_threshold_cap"]), int(risk_row["maximum_duration_points"]))
            if risk_row is not None else None
        )
        framework_rows.extend(decoded_rows(
            dataset, videos, prediction_by_sequence, configs, risk_config,
        ))
        for index in dataset.indices(videos):
            row = dataset.sequences[index]
            criterion = str(row["criterion"])
            config = configs[criterion]
            prediction = prediction_by_sequence[index]
            corrected = sigmoid_residual(
                row["m2c_probabilities"].numpy(), prediction["residual"], config[1],
            )
            for point, time_s in enumerate(row["center_times_s"]):
                prediction_rows.append({
                    "reference_id": row["reference_ids"][point],
                    "video_id": int(row["video_id"]), "criterion": criterion,
                    "center_time_s": float(time_s), "outer_oof_fold": f"fold{outer_index}",
                    "model_never_saw_video": True,
                    "outer_labels_used_for_model_or_policy_selection": False,
                    "m2c_probability": float(row["m2c_probabilities"][point]),
                    "direct_interval_probability": float(corrected[point]),
                    "raw_frame_residual": float(prediction["residual"][point]),
                    "onset_probability": float(prediction["onset_probability"][point]),
                    "offset_probability": float(prediction["offset_probability"][point]),
                    "pair_presence_probability": float(prediction["presence_probability"]),
                    "selected_config": jsonable_config(config),
                    "selected_presence_risk_rule": risk_row,
                })
    repetitions = int(dataset.frozen["paper_gate"]["paired_video_bootstrap_repetitions"])
    comparison = paired_bootstrap_comparison(
        baseline_rows, framework_rows, seed=seed_base + 99999,
        repetitions=repetitions, baseline_label="M2c",
        comparison_label=("RiskFilteredDirectInterval" if "presence_change_policy" in dataset.frozen else "DirectInterval"),
    )
    gate = paper_gate(
        baseline_rows, framework_rows, fold_by_video, comparison, dataset.frozen,
    )
    result = {
        "schema_version": (
            "recall_protected_direct_interval_nested_oof_result_v2"
            if "presence_change_policy" in dataset.frozen
            else "text_conditioned_direct_interval_nested_oof_result_v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_policy": str(args.frozen_policy.resolve()),
        "frozen_policy_sha256": sha256_file(args.frozen_policy),
        "protocol": {
            "outer_folds": 5,
            "inner_selection": "rotated_disjoint_video_fold_train_only",
            "outer_labels_used_for_model_or_policy_selection": False,
            "development_labels_or_predictions_loaded": False,
            "consumed_test_loaded": False,
        },
        "baseline": method_summary(baseline_rows, seed_base, repetitions),
        "direct_interval": method_summary(framework_rows, seed_base + 1, repetitions),
        "paired_comparison": comparison, "paper_gate": gate,
        "fold_audits": selected_by_fold,
        "LLM_or_MLLM_parameters_updated": False,
        "development_or_consumed_test_accessed": False,
    }
    (args.output_dir / "nested_oof_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_dir / "predictions_outer_oof.json").write_text(json.dumps({
        "schema_version": "direct_interval_outer_oof_predictions_v1",
        "annotation_files_loaded_during_prediction": False,
        "training_annotations_loaded_for_training_only": True,
        "development_or_consumed_test_accessed": False,
        "rows": prediction_rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str((args.output_dir / "nested_oof_result.json").resolve()),
        "paper_gate_passed": gate["passed"],
        "baseline_macro_iou": gate["baseline"]["macro_temporal_iou"],
        "direct_macro_iou": gate["framework"]["macro_temporal_iou"],
        "paired_macro_iou_ci": comparison["macro_temporal_iou"],
        "development_or_consumed_test_accessed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
