#!/usr/bin/env python3
"""Train and freeze the train-only text/spatial/temporal Endoscapes small model."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
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

from cvs_assessment.text_conditioned_temporal_cvs import (
    SharedTemporalCvsHead, TextConditionedSpatialFrameHead,
)
from run_cholec80_validation_ablation import CRITERIA
from train_direct_interval_nested_oof import sha256_file


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool); positive = int(labels.sum())
    if not positive:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    tp = previous = 0; value = 0.0; first = 0
    while first < len(scores):
        last = first + 1
        while last < len(scores) and scores[last] == scores[first]: last += 1
        tp += int(labels[first:last].sum())
        value += (tp - previous) / positive * tp / last
        previous = tp; first = last
    return float(value)


def binary_summary(labels: np.ndarray, scores: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    by = {}
    for index, criterion in enumerate(CRITERIA):
        y, score = labels[:, index].astype(bool), scores[:, index]
        pred = score >= thresholds[criterion]
        tp, fn = int((y & pred).sum()), int((y & ~pred).sum())
        tn, fp = int((~y & ~pred).sum()), int((~y & pred).sum())
        recall = tp / max(tp + fn, 1); specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        by[criterion] = {
            "average_precision": average_precision(y, score),
            "balanced_accuracy": 0.5 * (recall + specificity),
            "f1": f1, "sensitivity": recall, "specificity": specificity,
            "threshold": thresholds[criterion], "positive": int(y.sum()), "n": len(y),
        }
    return {
        "macro_average_precision": float(np.mean([row["average_precision"] for row in by.values()])),
        "macro_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in by.values()])),
        "macro_f1": float(np.mean([row["f1"] for row in by.values()])),
        "by_criterion": by,
    }


class Dataset:
    def __init__(self, protocol_path: Path, cache_audit_path: Path) -> None:
        self.protocol_path = protocol_path
        self.protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
        if not audit.get("complete") or audit.get("official_val_or_test_labels_loaded") is not False:
            raise ValueError("Unsafe training feature cache")
        if audit["protocol"]["sha256"] != sha256_file(protocol_path):
            raise ValueError("Feature cache belongs to another protocol")
        self.data = {}
        for row in audit["videos"]:
            path = Path(row["path"])
            if sha256_file(path) != row["sha256"]: raise ValueError(path)
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value.get("official_val_or_test_labels_loaded") is not False: raise ValueError(path)
            self.data[int(row["video_id"])] = value
        skill_path = Path(self.protocol["sources"]["frozen_skill_text_embeddings"]["path"])
        skill = torch.load(skill_path, map_location="cpu", weights_only=False)
        self.embedding = {key: value.float() for key, value in skill["criterion_embeddings"].items()}
        semantic = list(skill["semantic_types"])
        roi_classes = skill["task_visual_evidence_schema"]["criterion_roi_classes"]
        global_count = int(skill["token_layout"]["global_grid_tokens"])
        self.relevance = {
            criterion: torch.tensor(
                [1.0] * global_count + [float(name in set(roi_classes[criterion])) for name in semantic],
                dtype=torch.float16,
            ) for criterion in CRITERIA
        }
        self.train_ids = set(self.protocol["internal_train_only_split"]["training_video_ids"])
        self.dev_ids = set(self.protocol["internal_train_only_split"]["development_video_ids"])
        if set(self.data) != self.train_ids | self.dev_ids: raise ValueError("Training cache coverage mismatch")

    def samples(self, videos: set[int]) -> list[tuple[int, int, int]]:
        return [
            (video, frame, criterion)
            for video in sorted(videos)
            for frame in range(len(self.data[video]["frame_indices"]))
            for criterion in range(len(CRITERIA))
        ]

    def frame_batch(self, samples: list[tuple[int, int, int]], device: torch.device) -> dict[str, torch.Tensor]:
        peska, tokens, geometry, embedding, label = [], [], [], [], []
        for video, frame, criterion_index in samples:
            value = self.data[video]; criterion = CRITERIA[criterion_index]
            peska.append(value["peska_features"][frame])
            tokens.append(value["tokens"][frame])
            geometry.append(torch.cat([
                value["geometry"][frame], self.relevance[criterion][:, None],
            ], dim=-1))
            embedding.append(self.embedding[criterion])
            label.append(value["labels_soft_C1_C3_C2"][frame, criterion_index] >= 0.5)
        return {
            "peska": torch.stack(peska).to(device),
            "tokens": torch.stack(tokens).to(device),
            "geometry": torch.stack(geometry).to(device),
            "embedding": torch.stack(embedding).to(device),
            "label": torch.tensor(label, dtype=torch.float32, device=device),
        }

    def balance(self, videos: set[int]) -> torch.Tensor:
        labels = torch.cat([self.data[video]["labels_soft_C1_C3_C2"] >= 0.5 for video in videos])
        weights = []
        for index in range(len(CRITERIA)):
            positive = float(labels[:, index].sum()); negative = len(labels) - positive
            weights.append(min(12.0, max(1.0, negative / max(positive, 1.0))))
        return torch.tensor(weights)


def new_frame_model(dataset: Dataset, protocol: dict[str, Any], device: torch.device, seed: int) -> nn.Module:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    cfg = protocol["model"]
    example = next(iter(dataset.data.values()))
    return TextConditionedSpatialFrameHead(
        peska_dim=example["peska_features"].shape[-1], token_dim=example["tokens"].shape[-1],
        geometry_dim=example["geometry"].shape[-1] + 1,
        text_dim=next(iter(dataset.embedding.values())).numel(),
        hidden_dim=int(cfg["frame_hidden_dim"]), attention_heads=int(cfg["attention_heads"]),
        dropout=float(cfg["dropout"]),
    ).to(device)


def train_frame_epoch(
    model: nn.Module, dataset: Dataset, videos: set[int], optimizer: Any,
    device: torch.device, epoch: int, seed: int, batch_size: int,
) -> float:
    model.train(); samples = dataset.samples(videos)
    random.Random(seed * 1000 + epoch).shuffle(samples)
    balance = dataset.balance(videos).to(device); losses = []
    for start in range(0, len(samples), batch_size):
        current = samples[start:start + batch_size]
        batch = dataset.frame_batch(current, device)
        output = model(batch["peska"], batch["tokens"], batch["geometry"], batch["embedding"])
        weights = torch.tensor([balance[criterion] for _, _, criterion in current], device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            output["logit"], batch["label"], pos_weight=weights,
        )
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def predict_frame(model: nn.Module, dataset: Dataset, videos: set[int], device: torch.device, batch_size: int) -> dict[int, dict[str, torch.Tensor]]:
    model.eval(); output = {}
    for video in sorted(videos):
        value = dataset.data[video]; count = len(value["frame_indices"])
        logits = torch.zeros(count, len(CRITERIA)); hidden = torch.zeros(count, len(CRITERIA), model.hidden_dim)
        samples = [(video, frame, criterion) for frame in range(count) for criterion in range(len(CRITERIA))]
        for start in range(0, len(samples), batch_size):
            current = samples[start:start + batch_size]; batch = dataset.frame_batch(current, device)
            prediction = model(batch["peska"], batch["tokens"], batch["geometry"], batch["embedding"])
            for local, (_, frame, criterion) in enumerate(current):
                logits[frame, criterion] = prediction["logit"][local].cpu()
                hidden[frame, criterion] = prediction["hidden"][local].cpu()
        output[video] = {"logits": logits, "hidden": hidden}
    return output


def flatten_predictions(dataset: Dataset, predictions: dict[int, dict[str, torch.Tensor]]) -> tuple[np.ndarray, np.ndarray]:
    labels = torch.cat([dataset.data[video]["labels_soft_C1_C3_C2"] >= 0.5 for video in sorted(predictions)])
    scores = torch.cat([torch.sigmoid(predictions[video]["logits"]) for video in sorted(predictions)])
    return labels.numpy(), scores.numpy()


def new_temporal_model(dataset: Dataset, protocol: dict[str, Any], device: torch.device, seed: int) -> nn.Module:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    cfg = protocol["model"]
    return SharedTemporalCvsHead(
        frame_dim=int(cfg["frame_hidden_dim"]),
        text_dim=next(iter(dataset.embedding.values())).numel(),
        hidden_dim=int(cfg["temporal_hidden_dim"]), dilations=tuple(cfg["temporal_dilations"]),
        dropout=float(cfg["dropout"]),
    ).to(device)


def temporal_batch(
    dataset: Dataset, frame: dict[int, dict[str, torch.Tensor]],
    pairs: list[tuple[int, int]], device: torch.device,
) -> dict[str, torch.Tensor]:
    maximum = max(len(dataset.data[video]["frame_indices"]) for video, _ in pairs)
    batch = len(pairs); hidden_dim = next(iter(frame.values()))["hidden"].shape[-1]
    hidden = torch.zeros(batch, maximum, hidden_dim); logits = torch.zeros(batch, maximum)
    labels = torch.zeros(batch, maximum); valid = torch.zeros(batch, maximum, dtype=torch.bool)
    progress = torch.zeros(batch, maximum); embeddings = []
    for local, (video, criterion) in enumerate(pairs):
        length = len(dataset.data[video]["frame_indices"]); valid[local, :length] = True
        hidden[local, :length] = frame[video]["hidden"][:, criterion]
        logits[local, :length] = frame[video]["logits"][:, criterion]
        labels[local, :length] = (dataset.data[video]["labels_soft_C1_C3_C2"][:, criterion] >= 0.5).float()
        progress[local, :length] = torch.linspace(0, 1, length)
        embeddings.append(dataset.embedding[CRITERIA[criterion]])
    return {key: value.to(device) for key, value in {
        "hidden": hidden, "logits": logits, "labels": labels, "valid": valid,
        "progress": progress, "embeddings": torch.stack(embeddings),
    }.items()}


def train_temporal_epoch(
    model: nn.Module, dataset: Dataset, frame: dict[int, dict[str, torch.Tensor]],
    videos: set[int], optimizer: Any, device: torch.device,
    epoch: int, seed: int, batch_size: int,
) -> float:
    model.train(); pairs = [(video, criterion) for video in videos for criterion in range(len(CRITERIA))]
    random.Random(seed * 1000 + epoch).shuffle(pairs); balance = dataset.balance(videos).to(device); losses = []
    for start in range(0, len(pairs), batch_size):
        current = pairs[start:start + batch_size]; batch = temporal_batch(dataset, frame, current, device)
        output = model(batch["hidden"], batch["logits"], batch["embeddings"], batch["progress"], batch["valid"])
        terms = nn.functional.binary_cross_entropy_with_logits(output["logit"], batch["labels"], reduction="none")
        weights = torch.ones_like(terms)
        for local, (_, criterion) in enumerate(current): weights[local, batch["labels"][local] >= 0.5] *= balance[criterion]
        loss = (terms * weights * batch["valid"]).sum() / (weights * batch["valid"]).sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def predict_temporal(
    model: nn.Module, dataset: Dataset, frame: dict[int, dict[str, torch.Tensor]],
    videos: set[int], device: torch.device, batch_size: int,
) -> dict[int, dict[str, torch.Tensor]]:
    model.eval(); output = {video: {"logits": frame[video]["logits"].clone(), "hidden": frame[video]["hidden"]} for video in videos}
    pairs = [(video, criterion) for video in sorted(videos) for criterion in range(len(CRITERIA))]
    for start in range(0, len(pairs), batch_size):
        current = pairs[start:start + batch_size]; batch = temporal_batch(dataset, frame, current, device)
        value = model(batch["hidden"], batch["logits"], batch["embeddings"], batch["progress"], batch["valid"])
        for local, (video, criterion) in enumerate(current):
            length = len(dataset.data[video]["frame_indices"])
            output[video]["logits"][:, criterion] = value["logit"][local, :length].cpu()
    return output


def choose_thresholds(labels: np.ndarray, scores: np.ndarray, choices: list[float]) -> dict[str, float]:
    output = {}
    for index, criterion in enumerate(CRITERIA):
        best = None
        for threshold in choices:
            summary = binary_summary(labels[:, [index]], scores[:, [index]], {criterion: threshold}) if False else None
            y, pred = labels[:, index].astype(bool), scores[:, index] >= threshold
            tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            specificity = int((~y & ~pred).sum()) / max(int((~y).sum()), 1)
            key = (f1, specificity, -abs(threshold - 0.5))
            if best is None or key > best[0]: best = (key, float(threshold))
        output[criterion] = best[1]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    dataset = Dataset(args.protocol, args.cache_audit); protocol = dataset.protocol
    cfg = protocol["training"]; seed = int(cfg["seed"]); device = torch.device(args.device)
    batch_size = int(cfg["batch_size"]); args.output_dir.mkdir(parents=True, exist_ok=False)
    default_thresholds = {criterion: 0.5 for criterion in CRITERIA}

    frame_model = new_frame_model(dataset, protocol, device, seed)
    optimizer = torch.optim.AdamW(frame_model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_frame = None; frame_history = []
    for epoch in range(1, int(cfg["frame_epochs"]) + 1):
        loss = train_frame_epoch(frame_model, dataset, dataset.train_ids, optimizer, device, epoch, seed, batch_size)
        row = {"epoch": epoch, "loss": loss}
        if epoch in set(cfg["frame_selection_epochs"]):
            prediction = predict_frame(frame_model, dataset, dataset.dev_ids, device, batch_size)
            labels, scores = flatten_predictions(dataset, prediction)
            summary = binary_summary(labels, scores, default_thresholds); row["development"] = summary
            key = (summary["macro_average_precision"], summary["macro_balanced_accuracy"], summary["macro_f1"])
            if best_frame is None or key > best_frame[0]: best_frame = (key, epoch, deepcopy(frame_model.state_dict()), summary)
        frame_history.append(row); print(json.dumps({"stage": "frame", **row}), flush=True)
    frame_model.load_state_dict(best_frame[2]); selected_frame_epoch = int(best_frame[1])
    frame_all = predict_frame(frame_model, dataset, dataset.train_ids | dataset.dev_ids, device, batch_size)

    temporal_model = new_temporal_model(dataset, protocol, device, seed + 100)
    temporal_optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    labels, scores = flatten_predictions(dataset, {video: frame_all[video] for video in dataset.dev_ids})
    frame_summary = binary_summary(labels, scores, default_thresholds)
    best_temporal = ((frame_summary["macro_average_precision"], frame_summary["macro_balanced_accuracy"], frame_summary["macro_f1"]), 0, deepcopy(temporal_model.state_dict()), frame_summary)
    temporal_history = []
    for epoch in range(1, int(cfg["temporal_epochs"]) + 1):
        loss = train_temporal_epoch(temporal_model, dataset, frame_all, dataset.train_ids, temporal_optimizer, device, epoch, seed + 100, batch_size)
        row = {"epoch": epoch, "loss": loss}
        if epoch in set(cfg["temporal_selection_epochs"]):
            prediction = predict_temporal(temporal_model, dataset, frame_all, dataset.dev_ids, device, batch_size)
            labels, scores = flatten_predictions(dataset, prediction)
            summary = binary_summary(labels, scores, default_thresholds); row["development"] = summary
            key = (summary["macro_average_precision"], summary["macro_balanced_accuracy"], summary["macro_f1"])
            if key > best_temporal[0]: best_temporal = (key, epoch, deepcopy(temporal_model.state_dict()), summary)
        temporal_history.append(row); print(json.dumps({"stage": "temporal", **row}), flush=True)
    selected_temporal_epoch = int(best_temporal[1]); temporal_model.load_state_dict(best_temporal[2])
    selected_dev = (
        {video: frame_all[video] for video in dataset.dev_ids}
        if selected_temporal_epoch == 0 else
        predict_temporal(temporal_model, dataset, frame_all, dataset.dev_ids, device, batch_size)
    )
    dev_labels, dev_scores = flatten_predictions(dataset, selected_dev)
    thresholds = choose_thresholds(dev_labels, dev_scores, list(map(float, cfg["threshold_choices"])))
    selected_summary = binary_summary(dev_labels, dev_scores, thresholds)

    # Full retrain on all official train videos using only the frozen epoch counts.
    all_ids = dataset.train_ids | dataset.dev_ids
    full_frame = new_frame_model(dataset, protocol, device, seed + 10000)
    full_optimizer = torch.optim.AdamW(full_frame.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    full_frame_history = []
    for epoch in range(1, selected_frame_epoch + 1):
        loss = train_frame_epoch(full_frame, dataset, all_ids, full_optimizer, device, epoch, seed + 10000, batch_size)
        full_frame_history.append({"epoch": epoch, "loss": loss}); print(json.dumps({"stage": "full_frame", "epoch": epoch, "loss": loss}), flush=True)
    full_frame_predictions = predict_frame(full_frame, dataset, all_ids, device, batch_size)
    full_temporal = new_temporal_model(dataset, protocol, device, seed + 10100)
    full_temporal_optimizer = torch.optim.AdamW(full_temporal.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    full_temporal_history = []
    for epoch in range(1, selected_temporal_epoch + 1):
        loss = train_temporal_epoch(full_temporal, dataset, full_frame_predictions, all_ids, full_temporal_optimizer, device, epoch, seed + 10100, batch_size)
        full_temporal_history.append({"epoch": epoch, "loss": loss}); print(json.dumps({"stage": "full_temporal", "epoch": epoch, "loss": loss}), flush=True)

    checkpoint_path = args.output_dir / "full_train_deployment.pt"
    torch.save({
        "schema_version": "endoscapes_text_spatial_temporal_full_train_deployment_v1",
        "frame_model_state": {key: value.cpu() for key, value in full_frame.state_dict().items()},
        "temporal_model_state": {key: value.cpu() for key, value in full_temporal.state_dict().items()},
        "selected_frame_epoch": selected_frame_epoch,
        "selected_temporal_epoch": selected_temporal_epoch,
        "thresholds_internal_development_only": thresholds,
        "official_train_video_ids": sorted(all_ids),
        "official_val_or_test_labels_used": False,
        "LLM_or_MLLM_parameters_updated": False,
    }, checkpoint_path)
    result = {
        "schema_version": "endoscapes_text_spatial_temporal_train_selection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "cache_audit": {"path": str(args.cache_audit.resolve()), "sha256": sha256_file(args.cache_audit)},
        "selected_frame_epoch": selected_frame_epoch,
        "selected_temporal_epoch": selected_temporal_epoch,
        "thresholds": thresholds,
        "frame_development_at_0_5": best_frame[3],
        "selected_development": selected_summary,
        "frame_history": frame_history, "temporal_history": temporal_history,
        "full_frame_history": full_frame_history, "full_temporal_history": full_temporal_history,
        "deployment_checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
        "official_val_or_test_labels_used": False,
        "LLM_or_MLLM_parameters_updated": False,
    }
    result_path = args.output_dir / "train_selection_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(result_path.resolve()), "selected_frame_epoch": selected_frame_epoch,
        "selected_temporal_epoch": selected_temporal_epoch,
        "thresholds": thresholds, "development": selected_summary,
        "checkpoint": result["deployment_checkpoint"],
        "official_val_or_test_labels_used": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
