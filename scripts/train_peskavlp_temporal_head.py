"""Train a leakage-aware ordinal TCN and auxiliary CVS boundary head."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvs_assessment.models import OrdinalCriterionHead, TemporalOrdinalBoundaryHead
import train_peskavlp_ordinal_head as ordinal
import train_peskavlp_cvs_head as base


CRITERIA = ordinal.CRITERIA
DEFINITIONS = ("support_or_full", "full_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitized_cache_manifest(cache_dir: Path, test_ids: list[int]) -> dict:
    """Retain cache provenance without copying held-out label counts."""
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    for video_id in test_ids:
        video = manifest.get("videos", {}).get(str(video_id), {})
        video.pop("positive_counts", None)
        video["labels"] = "withheld_until_after_model_selection"
    return manifest


def load_videos(cache_dir: Path, video_ids: list[int]) -> dict[int, dict]:
    output = {}
    for video_id in video_ids:
        path = cache_dir / f"video{video_id:02d}.pt"
        value = torch.load(path, map_location="cpu", weights_only=False)
        value["features"] = value["features"].float()
        value["labels"] = value["labels"].long()
        output[video_id] = value
    return output


def frame_tensors(videos: dict[int, dict]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat([value["features"] for value in videos.values()]),
        torch.cat([value["labels"] for value in videos.values()]),
    )


def frame_logits(head: OrdinalCriterionHead, features: torch.Tensor) -> torch.Tensor:
    output = head(features)
    support = torch.stack([output[key]["support_or_full"] for key in CRITERIA], dim=-1)
    full = torch.stack([output[key]["full_only"] for key in CRITERIA], dim=-1)
    return torch.cat([support, full], dim=-1)


@torch.inference_mode()
def infer_frame_head(
    head: OrdinalCriterionHead, features: torch.Tensor, device: str,
    batch_size: int = 2048,
) -> torch.Tensor:
    head.eval()
    values = []
    for start in range(0, len(features), batch_size):
        values.append(frame_logits(head, features[start:start + batch_size].to(device)).cpu())
    return torch.cat(values)


def truth_bundle(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"support_or_full": labels >= 1, "full_only": labels >= 2}


def macro_frame_auprc(
    logits: torch.Tensor, labels: torch.Tensor,
    definitions: tuple[str, ...] = DEFINITIONS,
) -> float:
    truth = truth_bundle(labels)
    probabilities = torch.sigmoid(logits)
    values = []
    for definition in definitions:
        definition_index = DEFINITIONS.index(definition)
        for criterion_index in range(3):
            score = probabilities[:, definition_index * 3 + criterion_index]
            metric = base.binary_metrics(score, truth[definition][:, criterion_index], 0.5)
            if metric["auprc"] is not None:
                values.append(float(metric["auprc"]))
    return sum(values) / max(1, len(values))


def train_frame_head(
    train_videos: dict[int, dict], validation_videos: dict[int, dict],
    device: str, epochs: int, patience: int, batch_size: int,
    learning_rate: float,
) -> tuple[OrdinalCriterionHead, list[dict]]:
    train_features, train_labels = frame_tensors(train_videos)
    validation_features, validation_labels = frame_tensors(validation_videos)
    truths = truth_bundle(train_labels)
    support_positive = truths["support_or_full"].sum(0).float()
    full_positive = truths["full_only"].sum(0).float()
    support_weight = ((len(train_labels) - support_positive) / support_positive.clamp_min(1)).clamp(max=30).to(device)
    full_weight = ((len(train_labels) - full_positive) / full_positive.clamp_min(1)).clamp(max=30).to(device)
    support_loss = nn.BCEWithLogitsLoss(pos_weight=support_weight)
    full_loss = nn.BCEWithLogitsLoss(pos_weight=full_weight)
    head = OrdinalCriterionHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True,
    )
    best_value, best_state, stale, history = float("-inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = frame_logits(head, features)
            support_target = (labels >= 1).float()
            full_target = (labels >= 2).float()
            consistency = torch.relu(
                torch.sigmoid(logits[:, 3:]) - torch.sigmoid(logits[:, :3])
            ).mean()
            loss = support_loss(logits[:, :3], support_target)
            loss = loss + full_loss(logits[:, 3:], full_target) + 0.25 * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(features)
        validation_logits = infer_frame_head(head, validation_features, device)
        selection = macro_frame_auprc(validation_logits, validation_labels)
        row = {
            "epoch": epoch, "train_loss": total / len(train_labels),
            "validation_macro_ordinal_auprc": selection,
        }
        history.append(row)
        print(f"frame_epoch={epoch} loss={row['train_loss']:.5f} val_auprc={selection:.5f}", flush=True)
        if selection > best_value + 1e-4:
            best_value, best_state, stale = selection, copy.deepcopy(head.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Frame-head training selected no checkpoint")
    head.load_state_dict(best_state)
    head.eval().requires_grad_(False)
    return head, history


def prepare_sequences(
    videos: dict[int, dict], frame_head: OrdinalCriterionHead, device: str,
) -> dict[int, dict]:
    output = {}
    for video_id, value in videos.items():
        prepared = dict(value)
        prepared["frame_logits"] = infer_frame_head(frame_head, value["features"], device)
        output[video_id] = prepared
    return output


def boundary_targets(labels: torch.Tensor) -> torch.Tensor:
    full = (labels >= 2).float()
    previous = torch.cat([torch.zeros_like(full[:1]), full[:-1]], dim=0)
    following = torch.cat([full[1:], torch.zeros_like(full[:1])], dim=0)
    onset = ((full > previous) & (full > 0)).float()
    offset = ((full > following) & (full > 0)).float()
    targets = torch.stack([onset, offset], dim=-1)
    # A small tolerance is appropriate for interval annotations and 2-second sampling.
    expanded = targets.clone()
    for shift, weight in ((1, 0.6), (2, 0.25)):
        expanded[shift:] = torch.maximum(expanded[shift:], targets[:-shift] * weight)
        expanded[:-shift] = torch.maximum(expanded[:-shift], targets[shift:] * weight)
    return expanded


def early_negative_mask(labels: torch.Tensor) -> torch.Tensor:
    masks = []
    for minimum in (1, 2):
        definition = labels >= minimum
        current = torch.zeros_like(definition, dtype=torch.bool)
        for criterion in range(3):
            positives = torch.where(definition[:, criterion])[0]
            end = int(positives[0]) if len(positives) else len(definition)
            current[:end, criterion] = True
        masks.append(current)
    return torch.stack(masks, dim=-1)


class SequenceWindowDataset(Dataset):
    def __init__(
        self, videos: dict[int, dict], length: int, stride: int,
        hard_negative_weight: float, full_positive_window_weight: float,
    ) -> None:
        self.videos = videos
        self.length = length
        self.items: list[tuple[int, int]] = []
        self.weights: list[float] = []
        for video_id, value in videos.items():
            n = len(value["features"])
            starts = list(range(0, max(1, n - length + 1), stride))
            last = max(0, n - length)
            if not starts or starts[-1] != last:
                starts.append(last)
            labels = value["labels"]
            probabilities = torch.sigmoid(value["frame_logits"])
            truth = torch.cat([(labels >= 1), (labels >= 2)], dim=-1)
            early = torch.cat([
                early_negative_mask(labels)[..., 0], early_negative_mask(labels)[..., 1]
            ], dim=-1)
            hard = (probabilities >= 0.5) & (~truth) & early
            for start in starts:
                end = min(start + length, n)
                hard_fraction = float(hard[start:end].float().mean())
                positive = bool(truth[start:end].any())
                full_positive = bool(truth[start:end, 3:].any())
                self.items.append((video_id, start))
                self.weights.append(
                    1.0 + hard_negative_weight * hard_fraction
                    + (1.0 if positive else 0.0)
                    + (full_positive_window_weight if full_positive else 0.0)
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        video_id, start = self.items[index]
        source = self.videos[video_id]
        end = min(start + self.length, len(source["features"]))
        valid = end - start
        result = {
            "features": torch.zeros(self.length, source["features"].shape[-1]),
            "frame_logits": torch.zeros(self.length, 6),
            "progress": torch.zeros(self.length),
            "labels": torch.zeros(self.length, 3, dtype=torch.long),
            "mask": torch.zeros(self.length, dtype=torch.bool),
            "boundary": torch.zeros(self.length, 3, 2),
            "early_negative": torch.zeros(self.length, 3, 2, dtype=torch.bool),
        }
        result["features"][:valid] = source["features"][start:end]
        result["frame_logits"][:valid] = source["frame_logits"][start:end]
        result["progress"][:valid] = source["phase_progress"][start:end]
        result["labels"][:valid] = source["labels"][start:end]
        result["mask"][:valid] = True
        result["boundary"][:valid] = boundary_targets(source["labels"])[start:end]
        result["early_negative"][:valid] = early_negative_mask(source["labels"])[start:end]
        return result


def masked_bce(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight,
    )
    expanded = mask.unsqueeze(-1).expand_as(loss)
    return loss[expanded].mean()


def dice_loss(probabilities: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1).float()
    intersection = (probabilities * targets * valid).sum(dim=(0, 1))
    denominator = ((probabilities + targets) * valid).sum(dim=(0, 1))
    return (1.0 - (2 * intersection + 1.0) / (denominator + 1.0)).mean()


def masked_square_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a differentiable zero when a batch has no selected negatives."""
    selected = values[mask]
    return selected.square().mean() if selected.numel() else values.sum() * 0.0


@torch.inference_mode()
def infer_temporal(
    model: TemporalOrdinalBoundaryHead, videos: dict[int, dict], device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    model.eval()
    output = {}
    for video_id, value in videos.items():
        prediction = model(
            value["features"].unsqueeze(0).to(device),
            value["frame_logits"].unsqueeze(0).to(device),
            value["phase_progress"].unsqueeze(0).to(device),
        )
        output[video_id] = {
            key: torch.sigmoid(prediction[key][0]).cpu()
            for key in ("support_or_full", "full_only", "boundary")
        }
    return output


def intervals_from_binary(
    times: torch.Tensor, active: torch.Tensor, min_duration_s: float,
    max_gap_s: float,
) -> list[tuple[float, float]]:
    positive_times = [float(times[index]) for index in torch.where(active)[0]]
    if not positive_times:
        return []
    groups = [[positive_times[0]]]
    typical_step = float(torch.median(torch.diff(times))) if len(times) > 1 else 1.0
    for time_s in positive_times[1:]:
        if time_s - groups[-1][-1] <= typical_step + max_gap_s:
            groups[-1].append(time_s)
        else:
            groups.append([time_s])
    return [
        (group[0], group[-1]) for group in groups
        if group[-1] - group[0] >= min_duration_s
    ]


def interval_iou(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    def merged(values):
        output = []
        for start, end in sorted(values):
            if output and start <= output[-1][1]:
                output[-1] = (output[-1][0], max(output[-1][1], end))
            else:
                output.append((start, end))
        return output
    left, right = merged(left), merged(right)
    intersection = sum(
        max(0.0, min(a_end, b_end) - max(a_start, b_start))
        for a_start, a_end in left for b_start, b_end in right
    )
    left_duration = sum(end - start for start, end in left)
    right_duration = sum(end - start for start, end in right)
    union = left_duration + right_duration - intersection
    return intersection / union if union else 1.0


def temporal_metrics(
    predictions: dict[int, dict[str, torch.Tensor]], videos: dict[int, dict],
    thresholds: dict[str, dict[str, float]], min_duration_s: float,
    max_gap_s: float,
) -> dict:
    values, positive_values, per_video = [], [], {}
    definition_values = {definition: [] for definition in DEFINITIONS}
    definition_positive_values = {definition: [] for definition in DEFINITIONS}
    presence = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    absolute_onset_errors = []
    for video_id, definitions in predictions.items():
        source = videos[video_id]
        truth = truth_bundle(source["labels"])
        per_video[str(video_id)] = {}
        for definition in DEFINITIONS:
            per_video[str(video_id)][definition] = {}
            for criterion_index, criterion in enumerate(CRITERIA):
                predicted_intervals = intervals_from_binary(
                    source["timestamps_s"],
                    definitions[definition][:, criterion_index] >= thresholds[definition][criterion],
                    min_duration_s, max_gap_s,
                )
                truth_intervals = intervals_from_binary(
                    source["timestamps_s"], truth[definition][:, criterion_index], 0.0, 0.0,
                )
                iou = interval_iou(predicted_intervals, truth_intervals)
                predicted_present, truth_present = bool(predicted_intervals), bool(truth_intervals)
                presence[
                    "tp" if predicted_present and truth_present else
                    "fp" if predicted_present else
                    "fn" if truth_present else "tn"
                ] += 1
                onset_error = (
                    predicted_intervals[0][0] - truth_intervals[0][0]
                    if predicted_intervals and truth_intervals else None
                )
                per_video[str(video_id)][definition][criterion] = {
                    "temporal_iou": iou,
                    "predicted_intervals": predicted_intervals,
                    "truth_intervals": truth_intervals,
                    "presence_correct": predicted_present == truth_present,
                    "first_onset_error_s": onset_error,
                }
                values.append(iou)
                definition_values[definition].append(iou)
                if truth_present:
                    positive_values.append(iou)
                    definition_positive_values[definition].append(iou)
                if onset_error is not None:
                    absolute_onset_errors.append(abs(onset_error))
    total = sum(presence.values())
    positive_total = presence["tp"] + presence["fn"]
    negative_total = presence["tn"] + presence["fp"]
    return {
        "macro_temporal_iou": sum(values) / max(1, len(values)),
        "macro_positive_temporal_iou": sum(positive_values) / max(1, len(positive_values)),
        "presence_accuracy": (presence["tp"] + presence["tn"]) / max(1, total),
        "presence_sensitivity": presence["tp"] / max(1, positive_total),
        "presence_specificity": presence["tn"] / max(1, negative_total),
        "onset_mae_s": (
            sum(absolute_onset_errors) / len(absolute_onset_errors)
            if absolute_onset_errors else None
        ),
        "presence_confusion": presence,
        "by_definition": {
            definition: {
                "macro_temporal_iou": sum(definition_values[definition]) / max(1, len(definition_values[definition])),
                "macro_positive_temporal_iou": sum(definition_positive_values[definition]) / max(1, len(definition_positive_values[definition])),
            }
            for definition in DEFINITIONS
        },
        "per_video": per_video,
    }


def calibrate_thresholds(
    predictions: dict[int, dict[str, torch.Tensor]], videos: dict[int, dict],
    min_duration_s: float, max_gap_s: float,
) -> dict[str, dict[str, float]]:
    output = {definition: {} for definition in DEFINITIONS}
    for definition in DEFINITIONS:
        for criterion_index, criterion in enumerate(CRITERIA):
            best_threshold, best_value = 0.5, float("-inf")
            for threshold in np.linspace(0.1, 0.9, 33):
                values, positive_values = [], []
                for video_id, prediction in predictions.items():
                    source = videos[video_id]
                    truth = truth_bundle(source["labels"])[definition][:, criterion_index]
                    pred_intervals = intervals_from_binary(
                        source["timestamps_s"], prediction[definition][:, criterion_index] >= threshold,
                        min_duration_s, max_gap_s,
                    )
                    true_intervals = intervals_from_binary(source["timestamps_s"], truth, 0.0, 0.0)
                    value = interval_iou(pred_intervals, true_intervals)
                    values.append(value)
                    if bool(truth.any()):
                        positive_values.append(value)
                # Preserve hard-negative pressure without allowing an empty/empty
                # video to dominate threshold selection.
                score = 0.7 * (sum(positive_values) / max(1, len(positive_values)))
                score += 0.3 * (sum(values) / len(values))
                if score > best_value:
                    best_threshold, best_value = float(threshold), score
            output[definition][criterion] = best_threshold
    return output


def train_temporal_head(
    train_videos: dict[int, dict], validation_videos: dict[int, dict],
    device: str, args,
) -> tuple[TemporalOrdinalBoundaryHead, list[dict], dict]:
    dataset = SequenceWindowDataset(
        train_videos, args.window_length, args.window_stride, args.hard_negative_weight,
        args.full_positive_window_weight,
    )
    sampler = WeightedRandomSampler(dataset.weights, num_samples=len(dataset) * 2, replacement=True)
    loader = DataLoader(dataset, batch_size=args.sequence_batch_size, sampler=sampler)
    all_labels = torch.cat([value["labels"] for value in train_videos.values()])
    truth = truth_bundle(all_labels)
    support_positive = truth["support_or_full"].sum(0).float()
    full_positive = truth["full_only"].sum(0).float()
    support_weight = ((len(all_labels) - support_positive) / support_positive.clamp_min(1)).clamp(max=30).to(device)
    full_weight = ((len(all_labels) - full_positive) / full_positive.clamp_min(1)).clamp(max=30).to(device)
    boundary_weight = torch.full((3, 2), args.boundary_positive_weight, device=device)
    model = TemporalOrdinalBoundaryHead(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.temporal_learning_rate, weight_decay=1e-4)
    baseline_predictions = infer_temporal(model, validation_videos, device)
    baseline_thresholds = calibrate_thresholds(
        baseline_predictions, validation_videos,
        args.min_stable_seconds, args.max_gap_seconds,
    )
    baseline_temporal = temporal_metrics(
        baseline_predictions, validation_videos, baseline_thresholds,
        args.min_stable_seconds, args.max_gap_seconds,
    )
    _, baseline_labels = frame_tensors(validation_videos)
    baseline_logits = torch.cat([
        torch.cat([
            baseline_predictions[video_id]["support_or_full"],
            baseline_predictions[video_id]["full_only"],
        ], dim=-1)
        for video_id in validation_videos
    ])
    baseline_auprc = macro_frame_auprc(
        torch.logit(baseline_logits.clamp(1e-5, 1 - 1e-5)), baseline_labels,
        definitions=("full_only",),
    )
    baseline_full = baseline_temporal["by_definition"]["full_only"]
    baseline_selection = 0.5 * baseline_full["macro_temporal_iou"]
    baseline_selection += 0.5 * baseline_full["macro_positive_temporal_iou"]
    baseline_selection += 0.25 * baseline_auprc
    history = [{
        "epoch": 0,
        "train_loss": None,
        "validation_macro_temporal_iou_calibrated": baseline_full["macro_temporal_iou"],
        "validation_macro_positive_temporal_iou_calibrated": baseline_full["macro_positive_temporal_iou"],
        "validation_macro_full_auprc": baseline_auprc,
        "validation_thresholds": baseline_thresholds,
        "selection": baseline_selection,
        "frame_head_baseline": True,
    }]
    print(
        f"temporal_epoch=0 frame_baseline=true "
        f"val_full_iou={baseline_full['macro_temporal_iou']:.5f} "
        f"val_full_pos_iou={baseline_full['macro_positive_temporal_iou']:.5f} "
        f"val_auprc={baseline_auprc:.5f} selection={baseline_selection:.5f}",
        flush=True,
    )
    best_value = baseline_selection
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(1, args.temporal_epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            features = batch["features"].to(device)
            logits = batch["frame_logits"].to(device)
            progress = batch["progress"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            prediction = model(features, logits, progress)
            support_target = (labels >= 1).float()
            full_target = (labels >= 2).float()
            support_classification = masked_bce(
                prediction["support_or_full"], support_target, mask, support_weight,
            )
            full_classification = masked_bce(
                prediction["full_only"], full_target, mask, full_weight,
            )
            classification = support_classification + args.full_loss_weight * full_classification
            support_dice = dice_loss(
                torch.sigmoid(prediction["support_or_full"]), support_target, mask,
            )
            full_dice = dice_loss(
                torch.sigmoid(prediction["full_only"]), full_target, mask,
            )
            dice = support_dice + args.full_dice_weight * full_dice
            boundary_loss = nn.functional.binary_cross_entropy_with_logits(
                prediction["boundary"], batch["boundary"].to(device), reduction="none",
                pos_weight=boundary_weight,
            )
            boundary_mask = mask[..., None, None].expand_as(boundary_loss)
            boundary_loss = boundary_loss[boundary_mask].mean()
            support_probability = torch.sigmoid(prediction["support_or_full"])
            full_probability = torch.sigmoid(prediction["full_only"])
            early = batch["early_negative"].to(device)
            early_support = early[..., 0] & mask.unsqueeze(-1)
            early_full = early[..., 1] & mask.unsqueeze(-1)
            early_fp = masked_square_mean(support_probability, early_support)
            early_fp = early_fp + masked_square_mean(full_probability, early_full)
            consistency = torch.relu(full_probability - support_probability)[mask.unsqueeze(-1).expand_as(full_probability)].mean()
            adjacent_mask = mask[:, 1:] & mask[:, :-1]
            smooth = (
                (support_probability[:, 1:] - support_probability[:, :-1]).abs()
                + (full_probability[:, 1:] - full_probability[:, :-1]).abs()
            )
            smooth = smooth[adjacent_mask.unsqueeze(-1).expand_as(smooth)].mean()
            loss = classification
            loss = loss + args.dice_weight * dice
            loss = loss + args.boundary_weight * boundary_loss
            loss = loss + args.early_fp_weight * early_fp
            loss = loss + args.consistency_weight * consistency
            loss = loss + args.smoothness_weight * smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(features)
        val_predictions = infer_temporal(model, validation_videos, device)
        epoch_thresholds = calibrate_thresholds(
            val_predictions, validation_videos,
            args.min_stable_seconds, args.max_gap_seconds,
        )
        val_temporal = temporal_metrics(
            val_predictions, validation_videos, epoch_thresholds,
            args.min_stable_seconds, args.max_gap_seconds,
        )
        val_features, val_labels = frame_tensors(validation_videos)
        val_logits = torch.cat([
            torch.cat([
                val_predictions[video_id]["support_or_full"],
                val_predictions[video_id]["full_only"],
            ], dim=-1)
            for video_id in validation_videos
        ])
        val_auprc = macro_frame_auprc(
            torch.logit(val_logits.clamp(1e-5, 1 - 1e-5)), val_labels,
            definitions=("full_only",),
        )
        val_full = val_temporal["by_definition"]["full_only"]
        selection = 0.5 * val_full["macro_temporal_iou"]
        selection += 0.5 * val_full["macro_positive_temporal_iou"]
        selection += 0.25 * val_auprc
        row = {
            "epoch": epoch, "train_loss": total / (len(dataset) * 2),
            "validation_macro_temporal_iou_calibrated": val_full["macro_temporal_iou"],
            "validation_macro_positive_temporal_iou_calibrated": val_full["macro_positive_temporal_iou"],
            "validation_macro_full_auprc": val_auprc,
            "validation_thresholds": epoch_thresholds,
            "selection": selection,
        }
        history.append(row)
        print(
            f"temporal_epoch={epoch} loss={row['train_loss']:.5f} "
            f"val_full_iou={row['validation_macro_temporal_iou_calibrated']:.5f} "
            f"val_full_pos_iou={row['validation_macro_positive_temporal_iou_calibrated']:.5f} "
            f"val_auprc={val_auprc:.5f} selection={selection:.5f}", flush=True,
        )
        if selection > best_value + 1e-4:
            best_value, best_state, stale = selection, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= args.temporal_patience:
            break
    if best_state is None:
        raise RuntimeError("Temporal-head training selected no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {"best_selection": best_value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-epochs", type=int, default=40)
    parser.add_argument("--frame-patience", type=int, default=8)
    parser.add_argument("--frame-batch-size", type=int, default=256)
    parser.add_argument("--frame-learning-rate", type=float, default=3e-4)
    parser.add_argument("--temporal-epochs", type=int, default=50)
    parser.add_argument("--temporal-patience", type=int, default=10)
    parser.add_argument("--temporal-learning-rate", type=float, default=2e-4)
    parser.add_argument("--sequence-batch-size", type=int, default=4)
    parser.add_argument("--window-length", type=int, default=256)
    parser.add_argument("--window-stride", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hard-negative-weight", type=float, default=8.0)
    parser.add_argument("--full-positive-window-weight", type=float, default=4.0)
    parser.add_argument("--boundary-positive-weight", type=float, default=15.0)
    parser.add_argument("--dice-weight", type=float, default=0.35)
    parser.add_argument("--full-loss-weight", type=float, default=2.0)
    parser.add_argument("--full-dice-weight", type=float, default=2.0)
    parser.add_argument("--boundary-weight", type=float, default=0.20)
    parser.add_argument("--early-fp-weight", type=float, default=0.40)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--min-stable-seconds", type=float, default=12.0)
    parser.add_argument("--max-gap-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--reuse-frame-head", action="store_true",
        help="Reuse the ordinal head in --encoder-checkpoint instead of retraining it.",
    )
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Load held-out test labels after model selection. Omit during tuning.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_path = Path(args.split_json).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = [int(value) for value in split["train_video_ids"]]
    validation_ids = [int(value) for value in split["validation_video_ids"]]
    test_ids = [int(value) for value in split["test_video_ids"]]
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
        raise ValueError("Split contains overlapping video IDs")
    cache_dir = Path(args.cache_dir)
    train_videos = load_videos(cache_dir, train_ids)
    validation_videos = load_videos(cache_dir, validation_ids)
    print(f"loaded_train={train_ids} loaded_validation={validation_ids} test_withheld={test_ids}", flush=True)
    source_checkpoint = Path(args.encoder_checkpoint).resolve()
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if args.reuse_frame_head:
        if "head_state" not in source:
            raise ValueError("--reuse-frame-head requires head_state in the source checkpoint")
        frame_head = OrdinalCriterionHead().to(device)
        frame_head.load_state_dict(source["head_state"], strict=True)
        frame_head.eval().requires_grad_(False)
        frame_history = [{
            "reused": True,
            "source_checkpoint": str(source_checkpoint),
            "original_train_video_ids": source.get("metadata", {}).get("train_video_ids", []),
            "original_validation_video_ids": source.get("metadata", {}).get("validation_video_ids", []),
        }]
        print("frame_head_reused_from_source_checkpoint=true", flush=True)
    else:
        frame_head, frame_history = train_frame_head(
            train_videos, validation_videos, device,
            args.frame_epochs, args.frame_patience, args.frame_batch_size,
            args.frame_learning_rate,
        )
    train_videos = prepare_sequences(train_videos, frame_head, device)
    validation_videos = prepare_sequences(validation_videos, frame_head, device)
    temporal_head, temporal_history, selection = train_temporal_head(
        train_videos, validation_videos, device, args,
    )
    validation_predictions = infer_temporal(temporal_head, validation_videos, device)
    thresholds = calibrate_thresholds(
        validation_predictions, validation_videos,
        args.min_stable_seconds, args.max_gap_seconds,
    )
    validation_metrics = temporal_metrics(
        validation_predictions, validation_videos, thresholds,
        args.min_stable_seconds, args.max_gap_seconds,
    )

    test_metrics = None
    if args.evaluate_test:
        print("selection_complete_loading_test_features=true", flush=True)
        test_videos = prepare_sequences(load_videos(cache_dir, test_ids), frame_head, device)
        test_predictions = infer_temporal(temporal_head, test_videos, device)
        test_metrics = temporal_metrics(
            test_predictions, test_videos, thresholds,
            args.min_stable_seconds, args.max_gap_seconds,
        )
    else:
        print("selection_complete_test_features_remain_withheld=true", flush=True)
    metadata = {
        "tool_type": "peskavlp_frozen_encoder_temporal_ordinal_boundary_head",
        "head_type": "temporal_ordinal_boundary",
        "checkpoint_version": "0.4-temporal-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "criteria": CRITERIA,
        "encoder": "PeskaVLP ResNet-50 visual tower + 768-D projection (frozen)",
        "frame_head": (
            "reused ordinal criterion-conditioned v2 head"
            if args.reuse_frame_head else
            "ordinal criterion-conditioned head trained only on temporal13 train videos"
        ),
        "frame_head_reused": bool(args.reuse_frame_head),
        "temporal_head": "dilated residual TCN with ordinal correction and onset/offset auxiliary head",
        "parameter_counts": {
            "frozen_encoder": sum(parameter.numel() for parameter in source["encoder_state"].values()),
            "trained_frame_head": sum(parameter.numel() for parameter in frame_head.parameters()),
            "trained_temporal_head": sum(parameter.numel() for parameter in temporal_head.parameters()),
        },
        "split_manifest_path": str(split_path),
        "split_manifest": split,
        "train_video_ids": train_ids,
        "validation_video_ids": validation_ids,
        "test_video_ids": test_ids,
        "source_encoder_checkpoint": str(source_checkpoint),
        "source_encoder_checkpoint_sha256": sha256(source_checkpoint),
        "cache_manifest": sanitized_cache_manifest(cache_dir, test_ids),
        "frame_history": frame_history,
        "temporal_history": temporal_history,
        "model_selection": selection,
        "temporal_thresholds_calibrated_on_validation": thresholds,
        "ordinal_thresholds_calibrated_on_validation": thresholds,
        "decision_thresholds_calibrated_on_validation": thresholds["full_only"],
        "validation_temporal_metrics": validation_metrics,
        "test_temporal_metrics": test_metrics,
        "test_labels_accessed": bool(args.evaluate_test),
        "temporal_parameters": {
            "sample_every_s": float(next(iter(train_videos.values()))["sample_every_s"]),
            "min_stable_seconds": args.min_stable_seconds,
            "max_gap_seconds": args.max_gap_seconds,
            "window_length": args.window_length,
            "window_stride": args.window_stride,
            "hidden_dim": args.hidden_dim,
            "dilations": list(temporal_head.dilations),
            "dropout": args.dropout,
            "receptive_field_steps": temporal_head.receptive_field_steps,
            "loss_weights": {
                "dice": args.dice_weight, "boundary": args.boundary_weight,
                "early_false_positive": args.early_fp_weight,
                "consistency": args.consistency_weight,
                "smoothness": args.smoothness_weight,
                "full_classification": args.full_loss_weight,
                "full_dice": args.full_dice_weight,
                "full_positive_window": args.full_positive_window_weight,
            },
        },
        "pilot_only": True,
        "development_only": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": source["encoder_state"],
        "frame_head_state": frame_head.cpu().state_dict(),
        "temporal_head_state": temporal_head.cpu().state_dict(),
        "metadata": metadata,
    }, output)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"checkpoint={output}", flush=True)
    print(f"thresholds={thresholds}", flush=True)
    print(f"validation_macro_temporal_iou={validation_metrics['macro_temporal_iou']:.6f}", flush=True)
    if test_metrics is not None:
        print(f"held_out_test_macro_temporal_iou={test_metrics['macro_temporal_iou']:.6f}", flush=True)


if __name__ == "__main__":
    main()
