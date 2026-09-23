"""Train a lightweight CVS head on a frozen PeskaVLP image encoder.

The saved checkpoint is self-contained: it includes the frozen visual tower and
the trained head, while metadata retains the original PeskaVLP weight checksum.
Dataset splits are video-level.  Runs without held-out videos are explicitly
marked development-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.models import CriterionConditionedHead, PeskaVLPVisualEncoder

CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")
OFFICIAL_REPOSITORY = "https://github.com/CAMMA-public/SurgVLP"
OFFICIAL_WEIGHT_URL = "https://seafile.unistra.fr/f/65a2b1bf113e428280d0/?dl=1"


def labels_at(intervals, time_s: float) -> torch.Tensor:
    return torch.tensor([
        max((value for start, end, value in intervals[key] if start <= time_s <= end), default=0) / 2.0
        for key in CRITERIA
    ], dtype=torch.float32)


def decode_samples(video_root: Path, annotation_xlsx: str, video_ids: list[int], sample_every_s: float):
    images, labels, provenance = [], [], []
    for video_id in video_ids:
        video_path = video_root / f"video{video_id:02d}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        print(f"decode_video={video_id:02d} path={video_path}", flush=True)
        intervals = load_cvs_intervals(annotation_xlsx, video_id)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError(f"Cannot read FPS from {video_path}")
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        for time_s in np.arange(0.0, duration, sample_every_s):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            # Match PeskaVLP test preprocessing: Resize((360, 640)), then
            # CenterCrop(224), then ImageNet normalization.
            rgb = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)
            top, left = (360 - 224) // 2, (640 - 224) // 2
            crop = rgb[top:top + 224, left:left + 224].copy()
            images.append(torch.from_numpy(crop).permute(2, 0, 1))
            labels.append(labels_at(intervals, float(time_s)))
            provenance.append((video_id, float(time_s)))
        cap.release()
        print(f"decoded_video={video_id:02d} cumulative_samples={len(images)}", flush=True)
    if not images:
        raise RuntimeError("No samples were decoded")
    return torch.stack(images), torch.stack(labels), provenance


@torch.inference_mode()
def extract_features(encoder, images: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    loader = DataLoader(TensorDataset(images), batch_size=batch_size)
    features = []
    encoder.to(device).freeze()
    for (batch,) in loader:
        features.append(encoder(transform(batch.to(device))).cpu())
    return torch.cat(features)


def stack_logits(head, features: torch.Tensor) -> torch.Tensor:
    output = head(features)
    return torch.stack([output[key] for key in CRITERIA], dim=1)


@torch.inference_mode()
def predict_scores(head, features: torch.Tensor, device: str) -> torch.Tensor:
    head.eval()
    scores = torch.sigmoid(stack_logits(head, features.to(device))).cpu()
    head.train()
    return scores


def binary_metrics(scores: torch.Tensor, truth: torch.Tensor, threshold: float) -> dict:
    scores, truth = scores.float().flatten(), truth.bool().flatten()
    prediction = scores >= threshold
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    positives, negatives = int(truth.sum()), int((~truth).sum())

    auroc = None
    if positives and negatives:
        order = torch.argsort(scores)
        sorted_scores = scores[order]
        ranks = torch.empty_like(scores)
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        positive_rank_sum = float(ranks[truth].sum())
        auroc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    auprc = None
    if positives:
        order = torch.argsort(scores, descending=True)
        sorted_scores, sorted_truth = scores[order], truth[order]
        cumulative_tp = 0
        seen = 0
        previous_recall = 0.0
        auprc = 0.0
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            cumulative_tp += int(sorted_truth[start:end].sum())
            seen += end - start
            recall_at_threshold = cumulative_tp / positives
            precision_at_threshold = cumulative_tp / seen
            auprc += (recall_at_threshold - previous_recall) * precision_at_threshold
            previous_recall = recall_at_threshold
            start = end
    return {
        "threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
        "auroc": auroc, "auprc": auprc, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_positive": positives, "n_negative": negatives, "n_samples": len(truth),
    }


def evaluate_scores(scores: torch.Tensor, targets: torch.Tensor, thresholds: dict[str, float]) -> dict:
    metrics = {}
    for index, criterion in enumerate(CRITERIA):
        metrics[criterion] = {
            "support_or_full": binary_metrics(scores[:, index], targets[:, index] >= 0.5, thresholds[criterion]),
            "full_only": binary_metrics(scores[:, index], targets[:, index] >= 0.999, thresholds[criterion]),
            "brier_soft_target": float(torch.mean((scores[:, index] - targets[:, index]) ** 2)),
        }
    return metrics


def macro_metric(metrics: dict, metric: str, label_definition: str = "support_or_full") -> float:
    values = [value[label_definition][metric] for value in metrics.values() if value[label_definition][metric] is not None]
    return sum(values) / max(1, len(values))


def calibrate_thresholds(scores: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    thresholds = {}
    candidates = torch.linspace(0.05, 0.95, 91)
    for index, criterion in enumerate(CRITERIA):
        truth = targets[:, index] >= 0.5
        best_threshold, best_f1 = 0.5, -1.0
        for candidate in candidates:
            threshold = float(candidate)
            f1 = binary_metrics(scores[:, index], truth, threshold)["f1"]
            if f1 > best_f1:
                best_threshold, best_f1 = threshold, f1
        thresholds[criterion] = best_threshold
    return thresholds


def per_video_metrics(
    scores: torch.Tensor, targets: torch.Tensor, provenance: list[tuple[int, float]], thresholds: dict[str, float],
) -> dict[str, dict]:
    result = {}
    for video_id in sorted({video_id for video_id, _ in provenance}):
        indices = [index for index, (sample_video_id, _) in enumerate(provenance) if sample_video_id == video_id]
        result[str(video_id)] = evaluate_scores(scores[indices], targets[indices], thresholds)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CVS head on frozen PeskaVLP visual features.")
    parser.add_argument("--peskavlp-checkpoint", required=True)
    parser.add_argument("--annotation-xlsx", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--split-json", help="Fixed video-level train/validation/test split manifest.")
    parser.add_argument("--train-videos", nargs="+", type=int)
    parser.add_argument("--validation-videos", nargs="*", type=int, default=[])
    parser.add_argument("--test-videos", nargs="*", type=int, default=[])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-every-s", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-mirror-url", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split_manifest = None
    if args.split_json:
        split_path = Path(args.split_json).resolve()
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        train_video_ids = [int(item) for item in split_manifest["train_video_ids"]]
        validation_video_ids = [int(item) for item in split_manifest["validation_video_ids"]]
        test_video_ids = [int(item) for item in split_manifest["test_video_ids"]]
    else:
        if not args.train_videos:
            raise ValueError("Provide --split-json or --train-videos")
        split_path = None
        train_video_ids = args.train_videos
        validation_video_ids = args.validation_videos
        test_video_ids = args.test_videos
    named_splits = {
        "train": set(train_video_ids), "validation": set(validation_video_ids), "test": set(test_video_ids),
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = named_splits[left] & named_splits[right]
        if overlap:
            raise ValueError(f"{left}/{right} videos overlap: {sorted(overlap)}")
    print(f"split_train={train_video_ids}", flush=True)
    print(f"split_validation={validation_video_ids}", flush=True)
    print(f"split_test_held_out={test_video_ids}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_checkpoint = Path(args.peskavlp_checkpoint).resolve()

    encoder = PeskaVLPVisualEncoder()
    load_info = encoder.load_official_checkpoint(source_checkpoint)
    train_images, train_targets, train_provenance = decode_samples(
        Path(args.video_root), args.annotation_xlsx, train_video_ids, args.sample_every_s,
    )
    train_features = extract_features(encoder, train_images, args.batch_size, device)
    print(f"train_features={tuple(train_features.shape)}", flush=True)
    del train_images

    validation_features = validation_targets = validation_provenance = None
    if validation_video_ids:
        validation_images, validation_targets, validation_provenance = decode_samples(
            Path(args.video_root), args.annotation_xlsx, validation_video_ids, args.sample_every_s,
        )
        validation_features = extract_features(encoder, validation_images, args.batch_size, device)
        print(f"validation_features={tuple(validation_features.shape)}", flush=True)
        del validation_images

    head = CriterionConditionedHead(feature_dim=encoder.output_dim).to(device).train()
    positives = train_targets.sum(dim=0)
    pos_weight = ((len(train_targets) - positives) / positives.clamp_min(1.0)).clamp(max=30.0).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=args.batch_size, shuffle=True)

    history = []
    best_state = None
    best_epoch = 0
    best_selection_value = float("-inf")
    epochs_without_improvement = 0
    default_thresholds = {criterion: 0.5 for criterion in CRITERIA}
    for epoch in range(args.epochs):
        total = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            loss = loss_fn(stack_logits(head, features), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(features)
        mean_loss = total / len(train_targets)
        record = {"epoch": epoch + 1, "mean_loss": mean_loss}
        if validation_features is not None and validation_targets is not None:
            validation_epoch_scores = predict_scores(head, validation_features, device)
            validation_epoch_metrics = evaluate_scores(validation_epoch_scores, validation_targets, default_thresholds)
            selection_value = macro_metric(validation_epoch_metrics, "auprc")
            record["validation_macro_auprc_support_or_full"] = selection_value
        else:
            selection_value = -mean_loss
        history.append(record)
        if selection_value > best_selection_value + args.min_delta:
            best_selection_value = selection_value
            best_epoch = epoch + 1
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch + 1} mean_loss={mean_loss:.5f} selection={selection_value:.5f} "
            f"best_epoch={best_epoch}", flush=True,
        )
        if validation_features is not None and epochs_without_improvement >= args.patience:
            print(f"early_stopping_epoch={epoch + 1} patience={args.patience}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no selectable checkpoint")
    head.load_state_dict(best_state)
    train_scores = predict_scores(head, train_features, device)
    if validation_features is not None and validation_targets is not None:
        validation_scores = predict_scores(head, validation_features, device)
        calibrated_thresholds = calibrate_thresholds(validation_scores, validation_targets)
        validation_metrics = evaluate_scores(validation_scores, validation_targets, calibrated_thresholds)
        validation_video_metrics = per_video_metrics(
            validation_scores, validation_targets, validation_provenance, calibrated_thresholds,
        )
    else:
        calibrated_thresholds = default_thresholds
        validation_metrics = validation_video_metrics = None
    train_metrics = evaluate_scores(train_scores, train_targets, calibrated_thresholds)

    # Test videos are decoded only after model selection and validation-based
    # threshold calibration are complete.
    test_metrics = test_video_metrics = None
    if test_video_ids:
        print("model_selection_complete_now_loading_held_out_test=true", flush=True)
        test_images, test_targets, test_provenance = decode_samples(
            Path(args.video_root), args.annotation_xlsx, test_video_ids, args.sample_every_s,
        )
        test_features = extract_features(encoder, test_images, args.batch_size, device)
        print(f"test_features={tuple(test_features.shape)}", flush=True)
        del test_images
        test_scores = predict_scores(head, test_features, device)
        test_metrics = evaluate_scores(test_scores, test_targets, calibrated_thresholds)
        test_video_metrics = per_video_metrics(test_scores, test_targets, test_provenance, calibrated_thresholds)
    metadata = {
        "tool_type": "peskavlp_frozen_encoder_cvs_head",
        "checkpoint_version": "0.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "criteria": CRITERIA,
        "encoder": "PeskaVLP ResNet-50 visual tower + 768-D projection (frozen)",
        "head": "criterion-conditioned lightweight head",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_weight_url": OFFICIAL_WEIGHT_URL,
        "source_mirror_url": args.source_mirror_url,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "source_load_info": load_info,
        "license_note": "PeskaVLP repository states CC BY-NC-SA 4.0 for non-commercial scientific research.",
        "split_manifest_path": str(split_path) if split_path else None,
        "split_manifest": split_manifest,
        "pilot_only": bool(split_manifest and "pilot" in str(split_manifest.get("name", "")).lower()),
        "train_video_ids": train_video_ids,
        "validation_video_ids": validation_video_ids,
        "test_video_ids": test_video_ids,
        "development_only": not bool(validation_video_ids and test_video_ids),
        "sample_every_s": args.sample_every_s,
        "n_train_samples": len(train_provenance),
        "positive_target_mass": {criterion: float(positives[i]) for i, criterion in enumerate(CRITERIA)},
        "pos_weight": {criterion: float(pos_weight[i]) for i, criterion in enumerate(CRITERIA)},
        "history": history,
        "selection_metric": "validation_macro_auprc_support_or_full" if validation_video_ids else "negative_train_loss",
        "best_epoch": best_epoch,
        "best_selection_value": best_selection_value,
        "early_stopping_patience": args.patience,
        "decision_thresholds_calibrated_on_validation": calibrated_thresholds,
        "train_metrics_not_for_reporting": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_per_video_metrics": validation_video_metrics,
        "test_metrics": test_metrics,
        "test_per_video_metrics": test_video_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": encoder.cpu().state_dict(),
        "head_state": head.cpu().state_dict(),
        "metadata": metadata,
    }, output)
    print(f"checkpoint={output}", flush=True)
    print(f"development_only={metadata['development_only']}", flush=True)
    print(f"best_epoch={best_epoch} calibrated_thresholds={calibrated_thresholds}", flush=True)
    print(f"validation_metrics={validation_metrics}", flush=True)
    print(f"held_out_test_metrics={test_metrics}", flush=True)


if __name__ == "__main__":
    main()
