"""Train a versioned, criterion-conditioned CVS visual evidence tool.

Splits are always expressed as video IDs.  A checkpoint without held-out
validation videos is marked development_only and must not be reported as test
performance.  Frames are decoded once, kept as uint8 tensors, and trained in
batches with class-imbalance weighting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.models import CriterionConditionedCvsModel, SharedFrameEncoder, UnifiedThreeLabelHead

CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")


def labels_at(intervals, time_s):
    return torch.tensor([
        max((value for start, end, value in intervals[key] if start <= time_s <= end), default=0) / 2.0
        for key in CRITERIA
    ], dtype=torch.float32)


def decode_samples(video_root: Path, annotation_xlsx: str, video_ids: list[int], sample_every_s: float):
    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    provenance: list[tuple[int, float]] = []
    for video_id in video_ids:
        video = video_root / f"video{video_id:02d}.mp4"
        if not video.exists():
            raise FileNotFoundError(video)
        intervals = load_cvs_intervals(annotation_xlsx, video_id)
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        for time_s in np.arange(0.0, duration, sample_every_s):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB).copy()
            images.append(torch.from_numpy(rgb).permute(2, 0, 1))
            labels.append(labels_at(intervals, float(time_s)))
            provenance.append((video_id, float(time_s)))
        cap.release()
    if not images:
        raise RuntimeError("No training frames were decoded")
    return torch.stack(images), torch.stack(labels), provenance


def logits_for(model, images, model_type: str):
    output = model(images)
    return torch.stack([output[key] for key in CRITERIA], dim=1) if model_type == "conditioned" else output


@torch.inference_mode()
def evaluate(model, loader, transform, device, model_type: str):
    model.eval()
    logits, targets = [], []
    for images, target in loader:
        images = transform(images.to(device))
        logits.append(logits_for(model, images, model_type).cpu())
        targets.append(target)
    scores = torch.sigmoid(torch.cat(logits))
    truth = torch.cat(targets) >= 0.5
    prediction = scores >= 0.5
    metrics = {}
    for index, criterion in enumerate(CRITERIA):
        tp = int((prediction[:, index] & truth[:, index]).sum())
        fp = int((prediction[:, index] & ~truth[:, index]).sum())
        fn = int((~prediction[:, index] & truth[:, index]).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        metrics[criterion] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "n_positive": int(truth[:, index].sum()),
            "n_samples": len(truth),
        }
    model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a specialized CVS evidence tool.")
    parser.add_argument("--annotation-xlsx", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--train-videos", nargs="+", type=int, required=True)
    parser.add_argument("--validation-videos", nargs="*", type=int, default=[])
    parser.add_argument("--model", choices=["conditioned", "unified"], default="conditioned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-every-s", type=float, default=5.0)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    overlap = set(args.train_videos) & set(args.validation_videos)
    if overlap:
        raise ValueError(f"Train/validation videos overlap: {sorted(overlap)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_root = Path(args.video_root)
    train_images, train_targets, train_provenance = decode_samples(
        video_root, args.annotation_xlsx, args.train_videos, args.sample_every_s,
    )
    train_loader = DataLoader(TensorDataset(train_images, train_targets), batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(TensorDataset(train_images, train_targets), batch_size=args.batch_size)
    validation_loader = None
    if args.validation_videos:
        validation_images, validation_targets, _ = decode_samples(
            video_root, args.annotation_xlsx, args.validation_videos, args.sample_every_s,
        )
        validation_loader = DataLoader(TensorDataset(validation_images, validation_targets), batch_size=args.batch_size)

    if args.model == "conditioned":
        model = CriterionConditionedCvsModel(pretrained_backbone=args.pretrained_backbone)
    else:
        model = nn.Sequential(SharedFrameEncoder(pretrained=args.pretrained_backbone), UnifiedThreeLabelHead())
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    positives = train_targets.sum(dim=0)
    pos_weight = ((len(train_targets) - positives) / positives.clamp_min(1.0)).clamp(max=30.0).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    history = []
    for epoch in range(args.epochs):
        total, count = 0.0, 0
        for images, target in train_loader:
            images = transform(images.to(device))
            target = target.to(device)
            loss = loss_fn(logits_for(model, images, args.model), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            count += len(images)
        mean_loss = total / max(1, count)
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss})
        print(f"epoch={epoch + 1} mean_loss={mean_loss:.4f}", flush=True)

    train_metrics = evaluate(model, train_eval_loader, transform, device, args.model)
    validation_metrics = evaluate(model, validation_loader, transform, device, args.model) if validation_loader else None
    metadata = {
        "tool_type": "specialized_cvs_visual_evidence",
        "checkpoint_version": "0.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "criteria": CRITERIA,
        "model": args.model,
        "train_video_ids": args.train_videos,
        "validation_video_ids": args.validation_videos,
        "development_only": not bool(args.validation_videos),
        "pretrained_backbone": args.pretrained_backbone,
        "sample_every_s": args.sample_every_s,
        "n_train_samples": len(train_provenance),
        "positive_target_mass": {criterion: float(positives[i]) for i, criterion in enumerate(CRITERIA)},
        "pos_weight": {criterion: float(pos_weight[i]) for i, criterion in enumerate(CRITERIA)},
        "history": history,
        "train_metrics_not_for_reporting": train_metrics,
        "validation_metrics": validation_metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, output)
    print(f"checkpoint={output}", flush=True)
    print(f"development_only={metadata['development_only']}", flush=True)
    print(f"train_metrics_not_for_reporting={train_metrics}", flush=True)


if __name__ == "__main__":
    main()
