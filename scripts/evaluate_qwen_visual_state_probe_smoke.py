#!/usr/bin/env python3
"""OOF smoke test for a linear SOP-state adapter over frozen Qwen visual features."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.evidence_pack import symmetric_timestamps


def decode_frame(dataset_root: Path, video_id: int, time_s: float) -> Image.Image:
    capture = cv2.VideoCapture(str(dataset_root / "videos" / f"video{video_id:02d}.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video{video_id:02d}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode video{video_id:02d} at {time_s:.3f}s")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def state_at(intervals: list[tuple[float, float, int]], time_s: float) -> int | None:
    values = [value for start_s, end_s, value in intervals if start_s <= time_s <= end_s]
    return max(values) if values else None


def extract_features(
    model: Qwen3VLForConditionalGeneration, processor: AutoProcessor,
    images: list[Image.Image], batch_size: int, progress_every: int = 10,
) -> np.ndarray:
    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    pooled = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = processor(
            text=[prompt] * len(batch), images=batch, padding=True, return_tensors="pt",
        )
        with torch.inference_mode():
            image_features, _ = model.get_image_features(
                inputs.pixel_values.to(model.device, dtype=torch.bfloat16),
                inputs.image_grid_thw.to(model.device),
            )
        pooled.extend(value.float().mean(dim=0).cpu() for value in image_features)
        batch_index = start // batch_size + 1
        if batch_index % progress_every == 0 or start + batch_size >= len(images):
            print(f"feature_batch {min(start + batch_size, len(images))}/{len(images)}", flush=True)
    return torch.stack(pooled).numpy()


def extract_spatial_pyramid_features(
    model: Qwen3VLForConditionalGeneration, processor: AutoProcessor,
    images: list[Image.Image], batch_size: int, progress_every: int = 10,
) -> np.ndarray:
    """Pool frozen visual tokens without discarding coarse anatomical position.

    Qwen's vision encoder returns the post-merge tokens in raster order.  For
    every frame we concatenate six pooled descriptors: the complete view,
    central operative field, upper and lower central fields, and left and
    right context.  This keeps the foundation model frozen while giving the
    lightweight state adapter access to coarse spatial relationships.
    """
    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    pooled = []
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))

    def bounds(length: int, start_fraction: float, end_fraction: float) -> tuple[int, int]:
        start = min(length - 1, max(0, int(np.floor(length * start_fraction))))
        end = min(length, max(start + 1, int(np.ceil(length * end_fraction))))
        return start, end

    regions = (
        (0.00, 1.00, 0.00, 1.00),  # complete view
        (0.15, 0.90, 0.15, 0.85),  # central operative field
        (0.00, 0.60, 0.15, 0.85),  # upper central field
        (0.40, 1.00, 0.15, 0.85),  # lower central field
        (0.00, 1.00, 0.00, 0.60),  # left context
        (0.00, 1.00, 0.40, 1.00),  # right context
    )
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = processor(
            text=[prompt] * len(batch), images=batch, padding=True, return_tensors="pt",
        )
        grids = inputs.image_grid_thw
        with torch.inference_mode():
            image_features, _ = model.get_image_features(
                inputs.pixel_values.to(model.device, dtype=torch.bfloat16),
                grids.to(model.device),
            )
        if len(image_features) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} image feature groups, got {len(image_features)}"
            )
        for tokens, grid in zip(image_features, grids):
            temporal, height, width = (int(value) for value in grid.tolist())
            merged_height = height // merge_size
            merged_width = width // merge_size
            expected = temporal * merged_height * merged_width
            if tokens.shape[0] != expected:
                raise RuntimeError(
                    "Unexpected Qwen visual-token geometry: "
                    f"grid={tuple(grid.tolist())} merge={merge_size} "
                    f"tokens={tokens.shape[0]} expected={expected}"
                )
            token_map = tokens.float().reshape(temporal, merged_height, merged_width, -1)
            token_map = token_map.mean(dim=0)
            descriptors = []
            for y0f, y1f, x0f, x1f in regions:
                y0, y1 = bounds(merged_height, y0f, y1f)
                x0, x1 = bounds(merged_width, x0f, x1f)
                descriptors.append(token_map[y0:y1, x0:x1].mean(dim=(0, 1)))
            pooled.append(torch.cat(descriptors).cpu())
        batch_index = start // batch_size + 1
        if batch_index % progress_every == 0 or start + batch_size >= len(images):
            print(
                f"spatial_feature_batch {min(start + batch_size, len(images))}/{len(images)}",
                flush=True,
            )
    return torch.stack(pooled).numpy()


def fit_linear_probe(features: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    x = torch.from_numpy(features).float()
    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.from_numpy(labels).float()
    layer = torch.nn.Linear(x.shape[1], 1)
    positive = max(1, int((labels == 1).sum()))
    negative = max(1, int((labels == 0).sum()))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive]))
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.015, weight_decay=0.08)
    for _ in range(1200):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(layer(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
    return layer.weight.detach().squeeze(0), layer.bias.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--atlas-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--criterion", default="hepatocystic_triangle")
    parser.add_argument("--candidate-ids", nargs="+", required=True)
    parser.add_argument("--candidate-center-fraction", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    atlas = json.loads(args.atlas_manifest.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    by_id = {item["candidate_id"]: item for item in candidate_manifest["candidates"]}
    queries = [by_id[candidate_id] for candidate_id in args.candidate_ids]
    query_video_ids = {int(item["video_id"]) for item in queries}

    train_records = [
        item for item in atlas["records"]
        if item["criterion"] == args.criterion and int(item["video_id"]) not in query_video_ids
    ]
    records = []
    images = []
    for item in train_records:
        images.append(decode_frame(args.dataset_root, int(item["video_id"]), float(item["time_s"])))
        records.append({
            "role": "train", "video_id": int(item["video_id"]),
            "time_s": float(item["time_s"]), "label": int(item["label"] == "positive"),
            "prototype_id": item["prototype_id"],
        })
    annotation_path = args.dataset_root / "annotations" / "cholec80-CVS.xlsx"
    for candidate in queries:
        video_id = int(candidate["video_id"])
        interval_start_s, interval_end_s = map(float, candidate["interval_s"])
        center_s = interval_start_s + args.candidate_center_fraction * (interval_end_s - interval_start_s)
        evaluation_start_s, evaluation_end_s = map(float, candidate["evaluation_window_s"])
        timestamps = symmetric_timestamps(
            center_s, 13, 2.0, start_s=evaluation_start_s, end_s=evaluation_end_s,
        )
        annotated = load_cvs_intervals(annotation_path, video_id)[args.criterion]
        for time_s in timestamps:
            state = state_at(annotated, time_s)
            images.append(decode_frame(args.dataset_root, video_id, time_s))
            records.append({
                "role": "query", "candidate_id": candidate["candidate_id"],
                "video_id": video_id, "time_s": time_s, "expert_state": state,
                "label": int(state is not None and state >= 2),
            })

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
    ).eval()
    features = extract_features(model, processor, images, args.batch_size)
    train_indices = [index for index, item in enumerate(records) if item["role"] == "train"]
    query_indices = [index for index, item in enumerate(records) if item["role"] == "query"]
    train_labels = np.asarray([records[index]["label"] for index in train_indices], dtype=np.float32)
    weight, bias = fit_linear_probe(features[train_indices], train_labels)
    normalized = torch.nn.functional.normalize(torch.from_numpy(features).float(), dim=1)
    probabilities = torch.sigmoid(normalized @ weight + bias).numpy()

    query_results = {}
    for candidate in queries:
        candidate_id = candidate["candidate_id"]
        items = []
        for index in query_indices:
            if records[index]["candidate_id"] != candidate_id:
                continue
            items.append({**records[index], "probability": float(probabilities[index])})
        frame_predictions = [item["probability"] >= 0.5 for item in items]
        query_results[candidate_id] = {
            "frames": items,
            "mean_probability": float(np.mean([item["probability"] for item in items])),
            "positive_frame_count": int(sum(frame_predictions)),
            "majority_prediction": "positive" if sum(frame_predictions) >= 7 else "negative",
            "expert_states": dict(Counter(str(item["expert_state"]) for item in items)),
        }
    result = {
        "schema_version": "qwen_visual_state_probe_smoke_v1",
        "criterion": args.criterion,
        "foundation_model": str(args.model_path.resolve()),
        "adapter": "balanced_l2_linear_probe_on_frozen_mean_pooled_visual_tokens",
        "query_video_ids_excluded_from_training": sorted(query_video_ids),
        "training_examples": len(train_indices),
        "training_label_counts": dict(Counter(str(int(value)) for value in train_labels)),
        "query_results": query_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
