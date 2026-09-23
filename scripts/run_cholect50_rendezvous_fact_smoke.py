#!/usr/bin/env python3
"""Run a bounded CholecT50 atomic-fact smoke test with official Rendezvous weights.

This is deliberately not an official CholecT50 evaluation.  It emits only the
instrument/verb/target component heads as plugin facts; the model's triplet
head is never written to the facts file.  Expert labels are opened only after
inference and are written to a separate evaluation file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms


COMPONENT_SIZES = {"instrument": 6, "verb": 10, "target": 15}
ROW_COLUMNS = {"instrument": 1, "verb": 7, "target": 8}
COMPONENT_NAMES = {
    "instrument": ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"],
    "verb": [
        "grasp",
        "retract",
        "dissect",
        "coagulate",
        "clip",
        "cut",
        "aspirate",
        "irrigate",
        "pack",
        "null_verb",
    ],
    "target": [
        "gallbladder",
        "cystic_plate",
        "cystic_duct",
        "cystic_artery",
        "cystic_pedicle",
        "blood_vessel",
        "fluid",
        "abdominal_wall_cavity",
        "liver",
        "adhesion",
        "omentum",
        "peritoneum",
        "gut",
        "specimen_bag",
        "null_target",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument(
        "--checkpoint-for-video",
        action="append",
        default=None,
        metavar="VID=PATH",
        help="Repeat for each video to use held-out fold-specific checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-layer-norm", action="store_true")
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Optional video IDs such as VID68. Defaults to every available video.",
    )
    return parser.parse_args()


def canonical_frame_id(path: Path) -> int:
    stem = re.sub(r" \(1\)$", "", path.stem)
    return int(stem)


def canonical_images(video_dir: Path) -> list[Path]:
    """Deduplicate byte-identical official files with a trailing ' (1)'."""
    by_id: dict[int, Path] = {}
    for path in sorted(video_dir.glob("*.png")):
        frame_id = canonical_frame_id(path)
        current = by_id.get(frame_id)
        if current is None or (" (1)" in current.stem and " (1)" not in path.stem):
            by_id[frame_id] = path
    return [by_id[key] for key in sorted(by_id)]


def checkpoint_mapping(args: argparse.Namespace, videos: list[str]) -> dict[str, Path]:
    if args.checkpoint is not None:
        return {video: args.checkpoint for video in videos}
    parsed = {}
    for item in args.checkpoint_for_video:
        video, separator, checkpoint = item.partition("=")
        if not separator or not video or not checkpoint:
            raise ValueError(f"invalid --checkpoint-for-video value: {item!r}")
        parsed[video] = Path(checkpoint)
    missing = sorted(set(videos) - set(parsed))
    extra = sorted(set(parsed) - set(videos))
    if missing or extra:
        raise ValueError(f"checkpoint map mismatch: missing={missing}, extra={extra}")
    return parsed


def load_official_model(
    repo: Path, checkpoint: Path, device: torch.device, use_layer_norm: bool
) -> torch.nn.Module:
    # The released checkpoint contains the backbone.  Prevent the legacy
    # constructor from downloading ImageNet weights before loading it.
    original_resnet18 = torchvision.models.resnet18

    def resnet18_without_download(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        return original_resnet18(*args, **kwargs)

    torchvision.models.resnet18 = resnet18_without_download
    sys.path.insert(0, str(repo / "pytorch"))
    try:
        network = importlib.import_module("network")
        model = network.Rendezvous(
            "resnet18", hr_output=False, use_ln=use_layer_norm
        )
    finally:
        torchvision.models.resnet18 = original_resnet18

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval().to(device)
    return model


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
    return float((precision * ranked).sum() / positives)


def component_metrics(
    labels: np.ndarray, scores: np.ndarray, names: dict[str, str]
) -> dict:
    per_class = {}
    aps = []
    for index in range(labels.shape[1]):
        ap = average_precision(labels[:, index], scores[:, index])
        per_class[names[str(index)]] = {
            "positive_frames": int(labels[:, index].sum()),
            "ap": ap,
        }
        if ap is not None:
            aps.append(ap)
    positive_rows = labels.sum(axis=1) > 0
    top1 = scores.argmax(axis=1)
    top1_hit = labels[np.arange(len(labels)), top1].astype(bool)
    return {
        "macro_ap_over_present_classes": float(np.mean(aps)) if aps else None,
        "classes_with_positives": len(aps),
        "positive_frame_top1_hit_rate": (
            float(top1_hit[positive_rows].mean()) if positive_rows.any() else None
        ),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    labels_dir = args.dataset_dir / "labels"
    videos_dir = args.dataset_dir / "videos"
    videos = args.videos or sorted(path.name for path in videos_dir.glob("VID*"))
    checkpoints_by_video = checkpoint_mapping(args, videos)
    checkpoint_sha256 = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in set(checkpoints_by_video.values())
    }
    model_cache = {}
    records = []
    component_scores = {name: [] for name in COMPONENT_SIZES}

    # Prediction phase: do not open any label file here.
    for video in videos:
        checkpoint = checkpoints_by_video[video]
        checkpoint_key = str(checkpoint.resolve())
        if checkpoint_key not in model_cache:
            model_cache[checkpoint_key] = load_official_model(
                args.official_repo, checkpoint, device, args.use_layer_norm
            )
        model = model_cache[checkpoint_key]
        images = canonical_images(videos_dir / video)
        for start in range(0, len(images), args.batch_size):
            batch_paths = images[start : start + args.batch_size]
            batch = torch.stack(
                [transform(Image.open(path).convert("RGB")) for path in batch_paths]
            ).to(device)
            with torch.inference_mode():
                enc_i, enc_v, enc_t, _unused_triplet = model(batch)
                outputs = {
                    "instrument": torch.sigmoid(enc_i[1]).cpu().numpy(),
                    "verb": torch.sigmoid(enc_v[1]).cpu().numpy(),
                    "target": torch.sigmoid(enc_t[1]).cpu().numpy(),
                }
            for row_index, image_path in enumerate(batch_paths):
                record = {
                    "video": video,
                    "frame_id": canonical_frame_id(image_path),
                    "image": str(image_path.resolve()),
                    "model_checkpoint_sha256": checkpoint_sha256[checkpoint_key],
                    "facts": {},
                }
                for component, values in outputs.items():
                    vector = values[row_index]
                    component_scores[component].append(vector)
                    order = np.argsort(-vector, kind="stable")[:3]
                    record["facts"][component] = [
                        {
                            "class_id": int(index),
                            "name": COMPONENT_NAMES[component][int(index)],
                            "score": float(vector[index]),
                        }
                        for index in order
                    ]
                records.append(record)

    facts_path = args.output_dir / "FACTS.jsonl"
    with facts_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    scores_path = args.output_dir / "SCORES.npz"
    np.savez_compressed(
        scores_path,
        video=np.asarray([record["video"] for record in records]),
        frame_id=np.asarray([record["frame_id"] for record in records]),
        instrument=np.stack(component_scores["instrument"]),
        verb=np.stack(component_scores["verb"]),
        target=np.stack(component_scores["target"]),
    )

    # Evaluation phase: labels are loaded only after FACTS.jsonl is complete.
    category_names = None
    rows_by_frame = {}
    for video in videos:
        payload = json.loads((labels_dir / f"{video}.json").read_text())
        category_names = payload["categories"]
        for frame_id, rows in payload["annotations"].items():
            rows_by_frame[(video, int(frame_id))] = rows

    component_labels = {
        name: np.zeros((len(records), size), dtype=np.int64)
        for name, size in COMPONENT_SIZES.items()
    }
    atomic_tuple_hits = []
    positive_frames = []
    for record_index, record in enumerate(records):
        rows = rows_by_frame.get((record["video"], record["frame_id"]), [])
        positive_frames.append(bool(rows))
        truth_tuples = set()
        for row in rows:
            truth_tuples.add((int(row[1]), int(row[7]), int(row[8])))
            for component, column in ROW_COLUMNS.items():
                component_labels[component][record_index, int(row[column])] = 1
        predicted_tuple = tuple(
            int(np.argmax(component_scores[component][record_index]))
            for component in ("instrument", "verb", "target")
        )
        atomic_tuple_hits.append(predicted_tuple in truth_tuples)

    positive_frames_array = np.asarray(positive_frames, dtype=bool)
    atomic_tuple_hits_array = np.asarray(atomic_tuple_hits, dtype=bool)
    metrics = {}
    for component in COMPONENT_SIZES:
        metrics[component] = component_metrics(
            component_labels[component],
            np.stack(component_scores[component]),
            category_names[component],
        )

    evaluation = {
        "status": "development_smoke_only",
        "not_an_official_benchmark": True,
        "no_qwen_calls": True,
        "triplet_head_exposed_to_qwen": False,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "official_repo": str(args.official_repo.resolve()),
        "checkpoints_by_video": {
            video: {
                "path": str(checkpoints_by_video[video].resolve()),
                "sha256": checkpoint_sha256[str(checkpoints_by_video[video].resolve())],
            }
            for video in videos
        },
        "use_layer_norm": args.use_layer_norm,
        "videos": videos,
        "canonical_frames": len(records),
        "expert_positive_frames": int(positive_frames_array.sum()),
        "label_free_score_archive": str(scores_path.resolve()),
        "component_metrics": metrics,
        "independent_component_top1_tuple_hit_rate_on_positive_frames": (
            float(atomic_tuple_hits_array[positive_frames_array].mean())
            if positive_frames_array.any()
            else None
        ),
        "limitations": [
            "The public challenge-validation labels are consumed development data.",
            "These metrics are subset diagnostics, not official CholecT50 metrics.",
            "A checkpoint whose training videos overlap these clips cannot support a generalization claim.",
            "The full official cross-validation release is still required for a formal experiment.",
        ],
    }
    (args.output_dir / "EVALUATION.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
