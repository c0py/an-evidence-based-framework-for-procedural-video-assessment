#!/usr/bin/env python3
"""Evaluate frozen-Qwen visual state adapters on all training candidates with video OOF."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.annotations import load_cvs_intervals
from cvs_assessment.evidence_pack import symmetric_timestamps
from evaluate_qwen_visual_state_probe_smoke import extract_features, fit_linear_probe, state_at


CRITERIA = ("cystic_plate", "hepatocystic_triangle")


def metrics(labels: list[int], predictions: list[int], probabilities: list[float]) -> dict:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    positive_scores = [s for y, s in zip(labels, probabilities) if y == 1]
    negative_scores = [s for y, s in zip(labels, probabilities) if y == 0]
    auc = None
    if positive_scores and negative_scores:
        wins = sum(
            float(pos > neg) + 0.5 * float(pos == neg)
            for pos in positive_scores for neg in negative_scores
        )
        auc = wins / (len(positive_scores) * len(negative_scores))
    return {
        "n": len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": tp / max(1, tp + fp),
        "recall_sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "balanced_accuracy": 0.5 * (
            tp / max(1, tp + fn) + tn / max(1, tn + fp)
        ),
        "roc_auc": auc,
    }


def decode_unique_frames(dataset_root: Path, keys: list[tuple[int, float]]) -> list[Image.Image]:
    by_video: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for index, (video_id, time_s) in enumerate(keys):
        by_video[video_id].append((index, time_s))
    decoded: list[Image.Image | None] = [None] * len(keys)
    for progress, (video_id, values) in enumerate(sorted(by_video.items()), start=1):
        capture = cv2.VideoCapture(str(dataset_root / "videos" / f"video{video_id:02d}.mp4"))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video{video_id:02d}")
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        last_decodable_s = max(0.0, (frame_count - 2.0) / fps) if fps > 0 else None
        try:
            for index, time_s in sorted(values, key=lambda item: item[1]):
                capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
                ok, frame = capture.read()
                if not ok and last_decodable_s is not None and time_s > last_decodable_s:
                    capture.set(cv2.CAP_PROP_POS_MSEC, last_decodable_s * 1000.0)
                    ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode video{video_id:02d} at {time_s:.3f}s")
                decoded[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        print(f"decode_video {progress}/{len(by_video)} video={video_id:02d}", flush=True)
    if any(image is None for image in decoded):
        raise RuntimeError("Some requested frames were not decoded")
    return [image for image in decoded if image is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--atlas-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--candidate-center-fraction", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    atlas = json.loads(args.atlas_manifest.read_text(encoding="utf-8"))
    candidates = [
        item for item in json.loads(args.candidate_manifest.read_text(encoding="utf-8"))["candidates"]
        if item["criterion"] in CRITERIA
    ]
    annotation_path = args.dataset_root / "annotations" / "cholec80-CVS.xlsx"
    annotation_cache: dict[tuple[int, str], list[tuple[float, float, int]]] = {}
    records = []
    for item in atlas["records"]:
        if item["criterion"] not in CRITERIA:
            continue
        records.append({
            "role": "train", "criterion": item["criterion"],
            "video_id": int(item["video_id"]), "time_s": float(item["time_s"]),
            "label": int(item["label"] == "positive"),
            "prototype_id": item["prototype_id"],
        })
    for candidate in candidates:
        video_id, criterion = int(candidate["video_id"]), candidate["criterion"]
        key = (video_id, criterion)
        if key not in annotation_cache:
            annotation_cache[key] = load_cvs_intervals(annotation_path, video_id)[criterion]
        interval_start_s, interval_end_s = map(float, candidate["interval_s"])
        center_s = interval_start_s + args.candidate_center_fraction * (interval_end_s - interval_start_s)
        evaluation_start_s, evaluation_end_s = map(float, candidate["evaluation_window_s"])
        timestamps = symmetric_timestamps(
            center_s, 13, 2.0, start_s=evaluation_start_s, end_s=evaluation_end_s,
        )
        for time_s in timestamps:
            records.append({
                "role": "query", "candidate_id": candidate["candidate_id"],
                "candidate_class": candidate["candidate_class"], "criterion": criterion,
                "video_id": video_id, "time_s": time_s,
                "expert_state": state_at(annotation_cache[key], time_s),
            })

    unique_keys = sorted({(item["video_id"], item["time_s"]) for item in records})
    key_to_index = {key: index for index, key in enumerate(unique_keys)}
    print(
        f"oof_dataset candidates={len(candidates)} records={len(records)} unique_frames={len(unique_keys)}",
        flush=True,
    )
    images = decode_unique_frames(args.dataset_root, unique_keys)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
    ).eval()
    features = extract_features(model, processor, images, args.batch_size, progress_every=10)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "keys": unique_keys, "features": torch.from_numpy(features).to(torch.float16),
        "model_path": str(args.model_path.resolve()),
    }, args.output_dir / "frozen_visual_features.pt")

    query_records = [item for item in records if item["role"] == "query"]
    for criterion in CRITERIA:
        videos = sorted({item["video_id"] for item in query_records if item["criterion"] == criterion})
        for progress, video_id in enumerate(videos, start=1):
            train = [
                item for item in records if item["role"] == "train"
                and item["criterion"] == criterion and item["video_id"] != video_id
            ]
            train_indices = [key_to_index[(item["video_id"], item["time_s"])] for item in train]
            train_labels = np.asarray([item["label"] for item in train], dtype=np.float32)
            weight, bias = fit_linear_probe(features[train_indices], train_labels)
            current = [
                item for item in query_records
                if item["criterion"] == criterion and item["video_id"] == video_id
            ]
            current_indices = [key_to_index[(item["video_id"], item["time_s"])] for item in current]
            normalized = torch.nn.functional.normalize(
                torch.from_numpy(features[current_indices]).float(), dim=1,
            )
            probabilities = torch.sigmoid(normalized @ weight + bias).numpy()
            for item, probability in zip(current, probabilities):
                item["probability"] = float(probability)
            print(
                f"oof_probe {criterion} {progress}/{len(videos)} video={video_id:02d}", flush=True,
            )

    frame_reports = {}
    for criterion in (*CRITERIA, "overall"):
        selected = [
            item for item in query_records
            if (criterion == "overall" or item["criterion"] == criterion)
            and item["expert_state"] in {0, 2}
        ]
        labels = [int(item["expert_state"] == 2) for item in selected]
        probabilities = [item["probability"] for item in selected]
        predictions = [int(value >= 0.5) for value in probabilities]
        frame_reports[criterion] = metrics(labels, predictions, probabilities)

    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for item in query_records:
        by_candidate[item["candidate_id"]].append(item)
    candidate_rows = []
    for candidate in candidates:
        frames = by_candidate[candidate["candidate_id"]]
        full_count = sum(item["expert_state"] == 2 for item in frames)
        nonfull_known_count = sum(item["expert_state"] in {0, 1} for item in frames)
        if full_count >= 7:
            target = 1
        elif nonfull_known_count >= 7 and full_count == 0:
            target = 0
        else:
            target = None
        probabilities = [item["probability"] for item in frames]
        positive_count = sum(value >= 0.5 for value in probabilities)
        candidate_rows.append({
            "candidate_id": candidate["candidate_id"], "criterion": candidate["criterion"],
            "video_id": int(candidate["video_id"]), "candidate_class": candidate["candidate_class"],
            "target": target, "full_frame_count": full_count,
            "nonfull_known_frame_count": nonfull_known_count,
            "missing_frame_count": sum(item["expert_state"] is None for item in frames),
            "mean_probability": float(np.mean(probabilities)),
            "positive_frame_count": positive_count,
            "prediction": int(positive_count >= 7),
        })
    candidate_reports = {}
    for criterion in (*CRITERIA, "overall"):
        selected = [
            item for item in candidate_rows
            if item["target"] is not None
            and (criterion == "overall" or item["criterion"] == criterion)
        ]
        candidate_reports[criterion] = metrics(
            [item["target"] for item in selected],
            [item["prediction"] for item in selected],
            [item["mean_probability"] for item in selected],
        )
    result = {
        "schema_version": "qwen_visual_state_adapter_oof_v1",
        "scope": "training_candidates_video_level_oof",
        "foundation_model": str(args.model_path.resolve()),
        "adapter": "balanced_l2_linear_probe_on_frozen_mean_pooled_visual_tokens",
        "criteria": list(CRITERIA), "candidate_center_fraction": args.candidate_center_fraction,
        "candidate_count": len(candidates), "unique_frame_count": len(unique_keys),
        "unannotated_video_policy": "exclude None frames and candidates without >=7 known labels",
        "frame_reports_state0_vs_state2": frame_reports,
        "candidate_reports_majority_full_vs_nonfull": candidate_reports,
        "candidate_rows": candidate_rows,
    }
    (args.output_dir / "oof_evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "frame_reports": frame_reports, "candidate_reports": candidate_reports,
        "output": str((args.output_dir / 'oof_evaluation.json').resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
