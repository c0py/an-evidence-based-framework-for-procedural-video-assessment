"""Train ordinal CVS evidence heads on phase-balanced frozen PeskaVLP features."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvs_assessment.annotations import load_cvs_intervals, load_phase_starts
from cvs_assessment.models import OrdinalCriterionHead, PeskaVLPVisualEncoder
import train_peskavlp_cvs_head as base

CRITERIA = base.CRITERIA


def states_at(intervals, time_s: float) -> tuple[int, int, int]:
    return tuple(
        max((value for start, end, value in intervals[key] if start <= time_s <= end), default=0)
        for key in CRITERIA
    )


def sample_evenly(values: list[float], limit: int, rng: np.random.Generator) -> list[float]:
    if len(values) <= limit:
        return values
    indices = np.sort(rng.choice(len(values), size=limit, replace=False))
    return [values[index] for index in indices]


def phase_times(
    annotation_xlsx: str, phase_root: Path, video_id: int, strategy: str, sample_every_s: float,
    hard_negative_margin_s: float, negative_ratio: float, max_positive_per_category: int, seed: int,
    minimum_negative_per_video: int = 0, maximum_samples_per_video: int = 0,
) -> tuple[list[float], dict[str, int]]:
    phases = load_phase_starts(phase_root / f"video{video_id:02d}-phase.txt")
    start, end = phases["CalotTriangleDissection"], phases["ClippingCutting"]
    intervals = load_cvs_intervals(annotation_xlsx, video_id)
    if strategy == "phase_uniform":
        times = np.arange(start, end, sample_every_s).tolist()
        return times, {"uniform": len(times)}

    grid = np.arange(start, end, 1.0).tolist()
    full, partial, negative = [], [], []
    positive_intervals = [
        (interval_start, interval_end)
        for criterion in CRITERIA for interval_start, interval_end, value in intervals[criterion] if value >= 1
    ]
    for time_s in grid:
        states = states_at(intervals, time_s)
        if 2 in states:
            full.append(time_s)
        elif 1 in states:
            partial.append(time_s)
        else:
            negative.append(time_s)
    hard_negative = [
        time_s for time_s in negative
        if any(start_i - hard_negative_margin_s <= time_s <= end_i + hard_negative_margin_s
               for start_i, end_i in positive_intervals)
    ]
    hard_set = set(hard_negative)
    normal_negative = [time_s for time_s in negative if time_s not in hard_set]
    rng = np.random.default_rng(seed + video_id)
    full = sample_evenly(full, max_positive_per_category, rng)
    partial = sample_evenly(partial, max_positive_per_category, rng)
    positive_count = len(full) + len(partial)
    negative_target = max(
        int(round(negative_ratio * positive_count)), minimum_negative_per_video,
    )
    if maximum_samples_per_video > 0:
        negative_target = min(
            negative_target,
            max(0, maximum_samples_per_video - positive_count),
        )
    hard_target = min(len(hard_negative), (negative_target + 1) // 2)
    hard_negative = sample_evenly(hard_negative, hard_target, rng)
    normal_target = min(len(normal_negative), negative_target - len(hard_negative))
    normal_negative = sample_evenly(normal_negative, normal_target, rng)
    selected = sorted(full + partial + hard_negative + normal_negative)
    audit = {
        "full": len(full), "partial": len(partial), "hard_negative": len(hard_negative),
        "normal_negative": len(normal_negative), "selected": len(selected),
        "candidate_window_s": round(end - start, 3),
        "minimum_negative_per_video": minimum_negative_per_video,
        "maximum_samples_per_video": maximum_samples_per_video,
    }
    return selected, audit


def decode_phase_samples(
    video_root: Path, phase_root: Path, annotation_xlsx: str, video_ids: list[int], strategy: str,
    sample_every_s: float, hard_negative_margin_s: float, negative_ratio: float,
    max_positive_per_category: int, seed: int, allow_empty: bool = False,
    minimum_negative_per_video: int = 0, maximum_samples_per_video: int = 0,
):
    images, labels, provenance = [], [], []
    sampling_audit = {}
    for video_id in video_ids:
        video_path = video_root / f"video{video_id:02d}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        times, audit = phase_times(
            annotation_xlsx, phase_root, video_id, strategy, sample_every_s,
            hard_negative_margin_s, negative_ratio, max_positive_per_category, seed,
            minimum_negative_per_video, maximum_samples_per_video,
        )
        sampling_audit[str(video_id)] = audit
        print(f"decode_video={video_id:02d} strategy={strategy} audit={audit}", flush=True)
        intervals = load_cvs_intervals(annotation_xlsx, video_id)
        cap = cv2.VideoCapture(str(video_path))
        for time_s in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)
            top, left = (360 - 224) // 2, (640 - 224) // 2
            images.append(torch.from_numpy(rgb[top:top + 224, left:left + 224].copy()).permute(2, 0, 1))
            labels.append(torch.tensor(states_at(intervals, time_s), dtype=torch.float32) / 2.0)
            provenance.append((video_id, float(time_s)))
        cap.release()
        print(f"decoded_video={video_id:02d} cumulative_samples={len(images)}", flush=True)
    if not images and allow_empty:
        return (
            torch.empty((0, 3, 224, 224), dtype=torch.uint8),
            torch.empty((0, len(CRITERIA)), dtype=torch.float32),
            provenance,
            sampling_audit,
        )
    if not images:
        raise RuntimeError("No phase samples decoded")
    return torch.stack(images), torch.stack(labels), provenance, sampling_audit


def cached_features_for_videos(
    encoder: PeskaVLPVisualEncoder,
    video_root: Path,
    phase_root: Path,
    annotation_xlsx: str,
    video_ids: list[int],
    strategy: str,
    sample_every_s: float,
    hard_negative_margin_s: float,
    negative_ratio: float,
    max_positive_per_category: int,
    seed: int,
    batch_size: int,
    device: str,
    cache_dir: Path,
    source_checkpoint_sha256: str,
    minimum_negative_per_video: int = 0,
    maximum_samples_per_video: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, float]], dict]:
    """Decode and cache frozen encoder features one video at a time."""
    annotation_path = Path(annotation_xlsx).resolve()
    annotation_sha256 = base.sha256(annotation_path)
    strategy_dir = cache_dir / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    feature_chunks, target_chunks = [], []
    provenance: list[tuple[int, float]] = []
    sampling_audit = {}

    for video_id in video_ids:
        video_path = (video_root / f"video{video_id:02d}.mp4").resolve()
        phase_path = (phase_root / f"video{video_id:02d}-phase.txt").resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        if not phase_path.exists():
            raise FileNotFoundError(phase_path)
        video_stat = video_path.stat()
        signature = {
            "cache_schema_version": 1,
            "video_id": video_id,
            "video_path": str(video_path),
            "video_size_bytes": video_stat.st_size,
            "video_mtime_ns": video_stat.st_mtime_ns,
            "phase_annotation_sha256": base.sha256(phase_path),
            "cvs_annotation_sha256": annotation_sha256,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "encoder": "PeskaVLP_ResNet50_visual_768d",
            "strategy": strategy,
            "sample_every_s": sample_every_s,
            "hard_negative_margin_s": hard_negative_margin_s,
            "negative_ratio": negative_ratio,
            "max_positive_per_category": max_positive_per_category,
            "seed": seed,
        }
        if strategy == "phase_balanced" and (
            minimum_negative_per_video or maximum_samples_per_video
        ):
            signature.update({
                "cache_schema_version": 2,
                "minimum_negative_per_video": minimum_negative_per_video,
                "maximum_samples_per_video": maximum_samples_per_video,
            })
        cache_path = strategy_dir / f"video{video_id:02d}.pt"
        payload = None
        if cache_path.exists():
            candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
            if candidate.get("signature") == signature:
                payload = candidate
                print(
                    f"feature_cache_hit video={video_id:02d} strategy={strategy} "
                    f"samples={len(payload['targets'])}",
                    flush=True,
                )
            else:
                print(f"feature_cache_stale video={video_id:02d} strategy={strategy}", flush=True)

        if payload is None:
            images, targets, video_provenance, video_audit = decode_phase_samples(
                video_root, phase_root, annotation_xlsx, [video_id], strategy,
                sample_every_s, hard_negative_margin_s, negative_ratio,
                max_positive_per_category, seed, allow_empty=True,
                minimum_negative_per_video=minimum_negative_per_video,
                maximum_samples_per_video=maximum_samples_per_video,
            )
            if len(images):
                features = base.extract_features(encoder, images, batch_size, device)
            else:
                features = torch.empty((0, encoder.output_dim), dtype=torch.float32)
            payload = {
                "signature": signature,
                "features": features,
                "targets": targets,
                "provenance": video_provenance,
                "sampling_audit": video_audit[str(video_id)],
            }
            temporary_path = cache_path.with_suffix(".tmp")
            torch.save(payload, temporary_path)
            temporary_path.replace(cache_path)
            print(
                f"feature_cache_saved video={video_id:02d} strategy={strategy} "
                f"samples={len(targets)} path={cache_path}",
                flush=True,
            )
            del images

        feature_chunks.append(payload["features"])
        target_chunks.append(payload["targets"])
        provenance.extend(payload["provenance"])
        sampling_audit[str(video_id)] = payload["sampling_audit"]

    features = torch.cat(feature_chunks)
    targets = torch.cat(target_chunks)
    if not len(features):
        raise RuntimeError(f"No phase samples available for strategy={strategy}")
    return features, targets, provenance, sampling_audit


def ordinal_logits(head, features: torch.Tensor) -> dict[str, torch.Tensor]:
    output = head(features)
    return {
        definition: torch.stack([output[key][definition] for key in CRITERIA], dim=1)
        for definition in ("support_or_full", "full_only")
    }


@torch.inference_mode()
def predict(head, features: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    head.eval()
    logits = ordinal_logits(head, features.to(device))
    result = {key: torch.sigmoid(value).cpu() for key, value in logits.items()}
    head.train()
    return result


def truths(targets: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"support_or_full": targets >= 0.5, "full_only": targets >= 0.999}


def calibrate(scores: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, dict[str, float]]:
    target_bundle = truths(targets)
    output = {}
    for definition in scores:
        output[definition] = {}
        for index, criterion in enumerate(CRITERIA):
            criterion_targets = target_bundle[definition][:, index]
            # F1 threshold search is undefined when validation contains only
            # one class.  Returning the first grid value (0.05) would turn
            # almost every frame positive, so keep the model's neutral logit
            # threshold and record the missing-class limitation in metadata.
            if criterion_targets.all() or not criterion_targets.any():
                output[definition][criterion] = 0.5
                continue
            best_threshold, best_f1 = 0.5, -1.0
            for threshold_tensor in torch.linspace(0.05, 0.95, 91):
                threshold = float(threshold_tensor)
                f1 = base.binary_metrics(scores[definition][:, index], criterion_targets, threshold)["f1"]
                if f1 > best_f1:
                    best_threshold, best_f1 = threshold, f1
            output[definition][criterion] = best_threshold
    return output


def evaluate(
    scores: dict[str, torch.Tensor], targets: torch.Tensor, thresholds: dict[str, dict[str, float]],
) -> dict:
    target_bundle = truths(targets)
    result = {}
    for index, criterion in enumerate(CRITERIA):
        result[criterion] = {}
        for definition in scores:
            result[criterion][definition] = base.binary_metrics(
                scores[definition][:, index], target_bundle[definition][:, index], thresholds[definition][criterion],
            )
            result[criterion][definition]["brier"] = float(torch.mean(
                (scores[definition][:, index] - target_bundle[definition][:, index].float()) ** 2
            ))
    return result


def macro_auprc(metrics: dict) -> float:
    values = [
        metrics[criterion][definition]["auprc"]
        for criterion in CRITERIA for definition in ("support_or_full", "full_only")
        if metrics[criterion][definition]["auprc"] is not None
    ]
    return sum(values) / max(1, len(values))


def per_video_metrics(scores, targets, provenance, thresholds):
    output = {}
    for video_id in sorted({video_id for video_id, _ in provenance}):
        indices = [i for i, (sample_video_id, _) in enumerate(provenance) if sample_video_id == video_id]
        output[str(video_id)] = evaluate(
            {key: value[indices] for key, value in scores.items()}, targets[indices], thresholds,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peskavlp-checkpoint", required=True)
    parser.add_argument("--annotation-xlsx", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--phase-annotation-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--feature-cache-dir",
        help="Optional directory for restart-safe, per-video frozen PeskaVLP feature caches.",
    )
    parser.add_argument("--evaluation-sample-every-s", type=float, default=5.0)
    parser.add_argument("--hard-negative-margin-s", type=float, default=30.0)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--max-positive-per-category", type=int, default=400)
    parser.add_argument("--minimum-negative-per-video", type=int, default=0)
    parser.add_argument("--maximum-samples-per-video", type=int, default=0)
    parser.add_argument(
        "--positive-weight-cap", type=float, default=30.0,
        help="Cap for BCE positive weights; use 1 for unweighted BCE.",
    )
    parser.add_argument(
        "--video-balanced-sampling", action="store_true",
        help="Give each non-empty training video equal expected sampling mass per epoch.",
    )
    parser.add_argument(
        "--ordinal-parameterization",
        choices=("independent", "conditional_product"), default="independent",
        help="conditional_product guarantees P(full) <= P(support).",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--skip-test-evaluation", action="store_true",
        help="Keep the frozen test split sealed while selecting the model on validation.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split_path = Path(args.split_json).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = [int(value) for value in split["train_video_ids"]]
    validation_ids = [int(value) for value in split["validation_video_ids"]]
    test_ids = [int(value) for value in split["test_video_ids"]]
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
        raise ValueError("Split contains overlapping video IDs")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_root, phase_root = Path(args.video_root), Path(args.phase_annotation_root)

    encoder = PeskaVLPVisualEncoder()
    source_checkpoint = Path(args.peskavlp_checkpoint).resolve()
    source_checkpoint_sha256 = base.sha256(source_checkpoint)
    load_info = encoder.load_official_checkpoint(source_checkpoint)
    if args.feature_cache_dir:
        feature_cache_dir = Path(args.feature_cache_dir).resolve()
        train_features, train_targets, train_provenance, sampling_audit = cached_features_for_videos(
            encoder, video_root, phase_root, args.annotation_xlsx, train_ids, "phase_balanced",
            args.evaluation_sample_every_s, args.hard_negative_margin_s, args.negative_ratio,
            args.max_positive_per_category, args.seed, args.batch_size, device,
            feature_cache_dir, source_checkpoint_sha256,
            args.minimum_negative_per_video, args.maximum_samples_per_video,
        )
    else:
        feature_cache_dir = None
        train_images, train_targets, train_provenance, sampling_audit = decode_phase_samples(
            video_root, phase_root, args.annotation_xlsx, train_ids, "phase_balanced",
            args.evaluation_sample_every_s, args.hard_negative_margin_s, args.negative_ratio,
            args.max_positive_per_category, args.seed,
            minimum_negative_per_video=args.minimum_negative_per_video,
            maximum_samples_per_video=args.maximum_samples_per_video,
        )
        train_features = base.extract_features(encoder, train_images, args.batch_size, device)
        del train_images
    print(f"train_features={tuple(train_features.shape)}", flush=True)

    if feature_cache_dir:
        validation_features, validation_targets, validation_provenance, validation_audit = cached_features_for_videos(
            encoder, video_root, phase_root, args.annotation_xlsx, validation_ids, "phase_uniform",
            args.evaluation_sample_every_s, args.hard_negative_margin_s, args.negative_ratio,
            args.max_positive_per_category, args.seed, args.batch_size, device,
            feature_cache_dir, source_checkpoint_sha256,
        )
    else:
        validation_images, validation_targets, validation_provenance, validation_audit = decode_phase_samples(
            video_root, phase_root, args.annotation_xlsx, validation_ids, "phase_uniform",
            args.evaluation_sample_every_s, args.hard_negative_margin_s, args.negative_ratio,
            args.max_positive_per_category, args.seed,
        )
        validation_features = base.extract_features(encoder, validation_images, args.batch_size, device)
        del validation_images
    print(f"validation_features={tuple(validation_features.shape)}", flush=True)

    monotonic = args.ordinal_parameterization == "conditional_product"
    head = OrdinalCriterionHead(
        feature_dim=encoder.output_dim, monotonic=monotonic,
    ).to(device).train()
    train_truth = truths(train_targets)
    support_positive = train_truth["support_or_full"].sum(0).float()
    full_positive = train_truth["full_only"].sum(0).float()
    if args.positive_weight_cap < 1:
        raise ValueError("--positive-weight-cap must be at least 1")
    support_weight = ((len(train_targets) - support_positive) / support_positive.clamp_min(1)).clamp(
        min=1, max=args.positive_weight_cap,
    ).to(device)
    full_weight = ((len(train_targets) - full_positive) / full_positive.clamp_min(1)).clamp(
        min=1, max=args.positive_weight_cap,
    ).to(device)
    support_loss = torch.nn.BCEWithLogitsLoss(pos_weight=support_weight)
    full_loss = torch.nn.BCEWithLogitsLoss(pos_weight=full_weight)
    dataset = TensorDataset(train_features, train_targets)
    if args.video_balanced_sampling:
        video_counts = Counter(video_id for video_id, _ in train_provenance)
        sample_weights = torch.tensor(
            [1.0 / video_counts[video_id] for video_id, _ in train_provenance],
            dtype=torch.double,
        )
        sampler_generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True,
            generator=sampler_generator,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        video_counts = Counter(video_id for video_id, _ in train_provenance)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    default_thresholds = {
        definition: {criterion: 0.5 for criterion in CRITERIA}
        for definition in ("support_or_full", "full_only")
    }

    history, best_state = [], None
    best_epoch, best_value, stale = 0, float("-inf"), 0
    for epoch in range(args.epochs):
        total = 0.0
        for features, soft_targets in loader:
            features, soft_targets = features.to(device), soft_targets.to(device)
            logits = ordinal_logits(head, features)
            support_target = (soft_targets >= 0.5).float()
            full_target = (soft_targets >= 0.999).float()
            consistency = torch.relu(torch.sigmoid(logits["full_only"]) - torch.sigmoid(logits["support_or_full"])).mean()
            loss = support_loss(logits["support_or_full"], support_target) + full_loss(logits["full_only"], full_target)
            loss = loss + args.consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(features)
        val_scores = predict(head, validation_features, device)
        val_metrics = evaluate(val_scores, validation_targets, default_thresholds)
        selection = macro_auprc(val_metrics)
        mean_loss = total / len(train_targets)
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss, "validation_macro_ordinal_auprc": selection})
        if selection > best_value + args.min_delta:
            best_epoch, best_value, stale = epoch + 1, selection, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
        print(f"epoch={epoch+1} loss={mean_loss:.5f} selection={selection:.5f} best_epoch={best_epoch}", flush=True)
        if stale >= args.patience:
            print(f"early_stopping_epoch={epoch+1}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("No best ordinal head selected")
    head.load_state_dict(best_state)
    validation_scores = predict(head, validation_features, device)
    thresholds = calibrate(validation_scores, validation_targets)
    validation_metrics = evaluate(validation_scores, validation_targets, thresholds)
    validation_calibration_support = {
        definition: {
            criterion: {
                "n_positive": int(truths(validation_targets)[definition][:, index].sum()),
                "n_negative": int((~truths(validation_targets)[definition][:, index]).sum()),
                "threshold_source": (
                    "validation_f1"
                    if truths(validation_targets)[definition][:, index].any()
                    and (~truths(validation_targets)[definition][:, index]).any()
                    else "neutral_0.5_single_class_fallback"
                ),
            }
            for index, criterion in enumerate(CRITERIA)
        }
        for definition in ("support_or_full", "full_only")
    }

    test_audit = None
    test_metrics = None
    test_per_video = None
    if args.skip_test_evaluation:
        print("model_selection_complete_held_out_test_remains_sealed=true", flush=True)
    else:
        print("model_selection_complete_now_loading_held_out_test=true", flush=True)
        test_images, test_targets, test_provenance, test_audit = decode_phase_samples(
            video_root, phase_root, args.annotation_xlsx, test_ids, "phase_uniform",
            args.evaluation_sample_every_s, args.hard_negative_margin_s, args.negative_ratio,
            args.max_positive_per_category, args.seed,
        )
        test_features = base.extract_features(encoder, test_images, args.batch_size, device)
        del test_images
        test_scores = predict(head, test_features, device)
        test_metrics = evaluate(test_scores, test_targets, thresholds)
        test_per_video = per_video_metrics(
            test_scores, test_targets, test_provenance, thresholds,
        )
    metadata = {
        "tool_type": "peskavlp_frozen_encoder_ordinal_cvs_head",
        "head_type": "ordinal",
        "checkpoint_version": "0.4" if (
            monotonic or args.video_balanced_sampling
            or args.positive_weight_cap != 30.0
            or args.minimum_negative_per_video
            or args.maximum_samples_per_video
        ) else "0.3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "criteria": CRITERIA,
        "encoder": "PeskaVLP ResNet-50 visual tower + 768-D projection (frozen)",
        "split_manifest_path": str(split_path), "split_manifest": split,
        "pilot_only": bool(split.get("pilot_only", False)),
        "train_video_ids": train_ids, "validation_video_ids": validation_ids, "test_video_ids": test_ids,
        "development_only": False,
        "sampling_strategy": "candidate_phase_balanced",
        "sampling_audit": sampling_audit,
        "validation_sampling_audit": validation_audit,
        "test_sampling_audit": test_audit,
        "hard_negative_margin_s": args.hard_negative_margin_s,
        "negative_ratio": args.negative_ratio,
        "max_positive_per_category": args.max_positive_per_category,
        "minimum_negative_per_video": args.minimum_negative_per_video,
        "maximum_samples_per_video": args.maximum_samples_per_video,
        "positive_weight_cap": args.positive_weight_cap,
        "video_balanced_sampling": args.video_balanced_sampling,
        "training_samples_per_video": dict(sorted(video_counts.items())),
        "ordinal_parameterization": args.ordinal_parameterization,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_load_info": load_info,
        "feature_cache_dir": str(feature_cache_dir) if feature_cache_dir else None,
        "best_epoch": best_epoch, "best_selection_value": best_value,
        "selection_metric": "validation_macro_ordinal_auprc",
        "history": history,
        "decision_thresholds_calibrated_on_validation": thresholds["full_only"],
        "ordinal_thresholds_calibrated_on_validation": thresholds,
        "validation_calibration_support": validation_calibration_support,
        "validation_metrics": validation_metrics,
        "validation_per_video_metrics": per_video_metrics(validation_scores, validation_targets, validation_provenance, thresholds),
        "test_metrics": test_metrics,
        "test_per_video_metrics": test_per_video,
        "test_evaluation_skipped": args.skip_test_evaluation,
        "training_arguments": {
            "epochs": args.epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "batch_size": args.batch_size,
            "evaluation_sample_every_s": args.evaluation_sample_every_s,
            "hard_negative_margin_s": args.hard_negative_margin_s,
            "negative_ratio": args.negative_ratio,
            "max_positive_per_category": args.max_positive_per_category,
            "minimum_negative_per_video": args.minimum_negative_per_video,
            "maximum_samples_per_video": args.maximum_samples_per_video,
            "positive_weight_cap": args.positive_weight_cap,
            "video_balanced_sampling": args.video_balanced_sampling,
            "ordinal_parameterization": args.ordinal_parameterization,
            "learning_rate": args.learning_rate,
            "consistency_weight": args.consistency_weight,
            "seed": args.seed,
            "test_access_policy": (
                "sealed" if args.skip_test_evaluation
                else "evaluated_after_model_selection"
            ),
        },
        "ordinal_consistency_weight": args.consistency_weight,
        "positive_weights": {
            "support_or_full": {criterion: float(support_weight[i]) for i, criterion in enumerate(CRITERIA)},
            "full_only": {criterion: float(full_weight[i]) for i, criterion in enumerate(CRITERIA)},
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": encoder.cpu().state_dict(), "head_state": head.cpu().state_dict(), "metadata": metadata}, output)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"checkpoint={output} best_epoch={best_epoch}", flush=True)
    print(f"thresholds={thresholds}", flush=True)
    print(f"held_out_test_metrics={test_metrics}", flush=True)


if __name__ == "__main__":
    main()
