#!/usr/bin/env python3
"""Train fixed-hyperparameter subject-out JIGSAWS gesture TCN folds."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvs_assessment.tasks.jigsaws_gesture_model import (  # noqa: E402
    GESTURE_CLASSES,
    GestureSequence,
    JigsawsGestureTCN,
    class_weights,
    discover_suturing_sequences,
    stable_class_filter,
    training_normalization,
)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20_260_821
    downsample: int = 3
    source_fps: float = 30.0
    hidden_dim: int = 96
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = 0.10
    chunk_steps: int = 512
    chunk_stride: int = 256
    batch_size: int = 16
    epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    gradient_clip: float = 5.0
    majority_filter_width: int = 7


class ChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        sequences: list[GestureSequence],
        mean: np.ndarray,
        standard_deviation: np.ndarray,
        chunk_steps: int,
        chunk_stride: int,
    ) -> None:
        self.sequences = sequences
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.chunk_steps = chunk_steps
        self.items: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(sequences):
            starts = list(range(0, max(1, len(sequence.labels) - chunk_steps + 1), chunk_stride))
            last = max(0, len(sequence.labels) - chunk_steps)
            if not starts or starts[-1] != last:
                starts.append(last)
            self.items.extend((sequence_index, start) for start in sorted(set(starts)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_index, start = self.items[index]
        sequence = self.sequences[sequence_index]
        end = min(len(sequence.labels), start + self.chunk_steps)
        length = end - start
        features = np.zeros((self.chunk_steps, sequence.features.shape[1]), dtype=np.float32)
        labels = np.full(self.chunk_steps, -100, dtype=np.int64)
        features[:length] = (
            sequence.features[start:end] - self.mean
        ) / self.standard_deviation
        labels[:length] = sequence.labels[start:end]
        return torch.from_numpy(features), torch.from_numpy(labels)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collapse(labels: np.ndarray) -> list[int]:
    output = []
    for value in np.asarray(labels, dtype=np.int64):
        if int(value) == 0:
            continue
        if not output or output[-1] != int(value):
            output.append(int(value))
    return output


def edit_score(truth: np.ndarray, predicted: np.ndarray) -> float:
    left, right = collapse(truth), collapse(predicted)
    matrix = np.zeros((len(left) + 1, len(right) + 1), dtype=np.int32)
    matrix[:, 0] = np.arange(len(left) + 1)
    matrix[0, :] = np.arange(len(right) + 1)
    for i, left_value in enumerate(left, start=1):
        for j, right_value in enumerate(right, start=1):
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,
                matrix[i, j - 1] + 1,
                matrix[i - 1, j - 1] + int(left_value != right_value),
            )
    denominator = max(len(left), len(right))
    return 1.0 - float(matrix[-1, -1]) / denominator if denominator else 1.0


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    values = np.asarray(labels, dtype=np.int64)
    output: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            if int(values[start]) != 0:
                output.append((int(values[start]), start, index))
            start = index
    return output


def segment_f1(truth: np.ndarray, predicted: np.ndarray, threshold: float) -> float:
    true_segments, predicted_segments = segments(truth), segments(predicted)
    candidates = []
    for p_index, (p_class, p_start, p_end) in enumerate(predicted_segments):
        for t_index, (t_class, t_start, t_end) in enumerate(true_segments):
            if p_class != t_class:
                continue
            intersection = max(0, min(p_end, t_end) - max(p_start, t_start))
            union = max(p_end, t_end) - min(p_start, t_start)
            candidates.append((intersection / union if union else 0.0, p_index, t_index))
    used_p, used_t, matches = set(), set(), 0
    for iou, p_index, t_index in sorted(candidates, reverse=True):
        if iou < threshold:
            break
        if p_index in used_p or t_index in used_t:
            continue
        used_p.add(p_index)
        used_t.add(t_index)
        matches += 1
    precision = matches / len(predicted_segments) if predicted_segments else 0.0
    recall = matches / len(true_segments) if true_segments else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    by_class = {}
    f1_values = []
    for index, class_name in enumerate(GESTURE_CLASSES):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        f1 = (
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
            if 2 * true_positive + false_positive + false_negative else None
        )
        if f1 is not None:
            f1_values.append(f1)
        by_class[class_name] = {
            "support": int(confusion[index, :].sum()),
            "f1": f1,
        }
    return {
        "frame_accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_frame_f1": float(np.mean(f1_values)),
        "by_class": by_class,
        "confusion": confusion.tolist(),
    }


@torch.no_grad()
def predict_sequence(
    model: nn.Module,
    sequence: GestureSequence,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    device: torch.device,
    filter_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features = torch.from_numpy(
        ((sequence.features - mean) / standard_deviation)[None]
    ).to(device)
    probabilities = torch.softmax(model(features)[0], dim=-1).cpu().numpy()
    predicted = stable_class_filter(probabilities.argmax(axis=-1), width=filter_width)
    return predicted, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "jigsaws_gesture_tcn_loso_protocol_v1":
        raise ValueError("Unexpected JIGSAWS gesture protocol")
    if protocol["safety"].get("expert_GRS_read_by_training_code") is not False:
        raise PermissionError("Unsafe JIGSAWS gesture protocol")
    config = TrainingConfig(**{
        key: tuple(value) if key == "dilations" else value
        for key, value in protocol["training"].items()
    })
    set_seed(config.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    sequences = discover_suturing_sequences(
        Path(protocol["dataset_root"]), downsample=config.downsample,
    )
    expected_ids = protocol["trial_ids"]
    if [row.trial_id for row in sequences] != expected_ids:
        raise ValueError("JIGSAWS discovered trial order differs from the frozen protocol")
    subjects = protocol["subjects"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_confusion = np.zeros((len(GESTURE_CLASSES), len(GESTURE_CLASSES)), dtype=np.int64)
    trial_metrics, folds = {}, []
    all_edits, all_f1_10, all_f1_25, all_f1_50 = [], [], [], []
    for fold_index, subject in enumerate(subjects):
        fold_dir = args.output_dir / f"subject_{subject}_out"
        fold_result = fold_dir / "FOLD_RESULT.json"
        if args.resume and fold_result.is_file():
            result = json.loads(fold_result.read_text(encoding="utf-8"))
            folds.append(result)
            for trial_id, row in result["trial_metrics"].items():
                trial_metrics[trial_id] = row
                all_confusion += np.asarray(row["confusion"], dtype=np.int64)
                all_edits.append(row["edit_score"])
                all_f1_10.append(row["segment_f1_at_0.10"])
                all_f1_25.append(row["segment_f1_at_0.25"])
                all_f1_50.append(row["segment_f1_at_0.50"])
            continue
        train = [row for row in sequences if row.subject_id != subject]
        test = [row for row in sequences if row.subject_id == subject]
        if not train or not test or {row.subject_id for row in train}.intersection({subject}):
            raise RuntimeError(f"Invalid JIGSAWS subject-out fold: {subject}")
        mean, deviation = training_normalization(train)
        weights = class_weights(train)
        dataset = ChunkDataset(
            train, mean, deviation, config.chunk_steps, config.chunk_stride,
        )
        generator = torch.Generator().manual_seed(config.seed + fold_index)
        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=0, generator=generator,
        )
        set_seed(config.seed + fold_index)
        model = JigsawsGestureTCN(
            hidden_dim=config.hidden_dim,
            dilations=config.dilations,
            dropout=config.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(weights).to(device), ignore_index=-100,
        )
        history = []
        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss, batches = 0.0, 0
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            row = {"epoch": epoch, "training_loss": total_loss / max(1, batches)}
            history.append(row)
            print(json.dumps({
                "event": "epoch", "fold": subject,
                "fold_index": fold_index + 1, "fold_count": len(subjects), **row,
            }), flush=True)
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = fold_dir / "checkpoint.pt"
        torch.save({
            "schema_version": "jigsaws_gesture_tcn_fold_v1",
            "held_out_subject": subject,
            "model_state": model.state_dict(),
            "normalization_mean": mean,
            "normalization_standard_deviation": deviation,
            "gesture_classes": GESTURE_CLASSES,
            "training_config": asdict(config),
            "train_trial_ids": [row.trial_id for row in train],
            "test_trial_ids": [row.trial_id for row in test],
            "expert_GRS_used": False,
        }, checkpoint)
        fold_trials = {}
        for sequence in test:
            predicted, probabilities = predict_sequence(
                model, sequence, mean, deviation, device,
                config.majority_filter_width,
            )
            prediction_path = fold_dir / f"{sequence.trial_id}_prediction.npz"
            np.savez_compressed(
                prediction_path,
                predicted_class=predicted.astype(np.int16),
                class_probability=probabilities.astype(np.float16),
                source_frame_indices=sequence.source_frame_indices.astype(np.int32),
                gesture_classes=np.asarray(GESTURE_CLASSES),
            )
            confusion = np.zeros_like(all_confusion)
            np.add.at(confusion, (sequence.labels, predicted), 1)
            row = {
                "frame_accuracy": float(np.mean(predicted == sequence.labels)),
                "edit_score": edit_score(sequence.labels, predicted),
                "segment_f1_at_0.10": segment_f1(sequence.labels, predicted, 0.10),
                "segment_f1_at_0.25": segment_f1(sequence.labels, predicted, 0.25),
                "segment_f1_at_0.50": segment_f1(sequence.labels, predicted, 0.50),
                "confusion": confusion.tolist(),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
                "GRS_or_skill_level_in_prediction": False,
            }
            fold_trials[sequence.trial_id] = row
            trial_metrics[sequence.trial_id] = row
            all_confusion += confusion
            all_edits.append(row["edit_score"])
            all_f1_10.append(row["segment_f1_at_0.10"])
            all_f1_25.append(row["segment_f1_at_0.25"])
            all_f1_50.append(row["segment_f1_at_0.50"])
        fold_payload = {
            "schema_version": "jigsaws_gesture_tcn_fold_result_v1",
            "held_out_subject": subject,
            "train_subjects": sorted({row.subject_id for row in train}),
            "test_subjects": [subject],
            "train_trial_count": len(train),
            "test_trial_count": len(test),
            "checkpoint": {
                "path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint),
            },
            "history": history,
            "trial_metrics": fold_trials,
            "expert_GRS_used": False,
        }
        write_json(fold_result, fold_payload)
        folds.append(fold_payload)
    summary = {
        **confusion_metrics(all_confusion),
        "mean_edit_score": float(np.mean(all_edits)),
        "mean_segment_f1_at_0.10": float(np.mean(all_f1_10)),
        "mean_segment_f1_at_0.25": float(np.mean(all_f1_25)),
        "mean_segment_f1_at_0.50": float(np.mean(all_f1_50)),
    }
    gates = {
        "frame_accuracy_at_least_0.75": summary["frame_accuracy"] >= 0.75,
        "macro_frame_f1_at_least_0.60": summary["macro_frame_f1"] >= 0.60,
        "mean_edit_score_at_least_0.60": summary["mean_edit_score"] >= 0.60,
    }
    output = {
        "schema_version": "jigsaws_gesture_tcn_loso_result_v1",
        "protocol": {
            "path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol),
        },
        "training_config": asdict(config),
        "gesture_classes": GESTURE_CLASSES,
        "fold_count": len(subjects),
        "trial_count": len(sequences),
        "summary": summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "folds": [{
            "held_out_subject": row["held_out_subject"],
            "checkpoint": row["checkpoint"],
            "test_trial_count": row["test_trial_count"],
        } for row in folds],
        "trial_metrics": trial_metrics,
        "expert_GRS_used_for_training_or_selection": False,
        "foundation_model_parameters_updated": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "RESULT.json", output)
    print(json.dumps({
        "output": str((args.output_dir / "RESULT.json").resolve()),
        "summary": summary, "gates": gates,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

