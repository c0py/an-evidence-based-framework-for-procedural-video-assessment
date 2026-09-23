#!/usr/bin/env python3
"""Train a text-conditioned Qwen vision adapter with video-grouped holdout.

The learned contract is task-neutral: arbitrary criterion text plus an image is
mapped to absent/partial/full evidence. Cholec80 criterion IDs are used only for
sampling and reporting, never as model inputs or criterion-specific branches.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.visual_adapters import (
    CriterionConditionedVisualStateHead,
    add_qwen_vision_lora,
    trainable_adapter_state,
)
from evaluate_qwen_visual_state_adapter_oof import decode_unique_frames, metrics


CRITERION_TEXT = {
    "two_structures": (
        "The target procedural criterion is fully satisfied only when exactly two "
        "tubular structures, the cystic duct and cystic artery, are visibly entering "
        "the gallbladder."
    ),
    "cystic_plate": (
        "The target procedural criterion is fully satisfied only when the lower "
        "gallbladder is separated from the liver bed and the cystic plate is clearly exposed."
    ),
    "hepatocystic_triangle": (
        "The target procedural criterion is fully satisfied only when fat and fibrous "
        "tissue are cleared from the hepatocystic triangle so its anatomy is visibly exposed."
    ),
}


def grouped_video_folds(records: list[dict[str, Any]], folds: int, seed: int) -> list[list[int]]:
    videos = sorted({int(item["video_id"]) for item in records})
    rng = random.Random(seed)
    tie_break = {video_id: rng.random() for video_id in videos}
    signatures = {}
    for video_id in videos:
        present = {
            (item["criterion"], int(item["expert_state"]))
            for item in records if int(item["video_id"]) == video_id
        }
        signatures[video_id] = np.asarray([
            int((criterion, state) in present)
            for criterion in CRITERION_TEXT for state in (1, 2)
        ], dtype=np.int64)
    videos.sort(key=lambda value: (-int(signatures[value].sum()), tie_break[value]))
    assignments: list[list[int]] = [[] for _ in range(folds)]
    counts = np.zeros((folds, len(next(iter(signatures.values())))), dtype=np.int64)
    capacity = int(np.ceil(len(videos) / folds))
    for video_id in videos:
        signature = signatures[video_id]
        candidates = [index for index in range(folds) if len(assignments[index]) < capacity]
        selected = min(
            candidates,
            key=lambda index: (float(np.dot(counts[index], signature)), len(assignments[index]), index),
        )
        assignments[selected].append(video_id)
        counts[selected] += signature
    return [sorted(values) for values in assignments]


def cached_images(
    dataset_root: Path, records: list[dict[str, Any]], cache_dir: Path,
) -> list[Image.Image]:
    """Decode every unique training frame once and reuse it across OOF folds."""
    keys = [(int(item["video_id"]), float(item["time_s"])) for item in records]
    unique_keys = sorted(set(keys))
    cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(key: tuple[int, float]) -> Path:
        video_id, time_s = key
        return cache_dir / f"video{video_id:02d}" / f"t{round(time_s * 1000):010d}.jpg"

    missing = [key for key in unique_keys if not path_for(key).is_file()]
    if missing:
        decoded = decode_unique_frames(dataset_root, missing)
        for key, image in zip(missing, decoded):
            path = path_for(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="JPEG", quality=95, subsampling=0)
    images_by_key = {}
    for key in unique_keys:
        with Image.open(path_for(key)) as image:
            images_by_key[key] = image.convert("RGB").copy()
    print(json.dumps({
        "frame_cache": str(cache_dir.resolve()), "unique_frames": len(unique_keys),
        "newly_decoded": len(missing),
    }), flush=True)
    return [images_by_key[key] for key in keys]


def encode_criterion_texts(
    foundation: Qwen3VLForConditionalGeneration, processor: AutoProcessor,
) -> dict[str, torch.Tensor]:
    embedding = foundation.get_input_embeddings()
    output = {}
    for criterion, description in CRITERION_TEXT.items():
        tokenized = processor.tokenizer(description, return_tensors="pt", add_special_tokens=True)
        ids = tokenized.input_ids.to(embedding.weight.device)
        with torch.inference_mode():
            output[criterion] = embedding(ids)[0].float().mean(0).cpu()
    return output


def visual_features(
    visual: nn.Module, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    pixel_values = pixel_values.to(next(visual.parameters()).device, dtype=visual.dtype)
    grid_thw = grid_thw.to(pixel_values.device)
    embeddings, _ = visual(pixel_values, grid_thw=grid_thw)
    sizes = (grid_thw.prod(-1) // int(visual.spatial_merge_size) ** 2).tolist()
    return torch.split(embeddings, sizes)


def balanced_indices(records: list[dict[str, Any]], count: int, seed: int) -> list[int]:
    groups = Counter((item["criterion"], int(item["expert_state"])) for item in records)
    weights = torch.tensor([
        1.0 / groups[(item["criterion"], int(item["expert_state"]))]
        for item in records
    ], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    return torch.multinomial(weights, count, replacement=True, generator=generator).tolist()


def batches(values: list[int], batch_size: int) -> list[list[int]]:
    return [values[start:start + batch_size] for start in range(0, len(values), batch_size)]


def forward_batch(
    visual: nn.Module, head: CriterionConditionedVisualStateHead,
    processor: AutoProcessor, images: list[Any], records: list[dict[str, Any]],
    indices: list[int], criterion_embeddings: dict[str, torch.Tensor], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_images = [images[index] for index in indices]
    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    inputs = processor(
        text=[prompt] * len(indices), images=current_images, padding=True,
        return_tensors="pt",
    )
    tokens = visual_features(visual, inputs.pixel_values, inputs.image_grid_thw)
    criteria = torch.stack([
        criterion_embeddings[records[index]["criterion"]] for index in indices
    ]).to(device)
    targets = torch.tensor([
        int(records[index]["expert_state"]) for index in indices
    ], dtype=torch.long, device=device)
    return head(tokens, criteria), targets


@torch.inference_mode()
def evaluate(
    visual: nn.Module, head: CriterionConditionedVisualStateHead,
    processor: AutoProcessor, images: list[Any], records: list[dict[str, Any]],
    indices: list[int], criterion_embeddings: dict[str, torch.Tensor],
    device: torch.device, batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visual.eval()
    head.eval()
    rows = []
    for group in batches(indices, batch_size):
        logits, targets = forward_batch(
            visual, head, processor, images, records, group, criterion_embeddings, device,
        )
        probabilities = torch.softmax(logits.float(), dim=-1).cpu()
        for index, target, probability in zip(group, targets.cpu().tolist(), probabilities.tolist()):
            rows.append({
                "reference_id": records[index]["reference_id"],
                "video_id": int(records[index]["video_id"]),
                "time_s": float(records[index]["time_s"]),
                "criterion": records[index]["criterion"],
                "expert_state": int(target),
                "state_probabilities": probability,
                "full_probability": float(probability[2]),
            })
    reports = {}
    for criterion in (*CRITERION_TEXT, "overall"):
        selected = [row for row in rows if criterion == "overall" or row["criterion"] == criterion]
        labels = [int(row["expert_state"] == 2) for row in selected]
        scores = [row["full_probability"] for row in selected]
        reports[criterion] = metrics(labels, [int(value >= 0.5) for value in scores], scores)
        reports[criterion]["three_state_accuracy"] = float(np.mean([
            int(np.argmax(row["state_probabilities"]) == row["expert_state"]) for row in selected
        ]))
    return reports, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lora-last-blocks", type=int, default=6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--head-learning-rate", type=float, default=4e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=8e-5)
    parser.add_argument("--initialize-head-checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-selection", choices=("best", "final"), default="best",
        help="Use final for confirmatory folds after fixing the epoch count on a pilot fold.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("scope") != "training_split_only" or manifest.get("test_labels_accessed"):
        raise ValueError("A training-only manifest is required")
    records = list(manifest["records"])
    fold_ids = grouped_video_folds(records, args.folds, args.seed)
    if not args.train_all and not 0 <= args.fold_index < len(fold_ids):
        raise ValueError("fold-index is out of range")
    heldout = set() if args.train_all else set(fold_ids[args.fold_index])
    train_indices = [index for index, item in enumerate(records) if int(item["video_id"]) not in heldout]
    valid_indices = (
        train_indices if args.train_all else
        [index for index, item in enumerate(records) if int(item["video_id"]) in heldout]
    )
    keys = [(int(item["video_id"]), float(item["time_s"])) for item in records]
    print(json.dumps({
        "fold": args.fold_index, "heldout_video_ids": sorted(heldout),
        "train_records": len(train_indices), "valid_records": len(valid_indices),
        "unique_frames": len(set(keys)), "test_accessed": False,
    }), flush=True)
    images = cached_images(args.dataset_root, records, args.frame_cache_dir)

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    foundation = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": device.index or 0}, low_cpu_mem_usage=True,
    )
    criterion_embeddings = encode_criterion_texts(foundation, processor)
    visual = foundation.model.visual
    visual.requires_grad_(False)
    del foundation
    gc.collect()
    torch.cuda.empty_cache()
    replaced = []
    if args.lora_last_blocks > 0:
        replaced = add_qwen_vision_lora(
            visual, last_blocks=args.lora_last_blocks, rank=args.lora_rank,
        )
    visual.to(device)
    head = CriterionConditionedVisualStateHead().to(device)
    if args.initialize_head_checkpoint is not None:
        initial = torch.load(
            args.initialize_head_checkpoint, map_location="cpu", weights_only=False,
        )
        head.load_state_dict(initial["head"])
        print(json.dumps({
            "initialized_head_from": str(args.initialize_head_checkpoint.resolve()),
            "source_best_epoch": initial.get("best_epoch"),
        }), flush=True)
    lora_parameters = [value for value in visual.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": args.head_learning_rate, "weight_decay": 1e-3},
        {"params": lora_parameters, "lr": args.lora_learning_rate, "weight_decay": 1e-4},
    ])
    steps = args.steps_per_epoch or len(train_indices)
    best_state, best_score, best_epoch = None, -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        visual.train()
        head.train()
        order = balanced_indices(records=[records[index] for index in train_indices], count=steps,
                                 seed=args.seed * 100 + epoch)
        order = [train_indices[index] for index in order]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        groups = batches(order, args.batch_size)
        for step, group in enumerate(groups, start=1):
            logits, targets = forward_batch(
                visual, head, processor, images, records, group, criterion_embeddings, device,
            )
            state_loss = nn.functional.cross_entropy(logits.float(), targets)
            full_logit = logits[:, 2].float() - torch.logsumexp(logits[:, :2].float(), dim=1)
            full_loss = nn.functional.binary_cross_entropy_with_logits(
                full_logit, (targets == 2).float(),
            )
            expected_state = (torch.softmax(logits.float(), dim=1) * torch.arange(
                3, device=device, dtype=torch.float32,
            )).sum(1)
            ordinal_loss = nn.functional.smooth_l1_loss(expected_state, targets.float())
            loss = (state_loss + 0.35 * full_loss + 0.15 * ordinal_loss) / args.gradient_accumulation
            loss.backward()
            losses.append(float(loss) * args.gradient_accumulation)
            if step % args.gradient_accumulation == 0 or step == len(groups):
                nn.utils.clip_grad_norm_([*head.parameters(), *lora_parameters], 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(
                    f"fold={args.fold_index} epoch={epoch} batch={step}/{len(groups)} "
                    f"loss={np.mean(losses[-50:]):.5f}", flush=True,
                )
        if args.train_all and epoch < args.epochs:
            history.append({
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                "selection": "fixed_epoch_protocol_no_intermediate_train_evaluation",
            })
            print(json.dumps(history[-1]), flush=True)
            continue
        reports, rows = evaluate(
            visual, head, processor, images, records, valid_indices,
            criterion_embeddings, device, args.batch_size,
        )
        aucs = [reports[name]["roc_auc"] for name in CRITERION_TEXT if reports[name]["roc_auc"] is not None]
        score = float(np.mean(aucs)) if aucs else 0.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "macro_criterion_auc": score, "reports": reports})
        print(json.dumps(history[-1]), flush=True)
        if args.checkpoint_selection == "final" or score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                "head": copy.deepcopy(head.state_dict()),
                "lora": copy.deepcopy(trainable_adapter_state(visual)),
                "rows": rows, "reports": reports,
            }
    assert best_state is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = "final" if args.train_all else f"fold{args.fold_index}"
    checkpoint_path = args.output_dir / f"{artifact_stem}_best.pt"
    torch.save({
        "schema_version": "criterion_conditioned_qwen_visual_adapter_v1",
        "foundation_model": str(args.model_path.resolve()),
        "criterion_interface": "free_text_embedding_to_shared_absent_partial_full_head",
        "criterion_texts": CRITERION_TEXT,
        "lora": best_state["lora"], "head": best_state["head"],
        "lora_last_blocks": args.lora_last_blocks, "lora_rank": args.lora_rank,
        "replaced_modules": replaced, "best_epoch": best_epoch,
        "heldout_video_ids": sorted(heldout), "test_accessed": False,
    }, checkpoint_path)
    result = {
        "schema_version": "criterion_conditioned_qwen_visual_adapter_fold_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()), "fold_index": args.fold_index,
        "folds": fold_ids, "heldout_video_ids": sorted(heldout),
        "best_epoch": best_epoch, "best_macro_criterion_auc": best_score,
        "reports": best_state["reports"], "predictions": best_state["rows"],
        "history": history, "checkpoint": str(checkpoint_path.resolve()),
        "test_video_or_annotation_accessed": False,
    }
    result_path = args.output_dir / f"{artifact_stem}_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(result_path.resolve()), "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch, "best_macro_criterion_auc": best_score,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
