#!/usr/bin/env python3
"""Run the frozen CholecT50 Rendezvous fact plugin on Cholec80 transfer frames."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cholect50_absolute_common import ROOT, atomic_write_json, sha256_file
from run_cholect50_rendezvous_fact_smoke import COMPONENT_NAMES, load_official_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/cholec80_nonoverlap_transfer_v1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/cholec80_nonoverlap_transfer_v1")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "external/rendezvous_official_weights/rendezvous_l8_cholect50_challenge_k0_batchnorm_lowres.pth",
    )
    parser.add_argument("--official-repo", type=Path, default=ROOT / "external/rendezvous_official")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (
        args.run_dir / "DEVELOPMENT_SCORES.npz",
        args.run_dir / "DEVELOPMENT_FACTS.jsonl",
        args.run_dir / "FACT_EXECUTION_RECEIPT.json",
    )
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)
    protocol = json.loads((args.run_dir / "TRANSFER_PROTOCOL.json").read_text())
    if protocol.get("phase_labels_accessed_by_this_preparation") is not False:
        raise RuntimeError("preparation was not label blind")
    allowed = set(protocol["evaluation_split"]["videos"])
    images = sorted((args.data_dir / "images").glob("video*/*.jpg"))
    observed = {path.parent.name for path in images}
    if not images or observed != allowed:
        raise RuntimeError(f"transfer image set mismatch: observed={sorted(observed)}, expected={sorted(allowed)}")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    device = torch.device(args.device)
    model = load_official_model(args.official_repo, args.checkpoint, device, use_layer_norm=False)
    scores = {component: [] for component in ("instrument", "verb", "target")}
    records = []
    started = time.monotonic()
    for start in range(0, len(images), args.batch_size):
        batch_paths = images[start : start + args.batch_size]
        batch = torch.stack([transform(Image.open(path).convert("RGB")) for path in batch_paths]).to(device)
        with torch.inference_mode():
            enc_i, enc_v, enc_t, _triplet_not_exported = model(batch)
            outputs_by_component = {
                "instrument": torch.sigmoid(enc_i[1]).cpu().numpy(),
                "verb": torch.sigmoid(enc_v[1]).cpu().numpy(),
                "target": torch.sigmoid(enc_t[1]).cpu().numpy(),
            }
        for row_index, path in enumerate(batch_paths):
            record = {
                "video": path.parent.name,
                "frame_id": int(path.stem),
                "image": str(path.resolve()),
                "facts": {},
            }
            for component, matrix in outputs_by_component.items():
                vector = matrix[row_index]
                scores[component].append(vector)
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
        if start % (args.batch_size * 20) == 0 or start + args.batch_size >= len(images):
            print(json.dumps({"facts_completed": min(start + args.batch_size, len(images)), "total": len(images)}), flush=True)

    scores_path, facts_path, receipt_path = outputs
    temporary_facts = facts_path.with_suffix(".jsonl.tmp")
    with temporary_facts.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_facts.replace(facts_path)
    with scores_path.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(
            handle,
            video=np.asarray([row["video"] for row in records]),
            frame_id=np.asarray([row["frame_id"] for row in records]),
            **{component: np.stack(values) for component, values in scores.items()},
        )
    scores_path.with_suffix(".npz.tmp").replace(scores_path)
    receipt = {
        "status": "cholec80_transfer_fact_cache_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "frames": len(records),
        "videos": sorted(allowed),
        "phase_annotations_accessed": False,
        "cvs_annotations_accessed": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "phase_output_exported": False,
        "triplet_output_exported": False,
        "component_outputs_exported": ["instrument", "verb", "target"],
        "scores_sha256": sha256_file(scores_path),
        "facts_sha256": sha256_file(facts_path),
        "script_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
