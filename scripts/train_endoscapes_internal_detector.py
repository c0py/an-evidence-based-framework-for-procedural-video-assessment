#!/usr/bin/env python3
"""Train the frozen internal-train-only Endoscapes anatomy detector."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="train_v1")
    parser.add_argument("--device", default="5")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["training_code_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Training code differs from frozen protocol")
    manifest_path = Path(protocol["sources"]["dataset_manifest"]["path"])
    if sha256_file(manifest_path) != protocol["sources"]["dataset_manifest"]["sha256"]:
        raise ValueError("Dataset manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["development_video_ids_loaded"] or manifest["official_val_or_test_loaded"]:
        raise ValueError("Unsafe detector dataset")
    initial = Path(protocol["sources"]["initial_weights"]["path"])
    if sha256_file(initial) != protocol["sources"]["initial_weights"]["sha256"]:
        raise ValueError("Initial detector weights changed")
    cfg = protocol["training"]
    from ultralytics import YOLO
    model = YOLO(str(initial))
    model.train(
        data=protocol["sources"]["dataset_yaml"]["path"],
        epochs=int(cfg["epochs"]), imgsz=int(cfg["image_size"]),
        batch=int(cfg["batch_size"]), device=args.device, workers=int(cfg["workers"]),
        project=str(args.project.resolve()), name=args.name, exist_ok=False,
        pretrained=True, optimizer="AdamW", lr0=float(cfg["initial_learning_rate"]),
        weight_decay=float(cfg["weight_decay"]), cos_lr=True,
        close_mosaic=int(cfg["close_mosaic_epochs"]), patience=0,
        seed=int(cfg["seed"]), deterministic=True, val=False,
        plots=False, save=True, verbose=True,
    )
    run_dir = args.project.resolve() / args.name
    checkpoint = run_dir / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    audit = {
        "schema_version": "endoscapes_internal_detector_training_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "fixed_epoch": int(cfg["epochs"]), "checkpoint_selection": "last_epoch",
        "training_video_ids": manifest["observed_training_video_ids"],
        "internal_development_video_ids_used": [],
        "official_val_or_test_used": False,
        "LLM_or_MLLM_parameters_updated": False,
    }
    (run_dir / "TRAINING_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
