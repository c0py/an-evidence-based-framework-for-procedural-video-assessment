#!/usr/bin/env python3
"""Fine-tune a generic object localizer on leakage-controlled Endoscapes boxes."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("external/datasets/Endoscapes2023/yolo_bbox_clean/dataset.yaml"),
    )
    parser.add_argument("--initial-weights", default="yolo11m.pt")
    parser.add_argument("--device", default="5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="endoscapes_yolo11m_clean_seed17")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.initial_weights)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path("runs/localizers").resolve()),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        cos_lr=True,
        close_mosaic=10,
        patience=25,
        seed=17,
        deterministic=True,
        plots=True,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
