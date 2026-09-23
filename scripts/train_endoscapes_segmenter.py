"""Train a leakage-controlled six-class Endoscapes YOLO segmenter."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11m-seg.pt")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="endoscapes_yolo11m_seg_clean_seed31")
    parser.add_argument("--device", default="5")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=str(args.data.resolve()), project=str(args.project.resolve()), name=args.name,
        device=args.device, epochs=args.epochs, imgsz=args.image_size, batch=args.batch_size,
        seed=args.seed, deterministic=True, patience=35, workers=8, cache="disk",
        optimizer="AdamW", lr0=5e-4, weight_decay=5e-4, cos_lr=True,
        degrees=4.0, translate=0.08, scale=0.20, fliplr=0.5,
        hsv_h=0.01, hsv_s=0.25, hsv_v=0.20, mosaic=0.25, mixup=0.0,
        plots=True, save=True, verbose=True,
    )


if __name__ == "__main__":
    main()
