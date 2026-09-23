#!/usr/bin/env python3
"""Run the frozen MoCo visual trainer with the official full-frame transform."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_endoscapes_moco_visual as base


class FullFrameEndoscapesFrames(base.EndoscapesFrames):
    """Preserve the complete 854x480 surgical field as in the official code."""

    def __init__(self, data: base.TrainData, videos: set[int], training: bool) -> None:
        self.data = data
        self.rows = [
            (video, frame)
            for video in sorted(videos)
            for frame in range(len(data.cache[video]["frame_indices"]))
        ]
        transforms: list[nn.Module] = [v2.Resize((480, 854), antialias=True)]
        if training:
            transforms.extend([
                v2.RandomHorizontalFlip(0.5),
                v2.ColorJitter(saturation=(0.6, 1.4)),
            ])
        transforms.extend([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        self.transform = v2.Compose(transforms)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", type=Path, required=True)
    known, _ = parser.parse_known_args()
    protocol = json.loads(known.protocol.read_text(encoding="utf-8"))
    if protocol.get("runner_code_sha256") != base.sha256_file(Path(__file__)):
        raise ValueError("Full-frame runner code differs from frozen protocol")
    if protocol.get("preprocessing", {}).get("mode") != "official_full_frame_480x854":
        raise ValueError("Protocol does not authorize the full-frame runner")
    base.EndoscapesFrames = FullFrameEndoscapesFrames
    base.main()


if __name__ == "__main__":
    main()
