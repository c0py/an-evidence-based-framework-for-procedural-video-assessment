#!/usr/bin/env python3
"""Run strict SAGES nested OOF with group-wise frozen visual feature fusion."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cvs_assessment.detector_skill_cvs import (
    GroupedFeatureFusionProbe,
    SkillConditionedGroupedFeatureHead,
)
import run_sages_dinov2_nested_oof_outer_fold as engine
from train_direct_interval_nested_oof import sha256_file


def build(candidate: dict):
    family = candidate["family"]
    common = {
        "group_dims": tuple(map(int, candidate["group_dims"])),
        "branch_dim": int(candidate["branch_dim"]),
        "hidden_dim": int(candidate["hidden_dim"]),
        "dropout": float(candidate["dropout"]),
    }
    if family == "grouped_multilabel_probe":
        return GroupedFeatureFusionProbe(criterion_count=3, **common)
    if family == "grouped_skill_shared_frame":
        return SkillConditionedGroupedFeatureHead(text_dim=4096, **common)
    raise ValueError(f"Unknown grouped fusion family: {family}")


def protocol_path(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--protocol") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("--protocol is required") from error


def main() -> None:
    protocol = json.loads(protocol_path(sys.argv).read_text())
    engine_path = Path(engine.__file__).resolve()
    if protocol["engine_training_code_sha256"] != sha256_file(engine_path):
        raise ValueError("DINO nested OOF engine differs from frozen protocol")
    engine.base.build = build
    engine.__file__ = __file__
    engine.main()


if __name__ == "__main__":
    main()
