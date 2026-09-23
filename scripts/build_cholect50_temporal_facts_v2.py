#!/usr/bin/env python3
"""Build conservative compound-event memories from calibrated atomic facts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from cholect50_absolute_common import ROOT, atomic_write_json
from run_cholect50_rendezvous_fact_smoke import COMPONENT_NAMES


EVENTS = {
    "clip_application_observed": (("instrument", "clipper"), ("verb", "clip")),
    "scissor_cut_observed": (("instrument", "scissors"), ("verb", "cut")),
    "coagulation_tool_action_observed": (("instrument", "bipolar"), ("verb", "coagulate")),
    "hook_coagulation_observed": (("instrument", "hook"), ("verb", "coagulate")),
    "specimen_bag_observed": (("target", "specimen_bag"),),
    "irrigator_observed": (("instrument", "irrigator"),),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/cholect50_absolute_development_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.run_dir / "TEMPORAL_FACTS_V2.json"
    if output.exists():
        raise FileExistsError(output)
    calibration = json.loads((args.run_dir / "RELIABILITY_CALIBRATION.json").read_text())
    cases = json.loads((args.run_dir / "DEVELOPMENT_CASES.json").read_text())
    source = np.load(args.run_dir / "DEVELOPMENT_SCORES.npz")
    videos = source["video"].astype(str)
    frames = source["frame_id"].astype(int)
    class_index = {
        (component, name): index
        for component, names in COMPONENT_NAMES.items()
        for index, name in enumerate(names)
    }
    thresholds = {
        (component, name): float(row["threshold"])
        for component, rows in calibration["classes"].items()
        for name, row in rows.items()
    }
    enabled = {
        (component, name): bool(row["enabled"])
        for component, rows in calibration["classes"].items()
        for name, row in rows.items()
    }
    indexed = {
        (str(video), int(frame)): {
            component: source[component][row_index]
            for component in COMPONENT_NAMES
        }
        for row_index, (video, frame) in enumerate(zip(videos, frames))
    }

    def atomic_hit(video: str, frame: int, component: str, name: str) -> tuple[bool, float]:
        if not enabled.get((component, name), False):
            return False, 0.0
        score = float(indexed[(video, frame)][component][class_index[(component, name)]])
        return score >= thresholds[(component, name)], score

    records = []
    for case in cases:
        video = case["video"]
        anchor = int(case["anchor_frame"])
        usable = sorted(frame for candidate_video, frame in indexed if candidate_video == video and frame <= anchor + 2)
        event_rows = []
        for event_name, requirements in EVENTS.items():
            hits = []
            strengths = []
            for frame in usable:
                results = [atomic_hit(video, frame, component, name) for component, name in requirements]
                if all(result[0] for result in results):
                    hits.append(frame)
                    strengths.append(min(result[1] for result in results))
            if hits:
                event_rows.append(
                    {
                        "event": event_name,
                        "hit_count": len(hits),
                        "first_hit_seconds": int(hits[0]),
                        "last_hit_seconds": int(hits[-1]),
                        "seconds_since_last_hit": int(anchor - hits[-1]),
                        "maximum_joint_score": float(max(strengths)),
                        "current_30s": any(anchor - 30 <= frame <= anchor + 2 for frame in hits),
                        "recent_120s": any(anchor - 120 <= frame <= anchor + 2 for frame in hits),
                    }
                )
        event_rows.sort(key=lambda row: (row["first_hit_seconds"], row["event"]))

        current_atomic = []
        recent_frames = [frame for frame in usable if anchor - 10 <= frame <= anchor + 2]
        for component, names in COMPONENT_NAMES.items():
            for index, name in enumerate(names):
                if not enabled.get((component, name), False) or not recent_frames:
                    continue
                values = np.asarray([indexed[(video, frame)][component][index] for frame in recent_frames])
                hit_indices = np.flatnonzero(values >= thresholds[(component, name)])
                if len(hit_indices) == 0:
                    continue
                current_atomic.append(
                    {
                        "component": component,
                        "name": name,
                        "hit_count": int(len(hit_indices)),
                        "sampled_frames": len(recent_frames),
                        "maximum_score": float(values.max()),
                        "calibrated_precision": float(calibration["classes"][component][name]["precision"]),
                    }
                )
        current_atomic.sort(key=lambda row: (-row["calibrated_precision"], -row["hit_count"], row["name"]))
        records.append(
            {
                "case_id": case["case_id"],
                "center_seconds": anchor,
                "video_duration_seconds": int(case["video_frames"]) - 1,
                "elapsed_fraction": float(case["elapsed_fraction"]),
                "current_atomic_observations": current_atomic[:10],
                "compound_event_history": event_rows,
                "phase_label_or_probability_in_plugin": False,
            }
        )
    atomic_write_json(
        output,
        {
            "status": "conservative_compound_temporal_facts_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "event_definitions": {
                event: [{"component": component, "name": name} for component, name in requirements]
                for event, requirements in EVENTS.items()
            },
            "cases": records,
            "phase_output_generated": False,
            "ground_truth_phase_in_fact_records": False,
        },
    )
    print(json.dumps({"output": str(output), "cases": len(records), "events": list(EVENTS)}, indent=2))


if __name__ == "__main__":
    main()

