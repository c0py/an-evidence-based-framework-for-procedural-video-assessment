#!/usr/bin/env python3
"""Build the frozen v2 temporal facts with a large-cohort I/O optimization.

This is semantically identical to build_cholect50_temporal_facts_v2.py.  The
only change is that compressed NumPy component arrays are decompressed once,
instead of once per row, which matters for the 31k-frame transfer cache.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from cholect50_absolute_common import ROOT, atomic_write_json
from build_cholect50_temporal_facts_v2 import EVENTS
from run_cholect50_rendezvous_fact_smoke import COMPONENT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/cholec80_nonoverlap_transfer_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.run_dir / "TEMPORAL_FACTS_V2.json"
    if output.exists():
        raise FileExistsError(output)
    calibration = json.loads((args.run_dir / "RELIABILITY_CALIBRATION.json").read_text())
    cases = json.loads((args.run_dir / "DEVELOPMENT_CASES.json").read_text())
    with np.load(args.run_dir / "DEVELOPMENT_SCORES.npz") as source:
        videos = source["video"].astype(str)
        frames = source["frame_id"].astype(int)
        score_arrays = {component: np.asarray(source[component]) for component in COMPONENT_NAMES}
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
    per_video: dict[str, list[tuple[int, int]]] = defaultdict(list)
    frame_to_row: dict[tuple[str, int], int] = {}
    for row_index, (video, frame) in enumerate(zip(videos, frames)):
        key = (str(video), int(frame))
        if key in frame_to_row:
            raise RuntimeError(f"duplicate score row: {key}")
        frame_to_row[key] = row_index
        per_video[str(video)].append((int(frame), row_index))
    for rows in per_video.values():
        rows.sort()

    def atomic_hit(video: str, frame: int, component: str, name: str) -> tuple[bool, float]:
        if not enabled.get((component, name), False):
            return False, 0.0
        row_index = frame_to_row[(video, frame)]
        score = float(score_arrays[component][row_index, class_index[(component, name)]])
        return score >= thresholds[(component, name)], score

    records = []
    for case_index, case in enumerate(cases, start=1):
        video = case["video"]
        anchor = int(case["anchor_frame"])
        usable = [frame for frame, _ in per_video[video] if frame <= anchor + 2]
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
                row_indices = [frame_to_row[(video, frame)] for frame in recent_frames]
                values = score_arrays[component][row_indices, index]
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
        if case_index % 250 == 0 or case_index == len(cases):
            print(json.dumps({"temporal_cases_completed": case_index, "total": len(cases)}), flush=True)
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
            "implementation_note": "I/O-optimized equivalent of build_cholect50_temporal_facts_v2.py",
        },
    )
    print(json.dumps({"output": str(output), "cases": len(records), "events": list(EVENTS)}, indent=2))


if __name__ == "__main__":
    main()
