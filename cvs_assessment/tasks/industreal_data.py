"""Release-layout normalization for the real IndustReal PSR benchmark."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


FPS = 10.0
COMPONENT_NAMES = (
    "base", "front_chassis", "front_chassis_pin", "rear_chassis",
    "short_rear_chassis", "front_rear_chassis_pin", "rear_rear_chassis_pin",
    "front_bracket", "front_bracket_screw", "front_wheel_assembly",
    "rear_wheel_assembly",
)


@dataclass(frozen=True)
class ProcedureEvent:
    frame_index: int
    timestamp_s: float
    action_id: int
    description: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_procedure_info(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise ValueError("IndustReal procedure_info must be a nonempty list")
    ids = [int(row["id"]) for row in value]
    if ids != list(range(len(value))):
        raise ValueError("IndustReal procedure action IDs must be contiguous")
    return value


def load_event_csv(path: Path) -> list[ProcedureEvent]:
    output: list[ProcedureEvent] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 3:
                raise ValueError(f"Malformed IndustReal PSR row in {path}: {row}")
            frame_index = int(Path(row[0]).stem)
            output.append(ProcedureEvent(
                frame_index=frame_index,
                timestamp_s=frame_index / FPS,
                action_id=int(row[1]), description=str(row[2]),
            ))
    return output


def load_raw_states(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 1 + len(COMPONENT_NAMES):
                raise ValueError(f"Malformed IndustReal raw-state row in {path}: {row}")
            frame_index = int(Path(row[0]).stem)
            states = [int(value) for value in row[1:]]
            if any(value not in {-1, 0, 1} for value in states):
                raise ValueError("IndustReal raw component state must be -1, 0, or 1")
            output.append({
                "frame_index": frame_index,
                "timestamp_s": frame_index / FPS,
                "component_states": dict(zip(COMPONENT_NAMES, states)),
            })
    if not output or any(
        current["frame_index"] <= previous["frame_index"]
        for previous, current in zip(output, output[1:])
    ):
        raise ValueError(f"Raw states must be nonempty and chronological: {path}")
    return output


def procedure_kind(recording_id: str) -> str:
    if "_assy_" in recording_id:
        return "assy"
    if "_main_" in recording_id:
        return "main"
    raise ValueError(f"Cannot infer IndustReal procedure kind: {recording_id}")


def normalize_recording(
    recording_dir: Path, split: str, procedure_info: list[dict[str, Any]],
) -> dict[str, Any]:
    recording_id = recording_dir.name
    kind = procedure_kind(recording_id)
    required = {
        "correct_events": recording_dir / "PSR_labels.csv",
        "events_with_errors": recording_dir / "PSR_labels_with_errors.csv",
        "raw_states": recording_dir / "PSR_labels_raw.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete IndustReal labels: {missing}")
    correct_events = load_event_csv(required["correct_events"])
    events_with_errors = load_event_csv(required["events_with_errors"])
    raw_states = load_raw_states(required["raw_states"])
    expected_action_ids = [
        int(row["id"]) for row in procedure_info if bool(row[f"expected_in_{kind}"])
    ]
    observed_correct_ids = [event.action_id for event in correct_events]
    missing_expected = [
        action_id for action_id in expected_action_ids
        if action_id not in observed_correct_ids
    ]
    incorrect_events = [
        event for event in events_with_errors
        if procedure_info[event.action_id]["description"].lower().startswith("incorrectly")
    ]
    erroneous_components = sorted({
        component
        for row in raw_states
        for component, state in row["component_states"].items()
        if state == -1
    })
    duration_s = raw_states[-1]["timestamp_s"]
    return {
        "recording_id": recording_id,
        "split": split,
        "procedure_kind": kind,
        "fps": FPS,
        "duration_s": duration_s,
        "expected_action_ids": expected_action_ids,
        "missing_expected_action_ids": missing_expected,
        "correct_completion_events": [asdict(event) for event in correct_events],
        "completion_events_with_errors": [
            asdict(event) for event in events_with_errors
        ],
        "incorrect_completion_events": [asdict(event) for event in incorrect_events],
        "raw_state_transitions": raw_states,
        "erroneous_components": erroneous_components,
        "procedure_has_error": bool(
            missing_expected or incorrect_events or erroneous_components
        ),
        "label_sources": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in required.items()
        },
    }


def normalize_annotation_tree(
    annotations_root: Path, procedure_info_path: Path,
) -> dict[str, Any]:
    procedure_info = load_procedure_info(procedure_info_path)
    recordings: dict[str, dict[str, Any]] = {}
    for archive_dir in sorted(path for path in annotations_root.iterdir() if path.is_dir()):
        prefix = archive_dir.name.split("_p", 1)[0]
        split = "validation" if prefix == "val" else prefix
        if split not in {"train", "validation"}:
            raise PermissionError(
                f"Only train/validation annotations may be normalized during development: {archive_dir}"
            )
        for correct_path in sorted(archive_dir.rglob("PSR_labels.csv")):
            recording_dir = correct_path.parent
            record = normalize_recording(recording_dir, split, procedure_info)
            recording_id = record["recording_id"]
            if recording_id in recordings:
                raise ValueError(f"Duplicate normalized recording: {recording_id}")
            recordings[recording_id] = record
    if not recordings:
        raise ValueError(f"No IndustReal PSR recordings found under {annotations_root}")
    return {
        "schema_version": "normalized_industreal_psr_real_v1",
        "dataset": "IndustReal",
        "fps": FPS,
        "component_order": list(COMPONENT_NAMES),
        "procedure_info": procedure_info,
        "recording_count": len(recordings),
        "split_counts": {
            split: sum(row["split"] == split for row in recordings.values())
            for split in ("train", "validation")
        },
        "official_test_annotations_accessed": False,
        "recordings": recordings,
    }
