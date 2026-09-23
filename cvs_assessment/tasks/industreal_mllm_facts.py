"""Fact-only IndustReal assembly-state plugins for a frozen foundation MLLM."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..grounded_evidence import GroundedEvidenceFact, grounded_fact_payload
from ..mllm_orchestration import CriterionContract
from .industreal_data import COMPONENT_NAMES


# Official ASD class order from IndustReal PSR/psr_utils.py. Each binary string
# is an observed assembly-state pattern over COMPONENT_NAMES; the detector's
# error_state is deliberately not decoded into a task verdict.
ASD_STATE_PATTERNS: tuple[str, ...] = (
    "background",
    "10000000000", "10010010000", "10010100000", "10010110000",
    "11100000000", "11110010000", "11110100000", "11110110000",
    "11110111100", "11110111110", "11110110001", "11110111101",
    "11110111111", "11110101111", "11110011111", "11110011110",
    "11110101110", "11100001110", "11101101110", "11101011110",
    "11101111110", "11101111111",
    "error_state",
)


def assembly_criterion_contracts(
    procedure_info: list[dict[str, Any]], procedure_kind: str = "assy",
) -> list[CriterionContract]:
    if procedure_kind not in {"assy", "main"}:
        raise ValueError(f"Unknown IndustReal procedure kind: {procedure_kind}")
    output = []
    for row in procedure_info:
        if not bool(row[f"expected_in_{procedure_kind}"]):
            continue
        action_id = int(row["id"])
        state_idx = int(row["state_idx"])
        operation = "installed" if bool(row["install"]) else "removed"
        component = COMPONENT_NAMES[state_idx]
        output.append(CriterionContract(
            criterion_id=f"action_{action_id:02d}_{component}_{operation}",
            title=str(row["description"]),
            minimal_description=(
                f"The {component.replace('_', ' ')} has been correctly {operation}; "
                "an incorrect part/orientation or a merely attempted action is not full completion."
            ),
        ))
    return output


def criterion_component_index(criterion_id: str) -> int:
    # Longest-first avoids matching ``front_chassis`` inside
    # ``front_chassis_pin``.
    for index, component in sorted(
        enumerate(COMPONENT_NAMES), key=lambda item: len(item[1]), reverse=True,
    ):
        if f"_{component}_" in criterion_id:
            return index
    raise KeyError(f"Criterion does not name an IndustReal component: {criterion_id}")


def criterion_target_state(criterion_id: str) -> int:
    if criterion_id.endswith("_installed"):
        return 1
    if criterion_id.endswith("_removed"):
        return 0
    raise KeyError(f"Criterion has no installed/removed target state: {criterion_id}")


def labels_at_timestamps(
    normalized_record: dict[str, Any], criteria: list[CriterionContract],
    timestamps_s: list[float],
) -> np.ndarray:
    transitions = normalized_record["raw_state_transitions"]
    output = np.zeros((len(timestamps_s), len(criteria)), dtype=np.float32)
    for time_index, timestamp_s in enumerate(timestamps_s):
        eligible = [
            row for row in transitions if float(row["timestamp_s"]) <= timestamp_s
        ]
        state = eligible[-1]["component_states"] if eligible else transitions[0]["component_states"]
        for criterion_index, criterion in enumerate(criteria):
            component_index = criterion_component_index(criterion.criterion_id)
            component = COMPONENT_NAMES[component_index]
            target = criterion_target_state(criterion.criterion_id)
            # An error state (-1) never satisfies either correct install/remove.
            output[time_index, criterion_index] = float(state[component] == target)
    return output


def load_asd_prediction_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"clip", "framenr", "bb_class", "bb_conf", "bb_x", "bb_y", "bb_w", "bb_h"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"Unexpected IndustReal ASD columns: {reader.fieldnames}")
        for row in reader:
            class_index = int(row["bb_class"])
            if not 0 <= class_index < len(ASD_STATE_PATTERNS):
                raise ValueError(f"Unknown IndustReal ASD state class: {class_index}")
            rows.append({
                "recording_id": row["clip"],
                "frame_index": int(row["framenr"]),
                "state_class_index": class_index,
                "confidence": float(row["bb_conf"]),
                "bbox_xywh": [
                    float(row["bb_x"]), float(row["bb_y"]),
                    float(row["bb_w"]), float(row["bb_h"]),
                ],
            })
    return rows


def _decoded_components(class_index: int) -> dict[str, str] | None:
    pattern = ASD_STATE_PATTERNS[class_index]
    if pattern in {"background", "error_state"}:
        return None
    return {
        component: ("observed_installed" if bit == "1" else "observed_not_installed")
        for component, bit in zip(COMPONENT_NAMES, pattern)
    }


def visual_fact_payload(
    prediction_rows: list[dict[str, Any]], timestamps_s: list[float],
    frame_indices: list[int], frame_width: int = 1280, frame_height: int = 720,
) -> dict[str, Any]:
    if len(timestamps_s) != len(frame_indices):
        raise ValueError("IndustReal timestamps and frame indices must align")
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        by_frame.setdefault(int(row["frame_index"]), []).append(row)
    facts: list[GroundedEvidenceFact] = []
    for output_index, (source_frame, timestamp_s) in enumerate(zip(frame_indices, timestamps_s)):
        candidates = by_frame.get(int(source_frame), [])
        if not candidates:
            continue
        row = max(candidates, key=lambda item: float(item["confidence"]))
        class_index = int(row["state_class_index"])
        x, y, width, height = [float(value) for value in row["bbox_xywh"]]
        # Ultralytics ``box.xywh`` uses centre-x/centre-y coordinates.  Convert
        # to corners before normalising the grounding box exported to the MLLM.
        left = max(0.0, x - width / 2.0)
        top = max(0.0, y - height / 2.0)
        right = min(float(frame_width), x + width / 2.0)
        bottom = min(float(frame_height), y + height / 2.0)
        decoded = _decoded_components(class_index)
        facts.append(GroundedEvidenceFact(
            fact_id=f"assembly_state:{output_index}", fact_type="entity",
            subject=f"assembly:{output_index}", predicate="predicted_assembly_state",
            value={
                "state_class": ASD_STATE_PATTERNS[class_index],
                "component_observations": decoded,
                "error_state_observed": ASD_STATE_PATTERNS[class_index] == "error_state",
            },
            confidence=float(row["confidence"]), frame_index=output_index,
            timestamp_s=float(timestamp_s),
            grounding={
                "source": "official_industreal_assembly_state_detector",
                "source_video_frame_index": int(source_frame),
                "bbox_normalized_xyxy": [
                    left / frame_width, top / frame_height,
                    right / frame_width, bottom / frame_height,
                ],
                "fallible_observation": True,
                "not_a_procedure_step_verdict": True,
            },
        ))
    return grounded_fact_payload(
        facts, modality="visual",
        source_description=(
            "A frozen assembly-state detector reports fallible state patterns grounded "
            "to the visible assembly box. It does not decide step correctness or order."
        ),
        provenance={
            "adapter": "industreal_asd_visual_facts_v1",
            "official_state_class_order": True,
            "raw_task_verdict_exported": False,
        },
    )


def temporal_fact_payload(
    prediction_rows: list[dict[str, Any]], fps: float = 10.0,
    minimum_confidence: float = 0.25,
    minimum_stable_observations: int = 3,
    maximum_transition_facts: int = 80,
    maximum_gap_multiplier: float = 2.5,
    expected_cadence_frames: int | None = None,
) -> dict[str, Any]:
    """Decode dense detector observations into generic component transitions."""
    if (
        fps <= 0 or minimum_stable_observations < 1
        or maximum_transition_facts < 1 or maximum_gap_multiplier <= 0
        or (expected_cadence_frames is not None and expected_cadence_frames < 1)
    ):
        raise ValueError("IndustReal temporal fact configuration is invalid")
    best_by_frame: dict[int, dict[str, Any]] = {}
    for row in prediction_rows:
        frame = int(row["frame_index"])
        if float(row["confidence"]) < minimum_confidence:
            continue
        if frame not in best_by_frame or float(row["confidence"]) > float(best_by_frame[frame]["confidence"]):
            best_by_frame[frame] = row
    ordered = [best_by_frame[frame] for frame in sorted(best_by_frame)]
    ordered_frames = [int(row["frame_index"]) for row in ordered]
    cadence_frames = int(expected_cadence_frames or (
        int(round(float(np.median(np.diff(ordered_frames)))))
        if len(ordered_frames) > 1 else 1
    ))
    cadence_frames = max(1, cadence_frames)
    maximum_gap_frames = max(1, int(round(cadence_frames * maximum_gap_multiplier)))
    runs: list[list[dict[str, Any]]] = []
    for row in ordered:
        gap = (
            int(row["frame_index"]) - int(runs[-1][-1]["frame_index"])
            if runs else 0
        )
        if (
            not runs
            or int(runs[-1][-1]["state_class_index"]) != int(row["state_class_index"])
            or gap > maximum_gap_frames
        ):
            runs.append([row])
        else:
            runs[-1].append(row)
    stable_runs = [run for run in runs if len(run) >= minimum_stable_observations]
    facts: list[GroundedEvidenceFact] = []
    previous: dict[str, str] | None = None
    transition_count = 0
    for run_index, run in enumerate(stable_runs):
        row = run[0]
        frame = int(row["frame_index"])
        end_frame = int(run[-1]["frame_index"])
        class_index = int(row["state_class_index"])
        decoded = _decoded_components(class_index)
        if decoded is None:
            if ASD_STATE_PATTERNS[class_index] == "error_state":
                facts.append(GroundedEvidenceFact(
                    fact_id=f"error_state:{run_index}", fact_type="event",
                    subject="assembly", predicate="detector_error_state_observation",
                    value=True,
                    confidence=float(np.mean([item["confidence"] for item in run])),
                    start_s=frame / fps, end_s=(end_frame + cadence_frames) / fps,
                    grounding={
                        "source": "dense_official_industreal_asd_track",
                        "supporting_observation_count": len(run),
                        "fallible_observation": True,
                        "not_a_procedure_verdict": True,
                    },
                ))
            continue
        facts.append(GroundedEvidenceFact(
            fact_id=f"stable_state:{run_index}", fact_type="event",
            subject="assembly", predicate="assembly_state_observed_persistently",
            value={
                "state_class": ASD_STATE_PATTERNS[class_index],
                "component_observations": decoded,
            },
            confidence=float(np.mean([item["confidence"] for item in run])),
            start_s=frame / fps, end_s=(end_frame + cadence_frames) / fps,
            grounding={
                "source": "dense_official_industreal_asd_track",
                "supporting_observation_count": len(run),
                "fallible_observation": True,
                "not_a_procedure_step_verdict": True,
            },
        ))
        if previous is not None:
            for component in COMPONENT_NAMES:
                if decoded[component] == previous[component]:
                    continue
                if transition_count >= maximum_transition_facts:
                    break
                facts.append(GroundedEvidenceFact(
                    fact_id=f"transition:{component}:{run_index}", fact_type="event",
                    subject=component, predicate="observed_state_transition",
                    value={"from": previous[component], "to": decoded[component]},
                    confidence=float(row["confidence"]), timestamp_s=frame / fps,
                    grounding={
                        "source": "dense_official_industreal_asd_track",
                        "source_video_frame_index": frame,
                        "fallible_observation": True,
                        "not_a_procedure_step_verdict": True,
                    },
                ))
                transition_count += 1
        previous = decoded
    return grounded_fact_payload(
        facts, modality="temporal",
        source_description=(
            "Dense fallible assembly-state observations are converted to component state "
            "transitions. The MLLM must verify completion, correctness, and procedure order."
        ),
        provenance={
            "adapter": "industreal_asd_temporal_facts_v1",
            "fps": float(fps),
            "minimum_confidence": float(minimum_confidence),
            "minimum_stable_observations": int(minimum_stable_observations),
            "maximum_transition_facts": int(maximum_transition_facts),
            "observed_cadence_frames": cadence_frames,
            "maximum_contiguous_gap_frames": maximum_gap_frames,
            "final_procedure_log_exported": False,
        },
    )
