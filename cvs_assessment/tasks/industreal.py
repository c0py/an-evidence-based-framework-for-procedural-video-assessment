from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..schema import CriterionSpec, EvaluationWindowSpec, EvidenceQuery, TemporalPolicy
from ..tasking import TaskPackage


@dataclass
class IndustRealNormalizedAdapter:
    """Adapter for a normalized PSR/ASD export, isolated from the core runtime.

    The official IndustReal release can be converted once into JSON records with
    ``video_id``, ``start_s``, ``end_s``, ``criterion_id``, and ordinal ``state``.
    Dataset release-layout parsing deliberately belongs in this adapter rather
    than in SkillExecutor or the verifier.
    """

    task_id: str = "industreal_assembly"

    @staticmethod
    def _record(video_id: str | int, annotation_path: str) -> dict[str, Any]:
        value = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        videos = value.get("videos", value)
        key = str(video_id)
        if key not in videos:
            raise KeyError(f"Normalized IndustReal annotations contain no video {key}")
        return videos[key]

    def locate_window(self, video_id: str | int, **sources: Any) -> dict[str, float]:
        record = self._record(video_id, sources["annotation_path"])
        return {
            "start_s": float(record.get("start_s", 0.0)),
            "end_s": float(record["end_s"]),
            "anchor_s": float(record["end_s"]),
        }

    def load_intervals(
        self, video_id: str | int, **sources: Any,
    ) -> dict[str, list[tuple[float, float, int]]]:
        record = self._record(video_id, sources["annotation_path"])
        output: dict[str, list[tuple[float, float, int]]] = {}
        for step in record.get("steps", []):
            output.setdefault(str(step["criterion_id"]), []).append((
                float(step["start_s"]), float(step["end_s"]), int(step.get("state", 2)),
            ))
        return output


@dataclass
class IndustRealRealPSRAdapter:
    """Adapter for normalized real-release component-state transitions."""

    criterion_state_map: dict[str, tuple[str, int]]
    task_id: str = "industreal_psr_assembly"

    @staticmethod
    def _record(video_id: str | int, annotation_path: str) -> dict[str, Any]:
        value = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        key = str(video_id)
        if key not in value.get("recordings", {}):
            raise KeyError(f"Normalized real IndustReal annotations contain no video {key}")
        return value["recordings"][key]

    def locate_window(self, video_id: str | int, **sources: Any) -> dict[str, float]:
        record = self._record(video_id, sources["annotation_path"])
        return {
            "start_s": 0.0, "end_s": float(record["duration_s"]),
            "anchor_s": float(record["duration_s"]),
        }

    def load_intervals(
        self, video_id: str | int, **sources: Any,
    ) -> dict[str, list[tuple[float, float, int]]]:
        record = self._record(video_id, sources["annotation_path"])
        transitions = record["raw_state_transitions"]
        duration_s = float(record["duration_s"])
        output: dict[str, list[tuple[float, float, int]]] = {}
        for criterion_id, (component, target_state) in self.criterion_state_map.items():
            start_s: float | None = None
            intervals: list[tuple[float, float, int]] = []
            for index, row in enumerate(transitions):
                timestamp_s = float(row["timestamp_s"])
                satisfied = int(row["component_states"][component]) == target_state
                if satisfied and start_s is None:
                    start_s = timestamp_s
                if start_s is not None and not satisfied:
                    intervals.append((start_s, timestamp_s, 2))
                    start_s = None
                if index == len(transitions) - 1 and start_s is not None:
                    intervals.append((start_s, duration_s, 2))
            output[criterion_id] = intervals
        return output


def _step(key: str, title: str, requirement: str) -> CriterionSpec:
    negative = "The step is omitted, performed with an incorrect part, or does not reach the required assembly state."
    return CriterionSpec(
        key=key,
        title=title,
        visual_requirement=requirement,
        negative_evidence=negative,
        evidence_query=EvidenceQuery(
            criterion_id=key,
            requirement=requirement,
            negative_evidence=negative,
            required_capabilities=[
                "object_state_recognition", "relation_recognition", "step_completion_recognition",
            ],
            attributes={"domain": "industrial_assembly"},
        ),
        temporal_rule="persistent_state_transition",
        tool_sequence=["criterion_visual_evidence", "state_transition_aggregation"],
        decision_rule="The required assembly state must be reached and persist in the prescribed order.",
        temporal_policy=TemporalPolicy(
            operator="persistent_state_transition",
            source_state="not_completed",
            target_state="satisfied",
            parameters={
                "smoothing_seconds": 1.0,
                "on_threshold": 0.65,
                "off_threshold": 0.50,
                "min_stable_seconds": 2.0,
                "max_gap_seconds": 1.0,
            },
        ),
    )


def create_industreal_package() -> TaskPackage:
    # These neutral step roles form an executable example package. A converter
    # maps the release's procedure-specific PSR step identifiers onto them.
    criteria = {
        "base_assembly_completed": _step(
            "base_assembly_completed", "Base assembly completed",
            "The required base components are correctly joined.",
        ),
        "wheel_assembly_completed": _step(
            "wheel_assembly_completed", "Wheel assembly completed",
            "The required wheel components are correctly attached using the prescribed parts.",
        ),
        "final_assembly_verified": _step(
            "final_assembly_verified", "Final assembly verified",
            "The final construction-toy assembly matches the required completed state.",
        ),
    }
    return TaskPackage(
        task_id="industreal_assembly",
        task_name="IndustReal procedure-step assessment",
        description=(
            "Assess correct step completion, persistent assembly state, ordering, omissions, "
            "and execution errors in an industrial-like assembly procedure."
        ),
        criterion_catalog=criteria,
        aliases={
            "base_assembly_completed": ("base assembly", "base components", "chassis"),
            "wheel_assembly_completed": ("wheel assembly", "attach the wheel", "wheel components"),
            "final_assembly_verified": ("final assembly", "completed state", "verify assembly"),
        },
        allowed_tools={
            "locate_evaluation_window", "sample_timestamps",
            "criterion_visual_evidence", "state_transition_aggregation",
            "explicit_verifier",
        },
        window_spec=EvaluationWindowSpec(
            locator_tool="locate_evaluation_window",
            start_anchor="procedure_start",
            end_anchor="procedure_end",
            attributes={"scope": "complete_procedure"},
        ),
        default_temporal_policy=TemporalPolicy(
            operator="persistent_state_transition",
            source_state="not_completed",
            target_state="satisfied",
            parameters={
                "smoothing_seconds": 1.0, "on_threshold": 0.65,
                "off_threshold": 0.50, "min_stable_seconds": 2.0,
                "max_gap_seconds": 1.0,
            },
        ),
        ordering_constraints=[
            ("base_assembly_completed", "wheel_assembly_completed"),
            ("wheel_assembly_completed", "final_assembly_verified"),
        ],
        dataset_adapter=IndustRealNormalizedAdapter(),
        metadata={
            "domain": "industrial_assembly",
            "dataset": "IndustReal",
            "supported_tasks": ["procedure_step_recognition", "assembly_state_detection"],
        },
    )


def create_industreal_real_psr_package(
    procedure_info: list[dict[str, Any]], procedure_kind: str = "assy",
) -> TaskPackage:
    """Build the real benchmark Task Package from official procedure metadata."""
    if procedure_kind not in {"assy", "main"}:
        raise ValueError(f"Unknown IndustReal procedure kind: {procedure_kind}")
    component_names = (
        "base", "front_chassis", "front_chassis_pin", "rear_chassis",
        "short_rear_chassis", "front_rear_chassis_pin", "rear_rear_chassis_pin",
        "front_bracket", "front_bracket_screw", "front_wheel_assembly",
        "rear_wheel_assembly",
    )
    criteria: dict[str, CriterionSpec] = {}
    aliases: dict[str, tuple[str, ...]] = {}
    state_map: dict[str, tuple[str, int]] = {}
    action_ids: list[int] = []
    for row in procedure_info:
        if not bool(row[f"expected_in_{procedure_kind}"]):
            continue
        action_id = int(row["id"])
        component = component_names[int(row["state_idx"])]
        operation = "installed" if bool(row["install"]) else "removed"
        target_state = 1 if operation == "installed" else 0
        key = f"action_{action_id:02d}_{component}_{operation}"
        requirement = (
            f"The {component.replace('_', ' ')} is correctly {operation} with the "
            "required part and connection state."
        )
        criteria[key] = _step(key, str(row["description"]), requirement)
        aliases[key] = (
            str(row["description"]), component.replace("_", " "), key,
        )
        state_map[key] = (component, target_state)
        action_ids.append(action_id)
    criterion_ids = list(criteria)
    key_by_action_id = dict(zip(action_ids, criterion_ids))
    # PSR annotations may mark multiple components at the same frame, so do
    # not impose a total order between every criterion. These assembly
    # milestone edges encode only the unambiguous coarse progression.
    coarse_action_edges = (
        [(3, 9), (9, 18), (18, 21), (21, 27), (27, 30)]
        if procedure_kind == "assy" else []
    )
    return TaskPackage(
        task_id=f"industreal_psr_{'assembly' if procedure_kind == 'assy' else 'maintenance'}",
        task_name=f"IndustReal real {procedure_kind} procedure-step recognition",
        description=(
            "Assess correct component-state completion, persistence, ordering, omissions, "
            "and execution errors from real egocentric assembly video."
        ),
        criterion_catalog=criteria,
        aliases=aliases,
        allowed_tools={
            "locate_evaluation_window", "sample_timestamps",
            "criterion_visual_evidence", "state_transition_aggregation",
            "explicit_verifier",
        },
        window_spec=EvaluationWindowSpec(
            locator_tool="locate_evaluation_window", start_anchor="procedure_start",
            end_anchor="procedure_end", attributes={"scope": "complete_procedure"},
        ),
        default_temporal_policy=TemporalPolicy(
            operator="persistent_state_transition", source_state="not_completed",
            target_state="satisfied", parameters={
                "smoothing_seconds": 1.0, "on_threshold": 0.65,
                "off_threshold": 0.50, "min_stable_seconds": 2.0,
                "max_gap_seconds": 1.0,
            },
        ),
        ordering_constraints=[
            (key_by_action_id[left], key_by_action_id[right])
            for left, right in coarse_action_edges
            if left in key_by_action_id and right in key_by_action_id
        ],
        dataset_adapter=IndustRealRealPSRAdapter(
            criterion_state_map=state_map,
            task_id=f"industreal_psr_{'assembly' if procedure_kind == 'assy' else 'maintenance'}",
        ),
        metadata={
            "domain": "industrial_assembly", "dataset": "IndustReal",
            "release_annotations": "real_PSR_component_states",
            "procedure_kind": procedure_kind,
            "criterion_action_ids": action_ids,
            "foundation_mllm_finetuned": False,
        },
    )
