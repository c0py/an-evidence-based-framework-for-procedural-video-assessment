"""Public, task-neutral facade for the frozen-foundation plugin framework v2.

The facade deliberately contains no surgical or industrial decision rule.  A
task package supplies criterion identifiers, a Skill, and fact-only plugin
slots; the shared core supplies the two frozen-Qwen roles, their label-free
consensus prior, and bounded residual arbitration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .foundation_consensus_residual_arbitration import (
    ROLE_NAMES,
    build_consensus_prior,
    merge_consensus_residual_judgment,
    select_consensus_residual_frame_ids,
)


ALLOWED_PLUGIN_KINDS = ("visual", "temporal")


@dataclass(frozen=True)
class EvidencePluginSlot:
    """A replaceable plugin boundary; plugins may report facts, never verdicts."""

    slot_id: str
    plugin_kind: str
    fact_only: bool = True
    may_emit_final_task_prediction: bool = False

    def validate(self) -> None:
        if not self.slot_id.strip():
            raise ValueError("Plugin slot needs a nonempty id")
        if self.plugin_kind not in ALLOWED_PLUGIN_KINDS:
            raise ValueError(f"Unsupported plugin kind: {self.plugin_kind}")
        if not self.fact_only or self.may_emit_final_task_prediction:
            raise ValueError("Framework-v2 plugins must be fact-only")


@dataclass(frozen=True)
class ProceduralTaskPackageV2:
    """The only task-specific input accepted by the shared v2 core."""

    task_id: str
    skill_id: str
    criterion_ids: tuple[str, ...]
    plugin_slots: tuple[EvidencePluginSlot, ...]
    sampled_frame_count: int

    def validate(self) -> None:
        if not self.task_id.strip() or not self.skill_id.strip():
            raise ValueError("Task and Skill ids must be nonempty")
        if not self.criterion_ids or len(set(self.criterion_ids)) != len(self.criterion_ids):
            raise ValueError("Criterion ids must be nonempty and unique")
        if any(not value.strip() for value in self.criterion_ids):
            raise ValueError("Criterion ids must be nonempty strings")
        if self.sampled_frame_count < 1:
            raise ValueError("A task package must sample at least one frame")
        if not self.plugin_slots or len({slot.slot_id for slot in self.plugin_slots}) != len(self.plugin_slots):
            raise ValueError("Plugin slots must be nonempty and unique")
        for slot in self.plugin_slots:
            slot.validate()


@dataclass(frozen=True)
class FrozenFoundationFrameworkV2:
    """Shared method configuration and the sole numerical merge entry point."""

    foundation_model: str = "qwen3-vl-32b-sop"
    foundation_model_finetuned: bool = False
    role_order: tuple[str, str] = ROLE_NAMES
    consensus_aggregation: str = "unweighted_arithmetic_mean"
    residual_bound: float = 0.10
    raw_small_model_rows_used_as_final_prediction: bool = False

    def validate(self) -> None:
        if self.foundation_model_finetuned:
            raise ValueError("Framework v2 requires a frozen foundation model")
        if tuple(self.role_order) != ROLE_NAMES:
            raise ValueError("Framework-v2 frozen-Qwen role order is fixed")
        if self.consensus_aggregation != "unweighted_arithmetic_mean":
            raise ValueError("Framework-v2 consensus aggregation is fixed")
        if self.residual_bound != 0.10:
            raise ValueError("Framework-v2 residual bound is fixed at 0.10")
        if self.raw_small_model_rows_used_as_final_prediction:
            raise ValueError("Small-model rows cannot be final predictions")

    def manifest(self, task_package: ProceduralTaskPackageV2) -> dict[str, Any]:
        self.validate()
        task_package.validate()
        return {
            "schema_version": "frozen_foundation_plugin_framework_v2",
            "shared_core": asdict(self),
            "task_package": asdict(task_package),
            "execution": {
                "prior": "mean_of_two_complete_frozen_qwen_judgments",
                "route": "all_frames_with_any_complete_qwen_row_difference",
                "residual_actions": {"hold": 0.0, "upgrade": 0.1, "downgrade": -0.1},
                "final_semantic_judge": "same_frozen_qwen",
                "labels_required_during_inference": False,
            },
        }

    def build_prior(self, candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
        self.validate()
        return build_consensus_prior(candidates)

    def route_frames(
        self, candidates: Iterable[dict[str, Any]], *, maximum_frames: int,
    ) -> list[int]:
        self.validate()
        return select_consensus_residual_frame_ids(candidates, maximum_frames=maximum_frames)

    def merge(
        self,
        candidates: Iterable[dict[str, Any]],
        arbitration: dict[str, Any] | None,
        frame_ids: Iterable[int],
    ) -> dict[str, Any]:
        self.validate()
        output = merge_consensus_residual_judgment(candidates, arbitration, frame_ids)
        observed = float(output["consensus_residual_merge"]["residual_bound"])
        if observed != self.residual_bound:
            raise RuntimeError("Core merge residual differs from the v2 contract")
        return output

