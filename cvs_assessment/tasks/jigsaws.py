"""JIGSAWS surgical-skill task package for the task-neutral framework."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..foundation_framework_v2 import EvidencePluginSlot, ProceduralTaskPackageV2
from ..schema import CriterionSpec, EvaluationWindowSpec, EvidenceQuery, TemporalPolicy
from ..tasking import TaskPackage
from .jigsaws_data import (
    GRS_CRITERIA,
    grs_score_to_state,
    load_kinematics,
    trial_by_id,
)


_TITLES = {
    "respect_for_tissue": "Respect for tissue",
    "suture_needle_handling": "Suture and needle handling",
    "time_and_motion": "Time and motion",
    "flow_of_operation": "Flow of operation",
    "overall_performance": "Overall performance",
    "quality_of_final_product": "Quality of final product",
}

_REQUIREMENTS = {
    "respect_for_tissue": "Manipulation is controlled and avoids unnecessary force, contact, or rough handling.",
    "suture_needle_handling": "The needle and suture are handled securely and deliberately with suitable control.",
    "time_and_motion": "Movements are purposeful, economical, and contain limited avoidable travel or hesitation.",
    "flow_of_operation": "The procedure progresses coherently with limited interruption, repetition, or uncertainty.",
    "overall_performance": "The complete trial is controlled, efficient, and independently performed.",
    "quality_of_final_product": "The completed suture has suitable visible placement, form, and consistency.",
}


@dataclass
class JigsawsSkillAssessmentAdapter:
    """Evaluation adapter; label methods must never be called by inference code."""

    task_id: str = "jigsaws_suturing_skill"

    def locate_window(self, video_id: str | int, **sources: Any) -> dict[str, float]:
        trial = trial_by_id(Path(sources["dataset_root"]), str(video_id))
        duration_s = load_kinematics(trial.kinematics_path).shape[0] / 30.0
        return {"start_s": 0.0, "end_s": duration_s, "anchor_s": duration_s}

    def load_intervals(
        self, video_id: str | int, **sources: Any,
    ) -> dict[str, list[tuple[float, float, int]]]:
        """Expose N/P/F labels only to an explicit evaluation caller."""
        if sources.get("evaluation_labels_allowed") is not True:
            raise PermissionError(
                "JIGSAWS GRS labels are evaluation-only; set evaluation_labels_allowed=True"
            )
        trial = trial_by_id(Path(sources["dataset_root"]), str(video_id))
        duration_s = load_kinematics(trial.kinematics_path).shape[0] / 30.0
        encoding = {"N": 0, "P": 1, "F": 2}
        return {
            criterion_id: [(
                0.0,
                duration_s,
                encoding[grs_score_to_state(trial.grs_scores[criterion_id])],
            )]
            for criterion_id in GRS_CRITERIA
        }


def _quality_criterion(criterion_id: str) -> CriterionSpec:
    requirement = _REQUIREMENTS[criterion_id]
    negative = (
        "The trial contains clear low-quality execution for this dimension, such as "
        "rough, insecure, inefficient, interrupted, or visibly poor completion."
    )
    return CriterionSpec(
        key=criterion_id,
        title=_TITLES[criterion_id],
        visual_requirement=requirement,
        negative_evidence=negative,
        evidence_query=EvidenceQuery(
            criterion_id=criterion_id,
            requirement=requirement,
            negative_evidence=negative,
            modalities=["rgb", "robot_kinematics"],
            required_capabilities=[
                "procedural_quality_recognition",
                "motion_quality_observation",
                "trial_level_evidence_aggregation",
            ],
            output_states=["satisfied", "partial", "violated", "unknown"],
            attributes={
                "domain": "robotic_surgical_skill_assessment",
                "ground_truth_scale": "modified_GRS_1_to_5",
                "framework_state_mapping": {"N": "1-2", "P": "3", "F": "4-5"},
            },
        ),
        temporal_rule="stable_state",
        tool_sequence=[
            "criterion_visual_evidence",
            "kinematic_temporal_facts",
            "stable_evidence_aggregation",
        ],
        decision_rule=(
            "Judge the dimension from evidence across the complete trial; no single "
            "kinematic statistic or sampled image is an automatic verdict."
        ),
        temporal_policy=TemporalPolicy(
            operator="stable_state",
            target_state="satisfied",
            parameters={
                "smoothing_seconds": 12.0,
                "on_threshold": 0.60,
                "off_threshold": 0.45,
                "min_stable_seconds": 10.0,
                "max_gap_seconds": 8.0,
            },
        ),
    )


def create_jigsaws_suturing_package() -> TaskPackage:
    criteria = {
        criterion_id: _quality_criterion(criterion_id)
        for criterion_id in GRS_CRITERIA
    }
    return TaskPackage(
        task_id="jigsaws_suturing_skill",
        task_name="JIGSAWS suturing skill assessment",
        description=(
            "Assess six trial-level modified Global Rating Scale dimensions from "
            "stereo surgical video and synchronized robot motion."
        ),
        criterion_catalog=criteria,
        aliases={
            criterion_id: (
                _TITLES[criterion_id].lower(),
                criterion_id.replace("_", " "),
            )
            for criterion_id in GRS_CRITERIA
        },
        allowed_tools={
            "locate_evaluation_window",
            "sample_timestamps",
            "criterion_visual_evidence",
            "kinematic_temporal_facts",
            "stable_evidence_aggregation",
            "explicit_verifier",
        },
        window_spec=EvaluationWindowSpec(
            locator_tool="locate_evaluation_window",
            start_anchor="trial_start",
            end_anchor="trial_end",
            include_start=True,
            include_end=True,
            attributes={"scope": "complete_trial", "labels_required": False},
        ),
        default_temporal_policy=TemporalPolicy(
            operator="stable_state",
            target_state="satisfied",
            parameters={
                "smoothing_seconds": 12.0,
                "on_threshold": 0.60,
                "off_threshold": 0.45,
                "min_stable_seconds": 10.0,
                "max_gap_seconds": 8.0,
            },
        ),
        default_logic="AND",
        dataset_adapter=JigsawsSkillAssessmentAdapter(),
        metadata={
            "domain": "robotic_surgery",
            "dataset": "JIGSAWS",
            "task": "Suturing",
            "evaluation_unit": "complete_trial",
            "primary_ground_truth": "six_expert_modified_GRS_items",
            "self_reported_skill_is_auxiliary_only": True,
            "gesture_transcriptions_are_evaluation_or_training_only": True,
            "recommended_split": "leave_one_subject_out",
            "foundation_mllm_finetuned": False,
        },
    )


def create_jigsaws_suturing_v2_package() -> ProceduralTaskPackageV2:
    return ProceduralTaskPackageV2(
        task_id="jigsaws_suturing_skill",
        skill_id="jigsaws_suturing_grs_skill",
        criterion_ids=GRS_CRITERIA,
        plugin_slots=(
            EvidencePluginSlot("video_motion_and_tool_facts", "visual"),
            EvidencePluginSlot("robot_kinematic_motion_facts", "temporal"),
        ),
        sampled_frame_count=18,
    )

