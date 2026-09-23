"""Fact-only JIGSAWS kinematic evidence for the frozen foundation MLLM."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..grounded_evidence import GroundedEvidenceFact, grounded_fact_payload
from ..mllm_orchestration import CriterionContract, PluginEvidence
from .jigsaws_data import GRS_CRITERIA, JIGSAWS_FPS


_CRITERION_TITLES = {
    "respect_for_tissue": "Respect for tissue",
    "suture_needle_handling": "Suture and needle handling",
    "time_and_motion": "Time and motion",
    "flow_of_operation": "Flow of operation",
    "overall_performance": "Overall performance",
    "quality_of_final_product": "Quality of final product",
}

_CRITERION_DESCRIPTIONS = {
    "respect_for_tissue": "Movements avoid unnecessary force, repeated contact, and rough manipulation.",
    "suture_needle_handling": "Needle and suture are handled securely, deliberately, and with suitable control.",
    "time_and_motion": "The trial uses purposeful motion without avoidable travel or long hesitation.",
    "flow_of_operation": "The procedure progresses coherently with limited interruption or repetition.",
    "overall_performance": "The complete execution is controlled, efficient, and independently performed.",
    "quality_of_final_product": "The visible completed suture has suitable placement, form, and consistency.",
}


def skill_criterion_contracts() -> list[CriterionContract]:
    return [
        CriterionContract(
            criterion_id=criterion_id,
            title=_CRITERION_TITLES[criterion_id],
            minimal_description=_CRITERION_DESCRIPTIONS[criterion_id],
        )
        for criterion_id in GRS_CRITERIA
    ]


def _validate_kinematics(values: np.ndarray, fps: float, bin_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < bin_count or array.shape[1] != 76:
        raise ValueError("JIGSAWS fact adapter expects a [T,76] array with T >= bin_count")
    if not np.isfinite(array).all() or fps <= 0 or bin_count < 1:
        raise ValueError("Invalid JIGSAWS kinematic fact input")
    return array


def _arm_signals(array: np.ndarray, base: int, fps: float) -> dict[str, np.ndarray]:
    position = array[:, base:base + 3]
    recorded_velocity = array[:, base + 12:base + 15]
    speed = np.linalg.norm(recorded_velocity, axis=1)
    acceleration = np.diff(recorded_velocity, axis=0, prepend=recorded_velocity[:1]) * fps
    jerk = np.diff(acceleration, axis=0, prepend=acceleration[:1]) * fps
    gripper = array[:, base + 18]
    return {
        "position": position,
        "speed": speed,
        "jerk": np.linalg.norm(jerk, axis=1),
        "gripper_rate": np.abs(np.diff(gripper, prepend=gripper[:1])) * fps,
    }


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _arm_summary(signals: dict[str, np.ndarray]) -> dict[str, Any]:
    position = signals["position"]
    speed = signals["speed"]
    path_length = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
    displacement = float(np.linalg.norm(position[-1] - position[0]))
    p95_speed = float(np.percentile(speed, 95))
    adaptive_idle_threshold = max(1e-12, 0.05 * p95_speed)
    return {
        "path_length_dataset_units": _rounded(path_length),
        "endpoint_displacement_dataset_units": _rounded(displacement),
        "displacement_to_path_ratio": _rounded(displacement / path_length) if path_length else 0.0,
        "mean_speed_dataset_units_per_s": _rounded(np.mean(speed)),
        "p95_speed_dataset_units_per_s": _rounded(p95_speed),
        "speed_coefficient_of_variation": _rounded(
            np.std(speed) / np.mean(speed)
        ) if float(np.mean(speed)) > 0 else 0.0,
        "adaptive_low_motion_fraction": _rounded(np.mean(speed <= adaptive_idle_threshold)),
        "p95_jerk_dataset_units_per_s3": _rounded(np.percentile(signals["jerk"], 95)),
        "p95_absolute_gripper_rate_per_s": _rounded(
            np.percentile(signals["gripper_rate"], 95)
        ),
    }


def kinematic_temporal_fact_payload(
    values: np.ndarray,
    *,
    sample_id: str,
    fps: float = JIGSAWS_FPS,
    bin_count: int = 6,
) -> dict[str, Any]:
    """Summarize raw robot motion without emitting a skill score or verdict."""
    array = _validate_kinematics(values, fps, bin_count)
    left = _arm_signals(array, 38, fps)
    right = _arm_signals(array, 57, fps)
    duration_s = array.shape[0] / fps
    facts: list[GroundedEvidenceFact] = [
        GroundedEvidenceFact(
            fact_id="trial_duration",
            fact_type="attribute",
            subject="trial",
            predicate="observed_duration_seconds",
            value=_rounded(duration_s),
            grounding={
                "source": "synchronized_robot_kinematics",
                "sample_id": sample_id,
                "sample_count": int(array.shape[0]),
                "sampling_rate_hz": float(fps),
                "not_a_skill_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="left_arm_motion_summary",
            fact_type="attribute",
            subject="slave_left_tool",
            predicate="motion_summary",
            value=_arm_summary(left),
            grounding={
                "source": "jigsaws_columns_39_to_57",
                "fallible_observation": True,
                "not_a_skill_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id="right_arm_motion_summary",
            fact_type="attribute",
            subject="slave_right_tool",
            predicate="motion_summary",
            value=_arm_summary(right),
            grounding={
                "source": "jigsaws_columns_58_to_76",
                "fallible_observation": True,
                "not_a_skill_verdict": True,
            },
        ),
    ]

    left_speed = left["speed"]
    right_speed = right["speed"]
    correlation: float | None = None
    if float(np.std(left_speed)) > 0 and float(np.std(right_speed)) > 0:
        correlation = _rounded(np.corrcoef(left_speed, right_speed)[0, 1])
    facts.append(GroundedEvidenceFact(
        fact_id="bimanual_activity_relation",
        fact_type="relation",
        subject="slave_left_and_right_tools",
        predicate="speed_signal_correlation",
        value=correlation if correlation is not None else "undefined_constant_signal",
        grounding={
            "source": "synchronized_robot_kinematics",
            "interpretation": "correlation_only_not_coordination_or_skill_label",
            "not_a_skill_verdict": True,
        },
    ))

    boundaries = np.linspace(0, array.shape[0], bin_count + 1, dtype=int)
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        combined_speed = (left_speed[start:end] + right_speed[start:end]) / 2.0
        facts.append(GroundedEvidenceFact(
            fact_id=f"activity_window_{index}",
            fact_type="event",
            subject="bimanual_tools",
            predicate="windowed_motion_activity",
            value={
                "mean_combined_speed_dataset_units_per_s": _rounded(np.mean(combined_speed)),
                "p95_combined_speed_dataset_units_per_s": _rounded(np.percentile(combined_speed, 95)),
                "left_mean_speed_dataset_units_per_s": _rounded(np.mean(left_speed[start:end])),
                "right_mean_speed_dataset_units_per_s": _rounded(np.mean(right_speed[start:end])),
            },
            start_s=_rounded(start / fps),
            end_s=_rounded(end / fps),
            grounding={
                "source": "uniform_window_over_synchronized_robot_kinematics",
                "window_index": index,
                "not_a_gesture_transcription": True,
                "not_a_skill_verdict": True,
            },
        ))

    return grounded_fact_payload(
        facts,
        modality="temporal",
        source_description=(
            "Synchronized robot kinematics are converted to transparent motion, "
            "smoothness, gripper-activity, and time-window facts. These observations "
            "do not contain gesture transcripts, expert scores, or a skill verdict."
        ),
        provenance={
            "adapter": "jigsaws_kinematic_temporal_facts_v1",
            "sample_id": sample_id,
            "fps": float(fps),
            "bin_count": int(bin_count),
            "ground_truth_transcription_used": False,
            "expert_grs_used": False,
            "self_reported_skill_used": False,
            "raw_task_verdict_exported": False,
        },
    )


def kinematic_plugin_evidence(
    values: np.ndarray,
    *,
    sample_id: str,
    fps: float = JIGSAWS_FPS,
    bin_count: int = 6,
) -> PluginEvidence:
    payload = kinematic_temporal_fact_payload(
        values, sample_id=sample_id, fps=fps, bin_count=bin_count,
    )
    plugin = PluginEvidence(
        plugin_id="jigsaws_kinematic_temporal_facts",
        plugin_kind="temporal",
        description="Fact-only motion evidence from synchronized robot kinematics.",
        payload=payload,
        foundation_model_parameters_updated=False,
    )
    plugin.validate_fact_only()
    return plugin

