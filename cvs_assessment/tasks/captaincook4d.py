"""CaptainCook4D Task Package and fact-only plugin adapters.

This module translates the bounded pilot artifacts into the shared grounded
evidence schema.  It never reads or represents a correct/error target.
"""

from __future__ import annotations

import re
from typing import Any

from ..grounded_evidence import GroundedEvidenceFact, grounded_fact_payload
from ..mllm_orchestration import PluginEvidence


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def skill_step_for_case(
    task_package: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the label-free SOP requirement by recipe and canonical text."""
    if task_package.get("test_target_labels_accessed") is not False:
        raise ValueError("Task Package does not prove the test-label firewall")
    structured = task_package["skill_sources"]["structured_skill_package"]
    recipe = structured["recipes"].get(str(case["activity_slug"]))
    if recipe is None:
        raise ValueError(f"Unsupported CaptainCook4D recipe: {case['activity_slug']}")
    target = _normalized_text(case["canonical_step_description"])
    matches = [
        row for row in recipe["steps"]
        if _normalized_text(row["canonical_requirement"]) == target
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one SOP match for {case['sample_key']}, found {len(matches)}"
        )
    return matches[0]


def skill_text_for_case(
    task_package: dict[str, Any], case: dict[str, Any]
) -> str:
    step = skill_step_for_case(task_package, case)
    unknown = task_package["skill_sources"]["structured_skill_package"][
        "unknown_policy"
    ]
    lines = [
        "Structured SOP/Skill for the expected step:",
        f"- Canonical requirement: {step['canonical_requirement']}",
        f"- Required visible action: {step['required_action']}",
        f"- Required objects: {', '.join(step['required_objects']) or 'none specified'}",
        f"- Candidate tools: {', '.join(step['candidate_tools']) or 'none specified'}",
        "- Observable completion cues: "
        + ("; ".join(step["observable_completion_cues"]) or "none specified"),
        "- Nonvisual or weak requirements: "
        + ("; ".join(step["nonvisual_or_weak_requirements"]) or "none"),
        f"- RGB observability: {step['rgb_observability']}",
        "Unknown policy: exact quantity, temperature, internal state, events outside the "
        "window, and insufficient coverage must remain unknown unless directly supported.",
        "This Skill describes what should happen. It contains no observation or answer for "
        "this recording.",
    ]
    if any(value not in unknown for value in ("exact_quantity", "temperature", "outside_window_event")):
        raise ValueError("Structured Skill unknown policy is incomplete")
    return "\n".join(lines)


def _validate_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("final_task_verdict") is not False:
        raise ValueError("CaptainCook4D fact bundle contains a final task verdict")
    forbidden = {
        "has_errors", "is_error", "error_probability", "verdict", "final_answer",
        "error_description", "modified_description", "official_error_logit",
        "official_error_category",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            overlap = forbidden & set(value)
            if overlap:
                raise ValueError(f"CaptainCook4D fact bundle leaked target keys: {sorted(overlap)}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(bundle)


def _plugin(
    plugin_id: str,
    plugin_kind: str,
    description: str,
    facts: list[GroundedEvidenceFact],
    provenance: dict[str, Any],
) -> PluginEvidence:
    plugin = PluginEvidence(
        plugin_id=plugin_id,
        plugin_kind=plugin_kind,
        description=description,
        payload=grounded_fact_payload(
            facts,
            modality=plugin_kind,
            source_description=description,
            provenance=provenance,
        ),
        foundation_model_parameters_updated=False,
    )
    plugin.validate_fact_only()
    return plugin


def visual_action_plugin(bundle: dict[str, Any]) -> PluginEvidence:
    """Export only calibrated observed action verbs from the visual head."""
    _validate_bundle(bundle)
    facts = []
    for row in bundle["facts"]:
        if row["fact_type"] != "action_verb" or row["polarity"] != "observed":
            continue
        facts.append(GroundedEvidenceFact(
            fact_id=str(row["fact_id"]),
            fact_type="event",
            subject="observed_cooking_window",
            predicate="predicted_visible_action_verb",
            value=str(row["value"]),
            confidence=float(row["confidence"]),
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            grounding={
                "source": "captaincook4d_action_verb_head_v1",
                "coverage_fraction": float(row["coverage_fraction"]),
                "fallible_observation": True,
                "not_a_compliance_verdict": True,
            },
        ))
    return _plugin(
        "captaincook4d_visual_action_facts_v1",
        "visual",
        "Calibrated visible action-verb observations; these are fallible and do not decide compliance.",
        facts,
        {"adapter": "captaincook4d_visual_action_facts_v1", "final_verdict_exported": False},
    )


def deterministic_temporal_plugin(
    case: dict[str, Any], bundle: dict[str, Any], total_feature_count: int
) -> PluginEvidence:
    """Export window timing/coverage without predicting correctness or a step label."""
    _validate_bundle(bundle)
    if total_feature_count < 1:
        raise ValueError("total_feature_count must be positive")
    window = bundle["observation_window"]
    start = float(window["requested_start_s"])
    end = float(window["requested_end_s"])
    midpoint_index = (
        int(case["feature_start_index_inclusive"])
        + int(case["feature_end_index_exclusive"])
    ) / 2.0
    relative = min(1.0, max(0.0, midpoint_index / total_feature_count))
    phase = "early" if relative < 1 / 3 else "middle" if relative < 2 / 3 else "late"
    facts = [
        GroundedEvidenceFact(
            fact_id=f"{case['sample_key']}:window_duration",
            fact_type="attribute",
            subject="observation_window",
            predicate="duration_seconds",
            value=round(max(0.0, end - start), 6),
            start_s=start,
            end_s=end,
            grounding={
                "source": "frozen_observation_window",
                "coverage_fraction": float(window["coverage_fraction"]),
                "not_a_compliance_verdict": True,
            },
        ),
        GroundedEvidenceFact(
            fact_id=f"{case['sample_key']}:relative_position",
            fact_type="attribute",
            subject="observation_window",
            predicate="relative_position_in_recording",
            value={"fraction": round(relative, 6), "coarse_phase": phase},
            start_s=start,
            end_s=end,
            grounding={
                "source": "feature_index_over_unlabeled_recording_length",
                "total_one_second_feature_vectors": int(total_feature_count),
                "not_a_step_identity_or_compliance_verdict": True,
            },
        ),
    ]
    return _plugin(
        "captaincook4d_window_temporal_facts_v1",
        "temporal",
        "Deterministic observation-window duration, coverage, and relative recording position.",
        facts,
        {"adapter": "captaincook4d_window_temporal_facts_v1", "labels_used": False},
    )


def fused_step_plugin(bundle: dict[str, Any]) -> PluginEvidence:
    """Export only calibrated observed v2 coarse-step hypotheses."""
    _validate_bundle(bundle)
    facts = []
    for row in bundle["facts"]:
        if (
            row["fact_type"] != "coarse_step_hypothesis"
            or int(row.get("rank", 1)) != 1
            or row["polarity"] != "observed"
        ):
            continue
        facts.append(GroundedEvidenceFact(
            fact_id=str(row["fact_id"]),
            fact_type="event",
            subject="observed_cooking_window",
            predicate="predicted_coarse_step",
            value=str(row["value"]),
            confidence=float(row["confidence"]),
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            grounding={
                "source": "captaincook4d_visual_action_temporal_fact_head_v2",
                "coverage_fraction": float(row["coverage_fraction"]),
                "fallible_observation": True,
                "uses_train_only_temporal_prior": True,
                "not_a_compliance_verdict": True,
            },
        ))
    return _plugin(
        "captaincook4d_fused_step_facts_v2",
        "visual_temporal",
        "Calibrated coarse-step observations from visual, action, and train-only temporal evidence.",
        facts,
        {"adapter": "captaincook4d_fused_step_facts_v2", "final_verdict_exported": False},
    )


def plugins_for_case(
    case: dict[str, Any], bundle: dict[str, Any], total_feature_count: int
) -> list[PluginEvidence]:
    return [
        visual_action_plugin(bundle),
        deterministic_temporal_plugin(case, bundle, total_feature_count),
        fused_step_plugin(bundle),
    ]
