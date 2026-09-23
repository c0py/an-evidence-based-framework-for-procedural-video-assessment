"""Task-neutral reliability annotations for fact-only MLLM evidence plugins."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .mllm_orchestration import PluginEvidence


TIERS = ("calibrated_high", "supported", "context_only")


def _contains_unverified(value: Any) -> bool:
    if isinstance(value, str):
        return "unverified" in value.lower()
    if isinstance(value, dict):
        return any(_contains_unverified(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_unverified(child) for child in value)
    return False


def _is_direct_grounded_observation(fact: dict[str, Any]) -> bool:
    """Separate inspectable observations from retrieval-only hypotheses."""
    grounding = fact.get("grounding") if isinstance(fact.get("grounding"), dict) else {}
    if _contains_unverified(fact.get("value")) or _contains_unverified(grounding):
        return False
    if grounding.get("requires_image_verification") is True:
        return False
    return bool(
        grounding.get("fallible_observation") is True
        and (
            "bbox_normalized_xyxy" in grounding
            or "source_video_frame_index" in grounding
            or fact.get("frame_index") is not None
            or fact.get("timestamp_s") is not None
            or fact.get("start_s") is not None
        )
    )


def fact_reliability_tier(fact: dict[str, Any]) -> tuple[str, list[str]]:
    """Assign an interface tier without inferring a task label or verdict."""
    grounding = fact.get("grounding") if isinstance(fact.get("grounding"), dict) else {}
    value = fact.get("value")
    reasons: list[str] = []
    if _contains_unverified(value):
        reasons.append("explicit_unverified_or_retrieval_only")
        return "context_only", reasons

    # A high-reliability candidate has already passed a calibration policy
    # fitted outside the target sample/fold. It remains advisory, not truth.
    if isinstance(value, str) and value == "high_reliability_candidate":
        reasons.append("explicit_crossfit_high_reliability_band")
        return "calibrated_high", reasons
    if isinstance(value, dict) and value.get("candidate_rank_band") == "high_reliability_candidate":
        reasons.append("explicit_crossfit_high_reliability_band")
        return "calibrated_high", reasons

    if grounding.get("requires_image_verification") is True:
        reasons.append("requires_image_verification_without_calibrated_high_band")
        return "context_only", reasons
    if _contains_unverified(grounding):
        reasons.append("grounding_marks_unverified")
        return "context_only", reasons
    confidence = fact.get("confidence")
    support_count = grounding.get(
        "supporting_observation_count", grounding.get("supporting_frame_count"),
    )
    if support_count is not None and int(support_count) >= 3:
        reasons.append("at_least_three_temporal_supporting_observations")
        if confidence is None or float(confidence) >= 0.50:
            return "supported", reasons
    if confidence is not None:
        confidence = float(confidence)
        if confidence >= 0.75:
            reasons.append("high_raw_observation_confidence_but_not_task_calibrated")
            return "context_only", reasons
        if confidence < 0.35:
            reasons.append("low_raw_observation_confidence")
            return "context_only", reasons
        reasons.append("moderate_raw_observation_confidence_not_task_calibrated")
        return "context_only", reasons

    # Deterministic relations and summaries are useful context but have no
    # independently calibrated reliability signal.
    reasons.append("no_explicit_reliability_signal")
    return "context_only", reasons


def annotate_grounded_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy and annotate facts while preserving the fact-only safety contract."""
    if payload.get("schema_version") != "grounded_evidence_facts_v1":
        raise ValueError("Evidence adjudication requires grounded_evidence_facts_v1")
    if payload.get("final_task_prediction_provided") is not False:
        raise ValueError("Evidence adjudication cannot consume plugin task predictions")
    output = deepcopy(payload)
    counts = {tier: 0 for tier in TIERS}
    for fact in output.get("facts", []):
        if not isinstance(fact, dict):
            raise ValueError("Grounded evidence facts must be objects")
        tier, reasons = fact_reliability_tier(fact)
        counts[tier] += 1
        fact["adjudication"] = {
            "reliability_tier": tier,
            "reasons": reasons,
            "advisory_not_ground_truth": True,
            "may_supply_final_task_verdict": False,
            "requires_foundation_image_and_skill_adjudication": True,
        }
    provenance = output.setdefault("provenance", {})
    provenance["evidence_adjudication"] = {
        "policy": "task_neutral_fact_reliability_v1",
        "tier_counts": counts,
        "facts_removed": 0,
        "task_labels_accessed": False,
        "task_predictions_created": False,
    }
    return output


def adjudicated_plugin(plugin: PluginEvidence) -> PluginEvidence:
    return PluginEvidence(
        plugin_id=plugin.plugin_id,
        plugin_kind=plugin.plugin_kind,
        description=plugin.description,
        payload=annotate_grounded_payload(plugin.payload),
        foundation_model_parameters_updated=plugin.foundation_model_parameters_updated,
    )


def route_grounded_payload_for_decision(
    payload: dict[str, Any], *, maximum_context_facts: int = 8,
    retain_context_without_supported_anchor: bool = False,
    include_fact_annotations: bool = False,
) -> dict[str, Any]:
    """Keep decision-support facts and a bounded, diverse context tail.

    Context-only retrieval cues are not discarded from the upstream cache;
    they are omitted only from the final decision prompt after retrieval has
    already selected the submitted frames.
    """
    if maximum_context_facts < 0:
        raise ValueError("maximum_context_facts must be nonnegative")
    annotated = annotate_grounded_payload(payload)
    decision_facts = []
    context_facts = []
    for fact in annotated.get("facts", []):
        tier = fact["adjudication"]["reliability_tier"]
        if tier in {"calibrated_high", "supported"}:
            decision_facts.append(fact)
        else:
            context_facts.append(fact)

    # Deterministically retain diverse context across predicates/subjects and
    # chronology. This supplies orientation without allowing an unbounded list
    # of unverified retrieval peaks to dominate the foundation prompt.
    direct_context_facts = [
        fact for fact in context_facts if _is_direct_grounded_observation(fact)
    ]
    effective_context_limit = (
        maximum_context_facts
        if (
            decision_facts
            or retain_context_without_supported_anchor
            or direct_context_facts
        ) else 0
    )
    retained_context = []
    if effective_context_limit > 0:
        # Prefer observations tied to an inspectable frame/time/box. Retrieval
        # candidates without such grounding never displace direct evidence.
        ordered_context = direct_context_facts + [
            fact for fact in context_facts if fact not in direct_context_facts
        ]
        seen: set[tuple[str, str]] = set()
        for fact in ordered_context:
            key = (str(fact.get("subject", "")), str(fact.get("predicate", "")))
            if key in seen:
                continue
            retained_context.append(fact)
            seen.add(key)
            if len(retained_context) >= effective_context_limit:
                break
        if len(retained_context) < effective_context_limit:
            retained_ids = {id(fact) for fact in retained_context}
            for fact in ordered_context:
                if id(fact) in retained_ids:
                    continue
                retained_context.append(fact)
                if len(retained_context) >= effective_context_limit:
                    break

    retained = decision_facts + retained_context

    def sort_key(fact: dict[str, Any]) -> tuple[float, int, str]:
        raw_time = fact.get("timestamp_s")
        if raw_time is None:
            raw_time = fact.get("start_s", -1.0)
        raw_frame = fact.get("frame_index", -1)
        return (
            float(raw_time), int(-1 if raw_frame is None else raw_frame),
            str(fact.get("fact_id", "")),
        )

    retained.sort(key=sort_key)
    original_count = len(annotated.get("facts", []))
    if not include_fact_annotations:
        for fact in retained:
            fact.pop("adjudication", None)
    annotated["facts"] = retained
    audit = annotated["provenance"]["evidence_adjudication"]
    audit.update({
        "routing_policy": "retain_all_calibrated_high_and_supported_plus_bounded_diverse_context_v1",
        "maximum_context_facts": maximum_context_facts,
        "retain_context_without_supported_anchor": retain_context_without_supported_anchor,
        "direct_grounded_context_available": bool(direct_context_facts),
        "effective_context_limit": effective_context_limit,
        "fact_annotations_in_prompt": include_fact_annotations,
        "decision_support_fact_count": len(decision_facts),
        "retained_context_fact_count": len(retained_context),
        "facts_retained": len(retained),
        "facts_removed_from_final_decision_prompt": original_count - len(retained),
        "upstream_cache_modified": False,
    })
    return annotated


def detach_evidence_adjudication_audit(
    routed_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep router diagnostics out of the MLLM prompt while preserving them separately."""
    output = deepcopy(routed_payload)
    provenance = output.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Routed grounded payload has no provenance object")
    audit = provenance.pop("evidence_adjudication", None)
    if not isinstance(audit, dict):
        raise ValueError("Routed grounded payload has no evidence-adjudication audit")
    return output, audit


def plugin_reliability_manifest(
    plugins: Iterable[PluginEvidence],
) -> list[dict[str, Any]]:
    """Expose evidence strength counts, never criterion/task probabilities."""
    output = []
    for plugin in plugins:
        annotated = annotate_grounded_payload(plugin.payload)
        counts = annotated["provenance"]["evidence_adjudication"]["tier_counts"]
        output.append({
            "plugin_id": plugin.plugin_id,
            "plugin_kind": plugin.plugin_kind,
            "fact_count": sum(counts.values()),
            "tier_counts": counts,
            "final_task_prediction_provided": False,
            "foundation_model_must_resolve_conflicts": True,
        })
    return output
