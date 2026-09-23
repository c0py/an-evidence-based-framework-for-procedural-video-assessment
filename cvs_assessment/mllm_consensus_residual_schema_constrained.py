"""True JSON-Schema decoding for the minimal consensus-residual protocol.

The frozen foundation model still authors every semantic decision.  This mixin
only removes redundant identifiers and complementary arrays from the wire
format, then deterministically expands them into the already-frozen legacy
contract before its validators and merger run.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from jsonschema import Draft202012Validator

from .foundation_consensus_residual_arbitration import ROLE_NAMES
from .mllm_consensus_residual_orchestration import (
    ConsensusResidualJudgeRequest,
    _applicable_high_facts,
    disagreement_cells,
)
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest


SCOPES = ("image_only", "direct_plugin", "interval_plugin", "whole_timeline", "mixed")
ALL_BASES = (
    "real_image_consensus", "direct_image_support", "grounded_fact_support",
    "skill_applicability", "direct_image_contradiction", "grounding_mismatch",
    "applicability_mismatch",
)
DOWNGRADE_BASES = (
    "direct_image_contradiction", "grounding_mismatch", "applicability_mismatch",
)


def _array(items: dict[str, Any], count: int) -> dict[str, Any]:
    return {"type": "array", "items": items, "minItems": count, "maxItems": count}


def _object(properties: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object", "properties": properties,
        "required": list(required), "additionalProperties": False,
    }


def _expected_facts(request: ConsensusResidualJudgeRequest) -> list[tuple[int, str]]:
    high = _applicable_high_facts(request)
    return [
        (frame_id, fact_id)
        for frame_id in request.frame_ids for fact_id in sorted(high[frame_id])
    ]


def build_minimal_schema(
    request: ConsensusResidualJudgeRequest, ablation: MLLMAblation,
) -> dict[str, Any]:
    """Build a request-specific schema without consulting labels or annotations."""
    criterion_ids = [row.criterion_id for row in request.criteria]
    criterion_count = len(criterion_ids)
    selected_plugins = [
        row.plugin_id for row in request.plugin_evidence
        if row.plugin_kind in set(ablation.include_plugin_kinds)
    ]
    available_facts = sorted({fact_id for _, fact_id in _expected_facts(request)})
    evidence_item = {"type": "string", "enum": available_facts} if available_facts else {"type": "string"}

    frame_prediction = _object({
        "s": _array({"type": "string", "enum": ["N", "P", "F", "U"]}, criterion_count),
        "c": _array({"type": "string", "enum": ["l", "m", "h"]}, criterion_count),
        "v": _array({"type": "string", "enum": ["g", "l", "p"]}, criterion_count),
    }, ("s", "c", "v"))
    finding = _object({
        "frame_index": {"type": "integer", "enum": list(request.frame_ids)},
        "criterion_id": {"type": "string", "enum": criterion_ids},
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    }, ("frame_index", "criterion_id", "reason"))
    conflict = _object({
        "accepted_role_mask": {"type": "integer", "enum": [0, 1, 2, 3]},
        "evidence_scope": {"type": "string", "enum": list(SCOPES)},
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    }, ("accepted_role_mask", "evidence_scope", "reason"))

    residual_common = {
        "accepted_role_mask": {"type": "integer", "enum": [0, 1, 2, 3]},
        "evidence_ids": {"type": "array", "items": evidence_item, "uniqueItems": True, "maxItems": len(available_facts)},
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    }
    residual_variants = []
    for action, bases in (("H", ALL_BASES), ("U", ALL_BASES), ("D", DOWNGRADE_BASES)):
        residual_variants.append(_object({
            **residual_common,
            "action": {"const": action},
            "basis": {"type": "string", "enum": list(bases)},
        }, ("action", "accepted_role_mask", "evidence_ids", "basis", "reason")))
    residual = {"anyOf": residual_variants}

    criterion_subset = {
        "type": "array", "items": {"type": "string", "enum": criterion_ids},
        "minItems": 1, "maxItems": criterion_count, "uniqueItems": True,
    }
    fact_common = {
        "applicable_criterion_ids": criterion_subset,
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    }
    fact = {"anyOf": [
        _object({**fact_common, "status": {"const": status}, "rejection_basis": {"type": "null"}},
                ("status", "applicable_criterion_ids", "rejection_basis", "reason"))
        for status in ("decisive", "context")
    ] + [
        _object({**fact_common, "status": {"const": "rejected"},
                 "rejection_basis": {"type": "string", "enum": list(DOWNGRADE_BASES)}},
                ("status", "applicable_criterion_ids", "rejection_basis", "reason"))
    ]}

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object({
            "frame_predictions": _array(frame_prediction, len(request.frame_ids)),
            "plugin_use": _array({"type": "boolean"}, len(selected_plugins)),
            "key_findings": {"type": "array", "items": finding, "maxItems": 8},
            "case_summary": {"type": "string", "minLength": 1, "maxLength": 600},
            "conflict_decisions": _array(conflict, len(request.frame_ids)),
            "residual_decisions": _array(residual, len(disagreement_cells(request))),
            "fact_decisions": _array(fact, len(_expected_facts(request))),
        }, (
            "frame_predictions", "plugin_use", "key_findings", "case_summary",
            "conflict_decisions", "residual_decisions", "fact_decisions",
        )),
    }
    Draft202012Validator.check_schema(schema)
    return schema


def _accepted(mask: int) -> list[str]:
    return [name for index, name in enumerate(ROLE_NAMES) if mask & (1 << index)]


def expand_minimal_response(
    value: dict[str, Any], request: ConsensusResidualJudgeRequest,
    ablation: MLLMAblation,
) -> dict[str, Any]:
    """Expand fixed-order minimal decisions without creating semantic content."""
    selected_plugins = [
        row.plugin_id for row in request.plugin_evidence
        if row.plugin_kind in set(ablation.include_plugin_kinds)
    ]
    frames = [
        {"frame_index": frame_id, **row}
        for frame_id, row in zip(request.frame_ids, value["frame_predictions"], strict=True)
    ]
    conflicts = []
    for frame_id, row in zip(request.frame_ids, value["conflict_decisions"], strict=True):
        accepted = _accepted(int(row["accepted_role_mask"]))
        conflicts.append({
            "frame_index": frame_id,
            "accepted_hypotheses": accepted,
            "rejected_hypotheses": [name for name in ROLE_NAMES if name not in accepted],
            "evidence_scope": row["evidence_scope"], "reason": row["reason"],
        })

    grouped: dict[int, dict[str, Any]] = {}
    cells = disagreement_cells(request)
    for (frame_id, criterion_id), row in zip(cells, value["residual_decisions"], strict=True):
        target = grouped.setdefault(frame_id, {
            "frame_index": frame_id, "criterion_ids": [], "a": [], "h": [],
            "e": [], "b": [], "r": [],
        })
        target["criterion_ids"].append(criterion_id)
        target["a"].append(row["action"])
        target["h"].append(_accepted(int(row["accepted_role_mask"])))
        target["e"].append(row["evidence_ids"])
        target["b"].append(row["basis"])
        target["r"].append(row["reason"])

    accounting = []
    for (frame_id, fact_id), row in zip(_expected_facts(request), value["fact_decisions"], strict=True):
        accounting.append({"frame_index": frame_id, "fact_id": fact_id, **row})
    return {
        "criterion_order": [row.criterion_id for row in request.criteria],
        "frame_predictions": frames,
        "plugin_assessment": [
            {"plugin_id": plugin_id, "used": used}
            for plugin_id, used in zip(selected_plugins, value["plugin_use"], strict=True)
        ],
        "key_findings": value["key_findings"],
        "case_summary": value["case_summary"],
        "conflict_dispositions": conflicts,
        "residual_rows": list(grouped.values()),
        "calibrated_fact_accounting": accounting,
    }


class ConsensusResidualSchemaConstrainedMixin:
    """Use true constrained decoding and deterministic legacy-contract expansion."""

    def response_format(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> dict[str, Any]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Schema-constrained residual judging requires its dedicated request")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "consensus_residual_minimal_v1", "strict": True,
                "schema": build_minimal_schema(request, ablation),
            },
        }

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Schema-constrained residual judging requires its dedicated request")
        messages = super().build_messages(request, ablation)
        cells = disagreement_cells(request)
        facts = _expected_facts(request)
        selected_plugins = [
            row.plugin_id for row in request.plugin_evidence
            if row.plugin_kind in set(ablation.include_plugin_kinds)
        ]
        contract = f"""
SCHEMA-CONSTRAINED MINIMAL SERIALIZATION (supersedes only the earlier JSON shape)
The semantic decision rules above are unchanged. The decoder enforces the minimal JSON Schema.
Do not emit criterion_order or repeated frame/plugin/fact identifiers; the fixed array orders are:
frame_predictions/conflict_decisions frames={json.dumps(request.frame_ids)}
plugin_use plugin IDs={json.dumps(selected_plugins)}
residual_decisions cells={json.dumps(cells)}
fact_decisions facts={json.dumps(facts)}
accepted_role_mask bits are 1={ROLE_NAMES[0]}, 2={ROLE_NAMES[1]}, 3=both, 0=neither.
Each residual decision still belongs entirely to you. H/U may use any frozen allowed basis; D can
only use direct_image_contradiction, grounding_mismatch, or applicability_mismatch. Fixed-order
identities and complementary rejected-role lists are expanded mechanically after your response.
"""
        messages[0]["content"].append({"type": "text", "text": contract})
        return messages

    def validate_response(
        self, value: Any, request: MLLMJudgeRequest,
        available_plugin_ids: set[str] | None = None,
        probability_prior: list[list[float]] | None = None,
        log_odds_step: float = 0.5,
        decision_protocol: str = "direct_probability",
    ) -> dict[str, Any]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Schema-constrained residual response requires its dedicated request")
        if not isinstance(value, dict):
            raise RuntimeError("Schema-constrained residual response must be an object")
        # The deployed protocol always uses this frozen ablation shape.  Recover
        # the selected plugin kinds from the validated caller input, not labels.
        ablation = MLLMAblation(
            "full_framework_consensus_residual", True, ("visual", "temporal"),
            decision_protocol=decision_protocol, require_fact_only_plugins=True,
        )
        schema = build_minimal_schema(request, ablation)
        Draft202012Validator(schema).validate(value)
        expanded = expand_minimal_response(value, request, ablation)
        normalized = super().validate_response(
            expanded, request, available_plugin_ids, probability_prior,
            log_odds_step, decision_protocol,
        )
        schema_sha = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        normalized["schema_constraint_audit"] = {
            "protocol": "consensus_residual_minimal_v1",
            "true_token_level_json_schema_constraint": True,
            "deterministic_identity_and_partition_expansion": True,
            "schema_sha256": schema_sha,
            "labels_accessed": False,
            "foundation_model_parameters_updated": False,
            "semantic_decisions_authored_by_frozen_foundation_mllm": True,
        }
        return normalized
