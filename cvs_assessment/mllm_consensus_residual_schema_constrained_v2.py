"""Fixed-object JSON-Schema protocol for consensus-residual arbitration.

LM Format Enforcer 0.10 does not reliably enforce JSON-Schema array cardinality.
This protocol therefore represents every fixed cohort (frames, plugins,
criteria, conflicts, residual cells, and facts) as an object with frozen,
required opaque keys.  The frozen Qwen still authors all semantic values; the
adapter restores request identities and complementary role partitions.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from jsonschema import Draft202012Validator

from .foundation_consensus_residual_arbitration import ROLE_NAMES
from .mllm_consensus_residual_orchestration import (
    ConsensusResidualJudgeRequest,
    disagreement_cells,
)
from .mllm_consensus_residual_schema_constrained import (
    ALL_BASES,
    DOWNGRADE_BASES,
    SCOPES,
    _accepted,
    _expected_facts,
    _object,
)
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest


def _fixed_map(prefix: str, item_schemas: list[dict[str, Any]]) -> dict[str, Any]:
    properties = {f"{prefix}{index}": schema for index, schema in enumerate(item_schemas)}
    return _object(properties, tuple(properties))


def _selected_plugin_ids(
    request: ConsensusResidualJudgeRequest, ablation: MLLMAblation,
) -> list[str]:
    allowed = set(ablation.include_plugin_kinds)
    return [row.plugin_id for row in request.plugin_evidence if row.plugin_kind in allowed]


def build_fixed_object_schema(
    request: ConsensusResidualJudgeRequest, ablation: MLLMAblation,
) -> dict[str, Any]:
    """Build the label-blind request-specific fixed-object schema."""
    criterion_ids = [row.criterion_id for row in request.criteria]
    criterion_keys = [f"c{index}" for index in range(len(criterion_ids))]
    cell_count = len(disagreement_cells(request))
    fact_count = len(_expected_facts(request))

    criterion_decision = _object({
        "state": {"type": "string", "enum": ["N", "P", "F", "U"]},
        "confidence": {"type": "string", "enum": ["l", "m", "h"]},
        "visibility": {"type": "string", "enum": ["g", "l", "p"]},
    }, ("state", "confidence", "visibility"))
    per_frame = _fixed_map("c", [criterion_decision for _ in criterion_ids])

    conflict = _object({
        "accepted_role_mask": {"type": "integer", "enum": [0, 1, 2, 3]},
        "evidence_scope": {"type": "string", "enum": list(SCOPES)},
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    }, ("accepted_role_mask", "evidence_scope", "reason"))

    residual_common = {
        "accepted_role_mask": {"type": "integer", "enum": [0, 1, 2, 3]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    }
    residual = {"anyOf": [
        _object({**residual_common, "action": {"const": action},
                 "basis": {"type": "string", "enum": list(bases)}},
                ("action", "accepted_role_mask", "basis", "reason"))
        for action, bases in (("H", ALL_BASES), ("U", ALL_BASES), ("D", DOWNGRADE_BASES))
    ]}

    criterion_use = _fixed_map(
        "c", [{"type": "boolean"} for _ in criterion_ids],
    )
    # Require at least one applicable criterion while retaining an arbitrary
    # nonempty subset.  This is an object-level constraint, not array length.
    criterion_use["anyOf"] = [
        {"properties": {key: {"const": True}}, "required": [key]}
        for key in criterion_keys
    ]
    fact_common = {
        "criterion_use": criterion_use,
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    }
    fact = {"anyOf": [
        _object({**fact_common, "status": {"const": status},
                 "rejection_basis": {"type": "null"}},
                ("status", "criterion_use", "rejection_basis", "reason"))
        for status in ("decisive", "context")
    ] + [
        _object({**fact_common, "status": {"const": "rejected"},
                 "rejection_basis": {"type": "string", "enum": list(DOWNGRADE_BASES)}},
                ("status", "criterion_use", "rejection_basis", "reason"))
    ]}

    body = _object({
        "frame_decisions": _fixed_map(
            "f", [per_frame for _ in request.frame_ids],
        ),
        "plugin_use": _fixed_map(
            "p", [{"type": "boolean"} for _ in _selected_plugin_ids(request, ablation)],
        ),
        "case_summary": {"type": "string", "minLength": 1, "maxLength": 600},
        "conflict_decisions": _fixed_map(
            "f", [conflict for _ in request.frame_ids],
        ),
        "residual_decisions": _fixed_map(
            "r", [residual for _ in range(cell_count)],
        ),
        "fact_decisions": _fixed_map(
            "q", [fact for _ in range(fact_count)],
        ),
    }, (
        "frame_decisions", "plugin_use", "case_summary",
        "conflict_decisions", "residual_decisions", "fact_decisions",
    ))
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **body}
    Draft202012Validator.check_schema(schema)
    return schema


def expand_fixed_object_response(
    value: dict[str, Any], request: ConsensusResidualJudgeRequest,
    ablation: MLLMAblation,
) -> dict[str, Any]:
    """Restore frozen request identities without inferring Qwen decisions."""
    criterion_ids = [row.criterion_id for row in request.criteria]
    frames = []
    for frame_position, frame_id in enumerate(request.frame_ids):
        source = value["frame_decisions"][f"f{frame_position}"]
        decisions = [source[f"c{index}"] for index in range(len(criterion_ids))]
        frames.append({
            "frame_index": frame_id,
            "s": [row["state"] for row in decisions],
            "c": [row["confidence"] for row in decisions],
            "v": [row["visibility"] for row in decisions],
        })

    plugin_ids = _selected_plugin_ids(request, ablation)
    plugins = [
        {"plugin_id": plugin_id, "used": value["plugin_use"][f"p{index}"]}
        for index, plugin_id in enumerate(plugin_ids)
    ]
    conflicts = []
    for index, frame_id in enumerate(request.frame_ids):
        row = value["conflict_decisions"][f"f{index}"]
        accepted = _accepted(int(row["accepted_role_mask"]))
        conflicts.append({
            "frame_index": frame_id, "accepted_hypotheses": accepted,
            "rejected_hypotheses": [name for name in ROLE_NAMES if name not in accepted],
            "evidence_scope": row["evidence_scope"], "reason": row["reason"],
        })

    grouped: dict[int, dict[str, Any]] = {}
    for index, (frame_id, criterion_id) in enumerate(disagreement_cells(request)):
        row = value["residual_decisions"][f"r{index}"]
        target = grouped.setdefault(frame_id, {
            "frame_index": frame_id, "criterion_ids": [], "a": [], "h": [],
            "e": [], "b": [], "r": [],
        })
        target["criterion_ids"].append(criterion_id)
        target["a"].append(row["action"])
        target["h"].append(_accepted(int(row["accepted_role_mask"])))
        # Evidence identity arrays are deliberately empty here. Every calibrated
        # high fact is still semantically accounted below, and this non-scoring
        # field is not allowed to invent or misidentify a fact.
        target["e"].append([])
        target["b"].append(row["basis"])
        target["r"].append(row["reason"])

    accounting = []
    for index, (frame_id, fact_id) in enumerate(_expected_facts(request)):
        row = value["fact_decisions"][f"q{index}"]
        applicable = [
            criterion_id for criterion_position, criterion_id in enumerate(criterion_ids)
            if row["criterion_use"][f"c{criterion_position}"] is True
        ]
        accounting.append({
            "frame_index": frame_id, "fact_id": fact_id, "status": row["status"],
            "applicable_criterion_ids": applicable,
            "rejection_basis": row["rejection_basis"], "reason": row["reason"],
        })
    return {
        "criterion_order": criterion_ids, "frame_predictions": frames,
        "plugin_assessment": plugins, "key_findings": [],
        "case_summary": value["case_summary"],
        "conflict_dispositions": conflicts, "residual_rows": list(grouped.values()),
        "calibrated_fact_accounting": accounting,
    }


class ConsensusResidualSchemaConstrainedV2Mixin:
    """Fixed-object constrained decoding plus deterministic legacy expansion."""

    @staticmethod
    def _ablation(decision_protocol: str = "ordinal_state") -> MLLMAblation:
        return MLLMAblation(
            "full_framework_consensus_residual", True, ("visual", "temporal"),
            decision_protocol=decision_protocol, require_fact_only_plugins=True,
        )

    def response_format(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> dict[str, Any]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Fixed-object schema judging requires its dedicated request")
        return {"type": "json_schema", "json_schema": {
            "name": "consensus_residual_fixed_object_v2", "strict": True,
            "schema": build_fixed_object_schema(request, ablation),
        }}

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Fixed-object schema judging requires its dedicated request")
        messages = super().build_messages(request, ablation)
        criteria = [row.criterion_id for row in request.criteria]
        frame_map = {f"f{i}": frame_id for i, frame_id in enumerate(request.frame_ids)}
        criterion_map = {f"c{i}": criterion_id for i, criterion_id in enumerate(criteria)}
        plugin_map = {
            f"p{i}": plugin_id
            for i, plugin_id in enumerate(_selected_plugin_ids(request, ablation))
        }
        cell_map = {
            f"r{i}": [frame_id, criterion_id]
            for i, (frame_id, criterion_id) in enumerate(disagreement_cells(request))
        }
        fact_map = {
            f"q{i}": [frame_id, fact_id]
            for i, (frame_id, fact_id) in enumerate(_expected_facts(request))
        }
        reminder = f"""
FIXED-OBJECT SCHEMA v2 (supersedes only all earlier response serialization shapes)
The semantic rules above are unchanged. Return only the decoder-enforced object. Every opaque key
below is required exactly once; do not output any arrays or repeat request identities.
frame/conflict map={json.dumps(frame_map, separators=(',', ':'))}
criterion map inside every frame={json.dumps(criterion_map, separators=(',', ':'))}
plugin map={json.dumps(plugin_map, separators=(',', ':'))}
residual map={json.dumps(cell_map, separators=(',', ':'))}
fact map={json.dumps(fact_map, separators=(',', ':'))}
accepted_role_mask bits: 1={ROLE_NAMES[0]}, 2={ROLE_NAMES[1]}, 3=both, 0=neither.
For fact criterion_use, set every c-key boolean and at least one true. Residual evidence IDs are
not serialized in v2; account for every calibrated-high fact in its required q-key instead.
All semantic states, masks, scopes, actions, bases, statuses, reasons, and the summary remain your
decisions as the frozen foundation model.
"""
        messages[0]["content"].append({"type": "text", "text": reminder})
        return messages

    def validate_response(
        self, value: Any, request: MLLMJudgeRequest,
        available_plugin_ids: set[str] | None = None,
        probability_prior: list[list[float]] | None = None,
        log_odds_step: float = 0.5,
        decision_protocol: str = "direct_probability",
    ) -> dict[str, Any]:
        if not isinstance(request, ConsensusResidualJudgeRequest) or not isinstance(value, dict):
            raise RuntimeError("Fixed-object schema response/request is invalid")
        ablation = self._ablation(decision_protocol)
        schema = build_fixed_object_schema(request, ablation)
        Draft202012Validator(schema).validate(value)
        expanded = expand_fixed_object_response(value, request, ablation)
        normalized = super().validate_response(
            expanded, request, available_plugin_ids, probability_prior,
            log_odds_step, decision_protocol,
        )
        schema_sha = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        normalized["schema_constraint_audit"] = {
            "protocol": "consensus_residual_fixed_object_v2",
            "token_level_json_schema_constraint": True,
            "array_cardinality_keywords_avoided": True,
            "fixed_identity_and_partition_expansion": True,
            "schema_sha256": schema_sha, "labels_accessed": False,
            "foundation_model_parameters_updated": False,
            "semantic_decisions_authored_by_frozen_foundation_mllm": True,
        }
        return normalized
