"""Branch-free fixed-object Schema protocol for LMFE 0.10.

This final pilot variant avoids both array-cardinality keywords and JSON-Schema
``anyOf``.  Coupled semantic enums (action+basis and status+rejection basis)
encode only combinations accepted by the frozen legacy validator.
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
    ALL_BASES, DOWNGRADE_BASES, SCOPES, _accepted, _expected_facts, _object,
)
from .mllm_consensus_residual_schema_constrained_v2 import (
    _fixed_map, _selected_plugin_ids,
)
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest


ACTION_BASIS = tuple(
    f"{action}::{basis}"
    for action, bases in (("H", ALL_BASES), ("U", ALL_BASES), ("D", DOWNGRADE_BASES))
    for basis in bases
)
STATUS_BASIS = (
    "decisive::none", "context::none",
    *(f"rejected::{basis}" for basis in DOWNGRADE_BASES),
)


def build_branch_free_schema(
    request: ConsensusResidualJudgeRequest, ablation: MLLMAblation,
) -> dict[str, Any]:
    criterion_ids = [row.criterion_id for row in request.criteria]
    criterion_decision = _object({
        "state": {"type": "string", "enum": ["N", "P", "F", "U"]},
        "confidence": {"type": "string", "enum": ["l", "m", "h"]},
        "visibility": {"type": "string", "enum": ["g", "l", "p"]},
    }, ("state", "confidence", "visibility"))
    per_frame = _fixed_map("c", [criterion_decision for _ in criterion_ids])
    conflict = _object({
        "accepted_role_mask": {"type": "integer", "enum": [0, 1, 2, 3]},
        "evidence_scope": {"type": "string", "enum": list(SCOPES)},
        "reason": {"type": "string", "minLength": 1, "maxLength": 180},
    }, ("accepted_role_mask", "evidence_scope", "reason"))
    residual = _object({
        "action_basis": {"type": "string", "enum": list(ACTION_BASIS)},
        "accepted_role_mask": {"type": "integer", "enum": [0, 1, 2, 3]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 180},
    }, ("action_basis", "accepted_role_mask", "reason"))
    fact = _object({
        "status_basis": {"type": "string", "enum": list(STATUS_BASIS)},
        "primary_criterion": {
            "type": "string", "enum": [f"c{i}" for i in range(len(criterion_ids))],
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 180},
    }, ("status_basis", "primary_criterion", "reason"))
    body = _object({
        "frame_decisions": _fixed_map("f", [per_frame for _ in request.frame_ids]),
        "plugin_use": _fixed_map(
            "p", [{"type": "boolean"} for _ in _selected_plugin_ids(request, ablation)],
        ),
        "case_summary": {"type": "string", "minLength": 1, "maxLength": 420},
        "conflict_decisions": _fixed_map("f", [conflict for _ in request.frame_ids]),
        "residual_decisions": _fixed_map(
            "r", [residual for _ in disagreement_cells(request)],
        ),
        "fact_decisions": _fixed_map("q", [fact for _ in _expected_facts(request)]),
    }, (
        "frame_decisions", "plugin_use", "case_summary", "conflict_decisions",
        "residual_decisions", "fact_decisions",
    ))
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **body}
    Draft202012Validator.check_schema(schema)
    return schema


def expand_branch_free_response(
    value: dict[str, Any], request: ConsensusResidualJudgeRequest,
    ablation: MLLMAblation,
) -> dict[str, Any]:
    criterion_ids = [row.criterion_id for row in request.criteria]
    frames = []
    for frame_position, frame_id in enumerate(request.frame_ids):
        source = value["frame_decisions"][f"f{frame_position}"]
        decisions = [source[f"c{i}"] for i in range(len(criterion_ids))]
        frames.append({
            "frame_index": frame_id,
            "s": [row["state"] for row in decisions],
            "c": [row["confidence"] for row in decisions],
            "v": [row["visibility"] for row in decisions],
        })
    plugin_ids = _selected_plugin_ids(request, ablation)
    plugins = [
        {"plugin_id": plugin_id, "used": value["plugin_use"][f"p{i}"]}
        for i, plugin_id in enumerate(plugin_ids)
    ]
    conflicts = []
    for i, frame_id in enumerate(request.frame_ids):
        row = value["conflict_decisions"][f"f{i}"]
        accepted = _accepted(int(row["accepted_role_mask"]))
        conflicts.append({
            "frame_index": frame_id, "accepted_hypotheses": accepted,
            "rejected_hypotheses": [name for name in ROLE_NAMES if name not in accepted],
            "evidence_scope": row["evidence_scope"], "reason": row["reason"],
        })
    grouped: dict[int, dict[str, Any]] = {}
    for i, (frame_id, criterion_id) in enumerate(disagreement_cells(request)):
        row = value["residual_decisions"][f"r{i}"]
        action, basis = row["action_basis"].split("::", 1)
        target = grouped.setdefault(frame_id, {
            "frame_index": frame_id, "criterion_ids": [], "a": [], "h": [],
            "e": [], "b": [], "r": [],
        })
        target["criterion_ids"].append(criterion_id)
        target["a"].append(action)
        target["h"].append(_accepted(int(row["accepted_role_mask"])))
        target["e"].append([])
        target["b"].append(basis)
        target["r"].append(row["reason"])
    accounting = []
    for i, (frame_id, fact_id) in enumerate(_expected_facts(request)):
        row = value["fact_decisions"][f"q{i}"]
        status, basis = row["status_basis"].split("::", 1)
        criterion_position = int(row["primary_criterion"][1:])
        accounting.append({
            "frame_index": frame_id, "fact_id": fact_id, "status": status,
            "applicable_criterion_ids": [criterion_ids[criterion_position]],
            "rejection_basis": None if basis == "none" else basis,
            "reason": row["reason"],
        })
    return {
        "criterion_order": criterion_ids, "frame_predictions": frames,
        "plugin_assessment": plugins, "key_findings": [],
        "case_summary": value["case_summary"],
        "conflict_dispositions": conflicts, "residual_rows": list(grouped.values()),
        "calibrated_fact_accounting": accounting,
    }


class ConsensusResidualSchemaConstrainedV3Mixin:
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
            raise TypeError("Branch-free schema judging requires its dedicated request")
        return {"type": "json_schema", "json_schema": {
            "name": "consensus_residual_branch_free_v3", "strict": True,
            "schema": build_branch_free_schema(request, ablation),
        }}

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Branch-free schema judging requires its dedicated request")
        messages = super().build_messages(request, ablation)
        mapping = {
            "frames": {f"f{i}": frame for i, frame in enumerate(request.frame_ids)},
            "criteria": {f"c{i}": row.criterion_id for i, row in enumerate(request.criteria)},
            "plugins": {f"p{i}": plugin for i, plugin in enumerate(_selected_plugin_ids(request, ablation))},
            "residuals": {f"r{i}": [frame, criterion] for i, (frame, criterion) in enumerate(disagreement_cells(request))},
            "facts": {f"q{i}": [frame, fact] for i, (frame, fact) in enumerate(_expected_facts(request))},
        }
        reminder = f"""
BRANCH-FREE FIXED-OBJECT SCHEMA v3 (supersedes only earlier response shapes)
Semantic rules remain unchanged. The required-key mapping is
{json.dumps(mapping, separators=(',', ':'))}
No arrays and no conditional schema branches are used. Choose action_basis as one exact combined
enum such as H::real_image_consensus or D::direct_image_contradiction. Choose status_basis as
decisive::none, context::none, or rejected::<allowed rejection basis>. primary_criterion is the
single most directly applicable c-key for that fact. accepted_role_mask bits remain
1={ROLE_NAMES[0]}, 2={ROLE_NAMES[1]}, 3=both, 0=neither. Keep every reason concise. All semantic
values remain authored by you as the frozen foundation model.
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
            raise RuntimeError("Branch-free schema response/request is invalid")
        ablation = self._ablation(decision_protocol)
        schema = build_branch_free_schema(request, ablation)
        Draft202012Validator(schema).validate(value)
        expanded = expand_branch_free_response(value, request, ablation)
        normalized = super().validate_response(
            expanded, request, available_plugin_ids, probability_prior,
            log_odds_step, decision_protocol,
        )
        schema_sha = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        normalized["schema_constraint_audit"] = {
            "protocol": "consensus_residual_branch_free_v3",
            "token_level_json_schema_constraint": True,
            "array_cardinality_keywords_avoided": True,
            "conditional_schema_branches_avoided": True,
            "fixed_identity_and_partition_expansion": True,
            "schema_sha256": schema_sha, "labels_accessed": False,
            "foundation_model_parameters_updated": False,
            "semantic_decisions_authored_by_frozen_foundation_mllm": True,
        }
        return normalized
