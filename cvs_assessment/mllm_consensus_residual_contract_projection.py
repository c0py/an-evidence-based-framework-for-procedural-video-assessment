"""Label-free response-contract projection for frozen-Qwen residual outputs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import time
from typing import Any
from urllib.request import Request, urlopen

from .foundation_consensus_residual_arbitration import ROLE_NAMES
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest, _strip_json_fence


ALLOWED_BASES = {
    "real_image_consensus",
    "direct_image_support",
    "grounded_fact_support",
    "skill_applicability",
    "direct_image_contradiction",
    "grounding_mismatch",
    "applicability_mismatch",
}


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def canonical_action(value: Any) -> str | None:
    token = _token(value)
    aliases = {
        "h": "H", "hold": "H", "held": "H", "keep": "H", "k": "H",
        "no_change": "H", "unchanged": "H",
        "u": "U", "upgrade": "U", "increase": "U", "increased": "U",
        "d": "D", "downgrade": "D", "decrease": "D", "decreased": "D",
    }
    return aliases.get(token)


def canonical_basis(value: Any) -> str | None:
    token = _token(value)
    if token in ALLOWED_BASES:
        return token
    aliases = {
        "image_consensus": "real_image_consensus",
        "visual_consensus": "real_image_consensus",
        "consensus": "real_image_consensus",
        "image_support": "direct_image_support",
        "visual_support": "direct_image_support",
        "direct_visual_support": "direct_image_support",
        "fact_support": "grounded_fact_support",
        "grounded_support": "grounded_fact_support",
        "skill_support": "skill_applicability",
        "applicability_support": "skill_applicability",
        "image_contradiction": "direct_image_contradiction",
        "visual_contradiction": "direct_image_contradiction",
        "direct_visual_contradiction": "direct_image_contradiction",
        "grounded_fact_mismatch": "grounding_mismatch",
        "fact_mismatch": "grounding_mismatch",
        "skill_mismatch": "applicability_mismatch",
        "skill_applicability_mismatch": "applicability_mismatch",
    }
    if token in aliases:
        return aliases[token]
    for allowed in ALLOWED_BASES:
        if allowed in token:
            return allowed
    return None


def canonical_role_values(values: Any) -> list[str] | None:
    if not isinstance(values, list):
        return None
    output: list[str] = []
    for value in values:
        token = _token(value)
        if token in {"", "none", "neither", "no_hypothesis", "no_hypotheses"}:
            continue
        if token in {"all", "both", "both_hypotheses", "all_hypotheses"}:
            candidates = list(ROLE_NAMES)
        elif token == ROLE_NAMES[0] or (
            "visual" in token and "calibrated" not in token
        ) or token in {"perception_grounded", "visual_qwen"}:
            candidates = [ROLE_NAMES[0]]
        elif token == ROLE_NAMES[1] or "calibrated" in token or token in {
            "evidence_adjudicated", "full_framework", "calibrated_qwen",
        }:
            candidates = [ROLE_NAMES[1]]
        else:
            return None
        for candidate in candidates:
            if candidate not in output:
                output.append(candidate)
    return output


def deterministic_contract_projection(value: Any) -> tuple[Any, list[dict[str, Any]]]:
    projected = deepcopy(value)
    changes: list[dict[str, Any]] = []
    if not isinstance(projected, dict):
        return projected, changes
    for row_index, row in enumerate(projected.get("residual_rows", [])):
        if not isinstance(row, dict):
            continue
        actions = row.get("a")
        if isinstance(actions, list):
            for index, before in enumerate(list(actions)):
                after = canonical_action(before)
                if after is not None and after != before:
                    actions[index] = after
                    changes.append({"path": f"residual_rows[{row_index}].a[{index}]", "before": before, "after": after})
        bases = row.get("b")
        if isinstance(bases, list):
            for index, before in enumerate(list(bases)):
                after = canonical_basis(before)
                if after is not None and after != before:
                    bases[index] = after
                    changes.append({"path": f"residual_rows[{row_index}].b[{index}]", "before": before, "after": after})
    for row_index, row in enumerate(projected.get("conflict_dispositions", [])):
        if not isinstance(row, dict):
            continue
        for key in ("accepted_hypotheses", "rejected_hypotheses"):
            before = row.get(key)
            after = canonical_role_values(before)
            if after is not None and after != before:
                row[key] = after
                changes.append({"path": f"conflict_dispositions[{row_index}].{key}", "before": before, "after": after})
    return projected, changes


def _actions_by_cell(value: dict[str, Any]) -> dict[tuple[int, str], str | None]:
    output = {}
    for row in value.get("residual_rows", []):
        if not isinstance(row, dict):
            continue
        frame_id = int(row.get("frame_index", -1))
        ids, actions = row.get("criterion_ids"), row.get("a")
        if not isinstance(ids, list) or not isinstance(actions, list):
            continue
        for criterion_id, action in zip(ids, actions):
            output[(frame_id, str(criterion_id))] = canonical_action(action)
    return output


class ConsensusResidualContractProjectionMixin:
    """Project serialization aliases, with same-Qwen audit-field repair as fallback."""

    _repairable_errors = {
        "Residual action code must be H, U, or D",
        "Residual disposition has an invalid basis",
        "Conflict disposition must partition every candidate",
    }

    def _repair_contract_sections(
        self, value: dict[str, Any], request: MLLMJudgeRequest, trigger: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        sections = {
            "conflict_dispositions": value.get("conflict_dispositions"),
            "residual_rows": value.get("residual_rows"),
        }
        prompt = f"""You are the same frozen Qwen final judge repairing ONLY serialization audit fields from your immediately preceding answer.
The parsed answer failed this exact validator message: {trigger}

Return one JSON object containing exactly conflict_dispositions and residual_rows. Preserve all
frame identities, criterion identities, reasons, evidence IDs, and the semantic direction of every
residual action. Canonical action equivalents are hold/keep/K -> H, upgrade/increase -> U, and
downgrade/decrease -> D. Basis must use exactly one of {json.dumps(sorted(ALLOWED_BASES))}.
Accepted and rejected hypotheses must be a disjoint complete partition of
{json.dumps(list(ROLE_NAMES))} using only those exact strings. Correct only enum spelling,
equivalent basis categorization, and candidate partition serialization. These fields are audit
metadata; do not make a new image judgment and do not add or remove residual cells.

Original sections:
{json.dumps(sections, ensure_ascii=False, separators=(",", ":"))}
"""
        payload = {
            "model": self.model,
            # The local Qwen3-VL processor requires the canonical content-item
            # list even for a text-only follow-up turn.
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        started = time.monotonic()
        http_request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urlopen(http_request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        raw = body["choices"][0]["message"]["content"]
        repaired = json.loads(_strip_json_fence(raw))
        if set(repaired) != {"conflict_dispositions", "residual_rows"}:
            raise RuntimeError("Contract projection repair returned unexpected top-level keys")
        before_actions = _actions_by_cell(value)
        after_actions = _actions_by_cell(repaired)
        if before_actions != after_actions or any(action is None for action in before_actions.values()):
            raise RuntimeError("Contract projection attempted to change residual action semantics")
        output = deepcopy(value)
        output.update(repaired)
        return output, {
            "method": "same_frozen_qwen_text_only_audit_field_projection",
            "trigger": trigger,
            "latency_s": time.monotonic() - started,
            "usage": body.get("usage", {}),
            "original_sections_sha256": hashlib.sha256(
                json.dumps(sections, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "raw_repair_completion": raw,
            "residual_action_semantics_changed": False,
            "frame_predictions_changed": False,
        }

    def validate_response(
        self, value: Any, request: MLLMJudgeRequest,
        available_plugin_ids: set[str] | None = None,
        probability_prior: list[list[float]] | None = None,
        log_odds_step: float = 0.5,
        decision_protocol: str = "direct_probability",
    ) -> dict[str, Any]:
        projected, changes = deterministic_contract_projection(value)
        repair_audit = None
        try:
            normalized = super().validate_response(
                projected, request, available_plugin_ids, probability_prior,
                log_odds_step, decision_protocol,
            )
        except RuntimeError as exc:
            trigger = str(exc)
            if trigger not in self._repairable_errors or not isinstance(projected, dict):
                raise
            projected, repair_audit = self._repair_contract_sections(projected, request, trigger)
            projected, extra_changes = deterministic_contract_projection(projected)
            changes.extend(extra_changes)
            normalized = super().validate_response(
                projected, request, available_plugin_ids, probability_prior,
                log_odds_step, decision_protocol,
            )
        normalized["contract_projection_audit"] = {
            "schema_version": "frozen_qwen_contract_projection_v1",
            "deterministic_alias_changes": changes,
            "same_qwen_repair": repair_audit,
            "labels_accessed": False,
            "foundation_model_parameters_updated": False,
            "frame_predictions_changed": False,
            "residual_action_semantics_changed": False,
            "only_serialization_or_non_scoring_audit_fields_changed": True,
        }
        return normalized
