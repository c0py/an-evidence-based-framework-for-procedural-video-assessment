"""Per-criterion calibrated-evidence dispositions for the frozen final Qwen."""
from __future__ import annotations

import json
from typing import Any

from .foundation_multibranch_arbitration import _validate_candidates
from .mllm_multibranch_orchestration import (
    MultibranchJudgeRequest,
    SymmetricMultibranchPromptMixin,
)
from .mllm_orchestration import FrozenMLLMJudge, MLLMAblation, MLLMJudgeRequest


def conflict_cells(request: MultibranchJudgeRequest) -> list[tuple[int, str]]:
    _, names, frame_ids, maps = _validate_candidates(request.foundation_candidate_judgments)
    criterion_ids = [row.criterion_id for row in request.criteria]
    output = []
    for frame_id in frame_ids:
        rows_by_name = {
            name: {row["criterion_id"]: row for row in maps[name][frame_id]["criteria"]}
            for name in names
        }
        for criterion_id in criterion_ids:
            signatures = {
                tuple(rows_by_name[name][criterion_id].get(key) for key in (
                    "foundation_state", "foundation_confidence", "visibility",
                    "probability_satisfied",
                )) for name in names
            }
            if len(signatures) > 1:
                output.append((frame_id, criterion_id))
    return output


def fact_ids(request: MultibranchJudgeRequest) -> tuple[set[str], set[str]]:
    all_ids, calibrated = set(), set()
    for plugin in request.plugin_evidence:
        for fact in plugin.payload.get("facts", []):
            fact_id = str(fact.get("fact_id", ""))
            if not fact_id: continue
            all_ids.add(fact_id)
            annotation = fact.get("adjudication") or {}
            value = fact.get("value")
            if annotation.get("reliability_tier") == "calibrated_high" or value == "high_reliability_candidate" or (isinstance(value, dict) and value.get("candidate_rank_band") == "high_reliability_candidate"):
                calibrated.add(fact_id)
    return all_ids, calibrated


class CalibratedEvidencePromptMixin(SymmetricMultibranchPromptMixin):
    def build_messages(self, request: MLLMJudgeRequest, ablation: MLLMAblation) -> list[dict[str, Any]]:
        if not isinstance(request, MultibranchJudgeRequest):
            raise TypeError("Calibrated multibranch judging requires MultibranchJudgeRequest")
        messages = super().build_messages(request, ablation)
        cells = [{"frame_index": frame_id, "criterion_id": criterion_id} for frame_id, criterion_id in conflict_cells(request)]
        _, calibrated = fact_ids(request)
        contract = f"""
CALIBRATED-EVIDENCE-PRESERVING PER-CRITERION CONTRACT
Conflicting criterion cells that require an explicit disposition:
{json.dumps(cells, separators=(",", ":"))}
Applicable calibrated-high fact IDs in this request:
{json.dumps(sorted(calibrated), separators=(",", ":"))}

Add a top-level JSON key criterion_conflict_dispositions with exactly one object for every listed
cell, in listed order. Each object is:
{{"frame_index":0,"criterion_id":"criterion","accepted_hypotheses":["mllm_skill_visual"],"rejected_hypotheses":["mllm_skill","mllm_skill_temporal"],"decisive_evidence_ids":["fact-id"],"calibrated_evidence_rejections":[{{"fact_id":"fact-id","basis":"direct_image_contradiction"}}],"reason":"short criterion-specific reason"}}
Accepted and rejected hypotheses must partition all candidates. decisive_evidence_ids may be empty
only when the real image itself is decisive. Any applicable calibrated-high fact is reliable for
its stated observation (not automatically for criterion satisfaction): you must consider it and
may reject that observation only with basis direct_image_contradiction, grounding_mismatch, or
applicability_mismatch. Poor visibility or your own uncertainty is not a rejection basis. Supported
repetition remains context rather than visual proof. Decide every final criterion row yourself.
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
        normalized = super().validate_response(
            value, request, available_plugin_ids, probability_prior,
            log_odds_step, decision_protocol,
        )
        if not isinstance(request, MultibranchJudgeRequest):
            raise TypeError("Calibrated response requires MultibranchJudgeRequest")
        expected = conflict_cells(request)
        rows = value.get("criterion_conflict_dispositions") if isinstance(value, dict) else None
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise RuntimeError("Need one calibrated disposition per conflicting criterion cell")
        names = {row["ablation"]["name"] for row in request.foundation_candidate_judgments}
        available_facts, calibrated = fact_ids(request)
        allowed_bases = {"direct_image_contradiction", "grounding_mismatch", "applicability_mismatch"}
        validated = []
        for (frame_id, criterion_id), row in zip(expected, rows):
            if not isinstance(row, dict) or int(row.get("frame_index", -1)) != frame_id or row.get("criterion_id") != criterion_id:
                raise RuntimeError("Criterion conflict identity/order differs")
            accepted = set(map(str, row.get("accepted_hypotheses", [])))
            rejected = set(map(str, row.get("rejected_hypotheses", [])))
            if accepted & rejected or accepted | rejected != names:
                raise RuntimeError("Criterion disposition must partition every hypothesis")
            decisive = list(map(str, row.get("decisive_evidence_ids", [])))
            if not set(decisive).issubset(available_facts):
                raise RuntimeError("Criterion disposition cites an unavailable fact")
            rejection_rows = row.get("calibrated_evidence_rejections", [])
            if not isinstance(rejection_rows, list):
                raise RuntimeError("Calibrated evidence rejections must be a list")
            seen = set()
            for rejection in rejection_rows:
                fact_id = str(rejection.get("fact_id", "")); basis = str(rejection.get("basis", ""))
                if fact_id not in calibrated or fact_id in seen or basis not in allowed_bases:
                    raise RuntimeError("Invalid calibrated-evidence rejection audit")
                seen.add(fact_id)
            reason = str(row.get("reason", "")).strip()
            if not reason: raise RuntimeError("Criterion disposition needs a reason")
            validated.append({**row, "decisive_evidence_ids": decisive})
        normalized["criterion_conflict_dispositions"] = validated
        return normalized


class CalibratedEvidenceFrozenMLLMJudge(
    CalibratedEvidencePromptMixin, FrozenMLLMJudge,
):
    """Concrete calibrated-evidence-preserving final frozen-Qwen judge."""
