"""Frozen-Qwen residual decisions over a two-Qwen consensus prior."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .foundation_consensus_residual_arbitration import (
    ROLE_NAMES,
    build_consensus_prior,
)
from .foundation_multibranch_arbitration import _fact_support_tier
from .mllm_multibranch_orchestration import (
    MultibranchJudgeRequest,
    SymmetricMultibranchPromptMixin,
)
from .mllm_orchestration import FrozenMLLMJudge, MLLMAblation, MLLMJudgeRequest


def _applicable_high_facts(request: MLLMJudgeRequest) -> dict[int, set[str]]:
    output = {int(frame_id): set() for frame_id in request.frame_ids}
    time_by_frame = dict(zip(request.frame_ids, request.timestamps_s))
    for plugin in request.plugin_evidence:
        for fact in plugin.payload.get("facts", []):
            if _fact_support_tier(fact) != "calibrated_high":
                continue
            fact_id = str(fact.get("fact_id", ""))
            if not fact_id:
                continue
            for frame_id, timestamp in time_by_frame.items():
                applicable = (
                    fact.get("frame_index") is not None
                    and int(fact["frame_index"]) == int(frame_id)
                ) or (
                    fact.get("timestamp_s") is not None
                    and abs(float(fact["timestamp_s"]) - float(timestamp)) <= 1e-3
                ) or (
                    fact.get("start_s") is not None
                    and fact.get("end_s") is not None
                    and float(fact["start_s"]) <= float(timestamp) <= float(fact["end_s"])
                )
                if applicable:
                    output[int(frame_id)].add(fact_id)
    return output


def disagreement_cells(request: MultibranchJudgeRequest) -> list[tuple[int, str]]:
    """Return only cells on which the two complete frozen-Qwen roles differ."""
    first, second = request.foundation_candidate_judgments
    first_frames = {int(row["frame_index"]): row for row in first["prediction"]["frames"]}
    second_frames = {int(row["frame_index"]): row for row in second["prediction"]["frames"]}
    output = []
    for frame_id in request.frame_ids:
        left = {str(row["criterion_id"]): row for row in first_frames[frame_id]["criteria"]}
        right = {str(row["criterion_id"]): row for row in second_frames[frame_id]["criteria"]}
        for criterion in request.criteria:
            criterion_id = criterion.criterion_id
            keys = ("foundation_state", "foundation_confidence", "visibility", "probability_satisfied")
            if tuple(left[criterion_id].get(key) for key in keys) != tuple(right[criterion_id].get(key) for key in keys):
                output.append((frame_id, criterion_id))
    return output


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(_flatten_strings(item))
        return output
    return []


@dataclass
class ConsensusResidualJudgeRequest(MultibranchJudgeRequest):
    consensus_prior: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        expected = build_consensus_prior(self.foundation_candidate_judgments)
        if self.consensus_prior != expected:
            raise ValueError("Consensus prior is not the exact mean of the two frozen-Qwen roles")


class ConsensusResidualPromptMixin(SymmetricMultibranchPromptMixin):
    """Ask the final frozen Qwen for bounded semantic residual actions."""

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Consensus residual judging requires its dedicated request")
        messages = super().build_messages(request, ablation)
        prior_rows = [
            {
                "frame_index": frame["frame_index"],
                "p": [row["consensus_probability"] for row in frame["criteria"]],
            }
            for frame in request.consensus_prior["frames"]
        ]
        high_facts = {
            str(frame_id): sorted(fact_ids)
            for frame_id, fact_ids in _applicable_high_facts(request).items()
        }
        criterion_ids = [row.criterion_id for row in request.criteria]
        cells_by_frame = {
            frame_id: [criterion_id for fid, criterion_id in disagreement_cells(request) if fid == frame_id]
            for frame_id in request.frame_ids
        }
        contract = f"""
FROZEN-QWEN CONSENSUS PRIOR AND BOUNDED RESIDUAL CONTRACT
The numerical prior below is the unweighted mean of exactly two complete hypotheses made by this
same frozen foundation model under role-defined evidence paths: perception-grounded and
evidence-adjudicated. It contains no raw small-model prediction and is not ground truth.
criterion_order={json.dumps(criterion_ids)}
consensus_prior_rows={json.dumps(prior_rows, separators=(",", ":"))}
applicable_calibrated_high_fact_ids_by_frame={json.dumps(high_facts, separators=(",", ":"))}

Your ordinal frame_predictions remain an independent semantic inspection, but the deployed final
score is controlled ONLY by the residual action you author below. The only cells requiring a
residual are {json.dumps(cells_by_frame, separators=(",", ":"))}. Add top-level key residual_rows
with exactly one row for every frame having listed cells, in frame order:
{{"frame_index":0,"criterion_ids":["id1","id2"],"a":["H","U"],
"h":[["mllm_skill_visual"],["mllm_skill_visual","full_framework_calibrated_multibranch"]],
"e":[[],["fact-id"]],"b":["real_image_consensus","grounded_fact_support"],
"r":["short reason","short reason"]}}
Arrays align with criterion_ids. a codes are H=hold, U=upgrade, D=downgrade; H implies residual 0,
U/D imply exactly 0.1. h gives accepted hypotheses; its complement is rejected. Default to H.
Upgrade or downgrade only when your inspection of the real image,
Skill applicability, or a grounded fact identifies a specific semantic error in the consensus.
Downgrade basis must be direct_image_contradiction, grounding_mismatch, or applicability_mismatch.
h may contain either, both, or neither of {json.dumps(list(ROLE_NAMES))}.

Also add calibrated_fact_accounting with exactly one row for every applicable calibrated-high fact
listed above, ordered by frame then fact ID:
{{"frame_index":0,"fact_id":"id","status":"context","applicable_criterion_ids":["id"],
"rejection_basis":null,"reason":"short reason"}}
status is decisive, context, or rejected. applicable_criterion_ids must be a nonempty subset of the
criterion order. rejected requires one of the three allowed rejection bases; otherwise it must be
null. A reliable observation need not prove criterion satisfaction, but it may not be silently
ignored. You remain the sole final semantic decision maker.
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
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Consensus residual response requires its dedicated request")
        criterion_ids = [row.criterion_id for row in request.criteria]
        expected_cells = disagreement_cells(request)
        expected_by_frame = {
            frame_id: [criterion_id for fid, criterion_id in expected_cells if fid == frame_id]
            for frame_id in request.frame_ids
        }
        expected_by_frame = {frame_id: ids for frame_id, ids in expected_by_frame.items() if ids}
        rows = value.get("residual_rows") if isinstance(value, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("Need compact residual rows")
        allowed_bases = {
            "real_image_consensus", "direct_image_support", "grounded_fact_support",
            "skill_applicability", "direct_image_contradiction", "grounding_mismatch",
            "applicability_mismatch",
        }
        downgrade_bases = {
            "direct_image_contradiction", "grounding_mismatch", "applicability_mismatch",
        }
        high_by_frame = _applicable_high_facts(request)
        available_facts = set().union(*high_by_frame.values()) if high_by_frame else set()
        validated = []
        normalization = {
            "empty_h_expanded_frame_ids": [],
            "invalid_h_to_accept_neither_cells": [],
            "misaligned_evidence_union_frame_ids": [],
            "misaligned_basis_selection_frame_ids": [],
            "singleton_reason_repeated_frame_ids": [],
            "missing_reason_from_frame_disposition_frame_ids": [],
            "misaligned_reason_from_frame_disposition_frame_ids": [],
        }
        frame_reason_by_id = {
            int(row["frame_index"]): str(row.get("reason", "")).strip()
            for row in normalized.get("conflict_dispositions", [])
        }
        action_names = {"H": "hold", "U": "upgrade", "D": "downgrade"}
        by_frame = {}
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("Each compact residual row must be an object")
            frame_id = int(row.get("frame_index", -1))
            if frame_id in by_frame:
                raise RuntimeError("Compact residual frame is duplicated")
            by_frame[frame_id] = row
        if not set(expected_by_frame).issubset(by_frame):
            raise RuntimeError("One or more disagreement frames lack a residual row")
        for frame_id, row in by_frame.items():
            if frame_id not in expected_by_frame and row.get("criterion_ids"):
                raise RuntimeError("A non-disagreement frame contains residual criteria")
        for expected_frame, expected_ids in expected_by_frame.items():
            row = by_frame[expected_frame]
            ids = list(map(str, row.get("criterion_ids", [])))
            arrays = [row.get(key) for key in ("a", "h", "e", "b", "r")]
            if arrays[1] == [] and ids:
                # An empty compact hypothesis vector has one conservative,
                # semantics-preserving interpretation: accept neither role in
                # every aligned cell.  Never infer a positive acceptance.
                arrays[1] = [[] for _ in ids]
                normalization["empty_h_expanded_frame_ids"].append(expected_frame)
            reasons = arrays[4]
            if (
                isinstance(reasons, list) and len(reasons) == len(ids)
                and all(str(reason).strip() for reason in reasons)
            ):
                arrays[4] = [str(reason).strip() for reason in reasons]
            elif (
                isinstance(reasons, list) and len(reasons) == 1
                and str(reasons[0]).strip()
            ):
                arrays[4] = [str(reasons[0]).strip() for _ in ids]
                normalization["singleton_reason_repeated_frame_ids"].append(expected_frame)
            elif isinstance(reasons, str) and reasons.strip():
                arrays[4] = [reasons.strip() for _ in ids]
                normalization["singleton_reason_repeated_frame_ids"].append(expected_frame)
            else:
                fallback_reason = frame_reason_by_id.get(expected_frame, "")
                if not fallback_reason:
                    raise RuntimeError("Residual reason is absent and no Qwen frame reason is available")
                arrays[4] = [fallback_reason for _ in ids]
                key = (
                    "missing_reason_from_frame_disposition_frame_ids"
                    if reasons is None or reasons == []
                    else "misaligned_reason_from_frame_disposition_frame_ids"
                )
                normalization[key].append(expected_frame)
            if (len(ids) != len(set(ids)) or not set(ids).issubset(criterion_ids)
                    or not isinstance(arrays[0], list) or len(arrays[0]) != len(ids)
                    or not set(expected_ids).issubset(ids)):
                raise RuntimeError(
                    f"Compact residual action array differs from disagreement cells: "
                    f"frame={expected_frame}, expected={expected_ids}, observed={ids}, "
                    f"lengths={[len(items) if isinstance(items, list) else None for items in arrays]}"
                )
            hypotheses_aligned = isinstance(arrays[1], list) and len(arrays[1]) == len(ids)
            evidence_aligned = (
                isinstance(arrays[2], list) and len(arrays[2]) == len(ids)
                and all(isinstance(item, list) for item in arrays[2])
            )
            basis_aligned = isinstance(arrays[3], list) and len(arrays[3]) == len(ids)
            evidence_union = [
                item for item in dict.fromkeys(_flatten_strings(arrays[2]))
                if item in available_facts
            ]
            available_bases = [
                item for item in _flatten_strings(arrays[3]) if item in allowed_bases
            ]
            if not evidence_aligned:
                normalization["misaligned_evidence_union_frame_ids"].append(expected_frame)
            if not basis_aligned:
                normalization["misaligned_basis_selection_frame_ids"].append(expected_frame)
            positions = {criterion_id: index for index, criterion_id in enumerate(ids)}
            extra = set(ids) - set(expected_ids)
            for criterion_id in extra:
                if str(arrays[0][positions[criterion_id]]) != "H":
                    raise RuntimeError("An already-consistent criterion may only add a redundant H")
            for criterion_id in expected_ids:
                index = positions[criterion_id]
                code, reason = arrays[0][index], arrays[4][index]
                action = action_names.get(str(code)); step = 0.0 if action == "hold" else 0.1
                if action is None:
                    raise RuntimeError("Residual action code must be H, U, or D")
                hypotheses = arrays[1][index] if hypotheses_aligned else []
                accepted = set(map(str, hypotheses)) if isinstance(hypotheses, list) else set()
                if not accepted.issubset(ROLE_NAMES):
                    accepted = set()
                    normalization["invalid_h_to_accept_neither_cells"].append(
                        {"frame_index": expected_frame, "criterion_id": criterion_id}
                    )
                rejected = set(ROLE_NAMES) - accepted
                if evidence_aligned:
                    decisive = [
                        item for item in map(str, arrays[2][index]) if item in available_facts
                    ]
                else:
                    decisive = list(evidence_union)
                basis = str(arrays[3][index]) if basis_aligned else ""
                if basis not in allowed_bases:
                    candidates = [
                        item for item in available_bases
                        if action != "downgrade" or item in downgrade_bases
                    ]
                    basis = candidates[0] if candidates else ""
                if basis not in allowed_bases or (action == "downgrade" and basis not in downgrade_bases):
                    raise RuntimeError("Residual disposition has an invalid basis")
                if not str(reason).strip():
                    raise RuntimeError("Residual disposition needs a reason")
                validated.append({"frame_index": expected_frame, "criterion_id": criterion_id,
                                  "action": action, "residual_step": step,
                                  "accepted_hypotheses": [name for name in ROLE_NAMES if name in accepted],
                                  "rejected_hypotheses": [name for name in ROLE_NAMES if name in rejected],
                                  "decisive_evidence_ids": decisive, "basis": basis, "reason": str(reason)})

        expected_facts = [
            (frame_id, fact_id)
            for frame_id in request.frame_ids for fact_id in sorted(high_by_frame[frame_id])
        ]
        accounting = value.get("calibrated_fact_accounting") if isinstance(value, dict) else None
        if not isinstance(accounting, list) or len(accounting) != len(expected_facts):
            raise RuntimeError("Every applicable calibrated-high fact must be accounted exactly once")
        accounted = []
        for expected, row in zip(expected_facts, accounting):
            if not isinstance(row, dict) or (
                int(row.get("frame_index", -1)), str(row.get("fact_id", ""))
            ) != expected:
                raise RuntimeError("Calibrated fact accounting identity/order differs")
            status = str(row.get("status", ""))
            applicable = list(map(str, row.get("applicable_criterion_ids", [])))
            if status not in {"decisive", "context", "rejected"}:
                raise RuntimeError("Calibrated fact status is invalid")
            if not applicable or not set(applicable).issubset(criterion_ids):
                raise RuntimeError("Calibrated fact needs applicable criterion identities")
            rejection = row.get("rejection_basis")
            if (status == "rejected" and rejection not in downgrade_bases) or (
                status != "rejected" and rejection is not None
            ):
                raise RuntimeError("Calibrated fact rejection basis is invalid")
            if not str(row.get("reason", "")).strip():
                raise RuntimeError("Calibrated fact accounting needs a reason")
            accounted.append(row)
        normalized["residual_dispositions"] = validated
        normalized["calibrated_fact_accounting"] = accounted
        normalized["residual_contract_normalization"] = {
            "method": "conservative_compact_vector_alignment_v2",
            **normalization,
            "action_or_reason_changed": False,
            "positive_hypothesis_acceptance_inferred": False,
            "unavailable_evidence_id_retained": False,
            "new_semantic_content_created": False,
        }
        return normalized


class ConsensusResidualFrozenMLLMJudge(
    ConsensusResidualPromptMixin, FrozenMLLMJudge,
):
    """Concrete task-neutral consensus-residual frozen foundation judge."""
