"""Representation-only contract reminder for rejected residual responses."""
from __future__ import annotations

import json
from typing import Any

from .foundation_consensus_residual_arbitration import ROLE_NAMES
from .mllm_consensus_residual_orchestration import (
    ConsensusResidualJudgeRequest,
    disagreement_cells,
)
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest


class ConsensusResidualContractRetryPromptMixin:
    """Repeat exact enum and partition strings without prescribing decisions."""

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Contract retry requires ConsensusResidualJudgeRequest")
        messages = super().build_messages(request, ablation)
        cells = [
            {"frame_index": frame_id, "criterion_id": criterion_id}
            for frame_id, criterion_id in disagreement_cells(request)
        ]
        reminder = f"""
RETRY EXACT-STRING CONTRACT AUDIT (representation only)
The prior response was rejected before merge because one or more enum strings or candidate
partitions were noncanonical. Judge again from the unchanged images, Skill, facts, candidates, and
consensus prior. Do not copy the rejected response and do not change any decision rule.

Use only these exact candidate strings in every h vector and conflict_dispositions partition:
{json.dumps(list(ROLE_NAMES), separators=(",", ":"))}
Every conflict_dispositions row must place each exact candidate string once across accepted plus
rejected, with no aliases, descriptions, omissions, duplicates, or extra strings.

The exact residual cells are:
{json.dumps(cells, separators=(",", ":"))}
For every listed cell, a must be exactly H, U, or D. Basis b must be exactly one of:
real_image_consensus, direct_image_support, grounded_fact_support, skill_applicability,
direct_image_contradiction, grounding_mismatch, applicability_mismatch. A D action may use only
direct_image_contradiction, grounding_mismatch, or applicability_mismatch. Default to H when the
unchanged evidence does not justify U or D. Silently verify all exact strings, partitions, aligned
array lengths, and required identities before closing the compact JSON object. This reminder
changes representation only; it does not favor either candidate or any residual action.
"""
        messages[0]["content"].append({"type": "text", "text": reminder})
        return messages
