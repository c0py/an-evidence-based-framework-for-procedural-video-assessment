"""Representation-only concise-output reminder for one rejected residual response."""
from __future__ import annotations

from typing import Any

from .mllm_consensus_residual_orchestration import ConsensusResidualJudgeRequest
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest


class ConsensusResidualLengthRetryPromptMixin:
    """Constrain verbosity without changing any image/evidence decision rule."""

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Length retry requires ConsensusResidualJudgeRequest")
        messages = super().build_messages(request, ablation)
        reminder = """
RETRY JSON-LENGTH AUDIT (representation only; every semantic rule is unchanged)
The prior completion was rejected before merge because excessive prose ended inside a JSON string.
Judge again from the supplied images and evidence. Return the same required schema, but obey these
strict representation limits: use key_findings=[]; case_summary at most 20 words; every
conflict_dispositions reason, residual_rows r item, and calibrated_fact_accounting reason at most
8 words; never repeat a sentence; use compact JSON with no indentation. Include every required
frame, disagreement cell, candidate partition, action, basis, and fact-accounting row. Silently
verify that the final character closes the top-level JSON object. These limits do not favor any
hypothesis and do not change the consensus prior, residual action, or residual bound.
"""
        messages[0]["content"].append({"type": "text", "text": reminder})
        return messages
