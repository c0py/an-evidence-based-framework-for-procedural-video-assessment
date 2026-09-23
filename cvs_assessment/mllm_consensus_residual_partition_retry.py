"""Exact frame-level candidate partition reminder for residual retry."""
from __future__ import annotations

from typing import Any

from .foundation_consensus_residual_arbitration import ROLE_NAMES
from .mllm_consensus_residual_orchestration import ConsensusResidualJudgeRequest
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest


class ConsensusResidualPartitionRetryPromptMixin:
    """Require exact canonical candidate strings without changing decisions."""

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Partition retry requires its dedicated request")
        messages = super().build_messages(request, ablation)
        reminder = f"""
RETRY FRAME-CANDIDATE PARTITION AUDIT (representation only)
The prior response was rejected before merge only because at least one frame-level
conflict_dispositions candidate partition was noncanonical. In every such row, use only these two
exact strings: "{ROLE_NAMES[0]}" and "{ROLE_NAMES[1]}". accepted_hypotheses plus
rejected_hypotheses must be disjoint and contain each exact string once. Do not use aliases, role
descriptions, shortened names, or omit either string. Preserve all concise-output limits, the exact
four-row calibrated-fact checklist, and every unchanged image/evidence/residual decision rule.
"""
        messages[0]["content"].append({"type": "text", "text": reminder})
        return messages
