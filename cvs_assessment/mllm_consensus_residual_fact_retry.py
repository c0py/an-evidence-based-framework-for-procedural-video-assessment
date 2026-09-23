"""Exact calibrated-fact count reminder for a rejected residual response."""
from __future__ import annotations

import json
from typing import Any

from .mllm_consensus_residual_orchestration import (
    ConsensusResidualJudgeRequest,
    _applicable_high_facts,
)
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest


class ConsensusResidualFactCompletenessRetryPromptMixin:
    """Append an exact, representation-only calibrated-fact checklist."""

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, ConsensusResidualJudgeRequest):
            raise TypeError("Fact completeness retry requires its dedicated request")
        messages = super().build_messages(request, ablation)
        high = _applicable_high_facts(request)
        expected = [
            {"frame_index": frame_id, "fact_id": fact_id}
            for frame_id in request.frame_ids for fact_id in sorted(high[frame_id])
        ]
        reminder = f"""
RETRY CALIBRATED-FACT COMPLETENESS AUDIT (representation only)
The prior concise response parsed but was rejected before merge because calibrated_fact_accounting
did not have the exact required identities. Generate the judgment again; do not reuse the rejected
response. calibrated_fact_accounting must contain exactly {len(expected)} rows in this exact order:
{json.dumps(expected, separators=(",", ":"))}
Do not combine, omit, duplicate, or add fact rows. Each row still independently chooses decisive,
context, or rejected under the unchanged evidence rules. Keep every concise-output limit and every
consensus/residual decision rule unchanged. Silently verify the count and identities before closing
the JSON object.
"""
        messages[0]["content"].append({"type": "text", "text": reminder})
        return messages
