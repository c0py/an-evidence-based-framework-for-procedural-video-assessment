"""Generic completeness reminder for audited multibranch response retries."""
from __future__ import annotations

import json
from typing import Any

from .mllm_calibrated_multibranch_orchestration import conflict_cells
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest
from .mllm_multibranch_orchestration import MultibranchJudgeRequest


class MultibranchCompletenessRetryPromptMixin:
    """Append a representation-only count audit without changing decision rules."""

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, MultibranchJudgeRequest):
            raise TypeError("Completeness retry requires MultibranchJudgeRequest")
        messages = super().build_messages(request, ablation)
        frame_ids = [int(value) for value in request.frame_ids]
        cells = [
            {"frame_index": frame_id, "criterion_id": criterion_id}
            for frame_id, criterion_id in conflict_cells(request)
        ]
        reminder = f"""
RETRY RESPONSE-COMPLETENESS AUDIT (representation only; decision rules are unchanged)
The prior response for this sample was rejected before merge because at least one required
per-frame disposition was absent. Generate the judgment again from the supplied images and
evidence; do not reconstruct or reuse the rejected response.

Required conflict_dispositions frame IDs, exactly once and in this order:
{json.dumps(frame_ids, separators=(",", ":"))}
Required conflict_dispositions count: {len(frame_ids)}
Required criterion_conflict_dispositions count: {len(cells)}

A frame remains a conflict when candidates differ only in confidence, visibility, or probability,
even if their discrete states match. Do not omit such a frame. Before returning the JSON object,
silently count both arrays and verify their identities and order against the lists above. Keep all
existing calibrated-evidence rules, candidate partitions, final-Qwen decisions, and schema fields.
"""
        messages[0]["content"].append({"type": "text", "text": reminder})
        return messages

