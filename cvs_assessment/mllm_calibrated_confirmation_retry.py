"""Representation-only retry reminder for rejected calibrated responses."""
from __future__ import annotations

import json
from typing import Any

from .mllm_calibrated_multibranch_orchestration import conflict_cells, fact_ids
from .mllm_multibranch_orchestration import MultibranchJudgeRequest
from .mllm_orchestration import MLLMAblation, MLLMJudgeRequest


class CalibratedConfirmationRetryPromptMixin:
    """Require a complete, compact, schema-valid rendering of a new judgment."""

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, MultibranchJudgeRequest):
            raise TypeError("Calibrated confirmation retry requires MultibranchJudgeRequest")
        messages = super().build_messages(request, ablation)
        frame_ids = [int(value) for value in request.frame_ids]
        cells = [
            {"frame_index": frame_id, "criterion_id": criterion_id}
            for frame_id, criterion_id in conflict_cells(request)
        ]
        available, calibrated = fact_ids(request)
        reminder = f"""
RETRY RESPONSE-INTEGRITY AUDIT (representation only; semantic rules are unchanged)
The prior response was rejected before merge and scoring because its JSON representation was
truncated, incomplete, or violated an audit-field schema. Judge again from the supplied images,
candidates, Skill, and facts; do not copy or repair the rejected response.

Required conflict_dispositions frame IDs, exactly once and in this order:
{json.dumps(frame_ids, separators=(",", ":"))}
Required conflict_dispositions count: {len(frame_ids)}
Required criterion_conflict_dispositions cells, exactly once and in this order:
{json.dumps(cells, separators=(",", ":"))}
Required criterion_conflict_dispositions count: {len(cells)}
Calibrated-high fact IDs eligible for calibrated_evidence_rejections:
{json.dumps(sorted(calibrated), separators=(",", ":"))}
All fact IDs eligible for decisive_evidence_ids:
{json.dumps(sorted(available), separators=(",", ":"))}

In every calibrated_evidence_rejections array, cite only an eligible ID above, at most once, and
use only direct_image_contradiction, grounding_mismatch, or applicability_mismatch. If no eligible
calibrated-high observation is rejected for that cell, return an empty array. Do not place
ordinary decisive facts in calibrated_evidence_rejections; they belong only in
decisive_evidence_ids.

In every decisive_evidence_ids array, cite only an exact ID from the complete all-fact list above.
Never abbreviate, combine, translate, or invent an ID. If the image itself is decisive and no
listed fact applies, return an empty decisive_evidence_ids array.

Keep the JSON compact: key_findings=[]; case_summary at most 20 words; every frame-level and
criterion-level reason at most 8 words; no indentation and no repeated prose. Include every
required frame, cell, candidate partition, and audit array. Silently verify counts, identities,
candidate partitions, and that the final character closes the top-level JSON object. These are
representation constraints only and do not favor any candidate or change any decision rule.
"""
        messages[0]["content"].append({"type": "text", "text": reminder})
        return messages
