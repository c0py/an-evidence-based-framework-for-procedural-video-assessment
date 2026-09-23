"""Symmetric multi-branch prompting for the unchanged frozen foundation MLLM."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .mllm_orchestration import (
    FrozenMLLMJudge,
    MLLMAblation,
    MLLMJudgeRequest,
    compact_foundation_preliminary,
)


def compact_multibranch_candidate(
    request: MLLMJudgeRequest, judgment: dict[str, Any],
) -> dict[str, Any]:
    """Validate routed rows and retain label-free whole-timeline context."""
    compact = compact_foundation_preliminary(request, judgment)
    context = judgment.get("multibranch_slice_context")
    if not isinstance(context, dict):
        raise ValueError("Multibranch candidate lacks whole-timeline slice context")
    if context.get("source_ablation") != compact["source_ablation"]:
        raise ValueError("Multibranch context and candidate evidence path differ")
    original_ids = context.get("original_frame_ids")
    if (
        not isinstance(original_ids, list)
        or len(original_ids) != int(context.get("original_frame_count", -1))
        or context.get("routed_frame_ids") != request.frame_ids
        or context.get("ground_truth_or_labels_included") is not False
    ):
        raise ValueError("Multibranch candidate has invalid whole-timeline context")
    compact["whole_timeline_context"] = {
        "original_frame_ids": [int(value) for value in original_ids],
        "original_frame_count": len(original_ids),
        "routed_frame_ids": list(request.frame_ids),
        "case_summary": str(context.get("whole_timeline_case_summary", "")),
        "plugin_assessment": context.get("whole_timeline_plugin_assessment", []),
        "ground_truth_or_labels_included": False,
    }
    return compact


@dataclass
class MultibranchJudgeRequest(MLLMJudgeRequest):
    foundation_candidate_judgments: list[dict[str, Any]] = field(default_factory=list)
    plugin_coverage_manifests: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        super().validate()
        if self.foundation_preliminary_judgment is not None or self.foundation_auxiliary_judgments:
            raise ValueError("Symmetric candidates cannot be mixed with legacy primary/auxiliary fields")
        if len(self.foundation_candidate_judgments) < 2:
            raise ValueError("At least two symmetric frozen-foundation candidates are required")
        compact = [
            compact_multibranch_candidate(self, judgment)
            for judgment in self.foundation_candidate_judgments
        ]
        names = [row["source_ablation"] for row in compact]
        if len(names) != len(set(names)):
            raise ValueError("Symmetric foundation candidates must use distinct evidence paths")
        plugin_ids = {plugin.plugin_id for plugin in self.plugin_evidence}
        manifest_ids = set()
        for manifest in self.plugin_coverage_manifests:
            if manifest.get("schema_version") != "foundation_plugin_coverage_manifest_v1":
                raise ValueError("Unexpected plugin coverage manifest schema")
            plugin_id = str(manifest.get("plugin_id", ""))
            if not plugin_id or plugin_id in manifest_ids:
                raise ValueError("Plugin coverage manifests must have unique plugin IDs")
            manifest_ids.add(plugin_id)
            if (
                manifest.get("labels_accessed") is not False
                or manifest.get("final_task_prediction_provided") is not False
                or manifest.get("coverage_is_evidence_applicability_not_task_truth") is not True
            ):
                raise ValueError("Plugin coverage manifests must be label-free and fact-only")
            covered = set(int(value) for value in manifest.get("covered_frame_ids", []))
            uncovered = set(int(value) for value in manifest.get("uncovered_frame_ids", []))
            all_ids = set(int(value) for value in manifest.get("all_frame_ids", self.frame_ids))
            # Manifests are built on the whole timeline, so routed IDs need only
            # be a subset of their covered/uncovered partition.
            if covered & uncovered or not set(self.frame_ids).issubset(covered | uncovered | all_ids):
                raise ValueError("Plugin coverage manifest has an invalid frame partition")
        if manifest_ids != plugin_ids:
            raise ValueError("Plugin coverage manifests differ from supplied evidence plugins")


class SymmetricMultibranchPromptMixin:
    """Mixin that appends candidates only after the real images in the request."""

    def _compact_candidates(self, request: MultibranchJudgeRequest) -> list[dict[str, Any]]:
        compact = [
            compact_multibranch_candidate(request, judgment)
            for judgment in request.foundation_candidate_judgments
        ]
        for row in compact:
            if row["source_model"] != self.model:
                raise ValueError("Every candidate and final pass must use the same frozen model")
        names = [row["source_ablation"] for row in compact]
        if len(names) != len(set(names)):
            raise ValueError("Candidate evidence paths must be distinct")
        return compact

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        if not isinstance(request, MultibranchJudgeRequest):
            raise TypeError("Symmetric multibranch judging requires MultibranchJudgeRequest")
        request.validate()
        base_request = MLLMJudgeRequest(
            task_id=request.task_id, sample_id=request.sample_id,
            criteria=request.criteria, frame_ids=request.frame_ids,
            timestamps_s=request.timestamps_s, frame_jpegs=request.frame_jpegs,
            skill_text=request.skill_text, plugin_evidence=request.plugin_evidence,
        )
        messages = super().build_messages(base_request, ablation)
        candidates = self._compact_candidates(request)
        names = [row["source_ablation"] for row in candidates]
        arbitration_contract = f"""
SYMMETRIC CANDIDATE HYPOTHESES FROM THE SAME FROZEN FOUNDATION MLLM
(fallible evidence-path hypotheses, never votes, priors, or ground truth):
{json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))}

LABEL-FREE PLUGIN APPLICABILITY / COVERAGE MANIFESTS
(coverage says where evidence applies; uncovered never means visibly absent):
{json.dumps(request.plugin_coverage_manifests, ensure_ascii=False, separators=(",", ":"))}

This is the final symmetric foundation arbitration pass. The real images and criterion definitions
were presented before these hypotheses. No candidate is a stable primary. Independently inspect
each routed image, use the compact whole-timeline context, and accept or reject each hypothesis.
A low-confidence N under poor visibility is absence of direct support, not negative evidence.
Temporal persistence cannot manufacture a visually defined F, and a missing detection cannot lower
an image-supported F. Plugin evidence may affect only its covered frame or interval unless you
independently verify the claim in the real image.

Add one sixth top-level JSON key named conflict_dispositions. It must contain exactly one object per
supplied frame, in frame order, with this structure:
{{"frame_index":0,"accepted_hypotheses":["mllm_skill"],"rejected_hypotheses":["mllm_skill_visual","mllm_skill_temporal"],"evidence_scope":"image_only","reason":"short conflict resolution"}}
For every disposition, accepted_hypotheses and rejected_hypotheses must form a disjoint complete
partition of {json.dumps(names)}. accepted_hypotheses may be empty when you reject every candidate
and decide independently. evidence_scope must be one of image_only, direct_plugin, interval_plugin,
whole_timeline, or mixed. You remain the sole final decision maker.
"""
        # Base content is [instructions, frame marker, image, ...]. Appending
        # candidates here guarantees every real image precedes every hypothesis.
        messages[0]["content"].append({"type": "text", "text": arbitration_contract})
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
            raise TypeError("Symmetric response validation requires MultibranchJudgeRequest")
        candidates = self._compact_candidates(request)
        names = {row["source_ablation"] for row in candidates}
        dispositions = value.get("conflict_dispositions") if isinstance(value, dict) else None
        if not isinstance(dispositions, list) or len(dispositions) != len(request.frame_ids):
            raise RuntimeError("Multibranch response needs one conflict disposition per frame")
        validated = []
        scopes = {"image_only", "direct_plugin", "interval_plugin", "whole_timeline", "mixed"}
        for expected_frame, row in zip(request.frame_ids, dispositions):
            if not isinstance(row, dict) or int(row.get("frame_index", -1)) != expected_frame:
                raise RuntimeError("Conflict disposition frame identity/order differs")
            accepted = row.get("accepted_hypotheses")
            rejected = row.get("rejected_hypotheses")
            if not isinstance(accepted, list) or not isinstance(rejected, list):
                raise RuntimeError("Conflict disposition hypotheses must be lists")
            accepted_set = set(str(value) for value in accepted)
            rejected_set = set(str(value) for value in rejected)
            if accepted_set & rejected_set or accepted_set | rejected_set != names:
                raise RuntimeError("Conflict disposition must partition every candidate")
            scope = str(row.get("evidence_scope", ""))
            reason = str(row.get("reason", "")).strip()
            if scope not in scopes or not reason:
                raise RuntimeError("Conflict disposition needs a valid scope and reason")
            validated.append({
                "frame_index": expected_frame,
                "accepted_hypotheses": [name for name in accepted if name in names],
                "rejected_hypotheses": [name for name in rejected if name in names],
                "evidence_scope": scope, "reason": reason,
            })
        normalized["conflict_dispositions"] = validated
        return normalized


class SymmetricMultibranchFrozenMLLMJudge(
    SymmetricMultibranchPromptMixin, FrozenMLLMJudge,
):
    """Concrete symmetric judge for tasks without response-normalization subclasses."""

