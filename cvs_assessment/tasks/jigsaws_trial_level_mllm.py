"""Trial-level frozen-Qwen judging and conservative JIGSAWS arbitration."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from jsonschema import Draft202012Validator

from ..mllm_orchestration import (
    FrozenMLLMJudge,
    MLLMAblation,
    MLLMJudgeRequest,
    _data_url,
)


TRIAL_LEVEL_ABLATIONS: dict[str, MLLMAblation] = {
    "bare_trial_mllm": MLLMAblation(
        "bare_trial_mllm",
        False,
        (),
        decision_protocol="ordinal_state",
        require_fact_only_plugins=True,
    ),
    "skill_motion_trial_mllm": MLLMAblation(
        "skill_motion_trial_mllm",
        True,
        ("temporal",),
        decision_protocol="ordinal_state",
        require_fact_only_plugins=True,
    ),
}

ARBITRATION_ABLATION = MLLMAblation(
    "conservative_trial_framework",
    True,
    ("temporal",),
    decision_protocol="ordinal_state",
    require_fact_only_plugins=True,
)


def _object(
    properties: dict[str, Any], required: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def build_trial_level_schema(
    request: MLLMJudgeRequest,
    ablation: MLLMAblation,
) -> dict[str, Any]:
    frame_keys = [f"f{i}" for i in range(len(request.frame_ids))]
    decision = _object({
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": "string", "enum": ["l", "m", "h"]},
        "evidence_adequacy": {
            "type": "string", "enum": ["insufficient", "limited", "adequate"],
        },
        "supporting_frames": {
            "type": "array",
            "items": {"type": "string", "enum": frame_keys},
            "minItems": 0,
            "maxItems": 3,
            "uniqueItems": True,
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 220},
    }, ("score", "confidence", "evidence_adequacy", "supporting_frames", "reason"))
    criteria = _object(
        {f"c{i}": decision for i in range(len(request.criteria))},
        tuple(f"c{i}" for i in range(len(request.criteria))),
    )
    selected_plugins = sorted(
        plugin.plugin_id for plugin in request.plugin_evidence
        if plugin.plugin_kind in set(ablation.include_plugin_kinds)
    )
    plugins = _object(
        {f"p{i}": {"type": "boolean"} for i in range(len(selected_plugins))},
        tuple(f"p{i}" for i in range(len(selected_plugins))),
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object({
            "trial_decisions": criteria,
            "plugin_use": plugins,
            "case_summary": {"type": "string", "minLength": 1, "maxLength": 500},
        }, ("trial_decisions", "plugin_use", "case_summary")),
    }
    Draft202012Validator.check_schema(schema)
    return schema


def _selected_plugins(
    request: MLLMJudgeRequest,
    ablation: MLLMAblation,
) -> list[Any]:
    allowed = set(ablation.include_plugin_kinds)
    return [
        plugin for plugin in request.plugin_evidence
        if plugin.plugin_kind in allowed
    ]


def _identity_mapping(
    request: MLLMJudgeRequest,
    plugin_ids: list[str],
) -> dict[str, Any]:
    return {
        "frames": {
            f"f{i}": {
                "frame_index": frame_id,
                "timestamp_s": round(float(timestamp), 3),
            }
            for i, (frame_id, timestamp) in enumerate(
                zip(request.frame_ids, request.timestamps_s)
            )
        },
        "criteria": {
            f"c{i}": criterion.criterion_id
            for i, criterion in enumerate(request.criteria)
        },
        "plugins": {f"p{i}": plugin_id for i, plugin_id in enumerate(plugin_ids)},
    }


def _trial_prompt(
    request: MLLMJudgeRequest,
    ablation: MLLMAblation,
) -> str:
    plugins = _selected_plugins(request, ablation)
    plugin_ids = sorted(plugin.plugin_id for plugin in plugins)
    mapping = _identity_mapping(request, plugin_ids)
    criteria = "\n".join(
        f"- {criterion.criterion_id}: {criterion.minimal_description}"
        for criterion in request.criteria
    )
    additions: list[str] = []
    if ablation.include_skill:
        if not request.skill_text:
            raise ValueError("Trial-level Skill arm requires Skill text")
        additions.append("EXECUTABLE TRIAL-LEVEL SKILL:\n" + request.skill_text.strip())
    if plugins:
        additions.append(
            "ADVISORY FACT-ONLY MOTION EVIDENCE:\n" +
            json.dumps(
                [
                    {
                        "plugin_id": plugin.plugin_id,
                        "description": plugin.description,
                        "payload": plugin.payload,
                    }
                    for plugin in plugins
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    evidence = "\n\n".join(additions) if additions else (
        "No detailed Skill or motion plugin is available in this ablation."
    )
    return f"""You are the frozen foundation MLLM and the final judge of one complete robotic suturing trial. Assess the COMPLETE TRIAL, not one independent label per image. The images are sparse chronological observations; motion facts, when supplied, summarize synchronized measurements across the trial.

Use the modified Global Rating Scale directly:
- 1: clear, frequent, or severe problems for the criterion;
- 2: important recurring problems, though some execution is acceptable;
- 3: mixed or adequate execution, with neither clear poor performance nor consistent proficiency;
- 4: consistently proficient execution supported across the trial;
- 5: exceptional, near-flawless execution with strong repeated support.

Absence of an obvious error in sparse images is not evidence for score 4 or 5. Do not assume the score distribution. Use 3 when the evidence is genuinely mixed. Evidence adequacy and score are separate: limited evidence does not automatically mean a low score. Motion percentiles describe position within other-subject reference trials; a high or low percentile is not automatically good or bad. A predicted gesture identity is never itself a skill rating.

Minimal criterion identities:
{criteria}

{evidence}

Return the strict fixed object defined by the JSON Schema. Give exactly one trial-level score for every criterion. supporting_frames may cite at most three frame keys and may be empty when the decision mainly uses complete-trial motion evidence. plugin_use must contain every supplied plugin exactly once. Keep each reason concrete and concise. Never use filenames, operator identity, self-reported experience, expert labels, or ground-truth gesture transcripts.

Exact identity mapping: {json.dumps(mapping, ensure_ascii=False, separators=(',', ':'))}
"""


class JigsawsTrialLevelJudge(FrozenMLLMJudge):
    """Strict-schema complete-trial judge using the unchanged frozen Qwen."""

    def response_format(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> dict[str, Any]:
        schema = build_trial_level_schema(request, ablation)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "jigsaws_trial_level_grs_v1",
                "strict": True,
                "schema": schema,
            },
        }

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        self.validate_ablation_plugins(request, ablation)
        content: list[dict[str, Any]] = [{"type": "text", "text": _trial_prompt(request, ablation)}]
        for position, (frame_id, timestamp, jpeg) in enumerate(zip(
            request.frame_ids,
            request.timestamps_s,
            request.frame_jpegs,
        )):
            content.append({
                "type": "text",
                "text": (
                    f"FRAME f{position}: frame_index={frame_id}, "
                    f"timestamp_s={float(timestamp):.3f}"
                ),
            })
            content.append({"type": "image_url", "image_url": {"url": _data_url(jpeg)}})
        return [{"role": "user", "content": content}]

    def validate_response(
        self,
        value: Any,
        request: MLLMJudgeRequest,
        available_plugin_ids: set[str] | None = None,
        probability_prior: list[list[float]] | None = None,
        log_odds_step: float = 0.5,
        decision_protocol: str = "ordinal_state",
    ) -> dict[str, Any]:
        del probability_prior, log_odds_step, decision_protocol
        plugin_ids = sorted(available_plugin_ids or ())
        ablation = (
            TRIAL_LEVEL_ABLATIONS["skill_motion_trial_mllm"]
            if plugin_ids else TRIAL_LEVEL_ABLATIONS["bare_trial_mllm"]
        )
        schema = build_trial_level_schema(request, ablation)
        Draft202012Validator(schema).validate(value)
        frame_key_to_id = {
            f"f{i}": frame_id for i, frame_id in enumerate(request.frame_ids)
        }
        criteria = []
        for index, criterion in enumerate(request.criteria):
            row = value["trial_decisions"][f"c{index}"]
            score = int(row["score"])
            criteria.append({
                "criterion_id": criterion.criterion_id,
                "score": score,
                "probability_satisfied": (score - 1.0) / 4.0,
                "confidence": row["confidence"],
                "evidence_adequacy": row["evidence_adequacy"],
                "supporting_frame_indices": [
                    frame_key_to_id[key] for key in row["supporting_frames"]
                ],
                "reason": row["reason"],
            })
        used = [
            plugin_id for index, plugin_id in enumerate(plugin_ids)
            if value["plugin_use"][f"p{index}"] is True
        ]
        return {
            "criterion_order": [item.criterion_id for item in request.criteria],
            "trial_criteria": criteria,
            "plugin_assessment": [
                {"plugin_id": plugin_id, "used": plugin_id in used}
                for plugin_id in plugin_ids
            ],
            "case_summary": value["case_summary"],
            "trial_level_contract": {
                "one_score_per_complete_trial_criterion": True,
                "frame_level_state_averaging_used": False,
                "expert_labels_accessed": False,
            },
        }

    def judge(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> dict[str, Any]:
        output = super().judge(request, ablation)
        output["schema_version"] = "jigsaws_trial_level_foundation_judgment_v1"
        output["decision_mode"] = "complete_trial_modified_GRS"
        return output


def _trial_rows(document: dict[str, Any], request: MLLMJudgeRequest) -> dict[str, dict[str, Any]]:
    if document.get("foundation_model_parameters_updated") is not False:
        raise ValueError("JIGSAWS trial candidates must use a frozen foundation model")
    rows = document.get("prediction", {}).get("trial_criteria")
    if not isinstance(rows, list):
        raise ValueError("JIGSAWS trial candidate lacks trial criteria")
    by_id = {str(row.get("criterion_id")): row for row in rows if isinstance(row, dict)}
    expected = [criterion.criterion_id for criterion in request.criteria]
    if set(by_id) != set(expected) or len(by_id) != len(rows):
        raise ValueError("JIGSAWS trial candidate criteria changed")
    for criterion_id in expected:
        if int(by_id[criterion_id].get("score", 0)) not in {1, 2, 3, 4, 5}:
            raise ValueError("JIGSAWS trial candidate score is invalid")
    return by_id


def _plugin_fact_ids(request: MLLMJudgeRequest) -> list[str]:
    output = []
    for plugin in request.plugin_evidence:
        for fact in plugin.payload.get("facts", []):
            fact_id = str(fact.get("fact_id", ""))
            if fact_id and fact_id not in output:
                output.append(fact_id)
    return output


def build_trial_arbitration_schema(
    request: MLLMJudgeRequest,
    bare: dict[str, Any],
    enhanced: dict[str, Any],
) -> dict[str, Any]:
    bare_rows = _trial_rows(bare, request)
    enhanced_rows = _trial_rows(enhanced, request)
    fact_ids = _plugin_fact_ids(request)
    frame_keys = [f"f{i}" for i in range(len(request.frame_ids))]
    decisions = {}
    for index, criterion in enumerate(request.criteria):
        criterion_id = criterion.criterion_id
        bare_score = int(bare_rows[criterion_id]["score"])
        enhanced_score = int(enhanced_rows[criterion_id]["score"])
        actions = ["H"]
        if enhanced_score > bare_score:
            actions.append("U")
        elif enhanced_score < bare_score:
            actions.append("D")
        common = {
            "supporting_frames": {
                "type": "array",
                "items": {"type": "string", "enum": frame_keys},
                "minItems": 0,
                "maxItems": 3,
                "uniqueItems": True,
            },
            "reason": {"type": "string", "minLength": 1, "maxLength": 220},
        }
        required = (
            "action", "evidence_strength", "supporting_fact_id",
            "supporting_frames", "reason",
        )
        variants = [_object({
            **common,
            "action": {"const": "H"},
            "evidence_strength": {"type": "string", "enum": ["l", "m", "h"]},
            "supporting_fact_id": {
                "type": "string", "enum": ["none", *fact_ids],
            },
        }, required)]
        # A score change is token-level constrained to high-strength evidence
        # and a real plugin fact. Direct images may be cited in addition, but a
        # plugin-conditioned branch cannot move the bare prior without plugin
        # evidence that the same Qwen explicitly accepts.
        for action in actions[1:]:
            variants.append(_object({
                **common,
                "action": {"const": action},
                "evidence_strength": {"const": "h"},
                "supporting_fact_id": {
                    "type": "string", "enum": fact_ids,
                },
            }, required))
        decisions[f"c{index}"] = {"anyOf": variants}
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object({
            "criterion_decisions": _object(
                decisions, tuple(f"c{i}" for i in range(len(request.criteria))),
            ),
            "case_summary": {"type": "string", "minLength": 1, "maxLength": 500},
        }, ("criterion_decisions", "case_summary")),
    }
    Draft202012Validator.check_schema(schema)
    return schema


def merge_conservative_trial_decisions(
    request: MLLMJudgeRequest,
    bare: dict[str, Any],
    enhanced: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    """Apply only high-evidence, one-point Qwen-authored changes to bare Qwen."""
    schema = build_trial_arbitration_schema(request, bare, enhanced)
    Draft202012Validator(schema).validate(response)
    bare_rows = _trial_rows(bare, request)
    enhanced_rows = _trial_rows(enhanced, request)
    fact_ids = set(_plugin_fact_ids(request))
    frame_key_to_id = {f"f{i}": value for i, value in enumerate(request.frame_ids)}
    output_rows, audit = [], []
    for index, criterion in enumerate(request.criteria):
        criterion_id = criterion.criterion_id
        source = response["criterion_decisions"][f"c{index}"]
        action = str(source["action"])
        bare_score = int(bare_rows[criterion_id]["score"])
        enhanced_score = int(enhanced_rows[criterion_id]["score"])
        support_fact = str(source["supporting_fact_id"])
        support_frames = [frame_key_to_id[key] for key in source["supporting_frames"]]
        if action != "H":
            if source["evidence_strength"] != "h":
                raise ValueError("A conservative JIGSAWS score change requires high evidence")
            if support_fact == "none" and not support_frames:
                raise ValueError("A conservative JIGSAWS score change requires cited evidence")
            if support_fact != "none" and support_fact not in fact_ids:
                raise ValueError("JIGSAWS arbitration cited an unavailable fact")
        if action == "U":
            if enhanced_score <= bare_score:
                raise ValueError("JIGSAWS upgrade contradicts the enhanced candidate")
            final_score = min(5, bare_score + 1)
        elif action == "D":
            if enhanced_score >= bare_score:
                raise ValueError("JIGSAWS downgrade contradicts the enhanced candidate")
            final_score = max(1, bare_score - 1)
        else:
            final_score = bare_score
        row = deepcopy(bare_rows[criterion_id])
        row.update({
            "score": final_score,
            "probability_satisfied": (final_score - 1.0) / 4.0,
            "decision_action": {"H": "hold", "U": "upgrade", "D": "downgrade"}[action],
            "correction_magnitude_points": abs(final_score - bare_score),
            "prior_bare_score": bare_score,
            "enhanced_candidate_score": enhanced_score,
            "arbitration_evidence_strength": source["evidence_strength"],
            "arbitration_supporting_fact_id": support_fact,
            "arbitration_supporting_frame_indices": support_frames,
            "arbitration_reason": source["reason"],
        })
        output_rows.append(row)
        audit.append({
            "criterion_id": criterion_id,
            "bare_score": bare_score,
            "enhanced_score": enhanced_score,
            "action": action,
            "final_score": final_score,
            "supporting_fact_id": support_fact,
            "supporting_frame_indices": support_frames,
        })
    return {
        "criterion_order": [criterion.criterion_id for criterion in request.criteria],
        "trial_criteria": output_rows,
        "case_summary": response["case_summary"],
        "conservative_merge_audit": audit,
        "trial_level_contract": {
            "one_score_per_complete_trial_criterion": True,
            "frame_level_state_averaging_used": False,
            "bare_qwen_centered": True,
            "maximum_change_from_bare_per_criterion": 1,
            "change_requires_same_qwen_high_evidence": True,
            "raw_small_model_score_used_as_final_prediction": False,
            "expert_labels_accessed": False,
        },
    }


class JigsawsConservativeTrialArbitrator(FrozenMLLMJudge):
    """Same-Qwen bounded arbitration between bare and plugin-conditioned scores."""

    _bare: dict[str, Any] | None = None
    _enhanced: dict[str, Any] | None = None

    def judge_trial(
        self,
        request: MLLMJudgeRequest,
        bare: dict[str, Any],
        enhanced: dict[str, Any],
    ) -> dict[str, Any]:
        _trial_rows(bare, request)
        _trial_rows(enhanced, request)
        if bare.get("model") != self.model or enhanced.get("model") != self.model:
            raise ValueError("JIGSAWS arbitration candidates must use the same frozen Qwen")
        self._bare, self._enhanced = bare, enhanced
        try:
            output = super().judge(request, ARBITRATION_ABLATION)
        finally:
            self._bare, self._enhanced = None, None
        output["schema_version"] = "jigsaws_conservative_trial_framework_v1"
        output["decision_mode"] = "same_frozen_qwen_bare_centered_bounded_arbitration"
        output["conservative_merge"] = {
            "bare_qwen_centered": True,
            "maximum_change_from_bare_per_criterion": 1,
            "final_semantic_judge": "same_frozen_qwen",
            "raw_small_model_score_used_as_final_prediction": False,
            "labels_accessed": False,
            "foundation_model_parameters_updated": False,
        }
        return output

    def _candidates(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._bare is None or self._enhanced is None:
            raise RuntimeError("JIGSAWS arbitration candidates are unavailable")
        return self._bare, self._enhanced

    def response_format(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> dict[str, Any]:
        del ablation
        bare, enhanced = self._candidates()
        schema = build_trial_arbitration_schema(request, bare, enhanced)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "jigsaws_conservative_trial_arbitration_v1",
                "strict": True,
                "schema": schema,
            },
        }

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        self.validate_ablation_plugins(request, ablation)
        bare, enhanced = self._candidates()
        bare_rows = _trial_rows(bare, request)
        enhanced_rows = _trial_rows(enhanced, request)
        candidates = {
            criterion.criterion_id: {
                "bare_qwen": {
                    key: bare_rows[criterion.criterion_id].get(key)
                    for key in ("score", "confidence", "evidence_adequacy", "reason")
                },
                "skill_motion_qwen": {
                    key: enhanced_rows[criterion.criterion_id].get(key)
                    for key in ("score", "confidence", "evidence_adequacy", "reason")
                },
            }
            for criterion in request.criteria
        }
        facts = [
            {
                "plugin_id": plugin.plugin_id,
                "facts": plugin.payload.get("facts", []),
            }
            for plugin in request.plugin_evidence
            if plugin.plugin_kind in set(ablation.include_plugin_kinds)
        ]
        mapping = _identity_mapping(
            request, sorted(plugin.plugin_id for plugin in request.plugin_evidence),
        )
        prompt = f"""You are the SAME frozen foundation MLLM making the final conservative decision for one complete JIGSAWS suturing trial. Two complete hypotheses from you are available: bare_qwen and skill_motion_qwen. The bare hypothesis is the stable prior. The enhanced hypothesis and plugin facts are advisory, not votes or ground truth.

For each criterion:
- H holds the bare score;
- U is allowed only when the enhanced score is higher and concrete high-strength evidence supports moving exactly one point upward from bare;
- D is allowed only when the enhanced score is lower and concrete high-strength evidence supports moving exactly one point downward from bare.

Use U or D only with evidence_strength=h and cite at least one supplied fact ID or real frame. Otherwise use H. A motion percentile is descriptive, not automatically good or bad. Gesture identity is not skill. Lack of an obvious error is not sufficient for an upgrade. The small model never owns the final score; your H/U/D action does.

Candidate hypotheses: {json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}

Executable Skill: {request.skill_text or 'No detailed Skill supplied.'}

Advisory facts: {json.dumps(facts, ensure_ascii=False, separators=(',', ':'))}

Return only the strict JSON-Schema object. Exact identity mapping: {json.dumps(mapping, ensure_ascii=False, separators=(',', ':'))}
"""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for position, (frame_id, timestamp, jpeg) in enumerate(zip(
            request.frame_ids, request.timestamps_s, request.frame_jpegs,
        )):
            content.append({
                "type": "text",
                "text": f"FRAME f{position}: frame_index={frame_id}, timestamp_s={timestamp:.3f}",
            })
            content.append({"type": "image_url", "image_url": {"url": _data_url(jpeg)}})
        return [{"role": "user", "content": content}]

    def validate_response(
        self,
        value: Any,
        request: MLLMJudgeRequest,
        available_plugin_ids: set[str] | None = None,
        probability_prior: list[list[float]] | None = None,
        log_odds_step: float = 0.5,
        decision_protocol: str = "ordinal_state",
    ) -> dict[str, Any]:
        del available_plugin_ids, probability_prior, log_odds_step, decision_protocol
        bare, enhanced = self._candidates()
        return merge_conservative_trial_decisions(request, bare, enhanced, value)


def trial_schema_sha256(request: MLLMJudgeRequest, ablation: MLLMAblation) -> str:
    schema = build_trial_level_schema(request, ablation)
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
