"""Foundation-MLLM orchestration over interchangeable evidence plugins.

The MLLM is the final task judge.  Plugins may expose visual, temporal, or
task-skill evidence, but their outputs are advisory and never become the final
prediction without passing through the MLLM.  The module is deliberately
task-neutral; dataset adapters build the concrete evidence payloads.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import json
import math
import re
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class CriterionContract:
    criterion_id: str
    title: str
    minimal_description: str


@dataclass(frozen=True)
class PluginEvidence:
    plugin_id: str
    plugin_kind: str
    description: str
    payload: dict[str, Any]
    foundation_model_parameters_updated: bool = False

    def validate_fact_only(self) -> None:
        """Require advisory observations rather than a task-level prediction."""
        if self.payload.get("schema_version") != "grounded_evidence_facts_v1":
            raise ValueError(
                f"Fact-only ablation requires grounded_evidence_facts_v1: {self.plugin_id}"
            )
        if self.payload.get("final_task_prediction_provided") is not False:
            raise ValueError(
                f"Fact-only plugin must explicitly omit final task predictions: {self.plugin_id}"
            )
        forbidden = {
            "calibrated_probability_prior", "probability_satisfied",
            "criterion_probability", "criterion_probabilities",
            "criterion_scores", "final_prediction", "final_verdict",
        }

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                overlap = forbidden.intersection(value)
                if overlap:
                    raise ValueError(
                        f"Fact-only plugin {self.plugin_id} contains task prediction keys: "
                        f"{sorted(overlap)}"
                    )
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.payload)


@dataclass
class MLLMJudgeRequest:
    task_id: str
    sample_id: str
    criteria: list[CriterionContract]
    frame_ids: list[int]
    timestamps_s: list[float]
    frame_jpegs: list[bytes] = field(repr=False)
    skill_text: str | None = None
    plugin_evidence: list[PluginEvidence] = field(default_factory=list)
    foundation_preliminary_judgment: dict[str, Any] | None = None
    foundation_auxiliary_judgments: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        n = len(self.frame_ids)
        if n == 0 or len(self.timestamps_s) != n or len(self.frame_jpegs) != n:
            raise ValueError("frame_ids, timestamps_s, and frame_jpegs must have equal nonzero length")
        if len(set(self.frame_ids)) != n:
            raise ValueError("frame_ids must be unique")
        if any(b <= a for a, b in zip(self.timestamps_s, self.timestamps_s[1:])):
            raise ValueError("timestamps_s must be strictly increasing")
        if not self.criteria or len({item.criterion_id for item in self.criteria}) != len(self.criteria):
            raise ValueError("criteria must be nonempty and unique")
        if any(item.foundation_model_parameters_updated for item in self.plugin_evidence):
            raise ValueError("Frozen-MLLM runs cannot consume plugins that updated the foundation model")
        if self.foundation_preliminary_judgment is not None:
            compact_foundation_preliminary(self, self.foundation_preliminary_judgment)
        for judgment in self.foundation_auxiliary_judgments:
            compact_foundation_preliminary(self, judgment)


def compact_foundation_preliminary(
    request: MLLMJudgeRequest, judgment: dict[str, Any],
) -> dict[str, Any]:
    """Validate and compact a same-foundation first-pass judgment for re-adjudication."""
    if not isinstance(judgment, dict):
        raise ValueError("Foundation preliminary judgment must be an object")
    if not str(judgment.get("schema_version", "")).startswith(
        "foundation_mllm_plugin_judgment_"
    ):
        raise ValueError("Preliminary judgment must originate from the foundation MLLM")
    if judgment.get("foundation_model_parameters_updated") is not False:
        raise ValueError("Preliminary judgment must come from the frozen foundation MLLM")
    if judgment.get("task_id") != request.task_id or judgment.get("sample_id") != request.sample_id:
        raise ValueError("Preliminary judgment task/sample identity differs from the request")
    if judgment.get("decision_mode") != "ordinal_state":
        raise ValueError("Foundation re-adjudication requires an ordinal preliminary judgment")
    frames = judgment.get("prediction", {}).get("frames")
    if not isinstance(frames, list) or len(frames) != len(request.frame_ids):
        raise ValueError("Preliminary judgment frame count differs from the request")
    criterion_ids = [item.criterion_id for item in request.criteria]
    compact_frames = []
    for expected_id, frame in zip(request.frame_ids, frames):
        if not isinstance(frame, dict) or int(frame.get("frame_index", -1)) != expected_id:
            raise ValueError("Preliminary judgment frame identity/order differs from the request")
        rows = frame.get("criteria")
        if not isinstance(rows, list):
            raise ValueError("Preliminary judgment has no criterion rows")
        by_id = {
            row.get("criterion_id"): row for row in rows if isinstance(row, dict)
        }
        if set(by_id) != set(criterion_ids) or len(by_id) != len(rows):
            raise ValueError("Preliminary judgment criteria differ from the request")
        states, confidences, visibilities = [], [], []
        visibility_codes = {"good": "g", "limited": "l", "poor": "p"}
        for criterion_id in criterion_ids:
            row = by_id[criterion_id]
            state = str(row.get("foundation_state", "")).upper()
            confidence = str(row.get("foundation_confidence", "")).lower()
            visibility = str(row.get("visibility", "")).lower()
            if state not in {"N", "P", "F", "U"} or confidence not in {"l", "m", "h"}:
                raise ValueError("Preliminary judgment has invalid state/confidence codes")
            if visibility not in {"good", "limited", "poor"}:
                raise ValueError("Preliminary judgment has invalid visibility")
            states.append(state)
            confidences.append(confidence)
            visibilities.append(visibility_codes[visibility])
        compact_frames.append({
            "frame_index": expected_id,
            "states": states,
            "confidences": confidences,
            "visibilities": visibilities,
        })
    return {
        "source": "same_frozen_foundation_mllm_preliminary_judgment",
        "source_model": str(judgment.get("model", "")),
        "source_ablation": str(judgment.get("ablation", {}).get("name", "")),
        "criterion_order": criterion_ids,
        "frames": compact_frames,
        "ground_truth_or_labels_included": False,
        "plugin_task_verdict": False,
    }


@dataclass(frozen=True)
class MLLMAblation:
    name: str
    include_skill: bool
    include_plugin_kinds: tuple[str, ...]
    use_plugin_probability_prior: bool = False
    decision_protocol: str = "direct_probability"
    require_fact_only_plugins: bool = False


STANDARD_ABLATIONS: dict[str, MLLMAblation] = {
    "bare_mllm": MLLMAblation("bare_mllm", False, ()),
    "mllm_skill": MLLMAblation("mllm_skill", True, ()),
    "mllm_visual": MLLMAblation("mllm_visual", False, ("visual",), True),
    "full_framework": MLLMAblation(
        "full_framework", True, ("visual", "temporal"), True,
    ),
}


# Paper-facing ablations all retain the same frozen MLLM as the base and final
# judge.  Visual/temporal tools expose grounded facts only, never CVS/task
# probabilities.  The legacy bounded-prior variants above remain available to
# reproduce the v8 interface experiment.
FOUNDATION_CENTERED_ABLATIONS: dict[str, MLLMAblation] = {
    "bare_mllm": MLLMAblation(
        "bare_mllm", False, (), decision_protocol="ordinal_state",
        require_fact_only_plugins=True,
    ),
    "mllm_skill": MLLMAblation(
        "mllm_skill", True, (), decision_protocol="ordinal_state",
        require_fact_only_plugins=True,
    ),
    "mllm_skill_visual": MLLMAblation(
        "mllm_skill_visual", True, ("visual",),
        decision_protocol="ordinal_state", require_fact_only_plugins=True,
    ),
    "mllm_skill_temporal": MLLMAblation(
        "mllm_skill_temporal", True, ("temporal",),
        decision_protocol="ordinal_state", require_fact_only_plugins=True,
    ),
    "full_framework": MLLMAblation(
        "full_framework", True, ("visual", "temporal"),
        decision_protocol="ordinal_state", require_fact_only_plugins=True,
    ),
}


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _strip_json_fence(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return candidate


def _deterministic_json_syntax_repair(text: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Repair only known punctuation slips without generating semantic content."""
    candidate = _strip_json_fence(text)
    repairs = [
        (
            "insert_missing_reason_key_after_criterion_id",
            re.sub(
                r'("criterion_id"\s*:\s*"[^"]+")\s*:\s*"',
                r'\1,"reason":"', candidate,
            ),
        ),
        (
            "close_last_frame_object_before_plugin_assessment",
            re.sub(
                r'("v"\s*:\s*\[(?:\s*"[glp]"\s*,?)+\])\s*\],\s*("plugin_assessment")',
                r'\1}],\2', candidate, count=1,
            ),
        ),
        (
            "remove_trailing_commas",
            re.sub(r',\s*([}\]])', r'\1', candidate),
        ),
        (
            "close_truncated_final_case_summary_string_and_object",
            re.sub(
                r'("case_summary"\s*:\s*"(?:[^"\\]|\\.)*)\Z',
                r'\1"}', candidate,
            ),
        ),
    ]
    for rule, repaired in repairs:
        if repaired == candidate:
            continue
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, repaired, rule
    return None, None, None


class FrozenMLLMJudge:
    """OpenAI-compatible frozen MLLM used as the final evidence-fusion judge."""

    def __init__(
        self, base_url: str, model: str, api_key: str = "EMPTY",
        timeout_s: float = 900.0, max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def response_format(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> dict[str, Any]:
        """Return the decoding contract; subclasses may supply a frozen JSON Schema."""
        return {"type": "json_object"}

    def judge(self, request: MLLMJudgeRequest, ablation: MLLMAblation) -> dict[str, Any]:
        request.validate()
        self.validate_ablation_plugins(request, ablation)
        messages = self.build_messages(request, ablation)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format(request, ablation),
        }
        http_request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urlopen(http_request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MLLM request failed with HTTP {exc.code}: {detail[:2000]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach frozen MLLM at {self.base_url}: {exc.reason}") from exc
        latency_s = time.monotonic() - started
        try:
            raw_text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"MLLM response has no completion: {body}") from exc
        repaired_text = None
        repair_runtime: dict[str, Any] | None = None
        repair_method: str | None = None
        parsed: dict[str, Any] | None = None
        original_parse_error: json.JSONDecodeError | None = None
        try:
            parsed = json.loads(_strip_json_fence(raw_text))
        except json.JSONDecodeError as exc:
            original_parse_error = exc
            parsed, repaired_text, deterministic_rule = _deterministic_json_syntax_repair(raw_text)
            if parsed is not None:
                repair_method = "deterministic_punctuation_only"
                repair_runtime = {
                    "latency_s": 0.0,
                    "usage": {},
                    "original_json_error": str(original_parse_error),
                    "rule": deterministic_rule,
                }
            else:
                repair_method = "frozen_mllm_syntax_only"
        if parsed is None:
            complete_response = all(
                re.search(rf'"frame_index"\s*:\s*{frame_id}(?:\D|$)', raw_text)
                for frame_id in request.frame_ids
            ) and "case_summary" in raw_text
            if not complete_response:
                raise RuntimeError(
                    "MLLM returned truncated invalid JSON; refusing to invent missing decisions. "
                    f"Original error: {original_parse_error}; raw={raw_text[:4000]}"
                ) from original_parse_error
            repair_payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Repair only the JSON syntax in the object below. Preserve every key, "
                        "array item, number, string, and boolean exactly; do not add, remove, "
                        "reorder, summarize, or reinterpret decisions. Return only the corrected "
                        "JSON object with no markdown.\n\n" + raw_text
                    ),
                }],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            repair_request = Request(
                self.base_url + "/chat/completions",
                data=json.dumps(repair_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            repair_started = time.monotonic()
            try:
                with urlopen(repair_request, timeout=self.timeout_s) as response:
                    repair_body = json.loads(response.read().decode("utf-8"))
                repaired_text = repair_body["choices"][0]["message"]["content"]
                parsed = json.loads(_strip_json_fence(repaired_text))
            except (HTTPError, URLError, KeyError, IndexError, TypeError, json.JSONDecodeError) as repair_error:
                raise RuntimeError(
                    "MLLM returned complete but invalid JSON and syntax-only repair failed. "
                    f"Original error: {original_parse_error}; repair error: {repair_error}; "
                    f"raw={raw_text[:4000]}"
                ) from repair_error
            repair_runtime = {
                "latency_s": time.monotonic() - repair_started,
                "usage": repair_body.get("usage", {}),
                "original_json_error": str(original_parse_error),
            }
        available_plugin_ids = {
            item.plugin_id for item in request.plugin_evidence
            if item.plugin_kind in set(ablation.include_plugin_kinds)
        }
        prior = self.probability_prior(request, ablation)
        normalized = self.validate_response(
            parsed, request, available_plugin_ids, probability_prior=prior,
            decision_protocol=ablation.decision_protocol,
        )
        return {
            "schema_version": (
                "foundation_mllm_plugin_judgment_v3"
                if ablation.decision_protocol == "ordinal_state"
                else "foundation_mllm_plugin_judgment_v2"
            ),
            "task_id": request.task_id,
            "sample_id": request.sample_id,
            "ablation": asdict(ablation),
            "model": self.model,
            "foundation_model_parameters_updated": False,
            "prediction": normalized,
            "decision_mode": (
                "bounded_prior_correction" if prior is not None
                else ablation.decision_protocol
            ),
            "runtime": {
                "latency_s": latency_s,
                "usage": body.get("usage", {}),
            },
            "raw_completion": raw_text,
            "json_syntax_repair": {
                "used": repaired_text is not None,
                "method": repair_method,
                "runtime": repair_runtime,
                "raw_repaired_completion": repaired_text,
                "semantics_must_be_unchanged": True,
            },
        }

    def build_messages(
        self, request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[dict[str, Any]]:
        self.validate_ablation_plugins(request, ablation)
        allowed = set(ablation.include_plugin_kinds)
        plugins = [item for item in request.plugin_evidence if item.plugin_kind in allowed]
        criterion_text = "\n".join(
            f"- {item.criterion_id}: {item.minimal_description}"
            for item in request.criteria
        )
        additions = []
        if ablation.include_skill:
            if not request.skill_text:
                raise ValueError(f"Ablation {ablation.name} requires skill_text")
            additions.append("EXECUTABLE SKILL / SOP:\n" + request.skill_text.strip())
        if plugins:
            additions.append(
                "PLUGIN EVIDENCE (advisory; verify it against the images):\n" +
                json.dumps([asdict(item) for item in plugins], ensure_ascii=False, separators=(",", ":"))
            )
        preliminary = None
        if request.foundation_preliminary_judgment is not None:
            preliminary = compact_foundation_preliminary(
                request, request.foundation_preliminary_judgment,
            )
            if preliminary["source_model"] != self.model:
                raise ValueError(
                    "Preliminary and final passes must use the same frozen foundation model"
                )
            additions.append(
                "PRELIMINARY JUDGMENT FROM THE SAME FROZEN FOUNDATION MLLM "
                "(a hypothesis to re-adjudicate, not plugin ground truth):\n" +
                json.dumps(preliminary, ensure_ascii=False, separators=(",", ":"))
            )
        auxiliary = [
            compact_foundation_preliminary(request, judgment)
            for judgment in request.foundation_auxiliary_judgments
        ]
        for judgment in auxiliary:
            if judgment["source_model"] != self.model:
                raise ValueError(
                    "Auxiliary and final passes must use the same frozen foundation model"
                )
        source_ablations = [
            row["source_ablation"] for row in ([preliminary] if preliminary else []) + auxiliary
        ]
        if len(source_ablations) != len(set(source_ablations)):
            raise ValueError("Foundation preliminary branches must use distinct evidence paths")
        if auxiliary:
            additions.append(
                "AUXILIARY HYPOTHESES FROM THE SAME FROZEN FOUNDATION MLLM "
                "(alternate evidence branches, not votes or ground truth):\n" +
                json.dumps(auxiliary, ensure_ascii=False, separators=(",", ":"))
            )
        extra = "\n\n".join(additions) if additions else "No plugin evidence or detailed Skill is available in this ablation."
        frame_count = len(request.frame_ids)
        criterion_order = [item.criterion_id for item in request.criteria]
        supplied_plugin_ids = [item.plugin_id for item in plugins]
        prior = self.probability_prior(request, ablation)
        revision_contract = ""
        if preliminary is not None or auxiliary:
            revision_contract = """
This is a final foundation arbitration pass. The primary preliminary branch supplies the stable
foundation prior. Auxiliary branches are alternate hypotheses from the same frozen MLLM under
different evidence; they are not votes and must not be averaged. Reinspect every real frame. When
an auxiliary temporal branch upgrades an N/P to F, explicitly check that frame and nearby frames
against the Skill and grounded facts. Preserve the primary state when the image is genuinely
unclear, but revise it when direct image-grounded evidence supports the auxiliary hypothesis. A
missing tool detection is never a reason to lower F. You remain the sole final decision maker.
"""
        if ablation.decision_protocol == "ordinal_state":
            decision_contract = f"""For EVERY input frame and EVERY criterion, make the task judgment yourself from the real frame, the Skill when supplied, and advisory grounded facts when supplied.

Use one state code per criterion:
- N: the criterion is not fully satisfied.
- P: partial/supporting evidence exists, but the full criterion is not satisfied.
- F: the full criterion is satisfied.
- U: visibility is insufficient to judge.
Also provide confidence code l, m, or h (low, medium, high). Confidence describes confidence in the selected state, not tool confidence. Do not copy a tool verdict: tools provide observations only and you own every final state.

The state is instantaneous at that frame's timestamp. Judge whether the full visual condition is satisfied in that frame; do not require it to persist across neighboring frames before using F. Temporal persistence may support or weaken confidence and may inform the case summary, but it must not erase a clearly satisfied frame. Missing plugin facts or missing predicted boxes mean the tool did not report a reliable observation; they are not evidence that an entity or condition is absent. Inspect the image yourself.

Return ONLY one compact JSON object with these top-level keys: criterion_order, frame_predictions, plugin_assessment, key_findings, and case_summary. criterion_order must equal {json.dumps(criterion_order)}. Every frame_predictions object must contain frame_index plus arrays s, c, and v in that criterion order. Every plugin_assessment object contains plugin_id and used. Every optional key_findings object contains frame_index, criterion_id, and a short image-grounded reason.

Do not copy illustrative answers from the instructions: independently judge every frame and criterion."""
            row_requirement = (
                f"exactly {len(request.criteria)} state codes, "
                f"{len(request.criteria)} confidence codes, and "
                f"{len(request.criteria)} visibility codes"
            )
        elif prior is None:
            decision_contract = f"""For EVERY input frame and EVERY criterion, estimate probability_satisfied: the probability from 0 to 1 that the criterion is fully satisfied in that frame. Use uncertain visibility rather than inventing anatomy. Temporal plugin evidence may help with persistence and transitions, but the returned probability is still frame-specific.

Return ONLY one compact JSON object with this exact top-level structure:
{{"criterion_order":{json.dumps(criterion_order)},"frame_predictions":[{{"frame_index":0,"p":[0.0,0.0,0.0],"v":["g","l","p"]}}],"plugin_assessment":[{{"plugin_id":"id","used":true}}],"key_findings":[{{"frame_index":0,"criterion_id":"criterion_id","reason":"short image-grounded reason"}}],"case_summary":"short summary"}}

In each frame prediction, p contains probability_satisfied values and v contains visibility codes in the exact criterion_order."""
            row_requirement = (
                f"exactly {len(request.criteria)} p values and "
                f"{len(request.criteria)} v codes"
            )
        else:
            decision_contract = f"""The visual plugin contains a strict video-OOF calibrated_probability_prior for every frame and criterion. Treat that probability as the numerical prior. You are the final judge because you must decide whether to KEEP or correct every prior after inspecting the real annotated frame, structured detections, Skill, and temporal evidence.

Do NOT regenerate probabilities from zero. For each criterion emit an action code and magnitude:
- K: keep the calibrated prior; this is the default when there is no clear contradiction.
- U: increase the prior because specific visible evidence supports the criterion more strongly.
- D: decrease the prior because specific visible evidence contradicts it.
- X: visibility is insufficient; defer to the calibrated prior rather than turning it into zero.
Magnitude m is 0 for K/X and 1 or 2 for U/D. Each step is a bounded 0.5 log-odds correction applied after your response. U/D requires an image-grounded key_finding or a concrete structured-detection/temporal fact. A predicted box is fallible and is evidence, not ground truth.

Return ONLY one compact JSON object with this exact top-level structure:
{{"criterion_order":{json.dumps(criterion_order)},"frame_predictions":[{{"frame_index":0,"a":["K","U","D"],"m":[0,1,2],"v":["g","l","p"]}}],"plugin_assessment":[{{"plugin_id":"id","used":true}}],"key_findings":[{{"frame_index":0,"criterion_id":"criterion_id","reason":"short image-grounded reason"}}],"case_summary":"short summary"}}

In each frame prediction, a and m are decisions over the exact criterion_order; v contains visibility codes."""
            row_requirement = (
                f"exactly {len(request.criteria)} action codes, "
                f"{len(request.criteria)} magnitudes, and {len(request.criteria)} v codes"
            )
        prompt = f"""You are the frozen foundation MLLM and the FINAL decision maker for a procedural-video assessment.
Small visual models, temporal models, and Skills are evidence plugins. Inspect the real frames, use their structured evidence, resolve conflicts, and make the final decision yourself.

Task: {request.task_id}
The input contains {frame_count} real chronological frames. Each image is preceded by its exact frame_index and timestamp_s.

Minimal criterion identities:
{criterion_text}

{extra}

{decision_contract}
{revision_contract}
Visibility codes are g=good, l=limited, p=poor.

Requirements:
- Return exactly {frame_count} frame_predictions in the supplied order, each with {row_requirement}.
- Use only the supplied frame_index and criterion_id values.
- plugin_assessment must contain each supplied plugin ID exactly once ({json.dumps(supplied_plugin_ids)}); use [] when no plugins were supplied.
- key_findings is optional evidence for at most 8 important, uncertain, or plugin-conflict decisions. Do not explain all {frame_count * len(request.criteria)} predictions.
- The detailed Skill, when present, defines the task but does not itself prove that a frame is positive.
"""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame_index, timestamp_s, jpeg in zip(
            request.frame_ids, request.timestamps_s, request.frame_jpegs,
        ):
            content.append({
                "type": "text",
                "text": f"FRAME frame_index={frame_index}, timestamp_s={timestamp_s:.3f}",
            })
            content.append({"type": "image_url", "image_url": {"url": _data_url(jpeg)}})
        return [{"role": "user", "content": content}]

    @staticmethod
    def validate_ablation_plugins(
        request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> None:
        if ablation.decision_protocol not in {
            "direct_probability", "ordinal_state", "bounded_prior_correction",
        }:
            raise ValueError(f"Unknown MLLM decision protocol: {ablation.decision_protocol}")
        allowed = set(ablation.include_plugin_kinds)
        selected = [
            item for item in request.plugin_evidence if item.plugin_kind in allowed
        ]
        if ablation.require_fact_only_plugins:
            for item in selected:
                item.validate_fact_only()
        if ablation.decision_protocol == "ordinal_state" and ablation.use_plugin_probability_prior:
            raise ValueError("Ordinal foundation-centered judging cannot use a probability prior")

    @staticmethod
    def probability_prior(
        request: MLLMJudgeRequest, ablation: MLLMAblation,
    ) -> list[list[float]] | None:
        if not ablation.use_plugin_probability_prior:
            return None
        allowed = set(ablation.include_plugin_kinds)
        candidates = [
            item.payload["calibrated_probability_prior"]
            for item in request.plugin_evidence
            if item.plugin_kind in allowed
            and "calibrated_probability_prior" in item.payload
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Ablation {ablation.name} requires exactly one calibrated probability prior"
            )
        value = candidates[0]
        criterion_ids = [item.criterion_id for item in request.criteria]
        if value.get("criterion_order") != criterion_ids:
            raise ValueError("Plugin probability-prior criterion order differs from the request")
        probabilities = value.get("frame_probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(request.frame_ids):
            raise ValueError("Plugin probability prior has the wrong frame count")
        output = []
        for row in probabilities:
            if not isinstance(row, list) or len(row) != len(criterion_ids):
                raise ValueError("Plugin probability prior has the wrong criterion count")
            converted = [float(item) for item in row]
            if any(not 0.0 <= item <= 1.0 for item in converted):
                raise ValueError("Plugin probability prior must be in [0,1]")
            output.append(converted)
        return output

    @staticmethod
    def validate_response(
        value: Any, request: MLLMJudgeRequest,
        available_plugin_ids: set[str] | None = None,
        probability_prior: list[list[float]] | None = None,
        log_odds_step: float = 0.5,
        decision_protocol: str = "direct_probability",
    ) -> dict[str, Any]:
        if not isinstance(value, dict) or not isinstance(value.get("frame_predictions"), list):
            raise RuntimeError("MLLM response must contain a frame_predictions list")
        expected_frames = list(zip(request.frame_ids, request.timestamps_s))
        if value.get("criterion_order") != [item.criterion_id for item in request.criteria]:
            raise RuntimeError("MLLM criterion_order differs from the request")
        if len(value["frame_predictions"]) != len(expected_frames):
            raise RuntimeError("MLLM response does not contain exactly one row per input frame")
        criterion_ids = [item.criterion_id for item in request.criteria]
        allowed_plugins = (
            {item.plugin_id for item in request.plugin_evidence}
            if available_plugin_ids is None else set(available_plugin_ids)
        )
        plugin_rows = value.get("plugin_assessment", [])
        if not isinstance(plugin_rows, list):
            raise RuntimeError("MLLM plugin_assessment must be a list")
        by_plugin = {
            item.get("plugin_id"): item for item in plugin_rows if isinstance(item, dict)
        }
        if len(by_plugin) != len(plugin_rows) or set(by_plugin) != allowed_plugins:
            raise RuntimeError("MLLM plugin_assessment differs from the supplied plugins")
        used_plugins = [
            plugin_id for plugin_id, item in by_plugin.items() if item.get("used") is True
        ]
        findings = value.get("key_findings", [])
        if not isinstance(findings, list) or len(findings) > 8:
            raise RuntimeError("MLLM key_findings must be a list of at most 8 items")
        finding_reasons = {}
        for item in findings:
            if not isinstance(item, dict):
                raise RuntimeError("Each MLLM key finding must be an object")
            key = (int(item.get("frame_index")), item.get("criterion_id"))
            if key[0] not in request.frame_ids or key[1] not in criterion_ids:
                raise RuntimeError("MLLM key finding cites an unavailable frame or criterion")
            finding_reasons[key] = str(item.get("reason", ""))
        normalized_frames = []
        visibility_map = {"g": "good", "l": "limited", "p": "poor"}
        for frame_position, (row, (frame_id, timestamp_s)) in enumerate(
            zip(value["frame_predictions"], expected_frames)
        ):
            if int(row.get("frame_index")) != frame_id:
                raise RuntimeError("MLLM frame order or frame_index differs from the request")
            visibilities = row.get("v")
            if not isinstance(visibilities, list) or len(visibilities) != len(criterion_ids):
                raise RuntimeError("MLLM response has the wrong visibility vector length")
            if decision_protocol == "ordinal_state":
                states, confidences = row.get("s"), row.get("c")
                if not isinstance(states, list) or len(states) != len(criterion_ids):
                    raise RuntimeError("MLLM response has the wrong state vector length")
                if not isinstance(confidences, list) or len(confidences) != len(criterion_ids):
                    raise RuntimeError("MLLM response has the wrong confidence vector length")
                state_probability = {
                    "N": {"l": 0.30, "m": 0.15, "h": 0.05},
                    "P": {"l": 0.35, "m": 0.45, "h": 0.55},
                    "F": {"l": 0.65, "m": 0.80, "h": 0.95},
                    "U": {"l": 0.50, "m": 0.50, "h": 0.50},
                }
                decisions = []
                for state, confidence in zip(states, confidences):
                    state = str(state).upper()
                    confidence = str(confidence).lower()
                    if state not in state_probability or confidence not in {"l", "m", "h"}:
                        raise RuntimeError(
                            "MLLM state/confidence codes must be N/P/F/U and l/m/h"
                        )
                    decisions.append((state, confidence, state_probability[state][confidence]))
            elif probability_prior is None:
                probabilities = row.get("p")
                if not isinstance(probabilities, list) or len(probabilities) != len(criterion_ids):
                    raise RuntimeError("MLLM response has the wrong probability vector length")
                decisions = [("direct", None, float(item)) for item in probabilities]
            else:
                actions, magnitudes = row.get("a"), row.get("m")
                if not isinstance(actions, list) or len(actions) != len(criterion_ids):
                    raise RuntimeError("MLLM response has the wrong action vector length")
                if not isinstance(magnitudes, list) or len(magnitudes) != len(criterion_ids):
                    raise RuntimeError("MLLM response has the wrong magnitude vector length")
                decisions = []
                for index, (action, raw_magnitude) in enumerate(zip(actions, magnitudes)):
                    magnitude = int(raw_magnitude)
                    if action not in {"K", "U", "D", "X"}:
                        raise RuntimeError("MLLM action must be K, U, D, or X")
                    if magnitude not in {0, 1, 2}:
                        raise RuntimeError("MLLM correction magnitude must be 0, 1, or 2")
                    if (action in {"K", "X"}) != (magnitude == 0):
                        raise RuntimeError("K/X require magnitude 0; U/D require magnitude 1 or 2")
                    prior = float(probability_prior[frame_position][index])
                    if action in {"K", "X"}:
                        probability = prior
                    else:
                        clipped = min(max(prior, 1e-4), 1.0 - 1e-4)
                        logit = math.log(clipped / (1.0 - clipped))
                        direction = 1.0 if action == "U" else -1.0
                        probability = 1.0 / (1.0 + math.exp(-(logit + direction * log_odds_step * magnitude)))
                    decisions.append((action, magnitude, probability))
            normalized_criteria = []
            for index, criterion_id in enumerate(criterion_ids):
                action, magnitude, probability = decisions[index]
                visibility = visibility_map.get(str(visibilities[index]).lower())
                if not 0.0 <= probability <= 1.0:
                    raise RuntimeError("MLLM probability_satisfied must be in [0,1]")
                if visibility is None:
                    raise RuntimeError("MLLM visibility code must be g, l, or p")
                normalized_criteria.append({
                    "criterion_id": criterion_id,
                    "probability_satisfied": probability,
                    "visibility": visibility,
                    "rationale": finding_reasons.get((frame_id, criterion_id), ""),
                    "used_plugin_ids": used_plugins,
                    "decision_action": action,
                    "correction_magnitude": magnitude,
                    "foundation_state": (
                        action if decision_protocol == "ordinal_state" else None
                    ),
                    "foundation_confidence": (
                        magnitude if decision_protocol == "ordinal_state" else None
                    ),
                    "prior_probability": (
                        float(probability_prior[frame_position][index])
                        if probability_prior is not None else None
                    ),
                })
            normalized_frames.append({
                "frame_index": frame_id,
                "timestamp_s": timestamp_s,
                "criteria": normalized_criteria,
            })
        return {
            "frames": normalized_frames,
            "plugin_assessment": plugin_rows,
            "key_findings": findings,
            "case_summary": str(value.get("case_summary", "")),
        }


def select_plugins(
    plugins: Iterable[PluginEvidence], kinds: Iterable[str],
) -> list[PluginEvidence]:
    allowed = set(kinds)
    return [item for item in plugins if item.plugin_kind in allowed]
