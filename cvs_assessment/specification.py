from __future__ import annotations

from pathlib import Path
import json
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schema import (
    AssessmentPlan, CriterionSpec, EvidenceQuery, TemporalPolicy,
    VerificationExpression,
)
from .tasking import TaskPackage, load_task_package


# Backward-compatible names for older Cholec experiment scripts.  Their values
# come from the task plug-in; the planner core itself does not define CVS
# semantics, phases, or tool lists.
_LEGACY_CHOLEC_PACKAGE = load_task_package("cholec_cvs")
CATALOG = _LEGACY_CHOLEC_PACKAGE.criterion_catalog
ALIASES = _LEGACY_CHOLEC_PACKAGE.aliases
ALLOWED_TOOLS = _LEGACY_CHOLEC_PACKAGE.allowed_tools
ALLOWED_PHASES = {
    anchor for anchor in (
        _LEGACY_CHOLEC_PACKAGE.window_spec.start_anchor,
        _LEGACY_CHOLEC_PACKAGE.window_spec.end_anchor,
    ) if anchor
}


class SpecificationPlanner:
    """Deterministic planner; replace `plan` with an LLM adapter when desired."""

    def __init__(self, task_package: TaskPackage | None = None) -> None:
        self.task_package = task_package or load_task_package("cholec_cvs")

    def plan(self, specification: str) -> AssessmentPlan:
        text = specification.lower()
        criteria = self.task_package.select_criteria(specification)
        # "either" is an unambiguous intervention marker; word-boundary matching survives line wrapping.
        logic = "OR" if re.search(r"\beither\b", text) and "all three" not in text else "AND"
        invalid = not criteria or ("unrelated" in text) or ("do not assess" in text)
        window = self.task_package.window_spec
        return AssessmentPlan(
            plan_version="0.1",
            source_specification=specification,
            criteria=criteria,
            logic=logic,
            candidate_phase=window.start_anchor or "procedure_start",
            anchor_phase=window.end_anchor or "procedure_end",
            invalid_specification=invalid,
            task_id=self.task_package.task_id,
            task_name=self.task_package.task_name,
            window_spec=window,
            verification_expression=self.task_package.default_verification_expression(
                [criterion.key for criterion in criteria], logic,
            ),
        )


class LLMSpecificationPlanner:
    """Generate a validated assessment skill through an OpenAI-compatible LLM.

    The model may select criteria, evidence semantics, tools, temporal parameters,
    and overall logic.  It cannot invent executable tools or criterion identifiers:
    generated JSON is validated against the local registry contract before use.
    """

    def __init__(
        self, base_url: str, model: str, api_key: str = "EMPTY",
        timeout_s: float = 120.0, task_package: TaskPackage | None = None,
    ) -> None:
        if not model:
            raise ValueError("planner.model is required for the LLM planner")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.task_package = task_package or load_task_package("cholec_cvs")
        self.request_log: list[dict[str, Any]] = []

    def plan(self, specification: str) -> AssessmentPlan:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"Compile this SOP into an assessment skill:\n\n{specification}"},
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_error: RuntimeError | None = None
        for attempt in range(2):
            raw = self._post(payload)
            try:
                value = self._parse_json(raw)
                return self._validate_plan(value, specification)
            except RuntimeError as exc:
                last_error = exc
                if attempt == 1:
                    break
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"The generated skill failed compilation: {exc}. Return a corrected complete JSON object."},
                ])
        raise RuntimeError(f"LLM planner failed schema validation after one repair: {last_error}")

    def _system_prompt(self) -> str:
        catalog = {
            key: {"title": item.title, "canonical_visual_requirement": item.visual_requirement}
            for key, item in self.task_package.criterion_catalog.items()
        }
        window = self.task_package.window_spec
        temporal_tool = (
            "state_transition_aggregation"
            if self.task_package.default_temporal_policy.operator == "persistent_state_transition"
            else "stable_evidence_aggregation"
        )
        defaults = self.task_package.default_temporal_policy.parameters
        return (
            "You compile natural-language procedural-video specifications into executable assessment skills. "
            "You plan evidence acquisition but never issue a video verdict. Return only one JSON object. "
            f"Task package: {self.task_package.task_id}. Task description: {self.task_package.description}. "
            f"Allowed criterion catalog: {json.dumps(catalog)}. "
            f"Allowed tools: {sorted(self.task_package.allowed_tools)}. "
            f"Declared ordering constraints: {self.task_package.ordering_constraints}. "
            f"Use candidate_phase={window.start_anchor or 'procedure_start'!r} and "
            f"anchor_phase={window.end_anchor or 'procedure_end'!r}. "
            "Required JSON fields: criteria (array), logic ('AND' or 'OR'), candidate_phase, anchor_phase, "
            "invalid_specification (boolean), generation_notes (array of strings). Each criteria item must contain "
            "key, title, visual_requirement, negative_evidence, insufficient_evidence, tool_sequence, "
            "temporal_rule, temporal_parameters, and decision_rule. temporal_parameters must contain numeric "
            "smoothing_seconds, on_threshold, off_threshold, min_stable_seconds, and max_gap_seconds. "
            "Use only catalog keys and allowed tools. Include criterion_visual_evidence and "
            f"{temporal_tool} in every valid criterion tool_sequence. Use invalid_specification=true "
            "when the SOP has no grounded assessment target or is contradictory. Evidence insufficiency must map "
            "to uncertain; lack of observed positive evidence alone is not explicit negative evidence. "
            "negative_evidence and insufficient_evidence must be descriptive sentences, never booleans or label words. "
            f"Unless the SOP explicitly specifies temporal values, use these task defaults: {json.dumps(defaults)}."
        )

    def _post(self, payload: dict[str, Any]) -> str:
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM planner request failed with HTTP {exc.code}: {detail[:500]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach LLM planner at {self.base_url}: {exc.reason}") from exc
        self.request_log.append({
            "latency_s": time.monotonic() - started,
            "usage": body.get("usage", {}),
            "raw_response": body.get("choices", [{}])[0].get("message", {}).get("content"),
        })
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM planner returned no chat completion: {body}") from exc

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM planner returned invalid JSON: {text[:1000]}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("LLM planner output must be a JSON object")
        return value

    def _validate_plan(self, value: dict[str, Any], source: str) -> AssessmentPlan:
        try:
            logic = value["logic"]
            invalid = value["invalid_specification"]
            raw_criteria = value["criteria"]
            candidate_phase = str(value["candidate_phase"])
            anchor_phase = str(value["anchor_phase"])
        except KeyError as exc:
            raise RuntimeError(f"LLM planner omitted required field: {exc.args[0]}") from exc
        if logic not in {"AND", "OR"} or not isinstance(invalid, bool) or not isinstance(raw_criteria, list):
            raise RuntimeError("LLM planner returned invalid logic, invalid_specification, or criteria")
        window = self.task_package.window_spec
        allowed_anchors = {
            window.start_anchor or "procedure_start",
            window.end_anchor or "procedure_end",
        }
        if candidate_phase not in allowed_anchors or anchor_phase not in allowed_anchors or candidate_phase == anchor_phase:
            raise RuntimeError(f"LLM planner selected non-executable phases: {candidate_phase}, {anchor_phase}")
        criteria: list[CriterionSpec] = []
        seen: set[str] = set()
        for raw in raw_criteria:
            catalog = self.task_package.criterion_catalog
            if not isinstance(raw, dict) or raw.get("key") not in catalog:
                raise RuntimeError(f"LLM planner selected an unknown criterion: {raw}")
            key = raw["key"]
            if key in seen:
                raise RuntimeError(f"LLM planner duplicated criterion: {key}")
            seen.add(key)
            tools = raw.get("tool_sequence")
            if not isinstance(tools, list) or not tools or any(tool not in self.task_package.allowed_tools for tool in tools):
                raise RuntimeError(f"LLM planner selected invalid tools for {key}: {tools}")
            if len(set(tools)) != len(tools):
                raise RuntimeError(f"LLM planner duplicated executable tools for {key}: {tools}")
            temporal_tools = {"stable_evidence_aggregation", "state_transition_aggregation"}
            if "criterion_visual_evidence" not in tools or not temporal_tools.intersection(tools):
                raise RuntimeError(f"LLM planner omitted required evidence tools for {key}")
            aggregation_tool = next(tool for tool in tools if tool in temporal_tools)
            if tools.index("criterion_visual_evidence") > tools.index(aggregation_tool):
                raise RuntimeError(f"LLM planner placed aggregation before visual evidence for {key}")
            temporal = raw.get("temporal_parameters")
            temporal_keys = {"smoothing_seconds", "on_threshold", "off_threshold", "min_stable_seconds", "max_gap_seconds"}
            if not isinstance(temporal, dict) or not temporal_keys.issubset(temporal):
                raise RuntimeError(f"LLM planner returned incomplete temporal parameters for {key}")
            if any(not isinstance(temporal[name], (int, float)) for name in temporal_keys):
                raise RuntimeError(f"LLM planner returned nonnumeric temporal parameters for {key}")
            if float(temporal["off_threshold"]) > float(temporal["on_threshold"]):
                raise RuntimeError(f"LLM planner returned invalid hysteresis thresholds for {key}")
            negative = str(raw.get("negative_evidence", ""))
            insufficient = str(raw.get("insufficient_evidence", ""))
            if len(negative.split()) < 5 or len(insufficient.split()) < 5:
                raise RuntimeError(f"LLM planner returned non-descriptive evidence semantics for {key}")
            criteria.append(CriterionSpec(
                key=key,
                title=str(raw.get("title") or catalog[key].title),
                visual_requirement=str(raw.get("visual_requirement") or catalog[key].visual_requirement),
                negative_evidence=negative,
                insufficient_evidence=insufficient,
                tool_sequence=[str(tool) for tool in tools],
                temporal_rule=str(raw.get("temporal_rule", "stable_interval_before_anchor")),
                temporal_parameters={name: float(temporal[name]) for name in temporal_keys},
                decision_rule=str(raw.get("decision_rule", "")),
                evidence_query=EvidenceQuery(
                    criterion_id=key,
                    requirement=str(raw.get("visual_requirement") or catalog[key].visual_requirement),
                    negative_evidence=negative,
                    insufficient_evidence=insufficient,
                    required_capabilities=(
                        catalog[key].evidence_query.required_capabilities
                        if catalog[key].evidence_query else ["visual_state_recognition"]
                    ),
                ),
                temporal_policy=TemporalPolicy(
                    operator=(
                        catalog[key].temporal_policy.operator
                        if catalog[key].temporal_policy else str(raw.get("temporal_rule", "stable_state"))
                    ),
                    parameters={name: float(temporal[name]) for name in temporal_keys},
                ),
            ))
        if not invalid and not criteria:
            raise RuntimeError("LLM planner produced a valid plan without criteria")
        notes = value.get("generation_notes", [])
        if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
            raise RuntimeError("LLM planner generation_notes must be an array of strings")
        return AssessmentPlan(
            plan_version="0.2-llm",
            source_specification=source,
            criteria=criteria,
            logic=logic,
            candidate_phase=candidate_phase,
            anchor_phase=anchor_phase,
            invalid_specification=invalid,
            planner="llm_specification_planner",
            planner_model=self.model,
            generation_notes=notes,
            task_id=self.task_package.task_id,
            task_name=self.task_package.task_name,
            window_spec=self.task_package.window_spec,
            verification_expression=self.task_package.default_verification_expression(
                [criterion.key for criterion in criteria], logic,
            ),
        )


def load_plan_with_trace(
    spec_path: str | Path, planner_config: dict[str, Any] | None = None,
    task_package: TaskPackage | None = None,
) -> tuple[AssessmentPlan, list[dict[str, Any]]]:
    text = Path(spec_path).read_text(encoding="utf-8")
    config = planner_config or {"backend": "rule_based"}
    backend = config.get("backend", "rule_based")
    package = task_package or load_task_package(str(config.get("task_id", "cholec_cvs")))
    if backend == "rule_based":
        return SpecificationPlanner(package).plan(text), []
    if backend == "compiled_llm_skill":
        compiled_path = config.get("compiled_plan_path")
        if not compiled_path:
            raise ValueError("planner.compiled_plan_path is required for compiled_llm_skill")
        try:
            value = json.loads(Path(compiled_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to load compiled LLM skill: {compiled_path}") from exc
        compiled_source = str(value.get("source_specification", ""))
        if compiled_source.strip() != text.strip():
            raise RuntimeError(
                "Compiled LLM skill was generated for a different SOP; recompile before execution"
            )
        original_model = str(value.get("planner_model") or "compiled-llm")
        validator = LLMSpecificationPlanner(
            base_url="compiled://local", model=original_model, task_package=package,
        )
        try:
            plan = validator._validate_plan(value, compiled_source)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Compiled LLM skill failed executable contract validation: {compiled_path}: {exc}"
            ) from exc
        return plan, [{
            "compiled_skill_reuse": True,
            "compiled_plan_path": str(Path(compiled_path).resolve()),
            "original_planner_model": original_model,
        }]
    if backend == "llm":
        planner = LLMSpecificationPlanner(
            base_url=config.get("base_url", "http://127.0.0.1:8000/v1"),
            model=config.get("model", ""),
            api_key=config.get("api_key", "EMPTY"),
            timeout_s=float(config.get("timeout_s", 120)),
            task_package=package,
        )
        plan = planner.plan(text)
        return plan, planner.request_log
    raise ValueError(f"Unsupported planner backend: {backend}")


def load_plan(
    spec_path: str | Path, planner_config: dict[str, Any] | None = None,
    task_package: TaskPackage | None = None,
) -> AssessmentPlan:
    plan, _ = load_plan_with_trace(spec_path, planner_config, task_package)
    return plan
