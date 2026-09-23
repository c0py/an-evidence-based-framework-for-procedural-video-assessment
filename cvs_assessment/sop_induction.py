from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable


@dataclass(frozen=True)
class DemonstrationContract:
    """The evidence boundary for one demonstration used during SOP induction."""

    demonstration_id: str
    duration_s: float

    def __post_init__(self) -> None:
        if not self.demonstration_id or self.duration_s <= 0:
            raise ValueError("A demonstration requires a non-empty id and positive duration")


@dataclass(frozen=True)
class SOPValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    statistics: dict[str, Any]
    canonical_sha256: str | None


def make_executable_sop(
    value: dict[str, Any],
    demonstrations: Iterable[DemonstrationContract],
    *,
    boundary_tolerance_s: float = 0.25,
    minimum_order_support: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply label-free, representation-only repairs before execution.

    Near-boundary timestamp round-off is clamped to the source duration. Order
    edges are retained only when at least ``minimum_order_support`` demonstrations
    agree and no demonstration contradicts them. A unanimously reversed model
    edge is corrected; a mixed edge is removed rather than made mandatory.
    """

    if boundary_tolerance_s < 0 or minimum_order_support < 1:
        raise ValueError("Executable-SOP repair settings are invalid")
    output = deepcopy(value)
    demo_map = {item.demonstration_id: item for item in demonstrations}
    audit: dict[str, Any] = {
        "timestamp_clamps": [],
        "dropped_evidence": [],
        "text_repairs": [],
        "kept_order_edges": [],
        "reversed_order_edges": [],
        "dropped_order_edges": [],
        "semantic_fields_changed": False,
    }
    for step in output.get("steps", []):
        for field in ("name", "precondition", "action", "postcondition"):
            text = str(step.get(field, "")).strip()
            if text.count("(") > text.count(")"):
                repaired_text = text.rsplit("(", 1)[0].rstrip()
                if repaired_text:
                    step[field] = repaired_text
                    audit["text_repairs"].append({
                        "step_id": step.get("step_id"), "field": field,
                        "reason": "remove_incomplete_parenthetical_fragment",
                        "from": text, "to": repaired_text,
                    })
        valid_evidence = []
        for evidence in step.get("evidence", []):
            demo_id = str(evidence.get("demonstration_id", ""))
            if demo_id not in demo_map:
                audit["dropped_evidence"].append({
                    "step_id": step.get("step_id"), "reason": "unknown_demonstration",
                    "evidence": evidence,
                })
                continue
            try:
                start_s, end_s = float(evidence["start_s"]), float(evidence["end_s"])
            except (KeyError, TypeError, ValueError):
                audit["dropped_evidence"].append({
                    "step_id": step.get("step_id"), "reason": "invalid_timestamp",
                    "evidence": evidence,
                })
                continue
            duration_s = demo_map[demo_id].duration_s
            original = [start_s, end_s]
            if -boundary_tolerance_s <= start_s < 0:
                start_s = 0.0
            if duration_s < end_s <= duration_s + boundary_tolerance_s:
                end_s = duration_s
            if start_s < 0 or end_s <= start_s or end_s > duration_s:
                audit["dropped_evidence"].append({
                    "step_id": step.get("step_id"), "reason": "out_of_bounds",
                    "evidence": evidence,
                })
                continue
            repaired = dict(evidence, start_s=start_s, end_s=end_s)
            valid_evidence.append(repaired)
            if original != [start_s, end_s]:
                audit["timestamp_clamps"].append({
                    "step_id": step.get("step_id"),
                    "demonstration_id": demo_id,
                    "from": original,
                    "to": [start_s, end_s],
                })
        step["evidence"] = valid_evidence

    steps = {str(step.get("step_id")): step for step in output.get("steps", [])}

    def intervals(step_id: str) -> dict[str, tuple[float, float]]:
        return {
            str(item["demonstration_id"]): (float(item["start_s"]), float(item["end_s"]))
            for item in steps[step_id].get("evidence", [])
        }

    executable_edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for edge in output.get("order_edges", []):
        before, after = str(edge.get("before", "")), str(edge.get("after", ""))
        if before not in steps or after not in steps or before == after:
            audit["dropped_order_edges"].append({
                "edge": edge, "reason": "invalid_step_reference",
            })
            continue
        left, right = intervals(before), intervals(after)
        support = contradiction = overlap = 0
        for demo_id in sorted(set(left).intersection(right)):
            left_start, left_end = left[demo_id]
            right_start, right_end = right[demo_id]
            if left_end <= right_start:
                support += 1
            elif right_end <= left_start:
                contradiction += 1
            else:
                overlap += 1
        counts = {
            "support": support, "contradiction": contradiction, "overlap": overlap,
        }
        selected: tuple[str, str] | None = None
        if support >= minimum_order_support and contradiction == 0:
            selected = (before, after)
            audit["kept_order_edges"].append({"edge": edge, **counts})
        elif contradiction >= minimum_order_support and support == 0:
            selected = (after, before)
            audit["reversed_order_edges"].append({
                "from": [before, after], "to": [after, before], **counts,
            })
        else:
            audit["dropped_order_edges"].append({
                "edge": edge, "reason": "mixed_or_insufficient_demo_order", **counts,
            })
        if selected is not None and selected not in seen_edges:
            executable_edges.append({
                "before": selected[0], "after": selected[1],
                "basis": "observed_consensus",
            })
            seen_edges.add(selected)
    output["order_edges"] = executable_edges
    step_names = [str(step.get("name", "")).strip() for step in output.get("steps", [])]
    output["task_summary"] = "Procedure milestones: " + "; ".join(
        name for name in step_names if name
    ) + "."
    output["generation_note"] = (
        f"Consolidated from {len(demo_map)} successful demonstrations; source-boundary "
        "and cross-demonstration order checks are recorded in the execution audit."
    )
    audit["output_sha256"] = canonical_sha256(output)
    return output, audit


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _has_cycle(nodes: Iterable[str], edges: list[tuple[str, str]]) -> bool:
    adjacency = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for before, after in edges:
        adjacency[before].append(after)
        indegree[after] += 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited != len(adjacency)


def validate_induced_sop(
    value: dict[str, Any],
    demonstrations: Iterable[DemonstrationContract],
    allowed_tools: Iterable[str],
    *,
    require_grounding: bool = True,
) -> SOPValidationResult:
    """Validate a demonstration-derived SOP without deciding task semantics.

    This validator deliberately checks only properties that can be audited without
    an expert label: schema completeness, source-bounded timestamps, registered
    tools, and an acyclic partial-order graph. Semantic quality is evaluated by
    held-out replay and known counterfactual corruptions.
    """

    errors: list[str] = []
    warnings: list[str] = []
    demo_map = {item.demonstration_id: item for item in demonstrations}
    tools = set(allowed_tools)
    if not isinstance(value, dict):
        return SOPValidationResult(False, ("SOP must be a JSON object",), (), {}, None)
    if not str(value.get("task_summary", "")).strip():
        errors.append("task_summary is missing")
    steps = value.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("steps must be a non-empty array")
        steps = []
    step_ids: list[str] = []
    grounded_steps = 0
    cited_demos: set[str] = set()
    tool_bound_steps = 0
    for index, step in enumerate(steps):
        prefix = f"steps[{index}]"
        if not isinstance(step, dict):
            errors.append(f"{prefix} must be an object")
            continue
        step_id = str(step.get("step_id", "")).strip()
        if not step_id:
            errors.append(f"{prefix}.step_id is missing")
        elif step_id in step_ids:
            errors.append(f"duplicate step_id: {step_id}")
        else:
            step_ids.append(step_id)
        for field in ("name", "precondition", "action", "postcondition"):
            if not str(step.get(field, "")).strip():
                errors.append(f"{prefix}.{field} is missing")
        required_tools = step.get("required_tools")
        if not isinstance(required_tools, list) or not required_tools:
            errors.append(f"{prefix}.required_tools must be non-empty")
        else:
            unknown = sorted(set(map(str, required_tools)) - tools)
            if unknown:
                errors.append(f"{prefix} names unregistered tools: {unknown}")
            else:
                tool_bound_steps += 1
        evidence = step.get("evidence")
        valid_evidence = 0
        if not isinstance(evidence, list):
            errors.append(f"{prefix}.evidence must be an array")
            evidence = []
        for evidence_index, item in enumerate(evidence):
            evidence_prefix = f"{prefix}.evidence[{evidence_index}]"
            if not isinstance(item, dict):
                errors.append(f"{evidence_prefix} must be an object")
                continue
            demo_id = str(item.get("demonstration_id", ""))
            if demo_id not in demo_map:
                errors.append(f"{evidence_prefix} cites an unknown demonstration: {demo_id}")
                continue
            try:
                start_s, end_s = float(item["start_s"]), float(item["end_s"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{evidence_prefix} has invalid timestamps")
                continue
            if start_s < 0 or end_s <= start_s or end_s > demo_map[demo_id].duration_s + 1e-6:
                errors.append(
                    f"{evidence_prefix} interval [{start_s}, {end_s}] is outside "
                    f"[0, {demo_map[demo_id].duration_s}]"
                )
                continue
            valid_evidence += 1
            cited_demos.add(demo_id)
        if valid_evidence:
            grounded_steps += 1
        elif require_grounding:
            errors.append(f"{prefix} has no valid source evidence")
        else:
            warnings.append(f"{prefix} has no valid source evidence")

    edge_pairs: list[tuple[str, str]] = []
    edges = value.get("order_edges", [])
    if not isinstance(edges, list):
        errors.append("order_edges must be an array")
        edges = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"order_edges[{index}] must be an object")
            continue
        before, after = str(edge.get("before", "")), str(edge.get("after", ""))
        if before not in step_ids or after not in step_ids or before == after:
            errors.append(f"order_edges[{index}] references invalid steps: {before} -> {after}")
            continue
        pair = (before, after)
        if pair in edge_pairs:
            errors.append(f"duplicate order edge: {before} -> {after}")
            continue
        edge_pairs.append(pair)
    if step_ids and _has_cycle(step_ids, edge_pairs):
        errors.append("order_edges contain a cycle")

    statistics = {
        "step_count": len(steps),
        "grounded_step_count": grounded_steps,
        "grounded_step_rate": grounded_steps / len(steps) if steps else 0.0,
        "tool_bound_step_count": tool_bound_steps,
        "tool_bound_step_rate": tool_bound_steps / len(steps) if steps else 0.0,
        "order_edge_count": len(edge_pairs),
        "cited_demonstration_count": len(cited_demos),
        "available_demonstration_count": len(demo_map),
    }
    return SOPValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        statistics=statistics,
        canonical_sha256=canonical_sha256(value) if not errors else None,
    )
