from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schema import (
    CriterionEvidence, CriterionSpec, EvidenceQuery, ScorePoint,
    TemporalEvidencePoint, TemporalPolicy,
)
from .tools import ToolRegistry


@dataclass
class CriterionExecution:
    scores: list[ScorePoint]
    evidence: CriterionEvidence
    executed_tools: list[str]


class SkillExecutor:
    """Execute a compiled criterion skill through the registered tool contract."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute_criterion(
        self,
        criterion: CriterionSpec,
        timestamps: list[float],
        temporal_parameters: dict[str, float],
    ) -> CriterionExecution:
        raw_points: list[ScorePoint] | None = None
        smooth_points: list[ScorePoint] | None = None
        evidence: CriterionEvidence | None = None
        executed: list[str] = []

        for tool_name in criterion.tool_sequence:
            contract = self.registry.contract(tool_name)
            role = contract.role
            if role == "evidence_provider" or tool_name == "criterion_visual_evidence":
                query = criterion.evidence_query or EvidenceQuery(
                    criterion_id=criterion.key,
                    requirement=criterion.visual_requirement,
                    negative_evidence=criterion.negative_evidence,
                    insufficient_evidence=criterion.insufficient_evidence,
                )
                arguments = {
                    "criterion": criterion.key,
                    "timestamps": timestamps,
                    "visual_requirement": criterion.visual_requirement,
                }
                if role == "evidence_provider":
                    arguments["evidence_query"] = asdict(query)
                raw_points = self.registry.call(tool_name, **arguments)
                raw_points = [
                    point.to_score_point() if isinstance(point, TemporalEvidencePoint) else point
                    for point in raw_points
                ]
            elif role == "temporal_operator" or tool_name in {
                "stable_evidence_aggregation", "state_transition_aggregation",
            }:
                if raw_points is None:
                    raise RuntimeError(
                        f"Skill for {criterion.key} requested aggregation before visual evidence"
                    )
                policy = criterion.temporal_policy or TemporalPolicy(
                    operator=criterion.temporal_rule,
                    parameters=temporal_parameters,
                )
                arguments = {
                    "points": raw_points,
                    "temporal_parameters": temporal_parameters,
                }
                if role == "temporal_operator":
                    arguments["temporal_policy"] = asdict(policy)
                smooth_points, evidence = self.registry.call(tool_name, **arguments)
            elif role in {"context", "sampler"} or tool_name in {
                "locate_evaluation_window", "sample_timestamps",
            }:
                # These plan-level tools have already populated the execution
                # context. Recording reuse makes the generated workflow auditable
                # without decoding/sampling the same window once per criterion.
                self.registry.calls.append({
                    "tool": tool_name,
                    "arguments": {"criterion": criterion.key},
                    "result": {"reused_plan_context": True},
                })
            elif role == "verifier" or tool_name == "explicit_verifier":
                # Verification requires evidence from every criterion, so it is
                # deliberately executed once after all criterion workflows.
                self.registry.calls.append({
                    "tool": tool_name,
                    "arguments": {"criterion": criterion.key},
                    "result": {"deferred_until_all_criteria": True},
                })
            else:
                raise KeyError(f"Unsupported executable skill tool: {tool_name}")
            executed.append(tool_name)

        if raw_points is None or smooth_points is None or evidence is None:
            raise RuntimeError(
                f"Skill for {criterion.key} did not produce temporally aggregated visual evidence"
            )
        return CriterionExecution(smooth_points, evidence, executed)
