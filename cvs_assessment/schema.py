from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Verdict = Literal["pass", "fail", "uncertain"]
EvidenceState = Literal["positive", "negative", "unknown", "unclassified"]


@dataclass
class EvidenceQuery:
    """Task-neutral request passed from a generated skill to an evidence tool."""

    criterion_id: str
    requirement: str
    negative_evidence: str = ""
    insufficient_evidence: str = "The available observation is insufficient to judge."
    modalities: list[str] = field(default_factory=lambda: ["rgb"])
    required_capabilities: list[str] = field(
        default_factory=lambda: ["visual_state_recognition"],
    )
    output_states: list[str] = field(
        default_factory=lambda: ["satisfied", "violated", "unknown"],
    )
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalPolicy:
    """Declarative temporal operator configuration; no domain names are assumed."""

    operator: str = "stable_state"
    parameters: dict[str, Any] = field(default_factory=dict)
    source_state: str | None = None
    target_state: str | None = "satisfied"
    reference_criterion: str | None = None


@dataclass
class VerificationExpression:
    """Recursive task-neutral verification expression."""

    operator: str
    criterion_id: str | None = None
    expected_verdict: Verdict = "pass"
    arguments: list["VerificationExpression"] = field(default_factory=list)
    left_criterion: str | None = None
    right_criterion: str | None = None


@dataclass
class EvaluationWindowSpec:
    """How a task package identifies the video region eligible for evidence."""

    locator_tool: str = "locate_evaluation_window"
    start_anchor: str | None = None
    end_anchor: str | None = None
    include_start: bool = True
    include_end: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCapability:
    """Machine-readable contract advertised by an atomic tool."""

    tool_id: str
    role: Literal[
        "context", "sampler", "evidence_provider", "temporal_operator",
        "verifier", "artifact_writer", "other",
    ] = "other"
    capabilities: list[str] = field(default_factory=list)
    input_schema: str = "dict"
    output_schema: str = "dict"
    supported_tasks: list[str] = field(default_factory=lambda: ["*"])
    version: str = "1"


@dataclass
class CriterionSpec:
    key: str
    title: str
    visual_requirement: str
    tool: str = "criterion_visual_evidence"
    temporal_rule: str = "stable_interval_before_anchor"
    negative_evidence: str = ""
    insufficient_evidence: str = "The target state is not sufficiently visible to judge."
    tool_sequence: list[str] = field(default_factory=lambda: ["criterion_visual_evidence", "stable_evidence_aggregation"])
    temporal_parameters: dict[str, float] = field(default_factory=dict)
    decision_rule: str = "A stable positive interval is required before the anchor phase."
    evidence_query: EvidenceQuery | None = None
    temporal_policy: TemporalPolicy | None = None


@dataclass
class AssessmentPlan:
    plan_version: str
    source_specification: str
    criteria: list[CriterionSpec]
    logic: Literal["AND", "OR"]
    candidate_phase: str
    anchor_phase: str
    invalid_specification: bool = False
    planner: str = "rule_based_spec_parser"
    planner_model: str | None = None
    generation_notes: list[str] = field(default_factory=list)
    task_id: str = "unspecified_task"
    task_name: str = "Procedural video assessment"
    window_spec: EvaluationWindowSpec | None = None
    verification_expression: VerificationExpression | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScorePoint:
    time_s: float
    score: float
    raw_score: float
    quality: float
    # Structured tools set this explicitly. Legacy/checkpoint scorers leave it
    # unclassified and the temporal tool deterministically derives a three-way
    # state from its configured hysteresis thresholds.
    evidence_state: EvidenceState = "unclassified"
    state_confidence: float = 0.0


@dataclass
class TemporalEvidencePoint:
    """Universal evidence representation emitted by task-specific or general tools."""

    criterion_id: str
    time_s: float
    state: Literal["satisfied", "violated", "unknown"]
    confidence: float
    observability: float = 1.0
    source_tool: str = ""
    grounding: dict[str, Any] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def to_score_point(self) -> ScorePoint:
        mapped = {
            "satisfied": "positive",
            "violated": "negative",
            "unknown": "unknown",
        }[self.state]
        score = self.confidence if self.state == "satisfied" else 0.0
        return ScorePoint(
            time_s=self.time_s,
            score=score * self.observability,
            raw_score=score,
            quality=self.observability,
            evidence_state=mapped,
            state_confidence=self.confidence,
        )


@dataclass
class VisualEvidencePoint:
    """A model observation tied to one video time and one assessment criterion."""

    criterion: str
    time_s: float
    supports_criterion: Literal["yes", "no", "unknown"]
    confidence: float
    visibility: Literal["good", "limited", "poor"]
    observed_facts: list[str]
    rationale: str
    frame_times_s: list[float]


@dataclass
class EvidenceInterval:
    start_s: float
    end_s: float
    confidence: float
    mean_score: float
    duration_s: float
    representative_time_s: float
    representative_frame: str | None = None
    supporting_point_count: int = 0
    unknown_point_count: int = 0
    assessable_coverage: float = 1.0


@dataclass
class CriterionEvidence:
    """Temporally aggregated evidence with explicit observability semantics."""

    positive_intervals: list[EvidenceInterval] = field(default_factory=list)
    negative_intervals: list[EvidenceInterval] = field(default_factory=list)
    positive_point_count: int = 0
    negative_point_count: int = 0
    unknown_point_count: int = 0
    total_point_count: int = 0
    assessable_coverage: float = 0.0
    explicit_negative_confidence: float = 0.0
    evidence_semantics: str = "positive_negative_unknown"


@dataclass
class CriterionVerdict:
    key: str
    verdict: Verdict
    confidence: float
    evidence_intervals: list[EvidenceInterval] = field(default_factory=list)
    representative_frame: str | None = None
    reason: str | None = None
    assessable_coverage: float = 0.0
    positive_point_count: int = 0
    negative_point_count: int = 0
    unknown_point_count: int = 0


@dataclass
class AssessmentResult:
    video_id: str
    evaluation_window: dict[str, float]
    overall_verdict: Verdict
    overall_confidence: float
    criteria: list[CriterionVerdict]
    backend: str
    development_oracle: bool
    notes: list[str]
    task_id: str = "unspecified_task"
    task_name: str = "Procedural video assessment"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
