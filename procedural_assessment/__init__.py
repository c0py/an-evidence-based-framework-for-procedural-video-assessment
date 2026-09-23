"""Domain-neutral public API for foundation-MLLM procedural assessment.

``cvs_assessment`` remains as a compatibility namespace for existing experiment
scripts. New tasks should import this package and register a ``TaskPackage``.
"""

from cvs_assessment.execution import SkillExecutor
from cvs_assessment.evidence_backends import (
    EvidenceBackendRegistry,
    EvidenceToolInstance,
    default_evidence_backends,
)
from cvs_assessment.pipeline import AssessmentPipeline
from cvs_assessment.mllm_orchestration import (
    CriterionContract,
    FrozenMLLMJudge,
    MLLMAblation,
    MLLMJudgeRequest,
    PluginEvidence,
    FOUNDATION_CENTERED_ABLATIONS,
    STANDARD_ABLATIONS,
)
from cvs_assessment.grounded_evidence import (
    GroundedEvidenceFact,
    grounded_fact_payload,
)
from cvs_assessment.tool_reliability import (
    HighPrecisionToolPolicy,
    fit_high_precision_tool_policy,
)
from cvs_assessment.sop_induction import (
    DemonstrationContract,
    SOPValidationResult,
    canonical_sha256,
    make_executable_sop,
    validate_induced_sop,
)
from cvs_assessment.schema import (
    AssessmentPlan,
    AssessmentResult,
    CriterionEvidence,
    CriterionSpec,
    CriterionVerdict,
    EvaluationWindowSpec,
    EvidenceInterval,
    EvidenceQuery,
    ScorePoint,
    TemporalEvidencePoint,
    TemporalPolicy,
    ToolCapability,
    VerificationExpression,
)
from cvs_assessment.tasking import (
    DatasetAdapter,
    TaskPackage,
    TaskPackageRegistry,
    default_task_registry,
    load_task_package,
)
from cvs_assessment.temporal import (
    StableEvidenceAggregator,
    TemporalOperatorRegistry,
    default_temporal_operators,
)
from cvs_assessment.tools import ToolRegistry
from cvs_assessment.verifier import ExplicitVerifier

__all__ = [
    "AssessmentPipeline", "AssessmentPlan", "AssessmentResult", "CriterionEvidence",
    "CriterionContract", "FrozenMLLMJudge", "MLLMAblation", "MLLMJudgeRequest",
    "PluginEvidence", "STANDARD_ABLATIONS", "FOUNDATION_CENTERED_ABLATIONS",
    "GroundedEvidenceFact", "grounded_fact_payload",
    "HighPrecisionToolPolicy", "fit_high_precision_tool_policy",
    "DemonstrationContract", "SOPValidationResult", "canonical_sha256",
    "make_executable_sop", "validate_induced_sop",
    "CriterionSpec", "CriterionVerdict",
    "DatasetAdapter", "EvaluationWindowSpec", "EvidenceBackendRegistry",
    "EvidenceInterval", "EvidenceQuery", "EvidenceToolInstance", "ExplicitVerifier",
    "ScorePoint",
    "SkillExecutor", "StableEvidenceAggregator", "TaskPackage", "TaskPackageRegistry",
    "TemporalEvidencePoint", "TemporalOperatorRegistry", "TemporalPolicy",
    "ToolCapability", "ToolRegistry", "VerificationExpression",
    "default_evidence_backends", "default_task_registry",
    "default_temporal_operators", "load_task_package",
]
