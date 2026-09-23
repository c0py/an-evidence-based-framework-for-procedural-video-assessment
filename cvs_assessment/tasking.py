from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from .schema import (
    CriterionSpec,
    EvaluationWindowSpec,
    EvidenceQuery,
    TemporalPolicy,
    VerificationExpression,
)


class DatasetAdapter(Protocol):
    """Dataset-specific labels and anchors exposed through a neutral contract."""

    task_id: str

    def locate_window(self, video_id: str | int, **sources: Any) -> dict[str, float]: ...

    def load_intervals(
        self, video_id: str | int, **sources: Any,
    ) -> dict[str, list[tuple[float, float, int]]]: ...


@dataclass
class TaskPackage:
    """A replaceable task instantiation around the domain-neutral core."""

    task_id: str
    task_name: str
    description: str
    criterion_catalog: dict[str, CriterionSpec]
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    allowed_tools: set[str] = field(default_factory=set)
    window_spec: EvaluationWindowSpec = field(default_factory=EvaluationWindowSpec)
    default_temporal_policy: TemporalPolicy = field(default_factory=TemporalPolicy)
    default_logic: str = "AND"
    ordering_constraints: list[tuple[str, str]] = field(default_factory=list)
    allow_dynamic_criteria: bool = False
    dataset_adapter: DatasetAdapter | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def criterion(self, criterion_id: str) -> CriterionSpec:
        if criterion_id not in self.criterion_catalog:
            raise KeyError(f"Task {self.task_id} has no criterion: {criterion_id}")
        criterion = deepcopy(self.criterion_catalog[criterion_id])
        if criterion.evidence_query is None:
            criterion.evidence_query = EvidenceQuery(
                criterion_id=criterion.key,
                requirement=criterion.visual_requirement,
                negative_evidence=criterion.negative_evidence,
                insufficient_evidence=criterion.insufficient_evidence,
            )
        if criterion.temporal_policy is None:
            criterion.temporal_policy = deepcopy(self.default_temporal_policy)
        if not criterion.temporal_parameters:
            criterion.temporal_parameters = dict(criterion.temporal_policy.parameters)
        return criterion

    def artifact_stem(self, video_id: str | int) -> str:
        """Return a filesystem-safe run identifier using task-owned formatting."""
        raw = str(video_id)
        safe_video_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
        template = str(self.metadata.get("run_stem_template", "{task_id}_{video_id}"))
        try:
            numeric_video_id: int | str = int(raw)
        except ValueError:
            numeric_video_id = safe_video_id
        rendered = template.format(
            task_id=self.task_id,
            video_id=safe_video_id,
            video_id_int=numeric_video_id,
        )
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", rendered)

    def visual_evidence_schema(self) -> dict[str, Any]:
        """Return task-owned semantics for replaceable localization tools.

        The domain-neutral core never infers object classes from criterion IDs.
        A task package may omit this schema when no structured localizer exists.
        """

        value = deepcopy(self.metadata.get("visual_evidence_schema", {}))
        value.setdefault("semantic_types", [])
        value.setdefault("semantic_pairs", [])
        value.setdefault("criterion_roi_classes", {})
        unknown = set(value["criterion_roi_classes"]) - set(self.criterion_catalog)
        if unknown:
            raise ValueError(f"Visual evidence schema has unknown criteria: {sorted(unknown)}")
        return value

    def select_criteria(self, specification: str) -> list[CriterionSpec]:
        text = specification.lower()
        selected = []
        for criterion_id in self.criterion_catalog:
            aliases = self.aliases.get(criterion_id, (criterion_id.replace("_", " "),))
            if any(alias.lower() in text for alias in aliases):
                selected.append(self.criterion(criterion_id))
        return selected

    def default_verification_expression(
        self, criterion_ids: list[str], logic: str | None = None,
    ) -> VerificationExpression:
        operator = "ALL" if (logic or self.default_logic) == "AND" else "ANY"
        arguments = [
            VerificationExpression(operator="CRITERION", criterion_id=criterion_id)
            for criterion_id in criterion_ids
        ]
        selected = set(criterion_ids)
        arguments.extend(
            VerificationExpression(
                operator="BEFORE", left_criterion=left, right_criterion=right,
            )
            for left, right in self.ordering_constraints
            if left in selected and right in selected
        )
        return VerificationExpression(
            operator=operator,
            arguments=arguments,
        )

    def planner_context(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "description": self.description,
            "criteria": {
                key: {
                    "title": value.title,
                    "requirement": value.visual_requirement,
                    "required_capabilities": (
                        value.evidence_query.required_capabilities
                        if value.evidence_query else ["visual_state_recognition"]
                    ),
                }
                for key, value in self.criterion_catalog.items()
            },
            "allowed_tools": sorted(self.allowed_tools),
            "window_spec": self.window_spec,
            "allow_dynamic_criteria": self.allow_dynamic_criteria,
            "ordering_constraints": self.ordering_constraints,
        }


class TaskPackageRegistry:
    def __init__(self) -> None:
        self._packages: dict[str, TaskPackage] = {}

    def register(self, package: TaskPackage) -> None:
        if package.task_id in self._packages:
            raise ValueError(f"Task package already registered: {package.task_id}")
        self._packages[package.task_id] = package

    def get(self, task_id: str) -> TaskPackage:
        if task_id not in self._packages:
            raise KeyError(
                f"Unknown task package {task_id!r}; available={sorted(self._packages)}"
            )
        return self._packages[task_id]

    def available(self) -> list[str]:
        return sorted(self._packages)


_DEFAULT_REGISTRY: TaskPackageRegistry | None = None


def default_task_registry() -> TaskPackageRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        from .tasks.cholec_cvs import create_cholec_cvs_package
        from .tasks.industreal import create_industreal_package
        from .tasks.jigsaws import create_jigsaws_suturing_package

        registry = TaskPackageRegistry()
        registry.register(create_cholec_cvs_package())
        registry.register(create_industreal_package())
        registry.register(create_jigsaws_suturing_package())
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY


def load_task_package(task_id: str = "cholec_cvs") -> TaskPackage:
    return default_task_registry().get(task_id)
