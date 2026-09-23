"""Task-neutral grounded facts exposed by advisory evidence plugins.

The schema intentionally represents observations rather than task verdicts.
Dataset adapters may use domain vocabulary in ``subject``/``value`` and
grounding metadata, while the orchestration core remains domain independent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


_FACT_TYPES = {"entity", "attribute", "relation", "event", "visibility"}


@dataclass(frozen=True)
class GroundedEvidenceFact:
    fact_id: str
    fact_type: str
    subject: str
    predicate: str
    value: Any
    confidence: float | None = None
    frame_index: int | None = None
    timestamp_s: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    grounding: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.fact_id or not self.subject or not self.predicate:
            raise ValueError("Grounded facts require fact_id, subject, and predicate")
        if self.fact_type not in _FACT_TYPES:
            raise ValueError(f"Unsupported grounded fact type: {self.fact_type}")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Grounded fact confidence must be in [0,1]")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("Grounded fact intervals require both start_s and end_s")
        if self.start_s is not None and float(self.end_s) < float(self.start_s):
            raise ValueError("Grounded fact end_s must not precede start_s")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            key: value for key, value in asdict(self).items()
            if value is not None and value != {}
        }


def grounded_fact_payload(
    facts: Iterable[GroundedEvidenceFact], *, modality: str,
    source_description: str, provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a plugin payload that is structurally unable to own task output."""
    rows = [fact.to_dict() for fact in facts]
    return {
        "schema_version": "grounded_evidence_facts_v1",
        "evidence_role": "advisory_observations",
        "modality": str(modality),
        "source_description": str(source_description),
        "final_task_prediction_provided": False,
        "facts": rows,
        "provenance": dict(provenance or {}),
    }

