from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..annotations import load_cvs_intervals, load_phase_starts
from ..schema import CriterionSpec, EvaluationWindowSpec, EvidenceQuery, TemporalPolicy
from ..tasking import TaskPackage


@dataclass
class CholecCvsDatasetAdapter:
    task_id: str = "cholec_cvs"

    def locate_window(self, video_id: str | int, **sources: Any) -> dict[str, float]:
        phase_path = sources["phase_annotation_path"]
        starts = load_phase_starts(phase_path)
        start = starts["CalotTriangleDissection"]
        end = starts["ClippingCutting"]
        return {"start_s": start, "end_s": end, "anchor_s": end}

    def load_intervals(
        self, video_id: str | int, **sources: Any,
    ) -> dict[str, list[tuple[float, float, int]]]:
        return load_cvs_intervals(sources["annotation_path"], int(video_id))


def _criterion(
    key: str, title: str, requirement: str, negative: str,
) -> CriterionSpec:
    return CriterionSpec(
        key=key,
        title=title,
        visual_requirement=requirement,
        negative_evidence=negative,
        evidence_query=EvidenceQuery(
            criterion_id=key,
            requirement=requirement,
            negative_evidence=negative,
            required_capabilities=["anatomical_state_recognition"],
        ),
        temporal_policy=TemporalPolicy(
            operator="stable_state_before_anchor",
            target_state="satisfied",
            parameters={
                "smoothing_seconds": 9.0,
                "on_threshold": 0.68,
                "off_threshold": 0.52,
                "min_stable_seconds": 12.0,
                "max_gap_seconds": 5.0,
            },
        ),
    )


def create_cholec_cvs_package() -> TaskPackage:
    criteria = {
        "two_structures": _criterion(
            "two_structures",
            "Two structures entering gallbladder",
            "Exactly two tubular structures enter the gallbladder.",
            "The visible anatomy clearly shows fewer or more than exactly two tubular structures entering the gallbladder.",
        ),
        "cystic_plate": _criterion(
            "cystic_plate",
            "Cystic plate exposed",
            "Lower one third of gallbladder is dissected off the cystic plate.",
            "The visible lower third remains attached to the cystic plate or the plate is clearly not exposed.",
        ),
        "hepatocystic_triangle": _criterion(
            "hepatocystic_triangle",
            "Hepatocystic triangle cleared",
            "Triangle is cleared of fat and fibrous tissue.",
            "Visible fat or fibrous tissue clearly remains within the hepatocystic triangle.",
        ),
    }
    return TaskPackage(
        task_id="cholec_cvs",
        task_name="Critical View of Safety assessment",
        description="Assess the three Critical View of Safety criteria before clipping and cutting.",
        criterion_catalog=criteria,
        aliases={
            "two_structures": ("two structures", "two tubular", "cystic duct", "cystic artery"),
            "cystic_plate": ("cystic plate", "gallbladder plate", "lower one third", "liver bed"),
            "hepatocystic_triangle": (
                "hepatocystic triangle", "calot triangle", "calot's triangle", "fat and fibrous",
            ),
        },
        allowed_tools={
            "locate_evaluation_window", "sample_timestamps",
            "criterion_visual_evidence", "stable_evidence_aggregation",
            "explicit_verifier",
        },
        window_spec=EvaluationWindowSpec(
            locator_tool="locate_evaluation_window",
            start_anchor="CalotTriangleDissection",
            end_anchor="ClippingCutting",
            attributes={"evidence_constraint": "before_anchor"},
        ),
        default_temporal_policy=TemporalPolicy(
            operator="stable_state_before_anchor",
            parameters={
                "smoothing_seconds": 9.0, "on_threshold": 0.68,
                "off_threshold": 0.52, "min_stable_seconds": 12.0,
                "max_gap_seconds": 5.0,
            },
        ),
        dataset_adapter=CholecCvsDatasetAdapter(),
        metadata={
            "domain": "surgery",
            "dataset": "Cholec80-CVS",
            "run_stem_template": "video{video_id_int:02d}",
            "visual_evidence_schema": {
                "semantic_types": [
                    "cystic_plate", "calot_triangle", "cystic_artery",
                    "cystic_duct", "gallbladder", "tool",
                ],
                "semantic_pairs": [
                    ["cystic_duct", "cystic_artery"],
                    ["cystic_duct", "gallbladder"],
                    ["cystic_artery", "gallbladder"],
                    ["cystic_plate", "gallbladder"],
                    ["calot_triangle", "gallbladder"],
                    ["tool", "gallbladder"],
                ],
                "criterion_roi_classes": {
                    "two_structures": ["cystic_duct", "cystic_artery", "gallbladder"],
                    "cystic_plate": ["cystic_plate", "gallbladder"],
                    "hepatocystic_triangle": ["calot_triangle", "gallbladder"],
                },
            },
        },
    )
