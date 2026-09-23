"""SAGES adapter from predicted anatomy detections to task-neutral facts."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..grounded_evidence import GroundedEvidenceFact, grounded_fact_payload


def visual_fact_payload(
    detection_rows: list[dict[str, Any]], timestamps_s: list[float],
) -> dict[str, Any]:
    facts: list[GroundedEvidenceFact] = []
    for row, timestamp_s in zip(detection_rows, timestamps_s):
        frame_index = int(row["frame_index"])
        class_rank: dict[str, int] = {}
        for detection in row.get("detections", []):
            class_name = str(detection["class"])
            rank = class_rank.get(class_name, 0)
            class_rank[class_name] = rank + 1
            entity_id = f"{class_name}:{frame_index}:{rank}"
            facts.append(GroundedEvidenceFact(
                fact_id=f"entity:{entity_id}", fact_type="entity",
                subject=entity_id, predicate="detected_as", value=class_name,
                confidence=float(detection["confidence"]),
                frame_index=frame_index, timestamp_s=float(timestamp_s),
                grounding={
                    "bbox_normalized_xyxy": detection["bbox_normalized_xyxy"],
                    "area_fraction": detection["area_fraction"],
                    "source": "predicted_bbox",
                    "fallible_observation": True,
                },
            ))
        for predicate, value in row.get("derived_spatial_facts", {}).items():
            # The detector has imperfect recall.  Only expose positive spatial
            # observations; a false/zero derived value must not be interpreted
            # by the MLLM as proof that anatomy is absent.
            if isinstance(value, bool) and not value:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0:
                continue
            facts.append(GroundedEvidenceFact(
                fact_id=f"spatial:{frame_index}:{predicate}", fact_type="relation",
                subject=f"scene:{frame_index}", predicate=str(predicate), value=value,
                frame_index=frame_index, timestamp_s=float(timestamp_s),
                grounding={"source": "deterministic_relation_over_predicted_boxes"},
            ))
    return grounded_fact_payload(
        facts, modality="visual",
        source_description=(
            "Predicted entities, normalized locations, and spatial relations. "
            "Detections are fallible observations, not criterion labels."
        ),
        provenance={"adapter": "sages_predicted_anatomy_facts_v1"},
    )


def _active_intervals(
    active: np.ndarray, timestamps_s: list[float], cadence_s: float,
) -> list[tuple[int, int, float, float]]:
    output: list[tuple[int, int, float, float]] = []
    start: int | None = None
    for index, value in enumerate(active.astype(bool)):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            end = index - 1 if not value else index
            output.append((
                start, end,
                max(0.0, float(timestamps_s[start]) - cadence_s / 2.0),
                float(timestamps_s[end]) + cadence_s / 2.0,
            ))
            start = None
    return output


def temporal_fact_payload(
    detection_rows: list[dict[str, Any]], timestamps_s: list[float],
    detection_threshold: float = 0.25,
) -> dict[str, Any]:
    """Summarize entity/relation persistence without producing task scores."""
    if len(detection_rows) != len(timestamps_s) or not timestamps_s:
        raise ValueError("Detection rows and timestamps must have equal nonzero length")
    cadence_s = (
        float(np.median(np.diff(timestamps_s))) if len(timestamps_s) > 1 else 1.0
    )
    classes = sorted({
        str(item["class"])
        for row in detection_rows for item in row.get("detections", [])
    })
    facts: list[GroundedEvidenceFact] = []
    for class_name in classes:
        confidence = np.asarray([
            max((
                float(item["confidence"])
                for item in row.get("detections", [])
                if str(item["class"]) == class_name
            ), default=0.0)
            for row in detection_rows
        ], dtype=np.float32)
        active = confidence >= float(detection_threshold)
        intervals = _active_intervals(active, timestamps_s, cadence_s)
        for run_index, (start_index, end_index, start_s, end_s) in enumerate(intervals):
            facts.append(GroundedEvidenceFact(
                fact_id=f"track:{class_name}:{run_index}", fact_type="event",
                subject=class_name, predicate="detected_persistently", value=True,
                confidence=float(confidence[start_index:end_index + 1].mean()),
                start_s=start_s, end_s=end_s,
                grounding={
                    "source": "temporal_track_over_predicted_boxes",
                    "supporting_frame_indices": list(range(start_index, end_index + 1)),
                    "supporting_frame_count": end_index - start_index + 1,
                },
            ))
        facts.append(GroundedEvidenceFact(
            fact_id=f"track_summary:{class_name}", fact_type="attribute",
            subject=class_name, predicate="temporal_detection_summary",
            value={
                "observed_frame_fraction": round(float(active.mean()), 5),
                "longest_run_frames": max(
                    (end - start + 1 for start, end, _, _ in intervals), default=0,
                ),
                "peak_confidence": round(float(confidence.max()), 5),
            },
            grounding={"source": "temporal_track_over_predicted_boxes"},
        ))

    relation_keys = sorted({
        key for row in detection_rows
        for key, value in row.get("derived_spatial_facts", {}).items()
        if isinstance(value, bool)
    })
    for predicate in relation_keys:
        active = np.asarray([
            bool(row.get("derived_spatial_facts", {}).get(predicate, False))
            for row in detection_rows
        ])
        for run_index, (start_index, end_index, start_s, end_s) in enumerate(
            _active_intervals(active, timestamps_s, cadence_s)
        ):
            facts.append(GroundedEvidenceFact(
                fact_id=f"relation_track:{predicate}:{run_index}",
                fact_type="event", subject="scene",
                predicate=f"{predicate}_persisted", value=True,
                confidence=None, start_s=start_s, end_s=end_s,
                grounding={
                    "source": "temporal_relation_over_predicted_boxes",
                    "supporting_frame_indices": list(range(start_index, end_index + 1)),
                    "supporting_frame_count": end_index - start_index + 1,
                },
            ))
    return grounded_fact_payload(
        facts, modality="temporal",
        source_description=(
            "Persistence and transitions of predicted entities and spatial relations. "
            "No task criterion probability or verdict is supplied."
        ),
        provenance={
            "adapter": "sages_predicted_anatomy_temporal_facts_v1",
            "detection_threshold": float(detection_threshold),
            "cadence_s": cadence_s,
        },
    )


def specialist_candidate_fact_payload(
    scores: np.ndarray, criterion_ids: list[str], timestamps_s: list[float],
    strong_count: int = 2, moderate_count: int = 4,
    reliability_policies: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Expose rank bands from a specialist without exporting task probabilities."""
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (len(timestamps_s), len(criterion_ids)):
        raise ValueError("Specialist score shape differs from frames/criteria")
    facts: list[GroundedEvidenceFact] = []
    for criterion_index, criterion_id in enumerate(criterion_ids):
        ranking = np.argsort(-values[:, criterion_index], kind="stable")
        strong = set(int(item) for item in ranking[:strong_count])
        moderate = set(
            int(item) for item in ranking[strong_count:strong_count + moderate_count]
        )
        for frame_index in sorted(strong | moderate):
            policy = (reliability_policies or {}).get(str(criterion_id), {})
            threshold = policy.get("threshold") if policy.get("enabled") else None
            trusted = threshold is not None and float(values[frame_index, criterion_index]) >= float(threshold)
            facts.append(GroundedEvidenceFact(
                fact_id=f"specialist_candidate:{criterion_id}:{frame_index}",
                fact_type="attribute", subject=str(criterion_id),
                predicate="specialist_visual_support_band",
                value=(
                    "high_reliability_candidate" if trusted
                    else "unverified_retrieval_candidate"
                ),
                frame_index=frame_index, timestamp_s=float(timestamps_s[frame_index]),
                grounding={
                    "source": "strict_video_oof_specialist_visual_model",
                    "ranking_scope": "within_sample",
                    "not_a_task_verdict": True,
                    "raw_probability_withheld": True,
                    "requires_image_verification": True,
                    "reliability_source": "other_strict_oof_videos_only",
                },
            ))
    return grounded_fact_payload(
        facts, modality="visual",
        source_description=(
            "A specialist visual model nominates high-ranking candidate frames. "
            "Bands are retrieval hints, not calibrated probabilities or final states; "
            "the MLLM must verify them against the real image and Skill."
        ),
        provenance={
            "adapter": "ranked_specialist_candidate_facts_v1",
            "strict_video_oof": True,
            "raw_probability_exported": False,
            "strong_candidate_count_per_criterion": int(strong_count),
            "moderate_candidate_count_per_criterion": int(moderate_count),
            "crossfit_reliability_policies": dict(reliability_policies or {}),
        },
    )


def specialist_candidate_temporal_payload(
    scores: np.ndarray, criterion_ids: list[str], timestamps_s: list[float],
    strong_count: int = 2,
    reliability_policies: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Expose where specialist candidate peaks occur, without their scores."""
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (len(timestamps_s), len(criterion_ids)) or not timestamps_s:
        raise ValueError("Specialist score shape differs from frames/criteria")
    facts: list[GroundedEvidenceFact] = []
    denominator = max(1, len(timestamps_s) - 1)
    for criterion_index, criterion_id in enumerate(criterion_ids):
        ranking = np.argsort(-values[:, criterion_index], kind="stable")[:strong_count]
        for rank_index, raw_frame_index in enumerate(ranking):
            frame_index = int(raw_frame_index)
            relative = frame_index / denominator
            phase = "early" if relative < 1 / 3 else "middle" if relative < 2 / 3 else "late"
            policy = (reliability_policies or {}).get(str(criterion_id), {})
            threshold = policy.get("threshold") if policy.get("enabled") else None
            trusted = threshold is not None and float(values[frame_index, criterion_index]) >= float(threshold)
            facts.append(GroundedEvidenceFact(
                fact_id=f"specialist_peak:{criterion_id}:{rank_index}",
                fact_type="event", subject=str(criterion_id),
                predicate="specialist_candidate_peak", value={
                    "relative_phase": phase,
                    "candidate_rank_band": (
                        "high_reliability_candidate" if trusted
                        else "unverified_retrieval_candidate"
                    ),
                },
                frame_index=frame_index, timestamp_s=float(timestamps_s[frame_index]),
                grounding={
                    "source": "strict_video_oof_specialist_visual_model",
                    "not_a_task_verdict": True,
                    "raw_probability_withheld": True,
                    "requires_image_verification": True,
                    "reliability_source": "other_strict_oof_videos_only",
                },
            ))
    return grounded_fact_payload(
        facts, modality="temporal",
        source_description=(
            "A temporal retrieval tool locates specialist candidate peaks in the sequence. "
            "Peak locations are hints and do not assert that a criterion is satisfied."
        ),
        provenance={
            "adapter": "ranked_specialist_candidate_temporal_facts_v1",
            "strict_video_oof": True,
            "raw_probability_exported": False,
            "crossfit_reliability_policies": dict(reliability_policies or {}),
        },
    )
