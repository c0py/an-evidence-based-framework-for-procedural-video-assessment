from __future__ import annotations

from pathlib import Path

from .schema import AssessmentResult


def format_time(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def write_report(path: str | Path, result: AssessmentResult) -> None:
    lines = [
        f"# {result.task_name} Report", "",
        f"Task package: `{result.task_id}`", "",
        f"Overall verdict: **{result.overall_verdict}** (confidence {result.overall_confidence:.2f})", "",
        f"Evaluation window: {format_time(result.evaluation_window['start_s'])} to {format_time(result.evaluation_window['end_s'])}.", "",
    ]
    if result.development_oracle:
        lines += ["> Warning: this run uses `annotation_replay`, a development oracle. It is not a visual-model result.", ""]
    for criterion in result.criteria:
        lines.append(f"## {criterion.key}: {criterion.verdict} ({criterion.confidence:.2f})")
        lines.append(
            "- Evidence coverage: "
            f"{criterion.assessable_coverage:.2f} assessable; "
            f"positive={criterion.positive_point_count}, "
            f"negative={criterion.negative_point_count}, "
            f"unknown={criterion.unknown_point_count}"
        )
        if criterion.evidence_intervals:
            for interval in criterion.evidence_intervals:
                lines.append(f"- Stable evidence: {format_time(interval.start_s)}-{format_time(interval.end_s)}, mean={interval.mean_score:.2f}, confidence={interval.confidence:.2f}")
            if criterion.representative_frame:
                lines.append(f"- Representative frame: `{criterion.representative_frame}`")
        if criterion.reason:
            lines.append(f"- Reason: {criterion.reason}")
        lines.append("")
    if result.notes:
        lines += ["## Notes", *[f"- {note}" for note in result.notes], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
