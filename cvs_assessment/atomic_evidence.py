"""Task-neutral atomic visual facts for framework-controlled SOP verification."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence_pack import EvidencePack
from .object_observations import FrameObjectObservations


FACT_STATES = frozenset({"yes", "no", "unknown"})
FRAME_VISIBILITY = frozenset({"good", "limited", "poor"})


@dataclass(frozen=True)
class AtomicFact:
    key: str
    frame_states: tuple[str, ...]
    confidence: float
    observed_detail: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["frame_states"] = list(self.frame_states)
        return value


@dataclass(frozen=True)
class AtomicEvidenceObservation:
    criterion: str
    frame_visibility: tuple[str, ...]
    facts: tuple[AtomicFact, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "frame_visibility": list(self.frame_visibility),
            "facts": [fact.to_dict() for fact in self.facts],
            "notes": self.notes,
        }


def load_predicate_spec(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "atomic_predicate_spec_v1":
        raise ValueError("Unsupported atomic predicate specification")
    criteria = value.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("Predicate specification must define criteria")
    for criterion, config in criteria.items():
        facts = config.get("facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError(f"Criterion {criterion!r} has no atomic facts")
        keys = [item.get("key") for item in facts]
        if any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError(f"Criterion {criterion!r} has invalid or duplicate fact keys")
    return value


def build_atomic_fact_payload(
    pack: EvidencePack, criterion_spec: Mapping[str, Any], model: str,
    max_tokens: int = 1400,
    expert_prototypes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    facts = criterion_spec["facts"]
    fact_lines = "\n".join(
        f"- {item['key']}: {item['description']}" for item in facts
    )
    visual_calibration = criterion_spec.get("visual_calibration")
    calibration_instruction = (
        f"\nTask-adapter visual calibration: {visual_calibration}\n"
        if visual_calibration else ""
    )
    keys = [item["key"] for item in facts]
    scale_labels = pack.time_scale_labels or ("short_dense",) * pack.frame_count
    timestamps = ", ".join(
        f"{index}:{time_s:.1f}s[{scale_labels[index]}]"
        for index, time_s in enumerate(pack.timestamps_s)
    )
    prototype_instruction = ""
    if expert_prototypes:
        positive_count = sum(item.get("label") == "positive" for item in expert_prototypes)
        negative_count = sum(item.get("label") == "negative" for item in expert_prototypes)
        prototype_instruction = f"""
Before the query sequence, you will receive {len(expert_prototypes)} labeled expert reference
images ({positive_count} positive and {negative_count} negative). They define task-specific visual
anchors for this criterion and may come from different patients, viewpoints, and lighting. Use them
only to calibrate what the anatomical state looks like. They are NOT query frames: never emit
frame_visibility or frame_states for them, never copy their labels to the query, and do not require
the query to look identical. Judge every query frame from its own visible pixels.
"""
    image_description = (
        "Each image is an unmodified high-resolution raw surgical view without predicted overlays."
        if pack.view_layout == "raw_highres" else
        "Each image contains a large unmodified raw view followed by a fallible "
        "criterion-specific mask/box view. Predictions are hints and may be wrong; never "
        "repeat a predicted label unless the raw pixels support it."
    )
    instruction = f"""You are an atomic visual fact extractor inside an SOP monitoring framework.
Do not decide whether the SOP criterion passes, fails, is full, or is partial. The framework will
make that decision. Inspect only directly visible pixels in the {pack.frame_count} chronological
evidence images ({timestamps}). {image_description}
{prototype_instruction}

Frames labeled long_context are sparse earlier history used to understand anatomical state change.
Frames labeled short_dense are the current candidate interval. Report facts independently for every
frame. Do not require an earlier long_context frame to satisfy the current criterion: earlier frames
may deliberately show the uncleared or undissected state, while short_dense frames show completion.

Criterion context: {criterion_spec['requirement']}
{calibration_instruction}
Extract exactly these independent facts:
{fact_lines}

For every fact, output one state for every frame in chronological order. Use yes only when that fact
is directly visible in that frame, no only when the relevant region is adequately visible and the
fact is visibly contradicted, and unknown for blur, occlusion, insufficient context, or ambiguity.
Do not infer anatomy from surgical convention. Confidence describes the reliability of the entire
fact sequence, not task completion.

Return ONLY one JSON object with exactly these fields:
{{
  "criterion": "{pack.criterion}",
  "frame_visibility": ["good|limited|poor" exactly {pack.frame_count} entries],
  "facts": [
    {{"key": "one required key", "frame_states": ["yes|no|unknown" exactly {pack.frame_count} entries], "confidence": 0.0, "observed_detail": "short pixel-grounded description"}}
  ],
  "notes": "short note about ambiguity or contradictions"
}}
The facts array must contain each key exactly once in this order: {keys}."""
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    if expert_prototypes:
        content.append({
            "type": "text",
            "text": "EXPERT REFERENCE IMAGES START (calibration only; not part of output arrays).",
        })
        content.extend(
            {"type": "image_url", "image_url": {"url": item["image_data_url"]}}
            for item in expert_prototypes
        )
        content.append({
            "type": "text",
            "text": (
                "EXPERT REFERENCE IMAGES END. QUERY SEQUENCE START. "
                f"The next {pack.frame_count} images correspond exactly to frame indices "
                f"0 through {pack.frame_count - 1}."
            ),
        })
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in pack.image_data_urls
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }


def parse_atomic_evidence(
    text: str, criterion: str, criterion_spec: Mapping[str, Any], frame_count: int,
) -> AtomicEvidenceObservation:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid atomic evidence JSON: {text[:400]}") from exc
    required = {"criterion", "frame_visibility", "facts", "notes"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"Atomic evidence fields must be exactly {sorted(required)}")
    if value["criterion"] != criterion:
        raise ValueError("Atomic evidence criterion mismatch")
    visibility = value["frame_visibility"]
    if (
        not isinstance(visibility, list) or len(visibility) != frame_count
        or any(item not in FRAME_VISIBILITY for item in visibility)
    ):
        raise ValueError("frame_visibility must contain one valid state per frame")
    expected_keys = [item["key"] for item in criterion_spec["facts"]]
    facts = value["facts"]
    if not isinstance(facts, list) or [item.get("key") for item in facts if isinstance(item, dict)] != expected_keys:
        raise ValueError("Atomic facts must contain every configured key exactly once and in order")
    parsed = []
    for item in facts:
        if set(item) != {"key", "frame_states", "confidence", "observed_detail"}:
            raise ValueError(f"Invalid fields for atomic fact {item.get('key')!r}")
        states = item["frame_states"]
        if (
            not isinstance(states, list) or len(states) != frame_count
            or any(state not in FACT_STATES for state in states)
        ):
            raise ValueError(f"Fact {item['key']!r} needs one valid state per frame")
        confidence = item["confidence"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"Fact {item['key']!r} has invalid confidence")
        detail = item["observed_detail"]
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError(f"Fact {item['key']!r} needs an observed detail")
        parsed.append(AtomicFact(
            key=item["key"], frame_states=tuple(states), confidence=float(confidence),
            observed_detail=detail.strip(),
        ))
    notes = value["notes"]
    if not isinstance(notes, str):
        raise ValueError("Atomic evidence notes must be a string")
    return AtomicEvidenceObservation(
        criterion=criterion, frame_visibility=tuple(visibility),
        facts=tuple(parsed), notes=notes.strip(),
    )


def longest_run(states: Sequence[bool]) -> int:
    best = current = 0
    for state in states:
        current = current + 1 if state else 0
        best = max(best, current)
    return best


def temporal_atomic_features(
    observation: AtomicEvidenceObservation, criterion_spec: Mapping[str, Any],
    time_scale_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create task-neutral verifier features without deciding a task label."""
    by_key = {fact.key: fact for fact in observation.facts}
    required = [item["key"] for item in criterion_spec["facts"] if item.get("required", True)]
    usable = [state != "poor" for state in observation.frame_visibility]
    simultaneous_yes = [
        usable[index] and all(by_key[key].frame_states[index] == "yes" for key in required)
        for index in range(len(usable))
    ]
    simultaneous_no = [
        usable[index] and any(by_key[key].frame_states[index] == "no" for key in required)
        for index in range(len(usable))
    ]
    labels = list(time_scale_labels or ("short_dense",) * len(usable))
    if len(labels) != len(usable):
        raise ValueError("time_scale_labels must match the atomic evidence frame count")
    dense_indices = [index for index, label in enumerate(labels) if label == "short_dense"]
    context_indices = [index for index, label in enumerate(labels) if label == "long_context"]
    dense_yes = [simultaneous_yes[index] for index in dense_indices]
    dense_no = [simultaneous_no[index] for index in dense_indices]
    state_keys = [
        item["key"] for item in criterion_spec["facts"]
        if item.get("required", True) and item.get("predicate_type") == "state"
    ]
    state_transitions = {}
    for key in state_keys:
        context_states = [by_key[key].frame_states[index] for index in context_indices]
        dense_states = [by_key[key].frame_states[index] for index in dense_indices]
        context_no_fraction = (
            sum(value == "no" for value in context_states) / len(context_states)
            if context_states else 0.0
        )
        dense_yes_fraction = (
            sum(value == "yes" for value in dense_states) / len(dense_states)
            if dense_states else 0.0
        )
        state_transitions[key] = {
            "context_no_fraction": context_no_fraction,
            "dense_yes_fraction": dense_yes_fraction,
            "no_to_yes_transition_score": context_no_fraction * dense_yes_fraction,
        }
    return {
        "required_fact_keys": required,
        "per_fact_yes_runs": {
            key: longest_run([
                usable[index] and state == "yes"
                for index, state in enumerate(by_key[key].frame_states)
            ]) for key in required
        },
        "per_fact_no_runs": {
            key: longest_run([
                usable[index] and state == "no"
                for index, state in enumerate(by_key[key].frame_states)
            ]) for key in required
        },
        "simultaneous_yes_run": longest_run(simultaneous_yes),
        "any_required_no_run": longest_run(simultaneous_no),
        "dense_simultaneous_yes_run": longest_run(dense_yes),
        "dense_any_required_no_run": longest_run(dense_no),
        "state_transition_features": state_transitions,
        "good_frame_fraction": sum(state == "good" for state in observation.frame_visibility) / len(usable),
        "usable_frame_fraction": sum(usable) / len(usable),
        "minimum_fact_confidence": min(by_key[key].confidence for key in required),
        "mean_fact_confidence": sum(by_key[key].confidence for key in required) / len(required),
    }


def locator_consistency_features(
    frames: Sequence[FrameObjectObservations], criterion_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize whether replaceable localization tools support each predicate.

    These are model predictions, not facts or task labels.  Keeping them as
    separate features lets the explicit verifier learn conservative agreement
    rules without allowing either Qwen or a localizer to decide the SOP alone.
    """
    configured = {
        item["key"]: tuple(item.get("locator_classes", ()))
        for item in criterion_spec["facts"] if item.get("required", True)
    }
    present = [
        {item.semantic_type for item in frame.observations}
        for frame in frames
    ]
    per_fact_runs = {}
    per_fact_frame_fractions = {}
    for key, classes in configured.items():
        supported = [bool(classes) and all(name in names for name in classes) for names in present]
        per_fact_runs[key] = longest_run(supported)
        per_fact_frame_fractions[key] = sum(supported) / max(1, len(supported))
    all_classes = sorted({name for classes in configured.values() for name in classes})
    all_supported = [
        bool(all_classes) and all(name in names for name in all_classes)
        for names in present
    ]
    any_relevant = [any(name in names for name in all_classes) for names in present]
    return {
        "configured_locator_classes": all_classes,
        "per_fact_locator_runs": per_fact_runs,
        "per_fact_locator_frame_fractions": per_fact_frame_fractions,
        "all_locator_classes_run": longest_run(all_supported),
        "all_locator_classes_frame_fraction": sum(all_supported) / max(1, len(all_supported)),
        "any_relevant_locator_frame_fraction": sum(any_relevant) / max(1, len(any_relevant)),
    }
