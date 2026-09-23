"""Cholec score-cache adapter and opt-in graph-to-Qwen input adapter.

No labels, phase probabilities, or previous phase predictions are read here.
The label order below mirrors run_cholect50_rendezvous_fact_smoke.py without
importing its torch/torchvision inference dependencies.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
from pathlib import Path
from typing import Iterable, Iterator

from cvs_assessment.temporal_graph_memory import Atom, EventGraphMemory, Observation


COMPONENT_NAMES = {
    "instrument": ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"],
    "verb": ["grasp", "retract", "dissect", "coagulate", "clip", "cut", "aspirate",
             "irrigate", "pack", "null_verb"],
    "target": ["gallbladder", "cystic_plate", "cystic_duct", "cystic_artery",
               "cystic_pedicle", "blood_vessel", "fluid", "abdominal_wall_cavity",
               "liver", "adhesion", "omentum", "peritoneum", "gut", "specimen_bag",
               "null_target"],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_observations(scores_path: Path, calibration_path: Path, video: str,
                      cutoff_s: float, *, seconds_per_frame_id: float = 1.0) -> list[Observation]:
    """Load only threshold-passing marginal predictions, not inferred triplets.

    Existing Cholec transfer caches use frame_id as seconds (scale=1). For other
    caches the caller must explicitly supply the correct scale; it is never guessed.
    """
    import numpy as np

    if (not math.isfinite(seconds_per_frame_id) or seconds_per_frame_id <= 0 or
            not math.isfinite(cutoff_s) or cutoff_s < 0):
        raise ValueError("Invalid time scale or cutoff")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["classes"]
    source_hash = file_sha256(scores_path)
    output = []
    seen = set()
    with np.load(scores_path, allow_pickle=False) as source:
        videos = source["video"].astype(str)
        frame_ids = source["frame_id"]
        if videos.ndim != 1 or frame_ids.shape != videos.shape:
            raise ValueError("Malformed video/frame arrays")
        selected = np.flatnonzero(videos == video)
        if not len(selected):
            raise ValueError(f"Video absent from cache: {video}")
        arrays = {key: source[key] for key in COMPONENT_NAMES}
        for channel, names in COMPONENT_NAMES.items():
            if arrays[channel].shape != (len(videos), len(names)):
                raise ValueError(f"Unexpected {channel} score shape")
            if set(names) - set(calibration[channel]):
                raise ValueError(f"Calibration is missing {channel} classes")
        for index in selected:
            frame = float(frame_ids[index])
            timestamp = frame * seconds_per_frame_id
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("Invalid source timestamp")
            if timestamp > cutoff_s:
                continue
            if timestamp in seen:
                raise ValueError(f"Duplicate cache timestamp: {video}/{timestamp}")
            seen.add(timestamp)
            atoms = []
            for channel, names in COMPONENT_NAMES.items():
                for label_index, label in enumerate(names):
                    settings = calibration[channel][label]
                    if not isinstance(settings["enabled"], bool):
                        raise ValueError("Calibration enabled must be boolean")
                    if not settings["enabled"] or label.startswith("null_"):
                        continue
                    threshold = float(settings["threshold"])
                    score = float(arrays[channel][index, label_index])
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Invalid detector score")
                    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                        raise ValueError("Invalid calibration threshold")
                    if score >= threshold:
                        precision = settings.get("precision")
                        atoms.append(Atom(channel, label, score, threshold,
                                          None if precision is None else float(precision)))
            observation = Observation(
                observation_id=f"{video}:row{int(index)}", video=video, time_s=timestamp,
                atoms=tuple(atoms), source_file=str(scores_path.resolve()),
                source_sha256=source_hash, source_row=int(index),
            )
            observation.validate()
            output.append(observation)
    return sorted(output, key=lambda item: (item.time_s, item.observation_id))


def observation_batches(observations: Iterable[Observation], *, span_s: float = 10.0,
                        max_samples: int = 16,
                        boundaries: Iterable[float] = ()) -> Iterator[list[Observation]]:
    """Transport batches only; the model, not these boundaries, defines events.

    Split at requested evidence cutoffs so a query need not use a partially
    processed batch. Without such a boundary, recent raw observations remain
    available, but an event summary produced later will not be used retroactively.
    """
    if not math.isfinite(span_s) or span_s <= 0 or max_samples < 1:
        raise ValueError("Invalid batch size")
    cuts = sorted(set(float(t) for t in boundaries))
    if any(not math.isfinite(t) or t < 0 for t in cuts):
        raise ValueError("Invalid query boundary")
    batch: list[Observation] = []
    previous = -1.0
    for item in observations:
        if item.time_s <= previous:
            raise ValueError("Observations must have strictly increasing times")
        crosses_boundary = batch and any(batch[-1].time_s <= t < item.time_s for t in cuts)
        if batch and (item.time_s - batch[0].time_s >= span_s or
                      len(batch) >= max_samples or crosses_boundary):
            yield batch
            batch = []
        batch.append(item)
        previous = item.time_s
    if batch:
        yield batch


def graph_evidence_text(graph: EventGraphMemory, query_s: float, *, lookahead_s: float = 0.0,
                        query: str = "instrument verb target recent observations historical events",
                        **retrieval_options) -> str:
    if not math.isfinite(query_s) or query_s < 0 or not math.isfinite(lookahead_s) or lookahead_s < 0:
        raise ValueError("Invalid query time or lookahead")
    cutoff = query_s + lookahead_s
    payload = graph.retrieve(query, cutoff, **retrieval_options)
    payload["query_s"] = query_s
    payload["lookahead_s"] = lookahead_s
    payload["protocol"] = "strict_causal" if lookahead_s == 0 else "offline_with_explicit_lookahead"
    return ("Phase-free model-managed event graph. Detector observations are fallible; "
            "model summaries and semantic links are hypotheses, not additional observations. "
            "First/last times bound observed supports, not continuous duration. "
            "Use event IDs and times when citing evidence.\n" + json.dumps(payload, ensure_ascii=False))


def graph_judgment_messages(case: dict, graph: EventGraphMemory, *, skill_system_prompt: str,
                            lookahead_s: float = 2.0, **retrieval_options) -> list[dict]:
    """Prepare only; never perform a phase-model call or read case truth.

    Keeps the existing long-image context and caller-supplied skill. Default +2s
    matches the existing offline Cholec protocol. Strict online use requires both
    lookahead_s=0 AND a separately supplied causal image context.
    """
    if case["video"] != graph.video:
        raise ValueError("Case and graph belong to different videos")
    center = float(case["anchor_frame"])
    paths, times = case["long_images"], case["long_frames"]
    if not paths or len(paths) != len(times):
        raise ValueError("Image/time alignment is required")
    if any(not math.isfinite(float(t)) or float(t) < 0 or
           float(t) > center + lookahead_s for t in times):
        raise ValueError("Image context exceeds the explicitly allowed evidence cutoff")
    if any(float(a) > float(b) for a, b in zip(times, times[1:])):
        raise ValueError("Image context must be chronological")
    content = []
    for position, (raw_path, timestamp) in enumerate(zip(paths, times), start=1):
        path = Path(raw_path)
        marker = " CENTER" if float(timestamp) == center else ""
        content.append({"type": "text", "text": f"Chronological image {position}: t={timestamp} s{marker}."})
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    evidence = graph_evidence_text(graph, center, lookahead_s=lookahead_s, **retrieval_options)
    instruction = (f"Choose exactly one phase for the CENTER time.\n"
                   f"Temporal metadata: center={case['anchor_frame']} s, "
                   f"video duration={case['video_frames'] - 1} s, "
                   f"elapsed={100 * case['elapsed_fraction']:.1f}%.\n\n" + evidence)
    content.append({"type": "text", "text": instruction})
    return [{"role": "system", "content": skill_system_prompt},
            {"role": "user", "content": content}]
