#!/usr/bin/env python3
"""Shared definitions for CholecT50 absolute-performance development.

This module deliberately keeps the five official challenge test videos out of
all development helpers.  The specialist plugin may emit observations and
event history, but never a phase label or phase probability.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PHASES = {
    0: "preparation",
    1: "calot_triangle_dissection",
    2: "clipping_and_cutting",
    3: "gallbladder_dissection",
    4: "gallbladder_packaging",
    5: "cleaning_and_coagulation",
    6: "gallbladder_extraction",
}

CHALLENGE_TRAIN_VIDEOS = tuple(
    f"VID{value:02d}"
    for value in (
        1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57,
        68, 23, 35, 48, 60, 70, 13, 25, 49, 62, 75, 8, 12, 29, 50, 78,
        6, 51, 10, 73, 14, 32, 80, 42,
    )
)
CHALLENGE_VALIDATION_VIDEOS = tuple(
    f"VID{value:02d}" for value in (5, 18, 36, 65, 74)
)
FORBIDDEN_CHALLENGE_TEST_VIDEOS = ("VID92", "VID96", "VID103", "VID110", "VID111")

ARMS = (
    "vision_only_local",
    "skill_local",
    "skill_long_visual",
    "skill_long_gated_facts",
    "skill_long_gated_facts_v2",
)

LOCAL_OFFSETS = (-2, -1, 0, 1, 2)
LONG_OFFSETS = (-120, -60, -30, -10, -2, -1, 0, 1, 2)

PHASE_SKILL = """Task Skill: laparoscopic cholecystectomy phase recognition.
Judge the phase at the explicitly marked CENTER time.  Use the images as the
primary evidence and treat workflow order as a prior, not an inflexible rule.

0 preparation: initial access, exposure, adhesiolysis, or setup before sustained
  hepatocystic-triangle work.
1 calot_triangle_dissection: exposing and dissecting the cystic duct/artery in
  Calot's triangle, usually with grasper plus hook/bipolar, before division.
2 clipping_and_cutting: clip applier or scissors acts on the cystic duct/artery;
  this is usually a short transition.
3 gallbladder_dissection: after duct/artery division, the gallbladder is peeled
  away from the liver/cystic plate and the liver bed becomes visible.
4 gallbladder_packaging: the detached gallbladder is placed into a specimen bag.
5 cleaning_and_coagulation: inspection of the operative field/liver bed with
  coagulation, irrigation, or aspiration; it may recur around late phases.
6 gallbladder_extraction: the bagged specimen is pulled toward or through an
  abdominal-wall/trocar exit, often with a close view near the access port.

Important distinctions: repeated grasping/dissection alone does not identify a
phase.  Clipper/cut events are strong for phase 2; sustained separation from the
liver after those events supports phase 3; bag appearance/packing supports phase
4; irrigation/coagulation of the bed supports phase 5; movement of the bagged
specimen toward the abdominal wall supports phase 6."""

BASIC_SYSTEM_PROMPT = (
    "Classify the laparoscopic cholecystectomy phase at the CENTER frame. "
    "Return exactly one of: "
    + "; ".join(f"{index}={name}" for index, name in PHASES.items())
    + ". Return only the required JSON."
)

SKILL_SYSTEM_PROMPT = (
    BASIC_SYSTEM_PROMPT
    + "\n\n"
    + PHASE_SKILL
    + "\n\nSpecialist facts, when present, are fallible observations rather than answers. "
      "Do not copy them blindly."
)

WORKFLOW_ARBITRATION_GUIDANCE_V2 = """Workflow-memory arbitration guidance:
1. A reliable history containing both clip application and scissor cutting is
   positive evidence that cystic-structure division already occurred.  Do not
   fall back to phase 1 merely because the current frame again shows a hook,
   grasper, or generic dissection.  Post-division peeling from the liver is
   usually phase 3.
2. Before any clip/cut milestone, the very early part of a video can still be
   phase 0 even when a grasper retracts the gallbladder.  Require clear sustained
   work in Calot's triangle before choosing phase 1 over preparation.
3. Treat the first reliable specimen-bag appearance as a milestone.  Active bag
   filling/manipulation shortly after first appearance supports phase 4.  A bag
   that has been present for longer, especially late in the video and moving
   toward a port, supports phase 6.
4. Current/recent bipolar or hook coagulation and irrigator use at the exposed
   liver bed supports phase 5.  A bag merely being present does not override
   clear current cleaning/coagulation activity.
5. Late phases may alternate visually.  Use current images, event recency, time
   since first bag appearance, and overall elapsed time together; no single
   clock threshold or observation is an automatic answer."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "phase_id": {"type": "integer", "minimum": 0, "maximum": 6},
            "phase_name": {"type": "string", "enum": list(PHASES.values())},
            "rationale": {"type": "string"},
        },
        "required": ["phase_id", "phase_name", "rationale"],
        "additionalProperties": False,
    }


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def parse_prediction(payload: dict[str, Any]) -> int | None:
    parsed = payload.get("parsed")
    if not isinstance(parsed, dict):
        return None
    phase_id = parsed.get("phase_id")
    if phase_id not in PHASES or parsed.get("phase_name") != PHASES[phase_id]:
        return None
    return int(phase_id)
