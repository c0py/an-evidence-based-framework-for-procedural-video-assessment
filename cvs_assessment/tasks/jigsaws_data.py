"""Release-layout adapter for the official JIGSAWS dataset.

The module keeps inference inputs (video and robot kinematics) separate from
evaluation-only annotations (modified GRS scores and gesture transcripts).
This separation matters because the frozen foundation model and evidence
plugins must not receive the answer while a trial is being judged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


JIGSAWS_FPS = 30.0
JIGSAWS_TASKS = ("Knot_Tying", "Needle_Passing", "Suturing")
GRS_CRITERIA = (
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
)

_TRIAL_PATTERN = re.compile(
    r"^(Knot_Tying|Needle_Passing|Suturing)_([A-I])(\d{3})$"
)


@dataclass(frozen=True)
class GestureInterval:
    start_frame: int
    end_frame: int
    gesture_id: str

    def validate(self) -> None:
        if self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ValueError("JIGSAWS gesture interval has invalid frame bounds")
        if not re.fullmatch(r"G\d+", self.gesture_id):
            raise ValueError(f"Invalid JIGSAWS gesture id: {self.gesture_id}")


@dataclass(frozen=True)
class JigsawsTrial:
    trial_id: str
    task_name: str
    subject_id: str
    repetition: int
    self_reported_skill: str
    grs_total: int
    grs_scores: dict[str, int]
    capture1_video: Path
    capture2_video: Path
    kinematics_path: Path
    transcription_path: Path

    def validate(self) -> None:
        match = _TRIAL_PATTERN.fullmatch(self.trial_id)
        if match is None:
            raise ValueError(f"Invalid JIGSAWS trial id: {self.trial_id}")
        if match.group(1) != self.task_name or match.group(2) != self.subject_id:
            raise ValueError(f"Inconsistent JIGSAWS trial identity: {self.trial_id}")
        if int(match.group(3)) != self.repetition:
            raise ValueError(f"Inconsistent JIGSAWS repetition: {self.trial_id}")
        if self.self_reported_skill not in {"N", "I", "E"}:
            raise ValueError(f"Invalid self-reported skill: {self.self_reported_skill}")
        if set(self.grs_scores) != set(GRS_CRITERIA):
            raise ValueError(f"Incomplete JIGSAWS GRS row: {self.trial_id}")
        if any(not 1 <= int(value) <= 5 for value in self.grs_scores.values()):
            raise ValueError(f"JIGSAWS GRS item outside [1,5]: {self.trial_id}")
        if self.grs_total != sum(self.grs_scores.values()):
            raise ValueError(f"JIGSAWS GRS total does not equal six item scores: {self.trial_id}")
        missing = [
            path for path in (
                self.capture1_video, self.capture2_video,
                self.kinematics_path, self.transcription_path,
            ) if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Incomplete JIGSAWS trial {self.trial_id}: {missing}")

    def inference_record(self, *, capture: int = 1) -> dict[str, Any]:
        """Return an input-only record with all evaluation labels removed."""
        if capture not in {1, 2}:
            raise ValueError("JIGSAWS capture must be 1 (left) or 2 (right)")
        video = self.capture1_video if capture == 1 else self.capture2_video
        return {
            # JIGSAWS trial IDs contain the operator letter. A pseudonym keeps
            # that identity out of the MLLM prompt while the local paths remain
            # available to trusted loading code.
            "sample_id": "jig_" + hashlib.sha256(
                f"jigsaws-inference:{self.trial_id}".encode("utf-8")
            ).hexdigest()[:16],
            "task_name": self.task_name,
            "video_path": str(video.resolve()),
            "capture": capture,
            "kinematics_path": str(self.kinematics_path.resolve()),
            "fps": JIGSAWS_FPS,
            "labels_included": False,
            "transcription_included": False,
            "subject_identity_included_in_model_request": False,
        }

    def evaluation_record(self) -> dict[str, Any]:
        """Return labels for evaluation code; never pass this to inference."""
        return {
            "sample_id": self.trial_id,
            "subject_id": self.subject_id,
            "self_reported_skill": self.self_reported_skill,
            "grs_total": self.grs_total,
            "grs_scores": dict(self.grs_scores),
            "ordinal_states": {
                key: grs_score_to_state(value)
                for key, value in self.grs_scores.items()
            },
            "transcription_path": str(self.transcription_path.resolve()),
        }


def grs_score_to_state(score: int) -> str:
    """Map the five-point modified GRS to the framework's N/P/F states."""
    if score not in {1, 2, 3, 4, 5}:
        raise ValueError("Modified GRS score must be in [1,5]")
    return "N" if score <= 2 else ("P" if score == 3 else "F")


def _task_root(dataset_root: Path, task_name: str) -> Path:
    if task_name not in JIGSAWS_TASKS:
        raise ValueError(f"Unknown JIGSAWS task: {task_name}")
    root = Path(dataset_root)
    candidate = root / task_name
    if candidate.is_dir():
        return candidate
    if root.name == task_name and root.is_dir():
        return root
    raise FileNotFoundError(f"Cannot locate JIGSAWS task {task_name} under {root}")


def load_trials(dataset_root: Path, task_name: str = "Suturing") -> list[JigsawsTrial]:
    """Load fully labeled trials, using the task meta file as authority."""
    root = _task_root(Path(dataset_root), task_name)
    meta_path = root / f"meta_file_{task_name}.txt"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    trials: list[JigsawsTrial] = []
    for line_number, line in enumerate(meta_path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 9:
            raise ValueError(f"Malformed JIGSAWS meta row {meta_path}:{line_number}")
        trial_id, self_reported, total_text, *item_text = fields
        match = _TRIAL_PATTERN.fullmatch(trial_id)
        if match is None or match.group(1) != task_name:
            raise ValueError(f"Unexpected JIGSAWS trial id in {meta_path}: {trial_id}")
        scores = dict(zip(GRS_CRITERIA, (int(value) for value in item_text)))
        trial = JigsawsTrial(
            trial_id=trial_id,
            task_name=task_name,
            subject_id=match.group(2),
            repetition=int(match.group(3)),
            self_reported_skill=self_reported,
            grs_total=int(total_text),
            grs_scores=scores,
            capture1_video=root / "video" / f"{trial_id}_capture1.avi",
            capture2_video=root / "video" / f"{trial_id}_capture2.avi",
            kinematics_path=root / "kinematics" / "AllGestures" / f"{trial_id}.txt",
            transcription_path=root / "transcriptions" / f"{trial_id}.txt",
        )
        trial.validate()
        trials.append(trial)
    if not trials or len({trial.trial_id for trial in trials}) != len(trials):
        raise ValueError(f"JIGSAWS meta file is empty or has duplicate trials: {meta_path}")
    return trials


def load_kinematics(path: Path) -> np.ndarray:
    values = np.loadtxt(Path(path), dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 76:
        raise ValueError(f"Expected a nontrivial JIGSAWS [T,76] array: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"JIGSAWS kinematics contain non-finite values: {path}")
    return values


def load_transcription(path: Path) -> list[GestureInterval]:
    intervals: list[GestureInterval] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"Malformed JIGSAWS transcription row {path}:{line_number}")
        interval = GestureInterval(int(fields[0]), int(fields[1]), fields[2])
        interval.validate()
        if intervals and interval.start_frame <= intervals[-1].end_frame:
            raise ValueError(f"Overlapping/nonchronological JIGSAWS transcription: {path}")
        intervals.append(interval)
    if not intervals:
        raise ValueError(f"Empty JIGSAWS transcription: {path}")
    return intervals


def leave_one_subject_out_splits(
    trials: Iterable[JigsawsTrial],
) -> list[dict[str, Any]]:
    """Build deterministic user-out folds with trial IDs only."""
    rows = sorted(trials, key=lambda trial: trial.trial_id)
    subjects = sorted({trial.subject_id for trial in rows})
    output: list[dict[str, Any]] = []
    for subject_id in subjects:
        test_ids = [trial.trial_id for trial in rows if trial.subject_id == subject_id]
        train_ids = [trial.trial_id for trial in rows if trial.subject_id != subject_id]
        if not test_ids or set(test_ids).intersection(train_ids):
            raise RuntimeError(f"Invalid JIGSAWS subject-out fold: {subject_id}")
        output.append({
            "fold_id": f"subject_{subject_id}_out",
            "held_out_subject": subject_id,
            "train_trial_ids": train_ids,
            "test_trial_ids": test_ids,
        })
    return output


def trial_by_id(
    dataset_root: Path, trial_id: str,
) -> JigsawsTrial:
    match = _TRIAL_PATTERN.fullmatch(str(trial_id))
    if match is None:
        raise ValueError(f"Invalid JIGSAWS trial id: {trial_id}")
    by_id = {
        trial.trial_id: trial
        for trial in load_trials(dataset_root, match.group(1))
    }
    if trial_id not in by_id:
        raise KeyError(f"No fully labeled JIGSAWS trial: {trial_id}")
    return by_id[trial_id]
