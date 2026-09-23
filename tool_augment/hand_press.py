#!/usr/bin/env python3
"""Task-specific press-like hand pose prototype; no contact/force measurement.

Replays verified MediaPipe landmarks against the original video. Rules use
finger geometry, a configured work area and temporal stability, never frame IDs
or known action times. All labels describe visual candidates, not ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".runtime"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".runtime_settings/matplotlib"))
import numpy as np

HEADER_HEIGHT = 60
GESTURE_LABELS = {
    "PRESS-LIKE POSE (CANDIDATE)": "Press-like pose",
    "PINCH / GRIP": "Pinch / Grip",
    "MOVING / REPOSITIONING": "Repositioning",
    "CHECKING PRESS-LIKE POSE": "Observing",
    "OTHER HAND POSE": "Other",
}


@dataclass(frozen=True)
class Config:
    # Manually configured for this fixed-camera ostomy demonstration only.
    target_roi: tuple = (0.40, 0.40, 0.80, 0.82)
    index_pip_min_deg: float = 65.0
    index_pip_max_deg: float = 165.0
    pinch_max_palm_widths: float = 0.38
    max_tip_speed_hand_scales_s: float = 1.2
    evidence_dwell_s: float = 0.25
    release_s: float = 0.15
    max_observation_gap_s: float = 0.20
    smoothing_tau_s: float = 0.10


def angle(a, b, c):
    u, v = a - b, c - b
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / norm, -1, 1))))


def pose_features(landmarks, width, height, config):
    p = np.asarray(landmarks, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return {"valid": False}
    # MediaPipe image z is wrist-relative and roughly on the x scale.
    # It is used only for within-hand angles, never world motion/contact.
    xyz = p * [width, height, width]
    xy = xyz[:, :2]
    palm_width = float(np.linalg.norm(xy[5] - xy[17]))
    # An edge-on hand has near-zero projected width. Palm length prevents this
    # view from turning small keypoint motion into a very large normalized speed.
    palm = max(palm_width, float(np.linalg.norm(xy[0] - xy[9])))
    pip = angle(xyz[5], xyz[6], xyz[7])
    if palm < 12 or pip is None:
        return {"valid": False}
    pinch = float(np.linalg.norm(xyz[4] - xyz[8]) / max(np.linalg.norm(xyz[5] - xyz[17]), 1e-8))
    x0, y0, x1, y1 = config.target_roi
    in_target = x0 <= p[8, 0] <= x1 and y0 <= p[8, 1] <= y1
    return {"valid": True, "index_pip_deg": pip, "pinch_palm_widths": pinch,
            "tip_in_target": bool(in_target), "palm_width_px": palm_width, "palm_scale_px": palm,
            "tip_xy": xy[8].tolist(), "palm_xy": np.mean(xy[[0, 5, 9, 13, 17]], axis=0).tolist(),
            "press_shape": bool(in_target and config.index_pip_min_deg <= pip <= config.index_pip_max_deg
                                and pinch > config.pinch_max_palm_widths)}


class PoseState:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.previous_time = None
        self.previous_tip = None
        self.smoothed_speed = None
        self.support_since = None
        self.release_since = None
        self.active = False

    def update(self, timestamp, features):
        c = self.config
        if not features.get("valid"):
            self.reset()
            return {"state": "UNOBSERVED", "candidate": False, "speed": None, "dwell_s": 0.0}
        if self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError("Timestamps must increase")
        if self.previous_time is not None and timestamp - self.previous_time > c.max_observation_gap_s:
            self.reset()
        speed = None
        if self.previous_time is not None:
            dt = timestamp - self.previous_time
            speed = float(np.linalg.norm(np.asarray(features["tip_xy"]) - self.previous_tip)
                          / features.get("palm_scale_px", features["palm_width_px"]) / dt)
            alpha = 1 - math.exp(-dt / c.smoothing_tau_s)
            self.smoothed_speed = speed if self.smoothed_speed is None else (
                alpha * speed + (1 - alpha) * self.smoothed_speed)
        self.previous_time = timestamp
        self.previous_tip = np.asarray(features["tip_xy"])
        stable = self.smoothed_speed is not None and self.smoothed_speed <= c.max_tip_speed_hand_scales_s
        support = features["press_shape"] and stable
        if support:
            if self.support_since is None:
                self.support_since = timestamp
            self.release_since = None
            if timestamp - self.support_since >= c.evidence_dwell_s:
                self.active = True
        else:
            self.support_since = None
            if self.release_since is None:
                self.release_since = timestamp
            if timestamp - self.release_since >= c.release_s:
                self.active = False
        if self.active:
            state = "PRESS-LIKE POSE (CANDIDATE)"
        elif features["pinch_palm_widths"] <= c.pinch_max_palm_widths:
            state = "PINCH / GRIP"
        elif self.smoothed_speed is not None and not stable:
            state = "MOVING / REPOSITIONING"
        elif features["press_shape"]:
            state = "CHECKING PRESS-LIKE POSE"
        else:
            state = "OTHER HAND POSE"
        return {"state": state, "candidate": self.active, "current_frame_support": bool(support),
                "speed": self.smoothed_speed,
                "dwell_s": 0.0 if self.support_since is None else timestamp - self.support_since}


def match_tracks(features, tracks, timestamp, config):
    """Unique nearest-palm assignment; independent of MediaPipe list ordering."""
    available = [k for k, v in tracks.items() if timestamp - v["last_seen"] <= config.max_observation_gap_s]
    options = [[None] + available for _ in features]
    best, cost_best = None, float("inf")
    for assignment in itertools.product(*options):
        assigned = [k for k in assignment if k is not None]
        if len(set(assigned)) != len(assigned):
            continue
        cost = 0.0
        for f, k in zip(features, assignment):
            if k is None:
                cost += 1.5
            else:
                distance = np.linalg.norm(np.asarray(f["palm_xy"]) - tracks[k]["palm_xy"])
                cost += distance / max(f.get("palm_scale_px", f["palm_width_px"]), 12)
        if cost < cost_best:
            best, cost_best = assignment, cost
    return list(best or [])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(frame, message, xy, scale=0.6, color=(235, 235, 235)):
    import cv2
    cv2.putText(frame, message, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(frame, hands, timestamp, config):
    import cv2
    h, w = frame.shape[:2]
    canvas = np.zeros((h + HEADER_HEIGHT, w, 3), dtype=np.uint8)
    canvas[:] = (24, 22, 19)
    canvas[HEADER_HEIGHT:HEADER_HEIGHT + h] = frame
    text(canvas, "Hand Action Recognition", (20, 39), 0.9)
    view = canvas[HEADER_HEIGHT:HEADER_HEIGHT + h]
    colors = [(220, 170, 95), (105, 210, 240), (100, 200, 130), (175, 140, 225), (220, 160, 180)]
    for hand in hands:
        p = np.asarray(hand["landmarks"])[:, :2] * [w, h]
        p = p.astype(int)
        for finger, chain in enumerate([[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12],
                                         [9, 13, 14, 15, 16], [13, 17, 18, 19, 20]]):
            for a, b in zip(chain, chain[1:]):
                cv2.line(view, tuple(p[a]), tuple(p[b]), colors[finger], 2, cv2.LINE_AA)
            for index in chain[1:]:
                cv2.circle(view, tuple(p[index]), 3, colors[finger], -1, cv2.LINE_AA)
        cv2.line(view, tuple(p[0]), tuple(p[17]), colors[-1], 2, cv2.LINE_AA)
        col = (95, 240, 130) if hand["candidate"] else (230, 210, 120)
        cv2.circle(view, tuple(p[8]), 11, col, 2, cv2.LINE_AA)
        label = f"Hand {hand['track_id']} Gesture: {GESTURE_LABELS.get(hand['state'], 'Unobserved')}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = int(np.clip(p[0, 0] - 40, 8, w - tw - 8))
        y = int(np.clip(p[0, 1] - 18, th + 10, h - baseline - 8))
        cv2.rectangle(view, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 6), (24, 22, 19), -1)
        text(view, label, (x, y), 0.55, col)
    return canvas


def run(args):
    import cv2
    import imageio_ffmpeg
    cv2.setNumThreads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = Config(target_roi=tuple(args.target_roi))
    receipt = json.loads((args.landmarks.parent / "RUN_RECEIPT.json").read_text())
    if sha(args.source) != receipt["input_sha256"]:
        raise ValueError("Landmark cache does not match source video")
    cached = [json.loads(s) for s in args.landmarks.read_text().splitlines()]
    if len(cached) != receipt["frames_processed"] or any(r["frame_index"] != i for i, r in enumerate(cached)):
        raise ValueError("Incomplete or unordered landmark cache")
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError("Cannot decode source")
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    writer = cv2.VideoWriter(str(output / "raw_output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height + HEADER_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Cannot create video writer")
    tracks, rows, first_pts = {}, [], None
    for index, cache_row in enumerate(cached):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Landmark cache exceeds decoded video length")
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if first_pts is None:
            first_pts = pts
        timestamp = pts - first_pts
        if rows and timestamp <= rows[-1]["timestamp_s"]:
            raise RuntimeError("Nonmonotonic source timestamps")
        features, observations = [], []
        for hand in cache_row["hands"]:
            feature = pose_features(hand["landmarks"], width, height, config)
            if feature["valid"]:
                features.append(feature)
                observations.append(hand)
        assignment = match_tracks(features, tracks, timestamp, config)
        hands = []
        seen = set()
        for feature, observation, identity in zip(features, observations, assignment):
            if identity is None:
                identity = max(tracks, default=0) + 1
                tracks[identity] = {"classifier": PoseState(config)}
            track = tracks[identity]
            state = track["classifier"].update(timestamp, feature)
            track.update(last_seen=timestamp, palm_xy=feature["palm_xy"])
            seen.add(identity)
            hands.append({"track_id": identity, **feature, **state, "landmarks": observation["landmarks"]})
        for identity in tracks.keys() - seen:
            tracks[identity]["classifier"].reset()
        row = {"frame_index": index, "source_pts_s": pts, "timestamp_s": timestamp, "hands": hands,
               "candidate": any(h["candidate"] for h in hands)}
        rows.append(row)
        rendered = draw(frame, hands, timestamp, config)
        writer.write(rendered)
        if index % 20 == 0 or index == len(cached) - 1:
            cv2.imwrite(str(output / f"preview_{index:03d}.jpg"), rendered)
    if cap.read()[0]:
        raise RuntimeError("Source video has frames missing from landmark cache")
    cap.release()
    writer.release()
    # Restore each source-frame timestamp rather than using the previous
    # reproduction's index/average-FPS clock. This preserves VFR motion timing.
    expr = "+".join(f"{r['timestamp_s']:.9f}*eq(N\\,{i})" for i, r in enumerate(rows))
    filter_text = "settb=1/60000,setpts=(" + expr + ")/TB"
    (output / "timestamps.filter").write_text(filter_text)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "warning", "-n",
           "-i", str(output / "raw_output.mp4"), "-i", str(args.source),
           "-map", "0:v:0", "-map", "1:a?", "-vf", filter_text,
           "-fps_mode", "vfr", "-enc_time_base", "1:60000", "-video_track_timescale", "60000",
           "-c:v", "libx264", "-x264-params", "fps=30/1", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-movflags", "+faststart", str(output / "result.mp4")]
    subprocess.run(cmd, check=True)
    check = cv2.VideoCapture(str(output / "result.mp4"))
    decoded_pts = []
    while True:
        ok, _ = check.read()
        if not ok:
            break
        decoded_pts.append(check.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    check.release()
    if len(decoded_pts) != len(rows):
        raise RuntimeError("Final output lost or duplicated frames")
    max_error = max(abs(a - r["timestamp_s"]) for a, r in zip(decoded_pts, rows))
    if max_error > 0.002:
        raise RuntimeError(f"Video timestamp mismatch: {max_error}")
    episodes = []
    for identity in tracks:
        active = None
        for row in rows:
            hand = next((h for h in row["hands"] if h["track_id"] == identity), None)
            if hand and hand["candidate"]:
                if active is None:
                    active = {"track_id": identity, "start_s": row["timestamp_s"], "first_frame": row["frame_index"]}
                active.update(end_s=row["timestamp_s"], last_frame=row["frame_index"])
            elif active is not None:
                episodes.append(active)
                active = None
        if active is not None:
            episodes.append(active)
    for episode in episodes:
        episode["observed_span_s"] = episode["end_s"] - episode["start_s"]
    (output / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    states = Counter(h["state"] for r in rows for h in r["hands"])
    summary = {"status": "complete", "implementation": "manually configured pose-and-dwell heuristic",
               "title": "Hand Action Recognition", "config": asdict(config),
               "source": str(args.source.resolve()), "source_sha256": sha(args.source),
               "landmark_cache": str(args.landmarks.resolve()), "landmark_cache_sha256": sha(args.landmarks),
               "code_sha256": sha(__file__), "frames": len(rows), "candidate_frames": sum(r["candidate"] for r in rows),
               "hand_state_counts": dict(states), "candidate_episodes": episodes,
               "timing": {"source_first_pts_s": first_pts, "normalized_last_pts_s": rows[-1]["timestamp_s"],
                          "max_output_timestamp_error_s": max_error},
               "limits": ["Work area is manually configured for this video; no contact detection.",
                          "Candidate pose does not establish physical pressing, direction normal to skin, or force.",
                          "No supervised action training or held-out action labels; no accuracy estimate.",
                          "Rules can confuse pinching, resting and pressing; boundaries are heuristic."],
               "landmark_reference": "https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python",
               "result": str(output / "result.mp4")}
    (output / "RUN_RECEIPT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, default=HERE / "reproductions/20260922/hands/frames.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-roi", type=float, nargs=4, default=list(Config().target_roi),
                        metavar=("X0", "Y0", "X1", "Y1"))
    args = parser.parse_args()
    x0, y0, x1, y1 = args.target_roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        parser.error("Target ROI must be ordered normalized coordinates")
    run(args)
