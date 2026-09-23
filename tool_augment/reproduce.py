#!/usr/bin/env python3
"""Run the packaged scripts on their real inputs with auditable runtime adaptations.

Use the repository .venv-qwen3vl interpreter. Optional local wheel dependencies
live in tool_augment/.runtime; original source, inputs and weights are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "extracted/tool_augment_code"
sys.path.insert(0, str(HERE / ".runtime"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / ".runtime_settings"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".runtime_settings/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("GLOG_minloglevel", "2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def video_info(path, decode=False):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    info = dict(width=int(cap.get(3)), height=int(cap.get(4)), fps=cap.get(5),
                declared_frames=int(cap.get(7)))
    if decode:
        count = 0
        while cap.read()[0]:
            count += 1
        info["decoded_frames"] = count
    cap.release()
    return info


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one source match: {old!r}")
    return source.replace(old, new, 1)


def run(module_name, source, output, trocar_scale=1.0):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    output.mkdir(parents=True, exist_ok=False)
    mapping = {"trocar": "trocar完成版.py", "disinfection": "消毒细节检测.py", "hands": "hand_rec.py"}
    original = PACKAGE / "src" / mapping[module_name]
    original_text = original.read_text()
    adapted = original_text
    adaptations = ["headless execution", "explicit input/output/model paths", "per-frame audit callback"]
    if module_name == "trocar":
        adapted = replace_once(adapted, "model(frame, conf=0.1)",
                               'model(frame, conf=0.1, device="cpu", verbose=False)')
        adapted = replace_once(adapted, "        out.write(frame)",
                               "        _audit_frame(locals())\n        out.write(frame)")
        adapted = adapted.replace("cv2.imshow('Detection', frame)", "pass  # headless")
        adapted = adapted.replace("cv2.waitKey(1)", "-1")
        adaptations.append("CPU inference; original detection and angle thresholds retained")
        if trocar_scale != 1.0:
            adapted = replace_once(adapted, "    center_x, center_y = width // 2, height // 2",
                                   f"    width, height = round(width * {trocar_scale}), round(height * {trocar_scale})\n    center_x, center_y = width // 2, height // 2")
            adapted = replace_once(adapted, "        # 检测肚脐",
                                   "        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)\n        # 检测肚脐")
            adaptations.append(f"explicit {trocar_scale}x input resize for fixed pixel thresholds; output coordinates are resized pixels")
    elif module_name == "disinfection":
        adapted = replace_once(adapted, "'--no_circle',default=True", "'--no_circle',default=False")
        adapted = replace_once(adapted, "        if vw is not None:\n            vw.write(base)",
                               "        _audit_frame(locals())\n        if vw is not None:\n            vw.write(base)")
        adaptations.append("repair --no_circle default so online circle fitting is enabled; fitting gates unchanged")
    else:
        adapted = replace_once(adapted, 'title_text = "手部动作识别"',
                               'title_text = "Hand Action Recognition"')
        adapted = replace_once(adapted, "fps = int(cap.get(cv2.CAP_PROP_FPS))",
                               "fps = float(cap.get(cv2.CAP_PROP_FPS))")
        adapted = replace_once(adapted, 'ImageFont.truetype("simhei.ttf", 100)',
                               'ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 32)')
        adapted = replace_once(adapted, "                processed_frame, results = self.process_frame(frame)",
                               "                processed_frame, results = self.process_frame(frame)\n                _audit_frame(locals())")
        adaptations += ["retain fractional input FPS instead of truncating to int", "use installed CJK font at 32px", "English title: Hand Action Recognition"]
    adapted = adapted.replace("cv2.destroyAllWindows()", "pass  # headless cleanup")
    copy_path = output / original.name
    copy_path.write_text(adapted)
    (output / "ADAPTATIONS.diff").write_text("".join(difflib.unified_diff(
        original_text.splitlines(True), adapted.splitlines(True), fromfile=str(original), tofile=str(copy_path))))
    spec = importlib.util.spec_from_file_location(f"reproduced_{module_name}", copy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    info = video_info(source, decode=True)
    if not info["decoded_frames"]:
        raise RuntimeError("Input has no decodable frames")
    rows = []
    started = time.monotonic()
    audit_file = (output / "frames.jsonl").open("w")

    def audit(local):
        idx = len(rows)
        row = {"frame_index": idx, "timestamp_s": idx / info["fps"]}
        if module_name == "trocar":
            points, navel = local["tube_points"], local["navel"]
            row.update(trocar_boxes=sum(len(r.boxes) for r in local["results"] if r.boxes is not None),
                       endpoint_count=len(points), endpoints=points, navel=navel,
                       angle_analysis_triggered=len(points) == 3 and navel is not None)
            if row["angle_analysis_triggered"]:
                bottom_index = min(range(3), key=lambda i: mod.distance(navel, points[i]))
                bottom = points[bottom_index]
                others = [p for i, p in enumerate(points) if i != bottom_index]
                foot = mod.perpendicular_foot(bottom, *others)
                row.update(left_angle_deg=float(mod.calculate_angle(others[0], bottom, foot)),
                           right_angle_deg=float(mod.calculate_angle(others[1], bottom, foot)))
            frame = local["frame"]
        elif module_name == "disinfection":
            mask = local["target_mask"]
            ratio = None
            if mask is not None and np.count_nonzero(mask):
                ratio = float(np.count_nonzero(cv2.bitwise_and(local["coverage"], local["coverage"], mask=mask)) / np.count_nonzero(mask))
            row.update(swab_detected=local["swab_box"] is not None,
                       stoma_detected=local["stoma_box"] is not None,
                       swab_box=local["swab_box"], stoma_box=local["stoma_box"],
                       tip=local["tip"], coverage_ratio=ratio,
                       online_circle=local["circle_est"],
                       reused_previous_swab=local["swab_box"] is None and local["swab_circle"] is not None)
            frame = local["base"]
        else:
            result = local["results"]
            hands = result.multi_hand_landmarks or []
            row.update(hand_count=len(hands), hands=[{
                "handedness": result.multi_handedness[i].classification[0].label,
                "handedness_score": float(result.multi_handedness[i].classification[0].score),
                "landmarks": [[float(p.x), float(p.y), float(p.z)] for p in h.landmark]
            } for i, h in enumerate(hands)])
            frame = local["processed_frame"]
        # OpenCV Hough endpoints may be numpy.int32/int64, depending on whether
        # a line was found. Normalize telemetry without changing algorithm data.
        row = json.loads(json.dumps(row, default=lambda value: value.item()
                                   if isinstance(value, np.generic) else value.tolist()))
        rows.append(row)
        audit_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if idx in {0, info["decoded_frames"] // 2, info["decoded_frames"] - 1}:
            cv2.imwrite(str(output / f"preview_{idx:04d}.jpg"), frame)
        if idx % 30 == 0:
            print(json.dumps({"module": module_name, "frames": idx + 1, "total": info["decoded_frames"]}), flush=True)

    mod._audit_frame = audit
    weights = None
    raw_video = output / "raw_output.mp4"
    try:
        if module_name == "trocar":
            weights = PACKAGE / "models/trocar.pt"
            mod.detect_tubes_and_navel(str(weights), str(source), str(raw_video))
        elif module_name == "disinfection":
            weights = PACKAGE / "models/棉签造口.pt"
            sys.argv = [str(copy_path), "--source", str(source), "--model", str(weights),
                        "--device", "cpu", "--no_show", "--out_dir", str(output)]
            mod.main()
            raw_video = output / f"{source.stem}_traj.mp4"
        else:
            detector = mod.HandSkeletonDetector()
            detector.process_video(str(source), str(raw_video), show_window=False)
            del detector
    finally:
        audit_file.close()
    if len(rows) != info["decoded_frames"]:
        raise RuntimeError(f"Incomplete run: {len(rows)}/{info['decoded_frames']}")
    raw_info = video_info(raw_video, decode=True)
    if raw_info["decoded_frames"] != len(rows):
        raise RuntimeError(f"Output frame count mismatch: {raw_info}")
    import imageio_ffmpeg
    final_video = output / "result.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "warning", "-n",
               "-i", str(raw_video), "-an", "-c:v", "libx264", "-crf", "18",
               "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_video)]
    subprocess.run(command, check=True)
    final_info = video_info(final_video, decode=True)
    if final_info["decoded_frames"] != len(rows):
        raise RuntimeError("H264 conversion frame count mismatch")
    if module_name == "trocar":
        measurements = {"frames_with_boxes": sum(r["trocar_boxes"] > 0 for r in rows),
                        "frames_with_navel": sum(r["navel"] is not None for r in rows),
                        "endpoint_count_histogram": dict(Counter(r["endpoint_count"] for r in rows)),
                        "frames_with_angle_analysis": sum(r["angle_analysis_triggered"] for r in rows)}
    elif module_name == "disinfection":
        coverages = [r["coverage_ratio"] for r in rows if r["coverage_ratio"] is not None]
        measurements = {"frames_with_swab": sum(r["swab_detected"] for r in rows),
                        "frames_with_stoma": sum(r["stoma_detected"] for r in rows),
                        "frames_with_online_circle": sum(r["online_circle"] is not None for r in rows),
                        "frames_reusing_previous_swab": sum(r["reused_previous_swab"] for r in rows),
                        "final_coverage_ratio": rows[-1]["coverage_ratio"],
                        "maximum_coverage_ratio": max(coverages) if coverages else None,
                        "target_radius_pixels": 200, "coverage_threshold": 0.95}
    else:
        import mediapipe
        measurements = {"hand_count_histogram": dict(Counter(r["hand_count"] for r in rows)),
                        "frames_with_hands": sum(r["hand_count"] > 0 for r in rows),
                        "mediapipe_version": mediapipe.__version__}
    receipt = {"status": "complete", "module": module_name, "input": str(source),
               "input_sha256": digest(source), "input_video": info, "output_video": final_info,
               "frames_processed": len(rows), "elapsed_s": time.monotonic() - started,
               "original_script": str(original), "original_script_sha256": digest(original),
               "adapted_script_sha256": digest(copy_path), "runner_sha256": digest(__file__),
               "weights": str(weights) if weights else "MediaPipe bundled hand models",
               "weights_sha256": digest(weights) if weights else None,
               "adaptations": adaptations, "measurements": measurements,
               "versions": {"python": sys.version, "torch": torch.__version__, "opencv": cv2.__version__,
                            "numpy": np.__version__, "ultralytics": ultralytics.__version__},
               "device": "cpu", "result": str(final_video), "transcode_command": command,
               "trocar_input_scale": trocar_scale if module_name == "trocar" else None,
               "note": "Pipeline reproduction; no accuracy evaluation or clinical validation performed."}
    write_json(output / "RUN_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=("trocar", "disinfection", "hands"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trocar-scale", type=float, default=1.0,
                        help="Explicit preprocessing variant for fixed pixel thresholds (default: original resolution)")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.trocar_scale <= 0 or (args.module != "trocar" and args.trocar_scale != 1.0):
        parser.error("--trocar-scale must be positive and applies only to trocar")
    run(args.module, source, args.output.resolve(), args.trocar_scale)


if __name__ == "__main__":
    main()
