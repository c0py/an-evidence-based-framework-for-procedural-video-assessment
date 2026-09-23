"""Run a one-shot, direct-video Qwen-VL assessment baseline.

This baseline deliberately bypasses SpecificationPlanner, ToolRegistry,
StableEvidenceAggregator, and ExplicitVerifier.  It submits the complete
candidate-phase video clip and natural-language SOP in one multimodal request,
then preserves the model's direct criterion/verdict output for comparison with
the modular skill-tool pipeline.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from threading import Thread
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvs_assessment.tools import locate_evaluation_window


CRITERIA = ("two_structures", "cystic_plate", "hepatocystic_triangle")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


@contextmanager
def serve_directory(root: Path):
    handler = partial(QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def make_candidate_clip(video_path: Path, output_path: Path, start_s: float, end_s: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_s = end_s - start_s
    if duration_s <= 0:
        raise ValueError("Candidate phase must have positive duration")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_s:.3f}", "-i", str(video_path), "-t", f"{duration_s:.3f}",
        "-an", "-vf",
        "fps=1,scale=336:336:force_original_aspect_ratio=decrease,"
        "pad=336:336:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    subprocess.run(command, check=True)


def prompt_text(
    specification: str, duration_s: float, phase_start_s: float,
    sampled_frames: int | None = None,
) -> str:
    sampling_note = (
        f"The model input contains {sampled_frames} chronological frames sampled uniformly "
        "from that complete interval. "
        if sampled_frames is not None else ""
    )
    return f"""You are directly assessing a surgical procedure video against the SOP below.
Do not call tools and do not assume facts that are not visible. The submitted video is the complete
candidate phase before clipping/cutting. Its local timeline starts at 0.0 seconds and lasts
{duration_s:.3f} seconds; local time 0 corresponds to source-video time {phase_start_s:.3f} seconds.
{sampling_note}

SOP:
{specification}

Directly judge all three canonical criteria: two_structures, cystic_plate, and
hepatocystic_triangle. A pass requires sustained visible evidence, not a single ambiguous frame.
Use uncertain when visibility or temporal coverage is insufficient. Evidence interval timestamps
MUST be seconds on the submitted clip's LOCAL 0-to-{duration_s:.3f} timeline.

Return ONLY one JSON object with exactly this structure:
{{
  "criteria": [
    {{
      "key": "two_structures|cystic_plate|hepatocystic_triangle",
      "verdict": "pass|fail|uncertain",
      "confidence": 0.0,
      "evidence_intervals": [],
      "observed_facts": ["short visible fact"],
      "rationale": "short rationale"
    }}
  ],
  "overall_verdict": "pass|fail|uncertain",
  "overall_confidence": 0.0,
  "overall_rationale": "short rationale"
}}
Return all three criteria exactly once. Overall pass requires all three criteria to pass.
The empty array in the structure is a schema placeholder. Replace it with zero or more objects having
numeric start_s, end_s, and confidence fields only when positive evidence is actually visible. Estimate
times from the uniform chronological frame positions and the stated duration. Use an empty list when no
positive interval is visible; never copy example timestamps and never emit a zero-length interval.
"""


def post_completion(base_url: str, model: str, api_key: str, video_url: str, prompt: str, timeout_s: float) -> tuple[str, dict, float]:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }],
        "temperature": 0,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Direct Qwen-VL request failed with HTTP {exc.code}: {detail[:1200]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach Qwen-VL server at {base_url}: {exc.reason}") from exc
    elapsed_s = time.monotonic() - started
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Direct Qwen-VL returned no completion: {body}") from exc
    return text, body, elapsed_s


def parse_json(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Direct Qwen-VL returned invalid JSON: {text[:1600]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Direct Qwen-VL response must be a JSON object")
    return value


def validate_and_convert(value: dict, video_id: int, window: dict[str, float]) -> tuple[dict, dict]:
    duration_s = float(window["end_s"] - window["start_s"])
    criteria_values = value.get("criteria")
    if not isinstance(criteria_values, list):
        raise RuntimeError("Direct response criteria must be a list")
    by_key = {item.get("key"): item for item in criteria_values if isinstance(item, dict)}
    if set(by_key) != set(CRITERIA) or len(criteria_values) != len(CRITERIA):
        raise RuntimeError(f"Direct response must contain each canonical criterion once: {list(by_key)}")
    evidence, result_criteria = {}, []
    for criterion in CRITERIA:
        item = by_key[criterion]
        verdict = item.get("verdict")
        confidence = float(item.get("confidence"))
        if verdict not in {"pass", "fail", "uncertain"} or not 0 <= confidence <= 1:
            raise RuntimeError(f"Invalid direct criterion verdict: {item}")
        converted = []
        intervals = item.get("evidence_intervals", [])
        if not isinstance(intervals, list):
            raise RuntimeError(f"evidence_intervals must be a list: {item}")
        for interval in intervals:
            local_start, local_end = float(interval["start_s"]), float(interval["end_s"])
            interval_confidence = float(interval.get("confidence", confidence))
            # Some JSON-constrained models copy a 0/0 placeholder to express
            # no evidence. Treat only the exact zero-confidence, zero-length
            # sentinel as an empty interval and retain the raw response.
            if local_start == local_end == 0.0 and interval_confidence == 0.0:
                continue
            if not 0 <= local_start < local_end <= duration_s + 1.0 or not 0 <= interval_confidence <= 1:
                raise RuntimeError(f"Invalid direct evidence interval: {interval}")
            absolute_start = float(window["start_s"]) + local_start
            absolute_end = min(float(window["end_s"]), float(window["start_s"]) + local_end)
            converted.append({
                "start_s": absolute_start, "end_s": absolute_end,
                "confidence": interval_confidence, "mean_score": interval_confidence,
                "duration_s": absolute_end - absolute_start,
                "representative_time_s": (absolute_start + absolute_end) / 2,
                "representative_frame": None,
            })
        evidence[criterion] = converted
        result_criteria.append({
            "key": criterion, "verdict": verdict, "confidence": confidence,
            "evidence_intervals": converted, "representative_frame": None,
            "observed_facts": item.get("observed_facts", []),
            "reason": str(item.get("rationale", "")),
        })
    overall = value.get("overall_verdict")
    overall_confidence = float(value.get("overall_confidence"))
    if overall not in {"pass", "fail", "uncertain"} or not 0 <= overall_confidence <= 1:
        raise RuntimeError(f"Invalid direct overall verdict: {value}")
    result = {
        "video_id": str(video_id), "evaluation_window": window,
        "overall_verdict": overall, "overall_confidence": overall_confidence,
        "criteria": result_criteria, "backend": "direct_qwen_vl",
        "development_oracle": False,
        "notes": [
            "One-shot direct-video baseline; no planner, tools, temporal aggregator, or explicit verifier were used.",
            "The complete candidate-phase clip was submitted; the model server uniformly samples a preregistered bounded number of video frames.",
            "Model-proposed local timestamps were converted back to source-video time.",
        ],
        "overall_rationale": str(value.get("overall_rationale", "")),
    }
    return result, evidence


def write_report(path: Path, result: dict, runtime: dict) -> None:
    lines = [
        f"# Direct Qwen-VL baseline — video {result['video_id']}", "",
        f"Overall: **{result['overall_verdict']}** ({result['overall_confidence']:.3f})", "",
        f"Latency: {runtime['request_latency_s']:.2f} s", "",
    ]
    for criterion in result["criteria"]:
        lines.extend([
            f"## {criterion['key']}", "",
            f"Verdict: **{criterion['verdict']}** ({criterion['confidence']:.3f})", "",
            f"Rationale: {criterion['reason']}", "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot direct-video Qwen-VL CVS baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-s", type=float, default=900)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    video_id = int(cfg["video_id"])
    specification = Path(cfg["spec_path"]).read_text(encoding="utf-8")
    window = locate_evaluation_window(
        cfg["phase_annotation_path"], candidate_phase="CalotTriangleDissection",
        anchor_phase="ClippingCutting",
    )
    output_root = Path(cfg["output_root"]) / "direct_qwen_vl"
    run_dir = output_root / f"video{video_id:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    clip_path = run_dir / f"video{video_id:02d}_candidate_phase.mp4"
    make_candidate_clip(Path(cfg["video_path"]), clip_path, window["start_s"], window["end_s"])
    prompt = prompt_text(specification, window["end_s"] - window["start_s"], window["start_s"])
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    with serve_directory(run_dir) as media_base_url:
        raw_text, response_body, elapsed_s = post_completion(
            args.base_url, args.model, args.api_key,
            f"{media_base_url}/{clip_path.name}", prompt, args.timeout_s,
        )
    (run_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
    runtime = {
        "request_latency_s": elapsed_s,
        "model": args.model,
        "base_url": args.base_url,
        "clip_path": str(clip_path.resolve()),
        "clip_bytes": clip_path.stat().st_size,
        "usage": response_body.get("usage", {}),
        "direct_request_count": 1,
    }
    (run_dir / "raw_response_body.json").write_text(
        json.dumps(response_body, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (run_dir / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    value = parse_json(raw_text)
    result, evidence = validate_and_convert(value, video_id, window)
    (run_dir / "direct_visual_evidence.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(run_dir / "report.md", result, runtime)
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
