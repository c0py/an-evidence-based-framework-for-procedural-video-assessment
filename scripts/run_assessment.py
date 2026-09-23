from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvs_assessment.pipeline import AssessmentPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auditable procedural-video skill-tool assessment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", help="Optional specification intervention file.")
    parser.add_argument(
        "--backend",
        help="Override the registered evidence-backend id (built-in or externally registered).",
    )
    parser.add_argument("--planner", choices=["rule_based", "llm"], help="Override the specification planner backend.")
    parser.add_argument("--task", help="Override the registered task package id.")
    args = parser.parse_args()
    pipeline = AssessmentPipeline.from_yaml(args.config)
    if args.spec:
        pipeline.config["spec_path"] = str(Path(args.spec).resolve())
    if args.backend:
        pipeline.config["visual_backend"] = args.backend
    if args.planner:
        pipeline.config.setdefault("planner", {})["backend"] = args.planner
    if args.task:
        pipeline.config["task_id"] = args.task
    run_dir = pipeline.run()
    print(run_dir)


if __name__ == "__main__":
    main()
