from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .execution import SkillExecutor
from .evidence_backends import default_evidence_backends
from .reporting import write_report
from .schema import AssessmentResult, ToolCapability
from .specification import load_plan_with_trace
from .tasking import load_task_package
from .temporal import default_temporal_operators
from .tools import (ToolRegistry, export_monitoring_hit_frames,
                    export_representative_frame, locate_evaluation_window,
                    sample_timestamps, write_scores_csv)
from .tools import write_score_plot
from .verifier import ExplicitVerifier


class AssessmentPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AssessmentPipeline":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def run(self) -> Path:
        cfg = self.config
        task_id = str(cfg.get("task_id", cfg.get("task", {}).get("id", "cholec_cvs")))
        task_package = load_task_package(task_id)
        plan, planner_trace = load_plan_with_trace(
            cfg["spec_path"], cfg.get("planner"), task_package,
        )
        run_stem = task_package.artifact_stem(cfg["video_id"])
        run_dir = Path(cfg["output_root"]) / f"{run_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "assessment_plan.json").write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if planner_trace:
            (run_dir / "planner_runtime.json").write_text(
                json.dumps({
                    "planner": plan.planner,
                    "model": plan.planner_model,
                    "live_request_count": sum("latency_s" in item for item in planner_trace),
                    "trace_count": len(planner_trace),
                    "trace": planner_trace,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        registry = ToolRegistry()
        source_arguments = {
            "phase_annotation_path": cfg.get("phase_annotation_path"),
            "annotation_path": cfg.get("task_annotation_path", cfg.get("cvs_annotation_path")),
            "video_path": cfg.get("video_path"),
        }

        def locate_task_window(**_: Any) -> dict[str, float]:
            if task_package.dataset_adapter is None:
                return locate_evaluation_window(
                    cfg["phase_annotation_path"], plan.candidate_phase, plan.anchor_phase,
                )
            return task_package.dataset_adapter.locate_window(
                cfg["video_id"], **source_arguments,
            )

        locator_name = (
            plan.window_spec.locator_tool if plan.window_spec
            else "locate_evaluation_window"
        )
        registry.register(
            locator_name, locate_task_window,
            ToolCapability(
                tool_id=locator_name, role="context",
                capabilities=["evaluation_window_localization"],
                output_schema="EvaluationWindow",
            ),
        )
        registry.register(
            "sample_timestamps", sample_timestamps,
            ToolCapability(
                tool_id="sample_timestamps", role="sampler",
                capabilities=["temporal_sampling"], output_schema="TimestampSeries",
            ),
        )
        registry.register(
            "export_representative_frame", export_representative_frame,
            ToolCapability(
                tool_id="export_representative_frame", role="artifact_writer",
                capabilities=["representative_frame_export"],
            ),
        )
        window = registry.call(locator_name)
        timestamps = registry.call("sample_timestamps", start_s=window["start_s"], end_s=window["end_s"], sampling_fps=float(cfg["sampling_fps"]))

        backend = cfg["visual_backend"]
        evidence_tool = default_evidence_backends().build(
            backend, cfg, task_package, source_arguments,
        )
        scorer = evidence_tool.scorer
        oracle = evidence_tool.oracle

        if getattr(scorer, "metadata", None):
            (run_dir / "visual_tool_metadata.json").write_text(
                json.dumps(scorer.metadata, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            registry.calls.append({
                "tool": "load_visual_checkpoint",
                "arguments": {"backend": backend},
                "result": {
                    "tool_type": scorer.metadata.get("tool_type"),
                    "checkpoint_version": scorer.metadata.get("checkpoint_version"),
                    "development_only": scorer.metadata.get("development_only", False),
                    "source_checkpoint_sha256": scorer.metadata.get("source_checkpoint_sha256"),
                },
            })

        if evidence_tool.sampling_fps is not None:
            timestamps = registry.call(
                "sample_timestamps", start_s=window["start_s"], end_s=window["end_s"],
                sampling_fps=evidence_tool.sampling_fps,
            )

        if evidence_tool.accepts_requirement:
            def visual_evidence_tool(
                criterion, timestamps, visual_requirement, evidence_query=None,
            ):
                return scorer.score(criterion, timestamps, visual_requirement)
        else:
            def visual_evidence_tool(
                criterion, timestamps, visual_requirement, evidence_query=None,
            ):
                return scorer.score(criterion, timestamps)

        registry.register(
            "criterion_visual_evidence", visual_evidence_tool,
            ToolCapability(
                tool_id="criterion_visual_evidence", role="evidence_provider",
                capabilities=["visual_state_recognition", "temporal_visual_evidence"],
                input_schema="EvidenceQuery+TimestampSeries",
                output_schema="TemporalEvidenceSeries",
                supported_tasks=[task_id],
            ),
        )

        temporal_operators = default_temporal_operators()
        for tool_name, operator_name in (
            ("stable_evidence_aggregation", "stable_state"),
            ("state_transition_aggregation", "persistent_state_transition"),
        ):
            operator = temporal_operators.get(operator_name)

            def aggregate_evidence(
                points, temporal_parameters, temporal_policy=None, _operator=operator,
            ):
                return _operator(points, temporal_parameters, temporal_policy)

            registry.register(
                tool_name, aggregate_evidence,
                ToolCapability(
                    tool_id=tool_name, role="temporal_operator",
                    capabilities=[operator_name],
                    input_schema="TemporalEvidenceSeries+TemporalPolicy",
                    output_schema="CriterionEvidence",
                ),
            )
        registry.register(
            "explicit_verifier", lambda **_: None,
            ToolCapability(
                tool_id="explicit_verifier", role="verifier",
                capabilities=["logical_verification"],
                input_schema="VerificationExpression+CriterionEvidence",
                output_schema="AssessmentResult",
            ),
        )

        scores, evidence, skill_traces = {}, {}, {}
        temporal_cfg = cfg["temporal"]
        criterion_temporal_cfg = cfg.get("criterion_temporal", {})
        executor = SkillExecutor(registry)
        for criterion in plan.criteria:
            effective_temporal = {
                **temporal_cfg,
                **criterion.temporal_parameters,
                # Criterion-specific values are calibrated properties of the
                # selected visual tool. They override generic LLM defaults but
                # not SOP semantics such as criterion selection and logic.
                **criterion_temporal_cfg.get(criterion.key, {}),
            }
            execution = executor.execute_criterion(
                criterion=criterion,
                timestamps=timestamps,
                temporal_parameters=effective_temporal,
            )
            scores[criterion.key] = execution.scores
            evidence[criterion.key] = execution.evidence
            skill_traces[criterion.key] = execution.executed_tools
        (run_dir / "skill_execution.json").write_text(
            json.dumps(skill_traces, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        write_scores_csv(run_dir / "scores.csv", scores)
        if evidence_tool.runtime_kind in {
            "mllm", "small_mllm_fusion", "spatial_mllm_fusion",
        }:
            (run_dir / "visual_evidence.json").write_text(
                json.dumps({key: [asdict(item) for item in value] for key, value in scorer.observations.items()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            request_latencies = [item["latency_s"] for item in scorer.request_log]
            (run_dir / "qwen_runtime.json").write_text(json.dumps({
                "model": scorer.model,
                "request_count": len(scorer.request_log),
                "total_request_latency_s": sum(request_latencies),
                "mean_request_latency_s": sum(request_latencies) / max(1, len(request_latencies)),
                "requests": scorer.request_log,
                "frame_size": [scorer.frame_width, scorer.frame_height],
                "clip_frames": scorer.clip_frames,
                "max_queries_per_criterion": scorer.max_queries,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        if evidence_tool.runtime_kind in {"small_mllm_fusion", "spatial_mllm_fusion"}:
            (run_dir / "fusion_evidence.json").write_text(
                json.dumps(scorer.fusion_audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            hit_records = [
                record
                for criterion_audit in scorer.fusion_audit.values()
                for record in criterion_audit.get("queries", [])
            ]
            visual_cfg = cfg.get("monitoring_visualization", {})
            hit_artifacts = export_monitoring_hit_frames(
                cfg["video_path"], run_dir / "monitoring_hits", hit_records,
                context_label=str(visual_cfg.get(
                    "context_label", f"{task_package.task_name} | video {cfg['video_id']}",
                )),
                object_observations=getattr(
                    getattr(scorer, "small_scorer", None),
                    "object_observations", None,
                ),
            )
            (run_dir / "monitoring_hit_artifacts.json").write_text(
                json.dumps(hit_artifacts, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            registry.calls.append({
                "tool": "export_monitoring_hit_frames",
                "arguments": {"record_count": len(hit_records)},
                "result": hit_artifacts,
            })
        spatial_scorer = (
            scorer.small_scorer
            if evidence_tool.runtime_kind == "spatial_mllm_fusion"
            else scorer
        )
        if evidence_tool.runtime_kind in {"spatial_fusion", "spatial_mllm_fusion"}:
            (run_dir / "object_observations.json").write_text(
                json.dumps({
                    "meaning": "Predicted object observations; not task ground truth.",
                    "frames": [item.to_dict() for item in spatial_scorer.object_observations],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if getattr(scorer, "ordinal", False):
            ordinal_evidence = scorer.structured_ordinal_evidence(timestamps)
            ordinal_path = run_dir / "ordinal_visual_evidence.json"
            ordinal_path.write_text(
                json.dumps(ordinal_evidence, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            registry.calls.append({
                "tool": "structured_ordinal_visual_evidence",
                "arguments": {"n_timestamps": len(timestamps)},
                "result": {"path": str(ordinal_path), "criteria": list(ordinal_evidence)},
            })
        positive_intervals = {
            key: value.positive_intervals for key, value in evidence.items()
        }
        write_score_plot(run_dir / "score_curves.png", scores, positive_intervals)
        registry.calls.append({"tool": "write_score_plot", "arguments": {"n_criteria": len(scores)}, "result": {"path": str(run_dir / "score_curves.png")}})

        verifier_cfg = cfg["verifier"]
        verifier = ExplicitVerifier(
            float(verifier_cfg["pass_confidence"]),
            float(verifier_cfg["uncertain_confidence"]),
            fail_confidence=float(verifier_cfg.get("fail_confidence", verifier_cfg["pass_confidence"])),
            min_assessable_coverage=float(verifier_cfg.get("min_assessable_coverage", 0.5)),
            min_negative_points=int(verifier_cfg.get("min_negative_points", 2)),
        )
        criteria, overall, confidence = verifier.verify(plan, evidence)
        registry.calls.append({"tool": "explicit_verifier", "arguments": {"logic": plan.logic, **verifier_cfg}, "result": {"overall_verdict": overall, "overall_confidence": confidence}})
        for verdict in criteria:
            if verdict.evidence_intervals:
                best = max(verdict.evidence_intervals, key=lambda item: item.confidence)
                target = run_dir / "representative_frames" / f"{verdict.key}_{best.representative_time_s:.1f}s.jpg"
                best.representative_frame = registry.call("export_representative_frame", video_path=cfg["video_path"], time_s=best.representative_time_s, output_path=target)
                verdict.representative_frame = best.representative_frame

        notes = [
            "Verdicts are emitted by ExplicitVerifier; the planner and reporter cannot override them.",
            f"Evidence is constrained by task package {task_id!r} and its declared evaluation window.",
        ]
        if evidence_tool.checkpoint_backed and getattr(scorer, "metadata", {}).get("development_only"):
            notes.append("The specialized visual checkpoint is development-only and has no held-out validation result.")
        training_ids = {
            str(value) for value in getattr(scorer, "metadata", {}).get("train_video_ids", [])
        }
        if evidence_tool.checkpoint_backed and str(cfg["video_id"]) in training_ids:
            notes.append("This video was used to train the checkpoint; this run is an integration test, not test performance.")
        notes.extend(evidence_tool.notes)
        split_name = str(getattr(scorer, "metadata", {}).get("split_manifest", {}).get("name", ""))
        if "pilot" in split_name.lower():
            notes.append("This checkpoint uses a small preregistered pilot split; results are held-out pilot evidence, not final paper estimates.")
        elif getattr(scorer, "metadata", {}).get("pilot_only"):
            notes.append(
                "This checkpoint is an exploratory small-data pilot; it is not a final paper estimate."
            )
        result = AssessmentResult(
            video_id=str(cfg["video_id"]), evaluation_window=window, overall_verdict=overall,
            overall_confidence=confidence, criteria=criteria, backend=backend,
            development_oracle=oracle,
            notes=notes,
            task_id=task_id,
            task_name=task_package.task_name,
        )
        # Keep the original positive-interval artifact stable for existing metric
        # scripts, and add the complete three-state evidence artifact separately.
        (run_dir / "evidence.json").write_text(
            json.dumps({key: [asdict(item) for item in value.positive_intervals] for key, value in evidence.items()}, indent=2),
            encoding="utf-8",
        )
        (run_dir / "evidence_summary.json").write_text(
            json.dumps({key: asdict(value) for key, value in evidence.items()}, indent=2),
            encoding="utf-8",
        )
        (run_dir / "tool_calls.jsonl").write_text("\n".join(json.dumps(call, ensure_ascii=False) for call in registry.calls) + "\n", encoding="utf-8")
        (run_dir / "tool_manifest.json").write_text(
            json.dumps(registry.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(run_dir / "report.md", result)
        return run_dir
