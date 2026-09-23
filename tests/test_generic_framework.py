from __future__ import annotations

import unittest

import torch

from procedural_assessment import (
    CriterionEvidence,
    EvidenceInterval,
    ExplicitVerifier,
    SkillExecutor,
    TemporalEvidencePoint,
    ToolCapability,
    ToolRegistry,
    default_temporal_operators,
    load_task_package,
)
from cvs_assessment.models import TemporalOrdinalBoundaryHead
from cvs_assessment.specification import SpecificationPlanner


class TaskPackageTests(unittest.TestCase):
    def test_cholec_and_industreal_share_the_same_planner_class(self) -> None:
        cholec = load_task_package("cholec_cvs")
        industrial = load_task_package("industreal_assembly")
        cholec_plan = SpecificationPlanner(cholec).plan(
            "Assess two structures, cystic plate, and hepatocystic triangle."
        )
        industrial_plan = SpecificationPlanner(industrial).plan(
            "Assess base assembly, wheel assembly, and final assembly."
        )
        self.assertEqual(cholec_plan.task_id, "cholec_cvs")
        self.assertEqual(industrial_plan.task_id, "industreal_assembly")
        self.assertEqual(
            [criterion.temporal_policy.operator for criterion in industrial_plan.criteria],
            ["persistent_state_transition"] * 3,
        )
        order_rules = [
            item for item in industrial_plan.verification_expression.arguments
            if item.operator == "BEFORE"
        ]
        self.assertEqual(len(order_rules), 2)

    def test_industreal_skill_executes_with_generic_contracts(self) -> None:
        package = load_task_package("industreal_assembly")
        plan = SpecificationPlanner(package).plan(
            "Assess base assembly, wheel assembly, and final assembly."
        )
        transition_times = {
            "base_assembly_completed": 2.0,
            "wheel_assembly_completed": 6.0,
            "final_assembly_verified": 10.0,
        }
        registry = ToolRegistry()

        def evidence_tool(criterion, timestamps, visual_requirement, evidence_query):
            onset = transition_times[criterion]
            return [
                TemporalEvidencePoint(
                    criterion_id=criterion,
                    time_s=time_s,
                    state="satisfied" if time_s >= onset else "violated",
                    confidence=0.95,
                    source_tool="industreal_fixture_state_tool",
                )
                for time_s in timestamps
            ]

        registry.register(
            "criterion_visual_evidence", evidence_tool,
            ToolCapability(
                tool_id="criterion_visual_evidence", role="evidence_provider",
                capabilities=["object_state_recognition", "step_completion_recognition"],
            ),
        )
        transition = default_temporal_operators().get("persistent_state_transition")
        registry.register(
            "state_transition_aggregation", transition,
            ToolCapability(
                tool_id="state_transition_aggregation", role="temporal_operator",
                capabilities=["persistent_state_transition"],
            ),
        )
        executor = SkillExecutor(registry)
        timestamps = [float(value) for value in range(0, 15, 2)]
        evidence = {}
        for criterion in plan.criteria:
            execution = executor.execute_criterion(
                criterion, timestamps, criterion.temporal_policy.parameters,
            )
            evidence[criterion.key] = execution.evidence
            self.assertTrue(execution.evidence.positive_intervals)
        results, overall, _ = ExplicitVerifier(
            pass_confidence=0.6, uncertain_confidence=0.4,
        ).verify(plan, evidence)
        self.assertEqual(overall, "pass")
        self.assertEqual([item.verdict for item in results], ["pass", "pass", "pass"])

    def test_industreal_wrong_step_order_is_rejected_by_generic_verifier(self) -> None:
        package = load_task_package("industreal_assembly")
        plan = SpecificationPlanner(package).plan(
            "Assess base assembly, wheel assembly, and final assembly."
        )
        # Every state is individually satisfied, but the wheel appears before
        # the base.  The package-level BEFORE expression must reject the run.
        starts = {
            "base_assembly_completed": 8.0,
            "wheel_assembly_completed": 2.0,
            "final_assembly_verified": 12.0,
        }
        evidence = {
            key: CriterionEvidence(
                positive_intervals=[EvidenceInterval(
                    start_s=start, end_s=14.0, confidence=0.9,
                    mean_score=0.9, duration_s=14.0 - start,
                    representative_time_s=start,
                )]
            )
            for key, start in starts.items()
        }
        results, overall, confidence = ExplicitVerifier(
            pass_confidence=0.6, uncertain_confidence=0.4,
        ).verify(plan, evidence)
        self.assertEqual([item.verdict for item in results], ["pass", "pass", "pass"])
        self.assertEqual(overall, "fail")
        self.assertAlmostEqual(confidence, 0.9)


class GenericTemporalModelTests(unittest.TestCase):
    def test_temporal_head_supports_arbitrary_criterion_count(self) -> None:
        criteria = ("step_a", "step_b", "step_c", "step_d")
        model = TemporalOrdinalBoundaryHead(
            feature_dim=8, hidden_dim=16, dilations=(1, 2), criteria=criteria,
        )
        features = torch.randn(2, 7, 8)
        logits = torch.randn(2, 7, 2 * len(criteria))
        output = model(features, logits, torch.rand(2, 7))
        self.assertEqual(output["support_or_full"].shape, (2, 7, 4))
        self.assertEqual(output["full_only"].shape, (2, 7, 4))
        self.assertEqual(output["boundary"].shape, (2, 7, 4, 2))


if __name__ == "__main__":
    unittest.main()
