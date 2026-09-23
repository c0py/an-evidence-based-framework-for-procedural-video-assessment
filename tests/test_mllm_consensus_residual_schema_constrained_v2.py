from __future__ import annotations

import copy
import unittest

from jsonschema import ValidationError, validate

from cvs_assessment.foundation_consensus_residual_arbitration import build_consensus_prior
from cvs_assessment.mllm_consensus_residual_orchestration import (
    ConsensusResidualFrozenMLLMJudge, ConsensusResidualJudgeRequest,
)
from cvs_assessment.mllm_consensus_residual_schema_constrained_v2 import (
    ConsensusResidualSchemaConstrainedV2Mixin,
    build_fixed_object_schema,
    expand_fixed_object_response,
)
from cvs_assessment.mllm_orchestration import CriterionContract, MLLMAblation


def candidate(name, probability):
    return {
        "schema_version": "foundation_mllm_plugin_judgment_v3",
        "task_id": "task", "sample_id": "sample", "model": "qwen",
        "foundation_model_parameters_updated": False, "decision_mode": "ordinal_state",
        "ablation": {"name": name},
        "prediction": {"frames": [{"frame_index": 0, "timestamp_s": 0.0, "criteria": [{
            "criterion_id": "c", "foundation_state": "F", "foundation_confidence": "m",
            "visibility": "good", "probability_satisfied": probability,
        }]}]},
        "multibranch_slice_context": {
            "source_ablation": name, "original_frame_ids": [0], "original_frame_count": 1,
            "routed_frame_ids": [0], "whole_timeline_case_summary": "",
            "whole_timeline_plugin_assessment": [], "ground_truth_or_labels_included": False,
        },
    }


ABLATION = MLLMAblation(
    "full_framework_consensus_residual", True, ("visual", "temporal"),
    decision_protocol="ordinal_state", require_fact_only_plugins=True,
)


class Judge(ConsensusResidualSchemaConstrainedV2Mixin, ConsensusResidualFrozenMLLMJudge):
    pass


class FixedObjectSchemaTests(unittest.TestCase):
    def setUp(self):
        candidates = [candidate("mllm_skill_visual", .8),
                      candidate("full_framework_calibrated_multibranch", .6)]
        self.request = ConsensusResidualJudgeRequest(
            task_id="task", sample_id="sample",
            criteria=[CriterionContract("c", "C", "C")], frame_ids=[0],
            timestamps_s=[0.0], frame_jpegs=[b"x"], skill_text="skill",
            foundation_candidate_judgments=candidates, plugin_coverage_manifests=[],
            consensus_prior=build_consensus_prior(candidates),
        )
        self.value = {
            "frame_decisions": {"f0": {"c0": {
                "state": "F", "confidence": "m", "visibility": "g",
            }}},
            "plugin_use": {}, "case_summary": "visible consensus",
            "conflict_decisions": {"f0": {
                "accepted_role_mask": 1, "evidence_scope": "image_only", "reason": "visible",
            }},
            "residual_decisions": {"r0": {
                "action": "H", "accepted_role_mask": 1,
                "basis": "real_image_consensus", "reason": "hold consensus",
            }},
            "fact_decisions": {},
        }

    def test_wire_instance_has_no_arrays_and_expands(self):
        schema = build_fixed_object_schema(self.request, ABLATION)
        validate(self.value, schema)

        def visit(value):
            self.assertNotIsInstance(value, list)
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
        visit(self.value)
        expanded = expand_fixed_object_response(self.value, self.request, ABLATION)
        self.assertEqual(expanded["frame_predictions"][0]["s"], ["F"])
        self.assertEqual(expanded["conflict_dispositions"][0]["rejected_hypotheses"],
                         ["full_framework_calibrated_multibranch"])

    def test_missing_fixed_key_is_rejected(self):
        invalid = copy.deepcopy(self.value)
        invalid["frame_decisions"].pop("f0")
        with self.assertRaises(ValidationError):
            validate(invalid, build_fixed_object_schema(self.request, ABLATION))

    def test_downgrade_support_basis_is_rejected(self):
        invalid = copy.deepcopy(self.value)
        invalid["residual_decisions"]["r0"].update({
            "action": "D", "basis": "direct_image_support",
        })
        with self.assertRaises(ValidationError):
            validate(invalid, build_fixed_object_schema(self.request, ABLATION))

    def test_legacy_validation_accepts_expansion(self):
        normalized = Judge("http://unused", "qwen").validate_response(
            self.value, self.request, set(), decision_protocol="ordinal_state",
        )
        self.assertEqual(normalized["residual_dispositions"][0]["action"], "hold")
        self.assertTrue(normalized["schema_constraint_audit"]["array_cardinality_keywords_avoided"])


if __name__ == "__main__":
    unittest.main()
