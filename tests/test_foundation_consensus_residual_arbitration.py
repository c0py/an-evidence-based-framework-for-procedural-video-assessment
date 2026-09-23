import unittest

from cvs_assessment.foundation_consensus_residual_arbitration import (
    build_consensus_prior,
    merge_consensus_residual_judgment,
    select_consensus_residual_frame_ids,
)


def candidate(name, probabilities):
    return {
        "schema_version": "foundation_mllm_plugin_judgment_v3",
        "task_id": "task", "sample_id": "sample", "model": "qwen",
        "foundation_model_parameters_updated": False,
        "decision_mode": "ordinal_state", "ablation": {"name": name},
        "prediction": {"frames": [
            {"frame_index": frame_id, "timestamp_s": float(frame_id), "criteria": [
                {"criterion_id": "c", "foundation_state": "F" if p >= .6 else "P",
                 "foundation_confidence": "m", "visibility": "good",
                 "probability_satisfied": p}
            ]} for frame_id, p in enumerate(probabilities)
        ]},
    }


class ConsensusResidualTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            candidate("mllm_skill_visual", [.8, .5]),
            candidate("full_framework_calibrated_multibranch", [.6, .5]),
        ]

    def test_prior_and_route_are_label_free(self):
        prior = build_consensus_prior(self.candidates)
        self.assertAlmostEqual(prior["frames"][0]["criteria"][0]["consensus_probability"], .7)
        self.assertFalse(prior["raw_small_model_rows_used"])
        self.assertEqual(select_consensus_residual_frame_ids(self.candidates), [0])

    def test_merge_applies_only_bounded_qwen_residual(self):
        arbitration = candidate("final_consensus_residual", [.9])
        arbitration["prediction"]["residual_dispositions"] = [{
            "frame_index": 0, "criterion_id": "c", "action": "downgrade",
            "residual_step": .1,
        }]
        merged = merge_consensus_residual_judgment(self.candidates, arbitration, [0])
        frames = merged["prediction"]["frames"]
        self.assertAlmostEqual(frames[0]["criteria"][0]["probability_satisfied"], .6)
        self.assertAlmostEqual(frames[1]["criteria"][0]["probability_satisfied"], .5)
        self.assertFalse(merged["consensus_residual_merge"]["raw_small_model_rows_used_as_final_prediction"])

    def test_wrong_roles_are_rejected(self):
        wrong = [candidate("a", [.2]), candidate("b", [.4])]
        with self.assertRaises(ValueError):
            build_consensus_prior(wrong)


if __name__ == "__main__":
    unittest.main()
