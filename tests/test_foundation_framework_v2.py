import unittest

from cvs_assessment.foundation_framework_v2 import (
    EvidencePluginSlot,
    FrozenFoundationFrameworkV2,
    ProceduralTaskPackageV2,
)


class FrozenFoundationFrameworkV2Tests(unittest.TestCase):
    def test_two_domains_share_identical_core(self):
        core = FrozenFoundationFrameworkV2()
        sages = ProceduralTaskPackageV2(
            task_id="sages_cvs_2024", skill_id="sages_cvs_frame_skill",
            criterion_ids=("two_structures", "cystic_plate", "hepatocystic_triangle"),
            plugin_slots=(EvidencePluginSlot("anatomy", "visual"), EvidencePluginSlot("timeline", "temporal")),
            sampled_frame_count=18,
        )
        industrial = ProceduralTaskPackageV2(
            task_id="industreal_psr_assembly", skill_id="industreal_psr_assembly_skill",
            criterion_ids=("front_chassis", "front_wheel", "rear_wheel"),
            plugin_slots=(EvidencePluginSlot("assembly_state", "visual"), EvidencePluginSlot("state_track", "temporal")),
            sampled_frame_count=12,
        )
        left, right = core.manifest(sages), core.manifest(industrial)
        self.assertEqual(left["shared_core"], right["shared_core"])
        self.assertNotEqual(left["task_package"], right["task_package"])
        self.assertFalse(left["shared_core"]["foundation_model_finetuned"])
        self.assertFalse(left["shared_core"]["raw_small_model_rows_used_as_final_prediction"])

    def test_prediction_emitting_plugin_is_rejected(self):
        package = ProceduralTaskPackageV2(
            task_id="task", skill_id="skill", criterion_ids=("criterion",),
            plugin_slots=(EvidencePluginSlot("bad", "visual", may_emit_final_task_prediction=True),),
            sampled_frame_count=1,
        )
        with self.assertRaises(ValueError):
            FrozenFoundationFrameworkV2().manifest(package)

    def test_finetuned_foundation_is_rejected(self):
        with self.assertRaises(ValueError):
            FrozenFoundationFrameworkV2(foundation_model_finetuned=True).validate()


if __name__ == "__main__":
    unittest.main()

