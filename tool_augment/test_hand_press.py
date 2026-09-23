import unittest

import numpy as np

from hand_press import Config, PoseState, match_tracks, pose_features


def features(tip=(500.0, 300.0), shape=True, pinch=0.7):
    return {"valid": True, "tip_xy": list(tip), "palm_xy": [450.0, 250.0],
            "palm_width_px": 100.0, "press_shape": shape,
            "pinch_palm_widths": pinch}


class PoseRulesTest(unittest.TestCase):
    def test_stationary_shape_needs_dwell(self):
        state = PoseState(Config())
        results = [state.update(i * 0.05, features()) for i in range(10)]
        self.assertFalse(results[1]["candidate"])
        self.assertFalse(results[4]["candidate"])
        self.assertTrue(results[-1]["candidate"])

    def test_downward_image_motion_alone_is_not_pressing(self):
        state = PoseState(Config())
        results = [state.update(i * 0.05, features(tip=(500, 20 + 20 * i))) for i in range(15)]
        self.assertFalse(any(r["candidate"] for r in results))
        self.assertEqual(results[-1]["state"], "MOVING / REPOSITIONING")

    def test_rest_without_required_shape_never_triggers(self):
        state = PoseState(Config())
        for i in range(20):
            self.assertFalse(state.update(i * 0.05, features(shape=False))["candidate"])

    def test_missing_hand_and_time_gap_reset_evidence(self):
        state = PoseState(Config())
        for i in range(12):
            result = state.update(i * 0.05, features())
        self.assertTrue(result["candidate"])
        self.assertFalse(state.update(0.6, {"valid": False})["candidate"])
        self.assertFalse(state.update(0.65, features())["candidate"])
        for i in range(14, 24):
            state.update(i * 0.05, features())
        self.assertFalse(state.update(2.0, features())["candidate"])

    def test_pinching_does_not_pass_shape_gate(self):
        cfg = Config()
        p = np.zeros((21, 3))
        p[5], p[6], p[7], p[8], p[17] = [.45, .35, 0], [.5, .4, 0], [.5, .46, 0], [.5, .5, 0], [.6, .35, 0]
        p[4] = [.7, .5, 0]
        self.assertTrue(pose_features(p, 960, 544, cfg)["press_shape"])
        p[4] = p[8] + [.01, 0, 0]
        self.assertFalse(pose_features(p, 960, 544, cfg)["press_shape"])
        p[4] = [.7, .5, 0]
        outside = Config(target_roi=(0.0, 0.0, .3, .3))
        self.assertFalse(pose_features(p, 960, 544, outside)["press_shape"])

    def test_swapped_detection_order_keeps_hand_identity(self):
        tracks = {1: {"last_seen": 0.0, "palm_xy": [100, 100]},
                  2: {"last_seen": 0.0, "palm_xy": [500, 100]}}
        observations = [{"palm_xy": [502, 100], "palm_width_px": 100},
                        {"palm_xy": [102, 100], "palm_width_px": 100}]
        self.assertEqual(match_tracks(observations, tracks, .05, Config()), [2, 1])
        self.assertEqual(match_tracks(observations, tracks, 1.0, Config()), [None, None])

    def test_edge_on_hand_uses_length_for_motion_scale(self):
        cfg = Config()
        p = np.zeros((21, 3))
        p[0], p[5], p[17], p[9] = [.5, .1, 0], [.49, .4, 0], [.51, .4, 0], [.5, .4, 0]
        p[6], p[7], p[8], p[4] = [.5, .45, 0], [.52, .47, 0], [.54, .5, 0], [.7, .5, 0]
        f = pose_features(p, 960, 544, cfg)
        self.assertTrue(f['valid'])
        self.assertGreater(f['palm_scale_px'], 5 * f['palm_width_px'])

    def test_invalid_points_are_unobserved(self):
        p = np.zeros((21, 3))
        self.assertFalse(pose_features(p, 960, 544, Config())['valid'])
        p[4, 0] = float('nan')
        self.assertFalse(pose_features(p, 960, 544, Config())['valid'])


if __name__ == "__main__":
    unittest.main()
