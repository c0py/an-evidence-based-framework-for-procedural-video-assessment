"""Tests of executor boundaries, evidence identity and geometric semantics."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run as pilot
from jsonschema import ValidationError


class Contracts(unittest.TestCase):
    def setUp(self):
        self.item = {'case_id': 'case_01', 'criterion_id': 'trocar_layout', 'frames': 2,
                     'timestamps_s': [0., .04], 'criterion': {'tool_id': 'measure_trocar_geometry',
                                                            'roi_xyxy': [0., 0., 1., 1.],
                                                            'apex_reference_band_deg': [110., 130.]}}
        self.call = {'action': 'call_tool', 'tool_id': 'measure_trocar_geometry',
                     'input_id': 'case_01', 'criterion_id': 'trocar_layout',
                     'arguments': {'start_frame': 0, 'end_frame': 1, 'roi_xyxy': [0., 0., 1., 1.]},
                     'reason': 'Measure the requested image-plane geometry.'}

    def test_valid_call(self):
        pilot.validate_call(self.call, self.item)

    def test_wrong_tool_never_executes(self):
        self.call['tool_id'] = 'measure_hand_pose'
        with tempfile.TemporaryDirectory() as d, patch.object(pilot.subprocess, 'run') as process:
            with self.assertRaises(ValueError):
                pilot.execute(self.call, self.item, Path(d))
            process.assert_not_called()
            self.assertFalse((Path(d) / 'tool_execution').exists())

    def test_unregistered_paths_and_parameters_rejected(self):
        self.call['arguments']['source'] = '/tmp/arbitrary.mp4'
        with self.assertRaises(ValidationError):
            pilot.validate_call(self.call, self.item)

    def test_input_range_roi_cannot_change(self):
        for key, value in [('input_id', 'case_02'), ('criterion_id', 'hand_press')]:
            call = copy.deepcopy(self.call)
            call[key] = value
            with self.assertRaises(ValueError):
                pilot.validate_call(call, self.item)
        for key, value in [('start_frame', 1), ('end_frame', 4), ('roi_xyxy', [.1, 0., 1., 1.])]:
            call = copy.deepcopy(self.call)
            call['arguments'][key] = value
            with self.assertRaises(ValueError):
                pilot.validate_call(call, self.item)

    def test_triangle_angle_is_apex_not_half_angle(self):
        raw = [{'frame_index': 0, 'endpoints': [[0, 0], [-math.sqrt(3), 1], [math.sqrt(3), 1]],
                'navel': [0, -.1], 'left_angle_deg': 60., 'right_angle_deg': 60.},
               {'frame_index': 1, 'endpoints': [[0, 0]], 'navel': [0, -.1]}]
        measurements, evidence = pilot.geometric_observations(raw, self.item)
        self.assertAlmostEqual(measurements[0]['apex_angle_deg'], 120.)
        self.assertAlmostEqual(measurements[0]['side_length_ratio'], 1.)
        self.assertFalse(measurements[1]['valid'])
        self.assertEqual(evidence[0]['value']['valid_frames'], 1)
        self.assertEqual(evidence[4]['value']['triangle_vertices'], ['A', 'B', 'C'])
        self.assertEqual(evidence[4]['value']['samples'][0]['A_xy'], [0., 0.])
        self.assertEqual(evidence[4]['value']['samples'][0]['N_xy'], [0, -.1])
        self.assertNotEqual(evidence[4]['value']['samples'][0]['A_xy'], evidence[4]['value']['samples'][0]['N_xy'])

    def test_missing_geometry_returns_unknown_measurement(self):
        raw = [{'frame_index': 0, 'endpoints': [], 'navel': None}]
        measurements, evidence = pilot.geometric_observations(raw, self.item)
        self.assertFalse(measurements[0]['valid'])
        self.assertIsNone(evidence[1]['value'])

    def test_angle_band_inclusive_and_gaps_break_intervals(self):
        angles = [110., 130., None, 120., 131., 120., 109.]
        rows = [{'frame_index': i, 'timestamp_s': i * .1, 'valid': angle is not None,
                 'apex_angle_deg': angle} for i, angle in enumerate(angles)]
        result = pilot.angle_timeline(rows, [110., 130.])
        self.assertEqual(result['within_band_frames'], 4)
        self.assertEqual(result['outside_band_frames'], 2)
        self.assertEqual(result['missing_frames'], 1)
        self.assertAlmostEqual(result['within_band_fraction_of_valid'], 4/6)
        self.assertAlmostEqual(result['within_band_fraction_of_all'], 4/7)
        self.assertEqual([p['frames'] for p in result['within_band_contiguous_intervals']], [2, 1, 1])
        self.assertEqual([p['observed_span_s'] for p in result['within_band_contiguous_intervals']], [.1, 0., 0.])

    def test_angle_band_no_measurements_is_unknown(self):
        result = pilot.angle_timeline([{'frame_index': 0, 'timestamp_s': 0., 'valid': False}], [110., 130.])
        self.assertIsNone(result['within_band_fraction_of_valid'])
        self.assertEqual(result['within_band_contiguous_intervals'], [])

    def test_fabricated_evidence_rejected(self):
        judgment = {'criterion_id': 'trocar_layout', 'status': 'supported',
                    'supporting_evidence_ids': ['invented'], 'contradicting_evidence_ids': [],
                    'rationale': 'Example', 'criterion_checks': [], 'missing_or_limited_evidence': []}
        with self.assertRaises(ValueError):
            pilot.validate_judgment(judgment, self.item, ['real'])
        judgment['supporting_evidence_ids'] = ['real']
        pilot.validate_judgment(judgment, self.item, ['real'])

    def test_target_relative_path_removes_common_translation(self):
        raw = []
        for i, offset in enumerate([[10, 0], [0, 10], [-10, 0], [0, -10]]):
            center = [100 + i * 20, 100 + i * 30]
            raw.append({'frame_index': i, 'swab_detected': True, 'stoma_detected': True,
                        'reused_previous_swab': False,
                        'stoma_box': [center[0]-5, center[1]-5, center[0]+5, center[1]+5],
                        'tip': [center[0]+offset[0], center[1]+offset[1]]})
        points, arcs, span = pilot.relative_trajectory(raw, [0, .1, .2, .3])
        self.assertEqual([p['relative_xy_px'] for p in points], [[10, 0], [0, 10], [-10, 0], [0, -10]])
        self.assertAlmostEqual(span, 270.)
        self.assertAlmostEqual(arcs[0]['net_angular_displacement_deg'], 270.)

    def test_trajectory_gaps_and_reused_points_are_not_connected(self):
        raw = [{'frame_index': i, 'swab_detected': True, 'stoma_detected': True,
                'reused_previous_swab': i == 1, 'stoma_box': [-1, -1, 1, 1],
                'tip': p} for i, p in enumerate([[10, 0], [0, 10], [-10, 0]])]
        points, arcs, span = pilot.relative_trajectory(raw, [0, .1, .2])
        self.assertEqual(len(points), 2)
        self.assertEqual(len(arcs), 2)
        self.assertEqual([a['net_angular_displacement_deg'] for a in arcs], [0., 0.])

    def test_free_text_evidence_ids_are_validated(self):
        judgment = {'criterion_id': 'trocar_layout', 'status': 'supported',
                    'supporting_evidence_ids': ['real'], 'contradicting_evidence_ids': [],
                    'rationale': 'See case_01.measurement.99', 'criterion_checks': [],
                    'missing_or_limited_evidence': []}
        with self.assertRaises(ValueError):
            pilot.validate_judgment(judgment, self.item, ['real'])


if __name__ == '__main__':
    unittest.main()
