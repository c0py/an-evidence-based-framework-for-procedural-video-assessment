#!/usr/bin/env python3
"""Small, auditable constrained-call pilot. No training and no invented labels.

The local backend generates assistant JSON, which is schema-validated. An application
dispatcher validates and executes it, then sends an explicit tool-result
envelope back in the conversation. This is not native API function calling.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

HERE = Path(__file__).resolve().parent
TOOL_ROOT = HERE.parent
ROOT = TOOL_ROOT.parent
sys.path.insert(0, str(TOOL_ROOT / '.runtime'))
sys.path.insert(0, str(TOOL_ROOT))
import cv2
import numpy as np
from jsonschema import validate
from evidence_views import relative_trajectory, geometry_view, disinfection_view, hand_view, angle_timeline, angle_timeline_view

MODEL = 'qwen3-vl-32b-sop'
TOOLS = ['measure_trocar_geometry', 'measure_disinfection_path', 'measure_hand_pose']
SOP_PATH = Path(os.environ.get('SOP_CONFIG', str(HERE / 'sops.example.json')))
SYSTEM = '''Assess the visible procedure against the supplied natural-language
criterion. Use original images and, when available, specialist observations and
measurement diagrams. Tool outputs are observations, not compliance labels.
Judge the stated visible requirements: do not add requirements for clinical
efficacy, physical force, calibrated 3D measurements or unspecified thresholds.
Qualitative requirements can be assessed qualitatively; report numerical
departures from a nominal target without inventing acceptance tolerances.
Use supported for visible conformity; supported_with_deviations for a recognizable
intended action/configuration with concrete observed departures; violated for
a demonstrated incompatible action/configuration; and insufficient_evidence
when the relevant visible conditions cannot be determined. Qualified support
is not full compliance: retain actual deviations and do not invent numerical
tolerances. Do not assume an input is correct.
Use explicit numerical tolerances when provided; report both in-band and out-of-band
observations. Do not invent a required frame percentage or duration for clip-level
acceptance, select only favorable frames, or attribute fluctuations to camera motion
as an established cause without supporting measurements. Projection limits what
can be concluded about physical geometry, not what the measured numbers are.
In the vision-only baseline, judge what is visible; lack of tool output alone is not missing visual evidence.
Give a separate criterion_check for each relevant visible requirement, keeping
measurement limitations distinct from unfulfilled procedural requirements.
Do not invent measurements. Missing detections and rejected fits are not proof
that an action was absent. Before a tool executes, explain only why it is needed,
not what it will find. Cite actual IDs and include every ID mentioned in the
rationale in the supporting or contradicting ID arrays. Keep outputs concise,
in English, and in the requested JSON format.'''


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def progress(out, **state):
    path = out / 'PROGRESS.json'
    tmp = out / 'PROGRESS.json.tmp'
    tmp.write_text(json.dumps({'updated_utc': datetime.now(timezone.utc).isoformat(), **state}, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


CALL_SCHEMA = object_schema({
    'action': {'type': 'string', 'enum': ['call_tool']},
    'tool_id': {'type': 'string', 'enum': TOOLS},
    'input_id': {'type': 'string'},
    'criterion_id': {'type': 'string'},
    'arguments': object_schema({
        'start_frame': {'type': 'integer', 'minimum': 0},
        'end_frame': {'type': 'integer', 'minimum': 0},
        'roi_xyxy': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4},
    }),
    'reason': {'type': 'string'},
})
JUDGMENT_SCHEMA = object_schema({
    'criterion_id': {'type': 'string'},
    'status': {'type': 'string', 'enum': ['supported', 'supported_with_deviations', 'violated', 'insufficient_evidence']},
    'supporting_evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
    'contradicting_evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
    'rationale': {'type': 'string'},
    'criterion_checks': {'type': 'array', 'items': object_schema({
        'requirement': {'type': 'string'},
        'status': {'type': 'string', 'enum': ['supported', 'supported_with_deviations', 'violated', 'insufficient_evidence']},
        'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
        'finding': {'type': 'string'},
    })},
    'missing_or_limited_evidence': {'type': 'array', 'items': {'type': 'string'}},
})


def decode(source):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot decode source')
    frames, pts = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    cap.release()
    if not frames or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('Missing frames or nonmonotonic source PTS')
    times = [t - pts[0] for t in pts]
    return frames, pts, times


def prepare(out, case_manifest):
    if (out / 'PROTOCOL.json').exists():
        raise FileExistsError('Protocol already prepared; use run to execute the frozen manifest')
    cv2.setNumThreads(2)
    sop = read(SOP_PATH)
    manifest = []
    cases = read(case_manifest)
    if not isinstance(cases, list) or not cases:
        raise ValueError('Provide a nonempty local case manifest')
    seen = set()
    for entry in cases:
        case_id, criterion_id = entry['case_id'], entry['criterion_id']
        if not re.fullmatch(r'case_\d+', case_id) or case_id in seen:
            raise ValueError('Case IDs must be unique case_<digits> identifiers')
        seen.add(case_id)
        source = Path(entry['source']).expanduser().resolve()
        frames, pts, times = decode(source)
        case_dir = out / case_id
        case_dir.mkdir(parents=True, exist_ok=False)
        images = []
        for i in np.linspace(0, len(frames) - 1, 6, dtype=int).tolist():
            frame = frames[i].copy()
            h, w = frame.shape[:2]
            # Same fixed mask in all arms: suppress source instructional captions.
            frame[int(h * .86):] = 0
            if w > 640:
                frame = cv2.resize(frame, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
            path = case_dir / 'images' / f'frame_{i:04d}.jpg'
            path.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise RuntimeError('Image write failed')
            images.append({'evidence_id': f'{case_id}.frame.{i}', 'frame_index': i,
                           'timestamp_s': times[i], 'path': str(path.relative_to(out)), 'sha256': sha(path)})
        item = {'case_id': case_id, 'criterion_id': criterion_id, 'source': str(source),
                'source_sha256': sha(source), 'frames': len(frames), 'source_pts_s': pts,
                'timestamps_s': times, 'width': frames[0].shape[1], 'height': frames[0].shape[0],
                'criterion': sop['criteria'][criterion_id], 'images': images}
        save(case_dir / 'INPUT.json', item)
        manifest.append(item)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model': MODEL,
                'kind': 'assessment of user-supplied videos',
                'sop_provenance': sop['provenance'], 'arms': ['vision_sop', 'constrained_tool'],
                'image_policy': 'Same six original uniform frames in every call, bottom 14% masked, width <=640. Final tool judgment additionally sees neutral measurement diagrams (two for trocar, one for other cases). Tools process full raw clips; extra evidence access is part of the intervention.',
                'transport': 'Prompted assistant JSON with post-generation JSON-Schema validation, validated application dispatcher, explicit tool-result envelope in user role. Not native function-call API.',
                'tool_budget_per_case': 1, 'mllm_calls_per_case': 3,
                'temperature': 0.0, 'request_max_tokens': 384, 'judgment_max_tokens': 1000,
                'source_name_blinding': 'Only neutral case IDs enter model messages; no source filenames or pre-annotated outputs.',
                'ground_truth': 'No independent criterion labels; do not compute accuracy from tool rules or model agreement.',
                'code_sha256': sha(__file__), 'sops_sha256': sha(SOP_PATH),
                'reproduce_sha256': sha(TOOL_ROOT / 'reproduce.py'),
                'hand_press_sha256': sha(TOOL_ROOT / 'hand_press.py'),
                'evidence_views_sha256': sha(HERE / 'evidence_views.py'),
                'server_sha256': sha(ROOT / 'scripts/serve_qwen3_vl_openai.py'),
                'server_wrapper_sha256': sha(ROOT / 'scripts/serve_qwen3_vl_openai_control_char_recovery.py'),
                'cases': manifest}
    save(out / 'PROTOCOL.json', protocol)
    save(out / 'SCHEMAS.json', {'call': CALL_SCHEMA, 'judgment': JUDGMENT_SCHEMA})
    progress(out, status='prepared', completed_cases=0, total_cases=len(manifest))


def common_messages(out, item):
    # Model payload contains no source path, file name or previous outcome.
    spec = {'input_id': item['case_id'], 'criterion_id': item['criterion_id'],
            'criterion': item['criterion'], 'frames_in_full_clip': item['frames'],
            'sampled_frame_evidence': [{k: v for k, v in im.items() if k not in ('path', 'sha256')}
                                       for im in item['images']]}
    content = [{'type': 'text', 'text': json.dumps(spec, ensure_ascii=False)}]
    for im in item['images']:
        content.append({'type': 'text', 'text': f"{im['evidence_id']} at {im['timestamp_s']:.6f} s"})
        data = base64.b64encode((out / im['path']).read_bytes()).decode()
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + data}})
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]


def chat(base_url, messages, schema, cap, audit):
    format_message = {'role': 'user', 'content':
        'Return one compact valid JSON object, without Markdown, matching this schema: ' + json.dumps(schema) +
        '\nKeep reason/rationale at most 80 words. Use at most three criterion_checks with short findings, '
        'and at most three missing-evidence items of at most 20 words each. '
        'Do not repeat these instructions. End immediately after the closing JSON brace.'}
    payload = {'model': MODEL, 'messages': messages + [format_message], 'temperature': 0.0, 'max_tokens': cap}
    save(audit.with_suffix('.request.json'), payload)
    started = time.monotonic()
    opener = build_opener(ProxyHandler({}))
    req = Request(base_url.rstrip('/') + '/chat/completions',
                  data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with opener.open(req, timeout=900) as res:
            result = json.loads(res.read())
    except HTTPError as exc:
        save(audit.with_suffix('.error.json'), {'http_status': exc.code, 'body': exc.read().decode(errors='replace'),
                                               'elapsed_s': time.monotonic() - started})
        raise
    result['client_elapsed_s'] = time.monotonic() - started
    save(audit.with_suffix('.response.json'), result)
    choice = result['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('Truncated model output; retained without repair')
    parsed = json.loads(choice['message']['content'])
    validate(parsed, schema)
    return parsed, choice['message'], result


def validate_call(call, item):
    validate(call, CALL_SCHEMA)
    if call['input_id'] != item['case_id'] or call['criterion_id'] != item['criterion_id']:
        raise ValueError('Input or criterion substitution')
    if call['tool_id'] != item['criterion']['tool_id']:
        raise ValueError('Tool not permitted for this criterion')
    args = call['arguments']
    if args['start_frame'] != 0 or args['end_frame'] != item['frames'] - 1:
        raise ValueError('Pilot contract requires the full supplied clip')
    if args['roi_xyxy'] != item['criterion']['roi_xyxy']:
        raise ValueError('ROI does not match the frozen guide')


def quantiles(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return None
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite observation')
    return {'min': float(a.min()), 'median': float(np.median(a)), 'max': float(a.max())}


def add_evidence(evidence, item, kind, value, frame_indices, limitations):
    if not frame_indices:
        # Failure/count summaries remain admissible observations about tool coverage.
        support = []
    else:
        support = [int(i) for i in frame_indices]
    eid = f"{item['case_id']}.measurement.{len(evidence) + 1}"
    evidence.append({'evidence_id': eid, 'type': kind, 'value': value,
                     'source': {'input_id': item['case_id'], 'tool_id': item['criterion']['tool_id'],
                                'frame_count': len(support), 'first_frame': min(support) if support else None,
                                'last_frame': max(support) if support else None,
                                'first_timestamp_s': item['timestamps_s'][min(support)] if support else None,
                                'last_timestamp_s': item['timestamps_s'][max(support)] if support else None},
                     'limitations': limitations})


def geometric_observations(raw, item):
    measurements = []
    for row in raw:
        record = {'frame_index': row['frame_index'], 'timestamp_s': item['timestamps_s'][row['frame_index']],
                  'endpoints': row['endpoints'], 'navel': row['navel'], 'coordinate_frame': '2x resized image pixels',
                  'valid': False}
        if len(row['endpoints']) == 3 and row['navel'] is not None:
            p = np.asarray(row['endpoints'], dtype=float)
            apex_i = int(np.argmin(np.linalg.norm(p - row['navel'], axis=1)))
            b = p[apex_i]
            a, c = np.delete(p, apex_i, axis=0)
            u, v = a - b, c - b
            lengths = [float(np.linalg.norm(u)), float(np.linalg.norm(v))]
            if min(lengths) > 1e-6:
                dot = np.dot(u, v) / np.prod(lengths)
                angle = float(np.degrees(np.arccos(np.clip(dot, -1, 1))))
                record.update(valid=True, apex_angle_deg=angle,
                              side_length_ratio=max(lengths) / min(lengths),
                              half_angles_deg=[row['left_angle_deg'], row['right_angle_deg']],
                              apex_navel_distance_px=float(np.linalg.norm(b - row['navel'])),
                              apex_trocar_xy=b.tolist(),
                              base_trocars_xy=sorted([a.tolist(), c.tolist()], key=lambda point: point[0]))
        measurements.append(record)
    good = [r for r in measurements if r['valid']]
    ids = [r['frame_index'] for r in good]
    e = []
    add_evidence(e, item, 'geometry_measurement_coverage', {'valid_frames': len(good), 'total_frames': len(raw)},
                 list(range(len(raw))), ['Absent geometry is not a violation; detected navel is a heuristic anatomical reference.'])
    add_evidence(e, item, 'projected_apex_angle_degrees', quantiles([r['apex_angle_deg'] for r in good]), ids,
                 ['2D image-plane angle; three-dimensional insertion angle is not measured.'])
    add_evidence(e, item, 'projected_side_length_ratio', quantiles([r['side_length_ratio'] for r in good]), ids,
                 ['Ratio 1 denotes equal projected sides; image projection and endpoint localization affect the ratio.'])
    add_evidence(e, item, 'side_to_altitude_angles_degrees',
                 {'smaller': quantiles([min(r['half_angles_deg']) for r in good]),
                  'larger': quantiles([max(r['half_angles_deg']) for r in good])}, ids,
                 ['Angles are sorted per frame, not tracked anatomical left/right.'])
    samples = [good[i] for i in np.linspace(0, len(good) - 1, min(6, len(good)), dtype=int)] if good else []
    add_evidence(e, item, 'projected_landmark_samples',
                 {'point_identities': {'A': 'Trocar endpoint nearest the navel; triangle apex',
                                       'B': 'Other trocar endpoint at smaller image x; triangle vertex',
                                       'C': 'Other trocar endpoint at larger image x; triangle vertex',
                                       'N': 'Navel reference only; NOT a triangle vertex'},
                  'triangle_vertices': ['A', 'B', 'C'], 'angle_definition': 'Angle BAC, not angle BNC',
                  'samples': [{'frame_index': r['frame_index'], 'timestamp_s': r['timestamp_s'],
                               'A_xy': r['apex_trocar_xy'], 'B_xy': r['base_trocars_xy'][0],
                               'C_xy': r['base_trocars_xy'][1], 'N_xy': r['navel'],
                               'apex_angle_deg': r['apex_angle_deg'], 'side_length_ratio': r['side_length_ratio']}
                              for r in samples]}, [r['frame_index'] for r in samples],
                 ['Coordinates use 2x source pixels; samples selected uniformly among valid observations, not by target agreement.'])
    add_evidence(e, item, 'projected_apex_temporal_reference_band',
                 angle_timeline(measurements, item['criterion']['apex_reference_band_deg']),
                 [r['frame_index'] for r in measurements],
                 ['Inclusive complete-apex band from the supplied criterion; provenance is recorded in the SOP, separately from the legacy half-angle gate.',
                  'Frame fractions are descriptive; no minimum percentage or duration has been supplied for clip-level acceptance.',
                  'Intervals break at missing or out-of-band frames; observed span is last minus first timestamp, not continuous verified duration.',
                  'No smoothing, rectification or camera-cause attribution is applied.'])
    return measurements, e


def disinfection_observations(raw, item):
    measurements = []
    for r in raw:
        measurements.append({k: r[k] for k in ['frame_index', 'swab_detected', 'stoma_detected',
                                               'swab_box', 'stoma_box', 'tip', 'coverage_ratio',
                                               'online_circle', 'reused_previous_swab']})
        measurements[-1]['timestamp_s'] = item['timestamps_s'][r['frame_index']]
    valid = [r['frame_index'] for r in raw if r['swab_detected'] and r['stoma_detected']]
    circles = [r for r in raw if r['online_circle'] is not None]
    e = []
    add_evidence(e, item, 'detection_coverage',
                 {'total_frames': len(raw), 'swab_frames': sum(r['swab_detected'] for r in raw),
                  'target_frames': sum(r['stoma_detected'] for r in raw),
                  'jointly_detected_frames': len(valid),
                  'reused_swab_frames': sum(r['reused_previous_swab'] for r in raw)}, list(range(len(raw))),
                 ['Tracking gaps are unknown; target center can persist when the target is not freshly detected.'])
    add_evidence(e, item, 'image_plane_coverage_proxy',
                 {'final_fraction': raw[-1]['coverage_ratio'], 'target_radius_px': 200,
                  'image_width_px': item['width'], 'image_height_px': item['height'],
                  'footprint': 'swab bounding-box inscribed disk'}, list(range(len(raw))),
                 ['Describes footprint extent inside the displayed reference disk; not a binary quality label.'])
    add_evidence(e, item, 'online_circle_fit_support',
                 {'frames_with_online_circle_estimate': len(circles), 'total_frames': len(raw),
                  'fit_gate': {'minimum_points': 40, 'window_points': 120,
                               'minimum_angular_span_deg': 240, 'max_radial_std_over_radius': .20}},
                 [r['frame_index'] for r in circles],
                 ['An available estimate can persist across frames; this count is not a count of independent accepted fits.',
                  'Fit support is a tool validity result, not a final SOP verdict.',
                  'A rejected fit does not by itself prove that circular motion was absent.'])
    points, arcs, span = relative_trajectory(raw, item['timestamps_s'])
    by_frame = {p['frame_index']: p for p in points}
    for m in measurements:
        m['target_relative_observation'] = by_frame.get(m['frame_index'])
    samples = [points[i] for i in np.linspace(0, len(points) - 1, min(12, len(points)), dtype=int)] if points else []
    add_evidence(e, item, 'target_relative_swab_trajectory',
                 {'observed_angular_span_deg': span,
                  'distance_from_ostomy_center_px': quantiles([p['radius_px'] for p in points]),
                  'continuous_observation_segments': arcs, 'uniform_samples': samples,
                  'coordinate_convention': 'Current swab box center minus current ostomy box center; image x right, y down.'},
                 [p['frame_index'] for p in points],
                 ['Only fresh paired detections are used; gaps are not bridged.',
                  'Angular span summarizes observed positions, not the number of completed revolutions.'])
    return measurements, e


def hand_observations(raw, item):
    measurements, groups = [], {}
    for r in raw:
        for h in r['hands']:
            m = {k: h[k] for k in ['track_id', 'landmarks', 'index_pip_deg', 'pinch_palm_widths',
                                    'tip_in_target', 'tip_xy', 'speed', 'dwell_s']}
            m.update(frame_index=r['frame_index'], timestamp_s=item['timestamps_s'][r['frame_index']])
            # Derived rule support is explicitly identified, not an action class or verdict.
            m['development_pose_gate_support'] = bool(h['candidate'])
            measurements.append(m)
            groups.setdefault(h['track_id'], []).append(m)
    e = []
    for identity, hs in sorted(groups.items()):
        ids = [h['frame_index'] for h in hs]
        add_evidence(e, item, 'hand_landmark_geometry',
                     {'track_id': identity, 'observed_frames': len(hs),
                      'index_pip_angle_deg': quantiles([h['index_pip_deg'] for h in hs]),
                      'thumb_index_distance_palm_widths': quantiles([h['pinch_palm_widths'] for h in hs]),
                      'tip_in_manual_target_frames': sum(h['tip_in_target'] for h in hs),
                      'tip_speed_hand_scales_per_second': quantiles([h['speed'] for h in hs if h['speed'] is not None])}, ids,
                 ['Wrist-relative landmark geometry and a manually specified target ROI.'])
        intervals, active = [], None
        for h in hs:
            if h['development_pose_gate_support']:
                if active is None:
                    active = {'first_frame': h['frame_index'], 'start_s': h['timestamp_s'], 'frames': 0}
                active.update(last_frame=h['frame_index'], end_s=h['timestamp_s'], frames=active['frames'] + 1)
            elif active is not None:
                intervals.append(active)
                active = None
        if active is not None:
            intervals.append(active)
        add_evidence(e, item, 'development_pose_gate_temporal_support',
                     {'track_id': identity, 'intervals': intervals,
                      'gate': 'Index joint 65..165 degrees, tip in manual ROI, thumb-index separation >0.38 palm widths, speed <=1.2 hand scales/s; 0.25 s dwell with 0.15 s release hysteresis.'},
                     [h['frame_index'] for h in hs if h['development_pose_gate_support']],
                     ['Derived shape/stability support, not a compliance label.',
                      'Hysteresis may retain short departures from the instantaneous gate.'])
    return measurements, e


def execute(call, item, case_dir):
    validate_call(call, item)
    work = case_dir / 'tool_execution'
    work.mkdir(exist_ok=False)
    started = time.monotonic()
    module = {'measure_trocar_geometry': 'trocar', 'measure_disinfection_path': 'disinfection',
              'measure_hand_pose': 'hands'}[call['tool_id']]
    command = [sys.executable, str(TOOL_ROOT / 'reproduce.py'), module, '--source', item['source'],
               '--output', str(work / 'raw')]
    if module == 'trocar':
        command += ['--trocar-scale', '2']
    save(work / 'EXECUTION_REQUEST.json', {'validated_model_request': call, 'command': command,
                                         'cache_used': False, 'started_utc': datetime.now(timezone.utc).isoformat()})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4')
    with (work / 'execution.log').open('x') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        raw = rows(work / 'raw/frames.jsonl')
        if module == 'hands':
            pose_command = [sys.executable, str(TOOL_ROOT / 'hand_press.py'), '--source', item['source'],
                            '--landmarks', str(work / 'raw/frames.jsonl'), '--output', str(work / 'pose'),
                            '--target-roi', *map(str, call['arguments']['roi_xyxy'])]
            save(work / 'POSE_EXECUTION.json', {'command': pose_command, 'cache': 'fresh landmarks from this tool call'})
            subprocess.run(pose_command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            raw = rows(work / 'pose/frames.jsonl')
    if len(raw) != item['frames'] or sha(item['source']) != item['source_sha256']:
        raise ValueError('Source identity or decoded coverage mismatch')
    mapper = {'trocar': geometric_observations, 'disinfection': disinfection_observations, 'hands': hand_observations}[module]
    measurements, evidence = mapper(raw, item)
    save(work / 'MEASUREMENTS.json', {'source_sha256': item['source_sha256'], 'measurements': measurements})
    image_path = work / 'evidence.jpg'
    view = {'trocar': geometry_view, 'disinfection': disinfection_view, 'hands': hand_view}[module]
    image_meta = view(raw, item, image_path)
    artifacts = []
    if image_meta:
        artifacts.append({'evidence_id': f"{item['case_id']}.visual.1", **image_meta,
                          'path': str(image_path.relative_to(case_dir.parent)), 'sha256': sha(image_path),
                          'derived_from_evidence_ids': [e['evidence_id'] for e in evidence]})
    if module == 'trocar':
        timeline_path = work / 'angle_timeline.jpg'
        timeline_meta = angle_timeline_view(measurements, item['criterion']['apex_reference_band_deg'], timeline_path)
        artifacts.append({'evidence_id': f"{item['case_id']}.visual.2", **timeline_meta,
                          'path': str(timeline_path.relative_to(case_dir.parent)), 'sha256': sha(timeline_path),
                          'derived_from_evidence_ids': [evidence[-1]['evidence_id']]})
    result = {'input_id': item['case_id'], 'tool_id': call['tool_id'], 'criterion_id': item['criterion_id'],
              'status': 'observations_returned', 'cache_used': False,
              'elapsed_s': time.monotonic() - started, 'evidence': evidence, 'artifacts': artifacts,
              'full_measurements_sha256': sha(work / 'MEASUREMENTS.json')}
    save(work / 'TOOL_RESULT.json', result)
    return result


def validate_judgment(judgment, item, allowed_ids):
    validate(judgment, JUDGMENT_SCHEMA)
    if judgment['criterion_id'] != item['criterion_id']:
        raise ValueError('Wrong criterion judgment')
    cited = judgment['supporting_evidence_ids'] + judgment['contradicting_evidence_ids']
    cited += [eid for check in judgment['criterion_checks'] for eid in check['evidence_ids']]
    cited += re.findall(r'case_\d+\.(?:measurement|frame|visual)\.\d+', judgment['rationale'])
    if not set(cited) <= set(allowed_ids):
        raise ValueError('Fabricated evidence ID')
    if judgment['status'] != 'insufficient_evidence' and not cited:
        raise ValueError('Definitive verdict without any cited evidence')


def run(out, base_url):
    protocol = read(out / 'PROTOCOL.json')
    for path, key in [(Path(__file__), 'code_sha256'), (SOP_PATH, 'sops_sha256'),
                      (TOOL_ROOT / 'reproduce.py', 'reproduce_sha256'), (TOOL_ROOT / 'hand_press.py', 'hand_press_sha256'),
                      (HERE / 'evidence_views.py', 'evidence_views_sha256')]:
        if sha(path) != protocol[key]:
            raise ValueError('Frozen implementation changed: ' + key)
    started = time.monotonic()
    results, failures = [], []
    for item in protocol['cases']:
        case_id = item['case_id']
        case_dir = out / case_id
        (case_dir / 'calls').mkdir(exist_ok=False)
        progress(out, status='running', current_case=case_id, stage='baseline',
                 completed_cases=len(results), total_cases=len(protocol['cases']))
        try:
            if sha(item['source']) != item['source_sha256']:
                raise ValueError('Source changed since preparation')
            for im in item['images']:
                if sha(out / im['path']) != im['sha256']:
                    raise ValueError('Image changed since preparation')
            common = common_messages(out, item)
            initial_ids = [im['evidence_id'] for im in item['images']]
            baseline_messages = common + [{'role': 'user', 'content':
                'Baseline arm: no tool calls or tool results are available. Assess from the supplied images and criterion. '
                'Do not treat developer settings as observed measurements. Return the judgment JSON.'}]
            baseline, _, baseline_response = chat(base_url, baseline_messages, JUDGMENT_SCHEMA, 1000,
                                                  case_dir / 'calls/01_baseline')
            validate_judgment(baseline, item, initial_ids)
            save(case_dir / 'BASELINE.json', baseline)
            guide = {'available_tools': TOOLS, 'required_tool': item['criterion']['tool_id'],
                     'input_id': case_id, 'criterion_id': item['criterion_id'],
                     'required_arguments': {'start_frame': 0, 'end_frame': item['frames'] - 1,
                                            'roi_xyxy': item['criterion']['roi_xyxy']},
                     'budget': 'Exactly one registered tool invocation; tools internally chain detection and measurements.',
                     'instruction': 'Follow the criterion-to-tool guide; generate a call_tool request now, not a quality judgment. Explain relevance only; do not claim measurements before execution. No paths, shell commands, thresholds or extra tools may be supplied.'}
            call_messages = common + [{'role': 'user', 'content': json.dumps(guide)}]
            progress(out, status='running', current_case=case_id, stage='model_tool_request',
                     completed_cases=len(results), total_cases=len(protocol['cases']))
            call, assistant_message, call_response = chat(base_url, call_messages, CALL_SCHEMA, 384,
                                                          case_dir / 'calls/02_tool_request')
            validate_call(call, item)
            save(case_dir / 'MODEL_TOOL_REQUEST.json', call)
            progress(out, status='running', current_case=case_id, stage='tool_execution',
                     completed_cases=len(results), total_cases=len(protocol['cases']))
            tool_result = execute(call, item, case_dir)
            result_content = [{'type': 'text', 'text':
                'The execution system ran your validated request. TOOL_RESULT:\n' + json.dumps(tool_result) +
                '\nUsing this returned evidence and the original images, produce the criterion judgment JSON. '
                'Inspect the measurement diagram as well as the numbers. Reference IDs for tool-based claims. '
                'Assess only the specified visible requirements. Do not request additional tools.'}]
            for artifact in tool_result['artifacts']:
                path = out / artifact['path']
                if sha(path) != artifact['sha256']:
                    raise ValueError('Changed tool image')
                result_content += [{'type': 'text', 'text': artifact['evidence_id'] + ': ' + artifact['description']},
                                   {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()}}]
            result_messages = call_messages + [assistant_message, {'role': 'user', 'content': result_content}]
            progress(out, status='running', current_case=case_id, stage='evidence_grounded_judgment',
                     completed_cases=len(results), total_cases=len(protocol['cases']))
            judgment, _, judgment_response = chat(base_url, result_messages, JUDGMENT_SCHEMA, 1000,
                                                  case_dir / 'calls/03_tool_judgment')
            allowed = initial_ids + [e['evidence_id'] for e in tool_result['evidence']] + [a['evidence_id'] for a in tool_result['artifacts']]
            validate_judgment(judgment, item, allowed)
            save(case_dir / 'TOOL_JUDGMENT.json', judgment)
            response_items = [baseline_response, call_response, judgment_response]
            row = {'case_id': case_id, 'criterion_id': item['criterion_id'],
                   'baseline': baseline, 'constrained_tool': judgment, 'tool_id': call['tool_id'],
                   'tool_elapsed_s': tool_result['elapsed_s'],
                   'mllm_elapsed_s': sum(r['client_elapsed_s'] for r in response_items),
                   'prompt_tokens': sum(r['usage']['prompt_tokens'] for r in response_items),
                   'completion_tokens': sum(r['usage']['completion_tokens'] for r in response_items),
                   'request_valid': True, 'evidence_references_valid': True,
                   'rationale_citations_listed': set(re.findall(r'case_\d+\.(?:measurement|frame|visual)\.\d+', judgment['rationale'])) <= set(judgment['supporting_evidence_ids'] + judgment['contradicting_evidence_ids']),
                   'measurement_cited': any('.measurement.' in s for s in judgment['supporting_evidence_ids'] + judgment['contradicting_evidence_ids'])}
            save(case_dir / 'RESULT.json', row)
            results.append(row)
        except Exception as exc:
            failure = {'case_id': case_id, 'error_type': type(exc).__name__, 'error': str(exc)}
            save(case_dir / 'FAILURE.json', failure)
            failures.append(failure)
    summary = {'status': 'complete' if not failures else 'completed_with_failures',
               'cases_completed': len(results), 'cases_failed': len(failures), 'results': results,
               'failures': failures, 'elapsed_s': time.monotonic() - started,
               'ground_truth_accuracy': None,
               'limitations': ['Independent quality labels are not supplied by this runner.',
                              'Two-round tool arm vs one-round baseline; call budget is not matched.',
                              'Tools see full clips; tool judgment adds measurements and a neutral diagram to the same six raw frames.',
                              'Citation validation checks ID existence, not semantic correctness.',
                              'Structured SOP and numerical implementation parameters await expert review.']}
    save(out / 'SUMMARY.json', summary)
    progress(out, status=summary['status'], completed_cases=len(results), total_cases=len(protocol['cases']),
             failed_cases=len(failures), elapsed_s=summary['elapsed_s'])
    lines = ['# Constrained tool invocation pilot', '',
             'Real model requests → validated fresh tool execution → returned observations → model judgment.', '',
             '| Case | Criterion | Vision + SOP | With tool | Tool seconds | MLLM seconds |',
             '|---|---|---|---|---:|---:|']
    for r in results:
        lines.append(f"| {r['case_id']} | {r['criterion_id']} | {r['baseline']['status']} | {r['constrained_tool']['status']} | {r['tool_elapsed_s']:.1f} | {r['mllm_elapsed_s']:.1f} |")
    lines += ['', 'No accuracy estimate: these are development cases without independent criterion labels.', '']
    for r in results:
        lines += [f"## {r['case_id']}", '', '**Baseline:** ' + r['baseline']['rationale'], '',
                  '**Tool-supported:** ' + r['constrained_tool']['rationale'], '',
                  '**Limitations:** ' + '; '.join(r['constrained_tool']['missing_or_limited_evidence']), '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status': summary['status'], 'completed': len(results), 'failed': failures}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='Local JSON list of case_id, criterion_id and source')
    parser.add_argument('--base-url', default='http://127.0.0.1:18937/v1')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        if args.cases is None:
            parser.error('--cases is required for prepare')
        prepare(out, args.cases)
    else:
        run(out, args.base_url)
