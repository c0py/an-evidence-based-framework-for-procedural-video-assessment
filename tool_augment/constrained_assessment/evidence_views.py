"""Source-linked numeric geometry and neutral measurement diagrams.

No compliance labels or expected outcomes enter these artifacts.
"""
import math
from pathlib import Path

import cv2
import numpy as np


def angle_timeline(measurements, band):
    """Describe a fixed reference band without inventing a clip-level pass rule."""
    low, high = band
    if not (0 <= low <= high <= 180):
        raise ValueError('Invalid apex reference band')
    series, intervals = [], []
    for r in measurements:
        angle = r['apex_angle_deg'] if r['valid'] else None
        inside = low <= angle <= high if angle is not None else None
        p = {'frame_index': r['frame_index'], 'timestamp_s': r['timestamp_s'],
             'apex_angle_deg': angle, 'within_reference_band': inside}
        series.append(p)
        if inside:
            if (not intervals or r['frame_index'] != intervals[-1]['last_frame'] + 1):
                intervals.append({'first_frame': r['frame_index'], 'last_frame': r['frame_index'],
                                  'start_s': r['timestamp_s'], 'end_s': r['timestamp_s'], 'frames': 0})
            interval = intervals[-1]
            interval.update(last_frame=r['frame_index'], end_s=r['timestamp_s'], frames=interval['frames'] + 1)
            interval['observed_span_s'] = interval['end_s'] - interval['start_s']
    valid = sum(p['apex_angle_deg'] is not None for p in series)
    inside = sum(p['within_reference_band'] is True for p in series)
    return {'reference_band_deg_inclusive': list(band), 'total_frames': len(series),
            'valid_frames': valid, 'missing_frames': len(series) - valid,
            'within_band_frames': inside, 'outside_band_frames': valid - inside,
            'within_band_fraction_of_valid': inside / valid if valid else None,
            'within_band_fraction_of_all': inside / len(series) if series else None,
            'within_band_contiguous_intervals': intervals, 'series': series}


def angle_timeline_view(measurements, band, destination):
    timeline = angle_timeline(measurements, band)
    canvas = np.full((410, 800, 3), 25, np.uint8)
    text(canvas, 'Projected complete apex angle BAC over time', (20, 28), .65)
    low = min([100.] + [p['apex_angle_deg'] - 5 for p in timeline['series'] if p['apex_angle_deg'] is not None])
    high = max([150.] + [p['apex_angle_deg'] + 5 for p in timeline['series'] if p['apex_angle_deg'] is not None])
    end = max([p['timestamp_s'] for p in timeline['series']] + [.001])
    def xy(t, angle):
        return int(65 + 700 * t / end), int(330 - 260 * (angle - low) / (high - low))
    cv2.rectangle(canvas, xy(0, band[1]), xy(end, band[0]), (65, 60, 40), -1)
    for value in [low, *band, high]:
        y = xy(0, value)[1]
        cv2.line(canvas, (65, y), (765, y), (100, 100, 100), 1)
        text(canvas, f'{value:.0f}', (20, y + 5), .45)
    previous = None
    for p in timeline['series']:
        if p['apex_angle_deg'] is None:
            cv2.drawMarker(canvas, (xy(p['timestamp_s'], low)[0], 344), (130, 130, 240), cv2.MARKER_TILTED_CROSS, 5, 1)
            previous = None
            continue
        point = xy(p['timestamp_s'], p['apex_angle_deg'])
        if previous and p['frame_index'] == previous[0] + 1:
            cv2.line(canvas, previous[1], point, (230, 210, 80), 1, cv2.LINE_AA)
        cv2.circle(canvas, point, 3, (230, 210, 80), -1, cv2.LINE_AA)
        previous = p['frame_index'], point
    for t in np.linspace(0, end, 5):
        text(canvas, f'{t:.2f}s', (xy(t, low)[0] - 12, 367), .4)
    text(canvas, f"Reference: {band[0]:g}-{band[1]:g} deg | Within: {timeline['within_band_frames']}/{timeline['valid_frames']} valid frames", (65, 55), .5)
    text(canvas, 'Raw measurements; missing frames marked x; no smoothing or gap interpolation.', (20, 397), .46)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError('Angle timeline image write failed')
    return {'source_frame_indices': [r['frame_index'] for r in measurements],
            'description': 'Full raw projected apex time series with the supplied criterion reference band; missing frames retained.'}


def relative_trajectory(raw, timestamps):
    points = []
    for row in raw:
        if not (row['swab_detected'] and row['stoma_detected']) or row['reused_previous_swab']:
            continue
        box = row['stoma_box']
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        offset = np.asarray(row['tip']) - center
        points.append({'frame_index': row['frame_index'], 'timestamp_s': timestamps[row['frame_index']],
                       'ostomy_center_xy': center.tolist(), 'swab_center_xy': row['tip'],
                       'relative_xy_px': offset.tolist(), 'radius_px': float(np.linalg.norm(offset)),
                       'angle_deg': float(np.degrees(np.arctan2(offset[1], offset[0])))})
    segments = []
    for p in points:
        if not segments or p['frame_index'] != segments[-1][-1]['frame_index'] + 1:
            segments.append([])
        segments[-1].append(p)
    arcs = []
    for segment in segments:
        angles = np.unwrap(np.radians([p['angle_deg'] for p in segment]))
        arcs.append({'first_frame': segment[0]['frame_index'], 'last_frame': segment[-1]['frame_index'],
                     'observations': len(segment),
                     'net_angular_displacement_deg': float(np.degrees(angles[-1] - angles[0])),
                     'angular_range_deg': float(np.degrees(np.ptp(angles)))})
    angular_span = None
    if points:
        angles = np.sort(np.mod([p['angle_deg'] for p in points], 360))
        angular_span = float(360 - np.diff(np.r_[angles, angles[0] + 360]).max())
    return points, arcs, angular_span


def text(image, value, xy, scale=.5, color=(230, 230, 230)):
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def source_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    # Decode in order to keep VFR indices tied to the cached detections.
    frame = None
    for _ in range(index + 1):
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Source frame unavailable')
    cap.release()
    frame[int(frame.shape[0] * .86):] = 0
    return frame


def geometry_view(raw, item, destination):
    valid = [r for r in raw if len(r['endpoints']) == 3 and r['navel'] is not None]
    if not valid:
        return None
    # Fixed midpoint selection, not selection by closeness to the desired angle.
    r = min(valid, key=lambda row: abs(row['frame_index'] - (len(raw) - 1) / 2))
    frame = source_frame(item['source'], r['frame_index'])
    p = np.asarray(r['endpoints'], float) / 2
    navel = np.asarray(r['navel'], float) / 2
    apex = int(np.argmin(np.linalg.norm(p - navel, axis=1)))
    others = [i for i in range(3) if i != apex]
    others.sort(key=lambda i: p[i, 0])
    cv2.polylines(frame, [p.astype(np.int32)], True, (220, 230, 80), 2, cv2.LINE_AA)
    for name, index in zip(['A', 'B', 'C'], [apex, *others]):
        xy = tuple(p[index].astype(int))
        cv2.circle(frame, xy, 5, (80, 210, 255), -1, cv2.LINE_AA)
        text(frame, name, (xy[0] + 8, xy[1] + 15), .6, (30, 30, 30))
    xy = tuple(navel.astype(int))
    cv2.circle(frame, xy, 5, (255, 180, 60), -1, cv2.LINE_AA)
    text(frame, 'Navel', (xy[0] + 8, xy[1] - 9), .5, (30, 30, 30))
    canvas = np.full((frame.shape[0] + 64, frame.shape[1], 3), 25, np.uint8)
    canvas[64:] = frame
    text(canvas, 'Measured projected landmarks (not a rectified anatomical plane)', (12, 24), .47)
    text(canvas, f"Frame {r['frame_index']} | A: trocar nearest navel | B/C: lateral trocars", (12, 48), .45)
    cv2.imwrite(str(destination), canvas)
    return {'source_frame_indices': [r['frame_index']], 'description': 'Projected detector landmarks and anatomical reference on the original frame.'}


def disinfection_view(raw, item, destination):
    pts, arcs, angular_span = relative_trajectory(raw, item['timestamps_s'])
    canvas = np.full((480, 960, 3), 25, np.uint8)
    text(canvas, 'Target-relative swab trajectory', (16, 27), .63)
    text(canvas, 'Geometric footprint accumulation', (495, 27), .61)
    center = np.array([240, 258])
    extent = max([240.] + [p['radius_px'] for p in pts])
    scale = 190 / extent
    cv2.circle(canvas, tuple(center), int(200 * scale), (130, 130, 130), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, tuple(center), (255, 180, 60), cv2.MARKER_CROSS, 14, 2)
    text(canvas, 'Ostomy center', tuple(center + [10, 0]), .4)
    previous = None
    for p in pts:
        xy = tuple((center + np.array(p['relative_xy_px']) * scale).astype(int))
        color = (240, int(90 + 150 * p['frame_index'] / max(len(raw) - 1, 1)), 90)
        if previous and p['frame_index'] == previous[0] + 1:
            cv2.line(canvas, previous[1], xy, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, xy, 2, color, -1, cv2.LINE_AA)
        previous = p['frame_index'], xy
    text(canvas, f"Fresh paired observations: {len(pts)}/{len(raw)}", (16, 447), .46)
    text(canvas, 'Gaps are not bridged; gray guide radius: 200 px', (16, 470), .44)
    frame = source_frame(item['source'], len(raw) - 1)
    coverage = np.zeros(frame.shape[:2], np.uint8)
    target = None
    last_swab = None
    for r in raw:
        if r['stoma_detected']:
            b = r['stoma_box']
            target = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
        if r['swab_detected']:
            x0, y0, x1, y1 = r['swab_box']
            last_swab = ((int((x0 + x1) / 2), int((y0 + y1) / 2)), int(min(x1 - x0, y1 - y0) / 2))
        if last_swab:
            cv2.circle(coverage, last_swab[0], last_swab[1], 255, -1)
    target_mask = np.zeros_like(coverage)
    if target:
        cv2.circle(target_mask, target, 200, 255, -1)
    covered = (coverage > 0) & (target_mask > 0)
    ratio = float(covered.sum() / max((target_mask > 0).sum(), 1))
    if raw[-1]['coverage_ratio'] is not None and abs(ratio - raw[-1]['coverage_ratio']) > 1e-6:
        raise ValueError('Coverage diagram does not match the tool measurement')
    overlay = frame.copy()
    overlay[covered] = (40, 210, 120)
    frame = cv2.addWeighted(frame, .62, overlay, .38, 0)
    if target:
        cv2.circle(frame, target, 200, (220, 220, 80), 2, cv2.LINE_AA)
    # Suppress original captions even after drawing the measurement overlay.
    frame[int(frame.shape[0] * .86):] = 0
    frame = cv2.resize(frame, (460, round(frame.shape[0] * 460 / frame.shape[1])))
    canvas[75:75 + frame.shape[0], 490:950] = frame
    text(canvas, f"Reference-disk coverage: {100 * ratio:.2f}%", (495, 395), .58)
    text(canvas, 'Coverage extent is descriptive, not a pass/fail label.', (495, 425), .43)
    cv2.imwrite(str(destination), canvas)
    return {'source_frame_indices': list(range(len(raw))),
            'description': 'Target-relative fresh-detection trajectory without gap interpolation, plus image-space accumulated coverage.',
            'rendered_coverage_fraction': ratio}


def hand_view(raw, item, destination):
    # Use the same midpoint-nearest visible-hand rule regardless of pose state.
    valid = [r for r in raw if r['hands']]
    if not valid:
        return None
    r = min(valid, key=lambda row: abs(row['frame_index'] - (len(raw) - 1) / 2))
    frame = source_frame(item['source'], r['frame_index'])
    h, w = frame.shape[:2]
    roi = item['criterion']['roi_xyxy']
    cv2.rectangle(frame, (int(roi[0] * w), int(roi[1] * h)),
                  (int(roi[2] * w), int(roi[3] * h)), (170, 170, 170), 1)
    for hand in r['hands']:
        points = (np.asarray(hand['landmarks'])[:, :2] * [w, h]).astype(int)
        for chain in [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12],
                      [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [0, 17]]:
            cv2.polylines(frame, [points[chain]], False, (220, 210, 70), 2, cv2.LINE_AA)
        xy = tuple(np.clip(points[0] + [5, 20], [5, 20], [w - 100, h - 20]))
        text(frame, f"Hand {hand['track_id']}", xy, .6, (20, 20, 20))
    cv2.imwrite(str(destination), cv2.resize(frame, (640, round(h * 640 / w))))
    return {'source_frame_indices': [r['frame_index']], 'description': 'Measured hand landmarks and manual target ROI; no gesture or quality labels.'}
