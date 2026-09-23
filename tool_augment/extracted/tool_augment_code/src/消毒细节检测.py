import os
import cv2
import csv
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# PyTorch 2.6 safe load allowlist for Ultralytics checkpoints
try:
    from torch.serialization import add_safe_globals  # PyTorch >= 2.6
    from ultralytics.nn.tasks import DetectionModel
    add_safe_globals([DetectionModel])  # allowlist DetectionModel when weights_only=True
except Exception:
    # Older torch versions or import failures: ignore, default loader will handle
    pass

# Patch torch.load to handle missing DFLoss class
import torch
_torch_load = torch.load

def _patched_load(*args, **kwargs):
    # Force weights_only=False to avoid safe loading issues
    kwargs['weights_only'] = False
    try:
        return _torch_load(*args, **kwargs)
    except AttributeError as e:
        if "DFLoss" in str(e):
            # Create a dummy DFLoss class if missing
            import ultralytics.utils.loss as loss_module
            if not hasattr(loss_module, 'DFLoss'):
                class DFLoss:
                    def __init__(self, *args, **kwargs):
                        pass
                loss_module.DFLoss = DFLoss
            return _torch_load(*args, **kwargs)
        else:
            raise

torch.load = _patched_load



def parse():
    p = argparse.ArgumentParser(description="Minimal swab contact trajectory (based on detect_swab_min style)")
    p.add_argument('--model', type=str, default="棉签造口.pt", help='Path to disinfection.pt')
    p.add_argument('--source', type=str, default="错误消毒.mp4", help='Video path')
    p.add_argument('--conf', type=float, default=0.1, help='YOLO confidence threshold')
    p.add_argument('--device', type=str, default='', help="CUDA device like '0'; empty for auto")
    p.add_argument('--swab-id', type=int, default=0, help='Class id for swab')
    p.add_argument('--stoma-id', type=int, default=1, help='Class id for stoma (造口)')
    p.add_argument('--show', action='store_true', help='Show window')
    p.add_argument('--demo', action='store_true', help='Window demo only (no video write)')
    p.add_argument('--no_show', action='store_true', help='Do not show window (override default show)')
    p.add_argument('--no_circle',default=True, action='store_true', help='Do not fit/draw approximate circle')
    p.add_argument('--target_r', type=int, default=200, help='Target disinfection radius (px) centered at stoma')
    p.add_argument('--brush', type=int, default=8, help='Swab brush thickness (px) for coverage accumulation')
    p.add_argument('--cov_th', type=float, default=0.95, help='Coverage threshold to consider fully disinfected')
    p.add_argument('--lock_stoma', action='store_true', help='Lock target center to first stoma detection')
    p.add_argument('--out_dir', type=str, default='runs/swab_traj_min', help='Output directory')
    return p.parse_args()


def skin_mask(bgr_img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def endpoint_skin_score(frame_bgr: np.ndarray, pt, radius: int = 6) -> float:
    H, W = frame_bgr.shape[:2]
    x0, y0 = int(pt[0]), int(pt[1])
    r = max(2, radius)
    x1, y1 = max(0, x0 - r), max(0, y0 - r)
    x2, y2 = min(W, x0 + r + 1), min(H, y0 + r + 1)
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    m = skin_mask(patch)
    yy, xx = np.ogrid[:m.shape[0], :m.shape[1]]
    cy, cx = y0 - y1, x0 - x1
    circ = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if not np.any(circ):
        return 0.0
    return float(np.count_nonzero(m[circ])) / float(np.count_nonzero(circ))


def detect_longest_line(roi_bgr: np.ndarray):
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g, 80, 160)
    h, w = roi_bgr.shape[:2]
    min_len = max(30, min(h, w) // 6)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=min_len, maxLineGap=20)
    if lines is None:
        return None
    best, best_len = None, 0.0
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L > best_len:
            best_len = L
            best = (int(x1), int(y1), int(x2), int(y2))
    return best





def fit_circle_least_squares(points):
    """Fit circle (cx, cy, r) to 2D points using linear least squares. Returns (cx, cy, r) or None."""
    if points is None or len(points) < 3:
        return None
    pts = np.array(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c0 = theta
        r = float(np.sqrt(cx * cx + cy * cy + c0))
        return float(cx), float(cy), r
    except Exception:
        return None



def main():
    a = parse()
    # device auto-fallback: use CPU if CUDA not available
    try:
        import torch
        if (not a.device) or (a.device.lower() == 'auto'):
            a.device = '0' if torch.cuda.is_available() else 'cpu'
        elif a.device != 'cpu' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU (use --device cpu to silence).")
            a.device = 'cpu'
    except Exception:
        pass


    model = YOLO(a.model)
    # cache class names for labels if available
    try:
        CLASS_NAMES = model.names if hasattr(model, 'names') else (model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else None)
    except Exception:
        CLASS_NAMES = None


    src = a.source
    if not Path(src).exists():
        raise FileNotFoundError(f'Video not found: {src}')

    # online circle estimation (continuous update)
    WINDOW = 120              # sliding window size for fitting
    MIN_PTS = 40              # start fitting after MIN_PTS points
    MIN_SPAN_DEG = 240.0      # minimal angular coverage to trust the fit
    MAX_STD_RATIO = 0.20      # std(radius)/mean(radius) threshold
    ALPHA = 0.2               # exponential smoothing factor (0..1)
    circle_est = None         # current smoothed estimate (cx, cy, r)
    first_snapshot_saved = False


    os.makedirs(a.out_dir, exist_ok=True)
    stem = Path(src).stem

    out_video = str(Path(a.out_dir) / f'{stem}_traj.mp4')
    out_csv = str(Path(a.out_dir) / f'{stem}_traj.csv')
    out_png = str(Path(a.out_dir) / f'{stem}_traj.png')

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {src}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = None if a.demo else cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    # fading trail canvas (for tail effect)
    trail = None
    TRAIL_DECAY = 0.90   # 0..1, smaller -> faster fade
    TRAIL_THICK = 3

    # coverage accumulation mask and target center/mask (after W,H known)
    coverage = np.zeros((H, W), dtype=np.uint8)
    target_center = None  # (x, y)
    target_mask = None    # binary mask for target disk
    target_center_locked = False

    # create window by default unless --no_show
    if (not a.no_show) or a.show or a.demo:
        try:
            cv2.namedWindow('swab-trajectory', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('swab-trajectory', max(320, W), max(240, H))
        except Exception as e:
            print('Warning: failed to create window:', e)



    centers = []
    prev_tip = None
    first_frame = None
    prev_swab_circle = None


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame is None:
            first_frame = frame.copy()
        # YOLO detect
        results = model(frame, conf=a.conf, device=a.device, verbose=False)

        # select best swab and best stoma
        swab_box, stoma_box = None, None
        swab_conf, stoma_conf = -1.0, -1.0
        swab_id, stoma_id = int(a.swab_id), int(a.stoma_id)
        for r in results:
            if getattr(r, 'boxes', None) is None:
                continue
            for b in r.boxes:
                c = float(b.conf[0].cpu().numpy()) if hasattr(b, 'conf') else 0.0
                cls_id = int(b.cls[0].cpu().numpy()) if hasattr(b, 'cls') else -1
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                box_tuple = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                if cls_id == swab_id and c > swab_conf:
                    swab_conf, swab_box = c, box_tuple
                if cls_id == stoma_id and c > stoma_conf:
                    stoma_conf, stoma_box = c, box_tuple

        draw = frame.copy()
        tip = None
        swab_circle = None


        # draw swab (class a.swab_id)
        if swab_box is not None:
            sx1, sy1, sx2, sy2 = swab_box
            cv2.rectangle(draw, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            scx = 0.5 * (sx1 + sx2)
            scy = 0.5 * (sy1 + sy2)
            tip = (float(scx), float(scy))
            # inscribed circle of swab bbox (center and radius)
            rw = max(0, sx2 - sx1)
            rh = max(0, sy2 - sy1)
            r_in = int(0.5 * min(rw, rh))
            if r_in > 0:
                swab_circle = (int(scx), int(scy), r_in)
                prev_swab_circle = swab_circle
                cv2.circle(draw, (int(scx), int(scy)), r_in, (0, 255, 0), 2)
            cv2.circle(draw, (int(scx), int(scy)), 5, (0, 255, 0), -1)
            # label
            try:
                name_swab = 'swab'
                if CLASS_NAMES is not None:
                    if isinstance(CLASS_NAMES, (list, tuple)):
                        if 0 <= swab_id < len(CLASS_NAMES):
                            name_swab = str(CLASS_NAMES[swab_id])
                    elif isinstance(CLASS_NAMES, dict):
                        name_swab = str(CLASS_NAMES.get(swab_id, 'swab'))
                lab = f"{name_swab} {swab_conf:.2f}"
                cv2.putText(draw, lab, (sx1, max(15, sy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(draw, lab, (sx1, max(15, sy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception:
                pass
        else:
            # fallback: repeat previous center and circle
            if prev_tip is not None:
                tip = prev_tip
            if prev_swab_circle is not None:
                swab_circle = prev_swab_circle

        # draw stoma (class a.stoma_id)
        stoma_center = None
        if stoma_box is not None:
            tx1, ty1, tx2, ty2 = stoma_box
            cv2.rectangle(draw, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)
            tcx = 0.5 * (tx1 + tx2)
            tcy = 0.5 * (ty1 + ty2)
            stoma_center = (float(tcx), float(tcy))
            cv2.circle(draw, (int(tcx), int(tcy)), 5, (255, 0, 0), -1)
            # label
            try:
                name_stoma = 'stoma'
                if CLASS_NAMES is not None:
                    if isinstance(CLASS_NAMES, (list, tuple)):
                        if 0 <= stoma_id < len(CLASS_NAMES):
                            name_stoma = str(CLASS_NAMES[stoma_id])
                    elif isinstance(CLASS_NAMES, dict):
                        name_stoma = str(CLASS_NAMES.get(stoma_id, 'stoma'))
                labt = f"{name_stoma} {stoma_conf:.2f}"
                cv2.putText(draw, labt, (tx1, max(15, ty1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(draw, labt, (tx1, max(15, ty1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception:
                pass

        # optional: show distance from swab center to stoma center
        if (tip is not None) and (stoma_center is not None):
            x0, y0 = int(tip[0]), int(tip[1])
            x1, y1 = int(stoma_center[0]), int(stoma_center[1])
            cv2.line(draw, (x0, y0), (x1, y1), (255, 255, 0), 1)
            dist = float(np.hypot(x1 - x0, y1 - y0))
            midx, midy = (x0 + x1) // 2, (y0 + y1) // 2
            txt = f"d={dist:.1f}px"
            cv2.putText(draw, txt, (midx, midy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(draw, txt, (midx, midy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # update target center from stoma and build target mask
        if stoma_center is not None:
            if a.lock_stoma:
                if not target_center_locked:
                    target_center = (int(stoma_center[0]), int(stoma_center[1]))
                    target_center_locked = True
                    # build target disk mask
                    target_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.circle(target_mask, target_center, int(a.target_r), 255, -1)
            else:
                # dynamically follow latest stoma center
                target_center = (int(stoma_center[0]), int(stoma_center[1]))
                target_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.circle(target_mask, target_center, int(a.target_r), 255, -1)

        # accumulate coverage using the inscribed circle of the swab bbox
        if swab_circle is not None:
            cx_i, cy_i, r_i = swab_circle
            cv2.circle(coverage, (int(cx_i), int(cy_i)), int(r_i), 255, -1)



        if tip is not None:
            centers.append(tip)
            prev_tip = tip
        elif len(centers) > 0:
            centers.append(centers[-1])
        else:
            centers.append((float(W) / 2.0, float(H) / 2.0))

        # fading trail update
        if trail is None:
            trail = np.zeros_like(draw, dtype=np.uint8)
        else:
            trail = (trail * TRAIL_DECAY).astype(np.uint8)
        if len(centers) > 1:
            x0, y0 = map(int, centers[-2])
            x1, y1 = map(int, centers[-1])
            cv2.line(trail, (x0, y0), (x1, y1), (0, 0, 255), TRAIL_THICK)

        # composite: trail + current frame
        base = cv2.addWeighted(draw, 0.7, trail, 0.9, 0)

        # continuous online circle estimation (sliding window + quality gate + smoothing)
        if not a.no_circle:
            if len(centers) >= MIN_PTS:
                pts_f = np.array(centers[-WINDOW:], dtype=np.float64) if len(centers) > WINDOW else np.array(centers, dtype=np.float64)
                fc_live = fit_circle_least_squares(pts_f)
                if fc_live is not None:
                    cx_l, cy_l, r_l = fc_live
                    radii_l = np.hypot(pts_f[:, 0] - cx_l, pts_f[:, 1] - cy_l)
                    m_l = float(np.mean(radii_l))
                    s_l = float(np.std(radii_l))
                    std_ratio = (s_l / m_l) if m_l > 1e-6 else 1.0
                    ang = (np.degrees(np.arctan2(pts_f[:, 1] - cy_l, pts_f[:, 0] - cx_l)) + 360.0) % 360.0
                    ang.sort()
                    diffs = np.diff(np.concatenate([ang, ang[:1] + 360.0]))
                    max_gap = float(diffs.max()) if diffs.size else 360.0
                    ang_coverage = 360.0 - max_gap
                    if (ang_coverage >= MIN_SPAN_DEG) and (std_ratio <= MAX_STD_RATIO):
                        if circle_est is None:
                            circle_est = (cx_l, cy_l, r_l)
                            if not first_snapshot_saved:
                                # save first good circle snapshot
                                tmp = base.copy()
                                cv2.circle(tmp, (int(cx_l), int(cy_l)), max(1, int(r_l)), (255, 0, 0), 2)
                                cv2.circle(tmp, (int(cx_l), int(cy_l)), 4, (255, 255, 255), -1)
                                live_png = str(Path(a.out_dir) / f'{stem}_circle_live.png')
                                cv2.imwrite(live_png, tmp)
                                first_snapshot_saved = True
                        else:
                            # exponential smoothing update (guard None)
                            if circle_est is not None:
                                cx0, cy0, r0 = circle_est
                                cx_u = (1 - ALPHA) * cx0 + ALPHA * cx_l
                                cy_u = (1 - ALPHA) * cy0 + ALPHA * cy_l
                                r_u  = (1 - ALPHA) * r0  + ALPHA * r_l
                                circle_est = (cx_u, cy_u, r_u)


        # compute coverage ratio within target disk and draw overlays (after base is prepared)
        if target_mask is not None:
            tgt_area = int(np.count_nonzero(target_mask))
            if tgt_area > 0:
                cov_in = cv2.bitwise_and(coverage, coverage, mask=target_mask)
                cov_area = int(np.count_nonzero(cov_in))
                ratio = float(cov_area) / float(tgt_area)
                # draw target circle on base
                if target_center is not None:
                    cv2.circle(base, target_center, int(a.target_r), (0, 255, 255), 2)
                # draw semi-transparent coverage overlay (green) within target
                cov_color = np.zeros_like(base)
                cov_color[:, :] = (0, 200, 0)
                cov_alpha = 0.3
                mask_bool = (cov_in > 0)
                if np.any(mask_bool):
                    base[mask_bool] = (base[mask_bool] * (1 - cov_alpha) + cov_color[mask_bool] * cov_alpha).astype(np.uint8)
                # text feedback
                txt = f"coverage: {ratio*100:.1f}% ({cov_area}/{tgt_area})"
                cv2.putText(base, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(base, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if ratio >= float(a.cov_th):
                    ok = "OK"
                    cv2.putText(base, ok, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                    cv2.putText(base, ok, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


            if circle_est is not None:
                cx_l, cy_l, r_l = circle_est
                cv2.circle(base, (int(cx_l), int(cy_l)), max(1, int(r_l)), (255, 0, 0), 2)
                cv2.circle(base, (int(cx_l), int(cy_l)), 4, (255, 255, 255), -1)

        if vw is not None:
            vw.write(base)

        if (not a.no_show) or a.show or a.demo:
            cv2.imshow('swab-trajectory', base)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if a.show:
            cv2.imshow('swab-trajectory', draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()

    # save CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        wri = csv.writer(f)
        wri.writerow(['frame', 'x', 'y'])
        for i, (x, y) in enumerate(centers):
            wri.writerow([i, f'{x:.2f}', f'{y:.2f}'])

    # save PNG overlay + fit circle
    if first_frame is None:
        first_frame = np.zeros((H, W, 3), dtype=np.uint8)
    overlay = first_frame.copy()
    fitted_circle = None
    if len(centers) > 1:
        pts = np.array([[int(x), int(y)] for (x, y) in centers], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=False, color=(0, 0, 255), thickness=3)
        cv2.circle(overlay, (int(centers[0][0]), int(centers[0][1])), 5, (0, 255, 0), -1)
        cv2.circle(overlay, (int(centers[-1][0]), int(centers[-1][1])), 5, (0, 0, 255), -1)
        fc = fit_circle_least_squares(centers)
        if fc is not None:
            cx, cy, r = fc
            fitted_circle = (cx, cy, r)
            # draw circle
            cv2.circle(overlay, (int(cx), int(cy)), max(1, int(r)), (255, 0, 0), 2)
            cv2.circle(overlay, (int(cx), int(cy)), 4, (255, 255, 255), -1)
    cv2.imwrite(out_png, overlay)

    # save circle stats
    if fitted_circle is not None:
        cx, cy, r = fitted_circle
        radii = [float(np.hypot(x - cx, y - cy)) for (x, y) in centers]
        mean_r = float(np.mean(radii))
        std_r = float(np.std(radii))
        txt_path = str(Path(a.out_dir) / f'{stem}_circle.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f'cx: {cx:.2f}\ncy: {cy:.2f}\nr_fit: {r:.2f}\n')
            f.write(f'r_mean: {mean_r:.2f}\nr_std: {std_r:.2f}\n')
        print('Circle:', f'center=({cx:.2f},{cy:.2f}) r_fit={r:.2f} r_mean={mean_r:.2f} r_std={std_r:.2f}')
        print('TXT   :', txt_path)

    print('Done.')
    print('Video :', out_video)
    print('CSV   :', out_csv)
    print('PNG   :', out_png)


if __name__ == '__main__':
    main()

