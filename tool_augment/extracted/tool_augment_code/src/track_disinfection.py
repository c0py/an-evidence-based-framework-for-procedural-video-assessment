import os
import cv2
import csv
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Track and draw disinfection trajectory using sparse optical flow")
    p.add_argument("--video", type=str, default="圆形消毒.mp4", help="Input video path")
    p.add_argument("--output_video", type=str, default=None, help="Output annotated video path (.mp4)")
    p.add_argument("--output_csv", type=str, default=None, help="CSV to save (frame,x,y)")
    p.add_argument("--output_png", type=str, default=None, help="PNG image of the whole trajectory over the first frame")
    p.add_argument("--resize", type=float, default=1.0, help="Optional uniform resize factor (e.g., 0.5)")
    p.add_argument("--roi", type=str, default=None, help="Optional ROI in 'x,y,w,h'. If not set, an interactive selector will pop up on the first frame")
    p.add_argument("--show", action="store_true", help="Show live preview window")
    p.add_argument("--max_corners", type=int, default=150, help="Max corners to track")
    p.add_argument("--min_corners", type=int, default=20, help="Min corners before re-detecting")
    p.add_argument("--quality", type=float, default=0.02, help="Quality level for Shi-Tomasi")
    p.add_argument("--min_dist", type=float, default=7, help="Min distance between features")
    p.add_argument("--smooth", type=int, default=5, help="Moving-average window for trajectory smoothing; 1 disables")
    return p.parse_args()


def parse_roi(s):
    try:
        x, y, w, h = [int(float(t)) for t in s.split(",")]
        return (x, y, w, h)
    except Exception:
        raise ValueError("--roi must be like: x,y,w,h")


def ensure_output_paths(args):
    stem, _ = os.path.splitext(os.path.basename(args.video))
    if args.output_video is None:
        args.output_video = f"{stem}_带轨迹.mp4"
    if args.output_csv is None:
        args.output_csv = f"{stem}_轨迹.csv"
    if args.output_png is None:
        args.output_png = f"{stem}_轨迹.png"


def good_features(gray, mask, max_corners, quality, min_dist):
    feat = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist,
        blockSize=7,
        mask=mask,
        useHarrisDetector=False,
    )
    return feat


def make_mask(shape, roi):
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, w, h = roi
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def moving_average(points, k):
    if k <= 1 or len(points) == 0:
        return points
    k = max(1, int(k))
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    kernel = np.ones(k, dtype=np.float32) / k
    xs_s = np.convolve(xs, kernel, mode="same")
    ys_s = np.convolve(ys, kernel, mode="same")
    return [(float(x), float(y)) for x, y in zip(xs_s, ys_s)]


def main():
    args = parse_args()
    ensure_output_paths(args)

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame")

    if args.resize != 1.0:
        frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

    H, W = frame.shape[:2]
    first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ROI selection
    if args.roi is not None:
        roi = parse_roi(args.roi)
    else:
        # Interactive ROI selection
        sel = cv2.selectROI("选择消毒区域ROI (回车确认)", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("选择消毒区域ROI (回车确认)")
        roi = tuple(map(int, sel))
        if roi[2] == 0 or roi[3] == 0:
            # fallback to center square
            w = h = min(H, W) // 3
            x = (W - w) // 2
            y = (H - h) // 2
            roi = (x, y, w, h)
            print("未选择ROI，使用中心区域作为默认ROI:", roi)

    # Prepare feature points in ROI
    mask = make_mask(first_gray.shape, roi)
    p0 = good_features(first_gray, mask, args.max_corners, args.quality, args.min_dist)

    if p0 is None or len(p0) < 5:
        # widen ROI slightly
        x, y, w, h = roi
        grow = int(0.2 * max(w, h))
        x = max(0, x - grow)
        y = max(0, y - grow)
        w = min(W - x, w + 2 * grow)
        h = min(H - y, h + 2 * grow)
        roi = (x, y, w, h)
        mask = make_mask(first_gray.shape, roi)
        p0 = good_features(first_gray, mask, args.max_corners, args.quality, args.min_dist)

    if p0 is None:
        raise RuntimeError("Could not find enough features to track in the selected ROI. Try a different ROI.")

    # Video writer and drawing canvas
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (W, H))

    # colors for points
    np.random.seed(0)
    color = np.random.randint(0, 255, (max(len(p0), 100), 3), dtype=np.uint8)

    # lists to store trajectory
    centers = []  # (x,y)
    frame_idx = 0

    prev_gray = first_gray.copy()
    p_prev = p0

    # for LK params
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # draw ROI rectangle
    vis = frame.copy()
    x, y, w, h = roi
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    # initial center estimate
    c0 = (x + w // 2, y + h // 2)
    centers.append((float(c0[0]), float(c0[1])))

    # initial points visualization
    for i, pt in enumerate(p_prev.reshape(-1, 2)):
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(vis, (px, py), 2, (int(color[i % len(color)][0]), int(color[i % len(color)][1]), int(color[i % len(color)][2])), -1)
    writer.write(vis)
    if args.show:
        cv2.imshow("轨迹跟踪", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    frame_idx += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.resize != 1.0:
            frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute forward flow
        p_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p_prev, None, **lk_params)
        if p_next is None:
            # try re-detect around last center
            if len(centers) > 0:
                cx, cy = map(int, centers[-1])
                box_w = max(20, roi[2] // 2)
                box_h = max(20, roi[3] // 2)
                rx = max(0, cx - box_w)
                ry = max(0, cy - box_h)
                rw = min(W - rx, 2 * box_w)
                rh = min(H - ry, 2 * box_h)
                mask = make_mask(gray.shape, (rx, ry, rw, rh))
                p_prev = good_features(gray, mask, args.max_corners, args.quality, args.min_dist)
                prev_gray = gray.copy()
                # draw and continue
                vis = frame.copy()
                for i, pt in enumerate((p_prev or np.zeros((0,1,2))).reshape(-1, 2)):
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                # keep last center
                if len(centers) > 0:
                    centers.append(centers[-1])
                writer.write(vis)
                if args.show:
                    cv2.imshow("轨迹跟踪", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_idx += 1
                continue
            else:
                break

        st = st.reshape(-1) if st is not None else None
        if st is None:
            break
        good_new = p_next[st == 1]
        good_old = p_prev[st == 1]

        # forward-backward check to remove outliers
        p_back, st_back, err_back = cv2.calcOpticalFlowPyrLK(gray, prev_gray, good_new.reshape(-1, 1, 2), None, **lk_params)
        if p_back is not None and st_back is not None:
            fb_dist = np.linalg.norm(good_old.reshape(-1, 2) - p_back.reshape(-1, 2), axis=1)
            keep = fb_dist < 1.5  # threshold in pixels
            good_new = good_new[keep]
            good_old = good_old[keep]

        # compute center as robust median of good_new
        if len(good_new) > 0:
            cx = float(np.median(good_new[:, 0]))
            cy = float(np.median(good_new[:, 1]))
            centers.append((cx, cy))
        else:
            # fallback to last center
            if len(centers) > 0:
                centers.append(centers[-1])
            else:
                centers.append((float(W) / 2, float(H) / 2))

        # re-detect features if too few
        if len(good_new) < args.min_corners:
            # focus mask around current center
            cx_i, cy_i = map(int, centers[-1])
            box_w = max(20, roi[2] // 2)
            box_h = max(20, roi[3] // 2)
            rx = max(0, cx_i - box_w)
            ry = max(0, cy_i - box_h)
            rw = min(W - rx, 2 * box_w)
            rh = min(H - ry, 2 * box_h)
            mask = make_mask(gray.shape, (rx, ry, rw, rh))
            p_prev = good_features(gray, mask, args.max_corners, args.quality, args.min_dist)
            if p_prev is None:
                p_prev = good_features(gray, None, args.max_corners, args.quality, args.min_dist)
        else:
            p_prev = good_new.reshape(-1, 1, 2)

        prev_gray = gray.copy()

        # draw trajectory so far
        vis = frame.copy()
        # draw points
        for i, (new, old) in enumerate(zip(good_new.reshape(-1, 2), good_old.reshape(-1, 2))):
            a, b = int(new[0]), int(new[1])
            cv2.circle(vis, (a, b), 2, (int(color[i % len(color)][0]), int(color[i % len(color)][1]), int(color[i % len(color)][2])), -1)
            # optional: draw motion line
            oa, ob = int(old[0]), int(old[1])
            cv2.line(vis, (oa, ob), (a, b), (200, 200, 200), 1)

        # draw polyline of centers
        if len(centers) > 1:
            pts = np.array([[int(x), int(y)] for (x, y) in centers], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            # highlight current center
            cv2.circle(vis, (int(centers[-1][0]), int(centers[-1][1])), 4, (0, 0, 255), -1)

        writer.write(vis)
        if args.show:
            cv2.imshow("轨迹跟踪", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Optionally smooth and save results
    centers_sm = moving_average(centers, args.smooth)

    # Save CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["frame", "x", "y"]) 
        for i, (x, y) in enumerate(centers_sm):
            wri.writerow([i, f"{x:.2f}", f"{y:.2f}"])

    # Save an image with the full path over the first frame
    img = frame if 'frame' in locals() else None
    if img is None:
        # reload first frame for the backdrop
        cap2 = cv2.VideoCapture(args.video)
        ret2, img = cap2.read()
        cap2.release()
        if not ret2:
            img = np.zeros((H, W, 3), dtype=np.uint8)
    if args.resize != 1.0:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    overlay = img.copy()
    if len(centers_sm) > 1:
        pts = np.array([[int(x), int(y)] for (x, y) in centers_sm], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=False, color=(0, 0, 255), thickness=3)
        cv2.circle(overlay, (int(centers_sm[0][0]), int(centers_sm[0][1])), 5, (0, 255, 0), -1)
        cv2.circle(overlay, (int(centers_sm[-1][0]), int(centers_sm[-1][1])), 5, (0, 0, 255), -1)
    cv2.imwrite(args.output_png, overlay)

    print("\n完成：")
    print(f"- 轨迹视频: {args.output_video}")
    print(f"- 轨迹CSV:  {args.output_csv}")
    print(f"- 轨迹PNG:  {args.output_png}")


if __name__ == "__main__":
    main()

