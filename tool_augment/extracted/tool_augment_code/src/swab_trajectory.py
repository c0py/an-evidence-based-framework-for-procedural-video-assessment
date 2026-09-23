import os
import cv2
import csv
import argparse
import numpy as np
from typing import Optional, Tuple

# 可选：仅在提供模型时才导入YOLO，避免环境无该依赖时报错
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # 延迟判断


def parse_args():
    p = argparse.ArgumentParser(description="Detect cotton swab tip contact trajectory in disinfection video")
    p.add_argument("--video", type=str, default="圆形消毒.mp4", help="Input video path")
    p.add_argument("--model", type=str, default=None, help="Optional YOLO model path for swab detection (e.g., swab.pt)")
    p.add_argument("--output_video", type=str, default=None, help="Output annotated video path (.mp4)")
    p.add_argument("--output_csv", type=str, default=None, help="CSV to save (frame,x,y)")
    p.add_argument("--output_png", type=str, default=None, help="PNG image of the whole trajectory over the first frame")
    p.add_argument("--resize", type=float, default=1.0, help="Uniform resize factor for speed (e.g., 0.5)")
    p.add_argument("--show", action="store_true", help="Show live preview window")
    p.add_argument("--roi", type=str, default=None, help="Skin ROI in 'x,y,w,h'. If not set, interactive selector pops up on first frame")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold if model provided")
    p.add_argument("--smooth", type=int, default=5, help="Moving-average window for trajectory smoothing; 1 disables")
    p.add_argument("--debug", action="store_true", help="Draw intermediate edges/lines for debugging")
    return p.parse_args()


def ensure_outputs(args):
    stem, _ = os.path.splitext(os.path.basename(args.video))
    if args.output_video is None:
        args.output_video = f"{stem}_带轨迹.mp4"
    if args.output_csv is None:
        args.output_csv = f"{stem}_轨迹.csv"
    if args.output_png is None:
        args.output_png = f"{stem}_轨迹.png"


def parse_roi(s: str) -> Tuple[int, int, int, int]:
    try:
        x, y, w, h = [int(float(t)) for t in s.split(",")]
        return (x, y, w, h)
    except Exception:
        raise ValueError("--roi must be like: x,y,w,h")


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


def detect_swab_box(model, frame, conf=0.25):
    """使用YOLO检测到的最大置信度框；若无检测，返回None。
    注意：若未训练专门的'棉签'类别，该结果可能无效。
    """
    if model is None:
        return None
    try:
        results = model(frame, conf=conf)
        best = None
        best_conf = -1
        for r in results:
            if getattr(r, 'boxes', None) is not None:
                for box in r.boxes:
                    c = float(box.conf[0].cpu().numpy()) if getattr(box, 'conf', None) is not None else 0.0
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    if c > best_conf:
                        best_conf = c
                        best = (x1, y1, x2, y2, c)
        return best
    except Exception:
        return None


def detect_line_in_roi(img_bgr) -> Optional[Tuple[int, int, int, int]]:
    """在ROI图像内检测最长的直线段，返回端点坐标(相对ROI局部坐标)。"""
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=max(30, min(img_bgr.shape[:2])//6), maxLineGap=20)
    if lines is None:
        return None
    best = None
    best_len = 0.0
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > best_len:
            best_len = length
            best = (int(x1), int(y1), int(x2), int(y2))
    return best


def skin_mask(bgr_img: np.ndarray) -> np.ndarray:
    """简单YCrCb阈值的肤色分割，返回二值mask(0/255)。"""
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    # 常用阈值范围，适配不同肤色可再调节
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    # 适度形态处理平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    return skin


def endpoint_skin_score(frame_bgr: np.ndarray, pt: Tuple[int, int], radius: int = 6) -> float:
    """计算端点周围小圆邻域的肤色占比作为接触得分。"""
    H, W = frame_bgr.shape[:2]
    x0, y0 = int(pt[0]), int(pt[1])
    r = int(max(2, radius))
    x1, y1 = max(0, x0 - r), max(0, y0 - r)
    x2, y2 = min(W, x0 + r + 1), min(H, y0 + r + 1)
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    mask = skin_mask(patch)
    # 圆形权重掩膜，避免方块边角影响
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    cy, cx = y0 - y1, x0 - x1
    circ = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if np.any(circ):
        skin_pixels = float(np.count_nonzero(mask[circ]))
        total = float(np.count_nonzero(circ))
        return skin_pixels / max(1.0, total)
    else:
        return 0.0


def pick_tip(prev_tip: Optional[Tuple[float, float]], p1: Tuple[int, int], p2: Tuple[int, int], fallback_center: Tuple[int, int], frame_bgr: np.ndarray):
    """从线段两个端点中选择作为棉签尖端的点：
    1) 优先选择端点周围肤色占比更高者（更可能是接触皮肤的一端）；
    2) 若肤色得分相近，则使用与上一帧tip更近的端点；
    3) 若仍无法区分，则使用靠近搜索窗口中心的端点。
    """
    # 皮肤接触优先
    s1 = endpoint_skin_score(frame_bgr, p1)
    s2 = endpoint_skin_score(frame_bgr, p2)
    if abs(s1 - s2) > 0.05:  # 有明显差异
        return p1 if s1 > s2 else p2

    # 与上一帧的连续性
    if prev_tip is not None:
        d1 = (p1[0] - prev_tip[0])**2 + (p1[1] - prev_tip[1])**2
        d2 = (p2[0] - prev_tip[0])**2 + (p2[1] - prev_tip[1])**2
        if d1 != d2:
            return p1 if d1 <= d2 else p2

    # 回退：与窗口中心更近
    d1 = (p1[0] - fallback_center[0])**2 + (p1[1] - fallback_center[1])**2
    d2 = (p2[0] - fallback_center[0])**2 + (p2[1] - fallback_center[1])**2
    return p1 if d1 <= d2 else p2


def main():
    args = parse_args()
    ensure_outputs(args)

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    # YOLO模型（可选）
    yolo_model = None
    if args.model is not None:
        if YOLO is None:
            print("警告：未找到ultralytics.YOLO依赖，跳过YOLO检测。")
        elif not os.path.exists(args.model):
            print(f"警告：YOLO模型文件不存在: {args.model}，跳过YOLO检测。")
        else:
            try:
                yolo_model = YOLO(args.model)
                print(f"已加载YOLO模型: {args.model}")
            except Exception as e:
                print(f"加载YOLO模型失败，跳过YOLO: {e}")
                yolo_model = None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame")

    if args.resize != 1.0:
        frame0 = cv2.resize(frame0, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

    H, W = frame0.shape[:2]

    # 选择/解析皮肤区域ROI
    if args.roi is not None:
        roi = parse_roi(args.roi)
    else:
        sel = cv2.selectROI("选择皮肤消毒区域ROI (回车确认)", frame0, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("选择皮肤消毒区域ROI (回车确认)")
        roi = tuple(map(int, sel))
        if roi[2] == 0 or roi[3] == 0:
            w = h = min(H, W) // 2
            x = (W - w) // 2
            y = (H - h) // 2
            roi = (x, y, w, h)
            print("未选择ROI，使用居中默认ROI:", roi)

    # 输出准备
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (W, H))

    # 轨迹
    centers = []  # 记录tip (x,y)
    prev_tip: Optional[Tuple[float, float]] = None

    # 可视化第一帧：画ROI
    vis0 = frame0.copy()
    x, y, w, h = roi
    cv2.rectangle(vis0, (x, y), (x + w, y + h), (0, 255, 255), 2)
    writer.write(vis0)
    if args.show:
        cv2.imshow("棉签轨迹", vis0)
        cv2.waitKey(1)

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.resize != 1.0:
            frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

        draw = frame.copy()

        # 先用YOLO在全帧找大致框（若有模型），否则使用上一帧附近的局部窗口
        box = detect_swab_box(yolo_model, frame, conf=args.conf) if yolo_model is not None else None

        # 设定本帧检测的搜索窗口：优先YOLO框；否则在ROI内以上一tip为中心的局部窗口；都没有时用ROI
        search_rect = None  # (sx, sy, sw, sh)
        if box is not None:
            x1, y1, x2, y2, _ = box
            # 与ROI求交，避免落到ROI外部
            sx = max(x, min(x2, max(x1, x)))
            sy = max(y, min(y2, max(y1, y)))
            ex = min(x + w, max(x1, min(x2, x + w)))
            ey = min(y + h, max(y1, min(y2, y + h)))
            if ex > sx and ey > sy:
                search_rect = (sx, sy, ex - sx, ey - sy)
        if search_rect is None:
            if prev_tip is not None:
                cx, cy = int(prev_tip[0]), int(prev_tip[1])
                half = max(20, min(w, h)//4)
                sx = max(x, cx - half)
                sy = max(y, cy - half)
                ex = min(x + w, cx + half)
                ey = min(y + h, cy + half)
                if ex > sx and ey > sy:
                    search_rect = (sx, sy, ex - sx, ey - sy)
        if search_rect is None:
            search_rect = roi

        sx, sy, sw, sh = search_rect
        roi_img = frame[sy:sy+sh, sx:sx+sw]

        line_local = detect_line_in_roi(roi_img)
        tip = None
        if line_local is not None:
            lx1, ly1, lx2, ly2 = line_local
            p1 = (sx + lx1, sy + ly1)
            p2 = (sx + lx2, sy + ly2)
            fallback_center = (sx + sw // 2, sy + sh // 2)
            tip_pt = pick_tip(prev_tip, p1, p2, fallback_center, frame)
            tip = (float(tip_pt[0]), float(tip_pt[1]))

            # 可视化直线与端点
            cv2.line(draw, p1, p2, (0, 0, 255), 2)
            cv2.circle(draw, (int(tip[0]), int(tip[1])), 5, (0, 255, 0), -1)
        else:
            # 未检测到直线，延用上一tip
            if prev_tip is not None:
                tip = (prev_tip[0], prev_tip[1])

        # 记录并绘制轨迹
        if tip is not None:
            centers.append(tip)
            prev_tip = tip
        elif len(centers) > 0:
            # 重复上一点以保持长度一致
            centers.append(centers[-1])
        else:
            centers.append((float(x + w//2), float(y + h//2)))

        # ROI与搜索框绘制
        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.rectangle(draw, (sx, sy), (sx + sw, sy + sh), (255, 200, 0), 1)

        # 轨迹折线
        if len(centers) > 1:
            pts = np.array([[int(px), int(py)] for (px, py) in centers], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(draw, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.circle(draw, (int(centers[-1][0]), int(centers[-1][1])), 4, (0, 0, 255), -1)

        # 调试：显示边缘
        if args.debug:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            ph, pw = dbg.shape[:2]
            draw[sy:sy+ph, sx:sx+pw] = cv2.addWeighted(draw[sy:sy+ph, sx:sx+pw], 0.6, dbg, 0.4, 0)

        writer.write(draw)
        if args.show:
            cv2.imshow("棉签轨迹", draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 平滑与导出
    centers_sm = moving_average(centers, args.smooth)

    # CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["frame", "x", "y"]) 
        for i, (xv, yv) in enumerate(centers_sm):
            wri.writerow([i, f"{xv:.2f}", f"{yv:.2f}"])

    # PNG（整条轨迹覆盖在首帧）
    backdrop = frame0.copy()
    overlay = backdrop.copy()
    if len(centers_sm) > 1:
        pts = np.array([[int(px), int(py)] for (px, py) in centers_sm], dtype=np.int32).reshape(-1, 1, 2)
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

