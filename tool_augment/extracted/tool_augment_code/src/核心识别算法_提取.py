"""
从本项目提取的核心识别算法。

来源：
  - detect_swab.py
  - 消毒细节检测.py
  - swab_trajectory.py
  - track_disinfection.py
  - find_arm.py
  - hand_rec.py

这里保留算法本身，去掉了命令行、视频写入、GUI 和 PyTorch 兼容性补丁，方便
在其他程序中复用。模型仍由 Ultralytics/MediaPipe 加载。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]
Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# 1. YOLO 通用检测
# ---------------------------------------------------------------------------

def load_yolo(model_path: str):
    """加载本地 Ultralytics YOLO 权重。"""
    from ultralytics import YOLO
    return YOLO(model_path)


def _box_values(box: Any) -> Tuple[Box, float, int]:
    """将 Ultralytics Box 转为 (xyxy, confidence, class_id)。"""
    # 使用 tolist/item 优先，避免某些 torch 版本与 NumPy ABI 不兼容时，
    # 仅做框解析就触发 tensor.numpy() 异常。
    xyxy_value = box.xyxy[0]
    if hasattr(xyxy_value, "detach"):
        xyxy_value = xyxy_value.detach().cpu().tolist()
    elif hasattr(xyxy_value, "tolist"):
        xyxy_value = xyxy_value.tolist()
    xyxy = np.asarray(xyxy_value).astype(int)

    def scalar(value: Any, default: float) -> float:
        if value is None:
            return default
        value = value[0] if hasattr(value, "__getitem__") else value
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    conf = scalar(getattr(box, "conf", None), 0.0)
    cls_id = int(scalar(getattr(box, "cls", None), -1))
    return (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])), conf, cls_id


def select_best_box(results: Iterable[Any], class_id: Optional[int] = None,
                    conf_threshold: float = 0.0) -> Optional[Tuple[Box, float, int]]:
    """从一帧 YOLO 结果中选置信度最高的框，可按类别过滤。"""
    best = None
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            values = _box_values(box)
            _, conf, cls_id = values
            if conf < conf_threshold or (class_id is not None and cls_id != class_id):
                continue
            if best is None or conf > best[1]:
                best = values
    return best


def infer_yolo_frame(model: Any, frame: np.ndarray, conf: float = 0.5,
                     device: str = "") -> List[Tuple[Box, float, int]]:
    """对单帧执行 YOLO 检测并返回所有通过阈值的框。"""
    kwargs = {"conf": conf, "verbose": False}
    if device:
        kwargs["device"] = device
    results = model(frame, **kwargs)
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy, score, cls_id = _box_values(box)
            if score >= conf:
                detections.append((xyxy, score, cls_id))
    return detections


def draw_yolo_detections(frame: np.ndarray, detections: Sequence[Tuple[Box, float, int]],
                         class_names: Any = None) -> np.ndarray:
    """绘制 YOLO 框、类别和置信度。"""
    output = frame.copy()
    for (x1, y1, x2, y2), score, cls_id in detections:
        if isinstance(class_names, dict):
            name = class_names.get(cls_id, str(cls_id))
        elif isinstance(class_names, (list, tuple)) and 0 <= cls_id < len(class_names):
            name = class_names[cls_id]
        else:
            name = str(cls_id)
        label = f"{name} {score:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output


# ---------------------------------------------------------------------------
# 2. 圆形消毒：YOLO + 轨迹 + 覆盖率 + 圆拟合
# ---------------------------------------------------------------------------

def fit_circle_least_squares(points: Sequence[Point]) -> Optional[Tuple[float, float, float]]:
    """用线性最小二乘拟合圆，返回 (圆心x, 圆心y, 半径)。"""
    if points is None or len(points) < 3:
        return None
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c0 = theta
        radius = float(np.sqrt(cx * cx + cy * cy + c0))
        return float(cx), float(cy), radius
    except Exception:
        return None


def _angular_coverage(points: np.ndarray, cx: float, cy: float) -> float:
    angles = (np.degrees(np.arctan2(points[:, 1] - cy, points[:, 0] - cx)) + 360.0) % 360.0
    angles.sort()
    if len(angles) == 0:
        return 0.0
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    return float(360.0 - gaps.max())


@dataclass
class CircularDisinfectionTracker:
    """跟踪棉签和造口，并计算圆形消毒覆盖率。"""

    swab_id: int = 0
    stoma_id: int = 1
    target_radius: int = 200
    coverage_threshold: float = 0.95
    lock_stoma: bool = False
    window_size: int = 120
    min_points: int = 40
    min_angle_span: float = 240.0
    max_radius_std_ratio: float = 0.20
    circle_alpha: float = 0.2

    def __post_init__(self) -> None:
        self.centers: List[Point] = []
        self.coverage: Optional[np.ndarray] = None
        self.target_center: Optional[Tuple[int, int]] = None
        self.target_center_locked = False
        self.circle_est: Optional[Tuple[float, float, float]] = None
        self.previous_swab_circle: Optional[Tuple[int, int, int]] = None

    def update(self, results: Iterable[Any], frame_shape: Tuple[int, int],
               class_names: Any = None) -> Dict[str, Any]:
        """处理一帧 YOLO 结果，返回轨迹、圆和覆盖率状态。"""
        height, width = frame_shape[:2]
        if self.coverage is None or self.coverage.shape != (height, width):
            self.coverage = np.zeros((height, width), dtype=np.uint8)

        if not isinstance(results, (list, tuple)):
            results = list(results)
        # 结果可能是生成器，先物化后再分别筛选 swab 和 stoma。
        swab = select_best_box(results, self.swab_id)
        stoma = select_best_box(results, self.stoma_id)

        tip: Optional[Point] = None
        swab_circle = None
        if swab is not None:
            (x1, y1, x2, y2), swab_conf, _ = swab
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            tip = (cx, cy)
            radius = int(0.5 * min(max(0, x2 - x1), max(0, y2 - y1)))
            if radius > 0:
                swab_circle = (int(cx), int(cy), radius)
                self.previous_swab_circle = swab_circle
        elif self.centers:
            tip = self.centers[-1]
            swab_circle = self.previous_swab_circle

        stoma_center = None
        if stoma is not None:
            (x1, y1, x2, y2), stoma_conf, _ = stoma
            stoma_center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
            if self.lock_stoma:
                if not self.target_center_locked:
                    self.target_center = (int(stoma_center[0]), int(stoma_center[1]))
                    self.target_center_locked = True
            else:
                self.target_center = (int(stoma_center[0]), int(stoma_center[1]))

        if swab_circle is not None:
            cx, cy, radius = swab_circle
            cv2.circle(self.coverage, (cx, cy), radius, 255, -1)

        if tip is not None:
            self.centers.append(tip)
        elif self.centers:
            self.centers.append(self.centers[-1])
        else:
            self.centers.append((width / 2.0, height / 2.0))

        self._update_circle_estimate()
        ratio = self.coverage_ratio((height, width))
        return {
            "tip": tip,
            "swab": swab,
            "stoma": stoma,
            "stoma_center": stoma_center,
            "coverage_ratio": ratio,
            "disinfected": ratio >= self.coverage_threshold if ratio is not None else False,
            "circle": self.circle_est,
            "class_names": class_names,
        }

    def coverage_ratio(self, frame_shape: Tuple[int, int]) -> Optional[float]:
        if self.coverage is None or self.target_center is None:
            return None
        height, width = frame_shape[:2]
        target = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(target, self.target_center, self.target_radius, 255, -1)
        target_area = int(np.count_nonzero(target))
        if target_area == 0:
            return None
        covered = cv2.bitwise_and(self.coverage, self.coverage, mask=target)
        return float(np.count_nonzero(covered)) / target_area

    def _update_circle_estimate(self) -> None:
        if len(self.centers) < self.min_points:
            return
        points = np.asarray(self.centers[-self.window_size:], dtype=np.float64)
        fitted = fit_circle_least_squares(points)
        if fitted is None:
            return
        cx, cy, radius = fitted
        radii = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        mean_radius = float(np.mean(radii))
        std_ratio = float(np.std(radii)) / max(mean_radius, 1e-6)
        span = _angular_coverage(points, cx, cy)
        if span < self.min_angle_span or std_ratio > self.max_radius_std_ratio:
            return
        if self.circle_est is None:
            self.circle_est = fitted
        else:
            old_cx, old_cy, old_r = self.circle_est
            a = self.circle_alpha
            self.circle_est = (
                (1 - a) * old_cx + a * cx,
                (1 - a) * old_cy + a * cy,
                (1 - a) * old_r + a * radius,
            )


# ---------------------------------------------------------------------------
# 3. 圆形消毒的传统视觉备选：ROI + 角点 + 稀疏 LK 光流
# ---------------------------------------------------------------------------

def good_features(gray: np.ndarray, mask: Optional[np.ndarray], max_corners: int = 150,
                  quality: float = 0.02, min_distance: float = 7) -> Optional[np.ndarray]:
    return cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_distance, blockSize=7, mask=mask,
        useHarrisDetector=False,
    )


def optical_flow_center(prev_gray: np.ndarray, gray: np.ndarray,
                        previous_points: np.ndarray) -> Tuple[Optional[Point], np.ndarray, np.ndarray]:
    """一次 LK 前后向光流更新，使用通过一致性检查点的中位数作为中心。"""
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, previous_points, None, **lk)
    if next_points is None or status is None:
        return None, np.empty((0, 2)), np.empty((0, 2))
    good_new = next_points[status.reshape(-1) == 1]
    good_old = previous_points[status.reshape(-1) == 1]
    if len(good_new) == 0:
        return None, good_new, good_old

    back, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, good_new.reshape(-1, 1, 2), None, **lk)
    if back is not None and back_status is not None:
        fb_distance = np.linalg.norm(good_old - back.reshape(-1, 2), axis=1)
        keep = (back_status.reshape(-1) == 1) & (fb_distance < 1.5)
        good_new, good_old = good_new[keep], good_old[keep]
    if len(good_new) == 0:
        return None, good_new, good_old
    center = (float(np.median(good_new[:, 0])), float(np.median(good_new[:, 1])))
    return center, good_new, good_old


# ---------------------------------------------------------------------------
# 4. 棉签尖端：ROI 内 Hough 直线 + 肤色端点判别
# ---------------------------------------------------------------------------

def skin_mask(bgr_image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def detect_longest_line(roi_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 80, 160)
    h, w = roi_bgr.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                            minLineLength=max(30, min(h, w) // 6), maxLineGap=20)
    if lines is None:
        return None
    best, best_length = None, 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length > best_length:
            best_length = length
            best = (int(x1), int(y1), int(x2), int(y2))
    return best


def endpoint_skin_score(frame_bgr: np.ndarray, point: Tuple[int, int], radius: int = 6) -> float:
    height, width = frame_bgr.shape[:2]
    x0, y0 = int(point[0]), int(point[1])
    r = max(2, int(radius))
    x1, y1 = max(0, x0 - r), max(0, y0 - r)
    x2, y2 = min(width, x0 + r + 1), min(height, y0 + r + 1)
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    mask = skin_mask(patch)
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    cx, cy = x0 - x1, y0 - y1
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return float(np.count_nonzero(mask[circle])) / max(1, int(np.count_nonzero(circle)))


def pick_swab_tip(previous_tip: Optional[Point], p1: Tuple[int, int], p2: Tuple[int, int],
                  fallback_center: Tuple[int, int], frame_bgr: np.ndarray) -> Tuple[int, int]:
    """选择更可能接触皮肤的端点，并利用上一帧保持时间连续性。"""
    score1, score2 = endpoint_skin_score(frame_bgr, p1), endpoint_skin_score(frame_bgr, p2)
    if abs(score1 - score2) > 0.05:
        return p1 if score1 > score2 else p2
    if previous_tip is not None:
        d1 = (p1[0] - previous_tip[0]) ** 2 + (p1[1] - previous_tip[1]) ** 2
        d2 = (p2[0] - previous_tip[0]) ** 2 + (p2[1] - previous_tip[1]) ** 2
        return p1 if d1 <= d2 else p2
    d1 = (p1[0] - fallback_center[0]) ** 2 + (p1[1] - fallback_center[1]) ** 2
    d2 = (p2[0] - fallback_center[0]) ** 2 + (p2[1] - fallback_center[1]) ** 2
    return p1 if d1 <= d2 else p2


# ---------------------------------------------------------------------------
# 5. trocar.pt：YOLO 框内 Hough 线 + 皮肤轮廓求插入点
# ---------------------------------------------------------------------------

def find_insertion_point(line: Tuple[int, int, int, int], skin_contour: np.ndarray) -> Optional[Tuple[int, int]]:
    """沿 trocar 线段朝皮肤轮廓搜索交点。"""
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0:
        return None
    dx, dy = dx / length, dy / length
    moments = cv2.moments(skin_contour)
    if moments["m00"] == 0:
        return None
    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    d1 = (x1 - center[0]) ** 2 + (y1 - center[1]) ** 2
    d2 = (x2 - center[0]) ** 2 + (y2 - center[1]) ** 2
    if d1 < d2:
        start, direction = (x1, y1), (dx, dy)
    else:
        start, direction = (x2, y2), (-dx, -dy)
    for distance in range(0, 300, 2):
        point = (int(start[0] + distance * direction[0]), int(start[1] + distance * direction[1]))
        if -10 < cv2.pointPolygonTest(skin_contour, point, True) < 10:
            return point
    return None


# ---------------------------------------------------------------------------
# 6. 手部识别：MediaPipe Hands（项目没有手部 YOLO 权重）
# ---------------------------------------------------------------------------

class MediaPipeHandDetector:
    """MediaPipe 手部关键点/骨架检测。"""

    def __init__(self, max_num_hands: int = 2, model_complexity: int = 1,
                 min_detection_confidence: float = 0.25,
                 min_tracking_confidence: float = 0.25) -> None:
        import mediapipe as mp
        self.mp = mp
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: np.ndarray, draw: bool = True):
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        output = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if draw and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_style = self.mp_styles.get_default_hand_landmarks_style()
                connection_style = self.mp_styles.get_default_hand_connections_style()
                for spec in landmark_style.values():
                    spec.thickness, spec.circle_radius = 8, 8
                for spec in connection_style.values():
                    spec.thickness = 6
                self.mp_drawing.draw_landmarks(
                    output, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_style, connection_style,
                )
        return output, results

    def close(self) -> None:
        self.hands.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
