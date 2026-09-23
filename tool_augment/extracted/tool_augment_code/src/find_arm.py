import cv2
import numpy as np
from ultralytics import YOLO

def detect_tubes_with_insertion_points(model_path, video_path, output_path="tubes_with_points.mp4"):
    """检测导管并标记插入点"""
    
    # 最佳参数
    MARGIN = 0
    CANNY_LOW = 200
    CANNY_HIGH = 300
    HOUGH_THRESH = 69
    MIN_LENGTH = 200
    MAX_GAP = 100
    ANGLE_FILTER = 10
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 先检测皮肤区域（用于找插入点）
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # 获取皮肤轮廓
        skin_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        skin_contour = max(skin_contours, key=cv2.contourArea) if skin_contours else None
        
        results = model(frame, conf=0.1)
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 画trocar框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # ROI区域
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    # 边缘检测
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
                    
                    # 检测直线
                    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                                          minLineLength=MIN_LENGTH, maxLineGap=MAX_GAP)
                    
                    if lines is not None:
                        # 存储最佳线段
                        best_line = None
                        max_length = 0
                        
                        for line in lines:
                            x1l, y1l, x2l, y2l = line[0]
                            
                            # 角度过滤
                            angle = np.abs(np.arctan2(y2l-y1l, x2l-x1l) * 180 / np.pi)
                            if not (angle < ANGLE_FILTER or 
                                   90-ANGLE_FILTER < angle < 90+ANGLE_FILTER or 
                                   angle > 180-ANGLE_FILTER):
                                
                                length = np.sqrt((x2l-x1l)**2 + (y2l-y1l)**2)
                                if length > max_length:
                                    max_length = length
                                    best_line = (x1+x1l, y1+y1l, x1+x2l, y1+y2l)
                        
                        # 画最长的线
                        if best_line:
                            cv2.line(frame, (best_line[0], best_line[1]), 
                                    (best_line[2], best_line[3]), (0, 0, 255), 2)
                            
                            # 找插入点
                            if skin_contour is not None:
                                insertion_point = find_insertion_point(best_line, skin_contour)
                                if insertion_point:
                                    # 画蓝色圆点（像示例图一样）
                                    cv2.circle(frame, insertion_point, 12, (255, 0, 0), -1)
                                    # 白色边框
                                    cv2.circle(frame, insertion_point, 12, (255, 255, 255), 2)
        
        out.write(frame)
        cv2.imshow('Tube Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def find_insertion_point(line, skin_contour):
    """找到导管与皮肤的交点"""
    x1, y1, x2, y2 = line
    
    # 延长线段
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return None
    
    # 单位方向向量
    dx /= length
    dy /= length
    
    # 从线段中点向两端搜索
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # 确定搜索方向（朝向皮肤中心）
    M = cv2.moments(skin_contour)
    if M["m00"] != 0:
        skin_center_x = int(M["m10"] / M["m00"])
        skin_center_y = int(M["m01"] / M["m00"])
        
        # 判断哪个方向指向皮肤中心
        dist1 = (x1 - skin_center_x)**2 + (y1 - skin_center_y)**2
        dist2 = (x2 - skin_center_x)**2 + (y2 - skin_center_y)**2
        
        if dist1 < dist2:
            # 从点1向点2方向搜索
            start_x, start_y = x1, y1
            search_dx, search_dy = dx, dy
        else:
            # 从点2向点1方向搜索
            start_x, start_y = x2, y2
            search_dx, search_dy = -dx, -dy
    else:
        return None
    
    # 沿着导管方向搜索交点
    for t in range(0, 300, 2):
        test_x = int(start_x + t * search_dx)
        test_y = int(start_y + t * search_dy)
        
        # 检查是否在皮肤边界上
        dist = cv2.pointPolygonTest(skin_contour, (test_x, test_y), True)
        
        # 从外到内穿过皮肤边界
        if -10 < dist < 10:  # 接近边界
            return (test_x, test_y)
    
    return None

# 简化版：直接在导管端点画圆
def detect_tubes_simple_points(model_path, video_path, output_path="tubes_simple_points.mp4"):
    """简化版：在导管靠近皮肤的端点画圆"""
    
    # 最佳参数
    MARGIN = 0
    CANNY_LOW = 200
    CANNY_HIGH = 300
    HOUGH_THRESH = 69
    MIN_LENGTH = 200
    MAX_GAP = 100
    ANGLE_FILTER = 10
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 图像中心（通常皮肤在中心）
        center_x = width // 2
        center_y = height // 2
        
        results = model(frame, conf=0.1)
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 画trocar框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # ROI区域
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    # 边缘检测
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
                    
                    # 检测直线
                    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                                          minLineLength=MIN_LENGTH, maxLineGap=MAX_GAP)
                    
                    if lines is not None:
                        best_line = None
                        max_length = 0
                        
                        for line in lines:
                            x1l, y1l, x2l, y2l = line[0]
                            
                            # 角度过滤
                            angle = np.abs(np.arctan2(y2l-y1l, x2l-x1l) * 180 / np.pi)
                            if not (angle < ANGLE_FILTER or 
                                   90-ANGLE_FILTER < angle < 90+ANGLE_FILTER or 
                                   angle > 180-ANGLE_FILTER):
                                
                                length = np.sqrt((x2l-x1l)**2 + (y2l-y1l)**2)
                                if length > max_length:
                                    max_length = length
                                    best_line = (x1+x1l, y1+y1l, x1+x2l, y1+y2l)
                        
                        if best_line:
                            # 画线
                            cv2.line(frame, (best_line[0], best_line[1]), 
                                    (best_line[2], best_line[3]), (0, 0, 255), 2)
                            
                            # 选择靠近图像中心的端点作为插入点
                            dist1 = (best_line[0] - center_x)**2 + (best_line[1] - center_y)**2
                            dist2 = (best_line[2] - center_x)**2 + (best_line[3] - center_y)**2
                            
                            if dist1 < dist2:
                                insertion_x, insertion_y = best_line[0], best_line[1]
                            else:
                                insertion_x, insertion_y = best_line[2], best_line[3]
                            
                            # 画蓝色圆点
                            cv2.circle(frame, (insertion_x, insertion_y), 12, (255, 0, 0), -1)
                            cv2.circle(frame, (insertion_x, insertion_y), 12, (255, 255, 255), 2)
        
        out.write(frame)
        cv2.imshow('Tube Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 使用精确版（检测皮肤交点）
    # detect_tubes_with_insertion_points("best.pt", "findarm.mp4")
    
    # 或使用简化版（直接标记端点）
    detect_tubes_simple_points("trocar.pt", "findarm.mp4")