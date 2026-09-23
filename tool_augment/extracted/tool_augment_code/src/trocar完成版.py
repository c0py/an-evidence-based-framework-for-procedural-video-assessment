import cv2
import numpy as np
from ultralytics import YOLO
import math

def detect_tubes_and_navel(model_path, video_path, output_path="result.mp4"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x, center_y = width // 2, height // 2
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测肚脐
        navel = detect_navel(frame)
        if navel:
            cv2.circle(frame, navel, 20, (0, 255, 255), 3)
            cv2.putText(frame, "Navel", (navel[0]-20, navel[1]-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 检测导管并收集端点
        tube_points = []
        results = model(frame, conf=0.1)
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 200, 300)
                    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 69, minLineLength=200, maxLineGap=100)
                    
                    if lines is not None:
                        best_line = None
                        max_length = 0
                        
                        for line in lines:
                            x1l, y1l, x2l, y2l = line[0]
                            angle = np.abs(np.arctan2(y2l-y1l, x2l-x1l) * 180 / np.pi)
                            
                            if not (angle < 10 or 80 < angle < 100 or angle > 170):
                                length = np.sqrt((x2l-x1l)**2 + (y2l-y1l)**2)
                                if length > max_length:
                                    max_length = length
                                    best_line = (x1+x1l, y1+y1l, x1+x2l, y1+y2l)
                        
                        if best_line:
                            cv2.line(frame, (best_line[0], best_line[1]), 
                                    (best_line[2], best_line[3]), (0, 0, 255), 2)
                            
                            if (best_line[0] - center_x)**2 + (best_line[1] - center_y)**2 < \
                               (best_line[2] - center_x)**2 + (best_line[3] - center_y)**2:
                                point = (best_line[0], best_line[1])
                            else:
                                point = (best_line[2], best_line[3])
                            
                            cv2.circle(frame, point, 12, (255, 0, 0), -1)
                            cv2.circle(frame, point, 12, (255, 255, 255), 2)
                            tube_points.append(point)
        
        # 如果检测到3个导管端点且有肚脐，进行角度分析
        if len(tube_points) == 3 and navel:
            analyze_height_angles(frame, tube_points, navel)
        
        out.write(frame)
        cv2.imshow('Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def analyze_height_angles(frame, points, navel):
    """分析从肚脐点出发的边与高线的夹角"""
    if len(points) != 3:
        return
    
    p1, p2, p3 = points
    
    # 找出距离肚脐最近的点作为底部顶点
    distances = [distance(navel, p) for p in points]
    closest_idx = distances.index(min(distances))
    bottom_vertex = points[closest_idx]
    
    # 其他两个点作为上边的两个顶点
    other_points = [p for i, p in enumerate(points) if i != closest_idx]
    left_vertex, right_vertex = other_points[0], other_points[1]
    
    # 绘制三角形
    cv2.line(frame, bottom_vertex, left_vertex, (255, 0, 255), 2)
    cv2.line(frame, bottom_vertex, right_vertex, (255, 0, 255), 2)
    cv2.line(frame, left_vertex, right_vertex, (255, 0, 255), 4)  # 底边加粗
    
    # 计算底边的垂足（高线的终点）
    foot_point = perpendicular_foot(bottom_vertex, left_vertex, right_vertex)
    
    # 绘制高线
    if foot_point:
        cv2.line(frame, bottom_vertex, foot_point, (255, 255, 0), 3)  # 青色高线
        cv2.circle(frame, foot_point, 6, (255, 255, 0), -1)
        
        # 计算两个关键角度：左边与高线的夹角，右边与高线的夹角
        left_angle = calculate_angle(left_vertex, bottom_vertex, foot_point)
        right_angle = calculate_angle(right_vertex, bottom_vertex, foot_point)
        
        # 理想角度设置
        ideal_angle = 60.0
        angle_tolerance = 10.0
        
        # 分析左侧角度
        left_diff = abs(left_angle - ideal_angle)
        if left_diff <= angle_tolerance:
            left_color = (0, 255, 0)  # 绿色
            left_status = "GOOD"
        elif left_diff <= 20:
            left_color = (0, 255, 255)  # 黄色
            left_status = "OK"
        else:
            left_color = (0, 0, 255)  # 红色
            left_status = "ADJUST"
        
        # 分析右侧角度
        right_diff = abs(right_angle - ideal_angle)
        if right_diff <= angle_tolerance:
            right_color = (0, 255, 0)
            right_status = "GOOD"
        elif right_diff <= 20:
            right_color = (0, 255, 255)
            right_status = "OK"
        else:
            right_color = (0, 0, 255)
            right_status = "ADJUST"
        
        # 绘制角度弧线和标注
        draw_angle_arc(frame, bottom_vertex, left_vertex, foot_point, left_color, 30)
        draw_angle_arc(frame, bottom_vertex, foot_point, right_vertex, right_color, 40)
        
        # 显示左侧角度
        left_text_pos = get_angle_text_position(bottom_vertex, left_vertex, foot_point, 50)
        cv2.putText(frame, f"{left_angle:.1f}deg", left_text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
        cv2.putText(frame, left_status, (left_text_pos[0], left_text_pos[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)
        
        # 显示右侧角度
        right_text_pos = get_angle_text_position(bottom_vertex, foot_point, right_vertex, 50)
        cv2.putText(frame, f"{right_angle:.1f}deg", right_text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)
        cv2.putText(frame, right_status, (right_text_pos[0], right_text_pos[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)
        
        # 整体评估
        good_angles = sum(1 for angle in [left_angle, right_angle] 
                         if abs(angle - ideal_angle) <= angle_tolerance)
        
        if good_angles == 2:
            overall_status = "PERFECT POSITIONING"
            overall_color = (0, 255, 0)
        elif good_angles == 1:
            overall_status = "NEEDS ADJUSTMENT"
            overall_color = (0, 255, 255)
        else:
            overall_status = "POOR POSITIONING"
            overall_color = (0, 0, 255)
        
        cv2.putText(frame, overall_status, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, overall_color, 2)
        
        # 显示角度信息
        avg_angle = (left_angle + right_angle) / 2
        info_text = f"L:{left_angle:.1f} R:{right_angle:.1f} Avg:{avg_angle:.1f} (Target:60)"
        cv2.putText(frame, info_text, (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_angle_arc(frame, vertex, p1, p2, color, radius):
    """绘制角度的弧线标记"""
    # 计算两个向量的角度
    v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]])
    v2 = np.array([p2[0] - vertex[0], p2[1] - vertex[1]])
    
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    
    # 确保角度顺序正确
    if angle2 < angle1:
        angle1, angle2 = angle2, angle1
    
    # 如果角度差超过180度，调整
    if angle2 - angle1 > math.pi:
        angle1 += 2 * math.pi
        angle1, angle2 = angle2, angle1
    
    # 绘制弧线
    start_angle = int(math.degrees(angle1))
    end_angle = int(math.degrees(angle2))
    
    cv2.ellipse(frame, vertex, (radius, radius), 0, start_angle, end_angle, color, 2)

def get_angle_text_position(vertex, p1, p2, offset):
    """计算角度标注文字的位置"""
    # 计算角度的平分线方向
    v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]])
    v2 = np.array([p2[0] - vertex[0], p2[1] - vertex[1]])
    
    # 归一化向量
    v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
    v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
    
    # 计算平分线方向
    bisector = v1 + v2
    if np.linalg.norm(bisector) > 0:
        bisector = bisector / np.linalg.norm(bisector)
    
    # 计算文字位置
    text_x = int(vertex[0] + bisector[0] * offset)
    text_y = int(vertex[1] + bisector[1] * offset)
    
    return (text_x, text_y)

def calculate_angle(p1, vertex, p2):
    """计算角度"""
    v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]])
    v2 = np.array([p2[0] - vertex[0], p2[1] - vertex[1]])
    
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    
    if len1 == 0 or len2 == 0:
        return 0
    
    cos_angle = np.dot(v1, v2) / (len1 * len2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle

def distance(p1, p2):
    """计算两点间距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def perpendicular_foot(point, line_start, line_end):
    """计算点到线段的垂足"""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    line_length_sq = (x2-x1)**2 + (y2-y1)**2
    if line_length_sq == 0:
        return line_start
    
    t = ((x0-x1)*(x2-x1) + (y0-y1)*(y2-y1)) / line_length_sq
    t = max(0, min(1, t))
    
    foot_x = int(x1 + t * (x2 - x1))
    foot_y = int(y1 + t * (y2 - y1))
    
    return (foot_x, foot_y)

def detect_navel(frame):
    """检测肚脐位置"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        best_contour = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 5000:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > best_circularity and circularity > 0.5:
                        best_circularity = circularity
                        best_contour = contour
        
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    
    return None

if __name__ == "__main__":
    detect_tubes_and_navel("trocar.pt", "findarm.mp4")