"""
手部骨架识别程序
使用 MediaPipe 进行手部关键点检测和骨架绘制
"""

import os
os.environ['GLOG_minloglevel'] = '2'  # 减少日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志

print("开始导入库...")
import cv2
print("cv2 导入成功")
import mediapipe as mp
print("mediapipe 导入成功")
import numpy as np
print("numpy 导入成功")

class HandSkeletonDetector:
    def __init__(self):
        # 初始化 MediaPipe 手部检测
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 配置手部检测参数
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,                # 最多检测2只手
            model_complexity=1,             # 使用完整模型：0(lite), 1(full) - 提高准确度
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25
        )
    
    def process_frame(self, frame):
        """
        处理单帧图像，检测手部骨架
        
        Args:
            frame: 输入的BGR图像
            
        Returns:
            processed_frame: 绘制了手部骨架的图像
            results: 检测结果
        """
        # 转换颜色空间 BGR -> RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 提高性能：标记图像为不可写
        image_rgb.flags.writeable = False
        
        # 检测手部
        results = self.hands.process(image_rgb)
        
        # 恢复图像为可写
        image_rgb.flags.writeable = True
        
        # 转换回BGR用于显示
        processed_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 如果检测到手部，绘制骨架
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 获取手部类别（左手/右手）
                handedness = results.multi_handedness[hand_idx].classification[0].label
                
                # 获取默认彩色样式并加大加粗
                default_landmark_style = self.mp_drawing_styles.get_default_hand_landmarks_style()
                default_connection_style = self.mp_drawing_styles.get_default_hand_connections_style()
                
                # 修改样式参数 - 保持彩色但加大加粗
                for landmark_spec in default_landmark_style.values():
                    landmark_spec.thickness = 8      # 节点粗细（加粗）
                    landmark_spec.circle_radius = 8  # 节点半径（加大）
                
                for connection_spec in default_connection_style.values():
                    connection_spec.thickness = 6    # 连接线粗细（加粗）
                
                # 绘制手部骨架连接线（使用彩色样式）
                self.mp_drawing.draw_landmarks(
                    processed_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    default_landmark_style,      # 使用彩色节点样式
                    default_connection_style     # 使用彩色连接线样式
                )
                
                # 暂时不显示左右手标签
                # # 在手腕位置显示左手/右手标签
                # h, w, _ = processed_frame.shape
                # wrist = hand_landmarks.landmark[0]  # 手腕关键点
                # cx, cy = int(wrist.x * w), int(wrist.y * h)
                #
                # # 绘制标签背景
                # label = f"{handedness}"
                # (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # cv2.rectangle(processed_frame,
                #             (cx - 10, cy - text_h - 20),
                #             (cx + text_w + 10, cy - 10),
                #             (0, 255, 0), -1)
                #
                # # 绘制文字
                # cv2.putText(processed_frame, label,
                #           (cx, cy - 15),
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 在视频右上角添加"手部动作识别"标题
        _, w, _ = processed_frame.shape
        title_text = "手部动作识别"

        # 使用 PIL 来绘制中文（OpenCV 对中文支持不好）
        from PIL import Image, ImageDraw, ImageFont

        # 转换为 PIL 图像
        pil_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 尝试加载中文字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype("simhei.ttf", 100)  # 黑体，字号40
        except:
            try:
                font = ImageFont.truetype("msyh.ttf", 40)  # 微软雅黑
            except:
                font = ImageFont.load_default()

        # 获取文字尺寸
        bbox = draw.textbbox((0, 0), title_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 计算右上角位置（留20像素边距）
        x = w - text_w - 20
        y = 20

        # 绘制半透明背景
        padding = 10
        draw.rectangle(
            [(x - padding, y - padding), (x + text_w + padding, y + text_h + padding)],
            fill=(0, 0, 0, 180)  # 黑色半透明背景
        )

        # 绘制文字（白色）
        draw.text((x, y), title_text, font=font, fill=(255, 255, 255, 255))

        # 转换回 OpenCV 格式
        processed_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return processed_frame, results
    
    def process_video(self, video_path, output_path=None, show_window=True):
        """
        处理视频文件，检测手部骨架
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            show_window: 是否显示窗口
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息：")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {total_frames}")
        
        # 创建视频写入器（如果需要保存）
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"输出视频: {output_path}")
        
        frame_count = 0
        
        print("\n开始处理视频...")
        print("按 'q' 键退出，按 'p' 键暂停/继续")
        
        paused = False
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("\n视频处理完成！")
                    break
                
                # 处理当前帧
                processed_frame, results = self.process_frame(frame)
                
                # 添加帧信息
                info_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(processed_frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示检测到的手数
                if results.multi_hand_landmarks:
                    hands_count = len(results.multi_hand_landmarks)
                    hands_text = f"Hands: {hands_count}"
                    cv2.putText(processed_frame, hands_text, (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 保存处理后的帧
                if writer:
                    writer.write(processed_frame)
                
                # 显示窗口
                if show_window:
                    cv2.imshow('Hand Skeleton Detection', processed_frame)
                
                frame_count += 1
                
                # 显示进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"进度: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # 处理键盘输入
            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户中断处理")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("暂停" if paused else "继续")
        
        # 释放资源
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        
        print(f"\n处理完成！共处理 {frame_count} 帧")
    
    def __del__(self):
        """析构函数，释放资源"""
        self.hands.close()


def main():
    """主函数"""
    try:
        print("程序启动...")
        
        # 输入视频路径
        video_path = "ai_system_v2/手部.mp4"
        
        # 检查视频文件是否存在
        import os
        if not os.path.exists(video_path):
            print(f"错误：找不到视频文件 {video_path}")
            print(f"当前目录：{os.getcwd()}")
            print(f"目录下的文件：")
            for f in os.listdir('.'):
                if f.endswith('.mp4'):
                    print(f"  - {f}")
            return
        
        # 输出视频路径（可选）
        output_path = "trocar_hand_skeleton.mp4"
        
        print("=" * 50)
        print("手部骨架识别程序")
        print("=" * 50)
        
        # 创建检测器
        print("正在初始化手部检测器...")
        detector = HandSkeletonDetector()
        print("检测器初始化完成")
        
        # 处理视频
        detector.process_video(
            video_path=video_path,
            output_path=output_path,
            show_window=True
        )
        
        print("\n程序结束")
        
    except Exception as e:
        print(f"程序出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
