# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :hand_video.py
# @Time      :2025-07-24 20:48
# @Author    :lenovo
import cv2
import mediapipe as mp

# 初始化 Mediapipe 手部模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# 初始化绘图模块
mp_drawing = mp.solutions.drawing_utils

# 打开视频捕捉
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从 BGR 转换为 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # 将 RGB 图像转换回 BGR，以便使用 OpenCV 显示
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 绘制手部标记
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示结果
    cv2.imshow('Hand Tracking', image)

    # 保存带标记的图像
    if cv2.waitKey(1) & 0xFF == ord('s'):  # 按 's' 键保存图像
        cv2.imwrite('hand_tracking_result.png', image)
        print("图像已保存！")

    if cv2.waitKey(1) & 0xFF == 27:  # 按 'ESC' 键退出
        break

cap.release()
cv2.destroyAllWindows()
