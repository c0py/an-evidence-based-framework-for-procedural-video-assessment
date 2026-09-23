# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :hand.py.py
# @Time      :2025-07-24 20:46
# @Author    :lenovo
import cv2
import mediapipe as mp

# 初始化 Mediapipe 手部模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 初始化绘图模块
mp_drawing = mp.solutions.drawing_utils

# 打开视频文件（改为视频文件而不是摄像头）
cap = cv2.VideoCapture("ai_system_v2/手部.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从 BGR 转换为 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # 绘制手部标记
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示结果
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # 按 'ESC' 键退出
        break

cap.release()
cv2.destroyAllWindows()
