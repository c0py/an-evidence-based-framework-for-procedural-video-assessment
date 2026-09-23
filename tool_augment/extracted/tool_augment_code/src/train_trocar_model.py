#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
import cv2
from ultralytics import YOLO

def train_with_existing_dataset(yaml_path, epochs=100, batch_size=8, image_size=640, device='auto'):
    """
    使用现有的YOLO数据集训练模型
    
    参数:
        yaml_path: YOLO数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批量大小
        image_size: 图像尺寸
        device: 训练设备 ('cpu', 'cuda', 'auto')
    """
    print(f"加载配置: {yaml_path}")
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用nano版本，训练更快
    
    print(f"开始训练模型...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        patience=20,
        device=device
    )
    
    print(f"模型训练完成！")
    
    # 导出模型
    model.export(format='onnx')
    print(f"模型已导出为ONNX格式")
    
    return model

def convert_json_to_yolo(json_file, video_file=None, output_dir='trocar_dataset_new'):
    """
    将JSON标注转换为YOLO格式
    
    参数:
        json_file: JSON标注文件路径
        video_file: 对应的视频文件路径(用于提取帧)
        output_dir: 输出目录
    """
    print(f"从 {json_file} 创建YOLO格式数据集...")
    
    # 创建目录结构
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    
    # 加载JSON数据
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 确定图像尺寸
    if video_file and os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        # 默认尺寸
        width, height = 1280, 720
        print(f"警告: 未提供视频文件，使用默认尺寸 {width}x{height}")
    
    # 处理每一帧
    train_count = 0
    val_count = 0
    
    for frame_data in data:
        frame_num = frame_data["frame"]
        detections = frame_data["detections"]
        
        if not detections:
            continue
        
        # 每5帧作为验证集
        is_val = frame_num % 5 == 0
        
        # 帧文件名
        if is_val:
            frame_filename = f"{output_dir}/images/val/frame_{frame_num:06d}.jpg"
            label_filename = f"{output_dir}/labels/val/frame_{frame_num:06d}.txt"
            val_count += 1
        else:
            frame_filename = f"{output_dir}/images/train/frame_{frame_num:06d}.jpg"
            label_filename = f"{output_dir}/labels/train/frame_{frame_num:06d}.txt"
            train_count += 1
        
        # 如果有视频，提取帧
        if video_file and os.path.exists(video_file):
            cap = cv2.VideoCapture(video_file)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_filename, frame)
            cap.release()
        
        # 写入标签文件
        with open(label_filename, 'w') as f:
            for det in detections:
                # 计算YOLOv8格式的标注 (类别索引, 中心点x, 中心点y, 宽度, 高度)
                x_center = det["center"][0] / width
                y_center = det["center"][1] / height
                w = det["w"] / width
                h = det["h"] / height
                
                # 写入标签文件 (确保值在0-1范围内)
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    # 创建数据集配置文件
    yaml_content = f"""
# Trocar数据集配置
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# 类别
nc: 1
names: ['trocar']
"""
    
    yaml_path = f"{output_dir}/dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"创建了 {train_count} 个训练样本和 {val_count} 个验证样本")
    print(f"数据集配置文件: {yaml_path}")
    
    return yaml_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用现有标注训练Trocar检测模型")
    parser.add_argument('--json', type=str, default='trocar_detections_20250726_100515.json', 
                        help='JSON标注文件路径')
    parser.add_argument('--video', type=str, default='findarm.mp4',
                        help='视频文件路径(用于从JSON创建新数据集时提取帧)')
    parser.add_argument('--yaml', type=str, default='my_trocar_dataset/yolo/trocar_dataset.yaml',
                        help='现有YOLO数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, help='批量大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', type=str, default='cpu', help='训练设备(cpu/cuda/auto)')
    parser.add_argument('--mode', type=str, choices=['existing', 'json', 'both'], default='existing',
                        help='训练模式: existing=使用现有数据集, json=使用JSON创建新数据集, both=两者结合')
    
    args = parser.parse_args()
    
    if args.mode in ['json', 'both']:
        # 从JSON创建数据集
        yaml_path = convert_json_to_yolo(args.json, args.video)
        
        if args.mode == 'both' and os.path.exists(args.yaml):
            print("合并现有数据集和新数据集...")
            # 这里可以添加合并数据集的代码
            pass
    else:
        yaml_path = args.yaml
    
    # 训练模型
    model = train_with_existing_dataset(
        yaml_path=yaml_path,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device
    )
    
    print("处理完成！") 