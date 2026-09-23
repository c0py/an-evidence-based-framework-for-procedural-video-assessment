#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import cv2
from pathlib import Path

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None


def train_with_existing_dataset(yaml_path: str, epochs=100, batch_size=8, image_size=640, device='auto', project='runs/train_swab', name='exp'):
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. pip install ultralytics")
    print(f"加载配置: {yaml_path}")
    model = YOLO('yolov8n.pt')  # nano 版本，速度快
    print("开始训练模型...")
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        patience=20,
        device=device,
        project=project,
        name=name,
    )
    print("模型训练完成！开始验证...")
    model.val(data=yaml_path, imgsz=image_size, batch=batch_size, device=device)
    print("验证完成，导出 ONNX ...")
    try:
        model.export(format='onnx')
        print("模型已导出为 ONNX 格式")
    except Exception as e:
        print(f"导出失败（忽略）：{e}")
    return model


def ensure_dirs(base: str):
    for d in [
        f"{base}/images/train",
        f"{base}/images/val",
        f"{base}/labels/train",
        f"{base}/labels/val",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ============ Annotation helpers ============

def ensure_annot_dirs(root: str):
    Path(os.path.join(root, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(root, 'labels')).mkdir(parents=True, exist_ok=True)


def abs_rect_to_norm(x: int, y: int, w: int, h: int, W: int, H: int):
    # clamp
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    if w <= 0 or h <= 0:
        return 0.5, 0.5, 0.0, 0.0
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / W, cy / H, w / W, h / H


def save_yolo_label(txt_path: str, boxes):
    with open(txt_path, 'w', encoding='utf-8') as f:
        for cls_id, cx, cy, ww, hh in boxes:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")


def annotate_from_video(video_path: str, out_root: str, extract_every: int = 5, class_id: int = 0):
    """简单标注器：
    - 从视频每 N 帧抽 1 帧
    - 使用 OpenCV selectROIs 一次性多框选择
    - 保存有框的帧到 out_root/images，并保存对应 YOLO 标签到 out_root/labels
    """
    ensure_annot_dirs(out_root)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % max(1, extract_every) != 0:
            idx += 1
            continue
        H, W = frame.shape[:2]
        win = f"标注 - 第 {idx} 帧 (空格/回车确认当前框，Esc结束本帧)"
        boxes = []
        while True:
            r = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
            # r: (x, y, w, h); 若按 Esc 或未选则 w/h 为 0
            try:
                x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            except Exception:
                break
            if w <= 0 or h <= 0:
                break
            cx, cy, ww, hh = abs_rect_to_norm(x, y, w, h, W, H)
            if ww > 0 and hh > 0:
                boxes.append((class_id, cx, cy, ww, hh))
        cv2.destroyWindow(win)
        if len(boxes) > 0:
            img_name = f"frame_{idx:06d}.jpg"
            img_path = os.path.join(out_root, 'images', img_name)
            txt_path = os.path.join(out_root, 'labels', img_name.replace('.jpg', '.txt'))
            cv2.imwrite(img_path, frame)
            save_yolo_label(txt_path, boxes)
            saved += 1
        idx += 1
    cap.release()
    print(f"标注完成：保存了 {saved} 张已标注图像至 {out_root}/images")


def split_dataset(root: str, val_ratio: float = 0.1, seed: int = 42):
    import random
    random.seed(seed)
    images_dir = Path(root) / 'images'
    labels_dir = Path(root) / 'labels'
    all_imgs = sorted([p for p in images_dir.glob('*.jpg')])
    # 仅保留有对应 label 的图片
    imgs = [p for p in all_imgs if (labels_dir / (p.stem + '.txt')).exists()]
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio)) if len(imgs) > 0 else 0
    val = set(imgs[:n_val])

    # 目标目录
    ensure_dirs(root)
    def copy_pair(ip: Path, split: str):
        lp = labels_dir / (ip.stem + '.txt')
        dst_img = Path(root) / 'images' / split / ip.name
        dst_lbl = Path(root) / 'labels' / split / lp.name
        dst_img.write_bytes(ip.read_bytes())
        dst_lbl.write_bytes(lp.read_bytes())

    for ip in imgs:
        split = 'val' if ip in val else 'train'
        copy_pair(ip, split)
    print(f"切分完成：train={len(imgs)-len(val)}, val={len(val)}")


def build_data_yaml(root: str, classes):
    root_abs = os.path.abspath(root)
    yaml_text = "\n".join([
        f"path: {root_abs}",
        "train: images/train",
        "val: images/val",
        f"names: {classes}",
    ])
    ypath = os.path.join(root, 'dataset.yaml')
    with open(ypath, 'w', encoding='utf-8') as f:
        f.write(yaml_text + "\n")
    print(f"写入配置: {ypath}\n{yaml_text}")
    return ypath


def convert_json_to_yolo(json_file: str, video_file: str = None, output_dir: str = 'swab_dataset', classes=None) -> str:
    """将JSON标注转换为YOLO数据集。JSON 结构示例（与 trocar 类似）：
    [
      {"frame": 12, "detections": [
         {"center": [x, y], "w": w_abs, "h": h_abs, "cls": 0}
      ]},
      ...
    ]
    Notes:
      - center, w, h 为像素绝对值；cls 可选，缺省为 0。
      - 每5帧分一部分到 val。
    """
    ensure_dirs(output_dir)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 推断图像尺寸
    if video_file and os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        width, height = 1280, 720
        print(f"警告: 未提供视频或无法打开，默认尺寸 {width}x{height}")

    # 若提供了视频，则逐帧抽图
    cap = None
    if video_file and os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)

    train_count = 0
    val_count = 0

    for item in data:
        frame_num = int(item.get('frame', -1))
        dets = item.get('detections', [])
        if frame_num < 0 or not dets:
            continue
        is_val = (frame_num % 5 == 0)
        split = 'val' if is_val else 'train'
        img_path = f"{output_dir}/images/{split}/frame_{frame_num:06d}.jpg"
        label_path = f"{output_dir}/labels/{split}/frame_{frame_num:06d}.txt"

        # 导出当前帧图像
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(img_path, frame)
        else:
            # 未提供视频，跳过图像导出（要求已有同名图像）
            pass

        # 写标签
        with open(label_path, 'w', encoding='utf-8') as lf:
            for det in dets:
                cx = det['center'][0] / width
                cy = det['center'][1] / height
                w = det['w'] / width
                h = det['h'] / height
                cls = int(det.get('cls', 0))
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                lf.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        if is_val:
            val_count += 1
        else:
            train_count += 1

    if cap is not None:
        cap.release()

    # 写 YAML
    classes = classes or ['swab']
    yaml_content = f"""
# Swab 数据集
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# 类别
auto: False
names: {classes}
"""
    yaml_path = f"{output_dir}/dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yf.write(yaml_content)

    print(f"创建数据集：train={train_count}, val={val_count}")
    print(f"数据集配置文件: {yaml_path}")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(description='Swab YOLO annotation + training (trocar-style)')
    # 标注与数据集构建
    ap.add_argument('--stage', type=str, default='all', choices=['annotate', 'train', 'all'], help='只标注/只训练/两者都做')
    ap.add_argument('--video', type=str, default='圆形消毒.mp4', help='视频路径（用于标注与按帧导出图片）')
    ap.add_argument('--extract_every', type=int, default=5, help='每 N 帧抽 1 帧用于标注')
    ap.add_argument('--dataset_root', type=str, default='swab_dataset', help='数据集根目录')
    ap.add_argument('--val_ratio', type=float, default=0.1, help='验证集占比')
    ap.add_argument('--classes', type=str, default='swab', help='类别名，逗号分隔（标注默认使用第一个类别）')

    # 训练
    ap.add_argument('--yaml', type=str, default='swab_dataset/dataset.yaml', help='如已存在的数据集配置路径')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--project', type=str, default='runs/train_swab')
    ap.add_argument('--name', type=str, default='exp')
    ap.add_argument('--mode', type=str, choices=['existing', 'json', 'both'], default='existing',
                    help='保留兼容：existing=使用现有数据集, json=由JSON创建新数据集, both=两者结合(优先新建)')
    ap.add_argument('--json', type=str, default='swab_annotations.json', help='JSON标注文件路径（兼容模式）')

    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    assert len(classes) > 0, 'classes 不能为空'

    # 标注阶段
    if args.stage in ('annotate', 'all'):
        annotate_from_video(args.video, args.dataset_root, extract_every=args.extract_every, class_id=0)
        split_dataset(args.dataset_root, val_ratio=args.val_ratio, seed=42)
        yaml_path = build_data_yaml(args.dataset_root, classes)
    else:
        # 纯训练阶段
        if args.mode in ['json', 'both']:
            yaml_path = convert_json_to_yolo(args.json, args.video, output_dir=args.dataset_root, classes=classes)
        else:
            yaml_path = args.yaml

    # 训练阶段
    if args.stage in ('train', 'all'):
        train_with_existing_dataset(
            yaml_path=yaml_path,
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
        )

    print('处理完成！')


if __name__ == '__main__':
    main()

