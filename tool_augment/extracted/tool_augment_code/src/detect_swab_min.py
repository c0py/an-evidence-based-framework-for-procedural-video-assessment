import os
import cv2
import glob
import argparse
from pathlib import Path
from ultralytics import YOLO
# -------------------------- 修正后的兼容性补丁 --------------------------
import torch
import os  # 确保导入 os 模块

# 关键修复：直接移除所有外部传入的 indexing 参数，避免与 PyTorch 内部冲突
original_meshgrid = torch.meshgrid
def patched_meshgrid(*tensors, **kwargs):
    # 无论是否存在，直接删除 kwargs 中的 indexing 参数
    kwargs.pop('indexing', None)  # 核心修改：彻底移除 indexing 参数
    return original_meshgrid(*tensors, **kwargs)

# 覆盖原有的 torch.meshgrid
torch.meshgrid = patched_meshgrid

# 禁用 Ultralytics 内部检查，避免干扰补丁
os.environ.setdefault('YOLO_CHECKS', '0')

# 无需设置 UL_TAL.TORCH_1_10，补丁已处理参数冲突
# -------------------------- 补丁结束 --------------------------

def parse():
    p = argparse.ArgumentParser(description="Minimal YOLO detector (find_arm style)")
    p.add_argument('--model', type=str, required=True, help='Path to best.pt')
    p.add_argument('--source', type=str, required=True, help='Video/image/or image folder')
    p.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    p.add_argument('--device', type=str, default='', help="CUDA device like '0'; empty for auto")
    p.add_argument('--show', action='store_true', help='Show window')
    return p.parse_args()


def draw(frame, boxes, names=None):
    for b in boxes or []:
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
        conf = float(b.conf[0].cpu().numpy()) if hasattr(b, 'conf') else 0.0
        cls_id = int(b.cls[0].cpu().numpy()) if hasattr(b, 'cls') else -1
        label = str(cls_id)
        if names and isinstance(names, dict):
            label = names.get(cls_id, str(cls_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame


def run_image(model, path, out_dir, conf, device, show=False):
    img = cv2.imread(path)
    if img is None:
        print(f"Skip (read fail): {path}")
        return
    results = model(img, conf=conf, device=device, verbose=False)
    names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else None
    vis = draw(img.copy(), results[0].boxes if results else [], names)
    os.makedirs(out_dir, exist_ok=True)
    save_p = os.path.join(out_dir, Path(path).stem + '_det.jpg')
    cv2.imwrite(save_p, vis)
    if show:
        cv2.imshow('detect', vis)
        cv2.waitKey(1)
    print('Saved:', save_p)


def run_video(model, path, out_path, conf, device, show=False):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(Path(out_path).parent, exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else None

    i = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        results = model(f, conf=conf, device=device, verbose=False)
        vis = draw(f.copy(), results[0].boxes if results else [], names)
        vw.write(vis)
        if show:
            cv2.imshow('detect', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        i += 1
    cap.release()
    vw.release()
    cv2.destroyAllWindows()
    print('Video saved:', out_path)


def main():
    a = parse()
    model = YOLO(a.model)
    out_dir = 'runs/detect_swab_min'
    s = a.source
    if os.path.isdir(s):
        imgs = []
        for e in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            imgs += glob.glob(os.path.join(s, e))
        if len(imgs) == 0:
            print('No images found in folder:', s)
            return
        for p in imgs:
            run_image(model, p, out_dir, a.conf, a.device, a.show)
    elif Path(s).suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.mpg', '.mpeg'}:
        stem = Path(s).stem
        out_p = os.path.join(out_dir, stem + '_det.mp4')
        run_video(model, s, out_p, a.conf, a.device, a.show)
    else:
        run_image(model, s, out_dir, a.conf, a.device, a.show)


if __name__ == '__main__':
    main()

