import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from ultralytics.nn.modules.block import F

# 修复 torch.meshgrid 'indexing' 参数重复问题
original_meshgrid = torch.meshgrid
def patched_meshgrid(*tensors, **kwargs):
    if 'indexing' in kwargs:
        return original_meshgrid(*tensors, indexing=kwargs['indexing'])
    return original_meshgrid(*tensors)
torch.meshgrid = patched_meshgrid

# 禁用 Ultralytics 的 TORCH_1_10 检查（避免触发旧版逻辑）
try:
    import ultralytics.utils.tal as UL_TAL
    if hasattr(UL_TAL, 'TORCH_1_10'):
        UL_TAL.TORCH_1_10 = False
except Exception:
    pass

import torch
_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)  # 显式关闭安全模式
    return _torch_load(*args, **kwargs)
torch.load = _patched_load

def parse_args():
    parser = argparse.ArgumentParser(description="简洁YOLO检测程序")
    parser.add_argument("--model", default="runs/train_swab/exp12/weights/best.pt",type=str, required=False, help="模型路径")
    parser.add_argument("--source", default="圆形消毒.mp4",type=str, required=False, help="检测源")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--show", default=True,action="store_true", help="显示窗口")
    return parser.parse_args()

def draw_result(frame, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def detect_video(model, video_path, conf, show):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    
    save_dir = Path("runs/simple_detect")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(video_path).stem}_det.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf, verbose=False)
        vis_frame = draw_result(frame.copy(), results[0].boxes, model.names)
        out.write(vis_frame)
        if show:
            cv2.imshow("Detect", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"结果保存：{save_path}")

if __name__ == "__main__":
    args = parse_args()
    model_path = Path(args.model)
    source_path = Path(args.source)

    # 更友好的本地文件检查，避免因联网失败抛出代理错误
    if not model_path.is_file():
        raise FileNotFoundError(
            f"模型文件不存在：{model_path}\n"
            "请提供本地 .pt 模型的实际路径（当前环境无法联网，Ultralytics 无法从 GitHub 自动下载权重）。"
        )
    if not source_path.is_file():
        raise FileNotFoundError(f"视频文件不存在：{source_path}")

    # 可选：关闭 Ultralytics 的在线同步/检查，减少不必要的网络请求
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS["sync"] = False
    except Exception:
        pass

    model = YOLO(str(model_path))
    if source_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        detect_video(model, str(source_path), args.conf, args.show)
    else:
        print("请提供视频文件")

#python detect_swab.py --model "runs\train_swab\exp12\weights\best.pt" --source "圆形消毒.mp4" --conf 0.4 --show