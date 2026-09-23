import os
import cv2
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

# Fix meshgrid indexing issue for torch 2.0.1 + ultralytics 8.3.x compatibility
import torch
original_meshgrid = torch.meshgrid
def patched_meshgrid(*tensors, **kwargs):
    if 'indexing' in kwargs:
        # Remove duplicate indexing if it exists in both args and kwargs
        return original_meshgrid(*tensors, indexing=kwargs['indexing'])
    return original_meshgrid(*tensors, **kwargs)
# Work around torch.meshgrid 'indexing' kwarg duplication by forcing Ultralytics to not pass it
try:
    import ultralytics.utils.tal as UL_TAL  # type: ignore
    if hasattr(UL_TAL, 'TORCH_1_10'):
        UL_TAL.TORCH_1_10 = False  # this makes tal.make_anchors call torch.meshgrid without 'indexing='
except Exception:
    pass

# Also disable checks via env to avoid extra internal warmups that may hit the same path
os.environ.setdefault('YOLO_CHECKS', '0')

torch.meshgrid = patched_meshgrid

# Disable Ultralytics runtime checks to avoid AMP meshgrid issue on certain torch versions
try:
    from ultralytics.utils import SETTINGS as ULTRA_SETTINGS  # type: ignore
    if isinstance(ULTRA_SETTINGS, dict):
        ULTRA_SETTINGS.update({'checks': False})
except Exception:
    pass


HELP_TEXT = """
键盘快捷键:
  - 鼠标左键拖拽: 绘制或调整新框
  - 数字键 0-9: 选择类别
  - n / p: 下一张 / 上一张
  - s: 保存当前标注
  - x: 删除最近一个框
  - h: 显示/隐藏帮助
  - q: 退出标注器
"""


def parse_args():
    p = argparse.ArgumentParser(description="Annotation + YOLO training all-in-one")
    # 数据来源
    p.add_argument("--video", type=str, default="圆形消毒.mp4", help="Optional video to extract frames from")
    p.add_argument("--images_dir", type=str, default=None, help="Optional dir of images to annotate (jpg/png)")
    p.add_argument("--extract_every", type=int, default=5, help="Extract 1 frame every N frames from video")

    # 输出与类别
    p.add_argument("--dataset_root", type=str, default="swab_dataset", help="Dataset root to create: images/labels and train/val after split")
    p.add_argument("--classes", type=str, default="swab", help="Comma-separated class names, e.g., 'swab' or 'swab,hand'")

    # 阶段控制
    p.add_argument("--stage", type=str, default="all", choices=["annotate", "train", "all"], help="Run only annotation, only training, or both")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio when splitting dataset")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")

    # 训练参数
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model to finetune or yaml to train from scratch")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--project", type=str, default="runs/train_swab")
    p.add_argument("--name", type=str, default="exp")
    # 为兼容你当前的 torch/ultralytics 组合，默认关闭 AMP，避免 meshgrid 索引参数冲突
    p.add_argument("--amp", action="store_true", help="Enable AMP mixed precision (default: off)")

    return p.parse_args()


def ensure_dirs(root: Path):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def extract_frames(video_path: str, out_dir: Path, every: int = 5) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    idx = 0
    saved = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % max(1, every) == 0:
            fn = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(fn), frame)
            saved.append(fn)
        idx += 1
    cap.release()
    return saved


def yolo_box_to_txt(path: Path, boxes: List[Tuple[int, float, float, float, float]]):
    with open(path, "w", encoding="utf-8") as f:
        for cid, cx, cy, w, h in boxes:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def txt_to_yolo_boxes(path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ps = line.strip().split()
            if len(ps) == 5:
                out.append((int(ps[0]), float(ps[1]), float(ps[2]), float(ps[3]), float(ps[4])))
    return out


def abs_to_norm(x1, y1, x2, y2, W, H) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
    x1, x2 = max(0.0, min(x1, W - 1)), max(0.0, min(x2, W - 1))
    y1, y2 = max(0.0, min(y1, H - 1)), max(0.0, min(y2, H - 1))
    xa, xb = min(x1, x2), max(x1, x2)
    ya, yb = min(y1, y2), max(y1, y2)
    w, h = xb - xa, yb - ya
    if w <= 1e-6 or h <= 1e-6:
        return 0.5, 0.5, 0.0, 0.0
    cx, cy = xa + w / 2.0, ya + h / 2.0
    return cx / W, cy / H, w / W, h / H


def norm_to_abs(cx, cy, w, h, W, H) -> Tuple[int, int, int, int]:
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    return x1, y1, x2, y2


class Annotator:
    def __init__(self, images_dir: Path, labels_dir: Path, classes: List[str]):
        self.images = list_images(images_dir)
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {images_dir}")
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.classes = classes
        self.idx = 0
        self.boxes: List[Tuple[int, float, float, float, float]] = []
        self.dragging = False
        self.anchor: Optional[Tuple[int, int]] = None
        self.current_rect: Optional[Tuple[int, int, int, int]] = None
        self.current_class = 0
        self.show_help = True
        # Use ASCII window name to avoid rare Unicode issues on Windows
        self.win = "annotator"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.on_mouse)

    def load_boxes(self, img_path: Path):
        txt = self.labels_dir / (img_path.stem + ".txt")
        self.boxes = txt_to_yolo_boxes(txt)

    def save_boxes(self, img_path: Path):
        txt = self.labels_dir / (img_path.stem + ".txt")
        yolo_box_to_txt(txt, self.boxes)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.anchor = (x, y)
            self.current_rect = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            ax, ay = self.anchor if self.anchor else (x, y)
            self.current_rect = (ax, ay, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            if self.anchor is not None:
                ax, ay = self.anchor
                cx, cy, w, h = abs_to_norm(ax, ay, x, y, self.W, self.H)
                if w > 0 and h > 0:
                    self.boxes.append((self.current_class, cx, cy, w, h))
            self.anchor = None
            self.current_rect = None

    def draw(self, img):
        vis = img.copy()
        # draw existing boxes
        for cid, cx, cy, w, h in self.boxes:
            x1, y1, x2, y2 = norm_to_abs(cx, cy, w, h, self.W, self.H)
            color = (0, 255, 0) if cid == self.current_class else (0, 200, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{self.classes[cid]}"
            cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # draw current rect
        if self.current_rect is not None:
            x1, y1, x2, y2 = self.current_rect
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # help
        info = f"[{self.idx+1}/{len(self.images)}] cls={self.current_class}:{self.classes[self.current_class]}  boxes={len(self.boxes)}"
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
        if self.show_help:
            y0 = 50
            for i, line in enumerate(HELP_TEXT.strip().splitlines()):
                cv2.putText(vis, line, (10, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3)
                cv2.putText(vis, line, (10, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        return vis

    def run(self):
        while True:
            img_path = self.images[self.idx]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read {img_path}")
                if self.idx < len(self.images) - 1:
                    self.idx += 1
                    continue
                else:
                    break
            self.H, self.W = img.shape[:2]
            self.load_boxes(img_path)
            while True:
                vis = self.draw(img)
                cv2.imshow(self.win, vis)
                k = cv2.waitKey(10) & 0xFF
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif k == ord('n'):
                    self.save_boxes(img_path)
                    if self.idx < len(self.images) - 1:
                        self.idx += 1
                    break
                elif k == ord('p'):
                    self.save_boxes(img_path)
                    if self.idx > 0:
                        self.idx -= 1
                    break
                elif k == ord('s'):
                    self.save_boxes(img_path)
                    print(f"Saved: {img_path.stem}.txt")
                elif k == ord('x'):
                    if len(self.boxes) > 0:
                        self.boxes.pop()
                elif k == ord('h'):
                    self.show_help = not self.show_help
                elif ord('0') <= k <= ord('9'):
                    cid = k - ord('0')
                    if cid < len(self.classes):
                        self.current_class = cid
        cv2.destroyAllWindows()


def split_dataset(root: Path, val_ratio: float, seed: int):
    random.seed(seed)
    images = list_images(root / "images")
    labeled = [p for p in images if (root / "labels" / (p.stem + ".txt")).exists()]
    random.shuffle(labeled)
    n_val = max(1, int(len(labeled) * val_ratio)) if len(labeled) > 0 else 0
    val = set(labeled[:n_val])
    train = [p for p in labeled if p not in val]

    # make dirs
    for split in ["train", "val"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

    def cp(imgs: List[Path], split: str):
        for ip in imgs:
            lp = root / "labels" / (ip.stem + ".txt")
            shutil.copy2(ip, root / split / "images" / ip.name)
            shutil.copy2(lp, root / split / "labels" / lp.name)

    cp(train, "train")
    cp(list(val), "val")
    print(f"Split done. train={len(train)} val={len(val)}")


def build_data_yaml(root: Path, classes: List[str]) -> Path:
    # Use absolute 'path' and make train/val relative to it, per Ultralytics dataset spec
    base = root.resolve().as_posix()
    yaml_text = "\n".join([
        f"path: {base}",
        f"train: train/images",
        f"val: val/images",
        f"names: {classes}",
    ])
    ypath = root / "dataset.yaml"
    with open(ypath, "w", encoding="utf-8") as f:
        f.write(yaml_text + "\n")
    print(f"Wrote: {ypath}")
    print("Content:\n" + yaml_text)
    return ypath


def train_yolo(data_yaml: Path, args):
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. pip install ultralytics")
    model = YOLO(args.model)
    # 根据 --amp 开关决定是否启用混合精度；默认关闭以兼容你当前的 torch 版本
    train_kwargs = dict(
        data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, project=args.project, name=args.name,
    )
    if hasattr(args, 'amp') and not args.amp:
        train_kwargs['amp'] = False
    model.train(**train_kwargs)
    model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device)
    print("Training/Validation done. Check runs dir.")


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    ensure_dirs(root)
    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    assert len(classes) > 0, "--classes cannot be empty"

    images_dir = root / "images"
    labels_dir = root / "labels"

    # 1) 准备图像
    if args.stage in ("annotate", "all"):
        if args.video:
            print(f"Extracting frames from {args.video} ...")
            extract_frames(args.video, images_dir, every=args.extract_every)
        elif args.images_dir:
            # 拷贝图片到数据集 images
            srcs = list_images(Path(args.images_dir))
            for sp in srcs:
                shutil.copy2(sp, images_dir / sp.name)
            print(f"Copied {len(srcs)} images to {images_dir}")
        else:
            print(f"No --video or --images_dir provided. Using existing images under {images_dir} (if any)")

        # 2) 启动标注器
        if len(list_images(images_dir)) == 0:
            raise RuntimeError("No images to annotate. Provide --video or --images_dir")
        print("Launching annotator ...")
        Annotator(images_dir, labels_dir, classes).run()
        print("Annotator finished.")

    # 3) 切分数据集 + 写 YAML
    if args.stage in ("train", "all"):
        split_dataset(root, val_ratio=args.val_ratio, seed=args.seed)
        data_yaml = build_data_yaml(root, classes)
        # 4) 训练 YOLO
        train_yolo(data_yaml, args)


if __name__ == "__main__":
    main()

