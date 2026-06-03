# =============================================================================
# YOLO Training Studio — Relabel 页业务逻辑
# 直接从 dataset 的 images/labels 目录读取，修改后直接覆盖 .txt 文件
# =============================================================================

"""Relabel 业务逻辑 — 读取/写入 YOLO 格式标注文件"""

import json
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from main.config import load_paths


# ══════════════════════════════════════════════════════════════
# Annotation 数据模型（与 label tab 共用）
# ══════════════════════════════════════════════════════════════

class Annotation:
    """单个标注框 (YOLO format: class_id xc yc w h 归一化)"""
    __slots__ = ('class_id', 'xc', 'yc', 'w', 'h')

    def __init__(self, class_id, xc, yc, w, h):
        self.class_id = class_id
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h

    def to_yolo(self):
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"

    def to_dict(self):
        return {'class_id': self.class_id, 'xc': self.xc, 'yc': self.yc, 'w': self.w, 'h': self.h}

    @classmethod
    def from_dict(cls, d):
        return cls(d['class_id'], d['xc'], d['yc'], d['w'], d['h'])

    @classmethod
    def from_yolo(cls, line):
        parts = line.strip().split()
        if len(parts) >= 5:
            return cls(int(parts[0]), float(parts[1]), float(parts[2]),
                       float(parts[3]), float(parts[4]))
        return None


def resolve_class_name(class_id):
    """从配置的 names 映射中获取类名"""
    import main.core.base as base
    name = base.CLASS_NAMES.get(class_id, f'cls{class_id}')
    return str(list(name.values())[0]) if isinstance(name, dict) else name


# ══════════════════════════════════════════════════════════════
# 路径
# ══════════════════════════════════════════════════════════════

def dataset_images_root() -> Path:
    """返回 dataset_dir/images/"""
    p = load_paths().get('dataset_dir', '')
    return Path(p) / 'images' if p else Path()


def dataset_labels_root() -> Path:
    """返回 dataset_dir/labels/"""
    p = load_paths().get('dataset_dir', '')
    return Path(p) / 'labels' if p else Path()


def get_split_folders() -> list[str]:
    """返回 images/ 下的子目录名（如 train, val）"""
    root = dataset_images_root()
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


# ══════════════════════════════════════════════════════════════
# 标签文件读写
# ══════════════════════════════════════════════════════════════

def load_labels_for_image(img_path: Path, split: str) -> list[Annotation]:
    """从 dataset_dir/labels/{split}/ 读取对应图片的 .txt 标注"""
    lbl_root = dataset_labels_root()
    lbl_file = lbl_root / split / f"{img_path.stem}.txt"
    anns = []
    if lbl_file.exists():
        for line in lbl_file.read_text('utf-8').strip().splitlines():
            line = line.strip()
            if line:
                ann = Annotation.from_yolo(line)
                if ann:
                    anns.append(ann)
    return anns


def save_labels_for_image(img_path: Path, split: str, anns: list[Annotation]):
    """将标注写入 dataset_dir/labels/{split}/ 对应 .txt 文件"""
    lbl_root = dataset_labels_root()
    lbl_dir = lbl_root / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    lbl_file = lbl_dir / f"{img_path.stem}.txt"
    if anns:
        lines = [a.to_yolo() for a in anns]
        lbl_file.write_text('\n'.join(lines) + '\n', 'utf-8')
    else:
        # 无标注则写空文件（清除旧标注）
        lbl_file.write_text('', 'utf-8')


# ══════════════════════════════════════════════════════════════
# 快捷键加载
# ══════════════════════════════════════════════════════════════

def load_shortcut_keys() -> dict:
    from main.core.settings.service import resolve_shortcuts_file, load_shortcuts as _load_shortcuts
    sf = resolve_shortcuts_file()
    return _load_shortcuts(sf)


def get_cls_shortcuts() -> dict[str, int]:
    from main.core.settings.service import resolve_shortcuts_file, load_shortcuts as _load_shortcuts
    sf = resolve_shortcuts_file()
    data = _load_shortcuts(sf)
    result = {}
    for k, v in data.items():
        if k.startswith('cls_'):
            try:
                result[v] = int(k.split('_')[1])
            except Exception:
                pass
    if not result:
        for i in range(4):
            result[str(i + 1)] = i
    return result


# ══════════════════════════════════════════════════════════════
# 随机筛选算法
# ══════════════════════════════════════════════════════════════

def random_filter_paths(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """按顺序每 3 张一组，每组随机保留 1 张。返回 (to_keep, to_delete)"""
    import random
    to_keep = []
    to_delete = []
    for i in range(0, len(paths), 3):
        group = list(paths[i:i + 3])
        if not group:
            continue
        keep = random.choice(group)
        to_keep.append(keep)
        for p in group:
            if p != keep:
                to_delete.append(p)
    return to_keep, to_delete


# ══════════════════════════════════════════════════════════════
# AutoLabelWorker — 后台自动标注线程
# ══════════════════════════════════════════════════════════════

class AutoLabelWorker(QThread):
    """后台自动标注线程"""
    progress = pyqtSignal(int, int)
    image_done = pyqtSignal(str, list)
    done = pyqtSignal(bool, str)
    log = pyqtSignal(str)

    def __init__(self, model_path, image_paths, conf=0.25, iou=0.45):
        super().__init__()
        self._model_path = model_path
        self._image_paths = image_paths
        self._conf = conf
        self._iou = iou
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            from ultralytics import YOLO
            self.log.emit(f"加载模型: {Path(self._model_path).name}")
            model = YOLO(self._model_path)
            n = len(self._image_paths)
            self.log.emit(f"开始自动标注 {n} 张图片...")
            for i, img_path in enumerate(self._image_paths):
                if self._stop_flag:
                    self.done.emit(False, "Stopped")
                    return
                results = model(img_path, conf=self._conf, iou=self._iou, verbose=False)[0]
                anns = []
                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        xywhn = box.xywhn[0].tolist()
                        if len(xywhn) == 4:
                            anns.append(dict(class_id=cls_id, xc=xywhn[0], yc=xywhn[1],
                                             w=xywhn[2], h=xywhn[3]))
                self.image_done.emit(str(img_path), anns)
                self.progress.emit(i + 1, n)
            self.log.emit("自动标注完成")
            self.done.emit(True, "Auto-label complete")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log.emit(f" {e}")
            self.done.emit(False, str(e))
