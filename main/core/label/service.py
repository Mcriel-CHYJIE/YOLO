# =============================================================================
# YOLO Training Studio — 标注页业务逻辑服务
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""标注业务逻辑 — 数据模型、session 持久化、文件操作、后台 Worker"""

import json
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from main.core.settings.service import load_shortcuts as _load_shortcuts
from main.config import load_paths


# ── 路径解析 ──
_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ══════════════════════════════════════════════════════════════
# Annotation 数据模型（纯数据，无 Qt 依赖）
# ══════════════════════════════════════════════════════════════

class Annotation:
    """单个标注框 (YOLO format: class_id xc yc w h 归一化)"""

    __slots__ = ('class_id', 'xc', 'yc', 'w', 'h')

    def __init__(self, class_id, xc, yc, w, h):
        # 防护：NaN/Inf → 修复为有效值
        import math
        def _clean(v, default, lo=0.0, hi=1.0):
            if not math.isfinite(v) or v < lo:
                return default
            if v > hi:
                return hi
            return v
        # w/h 最小 0.005（对应 640 图上约 3 像素）
        self.class_id = int(class_id) if class_id is not None else 0
        self.xc = _clean(xc, 0.5, 0.0, 1.0)
        self.yc = _clean(yc, 0.5, 0.0, 1.0)
        self.w = max(0.005, _clean(w, 0.05, 0.0, 1.0))
        self.h = max(0.005, _clean(h, 0.05, 0.0, 1.0))

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

def after_root() -> Path:
    p = load_paths().get('label_dir', '')
    return Path(p) / 'after' if p else Path()


def label_root() -> Path:
    p = load_paths().get('label_dir', '')
    return Path(p) / 'label' if p else Path()


# ══════════════════════════════════════════════════════════════
# Session 持久化
# ══════════════════════════════════════════════════════════════

def load_session(session_path: Path):
    """
    从 _annotations.json 加载 session（相对路径转回绝对路径）。

    返回 (annotations_dict, current_idx)
        annotations_dict: {absolute_path_str: [Annotation, ...]}
        current_idx: int
    """
    data = {}
    current_idx = 0
    if session_path.exists():
        try:
            data = json.loads(session_path.read_text("utf-8"))
        except Exception:
            data = {}
    base = session_path.parent
    saved = data.get("annotations", {})
    annotations = {}
    for k, v in saved.items():
        ap = base / k  # 相对路径 → 绝对路径
        annotations[str(ap)] = [Annotation.from_dict(a) for a in v]
    current_idx = data.get("current_idx", 0)
    return annotations, current_idx


def save_session_file(session_path: Path, current_idx: int,
                      annotations: dict) -> dict:
    """
    将当前标注数据写入 session 文件（路径存为相对路径）。
    """
    if session_path is None:
        return {"ok": False, "saved": False, "error": "no session path"}
    base = session_path.parent
    annotations_serialized = {}
    for k, v in annotations.items():
        try:
            rel = Path(k).relative_to(base).as_posix()
        except ValueError:
            rel = k  # 不在同一目录下则保留绝对路径
        annotations_serialized[rel] = [a.to_dict() for a in v]
    data = {
        "current_idx": current_idx,
        "annotations": annotations_serialized,
    }
    try:
        session_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        return {"ok": True, "saved": True, "error": None}
    except Exception as e:
        return {"ok": False, "saved": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# 图片 / 文件操作
# ══════════════════════════════════════════════════════════════

def delete_image_file(img_path: Path) -> bool:
    """删除图片文件。返回 True 表示文件被删除。"""
    if img_path.exists():
        img_path.unlink()
        return True
    return False


def delete_label_file(img_path: Path, lbl_root: Path) -> bool:
    """删除对应的标注 .txt 文件。"""
    lbl = lbl_root / img_path.parent.name / f"{img_path.stem}.txt"
    if lbl.exists():
        lbl.unlink()
        return True
    return False


# ══════════════════════════════════════════════════════════════
# 随机筛选算法
# ══════════════════════════════════════════════════════════════

def random_filter_paths(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """
    按顺序每 3 张一组，每组随机保留 1 张。

    返回 (to_keep, to_delete)
    """
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
# Export config.yaml 生成
# ══════════════════════════════════════════════════════════════

def generate_export_yaml(out_root: Path, project_root: Path,
                         class_names: dict) -> str:
    """生成导出的 config.yaml 内容字符串。"""
    try:
        rel = out_root.relative_to(project_root)
    except ValueError:
        rel = out_root
    names_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(class_names.items()))
    return (
        f"# Auto-generated by LabelTab\n"
        f"path: {rel.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names:\n"
        f"{names_lines}\n"
    )


# ══════════════════════════════════════════════════════════════
# 快捷键加载
# ══════════════════════════════════════════════════════════════

def load_shortcut_keys() -> dict:
    """从 shortcuts.json 加载快捷键映射。"""
    return _load_shortcuts()


def get_cls_shortcuts() -> dict[str, int]:
    """
    从 settings.json 加载类别快捷键映射 {键名: class_id}。
    无配置时返回默认值 {'1': 0, '2': 1, '3': 2, '4': 3}。
    """
    data = _load_shortcuts()
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


# ══════════════════════════════════════════════════════════════
# ExportWorker — 后台数据集导出线程
# ══════════════════════════════════════════════════════════════

class ExportWorker(QThread):
    """后台数据集导出线程"""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, annotations, out_root, train_ratio, project_root):
        super().__init__()
        self._annotations = annotations
        self._out_root = out_root
        self._train_ratio = train_ratio
        self._project_root = project_root

    def run(self):
        import hashlib, shutil
        try:
            items = list(self._annotations.items())
            items.sort(key=lambda x: hashlib.md5(x[0].encode("utf-8")).hexdigest())
            split_idx = max(1, int(len(items) * self._train_ratio / 100.0))
            train_items = items[:split_idx]
            val_items = items[split_idx:]
            for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
                (self._out_root / sub).mkdir(parents=True, exist_ok=True)

            total = len(items)
            exported_count = 0
            skipped_count = 0

            for split_name, split_items in [("train", train_items), ("val", val_items)]:
                for orig_path, anns in split_items:
                    src = Path(orig_path)
                    if not src.exists():
                        continue
                    base_name = src.stem
                    dst_img = self._out_root / "images" / split_name / f"{base_name}.jpg"
                    dst_lbl = self._out_root / "labels" / split_name / f"{base_name}.txt"

                    if dst_img.exists() or dst_lbl.exists():
                        skipped_count += 1
                        continue

                    shutil.copy2(str(src), str(dst_img))
                    lines = [a.to_yolo() for a in anns]
                    dst_lbl.write_text("\n".join(lines) + "\n", "utf-8")

                    exported_count += 1
                    self.progress.emit(int(exported_count / total * 100))

            # 生成 data.yaml（使用相对路径，可移植）
            class_names = sorted(set(a.class_id for anns_list in self._annotations.values()
                                      for a in anns_list))
            data_yaml = {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "nc": len(class_names),
                "names": {i: f"class_{i}" for i in class_names},
            }
            try:
                import yaml
                yaml_path = self._out_root / "data.yaml"
                yaml_path.write_text(yaml.dump(data_yaml, default_flow_style=False), "utf-8")
            except Exception:
                pass

            msg = f"Exported {exported_count}, skipped {skipped_count}"
            self.log.emit(msg)
            self.done.emit(True, msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log.emit(f" {e}")
            self.done.emit(False, str(e))
