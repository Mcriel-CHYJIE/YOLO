"""推理业务逻辑 — 批量预测 + 统计 + 视频推理工作线程"""
from pathlib import Path
from collections import Counter
from main.core.base import ROOT, cfg
from main.config import load_paths
from PyQt5.QtCore import QObject, pyqtSignal
from datetime import datetime
import cv2, numpy as np, threading
from ultralytics import YOLO
from .visualizer import compute_heatmap, extract_feature_maps, render_feature_map_grid, draw_boxes


# ══════════════════════════════════════════════════════════════
# 批量推理
# ══════════════════════════════════════════════════════════════

def run_batch_inference(
    model_path: Path, source_path: Path, conf: float, iou: float,
    gpu_ok: bool, save: bool = False,
) -> dict:
    """执行批量推理，返回结果数据集"""
    import numpy as np

    _pred_path = load_paths().get('predict_output', '')
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source_path), conf=conf, iou=iou,
        imgsz=cfg['predict']['imgsz'],
        device='0' if gpu_ok else 'cpu',
        save=save, save_txt=False, verbose=False,
        project=_pred_path,
        name=f'predict_{datetime.now().strftime("%m%d_%H%M%S")}')

    total_imgs = len(results)
    total_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
    cls_counts = Counter()
    for r in results:
        if r.boxes is not None:
            for cid in r.boxes.cls:
                nm = model.names.get(int(cid), f'cls_{int(cid)}')
                cls_counts[nm] += 1

    save_dir = str(results[0].save_dir) if results and hasattr(results[0], 'save_dir') else ''

    # ── 热力图 / 特征图 预计算（批模式始终计算）──
    heatmaps = []
    featuremaps = []       # 合并图（保存用）
    fm_layers_all = []     # 逐层图 [每张图片 → [ (name, grid_img), ... ]]
    for r in results:
        orig = r.orig_img if hasattr(r, 'orig_img') else None
        if orig is None:
            orig = cv2.imread(r.path) if hasattr(r, 'path') and r.path else None
        if orig is None:
            heatmaps.append(None)
            featuremaps.append(None)
            fm_layers_all.append([])
            continue

        hm = None
        fm_layers = []
        try:
            hm = compute_heatmap(model, orig, conf_threshold=conf)
        except Exception:
            hm = None
        try:
            raw = extract_feature_maps(model, orig)  # [(name, grid_img, dims), ...]
            if raw:
                fm_layers = [(name, grid) for name, grid, _ in raw]
                fm = render_feature_map_grid(raw, orig)
            else:
                fm = None
        except Exception as e:
            import traceback
            traceback.print_exc()
            fm = None
        heatmaps.append(hm)
        featuremaps.append(fm)
        fm_layers_all.append(fm_layers)

    # ── 保存全部视图（若开启导出）──
    saved_views_dir = ''
    if save and results:
        saved_views_dir = _save_all_views(results, heatmaps, fm_layers_all, _pred_path)

    return dict(
        results=results,
        total_imgs=total_imgs,
        total_dets=total_dets,
        cls_counts=dict(cls_counts),
        save_dir=save_dir,
        saved_views_dir=saved_views_dir,
        heatmaps=heatmaps,
        featuremaps=featuremaps,
        fm_layers_all=fm_layers_all,
    )


def _save_all_views(results, heatmaps, fm_layers_all, base_dir):
    """保存每张图的 5 个视图到 predict_output 子目录"""
    try:
        ts = datetime.now().strftime('%m%d_%H%M%S')
        out = Path(base_dir) / f'views_{ts}'
        out.mkdir(parents=True, exist_ok=True)

        for i, r in enumerate(results):
            stem = f'{i:04d}'
            # 原图
            orig = r.orig_img if hasattr(r, 'orig_img') and r.orig_img is not None else None
            if orig is not None:
                cv2.imwrite(str(out / f'{stem}_orig.jpg'), orig)
            # 检测图
            det = r.plot() if hasattr(r, 'plot') else None
            if det is not None:
                cv2.imwrite(str(out / f'{stem}_det.jpg'), det)
            # 热力图
            if i < len(heatmaps) and heatmaps[i] is not None:
                cv2.imwrite(str(out / f'{stem}_hm.jpg'), heatmaps[i])
            # 特征图（每层单独保存）
            if i < len(fm_layers_all):
                for li, (lname, grid) in enumerate(fm_layers_all[i]):
                    cv2.imwrite(str(out / f'{stem}_fm{li}.jpg'), grid)

        return str(out)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ''


# ══════════════════════════════════════════════════════════════
# Detector — 视频推理工作线程
# ══════════════════════════════════════════════════════════════

class Detector(QObject):
    """视频推理管理器 — 非 QThread，用 threading.Thread 获得完整原生栈"""
    frame_ready = pyqtSignal(np.ndarray, int, int)
    fps_updated = pyqtSignal(float); stats_updated = pyqtSignal(dict)
    details_ready = pyqtSignal(list)
    heat_signal = pyqtSignal(int, int, int)  # idx, det_count, total
    log_signal = pyqtSignal(str); finished = pyqtSignal()

    def __init__(self, model_path, video_path, conf=0.25, iou=0.45, target_fps=None,
                 show_heatmap=False, loop=False, heat_target_ids=None):
        super().__init__()
        self.model_path = Path(model_path); self.video_path = Path(video_path)
        self.conf = conf; self.iou = iou
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self.export_path = None; self.target_fps = target_fps
        self._show_heatmap = show_heatmap
        self._loop = loop
        self._heat_target_ids = heat_target_ids or set()
        self._seek_target = -1
        self._thread = None

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()

    def toggle_pause(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
        else:
            self._pause_event.set()

    def isRunning(self):
        return self._thread is not None and self._thread.is_alive()

    def set_heatmap(self, enabled: bool):
        """运行时切换热力图"""
        self._show_heatmap = enabled

    def seek(self, frame_pos: int):
        """拖拽进度条跳转"""
        self._seek_target = frame_pos

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            model = YOLO(str(self.model_path))
            self.log_signal.emit(f' {self.model_path.name}')
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened(): self.log_signal.emit('Failed to open video'); self.finished.emit(); return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._total_frames = total
            self.log_signal.emit(f' {self.video_path.name}  {total}帧')
            writer = None
            if self.export_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(str(self.export_path), fourcc, fps, (w, h))
                if writer.isOpened(): self.log_signal.emit(f' Saving to: {Path(self.export_path).name}')
                else: writer = None
            idx, t_prev = 0, datetime.now()
            while not self._stop_event.is_set() and cap.isOpened():
                if self._pause_event.is_set():
                    self._pause_event.wait(0.05)
                    continue
                if self._seek_target >= 0:
                    target = self._seek_target
                    self._seek_target = -1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    idx = target
                    continue
                ret, frame = cap.read()
                if not ret:
                    if self._loop and not self._stop_event.is_set():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        idx = 0
                        self.log_signal.emit(' 🔄 循环播放')
                        continue
                    break
                idx += 1
                results = model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
                annotated = results.plot(line_width=2, font_size=8)
                # 热力图叠加
                display_frame = annotated
                if self._show_heatmap:
                    try:
                        hm = compute_heatmap(model, frame, conf_threshold=self.conf)
                        # 热力图 + 检测框叠加
                        hm = draw_boxes(hm, results, model.names, conf_threshold=self.conf)
                        display_frame = hm
                    except Exception:
                        pass
                if writer is not None:
                    writer.write(annotated if not self._show_heatmap else display_frame)
                now = datetime.now(); fps_val = 1.0 / max((now - t_prev).total_seconds(), 0.001); t_prev = now
                stats = {}
                dets = []
                heat_det_count = 0
                if results.boxes is not None:
                    for cls_id, conf in zip(results.boxes.cls, results.boxes.conf):
                        cid = int(cls_id)
                        name = model.names.get(cid, f'cls_{cid}')
                        stats[name] = stats.get(name, 0) + 1
                        dets.append(f'{name} {conf:.2f}')
                        if not self._heat_target_ids or cid in self._heat_target_ids:
                            heat_det_count += 1
                line = f'[{idx}/{total}]'
                if dets:
                    line += '  ' + ' | '.join(dets)
                else:
                    line += '  —'
                self.frame_ready.emit(display_frame, idx, total)
                self.fps_updated.emit(fps_val); self.stats_updated.emit(stats)
                self.details_ready.emit([line])
                self.heat_signal.emit(idx, heat_det_count, total)
            if writer is not None: writer.release(); self.log_signal.emit(f' Video saved: {Path(self.export_path).name}')
            cap.release()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_signal.emit(f' {e}')
        finally:
            self.finished.emit()
