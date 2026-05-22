"""视频推理工作线程 — 使用 threading.Thread 替代 QThread 避免原生栈溢出 (0xC0000409)"""
import cv2, numpy as np
from pathlib import Path
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal
from ultralytics import YOLO
import threading


class Detector(QObject):
    """视频推理管理器 — 非 QThread，用 threading.Thread 获得完整原生栈"""
    frame_ready = pyqtSignal(np.ndarray, int, int)
    fps_updated = pyqtSignal(float); stats_updated = pyqtSignal(dict)
    log_signal = pyqtSignal(str); finished = pyqtSignal()

    def __init__(self, model_path, video_path, conf=0.25, iou=0.45, target_fps=None):
        super().__init__()
        self.model_path = Path(model_path); self.video_path = Path(video_path)
        self.conf = conf; self.iou = iou
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self.export_path = None; self.target_fps = target_fps
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
                ret, frame = cap.read()
                if not ret: break
                idx += 1
                results = model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
                annotated = results.plot(line_width=2, font_size=8)
                if writer is not None: writer.write(annotated)
                now = datetime.now(); fps_val = 1.0 / max((now - t_prev).total_seconds(), 0.001); t_prev = now
                stats = {}
                if results.boxes is not None:
                    for cls_id in results.boxes.cls:
                        name = model.names.get(int(cls_id), f'cls_{int(cls_id)}')
                        stats[name] = stats.get(name, 0) + 1
                self.frame_ready.emit(annotated, idx, total)
                self.fps_updated.emit(fps_val); self.stats_updated.emit(stats)
            if writer is not None: writer.release(); self.log_signal.emit(f' Video saved: {Path(self.export_path).name}')
            cap.release()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_signal.emit(f' {e}')
        finally:
            self.finished.emit()
