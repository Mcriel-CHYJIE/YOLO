"""视频推理工作线程"""
import cv2, numpy as np
from pathlib import Path
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal


class Detector(QThread):
    frame_ready = pyqtSignal(np.ndarray, int, int)
    fps_updated = pyqtSignal(float); stats_updated = pyqtSignal(dict)
    log_signal = pyqtSignal(str); finished = pyqtSignal()

    def __init__(self, model_path, video_path, conf=0.25, iou=0.45, target_fps=24):
        super().__init__()
        self.model_path = Path(model_path); self.video_path = Path(video_path)
        self.conf = conf; self.iou = iou; self._pause = False; self._stop = False
        self.export_path = None; self.target_fps = target_fps

    def stop(self): self._stop = True; self._pause = False
    def toggle_pause(self): self._pause = not self._pause

    def run(self):
        try:
            from ultralytics import YOLO
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
            fi = 1.0 / self.target_fps
            while not self._stop and cap.isOpened():
                if self._pause: self.msleep(50); continue
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
                elapsed = (datetime.now() - t_prev).total_seconds()
                sleep_time = max(0, fi - elapsed)
                if sleep_time > 0: self.msleep(int(sleep_time * 1000))
            if writer is not None: writer.release(); self.log_signal.emit(f' Video saved: {Path(self.export_path).name}')
            cap.release()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_signal.emit(f' {e}')
        finally: self.finished.emit()
