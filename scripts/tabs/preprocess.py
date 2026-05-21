"""视频预处理标签页 — 重命名+缩放+抽帧"""
from scripts.tabs.base import *
from PyQt5 import uic
import cv2, os, time
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = 'error'


class VideoPreprocessWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    video_progress = pyqtSignal(int, int)
    done = pyqtSignal(bool, str)
    image_saved = pyqtSignal(str)

    def __init__(self, src_folder: str, out_folder: str, target_size: int = 640):
        super().__init__()
        self.src_folder = Path(src_folder)
        self.out_folder = Path(out_folder)
        self.target_size = target_size
        self._stop = False

    def stop(self):
        self._stop = True

    @staticmethod
    def _resize_image(img, target_size=640):
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(img, (nw, nh), interpolation=interpolation)

    def run(self):
        try:
            import random
            exts = VIDEO_EXTS
            videos = sorted([f for f in self.src_folder.iterdir()
                             if f.suffix.lower() in exts and f.is_file()])
            if not videos:
                self.log.emit(' 未找到视频文件')
                self.done.emit(False, 'No video files found'); return
            n = len(videos)
            folder_name = self.src_folder.name
            self.log.emit(f' {self.src_folder.name} — {n} 个视频')
            self.out_folder.mkdir(parents=True, exist_ok=True)
            self.log.emit(' 开始重命名视频...')
            renamed_videos = []
            for idx, src_path in enumerate(videos):
                if self._stop: self.log.emit(' 已停止'); self.done.emit(False, 'Stopped'); return
                new_name = f'{idx:02d}{src_path.suffix}'
                rp = self.src_folder / new_name
                if src_path.name != new_name:
                    if rp.exists():
                        renamed_videos.append(rp)
                    else:
                        src_path.rename(rp); renamed_videos.append(rp)
                else:
                    renamed_videos.append(rp)
            self.log.emit(f' 重命名完成，共 {len(renamed_videos)} 个视频')
            self.log.emit(' 开始抽帧处理（每秒1帧，随机帧）...')
            for idx, rp in enumerate(renamed_videos):
                if self._stop: self.log.emit(' 已停止'); self.done.emit(False, 'Stopped'); return
                vnum = f'{idx:02d}'; nm = rp.name
                cap = cv2.VideoCapture(str(rp))
                if not cap.isOpened(): self.log.emit(f' 无法打开 {nm}，跳过'); continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                secs = int(total / fps)
                self.log.emit(f' {nm} | {fps:.1f}fps | {total}帧 | 约{secs}秒')
                self.progress.emit(idx + 1, n)
                saved = 0; errs = 0
                for s in range(secs):
                    if self._stop: break
                    try:
                        sf = s * int(fps); ef = min((s + 1) * int(fps), total)
                        rf = random.randint(sf, ef - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, rf)
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            resized = self._resize_image(frame, self.target_size)
                            fn = f'{folder_name}-{vnum}-{s:04d}.jpg'
                            cv2.imwrite(str(self.out_folder / fn), resized,
                                        [cv2.IMWRITE_JPEG_QUALITY, 98])
                            saved += 1
                            self.video_progress.emit(saved, secs)
                            self.image_saved.emit(str(self.out_folder / fn))
                        else:
                            errs += 1
                            if errs <= 3: self.log.emit(f'  Frame decode failed at second {s}')
                    except Exception as e:
                        errs += 1
                        if errs <= 3: self.log.emit(f'  Error at {s}: {str(e)[:50]}')
                cap.release()
                self.log.emit(f' {nm} → {saved} 帧{" (" + str(errs) + " decode errors)" if errs else ""}')
            self.log.emit(f' 全部完成！共处理 {n} 个视频')
            self.done.emit(True, f'Complete — {n} videos processed')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log.emit(f' {e}'); self.done.emit(False, str(e))


class PreprocessTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._worker = None
        self._build()
        self._init_widgets()
        self._connect_signals()
        self._refresh_folder_list()

    @property
    def _before_root(self): return ROOT / 'original' / 'before'
    @property
    def _after_root(self): return ROOT / 'original' / 'after'

    def _build(self):
        ui_path = Path(__file__).resolve().parent.parent / 'ui' / 'preprocess.ui'
        uic.loadUi(str(ui_path), self)

    def _init_widgets(self):
        self.video_list.setStyleSheet(f'''
            QListWidget {{background:{BG};border:1px solid {BORDER};border-radius:4px;
                font-size:11px;color:{TEXT};padding:3px;}}
            QListWidget::item {{padding:4px 6px;border-radius:3px;}}
            QListWidget::item:selected {{background:{PRI};color:#fff;}}''')
        for placeholder, label, color, attr_v, attr_c in [
            (self.videoMetricCard, 'Video', TEXT, '_pv', '_pv_card'),
            (self.frameMetricCard, 'Frame', GREEN, '_pf', '_pf_card'),
        ]:
            card = MetricCard(label, color, '—')
            idx = self.metricRowPP.indexOf(placeholder)
            self.metricRowPP.removeWidget(placeholder)
            placeholder.deleteLater()
            self.metricRowPP.insertWidget(idx, card, 1)
            setattr(self, attr_v, card.value_label)
            setattr(self, attr_c, card)
            card.value_label.setStyleSheet(
                f'font-size:16px;font-weight:700;color:{color};qproperty-alignment:AlignCenter;')

    def _connect_signals(self):
        self.srcBrowseBtn.clicked.connect(self._browse_input)
        self.outBrowseBtn.clicked.connect(self._browse_output)
        self.resetBtn.clicked.connect(self._refresh_folder_list)
        self.start_btn.setObjectName('pri')
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.setObjectName('danger')
        self.stop_btn.clicked.connect(self._stop)

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Input Directory',
            str(self._before_root) if self._before_root.exists() else str(ROOT))
        if folder:
            fp = Path(folder)
            self.src_input.setText(str(fp))
            self.input_info.setText(f' {fp}')
            self.out_input.setText(str(self._after_root / fp.name))
            self._refresh_video_list(fp)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Output Directory',
            str(self._after_root) if self._after_root.exists() else str(ROOT))
        if folder: self.out_input.setText(str(Path(folder)))

    def _refresh_folder_list(self):
        if self._before_root.exists():
            dirs = sorted([d.name for d in self._before_root.iterdir() if d.is_dir()])
            if dirs:
                d = dirs[0]; s = self._before_root / d; o = self._after_root / d
                self.src_input.setText(str(s)); self.out_input.setText(str(o))
                self.input_info.setText(f' {s}'); self._refresh_video_list(s); return
        self.src_input.clear(); self.out_input.clear()
        if not self._before_root.exists():
            self.input_info.setText(f' 目录不存在: {self._before_root}')
            self.video_list.clear(); self.video_count.setText('0 videos')

    def _refresh_video_list(self, folder: Path):
        self.video_list.clear()
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in VIDEO_EXTS and f.is_file():
                self.video_list.addItem(f.name)
        self.video_count.setText(f'{self.video_list.count()} videos')

    def _start(self):
        s = self.src_input.text().strip(); o = self.out_input.text().strip()
        if not s or not Path(s).exists():
            QMessageBox.warning(self, 'Warning', '请选择有效输入目录'); return
        if not o: QMessageBox.warning(self, 'Warning', '请选择输出目录'); return
        exts = VIDEO_EXTS
        videos = [f for f in Path(s).iterdir() if f.suffix.lower() in exts and f.is_file()]
        if not videos: QMessageBox.warning(self, 'Warning', '未找到视频文件'); return
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.src_input.setEnabled(False); self.out_input.setEnabled(False)
        self.progress_bar.setValue(0); self._pv.setText('—'); self._pf.setText('—')
        self.log_panel.clear()
        self._log(f'▶ 开始预处理: {Path(s).name}')
        self._worker = VideoPreprocessWorker(src_folder=s, out_folder=o)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(lambda c, t: (
            self.progress_bar.setValue(int(c / t * 100)), self._pv.setText(f'{c}/{t}')))
        self._worker.video_progress.connect(lambda c, t: self._pf.setText(f'{c}'))
        self._worker.image_saved.connect(self._on_image_saved)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop(); self.stop_btn.setEnabled(False); self._log(' 停止中...')

    def _on_image_saved(self, image_path: str):
        try:
            c = int(self.preview_stats.text().split()[0]) + 1
            self.preview_stats.setText(f'{c} images processed')
            from PyQt5.QtGui import QPixmap
            p = QPixmap(image_path)
            if not p.isNull():
                self.preview_label.setPixmap(p.scaled(
                    self.preview_label.width(), self.preview_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.preview_label.setText('')
        except: pass

    def _on_done(self, ok, msg):
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.src_input.setEnabled(True); self.out_input.setEnabled(True)
        self.progress_bar.setValue(100 if ok else 0)
        self._log(f'{"" if ok else ""} {msg}')
        self.status_label.setText(msg); self._worker = None

    def _log(self, msg):
        self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), msg))
