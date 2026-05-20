"""推理标签页 — 融合实时视频检测 + 批量图片预测"""
from scripts.tabs.base import *
from PyQt5 import uic
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QCursor
from workers.detector import Detector


class PredictTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._model_path = None
        self._source_path = None
        self._worker = None
        self._mode = None
        self._image_results = []
        self._current_image_idx = 0
        self._build()
        self._init_widgets()
        self._connect_signals()
        self._load_latest()

    def _build(self):
        ui_path = Path(__file__).resolve().parent.parent / 'ui' / 'predict.ui'
        uic.loadUi(str(ui_path), self)

    def _init_widgets(self):
        # 替换 MetricCard 占位
        for placeholder, label, color, attr_card in [
            (self.imgCardP, 'Images/Frame', PRI, '_img_card'),
            (self.detCardP, 'Detections', GREEN, '_det_card'),
            (self.fallCardP, 'Fallen', RED, '_fal_card'),
        ]:
            card = MetricCard(label, color, '0')
            idx = self.statsRowP.indexOf(placeholder)
            self.statsRowP.removeWidget(placeholder)
            placeholder.deleteLater()
            self.statsRowP.insertWidget(idx, card, 1)
            setattr(self, attr_card, card)
        self._st_imgs = self._img_card.value_label
        self._st_dets = self._det_card.value_label
        self._st_fall = self._fal_card.value_label

        # 构建右侧 QStackedWidget 各页面
        self.video_view = QLabel('Load model & source then start')
        self.video_view.setAlignment(Qt.AlignCenter)
        self.video_view.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:7px;font-size:14px;')
        self.video_view.setMinimumHeight(400)
        self.stack.addWidget(self.video_view)

        self.preview_widget = QWidget()
        pl = QVBoxLayout(self.preview_widget); pl.setContentsMargins(0, 0, 0, 0); pl.setSpacing(4)
        self.image_view = QLabel('No image to display')
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:7px;font-size:14px;')
        self.image_view.setMinimumHeight(350)
        pl.addWidget(self.image_view, 1)
        nw = QWidget()
        nl = QHBoxLayout(nw); nl.setContentsMargins(4, 4, 4, 4); nl.setSpacing(8)
        self.btn_prev = QPushButton('◀ Previous'); self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._prev_image); self.btn_prev.setStyleSheet('padding:4px 12px;')
        self.btn_next = QPushButton('Next ▶'); self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._next_image); self.btn_next.setStyleSheet('padding:4px 12px;')
        self.lbl_image_index = QLabel('0 / 0')
        self.lbl_image_index.setStyleSheet(f'font-size:11px;font-weight:600;color:{TEXT};min-width:80px;')
        self.lbl_image_index.setAlignment(Qt.AlignCenter)
        nl.addWidget(self.btn_prev); nl.addWidget(self.lbl_image_index, 1); nl.addWidget(self.btn_next)
        pl.addWidget(nw)
        self.stack.addWidget(self.preview_widget)

        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            f'QTextEdit{{background:{CON};color:{CON_T};border:none;border-radius:7px;padding:10px 12px;font-family:"Consolas","Courier New",monospace;font-size:13px;}}')
        self.stack.addWidget(self.log_view)
        self.stack.setCurrentIndex(0)

    def _connect_signals(self):
        self.modelBrowseBtn.clicked.connect(self._browse_model)
        self.srcBrowseBtn.clicked.connect(self._browse_src)
        self.btn_start.setObjectName('pri')
        self.btn_start.clicked.connect(self._toggle_start)
        self.btn_stop.setObjectName('danger')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_run)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_run.setObjectName('pri')
        self.btn_run.clicked.connect(self._run_batch)

    def _load_latest(self):
        w = find_latest_best()
        if w: self.lbl_model.setText(f'[{Path(w).name}]'); self._model_path = Path(w)

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Select Model', 'runs', MODEL_FILTER)
        if p: self._model_path = Path(p); self.lbl_model.setText(f'[{self._model_path.name}]')

    def _browse_src(self):
        menu = QMenu(self)
        va = menu.addAction('Video File'); ia = menu.addAction('Image File'); fa = menu.addAction('Image Folder')
        act = menu.exec_(QCursor.pos())
        if act == va:
            p, _ = QFileDialog.getOpenFileName(self, 'Select Video', '', 'Video Files (*.mp4 *.avi *.mov *.mkv)')
            if p: self.lbl_src.setText(p); self._set_mode(p)
        elif act == ia:
            p, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.jpg *.png *.jpeg)')
            if p: self.lbl_src.setText(p); self._set_mode(p)
        elif act == fa:
            d = QFileDialog.getExistingDirectory(self, 'Select Image Folder')
            if d: self.lbl_src.setText(d); self._set_mode(d)

    def _set_mode(self, p):
        if not p: return
        pp = Path(p)
        if pp.is_file() and pp.suffix.lower() in VIDEO_EXTS:
            self._mode = 'video'; self._source_path = pp; self._show_video_controls()
        elif pp.is_file() and pp.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            self._mode = 'image'; self._source_path = pp; self._show_batch_controls()
        elif pp.is_dir():
            self._mode = 'folder'; self._source_path = pp; self._show_batch_controls()

    def _show_video_controls(self):
        self.stack.setCurrentIndex(0)
        for w in [self.btn_start, self.btn_stop, self.btn_pause, self.lbl_fps]: w.setVisible(True)
        self.btn_run.setVisible(False); self.slider.setEnabled(True)

    def _show_batch_controls(self):
        self.stack.setCurrentIndex(1)
        for w in [self.btn_start, self.btn_stop, self.btn_pause, self.lbl_fps]: w.setVisible(False)
        self.btn_run.setVisible(True); self.slider.setEnabled(False)

    def _toggle_start(self):
        if not self._model_path or not self._model_path.exists():
            QMessageBox.warning(self, 'Warning', 'Please select a model first'); return
        if not self._source_path or not self._source_path.exists():
            QMessageBox.warning(self, 'Warning', 'Please select a video first'); return
        if self._worker and self._worker.isRunning():
            self._worker.toggle_pause(); self.btn_pause.setText('Resume' if self._worker._pause else 'Pause'); return
        export_path = None
        if self.cb_export.isChecked():
            d = ROOT / 'output'; d.mkdir(parents=True, exist_ok=True)
            fn = str(d / f"detected_{self._source_path.stem}_{datetime.now().strftime('%m%d_%H%M')}.mp4")
            export_path, _ = QFileDialog.getSaveFileName(self, 'Save Detected Video', fn, 'Video Files (*.mp4 *.avi)')
            if not export_path: return
        self._worker = Detector(self._model_path, self._source_path, self.sp_conf.value(), self.sp_iou.value())
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.fps_updated.connect(lambda v: self.lbl_fps.setText(f'{v:.0f} FPS'))
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.finished.connect(self._on_finish)
        self._worker.log_signal.connect(lambda m: self.log.setText(m))
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_pause.setEnabled(True)
        self.btn_pause.setText('Pause'); self.lbl_fps.setText('')
        self._st_imgs.setText('0'); self._st_dets.setText('0'); self._st_fall.setText('0')
        self.lbl_cls.setText('Detecting...'); self._worker.export_path = export_path
        self._worker.start()

    def _toggle_pause(self):
        if self._worker and self._worker.isRunning():
            self._worker.toggle_pause(); self.btn_pause.setText('Resume' if self._worker._pause else 'Pause')

    def _stop_run(self):
        if self._worker and self._worker.isRunning(): self._worker.stop(); self.log.setText('Stopped')

    def _on_frame(self, frame, idx, total):
        h, w = frame.shape[:2]
        mw = self.video_view.width() - 10; mh = self.video_view.height() - 10
        sc = min(mw / max(w, 1), mh / max(h, 1), 1.0)
        if sc < 1.0: frame = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self.video_view.setPixmap(QPixmap.fromImage(qi)); self.video_view.setStyleSheet('')
        if total > 0: self.slider.setValue(int(idx / total * 100))
        self.lbl_pos.setText(f'{idx}/{total}'); self._st_imgs.setText(str(idx))

    def _on_stats(self, stats):
        t = sum(stats.values()); self._st_dets.setText(str(t))
        fallen = sum(c for n, c in stats.items() if n.lower() in ('fallen', 'fall', 'down'))
        self._st_fall.setText(str(fallen))
        lines = [f'{n}: {c}' for n, c in sorted(stats.items(), key=lambda x: -x[1])]
        self.lbl_cls.setText('\n'.join(lines) if lines else 'No detections')

    def _on_finish(self):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_pause.setEnabled(False)
        self.btn_start.setText('Restart'); self._worker = None

    def _run_batch(self):
        w = self._model_path; src = self._source_path
        if not w or not w.exists(): QMessageBox.warning(self, 'Warning', 'Please select a model first'); return
        if not src or not src.exists(): QMessageBox.warning(self, 'Warning', 'Please select images/folder first'); return
        self.log_view.clear(); self.btn_run.setEnabled(False)
        self.log_view.append('Running batch inference...'); QApplication.processEvents()
        try:
            from ultralytics import YOLO
            model = YOLO(str(w))
            save = self.cb_export.isChecked()
            results = model.predict(source=str(src), conf=self.sp_conf.value(), iou=self.sp_iou.value(),
                imgsz=cfg['predict']['imgsz'], device='0' if self.studio.gpu_ok else 'cpu',
                save=save, save_txt=False, verbose=False,
                project=str(ROOT / 'output'), name=f'predict_{datetime.now().strftime("%m%d_%H%M")}')
            self._image_results = results; self._current_image_idx = 0
            total_imgs = len(results)
            total_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
            fallen = 0; cls_counts = {}
            for r in results:
                if r.boxes is not None:
                    for cid in r.boxes.cls:
                        nm = model.names.get(int(cid), f'cls_{int(cid)}')
                        cls_counts[nm] = cls_counts.get(nm, 0) + 1
                        if nm.lower() in ('fallen', 'fall', 'down'): fallen += 1
            self._st_imgs.setText(str(total_imgs)); self._st_dets.setText(str(total_dets))
            self._st_fall.setText(str(fallen))
            self.lbl_cls.setText('\n'.join(f'{n}: {c}' for n, c in sorted(cls_counts.items(), key=lambda x: -x[1])))
            self.log_view.append(f'Done! {total_imgs} sources, {total_dets} detections')
            if save and results and hasattr(results[0], 'save_dir'):
                self.log_view.append(f'   Saved: {results[0].save_dir}')
            if total_imgs > 0: self._show_image_preview(0)
        except Exception as e:
            import traceback; traceback.print_exc(); self.log_view.append(f'Error: {e}')
        finally: self.btn_run.setEnabled(True)

    def _show_image_preview(self, idx):
        if not self._image_results or idx < 0 or idx >= len(self._image_results): return
        r = self._image_results[idx]
        img = r.plot() if hasattr(r, 'plot') else cv2.imread(r.path) if hasattr(r, 'path') else None
        if img is None: self.image_view.setText('Cannot load image'); return
        h, w = img.shape[:2]
        sc = min((self.image_view.width() - 10) / max(w, 1), (self.image_view.height() - 10) / max(h, 1), 1.0)
        if sc < 1.0: img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image_view.setPixmap(QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0],
            rgb.strides[0], QImage.Format_RGB888)))
        self.image_view.setStyleSheet('')
        self._current_image_idx = idx
        self.lbl_image_index.setText(f'{idx + 1} / {len(self._image_results)}')
        self.btn_prev.setEnabled(idx > 0); self.btn_next.setEnabled(idx < len(self._image_results) - 1)

    def _prev_image(self):
        if self._current_image_idx > 0: self._show_image_preview(self._current_image_idx - 1)

    def _next_image(self):
        if self._current_image_idx < len(self._image_results) - 1:
            self._show_image_preview(self._current_image_idx + 1)
