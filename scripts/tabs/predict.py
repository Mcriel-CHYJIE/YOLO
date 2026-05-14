"""推理标签页 — 融合实时视频检测 + 批量图片预测"""
from scripts.tabs.base import *
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QCursor


class PredictTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._model_path = None
        self._source_path = None
        self._worker = None
        self._mode = None
        def _card(label=None):
            cw = QWidget(); cw.setStyleSheet(f'background:{BG};border-radius:4px;')
            cl = QHBoxLayout(cw); cl.setContentsMargins(6,1,6,1); cl.setSpacing(4)
            if label:
                lbl = QLabel(label); lbl.setStyleSheet(f'font-size:9px;color:{TEXT};background:transparent;font-weight:500;')
                lbl.setFixedHeight(22); cl.addWidget(lbl)
            return cw, cl

        lo = QHBoxLayout(self); lo.setContentsMargins(4, 4, 4, 4); lo.setSpacing(6)
        left = QWidget(); left.setFixedWidth(280)
        ll = QVBoxLayout(left); ll.setSpacing(4); ll.setContentsMargins(0, 0, 0, 0)

        # Input
        g1 = QGroupBox('Input')
        g1l = QVBoxLayout(g1); g1l.setSpacing(4); g1l.setContentsMargins(8,12,8,8)
        g1l.addWidget(QLabel('Model Weights', styleSheet=f'font-size:9px;color:{TEXT2};font-weight:500;'))
        mr = QHBoxLayout()
        self.lbl_model = QLabel('Not selected'); self.lbl_model.setStyleSheet(f'font-size:10px;color:{TEXT3};')
        mr.addWidget(self.lbl_model,1)
        btn_model = QPushButton('Browse'); btn_model.clicked.connect(self._browse_model); mr.addWidget(btn_model)
        g1l.addLayout(mr)
        g1l.addWidget(QLabel('Source', styleSheet=f'font-size:9px;color:{TEXT2};font-weight:500;'))
        sr = QHBoxLayout()
        self.lbl_src = QLineEdit(''); self.lbl_src.setPlaceholderText('Image / Video / Folder')
        sr.addWidget(self.lbl_src,1)
        btn_src = QPushButton('Browse'); btn_src.clicked.connect(self._browse_src); sr.addWidget(btn_src)
        g1l.addLayout(sr); ll.addWidget(g1)

        # Parameters
        g2 = QGroupBox('Parameters')
        g2l = QGridLayout(g2); g2l.setSpacing(3); g2l.setContentsMargins(6,12,6,6)
        g2l.setColumnStretch(0,1); g2l.setColumnStretch(1,1)
        p = cfg['predict']
        self.sp_conf = QDoubleSpinBox(); self.sp_conf.setRange(0.01,0.99); self.sp_conf.setValue(p['conf'])
        self.sp_conf.setSingleStep(0.05); self.sp_conf.setDecimals(2)
        self.sp_iou = QDoubleSpinBox(); self.sp_iou.setRange(0.01,0.99); self.sp_iou.setValue(p['iou'])
        self.sp_iou.setSingleStep(0.05); self.sp_iou.setDecimals(2)
        for name, w, rc in [('Conf',self.sp_conf,(0,0)),('IoU',self.sp_iou,(0,1))]:
            cw, cl = _card(name)
            w.setMinimumHeight(22); cl.addWidget(w,1); g2l.addWidget(cw,*rc)
        ll.addWidget(g2)

        # Control
        g3 = QGroupBox('Control')
        g3l = QVBoxLayout(g3); g3l.setSpacing(4); g3l.setContentsMargins(8,12,8,8)
        # Video controls
        vc = QHBoxLayout()
        self.btn_start = QPushButton('Start'); self.btn_start.setObjectName('pri')
        self.btn_start.clicked.connect(self._toggle_start)
        self.btn_stop = QPushButton('Stop'); self.btn_stop.setObjectName('danger')
        self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._stop_run)
        vc.addWidget(self.btn_start,1); vc.addWidget(self.btn_stop,1)
        g3l.addLayout(vc)
        self.btn_pause = QPushButton('Pause'); self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        g3l.addWidget(self.btn_pause)
        self.btn_run = QPushButton('Run Batch'); self.btn_run.setObjectName('pri')
        self.btn_run.clicked.connect(self._run_batch); g3l.addWidget(self.btn_run)
        # Stats
        st = QHBoxLayout()
        self._img_card = MetricCard('Images/Frame', PRI)
        self._st_imgs = self._img_card.value_label; st.addWidget(self._img_card, 1)
        self._det_card = MetricCard('Detections', GREEN)
        self._st_dets = self._det_card.value_label; st.addWidget(self._det_card, 1)
        self._fal_card = MetricCard('Fallen', RED)
        self._st_fall = self._fal_card.value_label; st.addWidget(self._fal_card, 1)
        g3l.addLayout(st)
        self.lbl_fps = QLabel(''); self.lbl_fps.setStyleSheet(f'font-size:10px;color:{TEXT3};')
        g3l.addWidget(self.lbl_fps)
        # Class breakdown
        self.lbl_cls = QLabel(''); self.lbl_cls.setWordWrap(True)
        self.lbl_cls.setStyleSheet(f'font-size:10px;color:{TEXT3};padding:2px;')
        g3l.addWidget(self.lbl_cls); ll.addWidget(g3)

        # Slider
        sl_w = QWidget()
        sl_l = QHBoxLayout(sl_w); sl_l.setContentsMargins(0,0,0,0); sl_l.setSpacing(4)
        self.lbl_pos = QLabel('0 / 0'); self.lbl_pos.setStyleSheet(f'font-size:10px;color:{TEXT3};min-width:60px;')
        sl_l.addWidget(self.lbl_pos)
        self.slider = QSlider(Qt.Horizontal); self.slider.setEnabled(False)
        sl_l.addWidget(self.slider,1); ll.addWidget(sl_w)

        # Log
        self.log = QLabel('Ready'); self.log.setWordWrap(True)
        self.log.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:5px;padding:4px 8px;font-family:Consolas;font-size:11px;min-height:24px;')
        ll.addWidget(self.log)
        ll.addStretch(); lo.addWidget(left)

        # ── Right Panel ─
        self.stack = QStackedWidget()
        # Page 0: Video display
        self.video_view = QLabel('Load model & source then start')
        self.video_view.setAlignment(Qt.AlignCenter)
        self.video_view.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:7px;font-size:14px;')
        self.video_view.setMinimumHeight(400)
        self.stack.addWidget(self.video_view)
        # Page 1: Batch results log
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(f'QTextEdit{{background:{CON};color:{CON_T};border:none;border-radius:7px;padding:10px 12px;font-family:"Consolas","Courier New",monospace;font-size:13px;}}')
        self.stack.addWidget(self.log_view)
        self.stack.setCurrentIndex(0)
        lo.addWidget(self.stack, 1)
        self._load_latest()

    # ── Helpers ──
    def _load_latest(self):
        w = find_latest_best()
        if w: self.lbl_model.setText(f'[{Path(w).name}]'); self._model_path = Path(w)

    def _browse_model(self):
        p,_ = QFileDialog.getOpenFileName(self,'Select Model','runs',MODEL_FILTER)
        if p: self._model_path = Path(p); self.lbl_model.setText(f'[{self._model_path.name}]')

    def _browse_src(self):
        menu = QMenu(self)
        video_act = menu.addAction('Video File')
        image_act = menu.addAction('Image File')
        folder_act = menu.addAction('Image Folder')
        act = menu.exec_(QCursor.pos())
        if act == video_act:
            p, _ = QFileDialog.getOpenFileName(self, 'Select Video', '', 'Video Files (*.mp4 *.avi *.mov *.mkv)')
            if p:
                self.lbl_src.setText(p)
                self._set_mode(p)
        elif act == image_act:
            p, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.jpg *.png *.jpeg)')
            if p:
                self.lbl_src.setText(p)
                self._set_mode(p)
        elif act == folder_act:
            d = QFileDialog.getExistingDirectory(self, 'Select Image Folder')
            if d and Path(d).exists():
                self.lbl_src.setText(d)
                self._set_mode(d)

    def _set_mode(self, path):
        if not path:
            return
        p = Path(path)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            self._mode = 'video'; self._source_path = p
            self._show_video_controls()
        elif p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'):
            self._mode = 'image'; self._source_path = p
            self._show_batch_controls()
        elif p.is_dir():
            self._mode = 'folder'; self._source_path = p
            self._show_batch_controls()

    def _show_video_controls(self):
        self.stack.setCurrentIndex(0)
        self.btn_start.setVisible(True); self.btn_stop.setVisible(True); self.btn_pause.setVisible(True)
        self.btn_run.setVisible(False)
        self.lbl_fps.setVisible(True)
        self.slider.setEnabled(True)

    def _show_batch_controls(self):
        self.stack.setCurrentIndex(1)
        self.btn_start.setVisible(False); self.btn_stop.setVisible(False); self.btn_pause.setVisible(False)
        self.btn_run.setVisible(True)
        self.lbl_fps.setVisible(False)
        self.slider.setEnabled(False)

    # ── Video ──
    def _toggle_start(self):
        if not self._model_path or not self._model_path.exists():
            QMessageBox.warning(self,'Warning','Please select a model first'); return
        if not self._source_path or not self._source_path.exists():
            QMessageBox.warning(self,'Warning','Please select a video first'); return
        if self._worker and self._worker.isRunning():
            self._worker.toggle_pause()
            self.btn_pause.setText('Resume' if self._worker._pause else 'Pause'); return
        self._worker = DetectWorker(self._model_path, self._source_path,
                                    self.sp_conf.value(), self.sp_iou.value())
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.fps_updated.connect(lambda v: self.lbl_fps.setText(f'{v:.0f} FPS'))
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.finished.connect(self._on_finish)
        self._worker.log_signal.connect(lambda m: self.log.setText(m))
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_pause.setEnabled(True)
        self.btn_pause.setText('Pause')
        self.lbl_fps.setText(''); self._st_imgs.setText('0'); self._st_dets.setText('0'); self._st_fall.setText('0')
        self.lbl_cls.setText('Detecting...')
        self._worker.start()

    def _toggle_pause(self):
        if self._worker and self._worker.isRunning():
            self._worker.toggle_pause()
            self.btn_pause.setText('Resume' if self._worker._pause else 'Pause')

    def _stop_run(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop(); self.log.setText('Stopped')

    def _on_frame(self, frame, idx, total):
        h,w = frame.shape[:2]
        mw = self.video_view.width()-10; mh = self.video_view.height()-10
        sc = min(mw/max(w,1), mh/max(h,1), 1.0)
        if sc < 1.0: frame = cv2.resize(frame, (int(w*sc),int(h*sc)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self.video_view.setPixmap(QPixmap.fromImage(qi)); self.video_view.setStyleSheet('')
        if total>0: self.slider.setValue(int(idx/total*100))
        self.lbl_pos.setText(f'{idx}/{total}'); self._st_imgs.setText(str(idx))

    def _on_stats(self, stats):
        t = sum(stats.values()); self._st_dets.setText(str(t))
        fallen = sum(c for n,c in stats.items() if n.lower() in ('fallen','fall','down'))
        self._st_fall.setText(str(fallen))
        lines = [f'{n}: {c}' for n,c in sorted(stats.items(),key=lambda x:-x[1])]
        self.lbl_cls.setText('\n'.join(lines) if lines else 'No detections')

    def _on_finish(self):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_pause.setEnabled(False)
        self.btn_start.setText('Restart'); self._worker = None

    # ── Batch ──
    def _run_batch(self):
        w = self._model_path; src = self._source_path
        if not w or not w.exists():
            QMessageBox.warning(self,'Warning','Please select a model first'); return
        if not src or not src.exists():
            QMessageBox.warning(self,'Warning','Please select images/folder first'); return
        self.log_view.clear(); self.btn_run.setEnabled(False)
        self.log_view.append('Running batch inference...'); QApplication.processEvents()
        try:
            from ultralytics import YOLO
            model = YOLO(str(w))
            results = model.predict(source=str(src), conf=self.sp_conf.value(), iou=self.sp_iou.value(),
                imgsz=cfg['predict']['imgsz'], device='0' if self.studio.gpu_ok else 'cpu',
                save=True, save_txt=False, verbose=False,
                project='runs', name=f'predict_{datetime.now().strftime("%m%d_%H%M")}')
            total_imgs = len(results)
            total_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
            fallen = 0; cls_counts = {}
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for cls_id in r.boxes.cls:
                        nm = model.names.get(int(cls_id), f'cls_{int(cls_id)}')
                        cls_counts[nm] = cls_counts.get(nm, 0) + 1
                        if nm.lower() in ('fallen','fall','down'): fallen += 1
            self._st_imgs.setText(str(total_imgs))
            self._st_dets.setText(str(total_dets)); self._st_fall.setText(str(fallen))
            lines = [f'{n}: {c}' for n,c in sorted(cls_counts.items(), key=lambda x:-x[1])]
            self.lbl_cls.setText('\n'.join(lines) if lines else 'No detections')
            self.log_view.append(f'Done! {total_imgs} sources, {total_dets} detections')
            if results and hasattr(results[0],'save_dir'): self.log_view.append(f'   Saved: {results[0].save_dir}')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_view.append(f'Error: {e}')
        finally: self.btn_run.setEnabled(True)
