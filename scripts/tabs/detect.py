"""检测标签页"""
from scripts.tabs.base import *
import cv2

class DetectTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._det_model = None; self._det_video = None; self._det_worker = None
        self._build()

    def _build(self):
        lo = QHBoxLayout(self); lo.setContentsMargins(4,4,4,4); lo.setSpacing(6)
        left = QWidget(); left.setFixedWidth(300)
        ll = QVBoxLayout(left); ll.setSpacing(5); ll.setContentsMargins(0,0,0,0)

        g1 = QGroupBox('Input')
        g1l = QVBoxLayout(g1); g1l.setSpacing(4); g1l.setContentsMargins(8,12,8,8)
        g1l.addWidget(QLabel('Model Weights', styleSheet=f'font-size:9px;color:{TEXT2};font-weight:500;'))
        mr = QHBoxLayout()
        self.dl_model = QLabel('Not selected'); self.dl_model.setStyleSheet(f'font-size:10px;color:{TEXT3};')
        mr.addWidget(self.dl_model,1)
        bm = QPushButton('Browse'); bm.clicked.connect(lambda: self._dbm()); mr.addWidget(bm)
        g1l.addLayout(mr)
        g1l.addWidget(QLabel('Video', styleSheet=f'font-size:9px;color:{TEXT2};font-weight:500;'))
        vr = QHBoxLayout()
        self.dl_video = QLabel('Not selected'); self.dl_video.setStyleSheet(f'font-size:10px;color:{TEXT3};')
        vr.addWidget(self.dl_video,1)
        bv = QPushButton('Browse'); bv.clicked.connect(lambda: self._dbv()); vr.addWidget(bv)
        g1l.addLayout(vr); ll.addWidget(g1)

        g2 = QGroupBox('Parameters')
        g2l = QGridLayout(g2); g2l.setSpacing(3); g2l.setContentsMargins(6,12,6,6)
        g2l.setColumnStretch(0,1); g2l.setColumnStretch(1,1)
        self.dc_conf = QDoubleSpinBox(); self.dc_conf.setRange(0.01,0.99); self.dc_conf.setValue(0.25); self.dc_conf.setSingleStep(0.05); self.dc_conf.setDecimals(2)
        self.dc_iou = QDoubleSpinBox(); self.dc_iou.setRange(0.01,0.99); self.dc_iou.setValue(0.45); self.dc_iou.setSingleStep(0.05); self.dc_iou.setDecimals(2)
        for name, w, rc in [('Conf',self.dc_conf,(0,0)),('IoU',self.dc_iou,(0,1))]:
            cw = QWidget(); cw.setStyleSheet(f'background:{BG};border-radius:4px;')
            cl = QHBoxLayout(cw); cl.setContentsMargins(6,1,6,1); cl.setSpacing(4)
            cl.addWidget(QLabel(name, styleSheet=f'font-size:9px;color:{TEXT};font-weight:500;'))
            w.setMinimumHeight(22); cl.addWidget(w,1); g2l.addWidget(cw,*rc)
        ll.addWidget(g2)

        g3 = QGroupBox('Control')
        g3l = QVBoxLayout(g3); g3l.setSpacing(4); g3l.setContentsMargins(8,12,8,8)
        br = QHBoxLayout()
        self.det_play = QPushButton('▶ Start'); self.det_play.setObjectName('pri'); self.det_play.clicked.connect(self._det_toggle)
        self.det_stop = QPushButton('⏹ Stop'); self.det_stop.setObjectName('danger'); self.det_stop.setEnabled(False); self.det_stop.clicked.connect(self._det_stop)
        br.addWidget(self.det_play,1); br.addWidget(self.det_stop,1)
        g3l.addLayout(br)
        self.det_pause = QPushButton('⏸ Pause'); self.det_pause.setEnabled(False); self.det_pause.clicked.connect(self._det_pause)
        g3l.addWidget(self.det_pause); ll.addWidget(g3)

        g4 = QGroupBox('Stats')
        g4l = QVBoxLayout(g4); g4l.setSpacing(3); g4l.setContentsMargins(8,12,8,8)
        sr = QHBoxLayout()
        for lbl,col,attr in [('FPS',PRI,'_det_fps'),('Frame',GREEN,'_det_frame'),('Total',RED,'_det_total')]:
            cw = QWidget(); cw.setStyleSheet(f'background:{BG};border-radius:5px;')
            cl2 = QVBoxLayout(cw); cl2.setContentsMargins(6,4,6,4); cl2.setSpacing(0)
            cl2.addWidget(QLabel(lbl, styleSheet=f'font-size:7px;color:{TEXT3};font-weight:500;qproperty-alignment:AlignCenter;'))
            v = QLabel('0'); v.setStyleSheet(f'font-size:18px;font-weight:700;color:{col};qproperty-alignment:AlignCenter;')
            setattr(self,attr,v); cl2.addWidget(v); sr.addWidget(cw,1)
        g4l.addLayout(sr)
        self.det_cls = QLabel('Awaiting...'); self.det_cls.setWordWrap(True)
        self.det_cls.setStyleSheet(f'font-size:10px;color:{TEXT3};padding:2px;')
        g4l.addWidget(self.det_cls); ll.addWidget(g4)

        # 进度条
        pr = QHBoxLayout()
        self.det_pos = QLabel('0 / 0'); self.det_pos.setStyleSheet(f'font-size:10px;color:{TEXT3};min-width:60px;')
        pr.addWidget(self.det_pos)
        self.det_slider = QSlider(Qt.Horizontal); self.det_slider.setEnabled(False)
        pr.addWidget(self.det_slider,1); ll.addLayout(pr)

        # 日志
        self.det_log = QLabel('Ready'); self.det_log.setWordWrap(True)
        self.det_log.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:5px;padding:4px 8px;font-family:Consolas;font-size:11px;min-height:24px;')
        ll.addWidget(self.det_log)

        ll.addStretch(); lo.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right); rl.setSpacing(4); rl.setContentsMargins(0,0,0,0)
        self.det_video = QLabel('Load model & video then start')
        self.det_video.setAlignment(Qt.AlignCenter)
        self.det_video.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:7px;font-size:14px;')
        self.det_video.setMinimumHeight(400)
        rl.addWidget(self.det_video,1); lo.addWidget(right,1)

    def _dbm(self):
        p,_ = QFileDialog.getOpenFileName(self,'Select Model','runs','PyTorch (*.pt)')
        if p: self._det_model=Path(p); self.dl_model.setText(f'✅ {self._det_model.name}')
    def _dbv(self):
        p,_ = QFileDialog.getOpenFileName(self,'Select Video','','Video (*.mp4 *.avi *.mov *.mkv)')
        if p: self._det_video=Path(p); self.dl_video.setText(f'✅ {self._det_video.name}')
    def _det_toggle(self):
        if not self._det_model or not self._det_model.exists():
            QMessageBox.warning(self,'Warning','Please select a model first'); return
        if not self._det_video or not self._det_video.exists():
            QMessageBox.warning(self,'Warning','Please select a video first'); return
        if self._det_worker and self._det_worker.isRunning():
            self._det_worker.toggle_pause()
            self.det_pause.setText('▶ Resume' if self._det_worker._pause else '⏸ Pause'); return
        self._det_worker = DetectWorker(self._det_model,self._det_video,self.dc_conf.value(),self.dc_iou.value())
        self._det_worker.frame_ready.connect(self._det_frame)
        self._det_worker.fps_updated.connect(lambda v: self._det_fps.setText(f'{v:.0f}'))
        self._det_worker.stats_updated.connect(self._det_stats)
        self._det_worker.finished.connect(self._det_fin)
        self._det_worker.log_signal.connect(lambda m: self.det_log.setText(m))
        self.det_play.setEnabled(False); self.det_stop.setEnabled(True); self.det_pause.setEnabled(True)
        self.det_pause.setText('⏸ Pause')
        self._det_total.setText('0'); self._det_frame.setText('0'); self._det_fps.setText('0')
        self.det_cls.setText('Detecting...'); self._det_worker.start()
    def _det_stop(self):
        if self._det_worker and self._det_worker.isRunning():
            self._det_worker.stop(); self.det_log.setText('🛑 Stopped')
    def _det_pause(self):
        if self._det_worker and self._det_worker.isRunning():
            self._det_worker.toggle_pause()
            self.det_pause.setText('▶ Resume' if self._det_worker._pause else '⏸ Pause')
    def _det_frame(self, frame, idx, total):
        h,w = frame.shape[:2]
        mw = self.det_video.width()-10; mh = self.det_video.height()-10
        sc = min(mw/max(w,1), mh/max(h,1), 1.0)
        if sc < 1.0: frame = cv2.resize(frame, (int(w*sc),int(h*sc)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self.det_video.setPixmap(QPixmap.fromImage(qi)); self.det_video.setStyleSheet('')
        if total>0: self.det_slider.setValue(int(idx/total*100))
        self.det_pos.setText(f'{idx}/{total}'); self._det_frame.setText(str(idx))
    def _det_stats(self, stats):
        t = sum(stats.values()); self._det_total.setText(str(t))
        lines = [f'{n}: {c}' for n,c in sorted(stats.items(),key=lambda x:-x[1])]
        self.det_cls.setText('\n'.join(lines) if lines else 'No detections')
    def _det_fin(self):
        self.det_play.setEnabled(True); self.det_stop.setEnabled(False); self.det_pause.setEnabled(False)
        self.det_play.setText('▶ Restart'); self._det_worker = None
