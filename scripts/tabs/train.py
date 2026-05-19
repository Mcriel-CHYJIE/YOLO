"""训练标签页"""
from scripts.tabs.base import *

PROG_RE = re.compile(r'\d+%\s+\d+/\d+')
HTML_RE = re.compile(r'<[^>]+>')


class TrainTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.trainer = None
        self._build()

    def _build(self):
        lo = QHBoxLayout(self); lo.setContentsMargins(4,4,4,4); lo.setSpacing(6)

        # ── 左栏 ──
        left = QWidget(); left.setFixedWidth(340)
        ll = QVBoxLayout(left); ll.setSpacing(5); ll.setContentsMargins(0,0,0,0)

        g1 = QGroupBox('Configuration')
        g1l = QGridLayout(g1); g1l.setSpacing(3); g1l.setContentsMargins(6,12,6,6)
        g1l.setColumnStretch(0,1); g1l.setColumnStretch(1,1)
        t = cfg['training']
        self.m = QComboBox(); self.m.addItems(t['model_options'])
        self.m.setCurrentText(t['model'])
        labeled_field(self.m,'Model',g1l,0,0)
        self.ep = QSpinBox(); self.ep.setRange(1,2000); self.ep.setValue(t['epochs'])
        labeled_field(self.ep,'Epochs',g1l,0,1)
        self.bs = QSpinBox(); self.bs.setRange(1,256); self.bs.setValue(t['batch'])
        labeled_field(self.bs,'Batch',g1l,1,0)
        self.sz = QComboBox(); self.sz.addItems([str(v) for v in t['imgsz_options']])
        self.sz.setCurrentText(str(t['imgsz']))
        labeled_field(self.sz,'ImgSz',g1l,1,1)
        self.opt = QComboBox(); self.opt.addItems(t['optimizer_options'])
        self.opt.setCurrentText(t['optimizer'])
        labeled_field(self.opt,'Optimizer',g1l,2,0)
        self.dev = QComboBox(); self.dev.addItems(['GPU','CPU'])
        labeled_field(self.dev,'Device',g1l,2,1)
        self.sch = QComboBox(); self.sch.addItems(t['scheduler_options'])
        self.sch.setCurrentText(t['scheduler'].capitalize())
        labeled_field(self.sch,'Schedule',g1l,3,0)
        self.pt = QSpinBox(); self.pt.setRange(5,500); self.pt.setValue(t['patience'])
        labeled_field(self.pt,'Patience',g1l,3,1)
        self.lr0 = QDoubleSpinBox(); self.lr0.setRange(1e-6,1); self.lr0.setDecimals(6); self.lr0.setValue(t['lr0'])
        labeled_field(self.lr0,'LR',g1l,4,0)
        self.lrf = QDoubleSpinBox(); self.lrf.setRange(1e-6,1); self.lrf.setDecimals(6); self.lrf.setValue(t['lrf'])
        labeled_field(self.lrf,'LR Final',g1l,4,1)
        self.wu = QSpinBox(); self.wu.setRange(0,50); self.wu.setValue(t['warmup_epochs'])
        labeled_field(self.wu,'Warmup',g1l,5,0)
        self.wk = QSpinBox(); self.wk.setRange(1,32); self.wk.setValue(min(t['workers'], self.studio.cpu_count))
        labeled_field(self.wk,'Workers',g1l,5,1)
        if not self.studio.gpu_ok: self.dev.setCurrentIndex(1)
        else:
            self.dev.setCurrentText('GPU' if t['device'] in ('auto', '0', 'GPU') else 'CPU')
        ll.addWidget(g1)

        g2 = QGroupBox('Algorithm')
        g2l = QGridLayout(g2); g2l.setSpacing(3); g2l.setContentsMargins(6,12,6,6)
        g2l.setColumnStretch(0,1); g2l.setColumnStretch(1,1)
        a = cfg['training']
        self.iou_thresh = QDoubleSpinBox(); self.iou_thresh.setRange(0.1,0.9); self.iou_thresh.setDecimals(2); self.iou_thresh.setSingleStep(0.05)
        self.iou_thresh.setValue(a['iou'])
        labeled_field(self.iou_thresh,'IoU',g2l,0,0)
        self.cm = QSpinBox(); self.cm.setRange(0,100); self.cm.setValue(a['close_mosaic'])
        labeled_field(self.cm,'Close Mosaic',g2l,0,1)
        self.cp = QDoubleSpinBox(); self.cp.setRange(0,1); self.cp.setDecimals(2); self.cp.setSingleStep(.1)
        self.cp.setValue(a['copy_paste'])
        labeled_field(self.cp,'Copy-Paste',g2l,1,0)
        self.dg = QDoubleSpinBox(); self.dg.setRange(0,45); self.dg.setDecimals(1); self.dg.setSingleStep(5)
        self.dg.setValue(a['degrees'])
        labeled_field(self.dg,'Rotation',g2l,1,1)
        cw = QWidget(); cw.setStyleSheet(f'background:{BG};border-radius:4px;')
        cl = QHBoxLayout(cw); cl.setContentsMargins(6,1,6,1); cl.setSpacing(4)
        lbl = QLabel('Multi-Scale'); lbl.setStyleSheet(f'font-size:9px;color:{TEXT};background:transparent;font-weight:500;')
        self.ms = QCheckBox(); self.ms.setChecked(a['multi_scale'])
        cl.addWidget(lbl); cl.addWidget(self.ms); cl.addStretch()
        g2l.addWidget(cw,2,0)
        hw = QWidget(); hw.setStyleSheet(f'background:#eef2ff;border-radius:4px;')
        hwl = QHBoxLayout(hw); hwl.setContentsMargins(6,1,6,1); hwl.setSpacing(0)
        ht = QLabel(cfg['project']['tip'])
        ht.setStyleSheet(f'font-size:8px;font-weight:600;color:{PRI};background:transparent;')
        hwl.addWidget(ht); g2l.addWidget(hw,2,1)
        ll.addWidget(g2)

        self._params = [self.m, self.ep, self.bs, self.sz, self.opt, self.dev,
            self.sch, self.pt, self.lr0, self.lrf, self.wu, self.wk,
            self.iou_thresh, self.cm, self.cp, self.dg, self.ms]

        g3 = QGroupBox('Control')
        g3l = QVBoxLayout(g3); g3l.setSpacing(4); g3l.setContentsMargins(8,12,8,8)
        br = QHBoxLayout(); br.setSpacing(5)
        self.bs1 = QPushButton('Start'); self.bs1.setObjectName('pri'); self.bs1.clicked.connect(self._s)
        self.bs2 = QPushButton('Stop'); self.bs2.setObjectName('danger'); self.bs2.setEnabled(False); self.bs2.clicked.connect(self._st)
        br.addWidget(self.bs1,1); br.addWidget(self.bs2,1)
        g3l.addLayout(br)
        self.pg = QProgressBar(); self.pg.setTextVisible(False); g3l.addWidget(self.pg)

        mr = QHBoxLayout(); mr.setSpacing(8)
        self._me_card = MetricCard('Epoch', TEXT)
        self._me = self._me_card.value_label; mr.addWidget(self._me_card, 1)
        self._mm_card = MetricCard('mAP@0.5', GREEN, '—')
        self._mm = self._mm_card.value_label; mr.addWidget(self._mm_card, 1)
        self._mb_card = MetricCard('Best', PRI, '—')
        self._mb = self._mb_card.value_label; mr.addWidget(self._mb_card, 1)
        g3l.addLayout(mr)
        ll.addWidget(g3)
        lo.addWidget(left)

        # ── 右栏 ──
        right = QWidget()
        rl = QVBoxLayout(right); rl.setSpacing(5); rl.setContentsMargins(0,0,0,0)
        mid = QWidget()
        ml2 = QHBoxLayout(mid); ml2.setSpacing(5); ml2.setContentsMargins(0,0,0,0)
        for title, cls, col in [('Loss', LossChart, RED), ('mAP', MapChart, GREEN)]:
            cw = QWidget(); cw.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
            cl0 = QVBoxLayout(cw); cl0.setContentsMargins(8,4,8,2); cl0.setSpacing(1)
            hh = QHBoxLayout()
            hh.addWidget(QLabel('●', styleSheet=f'color:{col};font-size:7px;'))
            hh.addWidget(QLabel(title, styleSheet='font-weight:600;font-size:10px;')); hh.addStretch(); cl0.addLayout(hh)
            ch = cls(self); cl0.addWidget(ch); ml2.addWidget(cw)
            if title == 'Loss': self.lc = ch
            else: self.mc = ch
        rl.addWidget(mid, 3)

        # ── 下：日志 + 系统监控 ──
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0,0,0,0)
        bottom_layout.setSpacing(5)

        self.log_panel = LogPanel('● Console', max_lines=500)
        bottom_layout.addWidget(self.log_panel, 3)

        # 右：系统监控面板
        mw = QWidget(); mw.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        mw.setFixedWidth(100)
        mvl = QVBoxLayout(mw); mvl.setContentsMargins(8,8,8,8); mvl.setSpacing(4)

        title = QLabel('● System')
        title.setStyleSheet(f'font-size:10px;font-weight:600;color:{TEXT3};')
        mvl.addWidget(title)

        monitors = [
            ('CPU', 'CPU'),
            ('MEM', 'Memory'),
            ('DSK', 'Disk'),
        ]
        if self.studio.gpu_ok:
            monitors.append(('VRM', 'VRAM'))
            monitors.append(('GPU', 'GPU'))

        for key, label in monitors:
            item = QWidget()
            item.setStyleSheet('background:transparent;border:none;')
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(0,0,0,0)
            item_layout.setSpacing(0)
            lbl = QLabel(label)
            lbl.setStyleSheet(f'font-size:10px;font-weight:500;color:{TEXT2};')
            lbl.setAlignment(Qt.AlignLeft)
            lbl.setFixedWidth(35)
            item_layout.addWidget(lbl)
            val = QLabel('0%')
            val.setStyleSheet(f'font-size:10px;font-weight:600;color:{TEXT};')
            val.setAlignment(Qt.AlignRight)
            item_layout.addWidget(val, 1)
            mvl.addWidget(item)
            self.studio._sys_data[key] = [val]

        bottom_layout.addWidget(mw)
        rl.addWidget(bottom, 2)
        lo.addWidget(right, 1)

    def _config(self):
        s = lambda w: w.currentText()
        lr0_val = self.lr0.value()
        if lr0_val <= 0: lr0_val = 0.001
        g = cfg['training']
        return dict(model=s(self.m), epochs=self.ep.value(), batch=self.bs.value(), imgsz=int(s(self.sz)),
            lr0=lr0_val, lrf=self.lrf.value(), optimizer=s(self.opt), patience=self.pt.value(),
            device='0' if self.studio.gpu_ok and self.dev.currentIndex()==0 else 'cpu',
            cos_lr=self.sch.currentIndex()==0, warmup_epochs=self.wu.value(), workers=self.wk.value(),
            iou=self.iou_thresh.value(),
            close_mosaic=self.cm.value(), copy_paste=self.cp.value(), degrees=self.dg.value(),
            multi_scale=self.ms.isChecked(),
            hsv_h=g.get('hsv_h', 0.015), hsv_s=g.get('hsv_s', 0.7), hsv_v=g.get('hsv_v', 0.4),
            translate=g.get('translate', 0.15), scale=g.get('scale', 0.6))

    def _s(self):
        if self.trainer and self.trainer.isRunning():
            QMessageBox.warning(self, 'Warning', 'Training is in progress, please stop the current training first!'); return
        self.trainer = None
        self.bs1.setEnabled(False); self.bs2.setEnabled(True); self._enable_params(False)
        self.log_panel.clear(); self.pg.setValue(0)
        self._me.setText('0'); self._mm.setText('—'); self._mb.setText('—')
        self.lc.upd({}); self.mc.upd({})
        cfg = self._config()
        self._log(f'🚀 {cfg["model"]} | {cfg["epochs"]}ep | batch={cfg["batch"]}')
        self.trainer = Trainer(cfg)
        self.trainer.log.connect(self._log)
        self.trainer.status.connect(lambda t, p, b, c: (
            self._me.setText(t.split('/')[0].replace('Epoch ','')),
            self._mm.setText(f'{c:.4f}' if c>0 else '—'),
            self._mb.setText(f'{b:.4f}' if b>0 else '—'),
            self.pg.setValue(int(p*100))))
        self.trainer.chart.connect(lambda: (self.trainer and (self.lc.upd(self.trainer.history), self.mc.upd(self.trainer.history))))
        self.trainer.done.connect(self._dn)
        self.trainer.start()

    def _st(self):
        if self.trainer and self.trainer.isRunning():
            self.trainer.stop(); self.bs2.setEnabled(False); self.bs2.setText('Stopping…')
            self._log('⏳  Stopping after current epoch…')

    def _dn(self, ok, msg):
        self._log(msg); self.bs1.setEnabled(True); self.bs2.setEnabled(True); self.bs2.setText('Stop')
        self._enable_params(True); self.trainer = None
        if ok: self.pg.setValue(100)

    def _enable_params(self, on):
        for w in self._params: w.setEnabled(on)
        if on and not self.studio.gpu_ok: self.dev.setCurrentIndex(1)

    def _log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        is_prog = bool(PROG_RE.search(msg))
        if is_prog and self.log_panel._log_lines:
            last = self.log_panel._log_lines[-1]
            text = HTML_RE.sub('', last)
            if PROG_RE.search(text):
                self.log_panel.replace_last(format_log(ts, msg))
                return
        self.log_panel.append(format_log(ts, msg))
