"""训练标签页"""
from scripts.tabs.base import *

class TrainTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.trainer = None
        self._build()

    def _build(self):
        lo = QHBoxLayout(self); lo.setContentsMargins(4,4,4,4); lo.setSpacing(6)
        # ── 左栏 ──
        left = QWidget(); left.setFixedWidth(360)
        ll = QVBoxLayout(left); ll.setSpacing(5); ll.setContentsMargins(0,0,0,0)
        def _fi(w, label, r, c, grid=None):
            cw = QWidget(); cw.setStyleSheet(f'background:{BG};border-radius:4px;')
            cl = QHBoxLayout(cw); cl.setContentsMargins(6,1,6,1); cl.setSpacing(4)
            lbl = QLabel(label); lbl.setStyleSheet(f'font-size:9px;color:{TEXT};background:transparent;font-weight:500;')
            lbl.setFixedHeight(22); w.setMinimumHeight(22)
            cl.addWidget(lbl); cl.addWidget(w, 1); grid.addWidget(cw, r, c)

        g1 = QGroupBox('Configuration')
        g1l = QGridLayout(g1); g1l.setSpacing(3); g1l.setContentsMargins(6,12,6,6)
        g1l.setColumnStretch(0,1); g1l.setColumnStretch(1,1)
        t = cfg['training']
        self.m = QComboBox(); self.m.addItems(t['model_options'])
        self.m.setCurrentText(t['model'])
        _fi(self.m,'Model',0,0,grid=g1l)
        self.ep = QSpinBox(); self.ep.setRange(1,2000); self.ep.setValue(t['epochs'])
        _fi(self.ep,'Epochs',0,1,grid=g1l)
        self.bs = QSpinBox(); self.bs.setRange(1,256); self.bs.setValue(t['batch'])
        _fi(self.bs,'Batch',1,0,grid=g1l)
        self.sz = QComboBox(); self.sz.addItems([str(v) for v in t['imgsz_options']])
        self.sz.setCurrentText(str(t['imgsz']))
        _fi(self.sz,'ImgSz',1,1,grid=g1l)
        self.opt = QComboBox(); self.opt.addItems(t['optimizer_options'])
        self.opt.setCurrentText(t['optimizer'])
        _fi(self.opt,'Optimizer',2,0,grid=g1l)
        self.dev = QComboBox(); self.dev.addItems(['GPU','CPU'])
        _fi(self.dev,'Device',2,1,grid=g1l)
        self.sch = QComboBox(); self.sch.addItems(t['scheduler_options'])
        self.sch.setCurrentText(t['scheduler'].capitalize())
        _fi(self.sch,'Schedule',3,0,grid=g1l)
        self.pt = QSpinBox(); self.pt.setRange(5,500); self.pt.setValue(t['patience'])
        _fi(self.pt,'Patience',3,1,grid=g1l)
        self.lr0 = QDoubleSpinBox(); self.lr0.setRange(1e-6,1); self.lr0.setDecimals(6); self.lr0.setValue(t['lr0'])
        _fi(self.lr0,'LR',4,0,grid=g1l)
        self.lrf = QDoubleSpinBox(); self.lrf.setRange(1e-6,1); self.lrf.setDecimals(6); self.lrf.setValue(t['lrf'])
        _fi(self.lrf,'LR Final',4,1,grid=g1l)
        self.wu = QSpinBox(); self.wu.setRange(0,50); self.wu.setValue(t['warmup_epochs'])
        _fi(self.wu,'Warmup',5,0,grid=g1l)
        self.wk = QSpinBox(); self.wk.setRange(1,32); self.wk.setValue(min(t['workers'], self.studio.cpu_count))
        _fi(self.wk,'Workers',5,1,grid=g1l)
        if not self.studio.gpu_ok: self.dev.setCurrentIndex(1)
        else:
            self.dev.setCurrentText('GPU' if t['device'] in ('auto', '0', 'GPU') else 'CPU')
        ll.addWidget(g1)

        g2 = QGroupBox('Algorithm')
        g2l = QGridLayout(g2); g2l.setSpacing(3); g2l.setContentsMargins(6,12,6,6)
        g2l.setColumnStretch(0,1); g2l.setColumnStretch(1,1)
        a = cfg['training']
        self.fl = QDoubleSpinBox(); self.fl.setRange(0,5); self.fl.setDecimals(1); self.fl.setSingleStep(0.5)
        self.fl.setValue(a['fl_gamma'])
        _fi(self.fl,'Focal γ',0,0,grid=g2l)
        self.sm = QDoubleSpinBox(); self.sm.setRange(0,.5); self.sm.setDecimals(2); self.sm.setSingleStep(.05)
        self.sm.setValue(a['label_smoothing'])
        _fi(self.sm,'Smoothing',0,1,grid=g2l)
        self.iou_thresh = QDoubleSpinBox(); self.iou_thresh.setRange(0.1,0.9); self.iou_thresh.setDecimals(2); self.iou_thresh.setSingleStep(0.05)
        self.iou_thresh.setValue(a['iou'])
        _fi(self.iou_thresh,'IoU',1,0,grid=g2l)
        self.cm = QSpinBox(); self.cm.setRange(0,100); self.cm.setValue(a['close_mosaic'])
        _fi(self.cm,'Close Mosaic',1,1,grid=g2l)
        self.cp = QDoubleSpinBox(); self.cp.setRange(0,1); self.cp.setDecimals(2); self.cp.setSingleStep(.1)
        self.cp.setValue(a['copy_paste'])
        _fi(self.cp,'Copy-Paste',2,0,grid=g2l)
        self.dg = QDoubleSpinBox(); self.dg.setRange(0,45); self.dg.setDecimals(1); self.dg.setSingleStep(5)
        self.dg.setValue(a['degrees'])
        _fi(self.dg,'Rotation',2,1,grid=g2l)
        cw = QWidget(); cw.setStyleSheet(f'background:{BG};border-radius:4px;')
        cl = QHBoxLayout(cw); cl.setContentsMargins(6,1,6,1); cl.setSpacing(4)
        lbl = QLabel('Multi-Scale'); lbl.setStyleSheet(f'font-size:9px;color:{TEXT};background:transparent;font-weight:500;')
        self.ms = QCheckBox(); self.ms.setChecked(a['multi_scale'])
        cl.addWidget(lbl); cl.addWidget(self.ms); cl.addStretch()
        g2l.addWidget(cw,3,0)
        hw = QWidget(); hw.setStyleSheet(f'background:#eef2ff;border-radius:4px;')
        hwl = QHBoxLayout(hw); hwl.setContentsMargins(6,1,6,1); hwl.setSpacing(0)
        ht = QLabel(cfg['project']['tip'])
        ht.setStyleSheet(f'font-size:8px;font-weight:600;color:{PRI};background:transparent;')
        hwl.addWidget(ht); g2l.addWidget(hw,3,1)
        ll.addWidget(g2)

        self._params = [self.m, self.ep, self.bs, self.sz, self.opt, self.dev,
            self.sch, self.pt, self.lr0, self.lrf, self.wu, self.wk,
            self.fl, self.sm, self.iou_thresh, self.cm, self.cp, self.dg, self.ms]

        g3 = QGroupBox('Control')
        g3l = QVBoxLayout(g3); g3l.setSpacing(4); g3l.setContentsMargins(8,12,8,8)
        br = QHBoxLayout(); br.setSpacing(5)
        self.bs1 = QPushButton('Start'); self.bs1.setObjectName('pri'); self.bs1.clicked.connect(self._s)
        self.bs2 = QPushButton('Stop'); self.bs2.setObjectName('danger'); self.bs2.setEnabled(False); self.bs2.clicked.connect(self._st)
        br.addWidget(self.bs1,1); br.addWidget(self.bs2,1)
        g3l.addLayout(br)
        self.pg = QProgressBar(); self.pg.setTextVisible(False); g3l.addWidget(self.pg)
        mr = QHBoxLayout(); mr.setSpacing(8)
        for lbl, attr, col in [('Epoch','_me',TEXT), ('mAP@0.5','_mm',GREEN), ('Best','_mb',PRI)]:
            cw = QWidget(); cw.setStyleSheet(f'background:{BG};border-radius:6px;padding:6px;')
            cl2 = QVBoxLayout(cw); cl2.setContentsMargins(8,6,8,6); cl2.setSpacing(2)
            v = QLabel('0'); v.setStyleSheet(f'font-size:18px;font-weight:600;color:{col};qproperty-alignment:AlignCenter;')
            setattr(self, attr, v); cl2.addWidget(v)
            cl2.addWidget(QLabel(lbl, styleSheet=f'font-size:9px;color:{TEXT3};font-weight:500;qproperty-alignment:AlignCenter;'))
            mr.addWidget(cw,1)
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
        
        # 左：日志面板
        lw = QWidget(); lw.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        lo2 = QVBoxLayout(lw); lo2.setContentsMargins(6,4,6,6); lo2.setSpacing(4)
        h = QWidget(); h.setStyleSheet('background:transparent;border:none;')
        hl = QHBoxLayout(h); hl.setContentsMargins(2,0,2,0); hl.setSpacing(6)
        hl.addWidget(QLabel('● Console', styleSheet=f'font-size:10px;font-weight:600;color:{TEXT3};')); hl.addStretch()
        self.log_count_label = QLabel('0 lines'); self.log_count_label.setStyleSheet(f'font-size:9px;color:{TEXT3};')
        hl.addWidget(self.log_count_label)
        clear_btn = QPushButton('Clear'); clear_btn.setObjectName('danger')
        clear_btn.setStyleSheet('padding:2px 8px;min-height:18px;font-size:10px;')
        clear_btn.clicked.connect(self._clear_log)
        hl.addWidget(clear_btn); lo2.addWidget(h)
        self.lo = QTextEdit(); self.lo.setReadOnly(True)
        self.lo.setStyleSheet(f'QTextEdit{{background:{CON};color:{CON_T};border:none;border-radius:5px;padding:8px 10px;font-family:"Consolas","Courier New",monospace;font-size:13px;line-height:1.4;}}')
        self._log_lines = []; self._max_log_lines = 500
        lo2.addWidget(self.lo)
        bottom_layout.addWidget(lw, 1)
        
        # 右：系统监控面板
        mw = QWidget(); mw.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        mw.setFixedWidth(160)
        mvl = QVBoxLayout(mw); mvl.setContentsMargins(10,12,10,12); mvl.setSpacing(12)
        
        # 监控标题
        title = QLabel('● System Monitor')
        title.setStyleSheet(f'font-size:10px;font-weight:600;color:{TEXT3};')
        mvl.addWidget(title)
        
        # 监控指标
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
            item_layout.setSpacing(6)
            
            # 名称标签
            lbl = QLabel(label)
            lbl.setStyleSheet(f'font-size:10px;font-weight:500;color:{TEXT2};')
            lbl.setFixedWidth(50)
            item_layout.addWidget(lbl)
            
            # 进度条
            bar = QProgressBar()
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setStyleSheet(f'''
                QProgressBar {{
                    border: none;
                    border-radius: 4px;
                    height: 8px;
                    background: {BORDER};
                    text-align: center;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {PRI}, stop:1 #8b5cf6);
                    border-radius: 4px;
                }}
            ''')
            item_layout.addWidget(bar, 1)
            
            # 百分比数值
            val = QLabel('0%')
            val.setStyleSheet(f'font-size:10px;font-weight:600;color:{TEXT};')
            val.setFixedWidth(35)
            val.setAlignment(Qt.AlignCenter)
            item_layout.addWidget(val)
            
            mvl.addWidget(item)
            
            # 注册到studio的监控数据
            self.studio._sys_data[key] = [bar, val]
        
        bottom_layout.addWidget(mw)
        
        rl.addWidget(bottom, 2)
        lo.addWidget(right,1)

    def _config(self):
        s = lambda w: w.currentText()
        lr0_val = self.lr0.value()
        if lr0_val <= 0: lr0_val = 0.001
        return dict(model=s(self.m), epochs=self.ep.value(), batch=self.bs.value(), imgsz=int(s(self.sz)),
            lr0=lr0_val, lrf=self.lrf.value(), optimizer=s(self.opt), patience=self.pt.value(),
            device='0' if self.studio.gpu_ok and self.dev.currentIndex()==0 else 'cpu',
            cos_lr=self.sch.currentIndex()==0, warmup_epochs=self.wu.value(), workers=self.wk.value(),
            fl_gamma=self.fl.value(), label_smoothing=self.sm.value(), iou=self.iou_thresh.value(),
            close_mosaic=self.cm.value(), copy_paste=self.cp.value(), degrees=self.dg.value(),
            multi_scale=self.ms.isChecked())

    def _s(self):
        if self.trainer and self.trainer.isRunning():
            QMessageBox.warning(self, 'Warning', 'Training is in progress, please stop the current training first!'); return
        self.trainer = None
        self.bs1.setEnabled(False); self.bs2.setEnabled(True); self._enable_params(False)
        self.lo.clear(); self.pg.setValue(0); self._me.setText('0'); self._mm.setText('—'); self._mb.setText('—')
        self.lc.upd({}); self.mc.upd({}); self._log_lines = []; self.log_count_label.setText('0 lines')
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
        is_prog = bool(__import__('re').search(r'\d+%\s+\d+/\d+', msg))
        if is_prog and self._log_lines:
            last = self._log_lines[-1]
            text = __import__('re').sub(r'<[^>]+>', '', last)
            if __import__('re').search(r'\d+%\s+\d+/\d+', text):
                self._log_lines[-1] = self._fmt(ts, msg)
                self.lo.setHtml('\n'.join(self._log_lines))
                self.lo.verticalScrollBar().setValue(self.lo.verticalScrollBar().maximum())
                return
        self._log_lines.append(self._fmt(ts, msg))
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines:]
            self.lo.setHtml('\n'.join(self._log_lines))
        else:
            self.lo.append(self._log_lines[-1])
        self.lo.verticalScrollBar().setValue(self.lo.verticalScrollBar().maximum())
        self.log_count_label.setText(f'{len(self._log_lines)} lines')

    def _fmt(self, ts, msg):
        color = CON_T
        if '✅' in msg or 'Done' in msg: color = GREEN
        elif '❌' in msg or 'Failed' in msg: color = RED
        elif '⚠️' in msg: color = AMBER
        elif '🚀' in msg or 'Epoch' in msg: color = '#a5b4fc'
        elif '📁' in msg: color = TEXT3
        elif __import__('re').search(r'\d+%\s+\d+/\d+', msg): color = '#9ca3af'
        return f'<span style="color:#6b7280">[{ts}]</span> <span style="color:{color}">{msg}</span>'

    def _clear_log(self):
        self.lo.clear(); self._log_lines = []; self.log_count_label.setText('0 lines')
