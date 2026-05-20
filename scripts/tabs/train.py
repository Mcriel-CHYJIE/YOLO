"""训练标签页"""
from scripts.tabs.base import *
from PyQt5 import uic
from workers.trainer import Trainer

PROG_RE = re.compile(r'\d+%\s+\d+/\d+')
HTML_RE = re.compile(r'<[^>]+>')


class TrainTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.trainer = None
        self._build()
        self._init_widgets()
        self._connect_signals()

    def _build(self):
        ui_path = Path(__file__).resolve().parent.parent / 'ui' / 'train.ui'
        uic.loadUi(str(ui_path), self)
        # 设置 columnStretch (PyQt5 不支持在 UI 文件中直接设置)
        if hasattr(self, 'configGrid'):
            self.configGrid.setColumnStretch(0, 0)
            self.configGrid.setColumnStretch(1, 1)
            self.configGrid.setColumnStretch(2, 1)
        if hasattr(self, 'algoGrid'):
            self.algoGrid.setColumnStretch(0, 0)
            self.algoGrid.setColumnStretch(1, 1)
            self.algoGrid.setColumnStretch(2, 1)

    def _init_widgets(self):
        t = cfg['training']
        self.configGrid.setColumnStretch(0, 0)
        self.configGrid.setColumnStretch(1, 1)
        self.configGrid.setColumnStretch(2, 0)
        self.configGrid.setColumnStretch(3, 1)
        self.m.addItems(t['model_options'])
        self.m.setCurrentText(t['model'])
        self.sz.addItems([str(v) for v in t['imgsz_options']])
        self.sz.setCurrentText(str(t['imgsz']))
        self.opt.addItems(t['optimizer_options'])
        self.opt.setCurrentText(t['optimizer'])
        self.dev.addItems(['GPU', 'CPU'])
        self.sch.addItems(t['scheduler_options'])
        self.sch.setCurrentText(t['scheduler'].capitalize())
        self.tipLabel.setText(cfg['project']['tip'])
        if not self.studio.gpu_ok:
            self.dev.setCurrentIndex(1)
        else:
            self.dev.setCurrentText('GPU' if t['device'] in ('auto', '0', 'GPU') else 'CPU')

        # 替换 MetricCard 占位
        for placeholder, label, color, default, attr_val, attr_card in [
            (self.epochCard, 'Epoch', TEXT, '0', '_me', '_me_card'),
            (self.mapCard, 'mAP@0.5', GREEN, '—', '_mm', '_mm_card'),
            (self.bestCard, 'Best', PRI, '—', '_mb', '_mb_card'),
        ]:
            card = MetricCard(label, color, default)
            idx = self.metricRow.indexOf(placeholder)
            self.metricRow.removeWidget(placeholder)
            placeholder.deleteLater()
            self.metricRow.insertWidget(idx, card, 1)
            setattr(self, attr_val, card.value_label)
            setattr(self, attr_card, card)

        # 图表容器样式
        for w in [self.lcContainer, self.mcContainer]:
            w.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')

        # 系统监控面板
        mvl = QVBoxLayout(self.sysMonitorPanel)
        mvl.setContentsMargins(8, 8, 8, 8); mvl.setSpacing(4)
        mvl.addWidget(QLabel('● System', styleSheet=f'font-size:10px;font-weight:600;color:{TEXT3};'))
        monitors = [('CPU', 'CPU'), ('MEM', 'Memory'), ('DSK', 'Disk')]
        if self.studio.gpu_ok:
            monitors += [('VRM', 'VRAM'), ('GPU', 'GPU')]
        for key, label in monitors:
            item = QWidget(); item.setStyleSheet('background:transparent;border:none;')
            il = QHBoxLayout(item); il.setContentsMargins(0, 0, 0, 0); il.setSpacing(0)
            lbl = QLabel(label)
            lbl.setStyleSheet(f'font-size:10px;font-weight:500;color:{TEXT2};')
            lbl.setFixedWidth(35)
            il.addWidget(lbl)
            val = QLabel('0%')
            val.setStyleSheet(f'font-size:10px;font-weight:600;color:{TEXT};')
            val.setAlignment(Qt.AlignRight)
            il.addWidget(val, 1)
            mvl.addWidget(item)
            self.studio._sys_data[key] = [val]

        self._params = [self.m, self.ep, self.bs, self.sz, self.opt, self.dev,
            self.sch, self.pt, self.lr0, self.lrf, self.wu, self.wk,
            self.iou_thresh, self.cm, self.cp, self.dg, self.ms]

    def _connect_signals(self):
        self.bs1.setObjectName('pri')
        self.bs1.clicked.connect(self._s)
        self.bs2.setObjectName('danger')
        self.bs2.setEnabled(False)
        self.bs2.clicked.connect(self._st)

    def _config(self):
        s = lambda w: w.currentText()
        lr0_val = self.lr0.value()
        if lr0_val <= 0: lr0_val = 0.001
        g = cfg['training']
        return dict(model=s(self.m), epochs=self.ep.value(), batch=self.bs.value(), imgsz=int(s(self.sz)),
            lr0=lr0_val, lrf=self.lrf.value(), optimizer=s(self.opt), patience=self.pt.value(),
            device='0' if self.studio.gpu_ok and self.dev.currentIndex() == 0 else 'cpu',
            cos_lr=self.sch.currentIndex() == 0, warmup_epochs=self.wu.value(), workers=self.wk.value(),
            iou=self.iou_thresh.value(),
            close_mosaic=self.cm.value(), copy_paste=self.cp.value(), degrees=self.dg.value(),
            multi_scale=self.ms.isChecked(),
            hsv_h=g.get('hsv_h', 0.015), hsv_s=g.get('hsv_s', 0.7), hsv_v=g.get('hsv_v', 0.4),
            translate=g.get('translate', 0.15), scale=g.get('scale', 0.6))

    def _s(self):
        if self.trainer and self.trainer.isRunning():
            QMessageBox.warning(self, 'Warning',
                'Training is in progress, please stop the current training first!'); return
        self.trainer = None
        self.bs1.setEnabled(False); self.bs2.setEnabled(True); self._enable_params(False)
        self.log_panel.clear(); self.pg.setValue(0)
        self._me.setText('0'); self._mm.setText('—'); self._mb.setText('—')
        self._lc.upd({}); self._mc.upd({})
        cfg = self._config()
        self._log(f'🚀 {cfg["model"]} | {cfg["epochs"]}ep | batch={cfg["batch"]}')
        self.trainer = Trainer(cfg)
        self.trainer.log.connect(self._log)
        self.trainer.status.connect(lambda t, p, b, c: (
            self._me.setText(t.split('/')[0].replace('Epoch ', '')),
            self._mm.setText(f'{c:.4f}' if c > 0 else '—'),
            self._mb.setText(f'{b:.4f}' if b > 0 else '—'),
            self.pg.setValue(int(p * 100))))
        self.trainer.chart.connect(
            lambda: (self.trainer and (self._lc.upd(self.trainer.history), self._mc.upd(self.trainer.history))))
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
