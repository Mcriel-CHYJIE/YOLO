"""蒸馏标签页"""
from scripts.tabs.base import *
from PyQt5 import uic
from workers.distiller import Distiller


class DistillTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.dworker = None
        self._build()
        self._init_widgets()
        self._connect_signals()

    def _build(self):
        ui_path = Path(__file__).resolve().parent.parent / 'ui' / 'distill.ui'
        uic.loadUi(str(ui_path), self)
        # 按钮样式直接设置（规避全局 STYLE 传播时机问题）
        self.ds1.setStyleSheet(
            "QPushButton{background:#07C160;color:#fff;border:none;padding:5px 18px;min-height:26px;font-size:12px;font-weight:600;border-radius:4px;}QPushButton:hover{background:#06ad56;}QPushButton:disabled{background:#a5d6a5;}"
        )
        self.ds2.setStyleSheet(
            "QPushButton{background:#ef4444;color:#fff;border:none;padding:5px 18px;min-height:26px;border-radius:4px;}QPushButton:hover{background:#dc2626;}QPushButton:disabled{background:#fca5a5;}"
        )
        self.ds2.setEnabled(False)
        # 设置 columnStretch (PyQt5 不支持在 UI 文件中直接设置)
        if hasattr(self, 'paramsGridD'):
            self.paramsGridD.setColumnStretch(0, 0)
            self.paramsGridD.setColumnStretch(1, 1)
            self.paramsGridD.setColumnStretch(2, 1)

    def _init_widgets(self):
        self.de_sz.addItems(['416', '512', '640', '800'])
        self._replace_metric_cards()
        self._setup_chart_styles()
        self._load_defaults()

    def _replace_metric_cards(self):
        for placeholder, label, color, attr_val, attr_card in [
            (self.epochMetricD, 'Epoch', TEXT, '_de_ep_l', '_dep_card'),
            (self.mapMetricD, 'mAP', GREEN, '_de_m_l', '_dem_card'),
            (self.bestMetricD, 'Best', PRI, '_de_b_l', '_deb_card'),
        ]:
            card = MetricCard(label, color, '—')
            idx = self.metricRowD.indexOf(placeholder)
            self.metricRowD.removeWidget(placeholder)
            placeholder.deleteLater()
            self.metricRowD.insertWidget(idx, card, 1)
            setattr(self, attr_val, card.value_label)
            setattr(self, attr_card, card)

    def _setup_chart_styles(self):
        for w in [self.cw_loss, self.cw_map]:
            w.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        for ch, xlabel, ylabel in [
            (self.dc_loss, 'Epoch', 'Loss'),
            (self.dc_map, 'Epoch', 'mAP50'),
        ]:
            ch.ax.set_xlabel(xlabel, fontsize=8)
            ch.ax.set_ylabel(ylabel, fontsize=8)

    def _connect_signals(self):
        self.teacherBrowse.clicked.connect(self._browse_teacher)
        self.studentBrowse.clicked.connect(self._browse_student)
        self.dataBrowse.clicked.connect(self._browse_data)
        self.ds1.setObjectName('pri')
        self.ds1.clicked.connect(self._ds)
        self.ds2.setObjectName('danger')
        self.ds2.clicked.connect(self._dst)

    def _load_defaults(self):
        d = cfg.get('distill', {})
        teacher_path = (d.get('teacher') if d else '') or str((ROOT / 'runs/detect/runs').as_posix())
        self.de_teacher.setText(teacher_path)
        student_def = (d.get('student', 'models/yolo11n.pt') if d else 'models/yolo11n.pt')
        sp = str((ROOT / student_def).as_posix()) if not Path(student_def).is_absolute() else Path(student_def).as_posix()
        self.de_student.setText(sp)
        self.de_data.setText(Path(DATA_YAML).as_posix())
        if d:
            self.de_ep.setValue(d.get('epochs', 150))
            self.de_bs.setValue(d.get('batch', 24))
            self.de_lr.setValue(d.get('lr0', 0.002))
            self.de_al.setValue(d.get('alpha', 0.5))
            self.de_pt.setValue(d.get('patience', 40))
            self.de_sz.setCurrentText(str(d.get('imgsz', 640)))
            self.de_warmup.setValue(d.get('warmup_epochs', 5))
            self.de_wd.setValue(d.get('weight_decay', 0.0005))
            self.de_mom.setValue(d.get('momentum', 0.937))

    def _browse_teacher(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Select Teacher Model', str(ROOT / 'runs/detect/runs'), MODEL_FILTER)
        if p: self.de_teacher.setText(p)

    def _browse_student(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Select Student Model', str(ROOT / 'runs/detect/runs'), MODEL_FILTER)
        if p: self.de_student.setText(p)

    def _browse_data(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Select Data YAML', '', 'YAML Files (*.yaml *.yml)')
        if p: self.de_data.setText(p)

    def _ds(self):
        if self.dworker and self.dworker.isRunning():
            QMessageBox.warning(self, 'Warning', 'Distillation is already in progress'); return
        teacher_path = self.de_teacher.text().strip()
        if not teacher_path or not Path(teacher_path).exists():
            QMessageBox.critical(self, 'Invalid Teacher', f'Teacher model not found:\n{teacher_path}'); return
        student_path = self.de_student.text().strip()
        if not student_path or not Path(student_path).exists():
            QMessageBox.critical(self, 'Invalid Student', f'Student model not found:\n{student_path}'); return
        data_path = self.de_data.text().strip()
        if not data_path or not Path(data_path).exists():
            QMessageBox.critical(self, 'Invalid Data', f'Data YAML not found:\n{data_path}'); return
        lr0 = self.de_lr.value()
        if lr0 <= 0:
            QMessageBox.critical(self, 'Invalid LR', f'Learning rate must be > 0, got: {lr0}'); return
        epochs = self.de_ep.value()
        if epochs < 1:
            QMessageBox.critical(self, 'Invalid Epochs', f'Epochs must be >= 1, got: {epochs}'); return
        self.dworker = None
        cfg = dict(teacher=teacher_path, student=student_path, data=data_path,
            name=f'distill_{datetime.now().strftime("%m%d_%H%M")}', epochs=epochs, batch=self.de_bs.value(),
            lr0=lr0, alpha=self.de_al.value(), patience=self.de_pt.value(),
            imgsz=int(self.de_sz.currentText()), device='cuda:0' if self.studio.gpu_ok else 'cpu',
            workers=self.studio.cpu_count,
            warmup_epochs=self.de_warmup.value(),
            weight_decay=self.de_wd.value(),
            momentum=self.de_mom.value())
        self.dworker = Distiller(cfg)
        self.dworker.log.connect(lambda m: self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), m)))
        self.dworker.progress.connect(self._dp)
        self.dworker.done.connect(self._dd)
        self.ds1.setEnabled(False); self.ds2.setEnabled(True)
        self.log_panel.clear()
        self.dc_loss.clr(); self.dc_loss.rf()
        self.dc_map.clr(); self.dc_map.rf()
        self.dpg.setValue(0); self.dworker.start()

    def _dst(self):
        if self.dworker and self.dworker.isRunning():
            self.dworker.stop(); self.ds2.setEnabled(False); self.ds2.setText('Stopping…')

    def _dp(self, ep, data):
        h = data.get('history', {})
        if h.get('epoch'):
            self.dc_loss.clr()
            self.dc_loss.ax.plot(h['epoch'], h['loss'], 'r-', lw=1.5, label='Loss')
            self.dc_loss.ax.legend(loc='upper right', fontsize=7)
            self.dc_loss.ax.grid(True, alpha=0.4, color=BORDER, linewidth=0.5)
            self.dc_loss.rf()
            self.dc_map.clr()
            if any(v > 0 for v in h.get('map50', [])):
                self.dc_map.ax.plot(h['epoch'], h['map50'], 'g-', lw=1.5, label='mAP50')
                self.dc_map.ax.legend(loc='lower right', fontsize=7)
            self.dc_map.ax.grid(True, alpha=0.4, color=BORDER, linewidth=0.5)
            self.dc_map.rf()
        self._de_ep_l.setText(str(data['epoch']))
        self._de_m_l.setText(f'{data["map50"]:.4f}' if data['map50'] > 0 else '—')
        self._de_b_l.setText(f'{data["best_map50"]:.4f}' if data['best_map50'] > 0 else '—')
        self.dpg.setValue(int(data['epoch'] / data['total'] * 100))

    def _dd(self, ok, msg):
        self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), f'{"" if ok else ""} {msg}'))
        self.ds1.setEnabled(True); self.ds2.setEnabled(True); self.ds2.setText('Stop')
        self.dworker = None
        if ok: self.dpg.setValue(100)
