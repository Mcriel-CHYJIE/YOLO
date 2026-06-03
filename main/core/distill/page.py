# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""蒸馏标签页"""
from main.core.base import *
from PyQt5 import uic
from .service import Distiller
from .service import build_distill_config, load_distill_defaults


class DistillTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.dworker = None
        self._build_ui()
        self._init_widgets()
        self._connect()

    def _build_ui(self):
        ui_path = Path(__file__).resolve().parent / 'distill.ui'
        uic.loadUi(str(ui_path), self)
        # ── 标题 ──
        self.titleLabel.setStyleSheet(f'font-size:18px;font-weight:700;color:{TEXT};padding:0;margin:0;')
        self.titleLabel.setFixedHeight(24)
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
        from PyQt5.QtWidgets import QListView
        _lv = QListView()
        _lv.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.de_sz.setView(_lv)
        self.de_sz.setMaxVisibleItems(10)
        fm = self.de_sz.fontMetrics()
        self.de_sz.view().setMaximumHeight((fm.height() + 4) * 10 + 4)
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

    # ═══════════════════════════════════════════
    # Signal Wiring
    # ═══════════════════════════════════════════
    def _connect(self):
        self.teacherBrowse.clicked.connect(self._browse_teacher)
        self.studentBrowse.clicked.connect(self._browse_student)
        self.dataBrowse.clicked.connect(self._browse_data)
        self.ds1.setObjectName('pri')
        self.ds1.clicked.connect(self._ds)
        self.ds2.setObjectName('danger')
        self.ds2.clicked.connect(self._dst)

    def _load_defaults(self):
        d = load_distill_defaults()
        self.de_teacher.setText(d['teacher'])
        self.de_student.setText(d['student'])
        self.de_data.setText(d['data'])
        self.de_ep.setValue(d['epochs'])
        self.de_bs.setValue(d['batch'])
        self.de_lr.setValue(d['lr0'])
        self.de_al.setValue(d['alpha'])
        self.de_pt.setValue(d['patience'])
        self.de_sz.setCurrentText(d['imgsz'])
        self.de_warmup.setValue(d['warmup_epochs'])
        self.de_wd.setValue(d['weight_decay'])
        self.de_mom.setValue(d['momentum'])

    def _browse_teacher(self):
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        p, _ = QFileDialog.getOpenFileName(self, 'Select Teacher Model', str(ROOT / 'runs/detect/runs'), MODEL_FILTER, options=opts)
        if p: self.de_teacher.setText(p)

    def _browse_student(self):
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        p, _ = QFileDialog.getOpenFileName(self, 'Select Student Model', str(ROOT / 'runs/detect/runs'), MODEL_FILTER, options=opts)
        if p: self.de_student.setText(p)

    def _browse_data(self):
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        p, _ = QFileDialog.getOpenFileName(self, 'Select Data YAML', '', 'YAML Files (*.yaml *.yml)', options=opts)
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
        cfg = build_distill_config(
            teacher_path, student_path, data_path,
            epochs, self.de_bs.value(), lr0, self.de_al.value(),
            self.de_pt.value(), int(self.de_sz.currentText()), self.studio.gpu_ok,
            self.de_warmup.value(), self.de_wd.value(), self.de_mom.value())
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
