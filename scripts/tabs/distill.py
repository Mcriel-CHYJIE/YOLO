"""蒸馏标签页"""
from scripts.tabs.base import *

class DistillTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.dworker = None
        self._build()

    def _build(self):
        lo = QHBoxLayout(self); lo.setContentsMargins(4,4,4,4); lo.setSpacing(6)
        left = QWidget(); left.setFixedWidth(340)
        ll = QVBoxLayout(left); ll.setSpacing(5); ll.setContentsMargins(0,0,0,0)

        gm = QGroupBox('Models')
        gl = QVBoxLayout(gm); gl.setSpacing(3); gl.setContentsMargins(6,12,6,6)
        d = cfg.get('distill', {})
        teacher_def = d.get('teacher', '') if d else ''
        teacher_path = teacher_def if teacher_def else str((ROOT / 'runs/detect/runs').as_posix())
        self.de_teacher = QLineEdit(teacher_path)
        cw_t, cl_t = self._labeled_card('Teacher', self.de_teacher)
        btn_teacher = QPushButton('Browse'); btn_teacher.clicked.connect(self._browse_teacher); cl_t.addWidget(btn_teacher)
        gl.addWidget(cw_t)
        
        student_def = d.get('student', 'yolo11n.pt') if d else 'yolo11n.pt'
        student_path = str((ROOT / student_def).as_posix()) if not Path(student_def).is_absolute() else Path(student_def).as_posix()
        self.de_student = QLineEdit(student_path)
        cw_s, cl_s = self._labeled_card('Student', self.de_student)
        btn_student = QPushButton('Browse'); btn_student.clicked.connect(self._browse_student); cl_s.addWidget(btn_student)
        gl.addWidget(cw_s)
        
        self.de_data = QLineEdit(Path(DATA_YAML).as_posix())
        cw_d, cl_d = self._labeled_card('Data', self.de_data)
        btn_data = QPushButton('Browse'); btn_data.clicked.connect(self._browse_data); cl_d.addWidget(btn_data)
        gl.addWidget(cw_d)
        ll.addWidget(gm)

        gp = QGroupBox('Parameters')
        gpl = QGridLayout(gp); gpl.setSpacing(3); gpl.setContentsMargins(6,12,6,6)
        gpl.setColumnStretch(0,1); gpl.setColumnStretch(1,1)
        d = cfg.get('distill', {})
        
        # Row 0: Epochs, Batch
        self.de_ep = QSpinBox(); self.de_ep.setRange(1,500); self.de_ep.setValue(d.get('epochs', 150) if d else 150)
        labeled_field(self.de_ep,'Epochs',gpl,0,0)
        self.de_bs = QSpinBox(); self.de_bs.setRange(1,128); self.de_bs.setValue(d.get('batch', 24) if d else 24)
        labeled_field(self.de_bs,'Batch',gpl,0,1)
        
        # Row 1: LR, Alpha
        self.de_lr = QDoubleSpinBox(); self.de_lr.setRange(1e-6,1); self.de_lr.setDecimals(6); self.de_lr.setValue(d.get('lr0', 0.002) if d else 0.002)
        labeled_field(self.de_lr,'LR',gpl,1,0)
        self.de_al = QDoubleSpinBox(); self.de_al.setRange(0,1); self.de_al.setDecimals(2); self.de_al.setSingleStep(0.05); self.de_al.setValue(d.get('alpha', 0.5) if d else 0.5)
        labeled_field(self.de_al,'Alpha',gpl,1,1)
        
        # Row 2: Patience, ImgSz
        self.de_pt = QSpinBox(); self.de_pt.setRange(5,200); self.de_pt.setValue(d.get('patience', 40) if d else 40)
        labeled_field(self.de_pt,'Patience',gpl,2,0)
        self.de_sz = QComboBox(); self.de_sz.addItems(['416','512','640','800'])
        self.de_sz.setCurrentText(str(d.get('imgsz', 640) if d else 640))
        labeled_field(self.de_sz,'ImgSz',gpl,2,1)
        
        # Row 3: Warmup Epochs, Weight Decay
        self.de_warmup = QSpinBox(); self.de_warmup.setRange(0,50); self.de_warmup.setValue(d.get('warmup_epochs', 5) if d else 5)
        labeled_field(self.de_warmup,'Warmup',gpl,3,0)
        self.de_wd = QDoubleSpinBox(); self.de_wd.setRange(0,0.1); self.de_wd.setDecimals(6); self.de_wd.setSingleStep(0.0001); self.de_wd.setValue(d.get('weight_decay', 0.0005) if d else 0.0005)
        labeled_field(self.de_wd,'WeightDecay',gpl,3,1)
        
        # Row 4: Momentum
        self.de_mom = QDoubleSpinBox(); self.de_mom.setRange(0,1); self.de_mom.setDecimals(3); self.de_mom.setSingleStep(0.001); self.de_mom.setValue(d.get('momentum', 0.937) if d else 0.937)
        labeled_field(self.de_mom,'Momentum',gpl,4,0)
        
        ll.addWidget(gp)

        gc = QGroupBox('Control')
        gcl = QVBoxLayout(gc); gcl.setSpacing(4); gcl.setContentsMargins(8,12,8,8)
        br = QHBoxLayout(); br.setSpacing(5)
        self.ds1 = QPushButton('Start'); self.ds1.setObjectName('pri'); self.ds1.clicked.connect(self._ds)
        self.ds2 = QPushButton('Stop'); self.ds2.setObjectName('danger'); self.ds2.setEnabled(False); self.ds2.clicked.connect(self._dst)
        br.addWidget(self.ds1,1); br.addWidget(self.ds2,1)
        gcl.addLayout(br)
        self.dpg = QProgressBar(); self.dpg.setTextVisible(False); gcl.addWidget(self.dpg)

        mr = QHBoxLayout()
        self._dep_card = MetricCard('Epoch', TEXT)
        self._de_ep_l = self._dep_card.value_label; mr.addWidget(self._dep_card, 1)
        self._dem_card = MetricCard('mAP', GREEN, '—')
        self._de_m_l = self._dem_card.value_label; mr.addWidget(self._dem_card, 1)
        self._deb_card = MetricCard('Best', PRI, '—')
        self._de_b_l = self._deb_card.value_label; mr.addWidget(self._deb_card, 1)
        gcl.addLayout(mr); ll.addWidget(gc); lo.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right); rl.setSpacing(5); rl.setContentsMargins(0,0,0,0)
        
        # Chart row: Loss | mAP50 (stretch factor 3)
        chart_row = QWidget()
        chart_lo = QHBoxLayout(chart_row); chart_lo.setContentsMargins(0,0,0,0); chart_lo.setSpacing(5)
        
        # Loss Chart
        self.dc_loss = Chart(self)
        cw_loss = QWidget(); cw_loss.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        cl_loss = QVBoxLayout(cw_loss); cl_loss.setContentsMargins(8,4,8,2); cl_loss.setSpacing(1)
        hh_loss = QHBoxLayout()
        hh_loss.addWidget(QLabel('●', styleSheet=f'color:{RED};font-size:7px;'))
        hh_loss.addWidget(QLabel('Loss', styleSheet='font-weight:600;font-size:10px;')); hh_loss.addStretch()
        cl_loss.addLayout(hh_loss)
        self.dc_loss.ax.set_xlabel('Epoch', fontsize=8)
        self.dc_loss.ax.set_ylabel('Loss', fontsize=8)
        cl_loss.addWidget(self.dc_loss)
        chart_lo.addWidget(cw_loss, 1)
        
        # mAP50 Chart
        self.dc_map = Chart(self)
        cw_map = QWidget(); cw_map.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        cl_map = QVBoxLayout(cw_map); cl_map.setContentsMargins(8,4,8,2); cl_map.setSpacing(1)
        hh_map = QHBoxLayout()
        hh_map.addWidget(QLabel('●', styleSheet=f'color:{GREEN};font-size:7px;'))
        hh_map.addWidget(QLabel('mAP50', styleSheet='font-weight:600;font-size:10px;')); hh_map.addStretch()
        cl_map.addLayout(hh_map)
        self.dc_map.ax.set_xlabel('Epoch', fontsize=8)
        self.dc_map.ax.set_ylabel('mAP50', fontsize=8)
        cl_map.addWidget(self.dc_map)
        chart_lo.addWidget(cw_map, 1)
        
        rl.addWidget(chart_row, 3)
        
        # Log panel (stretch factor 2)
        self.log_panel = LogPanel('● Console', max_lines=500)
        rl.addWidget(self.log_panel, 2)
        lo.addWidget(right, 1)

    def _labeled_card(self, label, widget):
        """Create a labeled card with horizontal layout"""
        cw = QWidget()
        cw.setStyleSheet(f'background:{BG};border-radius:4px;')
        cl = QHBoxLayout(cw); cl.setContentsMargins(6, 1, 6, 1); cl.setSpacing(4)
        lbl = QLabel(label)
        lbl.setStyleSheet(f'font-size:9px;color:{TEXT};background:transparent;font-weight:500;')
        lbl.setFixedHeight(22); widget.setMinimumHeight(22)
        cl.addWidget(lbl); cl.addWidget(widget, 1)
        return cw, cl

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
            QMessageBox.warning(self,'Warning','Distillation is already in progress'); return
        # ── 参数校验 ──
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
        # ── 启动 ──
        self.dworker = None
        cfg = dict(teacher=teacher_path, student=student_path, data=data_path,
            name=f'distill_{datetime.now().strftime("%m%d_%H%M")}', epochs=epochs, batch=self.de_bs.value(),
            lr0=lr0, alpha=self.de_al.value(), patience=self.de_pt.value(),
            imgsz=int(self.de_sz.currentText()), device='cuda:0' if self.studio.gpu_ok else 'cpu',
            workers=self.studio.cpu_count,
            warmup_epochs=self.de_warmup.value(),
            weight_decay=self.de_wd.value(),
            momentum=self.de_mom.value())
        self.dworker = DistillWorker(cfg)
        self.dworker.log.connect(lambda m: self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), m)))
        self.dworker.progress.connect(self._dp)
        self.dworker.done.connect(self._dd)
        self.ds1.setEnabled(False); self.ds2.setEnabled(True)
        self.log_panel.clear()
        # Clear both charts
        self.dc_loss.clr(); self.dc_loss.rf()
        self.dc_map.clr(); self.dc_map.rf()
        self.dpg.setValue(0); self.dworker.start()

    def _dst(self):
        if self.dworker and self.dworker.isRunning():
            self.dworker.stop(); self.ds2.setEnabled(False); self.ds2.setText('Stopping…')

    def _dp(self, ep, data):
        h = data.get('history',{})
        if h.get('epoch'):
            # Update Loss chart
            self.dc_loss.clr()
            self.dc_loss.ax.plot(h['epoch'],h['loss'],'r-',lw=1.5,label='Loss')
            self.dc_loss.ax.legend(loc='upper right',fontsize=7)
            self.dc_loss.ax.grid(True,alpha=0.4,color=BORDER,linewidth=0.5)
            self.dc_loss.rf()
            
            # Update mAP50 chart
            self.dc_map.clr()
            if any(v>0 for v in h.get('map50',[])):
                self.dc_map.ax.plot(h['epoch'],h['map50'],'g-',lw=1.5,label='mAP50')
                self.dc_map.ax.legend(loc='lower right',fontsize=7)
            self.dc_map.ax.grid(True,alpha=0.4,color=BORDER,linewidth=0.5)
            self.dc_map.rf()
        self._de_ep_l.setText(str(data['epoch']))
        self._de_m_l.setText(f'{data["map50"]:.4f}' if data['map50']>0 else '—')
        self._de_b_l.setText(f'{data["best_map50"]:.4f}' if data['best_map50']>0 else '—')
        self.dpg.setValue(int(data['epoch']/data['total']*100))

    def _dd(self, ok, msg):
        self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), f'{"✅" if ok else "❌"} {msg}'))
        self.ds1.setEnabled(True); self.ds2.setEnabled(True); self.ds2.setText('Stop')
        self.dworker = None
        if ok: self.dpg.setValue(100)
