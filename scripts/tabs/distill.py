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
        left = QWidget(); left.setFixedWidth(360)
        ll = QVBoxLayout(left); ll.setSpacing(5); ll.setContentsMargins(0,0,0,0)

        gm = QGroupBox('Models')
        gl = QGridLayout(gm); gl.setSpacing(3); gl.setContentsMargins(6,12,6,6)
        gl.setColumnStretch(0,1); gl.setColumnStretch(1,1)
        d = cfg['distill']
        teacher_path = d['teacher'] if d['teacher'] else str(ROOT / 'runs/detect/train/weights/best.pt')
        self.de_teacher = QLineEdit(teacher_path)
        labeled_field(self.de_teacher,'Teacher',gl,0,0)
        self.de_student = QLineEdit(str(ROOT / d['student']) if not Path(d['student']).is_absolute() else d['student'])
        labeled_field(self.de_student,'Student',gl,0,1)
        self.de_data = QLineEdit(DATA_YAML); labeled_field(self.de_data,'Data',gl,1,0)
        self.de_name = QLineEdit(f'distill_{datetime.now().strftime("%m%d_%H%M")}')
        labeled_field(self.de_name,'Name',gl,1,1)
        self.de_ep = QSpinBox(); self.de_ep.setRange(1,500); self.de_ep.setValue(d['epochs'])
        labeled_field(self.de_ep,'Epochs',gl,2,0)
        self.de_bs = QSpinBox(); self.de_bs.setRange(1,128); self.de_bs.setValue(d['batch'])
        labeled_field(self.de_bs,'Batch',gl,2,1)
        self.de_lr = QDoubleSpinBox(); self.de_lr.setRange(1e-6,1); self.de_lr.setDecimals(6); self.de_lr.setValue(d['lr0'])
        labeled_field(self.de_lr,'LR',gl,3,0)
        self.de_al = QDoubleSpinBox(); self.de_al.setRange(0,1); self.de_al.setDecimals(2); self.de_al.setSingleStep(0.05); self.de_al.setValue(d['alpha'])
        labeled_field(self.de_al,'Alpha',gl,3,1)
        self.de_pt = QSpinBox(); self.de_pt.setRange(5,200); self.de_pt.setValue(d['patience'])
        labeled_field(self.de_pt,'Patience',gl,4,0)
        self.de_sz = QComboBox(); self.de_sz.addItems(['640','800','416'])
        self.de_sz.setCurrentText(str(d['imgsz']))
        labeled_field(self.de_sz,'ImgSz',gl,4,1)
        ll.addWidget(gm)

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
        self.dc = Chart(self)
        cw = QWidget(); cw.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        cl0 = QVBoxLayout(cw); cl0.setContentsMargins(8,4,8,2); cl0.setSpacing(1)
        hh = QHBoxLayout()
        hh.addWidget(QLabel('●', styleSheet=f'color:{PRI};font-size:7px;'))
        hh.addWidget(QLabel('Distill', styleSheet='font-weight:600;font-size:10px;')); hh.addStretch(); cl0.addLayout(hh)
        self.dc.ax.set_xlabel('Epoch', fontsize=8)
        self.dc.loss_line, = self.dc.ax.plot([],[],'r-',label='Loss',lw=1.5)
        self.dc.map_line, = self.dc.ax.plot([],[],'g-',label='mAP50',lw=1.5)
        self.dc.ax.legend(loc='upper right',fontsize=8); self.dc.rf()
        cl0.addWidget(self.dc); rl.addWidget(cw)

        self.log_panel = LogPanel('● Console', max_lines=500)
        rl.addWidget(self.log_panel, 1); lo.addWidget(right, 1)

    def _ds(self):
        if self.dworker and self.dworker.isRunning():
            QMessageBox.warning(self,'Warning','Distillation is already in progress'); return
        self.dworker = None
        cfg = dict(teacher=self.de_teacher.text(), student=self.de_student.text(), data=self.de_data.text(),
            name=self.de_name.text(), epochs=self.de_ep.value(), batch=self.de_bs.value(),
            lr0=self.de_lr.value(), alpha=self.de_al.value(), patience=self.de_pt.value(),
            imgsz=int(self.de_sz.currentText()), device='cuda:0' if self.studio.gpu_ok else 'cpu',
            workers=self.studio.cpu_count)
        self.dworker = DistillWorker(cfg)
        self.dworker.log.connect(lambda m: self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), m)))
        self.dworker.progress.connect(self._dp)
        self.dworker.done.connect(self._dd)
        self.ds1.setEnabled(False); self.ds2.setEnabled(True)
        self.log_panel.clear(); self.dc.ax.clear()
        self.dc.loss_line, = self.dc.ax.plot([],[],'r-',label='Loss',lw=1.5)
        self.dc.map_line, = self.dc.ax.plot([],[],'g-',label='mAP50',lw=1.5)
        self.dc.ax.legend(loc='upper right',fontsize=8); self.dc.rf()
        self.dpg.setValue(0); self.dworker.start()

    def _dst(self):
        if self.dworker and self.dworker.isRunning():
            self.dworker.stop(); self.ds2.setEnabled(False); self.ds2.setText('Stopping…')

    def _dp(self, ep, data):
        h = data.get('history',{})
        if h.get('epoch'):
            self.dc.ax.clear()
            self.dc.ax.plot(h['epoch'],h['loss'],'r-',lw=1.5,label='Loss')
            if any(v>0 for v in h.get('map50',[])):
                self.dc.ax.plot(h['epoch'],h['map50'],'g-',lw=1.5,label='mAP50')
            self.dc.ax.legend(loc='upper right',fontsize=8)
            self.dc.ax.set_xlabel('Epoch',fontsize=8)
            self.dc.ax.grid(True,alpha=0.4,color=BORDER,linewidth=0.5); self.dc.rf()
        self._de_ep_l.setText(str(data['epoch']))
        self._de_m_l.setText(f'{data["map50"]:.4f}' if data['map50']>0 else '—')
        self._de_b_l.setText(f'{data["best_map50"]:.4f}' if data['best_map50']>0 else '—')
        self.dpg.setValue(int(data['epoch']/data['total']*100))

    def _dd(self, ok, msg):
        self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), f'{"✅" if ok else "❌"} {msg}'))
        self.ds1.setEnabled(True); self.ds2.setEnabled(True); self.ds2.setText('Stop')
        self.dworker = None
        if ok: self.dpg.setValue(100)
