"""导出标签页"""
from scripts.tabs.base import *

class ExportTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._build()
        self._load_latest()

    def _build(self):
        lo = QHBoxLayout(self); lo.setContentsMargins(4,4,4,4); lo.setSpacing(6)
        left = QWidget(); left.setFixedWidth(360)
        ll = QVBoxLayout(left); ll.setSpacing(5); ll.setContentsMargins(0,0,0,0)

        g1 = QGroupBox('Model')
        g1l = QVBoxLayout(g1); g1l.setSpacing(4); g1l.setContentsMargins(8,12,8,8)
        g1l.addWidget(QLabel('Weights', styleSheet=f'font-size:9px;color:{TEXT2};font-weight:500;'))
        wr = QHBoxLayout()
        self.ex_w = QLineEdit(''); wr.addWidget(self.ex_w,1)
        bw = QPushButton('Browse'); bw.clicked.connect(lambda: self._ex_browse()); wr.addWidget(bw)
        g1l.addLayout(wr); ll.addWidget(g1)

        g2 = QGroupBox('Format')
        g2l = QVBoxLayout(g2); g2l.setSpacing(3); g2l.setContentsMargins(8,12,8,8)
        e = cfg['export']
        self.ex_fmt = QComboBox(); self.ex_fmt.addItems(e['format_options'])
        self.ex_fmt.setCurrentText(e['format'])
        g2l.addWidget(self.ex_fmt)
        g2l.addWidget(QLabel('Image Size:', styleSheet=f'font-size:9px;color:{TEXT2};'))
        self.ex_sz = QComboBox(); self.ex_sz.addItems([str(v) for v in e['imgsz_options']])
        self.ex_sz.setCurrentText(str(e['imgsz']))
        g2l.addWidget(self.ex_sz)
        self.ex_half = QCheckBox('FP16 Half Precision'); self.ex_half.setChecked(e['half']); g2l.addWidget(self.ex_half)
        self.ex_int8 = QCheckBox('INT8 Quantization'); self.ex_int8.setChecked(e['int8']); g2l.addWidget(self.ex_int8)
        self.ex_nms = QCheckBox('Integrated NMS'); self.ex_nms.setChecked(e['nms']); g2l.addWidget(self.ex_nms)
        ll.addWidget(g2)

        g3 = QGroupBox('Control')
        g3l = QVBoxLayout(g3); g3l.setSpacing(4); g3l.setContentsMargins(8,12,8,8)
        self.ex_btn = QPushButton('📦 Export'); self.ex_btn.setObjectName('pri')
        self.ex_btn.clicked.connect(self._ex_run); g3l.addWidget(self.ex_btn)
        ll.addWidget(g3); ll.addStretch(); lo.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right); rl.setSpacing(5); rl.setContentsMargins(0,0,0,0)
        self.log_panel = LogPanel('● Output', max_lines=500)
        rl.addWidget(self.log_panel, 1); lo.addWidget(right, 1)

    def _load_latest(self):
        w = find_latest_best()
        if w: self.ex_w.setText(w)

    def _ex_browse(self):
        p,_ = QFileDialog.getOpenFileName(self,'Select Model','runs',MODEL_FILTER)
        if p: self.ex_w.setText(p)

    def _ex_run(self):
        w = self.ex_w.text().strip()
        if not w or not Path(w).exists():
            w2 = find_latest_best()
            if w2: self.ex_w.setText(w2); w = w2
            else: QMessageBox.warning(self,'Error','No weights found'); return
        fmt = self.ex_fmt.currentText()
        self.log_panel.clear(); self.ex_btn.setEnabled(False)
        self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), f'📦 Exporting to {fmt.upper()}...'))
        QApplication.processEvents()
        try:
            from ultralytics import YOLO
            model = YOLO(w)
            out = model.export(format=fmt, imgsz=int(self.ex_sz.currentText()),
                half=self.ex_half.isChecked(), int8=self.ex_int8.isChecked(),
                nms=self.ex_nms.isChecked(), simplify=fmt=='onnx',
                device='0' if self.studio.gpu_ok else 'cpu')
            sz = Path(out).stat().st_size / 1e6
            self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), f'✅ {fmt.upper()} exported!'))
            self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), f'   Path: {out}'))
            self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), f'   Size: {sz:.2f} MB'))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), f'❌ Failed: {e}'))
        finally: self.ex_btn.setEnabled(True)
