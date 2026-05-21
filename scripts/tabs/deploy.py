"""验证 + 导出合并标签页"""
from scripts.tabs.base import *
from PyQt5 import uic


class DeployTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.validator = None
        self._build()
        self._init_widgets()
        self._connect_signals()
        self._load_latest()

    def _build(self):
        ui_path = Path(__file__).resolve().parent.parent / 'ui' / 'deploy.ui'
        uic.loadUi(str(ui_path), self)

    def _init_widgets(self):
        for c in [self.m50CardVE, self.m95CardVE, self.pCardVE, self.rCardVE]:
            c.setStyleSheet(f'background:{BG};border-radius:5px;')
        self._cm.ax.text(.5, .5, 'Run validation to view confusion matrix',
            ha='center', va='center', transform=self._cm.ax.transAxes,
            fontsize=11, color=TEXT3)
        self._cm.ax.axis('off'); self._cm.rf()
        e = cfg['export']
        self._fmt.addItems(e['format_options'])
        self._fmt.setCurrentText(e['format'])
        self._sz.addItems([str(v) for v in e['imgsz_options']])
        self._sz.setCurrentText(str(e['imgsz']))
        self._half.setChecked(e['half'])
        self._int8.setChecked(e['int8'])
        self._nms.setChecked(e['nms'])

    def _connect_signals(self):
        self.browseBtn.clicked.connect(self._browse)
        self._val_btn.setObjectName('pri')
        self._val_btn.clicked.connect(self._run_val)
        self._exp_btn.setObjectName('pri')
        self._exp_btn.clicked.connect(self._run_export)

    def _load_latest(self):
        w = find_latest_best()
        if w: self._w.setText(w)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Select Weights', 'runs', MODEL_FILTER)
        if p: self._w.setText(p)

    def _run_val(self):
        w = self._w.text().strip()
        if not w or not Path(w).exists():
            w2 = find_latest_best()
            if w2: self._w.setText(w2); w = w2
            else: QMessageBox.warning(self, 'Error', 'No weights found'); return
        self._val_btn.setEnabled(False); self._val_btn.setText('Validating…')
        gpu = self.studio.gpu_ok
        class V(QThread):
            d = pyqtSignal(object)
            def run(self):
                try:
                    from ultralytics import YOLO
                    v = cfg['validation']
                    r = YOLO(str(w)).val(data=DATA_YAML, device='0' if gpu else 'cpu',
                        imgsz=v['imgsz'], batch=v['batch'], conf=v['conf'], iou=v['iou'],
                        plots=True, verbose=False)
                    self.d.emit(r)
                except Exception as e:
                    import traceback; traceback.print_exc(); self.d.emit(e)
        self.validator = V(); self.validator.d.connect(self._vd); self.validator.start()

    def _vd(self, r):
        self._val_btn.setEnabled(True); self._val_btn.setText('Run Validation'); self.validator = None
        if isinstance(r, Exception):
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f' Validation failed: {r}'))
            return
        try:
            b = r.box
            self._vm50.setText(f'{b.map50:.4f}'); self._vm95.setText(f'{b.map:.4f}')
            self._vp.setText(f'{b.p.mean():.4f}'); self._vr.setText(f'{b.r.mean():.4f}')
            rows = ''
            for i, nm in enumerate(CLASSES):
                mi = b.maps50[i] if hasattr(b, 'maps50') and len(b.maps50) > i else 0
                pi = b.p[i] if hasattr(b, 'p') and len(b.p) > i else 0
                ri = b.r[i] if hasattr(b, 'r') and len(b.r) > i else 0
                rows += f'<tr><td style="padding:1px 6px">{nm}</td><td style="padding:1px 6px;color:{PRI}\">{mi:.4f}</td><td style="padding:1px 6px">{pi:.4f}</td><td style="padding:1px 6px">{ri:.4f}</td></tr>'
            self._vt.setText(
                f'<table style="width:100%;font-size:10px;border-collapse:collapse;"><tr style="color:{TEXT3}"><th>Class</th><th>mAP</th><th>P</th><th>R</th></tr>{rows}</table>')
            self._vt.setVisible(True)
            sd = Path(r.save_dir) if hasattr(r, 'save_dir') else None
            if sd:
                imgs = list(sd.glob('confusion_matrix*.png'))
                if imgs:
                    self._cm.ax.clear(); self._cm.ax.axis('on')
                    self._cm.ax.imshow(plt.imread(str(imgs[0]))); self._cm.ax.axis('off'); self._cm.rf()
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), ' Validation complete'))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f' Display error: {e}'))

    def _run_export(self):
        w = self._w.text().strip()
        if not w or not Path(w).exists():
            w2 = find_latest_best()
            if w2: self._w.setText(w2); w = w2
            else: QMessageBox.warning(self, 'Error', 'No weights found'); return
        fmt = self._fmt.currentText()
        self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f' Exporting to {fmt.upper()}...'))
        self._exp_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            from ultralytics import YOLO
            model = YOLO(w)
            out = model.export(format=fmt, imgsz=int(self._sz.currentText()),
                half=self._half.isChecked(), int8=self._int8.isChecked(),
                nms=self._nms.isChecked(), simplify=fmt == 'onnx',
                device='0' if self.studio.gpu_ok else 'cpu')
            sz = Path(out).stat().st_size / 1e6
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f' {fmt.upper()} exported!'))
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f'   Path: {out}'))
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f'   Size: {sz:.2f} MB'))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f' Failed: {e}'))
        finally:
            self._exp_btn.setEnabled(True)
