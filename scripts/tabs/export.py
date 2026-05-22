"""导出标签页 — 从 Deploy 拆分出来"""
from scripts.tabs.base import *
from PyQt5 import uic


class ExportTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._build()
        self._init_widgets()
        self._connect_signals()

    def _build(self):
        ui_path = Path(__file__).resolve().parent.parent / 'ui' / 'export.ui'
        uic.loadUi(str(ui_path), self)

    def _init_widgets(self):
        e = cfg['export']
        self._fmt.addItems(e['format_options'])
        self._fmt.setCurrentText(e['format'])
        self._sz.addItems([str(v) for v in e['imgsz_options']])
        self._sz.setCurrentText(str(e['imgsz']))
        self._half.setChecked(e['half'])
        self._int8.setChecked(e['int8'])
        self._nms.setChecked(e['nms'])
        self._load_latest()

    def _connect_signals(self):
        self.browseBtn.clicked.connect(self._browse)
        self._exp_btn.setObjectName('pri')
        self._exp_btn.clicked.connect(self._run_export)

    def _load_latest(self):
        w = find_latest_best()
        if w:
            self._w.setText(w)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Select Weights', 'runs', MODEL_FILTER)
        if p:
            self._w.setText(p)

    def _run_export(self):
        w = self._w.text().strip()
        if not w or not Path(w).exists():
            w2 = find_latest_best()
            if w2:
                self._w.setText(w2); w = w2
            else:
                QMessageBox.warning(self, 'Error', 'No weights found')
                return
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
