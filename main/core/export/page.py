# =============================================================================
# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# =============================================================================

"""导出标签页"""
from main.core.base import *
from PyQt5 import uic
from .service import run_export


class ExportTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._build_ui()
        self._init_widgets()
        self._connect()

    def _build_ui(self):
        ui_path = Path(__file__).resolve().parent / 'export.ui'
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

    # ═══════════════════════════════════════════
    # Signal Wiring
    # ═══════════════════════════════════════════
    def _connect(self):
        self.browseBtn.clicked.connect(self._browse)
        self._exp_btn.setObjectName('pri')
        self._exp_btn.clicked.connect(self._run_export)

    def _load_latest(self):
        w = find_latest_best()
        if w:
            self._w.setText(w)

    def _browse(self):
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        p, _ = QFileDialog.getOpenFileName(self, 'Select Weights', str(ROOT / 'models'), MODEL_FILTER, options=opts)
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
            kw = dict(
                format=fmt,
                imgsz=int(self._sz.currentText()),
                half=self._half.isChecked(),
                int8=self._int8.isChecked(),
                nms=self._nms.isChecked(),
                device='0' if self.studio.gpu_ok else 'cpu',
            )
            # ONNX 专用参数（沿用 tool/export_onnx.py 逻辑）
            if fmt == 'onnx':
                kw['opset'] = 12
                kw['simplify'] = True
                kw['dynamic'] = False

            out = model.export(**kw)

            # ONNX 文件路径处理：ultralytics 默认导出到 .pt 同目录，路径已在 out 返回
            sz = Path(out).stat().st_size / 1e6
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f' {fmt.upper()} exported!'))
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f'   Path: {out}'))
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f'   Size: {sz:.2f} MB'))
            if fmt == 'onnx':
                self._log.append(format_log(datetime.now().strftime('%H:%M:%S'),
                    f'   Opset: {kw.get("opset","—")} | Simplify: {kw.get("simplify","—")}'))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._log.append(format_log(datetime.now().strftime('%H:%M:%S'), f' Failed: {e}'))
        finally:
            self._exp_btn.setEnabled(True)
