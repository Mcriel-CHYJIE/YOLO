# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""视频预处理标签页 — 重命名+缩放+抽帧"""
from main.core.base import *
from main.config import load_paths
from PyQt5 import uic
from .service import VideoPreprocessWorker
import cv2


class PreprocessTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._worker = None
        self._build_ui()
        self._init_widgets()
        self._connect()
        self._refresh_folder_list()

    @property
    def _before_root(self):
        p = load_paths().get('preproc_before', '')
        return Path(p) if p else Path()
    @property
    def _after_root(self):
        p = load_paths().get('preproc_after', '')
        return Path(p) if p else Path()

    def _build_ui(self):
        ui_path = Path(__file__).resolve().parent / 'preprocess.ui'
        uic.loadUi(str(ui_path), self)

    def _init_widgets(self):
        # 浏览按钮加文件夹图标
        for btn in [self.srcBrowseBtn, self.outBrowseBtn]:
            btn.setText('')
            btn.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
            btn.setIconSize(QSize(16, 16))
        self.video_list.setStyleSheet(f'''
            QListWidget {{background:{BG};border:1px solid {BORDER};border-radius:4px;
                font-size:11px;color:{TEXT};padding:3px;}}
            QListWidget::item {{padding:4px 6px;border-radius:3px;}}
            QListWidget::item:selected {{background:{PRI};color:#fff;}}''')
        for placeholder, label, color, attr_v, attr_c in [
            (self.videoMetricCard, 'Video', TEXT, '_pv', '_pv_card'),
            (self.frameMetricCard, 'Frame', GREEN, '_pf', '_pf_card'),
        ]:
            card = MetricCard(label, color, '—')
            idx = self.metricRowPP.indexOf(placeholder)
            self.metricRowPP.removeWidget(placeholder)
            placeholder.deleteLater()
            self.metricRowPP.insertWidget(idx, card, 1)
            setattr(self, attr_v, card.value_label)
            setattr(self, attr_c, card)
            card.value_label.setStyleSheet(
                f'font-size:16px;font-weight:700;color:{color};qproperty-alignment:AlignCenter;')

    # ═══════════════════════════════════════════
    # Signal Wiring
    # ═══════════════════════════════════════════
    def _connect(self):
        self.srcBrowseBtn.clicked.connect(self._browse_input)
        self.outBrowseBtn.clicked.connect(self._browse_output)
        self.resetBtn.clicked.connect(self._refresh_folder_list)
        self.start_btn.setObjectName('pri')
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.setObjectName('danger')
        self.stop_btn.clicked.connect(self._stop)

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Input Directory',
            str(self._before_root) if self._before_root.exists() else str(ROOT))
        if folder:
            fp = Path(folder)
            self.src_input.setText(str(fp))
            self.out_input.setText(str(self._after_root / fp.name))
            self._refresh_video_list(fp)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Output Directory',
            str(self._after_root) if self._after_root.exists() else str(ROOT))
        if folder: self.out_input.setText(str(Path(folder)))

    def _refresh_folder_list(self):
        self.src_input.clear(); self.out_input.clear()
        self.video_list.clear(); self.video_count.setText('0 videos')

    def _refresh_video_list(self, folder: Path):
        self.video_list.clear()
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in VIDEO_EXTS and f.is_file():
                self.video_list.addItem(f.name)
        self.video_count.setText(f'{self.video_list.count()} videos')

    def _start(self):
        s = self.src_input.text().strip(); o = self.out_input.text().strip()
        if not s or not Path(s).exists():
            QMessageBox.warning(self, 'Warning', '请选择有效输入目录'); return
        if not o: QMessageBox.warning(self, 'Warning', '请选择输出目录'); return
        exts = VIDEO_EXTS
        videos = [f for f in Path(s).iterdir() if f.suffix.lower() in exts and f.is_file()]
        if not videos: QMessageBox.warning(self, 'Warning', '未找到视频文件'); return
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.src_input.setEnabled(False); self.out_input.setEnabled(False)
        self.progress_bar.setValue(0); self._pv.setText('—'); self._pf.setText('—')
        self.log_panel.clear()
        self._log(f'▶ 开始预处理: {Path(s).name}')
        self._worker = VideoPreprocessWorker(src_folder=s, out_folder=o)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(lambda c, t: (
            self.progress_bar.setValue(int(c / t * 100)), self._pv.setText(f'{c}/{t}')))
        self._worker.video_progress.connect(lambda c, t: self._pf.setText(f'{c}'))
        self._worker.image_saved.connect(self._on_image_saved)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop(); self.stop_btn.setEnabled(False); self._log(' 停止中...')

    def _on_image_saved(self, image_path: str):
        try:
            c = int(self.preview_stats.text().split()[0]) + 1
            self.preview_stats.setText(f'{c} images processed')
            from PyQt5.QtGui import QPixmap
            p = QPixmap(image_path)
            if not p.isNull():
                self.preview_label.setPixmap(p.scaled(
                    self.preview_label.width(), self.preview_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.preview_label.setText('')
        except: pass

    def _on_done(self, ok, msg):
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.src_input.setEnabled(True); self.out_input.setEnabled(True)
        self.progress_bar.setValue(100 if ok else 0)
        self._log(f'{"" if ok else ""} {msg}')
        self.status_label.setText(msg); self._worker = None

    def _log(self, msg):
        self.log_panel.append(format_log(datetime.now().strftime('%H:%M:%S'), msg))
