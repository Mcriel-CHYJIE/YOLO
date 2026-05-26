# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""Tools 标签页 — 导入视频到预处理目录 + 标签导入导出"""

import shutil, zipfile
from pathlib import Path
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal
from main.core.base import *
from main.config import ROOT, load_paths


_WORKER = None
_THREAD = None


class _ImportWorker(QObject):
    """后台复制整个文件夹到预处理目录的工作线程"""
    progress = pyqtSignal(int, str)   # (percent, folder_name)
    finished = pyqtSignal(int)        # total copied
    error = pyqtSignal(str)

    def __init__(self, src_dir, dst_base):
        super().__init__()
        self.src_dir = Path(src_dir)
        self.dst_base = Path(dst_base)
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            folder_name = self.src_dir.name
            dst_dir = self.dst_base / folder_name
            if dst_dir.exists():
                stem = folder_name
                count = 1
                while dst_dir.exists():
                    dst_dir = self.dst_base / f'{stem}_{count}'
                    count += 1
                folder_name = dst_dir.name

            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
            files = [p for p in sorted(self.src_dir.iterdir())
                     if p.is_file() and p.suffix.lower() in video_exts]
            if not files:
                self.error.emit(f'No video files found in:\n{self.src_dir}')
                return

            dst_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            total = len(files)
            for i, f in enumerate(files):
                if self._stopped:
                    self.error.emit('Import cancelled')
                    return
                shutil.copy2(str(f), str(dst_dir / f.name))
                copied += 1
                pct = int((i + 1) / total * 100)
                self.progress.emit(pct, f.name)
            self.finished.emit(copied)
        except Exception as e:
            self.error.emit(str(e))


class ToolsTab(QWidget):
    """工具标签页 — 视频导入工具"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._load_ui()
        self._post_process_ui()
        self._connect_signals()
        self._load_paths()

    # ═══════════════════════════════════════════
    # UI 加载
    # ═══════════════════════════════════════════

    def _load_ui(self):
        """加载 Qt Designer .ui 文件"""
        ui_path = Path(__file__).resolve().parent / 'tools.ui'
        uic.loadUi(str(ui_path), self)
        # objectName: titleLabel, colsLo,
        #   col1Lo, dataGroup, importLabel, importPath, importBtn, importProgress, dataStatus,
        #   otherGroup, otherGroup2, otherGroup3

    def _post_process_ui(self):
        """主题适配，stretch 比例"""
        # ── 三列等宽 ──
        self.colsLo.setStretch(0, 1)
        self.colsLo.setStretch(1, 1)
        self.colsLo.setStretch(2, 1)

        # ── 标题 ──
        self.titleLabel.setStyleSheet(f'font-size:18px;font-weight:700;color:{TEXT};padding:0;margin:0;')
        self.titleLabel.setFixedHeight(24)

        # ── 所有 QGroupBox 统一样式 ──
        for g in (self.otherGroup, self.otherGroup2, self.otherGroup3, self.otherPlaceholder):
            g.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            g.setStyleSheet(f'''
                QGroupBox{{font-weight:600;font-size:10px;color:{TEXT2};
                    border:1px solid {BORDER};border-radius:6px;
                    margin-top:8px;padding:10px 8px 8px;background:{CARD};}}
                QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 5px;
                    background:{CARD};}}
            ''')
        # Label group 自然高度，替代默认 Preferred
        self.otherGroup.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        # dataGroup 自然高度
        self.dataGroup.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.dataGroup.setStyleSheet(f'''
            QGroupBox{{font-weight:600;font-size:10px;color:{TEXT2};
                border:1px solid {BORDER};border-radius:6px;
                margin-top:8px;padding:10px 8px 8px;background:{CARD};}}
            QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 5px;
                background:{CARD};}}
        ''')

        # ── dataStatus ──
        self.dataStatus.setStyleSheet(f'font-size:9px;color:{TEXT3};padding:0;margin:0;')
        self.dataStatus.setFixedHeight(13)

        # ── Import label ──
        self.importLabel.setStyleSheet(
            f'font-size:10px;font-weight:600;color:{TEXT3};padding:0;margin:0;')
        self.importLabel.setFixedHeight(14)

        # ── Import path display ──
        self.importPath.setStyleSheet(f'''
            QLineEdit{{background:{BG};border:1px solid {BORDER};
                border-radius:3px;padding:1px 6px;font-size:10px;color:{TEXT3};}}
        ''')

        # ── Import button ──
        self.importBtn.setStyleSheet(f'''
            QPushButton{{background:{PRI};color:#fff;border:none;
                padding:4px 0;font-size:11px;font-weight:600;border-radius:4px;}}
            QPushButton:hover{{background:{PRI_H};}}
        ''')

        # ── Label import/export widgets ──
        self.labelPath.setStyleSheet(
            f'font-size:9px;color:{TEXT3};padding:0;margin:0;')
        self.labelStatus.setStyleSheet(
            f'font-size:9px;color:{TEXT3};padding:0;margin:0;')
        self.labelFolderCombo.setStyleSheet(f'''
            QComboBox{{border:1px solid {BORDER};border-radius:4px;
                padding:2px 6px;background:{CARD};font-size:10px;color:{TEXT};}}
            QComboBox:focus{{border-color:{PRI};}}
            QComboBox::drop-down{{border:none;width:16px;}}
        ''')
        self.labelExportBtn.setStyleSheet(f'''
            QPushButton{{background:{PRI};color:#fff;border:none;
                padding:4px 0;font-size:11px;font-weight:600;border-radius:4px;}}
            QPushButton:hover{{background:{PRI_H};}}
            QPushButton:disabled{{background:#a5d6a5;}}
        ''')
        self.labelImportBtn.setStyleSheet(f'''
            QPushButton{{background:{AMBER};color:#fff;border:none;
                padding:4px 0;font-size:11px;font-weight:600;border-radius:4px;}}
            QPushButton:hover{{background:#d97706;}}
            QPushButton:disabled{{background:#fcd34d;}}
        ''')

        # ── Progress bar ──
        self.importProgress.setStyleSheet(f'''
            QProgressBar{{border:none;border-radius:1px;height:3px;
                background:{BORDER};text-align:center;}}
            QProgressBar::chunk{{background:{PRI};border-radius:1px;}}
        ''')

    # ═══════════════════════════════════════════
    # Signals
    # ═══════════════════════════════════════════

    def _connect_signals(self):
        self.importBtn.clicked.connect(self._start_import)
        self.labelExportBtn.clicked.connect(self._export_label_zip)
        self.labelImportBtn.clicked.connect(self._import_label_zip)

    # ═══════════════════════════════════════════
    # Data Loading
    # ═══════════════════════════════════════════

    def _load_paths(self):
        """从 paths.json 读取路径并显示"""
        paths = load_paths()
        preproc = paths.get('preproc_before', '')
        self.importPath.setText(preproc)
        self._preproc_dir = preproc

        label_dir = paths.get('label_dir', '')
        if label_dir:
            after = Path(label_dir) / 'after'
            self.labelPath.setText(str(after) if after.exists() else '—')
            self._label_after = after
            # 填充子文件夹下拉
            self.labelFolderCombo.clear()
            if after.exists():
                dirs = sorted([d.name for d in after.iterdir() if d.is_dir()])
                if dirs:
                    self.labelFolderCombo.addItems(dirs)
        else:
            self.labelPath.setText('—')
            self._label_after = None

    # ═══════════════════════════════════════════
    # Import Logic
    # ═══════════════════════════════════════════

    def _start_import(self):
        dst = self._preproc_dir
        if not dst:
            self.dataStatus.setStyleSheet(f'font-size:9px;color:{RED};')
            self.dataStatus.setText('No preproc dir configured in Settings')
            return

        src = QFileDialog.getExistingDirectory(self, 'Select Source Video Folder',
            str(ROOT))
        if not src:
            return

        self.importBtn.setEnabled(False)
        self.importProgress.setValue(0)
        self.dataStatus.setStyleSheet(f'font-size:9px;color:{AMBER};')
        self.dataStatus.setText('Importing...')

        global _WORKER, _THREAD
        _WORKER = _ImportWorker(src, dst)
        _THREAD = QThread(self)
        _WORKER.moveToThread(_THREAD)

        _THREAD.started.connect(_WORKER.run)
        _WORKER.progress.connect(self._on_import_progress)
        _WORKER.finished.connect(self._on_import_done)
        _WORKER.error.connect(self._on_import_error)
        _WORKER.finished.connect(_THREAD.quit)
        _WORKER.error.connect(_THREAD.quit)
        _THREAD.finished.connect(_WORKER.deleteLater)
        _THREAD.finished.connect(self._on_import_thread_done)
        _THREAD.start()

    def _on_import_progress(self, pct, filename):
        self.importProgress.setValue(pct)
        self.dataStatus.setText(f'Importing... {filename}')

    def _on_import_done(self, count):
        self.dataStatus.setStyleSheet(f'font-size:9px;color:{GREEN};')
        self.dataStatus.setText(f'Done — {count} video(s) imported')

    def _on_import_error(self, msg):
        self.dataStatus.setStyleSheet(f'font-size:9px;color:{RED};')
        self.dataStatus.setText(msg)

    def _on_import_thread_done(self):
        self.importBtn.setEnabled(True)

    # ═══════════════════════════════════════════
    # Label Export / Import
    # ═══════════════════════════════════════════

    def _export_label_zip(self):
        """将选中的子文件夹压缩为 ZIP"""
        after = self._label_after
        if not after or not after.exists():
            self._set_label_status(RED, 'No after dir configured in Settings')
            return

        folder_name = self.labelFolderCombo.currentText()
        if not folder_name:
            self._set_label_status(RED, 'No folder selected')
            return

        src_folder = after / folder_name
        if not src_folder.exists():
            self._set_label_status(RED, f'Folder not found: {folder_name}')
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Export Label ZIP', str(after / f'{folder_name}.zip'),
            'ZIP Files (*.zip)', options=QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        if not save_path:
            return

        self._set_label_status(AMBER, 'Compressing...')
        self.labelExportBtn.setEnabled(False)
        self.labelImportBtn.setEnabled(False)
        QApplication.processEvents()

        try:
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in src_folder.rglob('*'):
                    if f.is_file():
                        arcname = str(f.relative_to(after))
                        zf.write(str(f), arcname)
            self._set_label_status(GREEN, f'Exported → {Path(save_path).name}')
        except Exception as e:
            self._set_label_status(RED, f'Export failed: {e}')
        finally:
            self.labelExportBtn.setEnabled(True)
            self.labelImportBtn.setEnabled(True)

    def _import_label_zip(self):
        """导入 ZIP 压缩包并解压到 original/after"""
        after = self._label_after
        if not after:
            self._set_label_status(RED, 'No after dir configured in Settings')
            return

        zip_path, _ = QFileDialog.getOpenFileName(
            self, 'Import Label ZIP', '', 'ZIP Files (*.zip)',
            options=QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        if not zip_path:
            return

        self._set_label_status(AMBER, 'Extracting...')
        self.labelExportBtn.setEnabled(False)
        self.labelImportBtn.setEnabled(False)
        QApplication.processEvents()

        try:
            extracted = 0
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    # 安全解压：防止 zip slip
                    name = info.filename
                    if '..' in name or name.startswith('/') or name.startswith('\\'):
                        continue
                    dst = (after / name).resolve()
                    if not str(dst).startswith(str(after.resolve())):
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    zf.extract(info, after)
                    extracted += 1
            self._set_label_status(GREEN, f'Extracted {extracted} file(s) to after/')
        except Exception as e:
            self._set_label_status(RED, f'Import failed: {e}')
        finally:
            self.labelExportBtn.setEnabled(True)
            self.labelImportBtn.setEnabled(True)

    def _set_label_status(self, color, msg):
        self.labelStatus.setStyleSheet(f'font-size:9px;color:{color};padding:0;margin:0;')
        self.labelStatus.setText(msg)
        self.labelStatus.repaint()

    # ═══════════════════════════════════════════
    # 主题刷新
    # ═══════════════════════════════════════════

    def on_theme_changed(self):
        """Settings 主题切换后刷新样式"""
        self._post_process_ui()
