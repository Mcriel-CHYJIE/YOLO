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

_CRAWLER_WORKER = None
_CRAWLER_THREAD = None


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


# ── 压缩包解压辅助函数 ──

def _extract_zip(path, dest):
    import zipfile
    count = 0
    with zipfile.ZipFile(path, 'r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if '..' in name or name.startswith('/') or name.startswith('\\'):
                continue
            dst = (dest / name).resolve()
            if not str(dst).startswith(str(dest.resolve())):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            zf.extract(info, dest)
            count += 1
    return count


def _extract_tar(path, dest):
    import tarfile
    count = 0
    stem = Path(path).name.lower()
    if stem.endswith('.tar.gz') or stem.endswith('.tgz'):
        mode = 'r:gz'
    elif stem.endswith('.tar.bz2'):
        mode = 'r:bz2'
    elif stem.endswith('.tar.xz'):
        mode = 'r:xz'
    else:
        mode = 'r'
    with tarfile.open(path, mode) as tf:
        for info in tf.getmembers():
            if not info.isfile():
                continue
            name = info.name
            if '..' in name or name.startswith('/') or name.startswith('\\'):
                continue
            dst = (dest / name).resolve()
            if not str(dst).startswith(str(dest.resolve())):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tf.extract(info, dest)
            count += 1
    return count


def _extract_patool(path, dest):
    import patoolib
    patoolib.extract_archive(path, outdir=str(dest))
    # count extracted files
    count = 0
    for f in dest.rglob('*'):
        if f.is_file():
            count += 1
    return count


class _CrawlerWorker(QObject):
    """百度图片爬虫工作线程"""
    progress = pyqtSignal(int, int)   # (downloaded, total)
    finished = pyqtSignal(int)         # total downloaded
    error = pyqtSignal(str)

    def __init__(self, keyword, dst_dir):
        super().__init__()
        self.keyword = keyword
        self.dst_dir = Path(dst_dir)
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            self.dst_dir.mkdir(parents=True, exist_ok=True)
            total = self._baidu_crawl()
            self.finished.emit(total)
        except Exception as e:
            self.error.emit(str(e))

    def _baidu_crawl(self):
        """百度图片搜索爬虫"""
        import urllib.request, urllib.parse, json, os, time, ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        keyword_enc = urllib.parse.quote(self.keyword)
        downloaded = 0
        target = 100  # 默认抓取100张
        page = 0

        while downloaded < target and not self._stopped:
            url = (f'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=&ipn=rj&ct=201326592&is=&fp=result&fr=&word={keyword_enc}'
                   f'&queryWord={keyword_enc}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&expermode=&nojc=&isAsync=&pn={page*30}&rn=30&gsm=1e&1669525513756=')

            req = urllib.request.Request(url, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=15)
                data = json.loads(resp.read().decode('utf-8', errors='ignore'))
            except:
                page += 1
                continue

            imgs = data.get('data', [])
            if not imgs:
                break

            for img in imgs:
                if self._stopped:
                    break
                img_url = img.get('thumbURL') or img.get('middleURL') or img.get('objURL') or ''
                if not img_url:
                    continue

                # 过滤掉gif
                ext = os.path.splitext(img_url.split('?')[0])[1].lower()
                if ext in ('.gif',):
                    continue
                if not ext:
                    ext = '.jpg'

                fname = f'{self.keyword}_{downloaded+1}{ext}'
                fpath = self.dst_dir / fname
                if fpath.exists():
                    continue

                try:
                    img_req = urllib.request.Request(img_url, headers=headers)
                    img_resp = urllib.request.urlopen(img_req, timeout=10)
                    with open(fpath, 'wb') as f:
                        f.write(img_resp.read())
                    downloaded += 1
                    self.progress.emit(downloaded, target)
                    if downloaded >= target:
                        break
                except:
                    continue
                time.sleep(0.2)

            page += 1

        return downloaded


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
        #   labelRow, labelLabel, labelFolderCombo, labelExportBtn,
        #   labelImportRow, labelImportLabel, labelImportPath, labelImportBtn, labelStatus

    def _post_process_ui(self):
        """主题适配，stretch 比例"""
        # ── 三列等宽 ──
        self.colsLo.setStretch(0, 1)
        self.colsLo.setStretch(1, 1)
        self.colsLo.setStretch(2, 1)

        # ── 第一列：爬虫组件自然高度，底部加 other 占位组件 ──
        self.otherCrawlerPlaceholder = QGroupBox('... Other')
        self.otherCrawlerPlaceholder.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.otherCrawlerPlaceholder.setStyleSheet(f'''
            QGroupBox{{font-weight:600;font-size:10px;color:{TEXT2};
                border:1px solid {BORDER};border-radius:6px;
                margin-top:8px;padding:10px 8px 8px;background:{CARD};}}
            QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 5px;
                background:{CARD};}}
        ''')
        self.col1Lo.addWidget(self.otherCrawlerPlaceholder, 1)

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
        self.labelLabel.setStyleSheet(
            f'font-size:10px;font-weight:600;color:{TEXT3};padding:0;margin:0;')
        self.labelLabel.setFixedHeight(14)
        self.labelStatus.setStyleSheet(
            f'font-size:9px;color:{TEXT3};padding:0;margin:0;')
        self.labelFolderCombo.setStyleSheet(f'''
            QComboBox{{border:1px solid {BORDER};border-radius:4px;
                padding:2px 6px;background:{CARD};font-size:10px;color:{TEXT};}}
            QComboBox:focus{{border-color:{PRI};}}
            QComboBox::drop-down{{border:none;width:16px;}}
        ''')
        from PyQt5.QtWidgets import QListView
        _lv = QListView()
        _lv.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.labelFolderCombo.setView(_lv)
        self.labelFolderCombo.setMaxVisibleItems(10)
        self.labelExportBtn.setStyleSheet(f'''
            QPushButton{{background:{PRI};color:#fff;border:none;
                padding:4px 0;font-size:11px;font-weight:600;border-radius:4px;}}
            QPushButton:hover{{background:{PRI_H};}}
            QPushButton:disabled{{background:#a5d6a5;}}
        ''')
        self.labelImportBtn.setStyleSheet(f'''
            QPushButton{{background:{PRI};color:#fff;border:none;
                padding:4px 0;font-size:11px;font-weight:600;border-radius:4px;}}
            QPushButton:hover{{background:{PRI_H};}}
            QPushButton:disabled{{background:#a5d6a5;}}
        ''')
        self.labelImportLabel.setStyleSheet(
            f'font-size:10px;font-weight:600;color:{TEXT3};padding:0;margin:0;')
        self.labelImportLabel.setFixedHeight(14)
        self.labelImportPath.setStyleSheet(f'''
            QLineEdit{{background:{BG};border:1px solid {BORDER};
                border-radius:3px;padding:1px 6px;font-size:10px;color:{TEXT3};}}
        ''')

        # ── 导入进度条 ──
        self.importProgress.setStyleSheet(f'''
            QProgressBar{{border:none;border-radius:1px;height:3px;
                background:{BORDER};text-align:center;}}
            QProgressBar::chunk{{background:{PRI};border-radius:1px;}}
        ''')

        # ── Image Crawler ──
        self.crawlerKeyword.setStyleSheet(f'''
            QPlainTextEdit{{background:{BG};border:1px solid {BORDER};
                border-radius:4px;padding:4px 8px;font-size:13px;color:{TEXT};}}
            QPlainTextEdit:focus{{border-color:{PRI};}}
        ''')
        self.crawlerKeyword.setPlaceholderText('Enter keywords to search...')
        self.crawlerPath.setStyleSheet(f'''
            QLineEdit{{background:{BG};border:1px solid {BORDER};
                border-radius:3px;padding:1px 6px;font-size:10px;color:{TEXT3};}}
        ''')
        self.crawlerBrowseBtn.setStyleSheet(f'''
            QPushButton{{background:{CARD};border:1px solid {BORDER};
                border-radius:3px;font-size:12px;padding:0;}}
            QPushButton:hover{{background:{BORDER};}}
        ''')
        self.crawlerStartBtn.setStyleSheet(f'''
            QPushButton{{background:{PRI};color:#fff;border:none;
                padding:4px 0;font-size:11px;font-weight:600;border-radius:4px;}}
            QPushButton:hover{{background:{PRI_H};}}
            QPushButton:disabled{{background:#a5d6a5;}}
        ''')
        self.crawlerProgress.setStyleSheet(f'''
            QProgressBar{{border:1px solid {BORDER};border-radius:3px;height:16px;
                background:{BG};text-align:center;font-size:8px;color:{TEXT3};}}
            QProgressBar::chunk{{background:{PRI};border-radius:2px;}}
        ''')
        self.crawlerInfo.setStyleSheet(
            f'font-size:9px;color:{TEXT3};padding:0;margin:0;')

    # ═══════════════════════════════════════════
    # Signals
    # ═══════════════════════════════════════════

    def _connect_signals(self):
        self.importBtn.clicked.connect(self._start_import)
        self.labelExportBtn.clicked.connect(self._export_label_zip)
        self.labelImportBtn.clicked.connect(self._import_label_zip)
        self.crawlerBrowseBtn.clicked.connect(self._crawler_browse)
        self.crawlerStartBtn.clicked.connect(self._crawler_start)

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
            self._label_after = after
            self.labelImportPath.setText(str(after) if after.exists() else '—')
            # 填充子文件夹下拉（Export 用）
            self.labelFolderCombo.clear()
            if after.exists():
                dirs = sorted([d.name for d in after.iterdir() if d.is_dir()])
                if dirs:
                    self.labelFolderCombo.addItems(dirs)
        else:
            self.labelImportPath.setText('—')
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
        """将选中的子文件夹（来自 label_dir/after）压缩为 ZIP"""
        after = self._label_after
        if not after:
            self._set_label_status(RED, 'No label dir configured in Settings')
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
            self._set_label_status(GREEN, f'Exported \u2192 {Path(save_path).name}')
        except Exception as e:
            self._set_label_status(RED, f'Export failed: {e}')
        finally:
            self.labelExportBtn.setEnabled(True)
            self.labelImportBtn.setEnabled(True)

    def _import_label_zip(self):
        """导入压缩包（zip/rar/7z/tar/…）并解压到 label_dir/after"""
        after = self._label_after
        if not after:
            self._set_label_status(RED, 'No label dir configured in Settings')
            return

        archive_path, _ = QFileDialog.getOpenFileName(
            self, 'Import Label Archive', '',
            'Archives (*.zip *.rar *.7z *.tar *.tar.gz *.tgz *.tar.bz2 *.tar.xz)',
            options=QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        if not archive_path:
            return

        self._set_label_status(AMBER, 'Extracting...')
        self.labelExportBtn.setEnabled(False)
        self.labelImportBtn.setEnabled(False)
        QApplication.processEvents()

        try:
            after.mkdir(parents=True, exist_ok=True)
            suffix = Path(archive_path).suffix.lower()
            stem = Path(archive_path).name.lower()

            if suffix == '.zip':
                extracted = _extract_zip(archive_path, after)
            elif suffix == '.rar' or suffix == '.7z':
                extracted = _extract_patool(archive_path, after)
            elif stem.endswith('.tar.gz') or stem.endswith('.tgz') \
                 or stem.endswith('.tar.bz2') or stem.endswith('.tar.xz') \
                 or suffix == '.tar':
                extracted = _extract_tar(archive_path, after)
            else:
                self._set_label_status(RED, f'Unsupported format: {suffix}')
                return
            self._set_label_status(GREEN, f'Extracted {extracted} file(s)')
        except Exception as e:
            msg = str(e)
            if 'Cannot find working tool' in msg:
                msg = 'Need 7-Zip or WinRAR installed to extract this format'
            self._set_label_status(RED, f'Import failed: {msg}')
        finally:
            self.labelExportBtn.setEnabled(True)
            self.labelImportBtn.setEnabled(True)

    def _set_label_status(self, color, msg):
        self.labelStatus.setStyleSheet(f'font-size:9px;color:{color};padding:0;margin:0;')
        self.labelStatus.setText(msg)
        self.labelStatus.repaint()

    # ═══════════════════════════════════════════
    # Image Crawler
    # ═══════════════════════════════════════════

    def _crawler_browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Select download directory',
            options=QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        if d:
            self.crawlerPath.setText(d)

    def _crawler_start(self):
        keyword = self.crawlerKeyword.toPlainText().strip()
        if not keyword:
            self.crawlerInfo.setText('Please enter a keyword')
            return
        dst = self.crawlerPath.text().strip()
        if not dst:
            self.crawlerInfo.setText('Please select a download directory')
            return

        self.crawlerStartBtn.setEnabled(False)
        self.crawlerStartBtn.setText('Running...')
        self.crawlerProgress.setValue(0)
        QApplication.processEvents()

        global _CRAWLER_WORKER, _CRAWLER_THREAD
        _CRAWLER_WORKER = _CrawlerWorker(keyword, dst)
        _CRAWLER_THREAD = QThread()
        _CRAWLER_WORKER.moveToThread(_CRAWLER_THREAD)
        _CRAWLER_THREAD.started.connect(_CRAWLER_WORKER.run)
        _CRAWLER_WORKER.finished.connect(self._crawler_done)
        _CRAWLER_WORKER.progress.connect(self._crawler_progress)
        _CRAWLER_WORKER.error.connect(self._crawler_error)
        _CRAWLER_THREAD.start()

    def _crawler_progress(self, count, total):
        self.crawlerProgress.setValue(int(count / total * 100))
        self.crawlerProgress.setFormat(f'{count}/{total}')
        self.crawlerInfo.setText(f'Downloading {self.crawlerKeyword.toPlainText().strip()}...')

    def _crawler_done(self, count):
        self.crawlerStartBtn.setEnabled(True)
        self.crawlerStartBtn.setText('▶ Start')
        self.crawlerProgress.setValue(100)
        self.crawlerProgress.setFormat(f'{count} images')
        self.crawlerInfo.setText(f'Done — saved to {self.crawlerPath.text()}')
        global _CRAWLER_THREAD
        if _CRAWLER_THREAD:
            _CRAWLER_THREAD.quit()
            _CRAWLER_THREAD = None

    def _crawler_error(self, msg):
        self.crawlerStartBtn.setEnabled(True)
        self.crawlerStartBtn.setText('▶ Start')
        self.crawlerProgress.setValue(0)
        self.crawlerProgress.setFormat('Error')
        global _CRAWLER_THREAD
        if _CRAWLER_THREAD:
            _CRAWLER_THREAD.quit()
            _CRAWLER_THREAD = None

    # ═══════════════════════════════════════════
    # 主题刷新
    # ═══════════════════════════════════════════

    def on_theme_changed(self):
        """Settings 主题切换后刷新样式"""
        self._post_process_ui()
