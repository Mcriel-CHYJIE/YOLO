# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""Tools 标签页 — 导入视频到预处理目录 + 标签导入导出"""

import shutil, zipfile, os
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


class _AnalyzeSignal(QObject):
    """从工作线程发射信号到主线程更新 UI"""
    done = pyqtSignal(str, str)       # summary, error
    progress_update = pyqtSignal(int, int)  # current, total


class ToolsTab(QWidget):
    """工具标签页 — 视频导入工具"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._analyze_signal = _AnalyzeSignal()
        self._analyze_signal.done.connect(self._analyze_done)
        self._analyze_signal.progress_update.connect(self._on_analyze_progress)
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
        self.colsLo.setStretch(2, 1)  # col3 占位

        # ── 标题 ──
        self.titleLabel.setStyleSheet(f'font-size:18px;font-weight:700;color:{TEXT};padding:0;margin:0;')
        self.titleLabel.setFixedHeight(24)

        # ── 所有 QGroupBox 统一样式 ──
        for g in (self.otherGroup, self.otherGroup3, self.otherPlaceholder, self.exportGroup):
            g.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            g.setStyleSheet(f'''
                QGroupBox{{font-weight:600;font-size:10px;color:{TEXT2};
                    border:1px solid {BORDER};border-radius:6px;
                    margin-top:8px;padding:10px 8px 8px;background:{CARD};}}
                QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 5px;
                    background:{CARD};}}
            ''')
        self.otherGroup3.setVisible(True)  # col3 占位
        # Label group 自然高度
        self.otherGroup.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        # exportGroup 自然高度
        # dataGroup 自然高度
        self.dataGroup.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        # Crawler 组填充剩余空间
        self.otherPlaceholder.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        # 关键字输入框填充组内剩余高度
        self.crawlerKeyword.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

        # ── Export widgets ──
        self.exportWeights.setStyleSheet(f'''
            QLineEdit{{background:{BG};border:1px solid {BORDER};
                border-radius:3px;padding:1px 6px;font-size:10px;color:{TEXT3};}}
        ''')
        self.exportBrowseBtn.setStyleSheet(f'''
            QPushButton{{background:{CARD};border:1px solid {BORDER};
                border-radius:3px;font-size:12px;padding:0;}}
            QPushButton:hover{{background:{BORDER};}}
        ''')
        self.exportFmt.setStyleSheet(f'''
            QComboBox{{border:1px solid {BORDER};border-radius:4px;
                padding:2px 6px;background:{CARD};font-size:10px;color:{TEXT};}}
            QComboBox:focus{{border-color:{PRI};}}
            QComboBox::drop-down{{border:none;width:16px;}}
        ''')
        self.exportSz.setStyleSheet(f'''
            QComboBox{{border:1px solid {BORDER};border-radius:4px;
                padding:2px 6px;background:{CARD};font-size:10px;color:{TEXT};}}
            QComboBox:focus{{border-color:{PRI};}}
            QComboBox::drop-down{{border:none;width:16px;}}
        ''')
        self.exportBtn.setStyleSheet(f'''
            QPushButton{{background:{PRI};color:#fff;border:none;
                padding:4px 0;font-size:11px;font-weight:600;border-radius:4px;}}
            QPushButton:hover{{background:{PRI_H};}}
            QPushButton:disabled{{background:#a5d6a5;}}
        ''')
        self.exportStatus.setStyleSheet(
            f'font-size:9px;color:{TEXT3};padding:0;margin:0;')

        # ── 模型分析组件（Column 2）──
        self.col2Placeholder.setTitle('🔎 Model Analysis')
        lo = self.col2Placeholder.layout()

        # 模型路径行：Model 标签 + 路径框 + 📁 按钮
        row1 = QWidget()
        r1 = QHBoxLayout(row1); r1.setContentsMargins(0,0,0,0); r1.setSpacing(4)
        r1.addWidget(QLabel('Model'))
        self.analyze_model_path = QLineEdit()
        self.analyze_model_path.setPlaceholderText('Auto (latest best.pt)')
        self.analyze_model_path.setStyleSheet(
            f'font-size:10px;padding:2px 6px;border:1px solid {BORDER};'
            f'border-radius:3px;background:{BG};color:{TEXT};')
        r1.addWidget(self.analyze_model_path)
        self.analyze_model_btn = QPushButton('📁')
        self.analyze_model_btn.setFixedSize(24,22)
        self.analyze_model_btn.setStyleSheet(f'''
            QPushButton{{background:{CARD};border:1px solid {BORDER};
                border-radius:3px;font-size:12px;padding:0;}}
            QPushButton:hover{{background:{BORDER};}}
        ''')
        self.analyze_model_btn.clicked.connect(self._analyze_browse_model)
        r1.addWidget(self.analyze_model_btn)
        lo.addWidget(row1)

        # 参数行：Split + Conf + spinbox + ▶ Analyze
        btn_row = QWidget()
        br = QHBoxLayout(btn_row); br.setContentsMargins(0,0,0,0); br.setSpacing(6)
        br.addWidget(QLabel('Split'))
        self.analyze_split = QComboBox()
        self.analyze_split.addItems(['val', 'test'])
        self.analyze_split.setCurrentText('test')
        self.analyze_split.setStyleSheet(
            f'font-size:10px;padding:2px 4px;border:1px solid {BORDER};'
            f'border-radius:3px;background:{CARD};color:{TEXT};')
        br.addWidget(self.analyze_split)
        br.addWidget(QLabel('Conf'))
        self.analyze_conf = QDoubleSpinBox()
        self.analyze_conf.setRange(0.0, 0.99); self.analyze_conf.setDecimals(2)
        self.analyze_conf.setSingleStep(0.05); self.analyze_conf.setValue(0.0)
        self.analyze_conf.setStyleSheet(
            f'font-size:10px;padding:2px 4px;border:1px solid {BORDER};'
            f'border-radius:3px;background:{CARD};color:{TEXT};')
        br.addWidget(self.analyze_conf)
        self.analyze_btn = QPushButton('▶ Analyze')
        self.analyze_btn.setStyleSheet(
            f'background:{PRI};color:#fff;border:none;padding:6px 12px;'
            f'font-size:12px;font-weight:600;border-radius:4px;')
        self.analyze_btn.clicked.connect(self._run_analyze)
        br.addWidget(self.analyze_btn)
        lo.addWidget(btn_row)

        # 进度条
        self.analyze_progress = QProgressBar()
        self.analyze_progress.setVisible(True)
        self.analyze_progress.setTextVisible(True)
        self.analyze_progress.setStyleSheet(f'''
            QProgressBar{{border:none;border-radius:2px;height:14px;
                background:{BORDER};text-align:center;font-size:8px;color:{TEXT};}}
            QProgressBar::chunk{{background:{PRI};border-radius:2px;}}
        ''')
        lo.addWidget(self.analyze_progress)

        # 状态
        self.analyze_status = QLabel('')
        self.analyze_status.setStyleSheet(f'font-size:9px;color:{TEXT3};padding:0;')
        self.analyze_status.setWordWrap(True)
        lo.addWidget(self.analyze_status)
        lo.addStretch()

        # Col 2 组件自然高度
        self.col2Placeholder.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.col2Placeholder.setStyleSheet(f'''
            QGroupBox{{font-weight:600;font-size:10px;color:{TEXT2};
                border:1px solid {BORDER};border-radius:6px;
                margin-top:8px;padding:10px 8px 8px;background:{CARD};}}
            QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 5px;
                background:{CARD};}}
        ''')
        self.col2Other.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.col2Other.setStyleSheet(f'''
            QGroupBox{{font-weight:600;font-size:10px;color:{TEXT2};
                border:1px solid {BORDER};border-radius:6px;
                margin-top:8px;padding:10px 8px 8px;background:{CARD};}}
            QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 5px;
                background:{CARD};}}
        ''')

    # ═══════════════════════════════════════════
    # Signals
    # ═══════════════════════════════════════════

    def _connect_signals(self):
        self.importBtn.clicked.connect(self._start_import)
        self.labelExportBtn.clicked.connect(self._export_label_zip)
        self.labelImportBtn.clicked.connect(self._import_label_zip)
        self.crawlerBrowseBtn.clicked.connect(self._crawler_browse)
        self.crawlerStartBtn.clicked.connect(self._crawler_start)
        self.exportBrowseBtn.clicked.connect(self._export_browse)
        self.exportBtn.clicked.connect(self._run_export)

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

        # ── Export 初始化 ──
        e = cfg['export']
        self.exportFmt.clear()
        self.exportFmt.addItems(e['format_options'])
        self.exportFmt.setCurrentText(e['format'])
        self.exportSz.addItems([str(v) for v in e['imgsz_options']])
        self.exportSz.setCurrentText(str(e['imgsz']))
        self.exportHalf.setChecked(e['half'])
        self.exportInt8.setChecked(e['int8'])
        self.exportNms.setChecked(e['nms'])
        from PyQt5.QtWidgets import QListView
        for cb in (self.exportFmt, self.exportSz):
            lv = QListView()
            lv.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            cb.setView(lv)
            cb.setMaxVisibleItems(10)
        self._export_load_latest()

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
        # 清理旧线程/工作器，防止快速重复点击
        if _THREAD is not None:
            _THREAD.quit()
            _THREAD.wait(3000)
            _THREAD = None
        if _WORKER is not None:
            _WORKER.deleteLater()
            _WORKER = None

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
        global _WORKER, _THREAD
        _WORKER = None
        _THREAD = None
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
                    if f.is_file() or f.is_symlink():
                        try:
                            arcname = str(f.relative_to(after))
                        except ValueError:
                            continue  # 跳过指向外部的符号链接
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
        # 清理旧线程
        if _CRAWLER_THREAD is not None:
            _CRAWLER_THREAD.quit()
            _CRAWLER_THREAD.wait(3000)
            _CRAWLER_THREAD = None
        if _CRAWLER_WORKER is not None:
            _CRAWLER_WORKER.deleteLater()
            _CRAWLER_WORKER = None

        _CRAWLER_WORKER = _CrawlerWorker(keyword, dst)
        _CRAWLER_THREAD = QThread(self)
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
        global _CRAWLER_THREAD, _CRAWLER_WORKER
        if _CRAWLER_THREAD:
            _CRAWLER_THREAD.quit()
            _CRAWLER_THREAD.wait(3000)
            _CRAWLER_THREAD = None
        if _CRAWLER_WORKER:
            _CRAWLER_WORKER.deleteLater()
            _CRAWLER_WORKER = None

    def _crawler_error(self, msg):
        self.crawlerStartBtn.setEnabled(True)
        self.crawlerStartBtn.setText('▶ Start')
        self.crawlerProgress.setValue(0)
        self.crawlerProgress.setFormat('Error')
        global _CRAWLER_THREAD, _CRAWLER_WORKER
        if _CRAWLER_THREAD:
            _CRAWLER_THREAD.quit()
            _CRAWLER_THREAD.wait(3000)
            _CRAWLER_THREAD = None
        if _CRAWLER_WORKER:
            _CRAWLER_WORKER.deleteLater()
            _CRAWLER_WORKER = None

    # ═══════════════════════════════════════════
    # Export Logic
    # ═══════════════════════════════════════════

    def _export_load_latest(self):
        from main.core.base import find_latest_best
        w = find_latest_best()
        if w:
            self.exportWeights.setText(w)

    def _export_browse(self):
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        p, _ = QFileDialog.getOpenFileName(self, 'Select Weights', str(ROOT / 'models'), MODEL_FILTER, options=opts)
        if p:
            self.exportWeights.setText(p)

    def _run_export(self):
        w = self.exportWeights.text().strip()
        if not w or not Path(w).exists():
            from main.core.base import find_latest_best
            w2 = find_latest_best()
            if w2:
                self.exportWeights.setText(w2); w = w2
            else:
                self._set_export_status(RED, 'No weights found')
                return
        fmt = self.exportFmt.currentText()
        self._set_export_status(AMBER, f'Exporting to {fmt.upper()}...')
        self.exportBtn.setEnabled(False)
        QApplication.processEvents()
        try:
            from ultralytics import YOLO
            model = YOLO(w)
            kw = dict(
                format=fmt,
                imgsz=int(self.exportSz.currentText()),
                half=self.exportHalf.isChecked(),
                int8=self.exportInt8.isChecked(),
                nms=self.exportNms.isChecked(),
                device='0' if self.studio.gpu_ok else 'cpu',
            )
            if fmt == 'onnx':
                kw['opset'] = 12
                kw['simplify'] = True
                kw['dynamic'] = False

            out = model.export(**kw)
            sz = Path(out).stat().st_size / 1e6
            self._set_export_status(GREEN, f'{fmt.upper()} → {Path(out).name} ({sz:.1f}MB)')
        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_export_status(RED, f'Failed: {e}')
        finally:
            self.exportBtn.setEnabled(True)

    def _set_export_status(self, color, msg):
        self.exportStatus.setStyleSheet(f'font-size:9px;color:{color};padding:0;margin:0;')
        self.exportStatus.setText(msg)

    # ═══════════════════════════════════════════
    # 模型分析
    # ═══════════════════════════════════════════

    def _analyze_browse_model(self):
        p, _ = QFileDialog.getOpenFileName(
            self, 'Select Model', 'runs', 'PyTorch (*.pt)')
        if p:
            self.analyze_model_path.setText(p)

    def _on_analyze_progress(self, current, total):
        """主线程更新进度条和状态"""
        self.analyze_progress.setMaximum(max(total, 1))
        self.analyze_progress.setValue(current)
        if total > 0:
            self.analyze_status.setText(f'{current}/{total}')

    def _run_analyze(self):
        """执行模型分析（后台线程 + 实时进度）"""
        import subprocess, sys, threading

        model_path = self.analyze_model_path.text().strip()
        split = self.analyze_split.currentText()
        conf = self.analyze_conf.value()
        export_dir = load_paths().get('export_dir', '') or ''

        self.analyze_btn.setEnabled(False)
        self.analyze_status.setStyleSheet(f'font-size:9px;color:{AMBER};padding:0;')
        self.analyze_status.setText('正在加载')
        self.analyze_progress.setVisible(True)
        self.analyze_progress.setValue(0)
        self.analyze_progress.setMaximum(0)

        def worker():
            import time as _t
            try:
                cmd = [sys.executable, 'main/core/tools/analyze.py', '--source', split]
                if export_dir:
                    cmd += ['--output', export_dir]
                if model_path:
                    cmd += ['--model', model_path]
                if conf > 0:
                    cmd += ['--conf', str(conf)]
                print(f'[Analyze] 启动子进程: {" ".join(cmd)}', flush=True)

                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                t0 = _t.time()
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, encoding='utf-8', errors='replace', env=env)
                output_lines = []
                total = 0
                for line in iter(proc.stdout.readline, ''):
                    if not line:
                        break
                    line = line.rstrip('\n\r')
                    print(f'  {line}', flush=True)  # 回显到控制台
                    output_lines.append(line)
                    # 解析进度
                    if line.startswith('[PROGRESS] '):
                        msg = line[11:]
                        if '/' in msg and msg.split('/')[0].strip().isdigit():
                            parts = msg.split('/')
                            cur, total = int(parts[0]), int(parts[1])
                            self._analyze_signal.progress_update.emit(cur, total)
                        else:
                            # 文字消息（推理开始/完成）— 设置总数
                            if '/' in msg:
                                parts = msg.split('/')
                                try:
                                    total = int(parts[-1])
                                    self._analyze_signal.progress_update.emit(0, total)
                                except: pass
                proc.wait(timeout=600)
                elapsed = _t.time() - t0
                print(f'[Analyze] 子进程完成, 耗时 {elapsed:.1f}s', flush=True)
                output = '\n'.join(output_lines)

                lines = output.strip().split('\n')
                summary = '\n'.join(l for l in lines if any(
                    kw in l for kw in ['TP:', 'FP:', 'FN:', 'Precision:', 'Recall:',
                                       'F1:', '最佳', '输出:', '曲线:', '按类别']))
                if not summary:
                    summary = output[-400:] if len(output) > 400 else output

                self._analyze_signal.done.emit(summary, '')

            except subprocess.TimeoutExpired:
                self._analyze_signal.done.emit('', 'Timed out')
            except Exception as e:
                import traceback; traceback.print_exc()
                self._analyze_signal.done.emit('', str(e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _analyze_done(self, summary, error=''):
        """回到主线程更新 UI（分析完成）"""
        self.analyze_btn.setEnabled(True)
        self.analyze_status.setStyleSheet(f'font-size:9px;color:{GREEN};padding:0;')
        if error:
            self.analyze_status.setStyleSheet(f'font-size:9px;color:{RED};padding:0;')
            self.analyze_status.setText('失败')
        else:
            self.analyze_status.setText('完成')

    # ═══════════════════════════════════════════
    # 主题刷新
    # ═══════════════════════════════════════════

    def on_theme_changed(self):
        """Settings 主题切换后刷新样式"""
        self._post_process_ui()
