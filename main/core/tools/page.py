# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""Tools 标签页 — 导入视频到预处理目录 + 标签导入导出"""

import shutil, zipfile, os
from pathlib import Path
from PIL import Image
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, QProcess
from main.core.base import *
from main.config import ROOT, load_paths


_WORKER = None
_THREAD = None

_CRAWLER_WORKER = None
_CRAWLER_THREAD = None

_BILI_DL_WORKER = None
_BILI_DL_THREAD = None


class _BiliExtractWorker(QObject):
    """B站 API 提取 Worker（QThread 内运行）"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        try:
            info = _bilibili_extract_info(self.url)
            if info:
                self.finished.emit(info)
            else:
                self.error.emit('B站 API 提取失败')
        except Exception as e:
            self.error.emit(str(e))


def _bilibili_extract_info(url):
    """B站官方 API 提取视频信息（yt-dlp 失效时 fallback）"""
    import urllib.request, json as _json, re as _re

    print(f'[BiliAPI] Input URL: {url}', flush=True)

    # b23.tv 短链接跳转
    if 'b23.tv' in url:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            url = urllib.request.urlopen(req, timeout=10).geturl()
            print(f'[BiliAPI] Redirected to: {url}', flush=True)
        except Exception as e:
            print(f'[BiliAPI] Redirect failed: {e}', flush=True)
            return None

    bv = _re.search(r'BV[\w]+', url)
    if not bv:
        print(f'[BiliAPI] No BV found in URL', flush=True)
        return None
    bv = bv.group()
    print(f'[BiliAPI] BV={bv}', flush=True)

    # 获取视频信息
    api = f'https://api.bilibili.com/x/web-interface/view?bvid={bv}'
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bilibili.com'}
    try:
        req = urllib.request.Request(api, headers=headers)
        resp = urllib.request.urlopen(req, timeout=15)
        data = _json.loads(resp.read())
        print(f'[BiliAPI] API response code={data.get("code")}', flush=True)
    except Exception as e:
        print(f'[BiliAPI] API request failed: {e}', flush=True)
        return None

    if data.get('code') != 0:
        print(f'[BiliAPI] API error: {data.get("message")}', flush=True)
        return None

    v = data['data']
    print(f'[BiliAPI] OK: title={v.get("title","?")}', flush=True)
    return {
        'title': v.get('title', ''),
        'duration': v.get('duration', 0),
        'uploader': v.get('owner', {}).get('name', ''),
        'extractor_key': 'BiliBili',
        'webpage_url': url,
        'formats': [],
        '_bvid': bv,
        '_cid': v.get('cid', 0),
    }


class _BiliDownloadWorker(QObject):
    """B站原生下载工作线程（仅下载，不合并）"""
    progress = pyqtSignal(int, int)   # (current, total) bytes
    log = pyqtSignal(str)
    filesReady = pyqtSignal(str, str, str)  # (vpath, apath, fpath)
    error = pyqtSignal(str)

    def __init__(self, info, output_dir, format_spec):
        super().__init__()
        self._info = info
        self._out = Path(output_dir).resolve()
        self._fmt = format_spec
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        import urllib.request, json as _json, ssl, re as _re
        ssl._create_default_https_context = ssl._create_unverified_context

        bv = self._info.get('_bvid')
        cid = self._info.get('_cid')
        if not bv or not cid:
            self.log.emit('Missing BV/CID')
            self.finished.emit(False, '')
            return

        qn_map = {'best': 116, '4k': 120, '1080p': 80, '720p': 64, '480p': 32}
        qn = qn_map.get(self._fmt, 116)

        play_url = f'https://api.bilibili.com/x/player/playurl?bvid={bv}&cid={cid}&qn={qn}&fnval=4048'
        headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bilibili.com'}
        try:
            req = urllib.request.Request(play_url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=15)
            data = _json.loads(resp.read())
        except Exception as e:
            self.log.emit(f'获取播放地址失败: {e}')
            self.finished.emit(False, '')
            return

        if data.get('code') != 0:
            self.log.emit(f'播放地址错误: {data.get("message")}')
            self.finished.emit(False, '')
            return

        d = data['data']
        title = _re.sub(r'[\\/:*?"<>|]', '_', self._info.get('title', bv)).strip()
        video_url = None

        dash = d.get('dash')
        if dash:
            videos = sorted(dash.get('video', []), key=lambda x: x.get('height', 0), reverse=True)

            if self._fmt != 'best':
                try:
                    target = int(self._fmt.rstrip('p'))
                    filtered = [v for v in videos if v.get('height', 0) <= target]
                    if filtered:
                        videos = [max(filtered, key=lambda x: x.get('height', 0))]
                except ValueError:
                    pass

            if videos:
                video_url = videos[0].get('baseUrl') or videos[0].get('base_url', '')
        else:
            durl = d.get('durl', [])
            if durl:
                video_url = durl[0].get('url', '')

        self._out.mkdir(parents=True, exist_ok=True)
        fpath = self._out / f'{title}.mp4'

        dl_headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bilibili.com'}

        def _dl(url, path):
            try:
                req2 = urllib.request.Request(url, headers=dl_headers)
                with urllib.request.urlopen(req2, timeout=300) as src:
                    total = int(src.headers.get('Content-Length', 0))
                    dl = 0
                    with open(path, 'wb') as f:
                        while True:
                            if self._stopped:
                                return False
                            chunk = src.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)
                            dl += len(chunk)
                            if total:
                                self.progress.emit(dl, total)
                return True
            except Exception as e:
                self.log.emit(f'下载失败: {e}')
                return False

        if video_url:
            self.log.emit('下载视频...')
            if _dl(video_url, fpath):
                self.filesReady.emit('', '', str(fpath))
            else:
                self.error.emit('Download failed')
        else:
            self.error.emit('未找到可下载的视频流')


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
    """百度图片爬虫工作线程（支持多关键词，换行分隔）"""
    progress = pyqtSignal(int, int)         # (downloaded, total)
    keyword_progress = pyqtSignal(str, int, int)  # (keyword, index, total_keywords)
    finished = pyqtSignal(int)               # total downloaded
    error = pyqtSignal(str)

    def __init__(self, keywords, base_dir, target=100):
        super().__init__()
        self.keywords = keywords
        self.base_dir = Path(base_dir)
        self.target = target
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        import json, os, time, re, random
        import requests

        # 初始化 session：先访问首页拿到 cookie，再补充 Referer
        sess = requests.Session()
        sess.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36',
        })
        try:
            sess.get('https://image.baidu.com/', timeout=10)
        except Exception:
            pass
        sess.headers.update({'Referer': 'https://image.baidu.com/'})

        total_downloaded = 0
        total_keywords = len(self.keywords)
        per_kw = max(1, self.target // total_keywords)

        for idx, keyword in enumerate(self.keywords):
            if self._stopped:
                break
            kw_dir = self.base_dir / keyword
            kw_dir.mkdir(parents=True, exist_ok=True)
            self.keyword_progress.emit(keyword, idx + 1, total_keywords)

            downloaded = 0
            pn = 0

            while downloaded < per_kw and not self._stopped:
                params = {
                    'tn': 'resultjson_com', 'ct': '201326592',
                    'queryWord': keyword, 'cl': 2, 'lm': -1,
                    'ie': 'utf-8', 'oe': 'utf-8',
                    'word': keyword, 'pn': pn, 'rn': 30,
                    'gsm': hex(pn)[2:] or '1e',
                }
                try:
                    r = sess.get('https://image.baidu.com/search/acjson',
                                 params=params, timeout=(5, 15))
                    text = re.sub(r'[\x00-\x1f\x7f]', '', r.text)
                    data = json.loads(text).get('data', [])
                except Exception:
                    pn += 30
                    time.sleep(3)
                    continue

                if not data:
                    break

                for img in data:
                    if self._stopped:
                        break
                    if not img:
                        continue
                    img_url = (img.get('thumbURL') or img.get('middleURL') or '').strip()
                    if not img_url.startswith('http'):
                        continue
                    if any(x in img_url.lower() for x in ['.svg', 'logo', 'icon']):
                        continue

                    ext = os.path.splitext(img_url.split('?')[0])[1].lower()
                    if ext in ('.gif',):
                        continue
                    if not ext:
                        ext = '.jpg'

                    fname = f'{keyword}_{downloaded + 1}{ext}'
                    fpath = kw_dir / fname
                    if fpath.exists():
                        continue

                    try:
                        r_img = sess.get(img_url, timeout=10)
                        r_img.raise_for_status()
                        with open(fpath, 'wb') as f:
                            f.write(r_img.content)
                        # 图片格式修正：百度常返回 WEBP/GIF 但后缀 .jpg
                        try:
                            _img = Image.open(fpath)
                            _fmt_map = {'.jpg': 'JPEG', '.jpeg': 'JPEG',
                                        '.png': 'PNG', '.bmp': 'BMP', '.webp': 'WEBP'}
                            _expect = _fmt_map.get(ext, '')
                            if _expect and _img.format != _expect:
                                if _img.mode in ('P', 'PA', 'RGBA', 'LA'):
                                    _img = _img.convert('RGB')
                                _img.save(fpath, 'JPEG', quality=95)
                        except Exception:
                            pass  # 格式校验失败则保留原始文件
                        downloaded += 1
                        total_downloaded += 1
                        self.progress.emit(downloaded, per_kw)
                        if downloaded >= per_kw:
                            break
                    except Exception:
                        continue
                    time.sleep(random.uniform(0.3, 0.8))

                pn += 30
                time.sleep(random.uniform(1.0, 2.5))

        sess.close()
        self.finished.emit(total_downloaded)


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
        #   col1Lo, dataGroup, importLabel, importPath, importBtn, dataStatus,
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
        self.crawlerCountLabel.setStyleSheet(
            f'font-size:10px;font-weight:600;color:{TEXT3};padding:0;margin:0;')
        self.crawlerCount.setStyleSheet(f'''
            QSpinBox{{border:1px solid {BORDER};border-radius:4px;
                padding:2px 6px;background:{CARD};font-size:10px;color:{TEXT};}}
            QSpinBox:focus{{border-color:{PRI};}}
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

        # ── Resume Training ──
        self.col2Placeholder.setTitle('🚀 Resume Training')
        self.col2Placeholder.setVisible(True)
        lo = self.col2Placeholder.layout()
        lo.setSpacing(5)

        # 运行目录行
        rd_row = QWidget()
        rdl = QHBoxLayout(rd_row); rdl.setContentsMargins(0, 0, 0, 0); rdl.setSpacing(4)
        rdl.addWidget(QLabel('Run dir'))
        self.resume_run_dir = QLineEdit()
        self.resume_run_dir.setPlaceholderText('Auto (latest runs/...)')
        self.resume_run_dir.setStyleSheet(
            f'font-size:10px;padding:2px 6px;border:1px solid {BORDER};'
            f'border-radius:3px;background:{BG};color:{TEXT};')
        rdl.addWidget(self.resume_run_dir)
        self.resume_browse_btn = QPushButton('📁')
        self.resume_browse_btn.setFixedSize(24, 22)
        self.resume_browse_btn.setStyleSheet(f'''
            QPushButton{{background:{CARD};border:1px solid {BORDER};
                border-radius:3px;font-size:12px;padding:0;}}
            QPushButton:hover{{background:{BORDER};}}
        ''')
        self.resume_browse_btn.clicked.connect(self._resume_browse)
        rdl.addWidget(self.resume_browse_btn)
        lo.addWidget(rd_row)

        # 覆盖参数行: Batch + Wkr + Dev + Epochs + LR 统一一行
        params_w = QWidget()
        pl = QHBoxLayout(params_w); pl.setContentsMargins(0, 0, 0, 0); pl.setSpacing(3)
        pl.addWidget(QLabel('B'))
        self.resume_batch = QSpinBox()
        self.resume_batch.setRange(0, 256); self.resume_batch.setSpecialValueText('—')
        self.resume_batch.setValue(0)
        self.resume_batch.setFixedWidth(50)
        self.resume_batch.setStyleSheet(
            f'font-size:10px;padding:2px 4px;border:1px solid {BORDER};'
            f'border-radius:3px;background:{CARD};color:{TEXT};')
        pl.addWidget(self.resume_batch)
        pl.addWidget(QLabel('W'))
        self.resume_workers = QSpinBox()
        self.resume_workers.setRange(0, 32); self.resume_workers.setSpecialValueText('—')
        self.resume_workers.setValue(0)
        self.resume_workers.setFixedWidth(50)
        self.resume_workers.setStyleSheet(
            f'font-size:10px;padding:2px 4px;border:1px solid {BORDER};'
            f'border-radius:3px;background:{CARD};color:{TEXT};')
        pl.addWidget(self.resume_workers)
        pl.addWidget(QLabel('D'))
        self.resume_device = QLineEdit()
        self.resume_device.setPlaceholderText('—')
        self.resume_device.setFixedWidth(32)
        self.resume_device.setStyleSheet(
            f'font-size:10px;padding:2px 4px;border:1px solid {BORDER};'
            f'border-radius:3px;background:{BG};color:{TEXT};')
        pl.addWidget(self.resume_device)
        pl.addWidget(QLabel('Ep'))
        self.resume_epochs = QSpinBox()
        self.resume_epochs.setRange(0, 9999); self.resume_epochs.setSpecialValueText('—')
        self.resume_epochs.setValue(0)
        self.resume_epochs.setFixedWidth(58)
        self.resume_epochs.setStyleSheet(
            f'font-size:10px;padding:2px 4px;border:1px solid {BORDER};'
            f'border-radius:3px;background:{CARD};color:{TEXT};')
        pl.addWidget(self.resume_epochs)
        pl.addWidget(QLabel('LR'))
        self.resume_lr = QDoubleSpinBox()
        self.resume_lr.setRange(0, 1); self.resume_lr.setDecimals(6)
        self.resume_lr.setSingleStep(0.0001); self.resume_lr.setSpecialValueText('—')
        self.resume_lr.setValue(0)
        self.resume_lr.setFixedWidth(64)
        self.resume_lr.setStyleSheet(
            f'font-size:10px;padding:2px 4px;border:1px solid {BORDER};'
            f'border-radius:3px;background:{CARD};color:{TEXT};')
        pl.addWidget(self.resume_lr)
        lo.addWidget(params_w)

        # Cache + Resume 按钮行
        btn_row = QWidget()
        br = QHBoxLayout(btn_row); br.setContentsMargins(0, 0, 0, 0); br.setSpacing(6)
        self.resume_no_cache = QCheckBox('No cache clear')
        self.resume_no_cache.setStyleSheet(
            f'font-size:9px;color:{TEXT2};')
        br.addWidget(self.resume_no_cache)
        br.addStretch()
        self.resume_btn = QPushButton('▶ Resume')
        self.resume_btn.setStyleSheet(
            f'background:{PRI};color:#fff;border:none;padding:5px 12px;'
            f'font-size:11px;font-weight:600;border-radius:4px;')
        self.resume_btn.clicked.connect(self._run_resume_train)
        br.addWidget(self.resume_btn)
        lo.addWidget(btn_row)

        # 日志区
        self.resume_log = QTextEdit()
        self.resume_log.setReadOnly(True)
        self.resume_log.setPlaceholderText('Output will appear here...')
        self.resume_log.setMinimumHeight(60)
        self.resume_log.setStyleSheet(f'''
            QTextEdit{{background:{BG};border:1px solid {BORDER};
                border-radius:4px;padding:4px 6px;font-size:10px;
                color:{TEXT2};font-family:Consolas,"Courier New",monospace;}}
        ''')
        self.resume_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lo.addWidget(self.resume_log, 1)

        self.col2Placeholder.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.col2Other.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.col2Other.setStyleSheet(f'''
            QGroupBox{{font-weight:600;font-size:10px;color:{TEXT2};
                border:1px solid {BORDER};border-radius:6px;
                margin-top:8px;padding:10px 8px 8px;background:{CARD};}}
            QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 5px;
                background:{CARD};}}
        ''')
        self._build_video_extractor_ui()

        self.col2Other2.setVisible(False)

        # ── 交换 Crawler ↔ Resume Training 位置 ──
        i1 = self.col1Lo.indexOf(self.otherPlaceholder)
        i2 = self.col2Lo.indexOf(self.col2Placeholder)
        if i1 >= 0 and i2 >= 0:
            self.col1Lo.insertWidget(i1, self.col2Placeholder)
            self.col2Lo.insertWidget(i2, self.otherPlaceholder)
        self.otherPlaceholder.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.col2Placeholder.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

    def _build_video_extractor_ui(self):
        """视频提取组件 UI — 添加到 col2Other"""
        lo = self.col2Other.layout()
        lo.setSpacing(5)

        # ── URL 输入行 ──
        url_row = QWidget()
        url_l = QHBoxLayout(url_row); url_l.setContentsMargins(0, 0, 0, 0); url_l.setSpacing(6)
        self.video_url_input = QLineEdit()
        self.video_url_input.setPlaceholderText('Paste video URL (Bilibili/YouTube/Douyin…)')
        self.video_url_input.setStyleSheet(f'''
            QLineEdit{{background:{BG};border:1px solid {BORDER};
                border-radius:4px;padding:4px 8px;font-size:11px;color:{TEXT};}}
            QLineEdit:focus{{border-color:{PRI};}}
        ''')
        url_l.addWidget(self.video_url_input, 1)
        self.btn_video_clear = QPushButton('✕')
        self.btn_video_clear.setFixedSize(24, 24)
        self.btn_video_clear.setToolTip('Clear URL')
        self.btn_video_clear.setStyleSheet(f'''
            QPushButton{{background:{CARD};border:1px solid {BORDER};
                border-radius:3px;font-size:10px;color:{TEXT3};padding:0;}}
            QPushButton:hover{{background:{BORDER};color:{TEXT};}}
        ''')
        url_l.addWidget(self.btn_video_clear)
        self.btn_video_extract = QPushButton('Extract')
        self.btn_video_extract.setFixedWidth(62)
        self.btn_video_extract.setStyleSheet(f'''
            QPushButton{{background:{PRI};color:#fff;border:none;
                padding:5px 0;font-size:11px;font-weight:600;border-radius:4px;}}
            QPushButton:hover{{background:{PRI_H};}}
            QPushButton:disabled{{background:#a5d6a5;}}
        ''')
        url_l.addWidget(self.btn_video_extract)
        lo.addWidget(url_row)

        # ── 视频信息 ──
        self.video_info_label = QLabel('')
        self.video_info_label.setWordWrap(True)
        self.video_info_label.setStyleSheet(f'''
            font-size:10px;color:{TEXT};line-height:1.4;
            padding:4px 6px;background:{BG};border-radius:4px;
            border:1px solid {BORDER};
        ''')
        self.video_info_label.setMinimumHeight(36)
        self.video_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lo.addWidget(self.video_info_label)

        # ── 设置行: 画质 + 目录 ──
        settings_row = QWidget()
        sr = QHBoxLayout(settings_row); sr.setContentsMargins(0, 0, 0, 0); sr.setSpacing(5)
        q_label = QLabel('Quality')
        q_label.setStyleSheet(f'font-size:10px;color:{TEXT3};padding:0;')
        q_label.setFixedWidth(38)
        sr.addWidget(q_label)
        self.video_quality_combo = QComboBox()
        self.video_quality_combo.addItems(['Best', '1080p', '720p', '480p', '360p'])
        self.video_quality_combo.setStyleSheet(f'''
            QComboBox{{border:1px solid {BORDER};border-radius:3px;
                padding:2px 5px;background:{CARD};font-size:10px;color:{TEXT};
                min-width:56px;}}
            QComboBox:focus{{border-color:{PRI};}}
        ''')

        sr.addWidget(self.video_quality_combo)
        d_label = QLabel('Dir')
        d_label.setStyleSheet(f'font-size:10px;color:{TEXT3};padding:0;')
        d_label.setFixedWidth(20)
        sr.addWidget(d_label)
        self.video_output_dir = QLineEdit('output')
        self.video_output_dir.setStyleSheet(f'''
            QLineEdit{{background:{BG};border:1px solid {BORDER};
                border-radius:3px;padding:2px 4px;font-size:10px;color:{TEXT};}}
            QLineEdit:focus{{border-color:{PRI};}}
        ''')
        sr.addWidget(self.video_output_dir, 1)
        self.btn_video_browse = QPushButton('📁')
        self.btn_video_browse.setStyleSheet(f'''
            QPushButton{{background:{CARD};border:1px solid {BORDER};
                border-radius:3px;font-size:12px;padding:0;}}
            QPushButton:hover{{background:{BORDER};}}
        ''')
        sr.addWidget(self.btn_video_browse)
        lo.addWidget(settings_row)

        # ── 下载按钮 ──
        self.btn_video_download = QPushButton('Download')
        self.btn_video_download.setEnabled(False)
        self.btn_video_download.setStyleSheet(f'''
            QPushButton{{background:{PRI};color:#fff;border:none;
                padding:5px 0;font-size:12px;font-weight:600;border-radius:4px;}}
            QPushButton:hover{{background:{PRI_H};}}
            QPushButton:disabled{{background:#a5d6a5;}}
        ''')
        lo.addWidget(self.btn_video_download)

        # ── 进度条 ──
        self.video_progress = QProgressBar()
        self.video_progress.setTextVisible(True)
        self.video_progress.setFixedHeight(16)
        self.video_progress.setStyleSheet(f'''
            QProgressBar{{border:1px solid {BORDER};border-radius:3px;
                background:{BG};text-align:center;font-size:9px;color:{TEXT3};}}
            QProgressBar::chunk{{background:{PRI};border-radius:2px;}}
        ''')
        lo.addWidget(self.video_progress)

        # ── 状态文本 ──
        self.video_status = QLabel('')
        self.video_status.setWordWrap(True)
        self.video_status.setStyleSheet(f'font-size:9px;color:{TEXT3};padding:0 2px;')
        lo.addWidget(self.video_status)

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
        self.btn_video_extract.clicked.connect(self._video_extract_start)
        self.btn_video_clear.clicked.connect(lambda: self.video_url_input.clear())
        self.btn_video_browse.clicked.connect(self._video_browse)
        self.btn_video_download.clicked.connect(self._video_download_start)
        self.video_url_input.returnPressed.connect(self._video_extract_start)

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

        # ── 视频下载默认目录 = export_dir ──
        export_dir = paths.get('export_dir', '')
        if export_dir:
            self.video_output_dir.setText(export_dir)

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
        self.studio.log_operation('Tools', f'导入视频到预处理目录 · {Path(src).name}')

    def _on_import_progress(self, pct, filename):
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
            self._set_label_status(GREEN, f'Exported → {Path(save_path).name}')
            self.studio.log_operation('Tools', f'导出标注 · {folder_name}.zip')
        except Exception as e:
            self._set_label_status(RED, f'Export failed: {e}')
            self.studio.log_operation('Tools', f'导出标注失败 · {e}')
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
            self.studio.log_operation('Tools', f'导入标注 · {Path(archive_path).name}')
        except Exception as e:
            msg = str(e)
            if 'Cannot find working tool' in msg:
                msg = 'Need 7-Zip or WinRAR installed to extract this format'
            self._set_label_status(RED, f'Import failed: {msg}')
            self.studio.log_operation('Tools', f'导入标注失败 · {msg}')
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
        raw = self.crawlerKeyword.toPlainText().strip()
        if not raw:
            self.crawlerInfo.setText('Please enter keywords (one per line)')
            return
        dst = self.crawlerPath.text().strip()
        if not dst:
            self.crawlerInfo.setText('Please select a download directory')
            return

        # 按换行拆分多关键词，过滤空行
        keywords = [k.strip() for k in raw.split('\n') if k.strip()]
        target = self.crawlerCount.value()

        self.crawlerStartBtn.setEnabled(False)
        self.crawlerStartBtn.setText('Running...')
        self.crawlerProgress.setValue(0)
        self.crawlerInfo.setText(f'{len(keywords)} keyword(s), {target} total (~{max(1, target // len(keywords))}/each)...')
        QApplication.processEvents()

        global _CRAWLER_WORKER, _CRAWLER_THREAD
        if _CRAWLER_THREAD is not None:
            _CRAWLER_THREAD.quit()
            _CRAWLER_THREAD.wait(3000)
            _CRAWLER_THREAD = None
        if _CRAWLER_WORKER is not None:
            _CRAWLER_WORKER.deleteLater()
            _CRAWLER_WORKER = None

        _CRAWLER_WORKER = _CrawlerWorker(keywords, dst, target)
        _CRAWLER_THREAD = QThread(self)
        _CRAWLER_WORKER.moveToThread(_CRAWLER_THREAD)
        _CRAWLER_THREAD.started.connect(_CRAWLER_WORKER.run)
        _CRAWLER_WORKER.progress.connect(self._crawler_progress)
        _CRAWLER_WORKER.keyword_progress.connect(self._on_crawler_keyword)
        _CRAWLER_WORKER.finished.connect(self._crawler_done)
        _CRAWLER_WORKER.error.connect(self._crawler_error)
        _CRAWLER_THREAD.start()
        self.studio.log_operation('Tools', f'图片爬虫开始 · {len(keywords)} 关键词, 目标 {target} 张')

    def _on_crawler_keyword(self, keyword, idx, total):
        self.crawlerInfo.setText(f'[{idx}/{total}] {keyword}')

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
        self.studio.log_operation('Tools', f'图片爬虫完成 · {count} 张')
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
        self.studio.log_operation('Tools', f'图片爬虫失败 · {msg}')
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
            self.studio.log_operation('Tools', f'模型导出 · {fmt.upper()} → {Path(out).name} ({sz:.1f}MB)')
        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_export_status(RED, f'Failed: {e}')
            self.studio.log_operation('Tools', f'模型导出失败 · {e}')
        finally:
            self.exportBtn.setEnabled(True)

    def _set_export_status(self, color, msg):
        self.exportStatus.setStyleSheet(f'font-size:9px;color:{color};padding:0;margin:0;')
        self.exportStatus.setText(msg)

    # ═══════════════════════════════════════════
    # Resume Training
    # ═══════════════════════════════════════════

    def _resume_browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Select Run Directory',
            str(ROOT / 'runs'), options=QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        if d:
            self.resume_run_dir.setText(d)

    def _run_resume_train(self):
        """执行抢救训练：以 QProcess 启动 scripts/resume_train.py"""
        import sys

        run_dir = self.resume_run_dir.text().strip()
        cmd = [sys.executable, str(ROOT / 'scripts' / 'resume_train.py')]
        if run_dir:
            cmd.append(run_dir)

        batch = self.resume_batch.value()
        workers = self.resume_workers.value()
        device = self.resume_device.text().strip()
        epochs = self.resume_epochs.value()
        lr = self.resume_lr.value()

        if batch > 0: cmd += ['--batch', str(batch)]
        if workers > 0: cmd += ['--workers', str(workers)]
        if device: cmd += ['--device', device]
        if epochs > 0: cmd += ['--epochs', str(epochs)]
        if lr > 0: cmd += ['--lr0', str(lr)]
        if self.resume_no_cache.isChecked(): cmd.append('--no-cache-clear')

        self.resume_btn.setEnabled(False)
        self.resume_log.clear()
        self.resume_log.append('▶ Starting resume training...')
        cmd_str = ' '.join(cmd)
        self.studio.log_operation('Tools', f'抢救训练启动 · {cmd_str}')

        self._resume_proc = QProcess(self)
        self._resume_proc.setWorkingDirectory(str(ROOT))
        self._resume_proc.setProcessChannelMode(QProcess.MergedChannels)
        self._resume_proc.readyReadStandardOutput.connect(self._on_resume_output)
        self._resume_proc.finished.connect(self._on_resume_finished)
        self._resume_proc.start(cmd[0], cmd[1:])

    def _on_resume_output(self):
        data = self._resume_proc.readAllStandardOutput().data().decode('utf-8', errors='replace')
        if data.strip():
            self.resume_log.append(data.rstrip())

    def _on_resume_finished(self, exit_code, status):
        self.resume_btn.setEnabled(True)
        if exit_code == 0:
            self.resume_log.append('✓ Done')
            self.studio.log_operation('Tools', '抢救训练完成 ✓')
        else:
            self.resume_log.append(f'✗ Failed (exit={exit_code})')
            self.studio.log_operation('Tools', f'抢救训练失败 (exit={exit_code})')

    # ═══════════════════════════════════════════
    # 视频提取 (yt-dlp)
    # ═══════════════════════════════════════════

    def _video_browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Select output directory',
            options=QFileDialog.Options() | QFileDialog.DontUseNativeDialog)
        if d:
            self.video_output_dir.setText(d)

    def _video_extract_start(self):
        raw = self.video_url_input.text().strip()
        if not raw:
            self._set_video_status(RED, 'Please enter a URL')
            return

        # 从混合文本提取 URL（兼容粘贴整行含中文标题的分享文本）
        import re
        m = re.search(r'https?://[^\s\u4e00-\u9fff<>\"\'—]+', raw)
        url = m.group() if m else raw
        self._extract_url = url

        self.btn_video_extract.setEnabled(False)
        self.btn_video_download.setEnabled(False)
        self.video_info_label.setText('')
        self.video_progress.setRange(0, 0)
        self.video_progress.setValue(0)
        self._set_video_status(AMBER, 'Analyzing...')

        # 清理旧 QProcess
        if hasattr(self, '_extract_proc') and self._extract_proc is not None:
            self._extract_proc.kill()
            self._extract_proc = None

        self._extract_proc = QProcess(self)
        self._extract_proc.finished.connect(self._on_extract_finished)
        self._extract_proc.errorOccurred.connect(self._on_extract_error)
        print(f'[Video] Starting extract: yt-dlp --dump-json --no-download {url}', flush=True)
        self._extract_proc.start('yt-dlp', ['--dump-json', '--no-download', '--no-check-certificates', url])
        self.studio.log_operation('Tools', f'视频信息提取 · {url[:60]}')

    def _on_extract_finished(self, exit_code, exit_status):
        self.btn_video_extract.setEnabled(True)
        proc = self._extract_proc
        print(f'[Video] Extract finished: exit_code={exit_code} exit_status={exit_status}', flush=True)
        if exit_status != QProcess.NormalExit or exit_code != 0:
            err = str(proc.readAllStandardError().data(), 'utf-8', errors='replace').strip()
            print(f'[Video] Extract stderr: {err or "(empty)"}', flush=True)

            # B站 yt-dlp 失败 → 走官方 API fallback（QThread 异步）
            if 'bilibili' in err.lower() or 'b23.tv' in self._extract_url:
                print('[Video] yt-dlp failed, trying B站 API fallback...', flush=True)
                self._bili_extract_worker = _BiliExtractWorker(self._extract_url)
                self._bili_extract_thread = QThread(self)
                self._bili_extract_worker.moveToThread(self._bili_extract_thread)
                self._bili_extract_thread.started.connect(self._bili_extract_worker.run)
                self._bili_extract_worker.finished.connect(self._show_extract_info)
                self._bili_extract_worker.error.connect(lambda e: self._set_video_status(RED, e))
                self._bili_extract_worker.finished.connect(self._bili_extract_thread.quit)
                self._bili_extract_worker.error.connect(self._bili_extract_thread.quit)
                self._bili_extract_thread.start()
                return

            self.video_info_label.setText('')
            self._set_video_status(RED, err or 'Failed to extract video info')
            return

        self._try_parse_extract_output()

    def _try_parse_extract_output(self):
        import json as _json
        proc = self._extract_proc
        try:
            raw = str(proc.readAllStandardOutput().data(), 'utf-8', errors='replace').strip()
            print(f'[Video] Extract stdout ({len(raw)} chars)', flush=True)
            info = _json.loads(raw)
            print(f'[Video] Parsed OK: title={info.get("title","?")}', flush=True)
        except Exception as e:
            print(f'[Video] JSON parse error: {e}', flush=True)
            self.video_info_label.setText('')
            self._set_video_status(RED, 'Failed to parse video info')
            return
        self._show_extract_info(info)

    def _show_extract_info(self, info):
        """填充视频信息到 UI"""
        self._extract_info = info
        title = info.get('title', 'N/A')
        duration = info.get('duration', 0)
        uploader = info.get('uploader') or info.get('channel', 'N/A')
        platform = info.get('extractor_key', 'N/A')

        m, s = divmod(int(duration or 0), 60)
        h, m_div = divmod(m, 60)
        dur_str = f'{h}:{m_div:02d}:{s:02d}' if h else f'{m_div}:{s:02d}'

        self.video_info_label.setText(
            f'Title: {title}\nDuration: {dur_str}  |  Platform: {platform}  |  Uploader: {uploader}'
        )

        formats = info.get('formats') or []
        seen = set()
        heights = set()
        for f in formats:
            h = f.get('height', 0)
            if h and h not in seen:
                seen.add(h)
                heights.add(h)

        self.video_quality_combo.clear()
        self.video_quality_combo.addItem('Best')
        for h in sorted(heights, reverse=True):
            self.video_quality_combo.addItem(f'{h}p')

        self.btn_video_download.setEnabled(True)
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        self.video_progress.setFormat('Ready')
        self._set_video_status(GREEN, 'Ready to download')

    def _on_extract_error(self, err):
        print(f'[Video] Extract error: {err}', flush=True)
        self.btn_video_extract.setEnabled(True)
        self.video_info_label.setText('')
        self._set_video_status(RED, 'yt-dlp not found. Install: pip install yt-dlp' if err == QProcess.FailedToStart else f'Error: {err}')

    def _video_download_start(self):
        raw = self.video_url_input.text().strip()
        output_dir = self.video_output_dir.text().strip() or 'output'
        fmt_raw = self.video_quality_combo.currentText()
        if fmt_raw == 'Best':
            fmt_spec = 'bestvideo+bestaudio/best'
        else:
            fmt_spec = fmt_raw

        # 从混合文本提取 URL
        import re
        m = re.search(r'https?://[^\s\u4e00-\u9fff<>\"\'—]+', raw)
        url = m.group() if m else raw

        self.btn_video_download.setEnabled(False)
        self.btn_video_extract.setEnabled(False)
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        self.video_progress.setFormat('0%')
        self._set_video_status(AMBER, 'Downloading...')

        # B站 → 用原生下载器
        info = getattr(self, '_extract_info', None)
        if info and info.get('_bvid'):
            self._bili_download_start(info, output_dir, fmt_spec)
            return

        out = str(Path(output_dir).resolve())
        # 清理旧 QProcess
        if hasattr(self, '_dl_proc') and self._dl_proc is not None:
            self._dl_proc.kill()
            self._dl_proc = None

        self._dl_proc = QProcess(self)
        self._dl_proc.setWorkingDirectory(out)
        self._dl_proc.setProcessChannelMode(QProcess.MergedChannels)
        self._dl_proc.readyReadStandardOutput.connect(self._on_dl_output)
        self._dl_proc.finished.connect(self._on_dl_finished)
        self._dl_proc.errorOccurred.connect(self._on_dl_error)
        print(f'[Video] Starting download: yt-dlp -f {fmt_spec} -o "%(title)s.%(ext)s" {url}', flush=True)
        self._dl_proc.start('yt-dlp', [
            '--newline', '--progress', '-o', '%(title)s.%(ext)s',
            '-f', fmt_spec, '--merge-output-format', 'mp4',
            '--no-playlist', '--no-check-certificates', url,
        ])
        self.studio.log_operation('Tools', f'视频下载开始 · {fmt_raw} · {url[:40]}...')

    def _on_dl_output(self):
        import re
        data = str(self._dl_proc.readAllStandardOutput().data(), 'utf-8', errors='replace')
        # 解析百分比
        pcts = re.findall(r'([\d.]+)%', data)
        if pcts:
            try:
                pct = float(pcts[-1])
                self.video_progress.setValue(int(pct))
                self.video_progress.setFormat(f'{pct:.1f}%')
            except ValueError:
                pass
        # 显示最后一行
        lines = [l.strip() for l in data.splitlines() if l.strip()]
        if lines:
            self._set_video_status(TEXT3, lines[-1][:120])

    def _on_dl_finished(self, exit_code, exit_status):
        print(f'[Video] Download finished: exit_code={exit_code} exit_status={exit_status}', flush=True)
        self.btn_video_download.setEnabled(True)
        self.btn_video_extract.setEnabled(True)
        ok = exit_status == QProcess.NormalExit and exit_code == 0
        self.video_progress.setValue(100 if ok else 0)
        self.video_progress.setFormat('100%' if ok else 'Failed')
        self._set_video_status(GREEN if ok else RED,
            'Download complete' if ok else 'Download failed')
        self.studio.log_operation('Tools', f'视频下载{"完成 ✓" if ok else "失败"}'  )

    def _on_dl_error(self, err):
        print(f'[Video] Download error: {err}', flush=True)
        self.btn_video_download.setEnabled(True)
        self.btn_video_extract.setEnabled(True)
        self.video_progress.setFormat('Error')
        self._set_video_status(RED, 'yt-dlp not found. Install: pip install yt-dlp' if err == QProcess.FailedToStart else f'Download error: {err}')

    def _set_video_status(self, color, msg):
        self.video_status.setStyleSheet(f'font-size:9px;color:{color};padding:0;')
        self.video_status.setText(msg)

    # ═══════════════════════════════════════════
    # B站原生下载
    # ═══════════════════════════════════════════

    def _bili_download_start(self, info, output_dir, fmt_spec):
        global _BILI_DL_WORKER, _BILI_DL_THREAD
        if _BILI_DL_THREAD is not None:
            _BILI_DL_THREAD.quit()
            _BILI_DL_THREAD.wait(3000)
            _BILI_DL_THREAD = None
        if _BILI_DL_WORKER is not None:
            _BILI_DL_WORKER.deleteLater()
            _BILI_DL_WORKER = None

        _BILI_DL_WORKER = _BiliDownloadWorker(info, output_dir, fmt_spec)
        _BILI_DL_THREAD = QThread(self)
        _BILI_DL_WORKER.moveToThread(_BILI_DL_THREAD)
        _BILI_DL_THREAD.started.connect(_BILI_DL_WORKER.run)
        _BILI_DL_WORKER.log.connect(self._on_bili_dl_log)
        _BILI_DL_WORKER.progress.connect(self._on_bili_dl_progress)
        _BILI_DL_WORKER.filesReady.connect(self._on_bili_files_ready)
        _BILI_DL_WORKER.error.connect(self._on_bili_dl_error)
        _BILI_DL_THREAD.start()

    def _on_bili_dl_log(self, msg):
        print(f'[BiliDL] {msg}', flush=True)
        self._set_video_status(AMBER, msg)

    def _on_bili_dl_progress(self, current, total):
        if total:
            pct = int(current / total * 100)
            self.video_progress.setValue(pct)
            self.video_progress.setFormat(f'{pct}%')

    def _on_bili_files_ready(self, vpath, apath, fpath):
        global _BILI_DL_THREAD, _BILI_DL_WORKER
        if _BILI_DL_THREAD:
            _BILI_DL_THREAD.quit()
            _BILI_DL_THREAD.wait(3000)
            _BILI_DL_THREAD = None
        if _BILI_DL_WORKER:
            _BILI_DL_WORKER.deleteLater()
            _BILI_DL_WORKER = None

        self.video_progress.setValue(100)
        self.video_progress.setFormat('100%')
        self._set_video_status(GREEN, f'Done: {Path(fpath).name}')
        self.btn_video_download.setEnabled(True)
        self.btn_video_extract.setEnabled(True)
        self.studio.log_operation('Tools', f'B站下载完成 · {Path(fpath).name}')

    def _on_bili_dl_error(self, msg):
        global _BILI_DL_THREAD, _BILI_DL_WORKER
        if _BILI_DL_THREAD:
            _BILI_DL_THREAD.quit()
            _BILI_DL_THREAD.wait(3000)
            _BILI_DL_THREAD = None
        if _BILI_DL_WORKER:
            _BILI_DL_WORKER.deleteLater()
            _BILI_DL_WORKER = None
        self.btn_video_download.setEnabled(True)
        self.btn_video_extract.setEnabled(True)
        self.video_progress.setFormat('Error')
        self._set_video_status(RED, msg)
        self.studio.log_operation('Tools', f'B站下载失败 · {msg}')

    # ═══════════════════════════════════════════
    # 主题刷新
    # ═══════════════════════════════════════════

    def on_theme_changed(self):
        """Settings 主题切换后刷新样式"""
        self._post_process_ui()
