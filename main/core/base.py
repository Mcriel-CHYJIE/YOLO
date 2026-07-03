# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""共享基础模块 — 常量、UI 组件、工具函数（WeChat 风格配色）"""
import sys, os, json, time, re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent

from main.config import cfg, SETTINGS_FILE, DATA_YAML

# ── 主题加载（必须在颜色常量之前）──
_IS_DARK = False
try:
    if SETTINGS_FILE.exists():
        _IS_DARK = bool(json.loads(SETTINGS_FILE.read_text(encoding='utf-8')).get('theme', {}).get('dark', False))
except:
    pass

if _IS_DARK:
    # 深色主题
    BG, CARD, BORDER = '#1e1e1e', '#2d2d2d', '#3d3d3d'
    TEXT, TEXT2, TEXT3 = '#e0e0e0', '#999999', '#666666'
    PRI, PRI_H = '#07C160', '#06ad56'
    GREEN, RED, AMBER = '#07C160', '#ef4444', '#f59e0b'
    BLUE, PURPLE = '#3b82f6', '#8b5cf6'
    CON, CON_T = '#111111', '#d4d4d4'
    SIDE_BG, SIDE_HOVER, SIDE_ACTIVE = '#252526', '#333333', '#2d2d2d'
    TOP_BG, BOT_BG = '#2d2d2d', '#1a1a1a'
    BTN_HOVER = '#383838'
    SEC_HOVER = '#1a3d2d'
    WARN_HOVER = '#3d3010'
    SCROLL_H = '#555'
    SCROLL_HH = '#777'
else:
    # 浅色主题（WeChat 风格）
    BG, CARD, BORDER = '#f7f7f7', '#ffffff', '#e5e5e5'
    TEXT, TEXT2, TEXT3 = '#1a1a1a', '#7a7a7a', '#b0b0b0'
    PRI, PRI_H = '#07C160', '#06ad56'
    GREEN, RED, AMBER = '#07C160', '#ef4444', '#f59e0b'
    BLUE, PURPLE = '#3b82f6', '#8b5cf6'
    CON, CON_T = '#1e1e1e', '#d4d4d4'
    SIDE_BG, SIDE_HOVER, SIDE_ACTIVE = '#f0f0f0', '#e5e5e5', '#ffffff'
    TOP_BG, BOT_BG = '#ffffff', '#2b2b2b'
    BTN_HOVER = '#f0efed'
    SEC_HOVER = '#e8f5e9'
    WARN_HOVER = '#fff3e0'
    SCROLL_H = '#c0c0c0'
    SCROLL_HH = '#a0a0a0'
IS_FALL = any(k.lower() in ('fallen', 'fall') for k in cfg['project']['classes'])
TITLE = cfg['project']['name']
CLASSES = cfg['project']['classes']
CLASS_NAMES = cfg['project'].get('names', {i: name for i, name in enumerate(CLASSES)})

import cv2, numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer, QEvent, QSize
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QTextCursor, QIcon, QColor, QIntValidator, QDoubleValidator, QImage
from PyQt5.QtWidgets import QGraphicsOpacityEffect

STYLE = f"""
QMainWindow,QWidget{{background:{BG};}}
QStackedWidget{{background:{CARD};}}
QGroupBox{{font-weight:600;font-size:10px;color:{TEXT};border:1px solid {BORDER};
    border-radius:6px;margin-top:8px;padding:10px 8px 8px;background:{CARD};}}
QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 5px;
    background:{CARD};color:{TEXT3};}}
QPushButton{{border-radius:4px;padding:4px 14px;border:1px solid {BORDER};
    color:{TEXT};min-height:22px;font-size:11px;}}
QPushButton:hover{{background:{BTN_HOVER};}}
QPushButton#pri{{background:{PRI};color:#fff;border:none;padding:5px 18px;min-height:26px;font-size:12px;font-weight:600;border-radius:4px;}}
QPushButton#pri:hover{{background:{PRI_H};}}
QPushButton#pri:disabled{{background:#a5d6a5;}}
QPushButton#danger{{background:{RED};color:#fff;border:none;padding:5px 18px;min-height:26px;border-radius:4px;}}
QPushButton#danger:hover{{background:#dc2626;}}
QPushButton#danger:disabled{{background:#fca5a5;}}
QPushButton#sec{{background:{CARD};color:{PRI};border:1px solid {PRI};min-height:22px;font-size:11px;}}
QPushButton#sec:hover{{background:{SEC_HOVER};}}
QPushButton#warn{{background:{CARD};color:{AMBER};border:1px solid {AMBER};min-height:22px;font-size:11px;}}
QPushButton#warn:hover{{background:{WARN_HOVER};}}
QComboBox{{border:1px solid {BORDER};border-radius:4px;padding:2px 6px;
    background:{CARD};min-height:22px;color:{TEXT};font-size:11px;}}
QComboBox:focus{{border-color:{PRI};}}
QComboBox::drop-down{{border:none;width:16px;}}
QSpinBox,QDoubleSpinBox{{border:1px solid {BORDER};border-radius:4px;padding:2px 6px;
    background:{CARD};min-height:22px;color:{TEXT};font-size:11px;}}
QSpinBox:focus,QDoubleSpinBox:focus{{border-color:{PRI};}}
QSpinBox::up-button,QDoubleSpinBox::up-button{{width:0;padding:0;border:none;}}
QSpinBox::down-button,QDoubleSpinBox::down-button{{width:0;padding:0;border:none;}}
QProgressBar{{border:none;border-radius:1px;height:3px;background:{BORDER};text-align:center;}}
QProgressBar::chunk{{background:{PRI};border-radius:1px;}}
QTextEdit{{background:{CON};color:{CON_T};border:none;border-radius:5px;
    padding:8px;font-family:Consolas,Courier New;font-size:11px;}}
QCheckBox{{spacing:5px;font-size:11px;color:{TEXT};}}
QScrollBar:vertical{{width:6px;background:transparent;}}
QScrollBar::handle:vertical{{background:{SCROLL_H};border-radius:3px;min-height:30px;}}
QScrollBar::handle:vertical:hover{{background:{SCROLL_HH};}}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}
QScrollBar:horizontal{{height:6px;background:transparent;}}
QScrollBar::handle:horizontal{{background:{SCROLL_H};border-radius:3px;min-width:30px;}}
QScrollBar::handle:horizontal:hover{{background:{SCROLL_HH};}}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{{width:0;}}
"""

# ── 共享常量 ──
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
MODEL_FILTER = 'PyTorch (*.pt);;ONNX (*.onnx)'
CLASS_COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444',
                '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']

# ── 共享 UI 工具 ──

def labeled_field(widget, label, grid, r, c, height=22):
    cw = QWidget()
    cw.setStyleSheet(f'background:{BG};border-radius:4px;')
    cl = QHBoxLayout(cw); cl.setContentsMargins(6, 1, 6, 1); cl.setSpacing(4)
    lbl = QLabel(label)
    lbl.setStyleSheet(f'font-size:9px;color:{TEXT};background:transparent;font-weight:500;')
    lbl.setFixedHeight(height); widget.setMinimumHeight(height)
    cl.addWidget(lbl); cl.addWidget(widget, 1)
    grid.addWidget(cw, r, c)

def make_hparam_row(label, widget, height=24):
    cw = QWidget()
    cw.setStyleSheet(f'background:{BG};border-radius:4px;')
    cl = QHBoxLayout(cw); cl.setContentsMargins(8, 2, 8, 2); cl.setSpacing(6)
    lbl = QLabel(label)
    lbl.setStyleSheet(f'font-size:10px;color:{TEXT};font-weight:500;')
    cl.addWidget(lbl); cl.addWidget(widget, 1)
    return cw


class MetricCard(QWidget):
    def __init__(self, label, color=TEXT, default='0', parent=None):
        super().__init__(parent)
        self.setStyleSheet(f'background:{BG};border-radius:6px;padding:6px;')
        lo = QVBoxLayout(self); lo.setContentsMargins(8, 6, 8, 6); lo.setSpacing(2)
        self.value_label = QLabel(default)
        self.value_label.setStyleSheet(
            f'font-size:18px;font-weight:600;color:{color};text-align:center;')
        self.value_label.setAlignment(Qt.AlignCenter)
        lo.addWidget(self.value_label)
        title_label = QLabel(label)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f'font-size:9px;color:{TEXT3};font-weight:500;')
        lo.addWidget(title_label)


class LogPanel(QWidget):
    def __init__(self, title='● Console', parent=None, max_lines=500):
        super().__init__(parent)
        self._max_lines = max_lines; self._log_lines = []
        self.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        lo = QVBoxLayout(self); lo.setContentsMargins(4, 4, 4, 4); lo.setSpacing(0)

        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setStyleSheet(f"""
            QTextEdit {{
                background: {CON};
                color: {CON_T};
                border: none;
                border-radius: 5px;
                padding: 8px 10px;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 13px;
                line-height: 1.4;
            }}
        """)
        lo.addWidget(self.editor, 1)

    def append(self, html_line):
        self._log_lines.append(html_line)
        if len(self._log_lines) > self._max_lines:
            self._log_lines = self._log_lines[-self._max_lines:]
            self.editor.setHtml('\n'.join(self._log_lines))
        else:
            self.editor.append(html_line)
        self.editor.verticalScrollBar().setValue(self.editor.verticalScrollBar().maximum())

    def replace_last(self, html_line):
        if self._log_lines:
            self._log_lines[-1] = html_line
            self.editor.setHtml('\n'.join(self._log_lines))
            self.editor.verticalScrollBar().setValue(self.editor.verticalScrollBar().maximum())

    def clear(self):
        self.editor.clear(); self._log_lines = []


# ── 共享日志格式化 ──

def format_log(ts, msg):
    color = TEXT2
    if ('Done' in msg or 'complete' in msg.lower() or 'exported' in msg
        or 'saved' in msg or 'New best' in msg or 'Best mAP' in msg):
        color = GREEN
    elif ('Failed' in msg or 'fail' in msg.lower() or 'Error' in msg
          or 'error' in msg.lower()):
        color = RED
    elif ('Warning' in msg or 'warn' in msg.lower() or 'Skip' in msg
          or 'skip' in msg.lower()):
        color = AMBER
    elif ('Epoch' in msg or 'Start' in msg or 'Starting' in msg):
        color = '#a5b4fc'
    return f'<span style="color:#6b7280">[{ts}]</span> <span style="color:{color}">{msg}</span><br>'


# ── StreamEmitter ──
class StreamEmitter:
    def __init__(self, signal):
        self.signal = signal; self.buffer = ''
    def write(self, text):
        if text:
            text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
            self.buffer += text
            self.buffer = self.buffer.replace('\r', '\n')
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                if line.strip(): self.signal.emit(line)
    def flush(self):
        if self.buffer.strip():
            c = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', self.buffer.strip().replace('\r', ''))
            self.signal.emit(c); self.buffer = ''


# ── Chart 基类 ──
import matplotlib; matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Chart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(facecolor='white'); self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('white'); self.ax.tick_params(labelsize=7, colors=TEXT3)
        for s in ['top','right']: self.ax.spines[s].set_visible(False)
        for s in ['left','bottom']: self.ax.spines[s].set_color(BORDER)
        self.ax.grid(True, alpha=0.4, color=BORDER, linewidth=0.5)
        super().__init__(self.fig); self.setParent(parent); self.fig.tight_layout(pad=0.6)
    def clr(self):
        self.ax.clear(); self.ax.set_facecolor('white')
        for s in ['top','right']: self.ax.spines[s].set_visible(False)
        for s in ['left','bottom']: self.ax.spines[s].set_color(BORDER)
        self.ax.grid(True, alpha=0.4, color=BORDER, linewidth=0.5)
        self.ax.tick_params(labelsize=7, colors=TEXT3)
    def rf(self): self.fig.tight_layout(pad=0.6); self.draw()

class LossChart(Chart):
    def upd(self, h):
        self.clr()
        if h.get('epoch') and len(h['epoch']):
            self.ax.plot(h['epoch'], h['train_loss'], color=RED, lw=1.5)
            self.ax.fill_between(h['epoch'], h['train_loss'], alpha=0.04, color=RED)
            self.ax.set_title('Training Loss', fontsize=9, color=TEXT, fontweight='bold')
            self.ax.set_xlabel('Epoch', fontsize=7, color=TEXT2)
            self.ax.set_ylabel('Loss', fontsize=7, color=TEXT2)
        self.rf()

    def save(self, path):
        try:
            self.fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
            return True
        except Exception as e:
            print(f'Failed to save loss chart: {e}')
            return False

class MapChart(Chart):
    def upd(self, h):
        self.clr()
        if h.get('epoch') and len(h['epoch']):
            self.ax.plot(h['epoch'], h['mAP50'], color=GREEN, lw=1.5)
            self.ax.fill_between(h['epoch'], h['mAP50'], alpha=0.04, color=GREEN)
            if h.get('mAP50_95') and any(v>0 for v in h['mAP50_95']):
                self.ax.plot(h['epoch'], h['mAP50_95'], color=AMBER, lw=1, ls='--')
            self.ax.set_title('mAP Metrics', fontsize=9, color=TEXT, fontweight='bold')
            self.ax.set_xlabel('Epoch', fontsize=7, color=TEXT2)
            self.ax.set_ylabel('mAP', fontsize=7, color=TEXT2)
            legend_labels = ['mAP@0.5']
            if h.get('mAP50_95') and any(v>0 for v in h['mAP50_95']):
                legend_labels.append('mAP@0.5:0.95')
            self.ax.legend(legend_labels, loc='lower right', fontsize=6, framealpha=0.8)
        self.rf()

    def save(self, path):
        try:
            self.fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
            return True
        except Exception as e:
            print(f'Failed to save mAP chart: {e}')
            return False


class PrChart(Chart):
    """Precision + Recall 双曲线"""
    def upd(self, h):
        self.clr()
        if h.get('epoch') and len(h['epoch']):
            if h.get('precision') and any(v>0 for v in h['precision']):
                self.ax.plot(h['epoch'], h['precision'], color=GREEN, lw=1.5, label='Precision')
            if h.get('recall') and any(v>0 for v in h['recall']):
                self.ax.plot(h['epoch'], h['recall'], color=BLUE, lw=1.5, label='Recall')
            if h.get('precision') and any(v>0 for v in h['precision']) or \
               h.get('recall') and any(v>0 for v in h['recall']):
                self.ax.legend(loc='lower right', fontsize=6, framealpha=0.8)
            self.ax.set_title('Precision / Recall', fontsize=9, color=TEXT, fontweight='bold')
            self.ax.set_xlabel('Epoch', fontsize=7, color=TEXT2)
            self.ax.set_ylabel('Score', fontsize=7, color=TEXT2)
        self.rf()


class LrChart(Chart):
    """Learning Rate 曲线"""
    def upd(self, h):
        self.clr()
        if h.get('lr') and len(h['lr']) and any(v != 0 for v in h['lr']):
            epochs = h.get('epoch', list(range(len(h['lr']))))
            self.ax.plot(epochs, h['lr'], color=PURPLE, lw=1.5)
            self.ax.fill_between(epochs, h['lr'], alpha=0.04, color=PURPLE)
            self.ax.set_title('Learning Rate', fontsize=9, color=TEXT, fontweight='bold')
            self.ax.set_xlabel('Epoch', fontsize=7, color=TEXT2)
            self.ax.set_ylabel('LR', fontsize=7, color=TEXT2)
        self.rf()


def find_latest_best():
    d = ROOT / 'runs'
    if not d.exists(): return None
    m = sorted(d.rglob('weights/best5.20.pt'), key=lambda p: p.stat().st_mtime)
    return str(m[-1]) if m else None


# ── YOLO.png 水印 ──
_YOLO_PIXMAP = None

def _load_yolo_pixmap(size=200):
    global _YOLO_PIXMAP
    if _YOLO_PIXMAP is None:
        p = ROOT / 'assets' / 'YOLO.png'
        if p.exists():
            pm = QPixmap(str(p))
            if not pm.isNull():
                _YOLO_PIXMAP = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return _YOLO_PIXMAP


class _WatermarkFilter(QObject):
    def __init__(self, parent, label, pixmap):
        super().__init__(parent)
        self._label = label
        self._pm = pixmap
        parent.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            pw, ph = obj.width(), obj.height()
            lw, lh = self._pm.width(), self._pm.height()
            self._label.setGeometry((pw - lw) // 2, (ph - lh) // 2, lw, lh)
        return super().eventFilter(obj, event)


def set_watermark(container, size=200, opacity=0.25):
    pm = _load_yolo_pixmap(size)
    if pm is None:
        return
    label = QLabel(container)
    label.setPixmap(pm)
    label.setAttribute(Qt.WA_TransparentForMouseEvents)
    label.setStyleSheet('background:transparent;')
    effect = QGraphicsOpacityEffect(label)
    effect.setOpacity(opacity)
    label.setGraphicsEffect(effect)
    label.lower()
    _WatermarkFilter(container, label, pm)
    pw, ph = container.width(), container.height()
    lw, lh = pm.width(), pm.height()
    label.setGeometry((pw - lw) // 2, (ph - lh) // 2, lw, lh)


class ToggleSwitch(QWidget):
    """左右滑动开关按钮"""
    toggled = pyqtSignal(bool)

    def __init__(self, checked=True, parent=None):
        super().__init__(parent)
        self._checked = checked
        self.setFixedSize(44, 22)
        self.setCursor(Qt.PointingHandCursor)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        if self._checked != checked:
            self._checked = checked
            self.toggled.emit(checked)
            self.update()

    def mousePressEvent(self, e):
        self.setChecked(not self._checked)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # 背景圆角矩形
        bg = QColor('#07C160') if self._checked else QColor('#e7e5e4')
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(0, 0, w, h, h // 2, h // 2)
        # 滑块
        knob = QColor('#fff')
        p.setBrush(knob)
        pad = 2
        ks = h - pad * 2
        kx = w - ks - pad if self._checked else pad
        p.drawEllipse(kx, pad, ks, ks)
        p.end()
