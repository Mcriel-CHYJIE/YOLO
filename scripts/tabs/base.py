"""共享基础模块 — 常量、UI 组件、工具函数"""
import sys, os, json, time, re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent

from scripts.tabs.config import cfg
IS_FALL = any(k.lower() in ('fallen', 'fall') for k in cfg['project']['classes'])
DATA_YAML = str(ROOT / cfg['project']['data_yaml'])
TITLE = cfg['project']['name']
CLASSES = cfg['project']['classes']
CLASS_NAMES = cfg['project'].get('names', {i: name for i, name in enumerate(CLASSES)})

import cv2, numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QTextCursor, QIcon, QColor, QIntValidator, QDoubleValidator, QImage

# ── 配色 ──
BG, CARD, BORDER = '#f5f5f4', '#ffffff', '#e7e5e4'
TEXT, TEXT2, TEXT3 = '#1c1917', '#78716c', '#a8a29e'
PRI, PRI_H = '#6366f1', '#4f46e5'
GREEN, RED, AMBER = '#10b981', '#ef4444', '#f59e0b'
CON, CON_T = '#0c0a09', '#e7e5e4'

STYLE = f"""
QMainWindow,QWidget{{background:{BG};}}
QGroupBox{{font-weight:600;font-size:10px;color:{TEXT};border:1px solid {BORDER};
    border-radius:7px;margin-top:9px;padding:10px 8px 8px;background:{CARD};}}
QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 5px;
    background:{CARD};color:{TEXT3};letter-spacing:.4px;}}
QPushButton{{border-radius:4px;padding:4px 14px;border:1px solid {BORDER};
    background:{CARD};color:{TEXT};min-height:22px;font-size:11px;}}
QPushButton:hover{{background:#f0efed;}}
QPushButton#pri{{background:{PRI};color:#fff;border:none;padding:5px 18px;min-height:26px;font-size:12px;}}
QPushButton#pri:hover{{background:{PRI_H};}}
QPushButton#pri:disabled{{background:#a5b4fc;}}
QPushButton#danger{{background:{RED};color:#fff;border:none;padding:5px 18px;min-height:26px;}}
QPushButton#danger:hover{{background:#dc2626;}}
QPushButton#danger:disabled{{background:#fca5a5;}}
QComboBox{{border:1px solid {BORDER};border-radius:4px;padding:2px 5px;
    background:{CARD};min-height:22px;color:{TEXT};font-size:11px;}}
QComboBox:focus{{border-color:{PRI};}}
QComboBox::drop-down{{border:none;width:16px;}}
QSpinBox,QDoubleSpinBox{{border:1px solid {BORDER};border-radius:4px;padding:2px 5px;
    background:{CARD};min-height:22px;color:{TEXT};font-size:11px;}}
QSpinBox:focus,QDoubleSpinBox:focus{{border-color:{PRI};}}
QSpinBox::up-button,QDoubleSpinBox::up-button{{width:0;padding:0;border:none;}}
QSpinBox::down-button,QDoubleSpinBox::down-button{{width:0;padding:0;border:none;}}
QProgressBar{{border:none;border-radius:1px;height:3px;background:{BORDER};text-align:center;}}
QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0{PRI},stop:1 '#8b5cf6');border-radius:1px;}}
QTextEdit{{background:{CON};color:{CON_T};border:none;border-radius:6px;
    padding:8px;font-family:Consolas,Courier New;font-size:11px;}}
QTabBar::tab{{padding:4px 16px;font-size:11px;font-weight:500;}}
QTabBar::tab:selected{{background:{CARD};border:1px solid {BORDER};border-bottom:none;font-weight:600;}}
QTabBar::tab:!selected{{background:#e7e5e4;color:{TEXT2};}}
QTabWidget::pane{{border:1px solid {BORDER};border-radius:6px;background:{CARD};}}
QCheckBox{{spacing:5px;font-size:11px;color:{TEXT};}}
"""

# ── 共享常量 ──
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
MODEL_FILTER = 'PyTorch (*.pt)'
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
            f'font-size:18px;font-weight:600;color:{color};qproperty-alignment:AlignCenter;')
        lo.addWidget(self.value_label)
        lo.addWidget(QLabel(label, styleSheet=f'font-size:9px;color:{TEXT3};font-weight:500;qproperty-alignment:AlignCenter;'))


class LogPanel(QWidget):
    def __init__(self, title='● Console', parent=None, max_lines=500):
        super().__init__(parent)
        self._max_lines = max_lines; self._log_lines = []
        self.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        lo = QVBoxLayout(self); lo.setContentsMargins(6, 4, 6, 6); lo.setSpacing(4)
        h = QWidget(); h.setStyleSheet('background:transparent;border:none;')
        hl = QHBoxLayout(h); hl.setContentsMargins(2, 0, 2, 0); hl.setSpacing(6)
        hl.addWidget(QLabel(title, styleSheet=f'font-size:10px;font-weight:600;color:{TEXT3};'))
        hl.addStretch()
        self.line_count = QLabel('0 lines')
        self.line_count.setStyleSheet(f'font-size:9px;color:{TEXT3};')
        hl.addWidget(self.line_count)
        clear_btn = QPushButton('Clear')
        clear_btn.setObjectName('danger')
        clear_btn.setStyleSheet('padding:2px 8px;min-height:18px;font-size:10px;')
        clear_btn.clicked.connect(self.clear)
        hl.addWidget(clear_btn)
        lo.addWidget(h)
        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setStyleSheet(
            f'QTextEdit{{background:{CON};color:{CON_T};border:none;border-radius:5px;'
            'padding:8px 10px;font-family:"Consolas","Courier New",monospace;font-size:13px;line-height:1.4;}}')
        lo.addWidget(self.editor)

    def append(self, html_line):
        self._log_lines.append(html_line)
        if len(self._log_lines) > self._max_lines:
            self._log_lines = self._log_lines[-self._max_lines:]
            self.editor.setHtml('\n'.join(self._log_lines))
        else:
            self.editor.append(html_line)
        self.editor.verticalScrollBar().setValue(self.editor.verticalScrollBar().maximum())
        self.line_count.setText(f'{len(self._log_lines)} lines')

    def replace_last(self, html_line):
        if self._log_lines:
            self._log_lines[-1] = html_line
            self.editor.setHtml('\n'.join(self._log_lines))
            self.editor.verticalScrollBar().setValue(self.editor.verticalScrollBar().maximum())

    def clear(self):
        self.editor.clear(); self._log_lines = []; self.line_count.setText('0 lines')


# ── 共享日志格式化 ──

def format_log(ts, msg):
    color = CON_T
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
    return f'<span style="color:#6b7280">[{ts}]</span> <span style="color:{color}">{msg}</span>'


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
        self.rf()

class MapChart(Chart):
    def upd(self, h):
        self.clr()
        if h.get('epoch') and len(h['epoch']):
            self.ax.plot(h['epoch'], h['mAP50'], color=GREEN, lw=1.5)
            self.ax.fill_between(h['epoch'], h['mAP50'], alpha=0.04, color=GREEN)
            if h.get('mAP50_95') and any(v>0 for v in h['mAP50_95']):
                self.ax.plot(h['epoch'], h['mAP50_95'], color=AMBER, lw=1, ls='--')
        self.rf()


def find_latest_best():
    d = ROOT / 'runs'
    if not d.exists(): return None
    m = sorted(d.rglob('weights/best5.20.pt'), key=lambda p: p.stat().st_mtime)
    return str(m[-1]) if m else None
