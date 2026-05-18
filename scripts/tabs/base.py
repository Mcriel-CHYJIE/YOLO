"""共享基础模块 — 常量、Chart、Worker 类"""
import sys, os, json, time, re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Config-driven constants ──
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
    """网格布局中放置带标签的控件"""
    cw = QWidget()
    cw.setStyleSheet(f'background:{BG};border-radius:4px;')
    cl = QHBoxLayout(cw); cl.setContentsMargins(6, 1, 6, 1); cl.setSpacing(4)
    lbl = QLabel(label)
    lbl.setStyleSheet(f'font-size:9px;color:{TEXT};background:transparent;font-weight:500;')
    lbl.setFixedHeight(height); widget.setMinimumHeight(height)
    cl.addWidget(lbl); cl.addWidget(widget, 1)
    grid.addWidget(cw, r, c)

def make_hparam_row(label, widget, height=24):
    """水平布局中放置带标签的控件"""
    cw = QWidget()
    cw.setStyleSheet(f'background:{BG};border-radius:4px;')
    cl = QHBoxLayout(cw); cl.setContentsMargins(8, 2, 8, 2); cl.setSpacing(6)
    lbl = QLabel(label)
    lbl.setStyleSheet(f'font-size:10px;color:{TEXT};font-weight:500;')
    cl.addWidget(lbl); cl.addWidget(widget, 1)
    return cw


class MetricCard(QWidget):
    """统计卡片：大号数值 + 描述文字"""

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
    """统一日志面板：标题 + 行计数 + 清除按钮 + QTextEdit"""

    def __init__(self, title='● Console', parent=None, max_lines=500):
        super().__init__(parent)
        self._max_lines = max_lines
        self._log_lines = []

        self.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        lo = QVBoxLayout(self); lo.setContentsMargins(6, 4, 6, 6); lo.setSpacing(4)

        # 头部
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

        # 正文
        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setStyleSheet(
            f'QTextEdit{{background:{CON};color:{CON_T};border:none;border-radius:5px;'
            'padding:8px 10px;font-family:"Consolas","Courier New",monospace;font-size:13px;line-height:1.4;}}')
        lo.addWidget(self.editor)

    def append(self, html_line):
        """追加一行 HTML 日志"""
        self._log_lines.append(html_line)
        if len(self._log_lines) > self._max_lines:
            self._log_lines = self._log_lines[-self._max_lines:]
            self.editor.setHtml('\n'.join(self._log_lines))
        else:
            self.editor.append(html_line)
        self.editor.verticalScrollBar().setValue(self.editor.verticalScrollBar().maximum())
        self.line_count.setText(f'{len(self._log_lines)} lines')

    def replace_last(self, html_line):
        """替换最后一行（用于进度条更新）"""
        if self._log_lines:
            self._log_lines[-1] = html_line
            self.editor.setHtml('\n'.join(self._log_lines))
            self.editor.verticalScrollBar().setValue(self.editor.verticalScrollBar().maximum())

    def clear(self):
        self.editor.clear(); self._log_lines = []; self.line_count.setText('0 lines')


# ── 共享日志格式化 ──

def format_log(ts, msg):
    """统一日志格式（返回 HTML 行）"""
    color = CON_T
    if '✅' in msg or '🎉' in msg or 'Done' in msg:
        color = GREEN
    elif '❌' in msg or 'Failed' in msg:
        color = RED
    elif '⚠️' in msg:
        color = AMBER
    elif '🚀' in msg or 'Epoch' in msg:
        color = '#a5b4fc'
    elif '📁' in msg or '📄' in msg:
        color = TEXT3
    elif '🎬' in msg:
        color = PRI
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


# ── Trainer ──
class Trainer(QThread):
    log = pyqtSignal(str); status = pyqtSignal(str,float,float,float); chart = pyqtSignal(); done = pyqtSignal(bool,str)
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self._stop = False
        self.history = {'epoch':[],'train_loss':[],'mAP50':[],'mAP50_95':[],'precision':[],'recall':[]}
        self.best_map = 0.0
        self.stdout_emitter = StreamEmitter(self.log)
        self.stderr_emitter = StreamEmitter(self.log)
    def stop(self): self._stop = True
    
    def _check_gpu_memory(self):
        """Check GPU memory and return available memory in GB"""
        import torch
        if not torch.cuda.is_available():
            return None
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)
            free_mem = total_mem - allocated_mem
            return {
                'total': total_mem,
                'allocated': allocated_mem,
                'reserved': reserved_mem,
                'free': free_mem
            }
        except Exception as e:
            self.log.emit(f'⚠️ GPU memory check error: {e}')
            return None
    def _save_log(self, ok, error=''):
        try:
            d = ROOT / 'runs' / 'training_logs'; d.mkdir(parents=True, exist_ok=True)
            best_m95 = 0.0
            if self.history['mAP50_95'] and any(v > 0 for v in self.history['mAP50_95']):
                best_m95 = max(self.history['mAP50_95'])
            entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'project': ROOT.name, 'status': 'success' if ok else 'failed',
                'error': error,
                'config': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v for k, v in self.cfg.items()},
                'results': {
                    'best_mAP50': round(self.best_map, 4), 'best_mAP50_95': round(best_m95, 4),
                    'final_mAP50': round(self.history['mAP50'][-1], 4) if self.history['mAP50'] else 0,
                    'final_loss': round(self.history['train_loss'][-1], 4) if self.history['train_loss'] else 0,
                    'epochs_trained': len(self.history['epoch']),
                },
                'history': {k: [round(v, 4) for v in vals] for k, vals in self.history.items()},
            }
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(d / f'{ts}.json', 'w', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            hp = d / 'training_history.json'
            h = json.loads(hp.read_text('utf-8')) if hp.exists() else []
            h.append(entry)
            hp.write_text(json.dumps(h, ensure_ascii=False, indent=2), 'utf-8')
            self.log.emit(f'📁 Log saved: {ts}.json')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log.emit(f'⚠️ Log save failed: {e}')
    def run(self):
        import sys, torch
        original_stdout = sys.stdout; original_stderr = sys.stderr
        try:
            sys.stdout = self.stdout_emitter; sys.stderr = self.stderr_emitter
            from ultralytics import YOLO
            cfg = self.cfg; m = YOLO(cfg['model'])
            exp = f'{ROOT.name}_{datetime.now().strftime("%m%d_%H%M")}'
            self.log.emit(f'🚀 {cfg["model"]} | {cfg["epochs"]}ep | batch={cfg["batch"]}')
            if cfg.get('lr0', 0) <= 0:
                cfg['lr0'] = 0.001; self.log.emit(f'⚠️  LR0 was 0, reset to 0.001')
            
            # Monitor GPU memory before training
            if torch.cuda.is_available() and cfg['device'] != 'cpu':
                try:
                    gpu_info = self._check_gpu_memory()
                    if gpu_info:
                        self.log.emit(f'💻 GPU Memory: {gpu_info["total"]:.1f}GB total, {gpu_info["free"]:.1f}GB free')
                        # Warn if free memory is low
                        if gpu_info['free'] < 2.0:  # Less than 2GB free
                            self.log.emit('⚠️ Low GPU memory! Consider reducing batch size or image size')
                    # Clear cache
                    torch.cuda.empty_cache()
                except Exception as e:
                    self.log.emit(f'⚠️ GPU info error: {e}')
            
            def cb(t):
                if self._stop: t.stop = True; return
                try:
                    ep = t.epoch; loss = 0.0
                    try:
                        if hasattr(t,'loss') and t.loss is not None: loss = float(t.loss)
                        if loss == 0.0 and hasattr(t,'tloss') and t.tloss is not None: loss = float(t.tloss)
                    except: pass
                    mt = getattr(t,'metrics',None) or {}
                    m50 = float(mt.get('metrics/mAP50(B)',0)); m95 = 0.0
                    for k in ['metrics/mAP50_95(B)','metrics/mAP50_95','metrics/mAP_0.5:0.95']:
                        v = mt.get(k)
                        if v is not None:
                            try: m95 = float(v); break
                            except: pass
                    p = float(mt.get('metrics/precision(B)',0)); r = float(mt.get('metrics/recall(B)',0))
                    self.history['epoch'].append(ep); self.history['train_loss'].append(loss)
                    self.history['mAP50'].append(m50); self.history['mAP50_95'].append(m95)
                    self.history['precision'].append(p); self.history['recall'].append(r)
                    if m50 > self.best_map: self.best_map = m50
                    total_epochs = max(cfg['epochs'], 1)
                    progress = min(ep / total_epochs, 1.0)
                    self.status.emit(f'Epoch {ep}/{cfg["epochs"]}', progress, self.best_map, m50)
                    self.chart.emit()
                    
                    # Monitor GPU memory every 10 epochs
                    if ep % 10 == 0 and torch.cuda.is_available() and cfg['device'] != 'cpu':
                        try:
                            gpu_info = self._check_gpu_memory()
                            if gpu_info:
                                self.log.emit(f'📊 GPU Memory (Epoch {ep}): {gpu_info["allocated"]:.1f}GB allocated, {gpu_info["free"]:.1f}GB free')
                        except Exception as e:
                            pass  # Don't let memory check errors interrupt training
                except Exception as e:
                    try:
                        self.log.emit(f'⚠️ Callback error: {e}')
                    except: pass
            m.add_callback('on_train_epoch_end', cb)
            train_args = dict(data=DATA_YAML, epochs=cfg['epochs'], batch=cfg['batch'], imgsz=cfg['imgsz'],
                lr0=cfg['lr0'], lrf=cfg['lrf'], optimizer=cfg['optimizer'], patience=cfg['patience'],
                device=cfg['device'], warmup_epochs=3, warmup_momentum=0.8, cos_lr=cfg['cos_lr'],
                flipud=0.0 if IS_FALL else 0.3, fliplr=0.5, mosaic=1.0, mixup=0.2, workers=cfg.get('workers', 4),
                fl_gamma=cfg.get('fl_gamma', 1.5),
                label_smoothing=cfg['label_smoothing'] if cfg['label_smoothing'] > 0 else 0,
                iou=cfg['iou'], close_mosaic=cfg['close_mosaic'],
                copy_paste=cfg['copy_paste'] if cfg['copy_paste'] > 0 else 0,
                degrees=cfg['degrees'] if cfg['degrees'] > 0 else 0, multi_scale=cfg['multi_scale'],
                hsv_h=cfg.get('hsv_h', 0.015), hsv_s=cfg.get('hsv_s', 0.7), hsv_v=cfg.get('hsv_v', 0.4),
                translate=cfg.get('translate', 0.15), scale=cfg.get('scale', 0.6),
                project='runs', name=exp, exist_ok=True, amp=True, verbose=False)
            train_args = {k: v for k, v in train_args.items() if v is not None}
            
            # Add error handling for training
            try:
                m.train(**train_args)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    self.log.emit(f'❌ CUDA OOM Error: {e}')
                    self.log.emit('💡 Suggestion: Reduce batch size or image size')
                    raise
                else:
                    self.log.emit(f'❌ Training Runtime Error: {e}')
                    raise
            except Exception as e:
                self.log.emit(f'❌ Training Error: {e}')
                raise
                
            stopped = self._stop
            self._save_log(ok=not stopped, error='Stopped by user' if stopped else '')
            self.done.emit(not stopped, f'⏹ Stopped' if stopped else f'✅ Done | Best mAP@0.5 = {self.best_map:.4f}')
        except BaseException as e:
            import traceback; traceback.print_exc()
            self._save_log(ok=False, error=str(e))
            self.log.emit(f'❌ {e}')
            self.done.emit(False, f'Failed: {e}')
        finally:
            sys.stdout = original_stdout; sys.stderr = original_stderr


def find_latest_best():
    d = ROOT / 'runs'
    if not d.exists(): return None
    m = sorted(d.rglob('weights/best.pt'), key=lambda p: p.stat().st_mtime)
    return str(m[-1]) if m else None


# ── Distillation Loss Wrapper (picklable) ──
class DistillationLossWrapper:
    """Wrapper class for distillation loss (picklable for multiprocessing)"""
    def __init__(self, orig_loss, teacher_model, alpha, device):
        self.orig_loss = orig_loss
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.device = device
    
    def __call__(self, preds, batch):
        import torch
        import torch.nn.functional as F
        det_loss, loss_items = self.orig_loss(batch, preds=preds)
        with torch.no_grad():
            to = self.teacher_model(batch['img'].to(self.device))
            t_feats = to[1]['feats'] if isinstance(to, tuple) else to.get('feats', [])
        s_feats = preds.get('feats', [])
        distill = 0.0
        n = min(len(s_feats), len(t_feats))
        for i in range(n):
            sp = s_feats[i].view(s_feats[i].size(0), -1)
            tp = t_feats[i].view(t_feats[i].size(0), -1)
            md = min(sp.size(-1), tp.size(-1))
            distill += F.mse_loss(sp[:, :md], tp[:, :md].detach())
        return (1.0 - self.alpha) * det_loss.sum() + self.alpha * distill / max(n, 1), loss_items


# ── DistillWorker ──
class DistillWorker(QThread):
    log = pyqtSignal(str); progress = pyqtSignal(int, dict); done = pyqtSignal(bool, str)
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self._stop = False
    def stop(self): self._stop = True
    def run(self):
        try: self._train()
        except BaseException as e:
            import traceback; traceback.print_exc()
            self.log.emit(f'❌ {e}')
            self.done.emit(False, str(e))
    def _train(self):
        import torch, torch.nn.functional as F, torch.optim as optim
        from torch.optim.lr_scheduler import CosineAnnealingLR
        import numpy as np, shutil, json, time
        from ultralytics import YOLO
        from ultralytics.data import build_dataloader, build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils import colorstr
        from types import SimpleNamespace; from copy import deepcopy
        cfg = self.cfg; device = cfg['device']; ROOT2 = ROOT
        
        # Monitor GPU memory before distillation
        if torch.cuda.is_available() and 'cuda' in str(device):
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                free_mem = gpu_mem - allocated
                self.log.emit(f'💻 GPU Memory: {gpu_mem:.1f}GB total, {free_mem:.1f}GB free')
                if free_mem < 2.0:  # Less than 2GB free
                    self.log.emit('⚠️ Low GPU memory! Consider reducing batch size or image size')
                torch.cuda.empty_cache()
            except Exception as e:
                self.log.emit(f'⚠️ GPU info error: {e}')
        
        self.log.emit(f'[Teacher] {cfg["teacher"]}')
        teacher = YOLO(cfg['teacher']); teacher.model.to(device); teacher.model.eval()
        for p in teacher.model.parameters(): p.requires_grad = False
        self.log.emit(f'[Student] {cfg["student"]}')
        student = YOLO(cfg['student'])
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.utils import IterableSimpleNamespace
        student.model.args = IterableSimpleNamespace(**(DEFAULT_CFG_DICT | student.model.args))
        for p in student.model.parameters(): p.requires_grad = True
        student.model.to(device); student.model.train()
        alpha = cfg['alpha']; orig_loss = student.model.loss
        # Use picklable wrapper class instead of local function
        loss_wrapper = DistillationLossWrapper(orig_loss, teacher.model, alpha, device)
        student.model.loss = loss_wrapper
        self.log.emit(f'[Data] {cfg["data"]}')
        data_dict = check_det_dataset(cfg['data'])
        cfg_ns = SimpleNamespace(imgsz=cfg['imgsz'], batch=cfg['batch'], workers=0,
            lr0=cfg['lr0'], weight_decay=cfg.get('weight_decay', 0.0005), momentum=cfg.get('momentum', 0.937), rect=False, cache=None,
            single_cls=False, stride=32, pad=0.0, prefix=colorstr('train: '), task='detect',
            classes=None, fraction=1.0, augment=True, hyp=None, data=data_dict,
            mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode='flip', cutmix=0.0, degrees=0.0,
            translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0,
            fliplr=0.5, bgr=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4, mask_ratio=4, overlap_mask=False)
        ds = build_yolo_dataset(cfg_ns, data_dict['train'], cfg['batch'], data_dict, mode='train', rect=False, stride=32)
        # Use workers=0 to avoid pickle issues on Windows with local functions/classes
        dl = build_dataloader(ds, batch=cfg['batch'], workers=0, shuffle=True)
        self.log.emit(f'[Data] {len(ds)} samples, {len(dl)} batches/epoch')
        # Optimizer with configurable parameters
        weight_decay = cfg.get('weight_decay', 0.0005)
        momentum_beta = cfg.get('momentum', 0.937)
        opt = optim.AdamW(student.model.parameters(), lr=cfg['lr0'], weight_decay=weight_decay, betas=(momentum_beta, 0.999))
        warmup_epochs = cfg.get('warmup_epochs', 5)
        sch = CosineAnnealingLR(opt, T_max=cfg['epochs'] - warmup_epochs, eta_min=cfg['lr0'] * 0.01)
        save_dir = ROOT2 / 'runs' / 'distill' / cfg['name']; save_dir.mkdir(parents=True, exist_ok=True)
        best_map50, best_epoch, pc, hist = 0.0, 0, 0, {'epoch':[],'loss':[],'map50':[],'map50_95':[]}
        total_batches = len(dl)
        self.log.emit(f'\n🚀 Starting distillation training...')
        for ep in range(1, cfg['epochs'] + 1):
            if self._stop: break
            # Warmup learning rate
            if ep <= warmup_epochs:
                warmup_factor = ep / warmup_epochs
                for param_group in opt.param_groups:
                    param_group['lr'] = cfg['lr0'] * warmup_factor
            student.model.train(); el, nb = 0.0, 0
            # Log epoch start with progress
            epoch_progress = f'{ep}/{cfg["epochs"]}'
            self.log.emit(f'\n[Epoch {epoch_progress}] LR: {opt.param_groups[0]["lr"]:.6f} | Batches: {total_batches}')
            batch_start_time = time.time()
            for bi, batch in enumerate(dl):
                if self._stop: break
                try:
                    img = batch['img'].to(device, non_blocking=True)
                    img = img.float() / 255.0 if img.dtype == torch.uint8 else img; batch['img'] = img
                    for k in ['cls','bboxes','batch_idx']:
                        if k in batch and isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k].to(device, non_blocking=True)
                    loss, _ = student.model.loss(student.model(img), batch)
                    opt.zero_grad(); loss.backward(); opt.step()
                    el += loss.item(); nb += 1
                    
                    # Log batch progress every 10% or last batch
                    batch_idx = bi + 1
                    if batch_idx % max(total_batches // 10, 1) == 0 or batch_idx == total_batches:
                        batch_pct = batch_idx / total_batches * 100
                        elapsed = time.time() - batch_start_time
                        avg_time = elapsed / batch_idx
                        eta = avg_time * (total_batches - batch_idx)
                        self.log.emit(f'  [{batch_idx}/{total_batches}] ({batch_pct:.0f}%) | Loss: {loss.item():.4f} | ETA: {eta:.0f}s')
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        self.log.emit('[OOM] skip batch'); torch.cuda.empty_cache(); continue
                    raise
            sch.step(); avg = el / max(nb, 1); vm50, vm95 = 0.0, 0.0
            # Validate every epoch for distillation
            validate_now = True
            if validate_now:
                try:
                    # Use a fresh YOLO instance for validation to avoid custom loss interference
                    vm = YOLO(cfg['student'])
                    vm.model.load_state_dict(student.model.state_dict())
                    vm.model.to(device)
                    vm.model.eval()
                    r = vm.val(data=cfg['data'], device=device, batch=min(cfg['batch'], 16), imgsz=cfg['imgsz'], 
                              plots=False, verbose=False, save=False, rect=True)
                    vm50, vm95 = float(r.box.map50), float(r.box.map)
                    self.log.emit(f'  [Val] mAP50: {vm50:.4f}, mAP50-95: {vm95:.4f}')
                except Exception as ve:
                    # Log validation errors for debugging
                    import traceback
                    self.log.emit(f'[Val Error] {ve}')
                    self.log.emit(traceback.format_exc())
                    vm50, vm95 = 0.0, 0.0
                if vm50 > best_map50:
                    best_map50, best_epoch, pc = vm50, ep, 0
                    # Save only model weights (state_dict) to reduce file size
                    torch.save({
                        'model': student.model.state_dict(),
                        'names': student.names,
                        'args': student.model.args,
                        'version': '8.0.0'
                    }, str(save_dir / 'best.pt'))
                    self.log.emit(f'  🎉 New best mAP50: {vm50:.4f} (epoch {ep})')
                else: pc += 1
            if ep % 10 == 0:
                # Save only model weights for checkpoint files
                torch.save({
                    'model': student.model.state_dict(),
                    'names': student.names,
                    'args': student.model.args,
                    'version': '8.0.0'
                }, str(save_dir / f'epoch_{ep}.pt'))
            hist['epoch'].append(ep); hist['loss'].append(avg); hist['map50'].append(vm50); hist['map50_95'].append(vm95)
            self.log.emit(f'Epoch {ep}/{cfg["epochs"]} | Loss: {avg:.4f} | mAP50: {vm50:.4f}' + (f' | Best: {best_map50:.4f}@{best_epoch}' if best_map50 > 0 else ''))
            
            # Monitor GPU memory every 10 epochs
            if ep % 10 == 0 and torch.cuda.is_available() and 'cuda' in str(device):
                try:
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    self.log.emit(f'📊 GPU Memory (Epoch {ep}): {allocated:.1f}GB allocated, {reserved:.1f}GB reserved')
                except Exception as e:
                    pass  # Don't let memory check errors interrupt training
            
            self.progress.emit(ep, {'epoch': ep, 'total': cfg['epochs'], 'loss': avg, 'map50': vm50, 'best_map50': best_map50, 'best_epoch': best_epoch, 'lr': opt.param_groups[0]['lr'], 'history': dict(hist)})
            # Early stopping: only after warmup and sufficient training
            if pc >= cfg['patience'] and ep > warmup_epochs + 20: 
                self.log.emit(f'[Early Stop] No improvement for {pc} epochs'); break
        # Save last checkpoint with only weights
        torch.save({
            'model': student.model.state_dict(),
            'names': student.names,
            'args': student.model.args,
            'version': '8.0.0'
        }, str(save_dir / 'last.pt'))
        with open(str(save_dir / 'history.json'), 'w') as f: json.dump(hist, f, indent=2)
        self.log.emit(f'Done! Best mAP50: {best_map50:.4f}')
        self.done.emit(True, str(save_dir))


# ── DetectWorker ──
class DetectWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, int, int)
    fps_updated = pyqtSignal(float); stats_updated = pyqtSignal(dict)
    log_signal = pyqtSignal(str); finished = pyqtSignal()
    def __init__(self, model_path, video_path, conf=0.25, iou=0.45):
        super().__init__()
        self.model_path = Path(model_path); self.video_path = Path(video_path)
        self.conf = conf; self.iou = iou; self._pause = False; self._stop = False
        self.export_path = None  # Path to save detected video
    def stop(self): self._stop = True; self._pause = False
    def toggle_pause(self): self._pause = not self._pause
    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(str(self.model_path))
            self.log_signal.emit(f'✅ {self.model_path.name}')
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened(): self.log_signal.emit('Failed to open video'); self.finished.emit(); return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.log_signal.emit(f'🎬 {self.video_path.name}  {total}帧')
            
            # Setup video writer if export is enabled
            writer = None
            if self.export_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(str(self.export_path), fourcc, fps, (w, h))
                if writer.isOpened():
                    self.log_signal.emit(f'💾 Saving to: {Path(self.export_path).name}')
                else:
                    self.log_signal.emit('⚠️ Failed to create video writer')
                    writer = None
            
            idx, t_prev = 0, datetime.now()
            while not self._stop and cap.isOpened():
                if self._pause: self.msleep(50); continue
                ret, frame = cap.read()
                if not ret: break
                idx += 1
                results = model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
                annotated = results.plot(line_width=2, font_size=8)
                
                # Write frame to video if exporting
                if writer is not None:
                    writer.write(annotated)
                
                now = datetime.now(); fps = 1.0 / max((now - t_prev).total_seconds(), 0.001); t_prev = now
                stats = {}
                if results.boxes is not None:
                    for cls_id in results.boxes.cls:
                        name = model.names.get(int(cls_id), f'cls_{int(cls_id)}')
                        stats[name] = stats.get(name, 0) + 1
                self.frame_ready.emit(annotated, idx, total)
                self.fps_updated.emit(fps); self.stats_updated.emit(stats)
            
            # Release resources
            if writer is not None:
                writer.release()
                self.log_signal.emit(f'✅ Video saved: {Path(self.export_path).name}')
            cap.release()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log_signal.emit(f'❌ {e}')
        finally: self.finished.emit()
