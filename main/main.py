# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""YOLO Training Studio — WeChat 风格侧边栏，自适应布局"""
import sys, os, json
from pathlib import Path
from PyQt5.QtGui import QIcon, QCursor
import random


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

if sys.platform == 'win32':
    import io
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    elif hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')
    elif hasattr(sys.stderr, 'buffer') and sys.stderr.buffer is not None:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from main.core.train import TrainTab
from main.core.distill import DistillTab
from main.core.predict import PredictTab
from main.core.preprocess import PreprocessTab
from main.core.label import LabelTab
from main.core.review import RelabelTab
from main.core.guide import GuideTab
from main.core.settings import SettingsTab
from main.core.agent import AgentTab
from main.core.tools import ToolsTab
from main.core.base import *


SIDE_W = 110  # 侧边栏宽度
NAV_ITEMS = [
    ('🎯  Training',   'training'),
    ('👁️  Predict',   'predict'),
    ('🎞️  Preproc',    'preprocess'),
    ('🏷️  Label',      'label'),
    ('🔁  Review',     'review'),
    ('🔬  Distill',    'distill'),
    ('🤖  MIRO',       'agent'),
    ('🛠  Tools',      'tools'),
    ('⚙️  Settings',   'settings'),
    ('📖  Guide',      'guide'),
]


class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.validator = None
        self._sys_data = {}
        self._nav_btns = []
        self._ss_enabled = False  # 屏保开关，默认关闭
        self._detect_gpu()
        self._build()
        self._start_sys_monitor()
        self._start_ss_monitor()
        self._ref_w = 1400  # 参考宽度

    def resizeEvent(self, event):
        """窗口缩放时等比例调整字体"""
        super().resizeEvent(event)
        s = max(0.7, min(1.5, self.width() / self._ref_w))
        from PyQt5.QtGui import QFont
        app = QApplication.instance()
        f = app.font()
        f.setPointSize(round(9 * s))
        app.setFont(f)

    def _detect_gpu(self):
        self.gpu_ok = False; self.gpu_name = 'N/A'; self.gpu_mem = ''
        self.cpu_name = 'N/A'; self.cpu_count = 4
        try:
            import torch
            self.gpu_ok = torch.cuda.is_available()
            self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_ok else 'N/A'
            self.gpu_mem = f'{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB' if self.gpu_ok else ''
        except: pass
        import multiprocessing, platform
        self.cpu_count = max(1, multiprocessing.cpu_count() // 2)
        try:
            import pywintypes, winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            self.cpu_name = winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
            winreg.CloseKey(k)
        except: pass

    # ── 构建主界面 ──
    def _build(self):
        ico = str(ROOT / 'assets' / 'YOLO.ico')
        self.setWindowIcon(QIcon(_win_ico(ico)))
        self.setWindowTitle(f'{TITLE}')
        self.setMinimumSize(960, 600)
        self.setStyleSheet(STYLE)

        c = QWidget(); self.setCentralWidget(c)
        ml = QVBoxLayout(c); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)

        # 顶栏
        ml.addWidget(self._build_top())

        # 中间：侧边栏 + 内容
        mid = QWidget()
        mid_lo = QHBoxLayout(mid); mid_lo.setContentsMargins(0,0,0,0); mid_lo.setSpacing(0)
        mid_lo.addWidget(self._build_sidebar())

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet('color:#e5e5e5;')
        mid_lo.addWidget(sep)

        self.stack = QStackedWidget()
        mid_lo.addWidget(self.stack, 1)
        ml.addWidget(mid, 1)

        # 底栏
        ml.addWidget(self._build_bottom())

        # 创建各页
        self._create_tabs()

    # ── 顶栏 ──
    def _build_top(self):
        w = QWidget()
        w.setFixedHeight(44)
        w.setStyleSheet(f'background:{TOP_BG};')
        lo = QHBoxLayout(w); lo.setContentsMargins(16,0,16,0); lo.setSpacing(8)

        title = QLabel('YOLO Training')
        title.setStyleSheet(
            f'font-size:14px;font-weight:700;color:{TEXT};background:transparent;letter-spacing:.3px;')
        lo.addWidget(title)

        self._tab_label = QLabel('Training')
        self._tab_label.setStyleSheet(
            f'font-size:12px;font-weight:400;color:{TEXT2};background:transparent;margin-left:2px;')
        lo.addWidget(self._tab_label)

        lo.addStretch()

        # 刷新按钮
        self._refresh_btn = QPushButton('⟳')
        self._refresh_btn.setToolTip('Refresh all paths and file lists')
        self._refresh_btn.setFixedSize(32, 28)
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.setFont(QFont('Segoe UI Symbol', 14))
        self._refresh_btn.setStyleSheet(f'''
            QPushButton{{background:{BG};border:1px solid {BORDER};
                border-radius:4px;color:{TEXT2};padding:0;}}
            QPushButton:hover{{background:{BTN_HOVER};color:{TEXT};}}
        ''')
        self._refresh_btn.clicked.connect(self.refresh_all)
        lo.addWidget(self._refresh_btn)

        # 细底部分隔线
        w.setStyleSheet(f'background:{TOP_BG};border-bottom:1px solid {BORDER};')
        self._top_bar = w
        return w

    # ── 侧边栏 ──
    def _build_sidebar(self):
        w = QWidget()
        w.setFixedWidth(SIDE_W)
        w.setStyleSheet(f'background:{SIDE_BG};')
        lo = QVBoxLayout(w); lo.setContentsMargins(0,8,0,8); lo.setSpacing(2)

        for label, key in NAV_ITEMS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(42)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f'''
                QPushButton{{
                    background:transparent;border:none;border-radius:0;
                    text-align:left;padding:0 14px;
                    font-size:14px;font-weight:500;color:{TEXT2};
                    min-height:42px;max-height:42px;
                }}
                QPushButton:hover{{
                    background:{SIDE_HOVER};color:{TEXT};
                }}
                QPushButton:checked{{
                    background:{SIDE_ACTIVE};color:{TEXT};font-weight:600;
                    border-left:3px solid {PRI};padding:0 11px;
                }}
            ''')
            btn.clicked.connect(lambda checked, k=key: self._switch(k))
            lo.addWidget(btn)
            self._nav_btns.append((btn, key))

        lo.addStretch()
        self._sidebar = w
        return w

    # ── 底栏 ──
    def _build_bottom(self):
        w = QWidget()
        w.setFixedHeight(30)
        w.setStyleSheet(f'background:{BOT_BG};')
        lo = QHBoxLayout(w); lo.setContentsMargins(12,0,12,0); lo.setSpacing(16)

        # 设备信息（最左）
        dev_lbl = QLabel(
            f'{self.gpu_name} ({self.gpu_mem}) · {self.cpu_count}核' if self.gpu_ok
            else f'{self.cpu_name} · {self.cpu_count}核')
        dev_lbl.setStyleSheet(
            'font-size:10px;font-weight:600;color:#ddd;background:transparent;')
        lo.addWidget(dev_lbl)

        def mk_item(label, key):
            item = QWidget()
            item.setStyleSheet('background:transparent;')
            il = QHBoxLayout(item); il.setContentsMargins(0,0,0,0); il.setSpacing(4)
            lbl = QLabel(label)
            lbl.setStyleSheet('font-size:9px;color:#888;background:transparent;font-weight:500;')
            il.addWidget(lbl)
            val = QLabel('—')
            val.setStyleSheet('font-size:9px;color:#ccc;background:transparent;font-weight:600;')
            il.addWidget(val)
            lo.addWidget(item)
            self._sys_data[key] = [val]

        mk_item('CPU', 'CPU')
        mk_item('MEM', 'MEM')
        mk_item('DSK', 'DSK')
        if self.gpu_ok:
            mk_item('GPU', 'GPU')
            mk_item('VRAM', 'VRM')

        lo.addStretch()
        ver = QLabel(f'v{cfg["project"].get("version", "1.0")}')
        ver.setStyleSheet('font-size:9px;color:#555;background:transparent;')
        lo.addWidget(ver)
        self._bottom_bar = w
        return w

    # ── 导航切换 ──
    def _switch(self, key):
        for btn, k in self._nav_btns:
            btn.setChecked(k == key)
        tab = self._tabs.get(key)
        if tab:
            self.stack.setCurrentWidget(tab)
        # 更新顶栏标签页名称
        name_map = dict((k, l) for l, k in NAV_ITEMS)
        self._current_tab_name = key
        self._tab_label.setText(name_map.get(key, key))

    # ── 主题刷新 ──
    def refresh_theme(self):
        """切换主题后刷新所有已知内联样式"""
        import main.core.base as base_mod
        import importlib
        base_mod = importlib.reload(base_mod)
        # 顶栏
        if hasattr(self, '_top_bar'):
            self._top_bar.setStyleSheet(
                f'background:{base_mod.TOP_BG};border-bottom:1px solid {base_mod.BORDER};')
        if hasattr(self, '_tab_label'):
            self._tab_label.setStyleSheet(
                f'font-size:12px;font-weight:400;color:{base_mod.TEXT2};background:transparent;margin-left:2px;')
        # 侧边栏
        if hasattr(self, '_sidebar'):
            self._sidebar.setStyleSheet(f'background:{base_mod.SIDE_BG};')
        if hasattr(self, '_nav_btns'):
            for btn, _ in self._nav_btns:
                btn.setStyleSheet(f'''
                    QPushButton{{background:transparent;border:none;border-radius:0;
                        text-align:left;padding:0 14px;font-size:14px;font-weight:500;
                        color:{base_mod.TEXT2};min-height:42px;max-height:42px;
                    }}
                    QPushButton:hover{{background:{base_mod.SIDE_HOVER};color:{base_mod.TEXT};}}
                    QPushButton:checked{{background:{base_mod.SIDE_ACTIVE};color:{base_mod.TEXT};
                        font-weight:600;border-left:3px solid {base_mod.PRI};padding:0 11px;
                    }}
                ''')
        # 底栏
        if hasattr(self, '_bottom_bar'):
            self._bottom_bar.setStyleSheet(f'background:{base_mod.BOT_BG};')
        # 通知各标签页
        for key, tab in self._tabs.items():
            if hasattr(tab, 'on_theme_changed'):
                try:
                    tab.on_theme_changed()
                except Exception as e:
                    print(f'Theme refresh error in {key}: {e}')

    def refresh_all(self):
        """刷新所有标签页的文件读取与状态"""
        from main.config import load_paths
        import main.core.base as base_mod
        import importlib
        base_mod = importlib.reload(base_mod)

        paths = load_paths()
        for key, tab in self._tabs.items():
            try:
                # 各标签页的刷新入口
                if hasattr(tab, '_load_paths'):
                    tab._load_paths()
                if hasattr(tab, '_refresh_source_folders'):
                    tab._refresh_source_folders()
                if hasattr(tab, '_refresh_folder_list'):
                    tab._refresh_folder_list()
                if hasattr(tab, 'rebuild_class_buttons'):
                    tab.rebuild_class_buttons()
            except Exception as e:
                print(f'Refresh error in {key}: {e}')
        self._tab_label.setText('Refreshed')
        # 2秒后恢复
        from PyQt5.QtCore import QTimer
        name_map = dict((k, l) for l, k in NAV_ITEMS)
        original = name_map.get(self._current_tab_name, '')
        if original:
            QTimer.singleShot(2000, lambda o=original: self._tab_label.setText(o))

    # ── 创建各标签页 ──
    def _create_tabs(self):
        self._tabs = {}

        pages = [
            ('training',   TrainTab(self)),
            ('predict',    PredictTab(self)),
            ('preprocess', PreprocessTab(self)),
            ('label',      LabelTab(self)),
            ('review',    RelabelTab(self)),
            ('distill',    DistillTab(self)),
            ('agent',      AgentTab(self)),
            ('tools',      ToolsTab(self)),
            ('settings',   SettingsTab(self)),
            ('guide',      GuideTab(self)),
        ]
        for key, tab in pages:
            self._tabs[key] = tab
            self.stack.addWidget(tab)

        # 默认选中第一项
        if self._nav_btns:
            self._nav_btns[0][0].setChecked(True)
            self._switch('training')

    # ── 关闭窗口确认 ──
    def closeEvent(self, event):
        """训练中退出时弹出确认框"""
        tab = self._tabs.get('training')
        if tab and hasattr(tab, 'trainer') and tab.trainer and tab.trainer.isRunning():
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(self, '训练正在进行',
                '模型训练正在进行中，确定要退出吗？\n\n退出后训练将被中断，进度可能会丢失。',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        # 检查 label / review 页未保存的标注
        for key, name in [('label', 'Label'), ('review', 'Review')]:
            tab = self._tabs.get(key)
            if tab and getattr(tab, '_has_unsaved', False):
                from PyQt5.QtWidgets import QMessageBox
                reply = QMessageBox.question(self, f'{name} 未保存',
                    f'{name} 页有未保存的标注，确定要退出吗？\n\n退出后未保存的更改将丢失。',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return

        event.accept()

    # ── 系统监控 ──
    def _start_sys_monitor(self):
        self._sys_timer = QTimer()
        self._sys_timer.timeout.connect(self._update_sys)
        self._sys_timer.start(2000)

    # ── 屏幕保护：鼠标离开窗口 → 暗色蒙版 + YOLO 弹跳 ──
    def _ss_check(self):
        """轮询鼠标位置，控制屏幕保护覆盖层"""
        if not hasattr(self, '_ss_overlay'):
            return
        # 屏保被关闭 → 隐藏覆盖层并跳过
        if not self._ss_enabled:
            if self._ss_active:
                self._ss_active = False
                self._ss_overlay.hide_overlay()
            return
        cursor = QCursor.pos()
        rect = self.geometry()
        on_window = rect.contains(cursor) and not self.isMinimized() and self.isVisible()
        if on_window:
            if self._ss_active:
                self._ss_active = False
                self._ss_overlay.hide_overlay()
        else:
            if not self._ss_active:
                self._ss_active = True
                self._ss_overlay.show_overlay()
            else:
                self._ss_overlay.setGeometry(0, 0, self.width(), self.height())

    def _start_ss_monitor(self):
        self._ss_active = False
        self._ss_overlay = _ScreenSaverOverlay(self)
        self._ss_timer = QTimer(self)
        self._ss_timer.timeout.connect(self._ss_check)
        self._ss_timer.start(100)

    def _update_sys(self):
        if not hasattr(self, '_sys_data'):
            return
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            if 'CPU' in self._sys_data:
                self._sys_data['CPU'][0].setText(f'{cpu:.0f}%')
            if 'MEM' in self._sys_data:
                self._sys_data['MEM'][0].setText(f'{mem:.0f}%')
            if 'DSK' in self._sys_data:
                self._sys_data['DSK'][0].setText(f'{disk:.0f}%')
        except: pass
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vram = (total - free) / total * 100
                if 'VRM' in self._sys_data:
                    self._sys_data['VRM'][0].setText(f'{vram:.0f}%')
        except: pass
        try:
            import subprocess
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            r = subprocess.run(
                ['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2,
                startupinfo=si, creationflags=subprocess.CREATE_NO_WINDOW)
            if r.returncode == 0 and r.stdout.strip():
                gpu = float(r.stdout.strip())
                if 'GPU' in self._sys_data:
                    self._sys_data['GPU'][0].setText(f'{gpu:.0f}%')
        except: pass


class _ScreenSaverOverlay(QWidget):
    """全窗口暗色蒙版 + YOLO.png DVD 待机弹跳动画"""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hide()
        # 加载 YOLO.png
        p = ROOT / 'assets' / 'YOLO.png'
        self._raw = QPixmap(_win_ico(str(p))) if p.exists() else QPixmap()
        self._scaled = QPixmap()  # 预缩放缓存，避免每帧 scaled()
        self._logo_w = self._logo_h = 200
        self._x = self._y = 0
        self._dx = self._dy = 1  # 1px/帧 @ 60fps = 60px/s
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _update_scaled(self):
        """缓存预缩放的 pixmap，避免每帧 paintEvent 中 scaled()"""
        if not self._raw.isNull() and self._logo_w > 0 and self._logo_h > 0:
            self._scaled = self._raw.scaled(
                self._logo_w, self._logo_h,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            self._scaled = QPixmap()

    def show_overlay(self):
        p = self.parentWidget()
        if not p:
            return
        self.setGeometry(0, 0, p.width(), p.height())
        w, h = self.width(), self.height()
        self._logo_w = self._logo_h = min(200, w // 3, h // 3)
        self._x = random.randint(0, max(1, w - self._logo_w))
        self._y = random.randint(0, max(1, h - self._logo_h))
        self._dx = random.choice([-1, 1])
        self._dy = random.choice([-1, 1])
        self._update_scaled()  # 预缩放一次，后续动画不再调用 scaled()
        self.show()
        self.raise_()
        self._timer.start(16)  # ~60fps

    def hide_overlay(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        if not self.isVisible():
            return
        w, h = self.width(), self.height()
        self._x += self._dx
        self._y += self._dy
        if self._x <= 0:
            self._x = 0
            self._dx = abs(self._dx)
        elif self._x + self._logo_w >= w:
            self._x = w - self._logo_w
            self._dx = -abs(self._dx)
        if self._y <= 0:
            self._y = 0
            self._dy = abs(self._dy)
        elif self._y + self._logo_h >= h:
            self._y = h - self._logo_h
            self._dy = -abs(self._dy)
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.fillRect(0, 0, self.width(), self.height(), QColor(0, 0, 0, 200))
        if not self._scaled.isNull():
            qp.drawPixmap(self._x, self._y, self._scaled)


def _win_ico(path):
    '''Convert WSL /mnt/ path to Windows D:\\ path for QIcon.'''
    s = str(path)
    if sys.platform == 'win32' and s.startswith('/mnt/'):
        parts = s.split('/')
        s = f'{parts[2].upper()}:\\' + '\\'.join(parts[3:])
    return s


def main():
    import ctypes
    # 设置 AppUserModelID（非空！空字符串 Windows 会忽略，导致任务栏图标不显示）
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('YOLOTrainingStudio')
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    from main.core.settings import THEME_FILE, DARK_QSS
    try:
        if THEME_FILE.exists():
            dark = json.loads(THEME_FILE.read_text(encoding='utf-8')).get('dark', False)
            if dark:
                app.setStyleSheet(DARK_QSS)
    except:
        pass
    app.setWindowIcon(QIcon(_win_ico(ROOT / 'assets' / 'YOLO.ico')))
    app.setStyle('Fusion')
    app.setFont(QFont('Segoe UI', 9))

    window = Studio()
    app.window = window; window.show()

    # ── 启动时检查路径配置 ──
    from main.config import check_paths
    missing = check_paths()
    if missing:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(window, 'Paths Not Configured',
            '以下目录尚未配置，请在 Settings 页设置或点击 Init 初始化：\n\n' +
            '\n'.join(f'  • {m}' for m in missing))

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
