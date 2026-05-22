"""YOLO Training Studio — WeChat 风格侧边栏，自适应布局"""
import sys, os
from pathlib import Path
from PyQt5.QtGui import QIcon

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from scripts.tabs.train import TrainTab
from scripts.tabs.distill import DistillTab
from scripts.tabs.dataset import DatasetTab
from scripts.tabs.predict import PredictTab
from scripts.tabs.preprocess import PreprocessTab
from scripts.tabs.label import LabelTab
from scripts.tabs.guide import GuideTab
from scripts.tabs.validate import ValidateTab
from scripts.tabs.export import ExportTab
from scripts.tabs.base import *


SIDE_W = 130  # 侧边栏宽度
NAV_ITEMS = [
    ('🎯  Training',   'training'),
    ('👁️  Predict',   'predict'),
    ('📁  Dataset',    'dataset'),
    ('⚙️  Preprocess', 'preprocess'),
    ('🏷️  Label',      'label'),
    ('🔬  Distill',    'distill'),
    ('✅  Validate',   'validate'),
    ('📦  Export',     'export'),
    ('📖  Guide',      'guide'),
]


class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.validator = None
        self._sys_data = {}
        self._nav_btns = []
        self._detect_gpu()
        self._build()
        self._start_sys_monitor()

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
        ico = str(ROOT / 'YOLO.ico')
        self.setWindowIcon(QIcon(ico))
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

        # 细底部分隔线
        w.setStyleSheet(f'background:{TOP_BG};border-bottom:1px solid {BORDER};')
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
        self._tab_label.setText(name_map.get(key, key))

    # ── 创建各标签页 ──
    def _create_tabs(self):
        self._tabs = {}

        pages = [
            ('training',   TrainTab(self)),
            ('predict',    PredictTab(self)),
            ('dataset',    DatasetTab(self)),
            ('preprocess', PreprocessTab(self)),
            ('label',      LabelTab(self)),
            ('distill',    DistillTab(self)),
            ('validate',   ValidateTab(self)),
            ('export',     ExportTab(self)),
            ('guide',      GuideTab(self)),
        ]
        for key, tab in pages:
            self._tabs[key] = tab
            self.stack.addWidget(tab)

        # 默认选中第一项
        if self._nav_btns:
            self._nav_btns[0][0].setChecked(True)
            self._switch('training')

    # ── 系统监控 ──
    def _start_sys_monitor(self):
        self._sys_timer = QTimer()
        self._sys_timer.timeout.connect(self._update_sys)
        self._sys_timer.start(2000)

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
            r = subprocess.run(
                ['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2)
            if r.returncode == 0 and r.stdout.strip():
                gpu = float(r.stdout.strip())
                if 'GPU' in self._sys_data:
                    self._sys_data['GPU'][0].setText(f'{gpu:.0f}%')
        except: pass


def main():
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('nous.yolo.training.studio')
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(ROOT / 'YOLO.ico')))
    app.setStyle('Fusion')
    app.setFont(QFont('Segoe UI', 9))

    window = Studio()
    app.window = window; window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
