"""YOLO Training Studio — 集成主界面"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from scripts.tabs.train import TrainTab
from scripts.tabs.val import ValTab
from scripts.tabs.distill import DistillTab
from scripts.tabs.dataset import DatasetTab
from scripts.tabs.export import ExportTab
from scripts.tabs.predict import PredictTab
from scripts.tabs.preprocess import PreprocessTab
from scripts.tabs.label import LabelTab
from scripts.tabs.base import *


class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.validator = None
        self._sys_data = {}  # 初始化系统监控数据
        self._detect_gpu()
        self._build()
        self._start_sys_monitor()

    def _detect_gpu(self):
        self.gpu_ok = False; self.gpu_name = 'N/A'; self.gpu_mem = ''; self.cpu_count = 4
        try:
            import torch
            self.gpu_ok = torch.cuda.is_available()
            self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_ok else 'N/A'
            self.gpu_mem = f'{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB' if self.gpu_ok else ''
        except: pass
        import multiprocessing
        self.cpu_count = max(1, multiprocessing.cpu_count() // 2)

    def _build(self):
        self.setWindowTitle(f'YOLO Training — {TITLE}')
        # 计算固定窗口大小：左侧280 + 中间648(画布640+边距8) + 右侧280 + 额外边距 ≈ 1230
        self.setFixedSize(1230, 720)
        self.setStyleSheet(STYLE)
        c = QWidget(); self.setCentralWidget(c)
        ml = QVBoxLayout(c); ml.setContentsMargins(8,6,8,6); ml.setSpacing(4)

        h = QWidget(); h.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:6px;')
        hl = QHBoxLayout(h); hl.setContentsMargins(12,4,12,4)
        hl.addWidget(QLabel(f'◈ YOLO Training Studio', styleSheet=f'font-size:12px;font-weight:600;color:{TEXT};border:none;'))
        hl.addWidget(QLabel(TITLE, styleSheet=f'font-size:11px;color:{TEXT2};border:none;')); hl.addStretch()
        gu = f'GPU: {self.gpu_name} ({self.gpu_mem})' if self.gpu_ok else 'CPU only'
        hl.addWidget(QLabel(gu, styleSheet=f'font-size:10px;color:{TEXT2};border:none;'))
        ml.addWidget(h)

        tabs = QTabWidget(); ml.addWidget(tabs, 1)
        self.train_tab = TrainTab(self); tabs.addTab(self.train_tab, 'Training')
        self.val_tab = ValTab(self); tabs.addTab(self.val_tab, 'Validate')

        self.predict_tab = PredictTab(self); tabs.addTab(self.predict_tab, 'Predict')
        self.dataset_tab = DatasetTab(self); tabs.addTab(self.dataset_tab, 'Dataset')
        
        self.preprocess_tab = PreprocessTab(self); tabs.addTab(self.preprocess_tab, 'Preprocess')
        self.label_tab = LabelTab(self); tabs.addTab(self.label_tab, 'Label')
        self.distill_tab = DistillTab(self); tabs.addTab(self.distill_tab, 'Distill')
        self.export_tab = ExportTab(self); tabs.addTab(self.export_tab, 'Export')

    def _start_sys_monitor(self):
        self._sys_data = {}
        self._sys_timer = QTimer()
        self._sys_timer.timeout.connect(self._update_sys)
        self._sys_timer.start(2000)

    def _update_sys(self):
        # 移除空字典检查，允许后续注册监控组件
        if not hasattr(self, '_sys_data'):
            return
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            if 'CPU' in self._sys_data:
                self._sys_data['CPU'][0].setValue(int(cpu))
                self._sys_data['CPU'][1].setText(f'{cpu:.0f}%')
            if 'MEM' in self._sys_data:
                self._sys_data['MEM'][0].setValue(int(mem))
                self._sys_data['MEM'][1].setText(f'{mem:.0f}%')
            if 'DSK' in self._sys_data:
                self._sys_data['DSK'][0].setValue(int(disk))
                self._sys_data['DSK'][1].setText(f'{disk:.0f}%')
        except: pass
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vram = (total - free) / total * 100
                if 'VRM' in self._sys_data:
                    self._sys_data['VRM'][0].setValue(int(vram))
                    self._sys_data['VRM'][1].setText(f'{vram:.0f}%')
        except: pass
        try:
            import subprocess
            r = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2)
            if r.returncode == 0 and r.stdout.strip():
                gpu = float(r.stdout.strip())
                if 'GPU' in self._sys_data:
                    self._sys_data['GPU'][0].setValue(int(gpu))
                    self._sys_data['GPU'][1].setText(f'{gpu:.0f}%')
        except: pass


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Segoe UI', 9))

    window = Studio()
    app.window = window; window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
