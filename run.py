# =============================================================================
# YOLO Training Studio — 启动入口
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================
"""YOLO Training Studio 启动脚本。在项目根目录运行即可。"""

import sys
from multiprocessing import freeze_support
freeze_support()
from pathlib import Path
import os
import traceback
from datetime import datetime

# ── 全局异常捕获 ──
_CRASH_LOG = Path(os.environ.get('APPDATA', '.')) / 'YOLO Training Studio' / 'crash.log'

def _ensure_crash_dir():
    _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)

def _write_crash_log(exc_type, exc_value, exc_tb):
    _ensure_crash_dir()
    with open(_CRASH_LOG, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Crash at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Type: {exc_type.__name__}\n")
        f.write(f"Value: {exc_value}\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        f.write(f"{'='*60}\n")

def _crash_handler(exc_type, exc_value, exc_tb):
    """全局未捕获异常处理 — 写日志 + 弹窗提示用户"""
    _write_crash_log(exc_type, exc_value, exc_tb)
    # 如果 Qt 还在运行，弹消息框告知用户
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        if app:
            msg = f"程序遇到意外错误，即将退出。\n\n错误: {exc_type.__name__}\n{exc_value}\n\n详细信息已保存到:\n{_CRASH_LOG}"
            QMessageBox.critical(None, '程序异常退出', msg)
    except:
        pass
    # 默认行为
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _crash_handler

# ── 线程异常捕获 ──
import threading
def _thread_crash_handler(args):
    _write_crash_log(args.exc_type, args.exc_value, args.exc_tb)
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        if app:
            msg = f"线程异常退出。\n\n错误: {args.exc_type.__name__}\n{args.exc_value}\n\n详细信息已保存到:\n{_CRASH_LOG}"
            QMessageBox.critical(None, '程序异常退出', msg)
    except:
        pass
threading.excepthook = _thread_crash_handler

# ── 确保项目根目录在 Python 路径中 ──
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.main import main

if __name__ == '__main__':
    main()
