# =============================================================================
# YOLO Training Studio — 启动入口
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================
"""YOLO Training Studio 启动脚本。在项目根目录运行即可。"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.main import main

if __name__ == '__main__':
    main()
