# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""
Config Loader — reads main/project.yaml as the single source of truth.

Usage:
    from main.config import cfg
    cfg['project']['name']       # → 'YOLO Training Studio'
    cfg['training']['epochs']    # → 300
    cfg['predict']['conf']       # → 0.35

JSON config files (attention, shortcuts, theme) are stored in this directory.
"""

import yaml
import json
from pathlib import Path
from ultralytics import settings as _ultra_settings

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = Path(__file__).resolve().parent.parent / 'project.yaml'

# ── JSON 配置文件路径（统一在 main/config/ 下） ──
_CFG_DIR = Path(__file__).resolve().parent
ATTENTION_FILE = _CFG_DIR / 'attention.json'
THEME_FILE = _CFG_DIR / 'theme.json'
SHORTCUTS_FILE = _CFG_DIR / 'shortcuts.json'
PATHS_FILE = _CFG_DIR / 'paths.json'

# ── 工作目录路径配置 ──
_DEFAULT_PATHS = {
    'train_output': '',
    'predict_output': '',
    'dataset_dir': '',
    'preproc_dir': '',
    'label_dir': '',
    'export_dir': '',
    'models_dir': '',
}


def load_paths() -> dict:
    """从 paths.json 读取目录配置，缺失项用空字符串"""
    defaults = dict(_DEFAULT_PATHS)
    if PATHS_FILE.exists():
        try:
            saved = json.loads(PATHS_FILE.read_text('utf-8'))
            defaults.update(saved)
        except:
            pass
    return defaults


def check_paths() -> list:
    """检查路径配置是否完整，返回未配置的项列表"""
    paths = load_paths()
    labels = {
        'train_output': 'Training output',
        'predict_output': 'Predict output',
        'dataset_dir': 'Dataset dir',
        'preproc_dir': 'Preprocess dir',
        'label_dir': 'Label dir',
        'export_dir': 'Export dir',
        'models_dir': 'Models dir',
    }
    return [labels[k] for k in labels if not paths.get(k)]


# ── 重定向模型下载到配置目录 ──
def _setup_models_dir():
    paths = load_paths()
    md = paths.get('models_dir') or str(ROOT / 'models')
    Path(md).mkdir(parents=True, exist_ok=True)
    _ultra_settings.update({'weights_dir': md})
_setup_models_dir()
_FALLBACK = {
    'project': {'name': 'YOLO Studio', 'task': 'detect', 'classes': ['object'],
                'data_yaml': 'config.yaml'},
    'training': {'model': 'yolov8n.pt'},
    'predict': {'conf': 0.25},
    'export': {'format': 'onnx'},
}


def _load_raw():
    """Load project.yaml as the single source of truth."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding='utf-8') as f:
            return yaml.safe_load(f) or _FALLBACK
    return _FALLBACK


# ── Singleton config ──
cfg = _load_raw()

# ── Override classes from datasets/config.yaml (single source of truth for class names) ──
_DATAYAML_PATH = ROOT / cfg['project'].get('data_yaml', 'config.yaml')
if _DATAYAML_PATH.exists():
    try:
        with open(_DATAYAML_PATH, encoding='utf-8') as f:
            dy = yaml.safe_load(f) or {}
        dy_names = dy.get('names', {})
        if dy_names:
            sorted_keys = sorted(dy_names, key=lambda k: int(k) if isinstance(k, str) else k)
            cfg['project']['classes'] = [dy_names[k] for k in sorted_keys]
            cfg['project']['names'] = {i: n for i, n in enumerate(cfg['project']['classes'])}
    except Exception:
        pass  # fall back to project.yaml values

# ── Ensure classes always exists ──
if 'classes' not in cfg['project']:
    cfg['project']['classes'] = ['object']
if 'names' not in cfg['project']:
    cfg['project']['names'] = {0: 'object'}
