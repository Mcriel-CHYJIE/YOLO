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
import os, sys
from pathlib import Path
from ultralytics import settings as _ultra_settings

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = Path(__file__).resolve().parent.parent / 'project.yaml'

# ── JSON 配置文件路径（frozen 时存到 %APPDATA% 避免重装丢失） ──
if getattr(sys, 'frozen', False):
    _CFG_DIR = Path(os.environ.get('APPDATA', ROOT)) / 'YOLO Training Studio'
else:
    _CFG_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = _CFG_DIR / 'settings.json'
PATHS_FILE = _CFG_DIR / 'paths.json'

# ── 工作目录路径配置 ──
_DEFAULT_PATHS = {
    'train_output': '',
    'predict_output': '',
    'dataset_dir': '',
    'preproc_before': '',
    'preproc_after': '',
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
        'preproc_before': 'Preproc input',
        'preproc_after': 'Preproc output',
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
                'data_yaml': 'data.yaml'},
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

# ══════════════════════════════════════════════════════════════
# DATA_YAML — 运行时按需解析（只用 dataset_dir/data.yaml）
# ══════════════════════════════════════════════════════════════

def get_data_yaml() -> str:
    """返回 dataset_dir/data.yaml（用户通过 Settings 配置）"""
    ds = load_paths().get('dataset_dir', '')
    if ds:
        return str(Path(ds) / 'data.yaml')
    return ''

class _DataYamlProxy(str):
    """延迟计算 DATA_YAML，兼容 str 操作"""
    def __new__(cls):
        return str.__new__(cls, '')
    def __str__(self):
        return get_data_yaml()
    def __repr__(self):
        return get_data_yaml()
    def __eq__(self, other):
        return str(self) == str(other) if isinstance(other, (str, _DataYamlProxy)) else NotImplemented

DATA_YAML = _DataYamlProxy()

# ── 从 data.yaml 初始化 classes（如果 data.yaml 已存在） ──
def _load_classes_from_data():
    """加载 data.yaml 中的类别名称到 cfg"""
    p = Path(get_data_yaml())
    if p.exists():
        try:
            with open(p, encoding='utf-8') as f:
                dy = yaml.safe_load(f) or {}
            dy_names = dy.get('names', {})
            if dy_names:
                sk = sorted(dy_names, key=lambda k: int(k) if isinstance(k, str) else k)
                cfg['project']['classes'] = [dy_names[k] for k in sk]
                cfg['project']['names'] = {i: n for i, n in enumerate(cfg['project']['classes'])}
                return
        except Exception:
            pass
    cfg.setdefault('project', {})
    cfg['project'].setdefault('classes', ['object'])
    cfg['project'].setdefault('names', {0: 'object'})

_load_classes_from_data()

