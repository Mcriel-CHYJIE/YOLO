"""
Config Loader — reads project.yaml from project root.

Usage:
    from scripts.tabs.config import cfg
    cfg['project']['name']       # → 'YOLO Training Studio'
    cfg['training']['epochs']    # → 300
    cfg['predict']['conf']       # → 0.25

The loader merges the user's project.yaml over sensible defaults,
so a minimal file only needs to override what differs.
"""

import yaml, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / 'project.yaml'

# ── Full default config ──
# Any field not present in the user's project.yaml falls back to these.
_DEFAULTS = {
    'project': {
        'name': 'YOLO Studio',
        'task': 'detect',
        'classes': ['object'],
        'data_yaml': 'data.yaml',
        'tip': '',
    },
    'training': {
        'model': 'yolo11n.pt',
        'model_options': ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
        'epochs': 300,
        'batch': 32,
        'imgsz': 640,
        'imgsz_options': [416, 512, 640, 800],
        'optimizer': 'AdamW',
        'optimizer_options': ['AdamW', 'Adam', 'SGD'],
        'device': 'auto',
        'scheduler': 'cosine',
        'scheduler_options': ['Cosine', 'Linear'],
        'patience': 50,
        'lr0': 0.001,
        'lrf': 0.01,
        'warmup_epochs': 3,
        'workers': 8,
        'iou': 0.7,
        'close_mosaic': 15,
        'copy_paste': 0.0,
        'degrees': 15.0,
        'multi_scale': False,
    },
    'validation': {
        'imgsz': 640,
        'batch': 32,
        'conf': 0.25,
        'iou': 0.45,
    },
    'distill': {
        'teacher': '',
        'student': 'yolo11n.pt',
        'epochs': 100,
        'batch': 16,
        'lr0': 0.001,
        'alpha': 0.3,
        'patience': 30,
        'imgsz': 640,
    },
    'predict': {
        'conf': 0.25,
        'iou': 0.45,
        'imgsz': 640,
    },
    'export': {
        'format': 'onnx',
        'format_options': ['onnx', 'torchscript', 'ncnn', 'openvino', 'tensorrt', 'tflite', 'edgetpu', 'coreml'],
        'imgsz': 640,
        'imgsz_options': [320, 416, 640, 800, 1280],
        'half': False,
        'int8': False,
        'nms': False,
    },
}


def _deep_merge(base, override):
    """Recursively merge override dict into base dict (modifies base)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def _load_raw():
    """Load project.yaml as a plain Python dict, merged over defaults."""
    cfg = copy.deepcopy(_DEFAULTS)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding='utf-8') as f:
            user = yaml.safe_load(f) or {}
        _deep_merge(cfg, user)
    return cfg


# ── Singleton config ──
cfg = _load_raw()
