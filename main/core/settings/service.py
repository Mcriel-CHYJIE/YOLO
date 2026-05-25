# =============================================================================
# YOLO Training Studio — 设置页业务逻辑服务
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""设置页业务逻辑 — 类别名称、快捷键、主题的 纯数据操作（无 Qt 依赖）"""

import json
import yaml
from pathlib import Path
from main.config import THEME_FILE, SHORTCUTS_FILE, PATHS_FILE


# ── 路径解析 ──
_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ══════════════════════════════════════════════════════════════
# Config / Data.yaml 操作
# ══════════════════════════════════════════════════════════════

def resolve_data_yaml(rel_path: str = 'data.yaml') -> Path:
    """返回 data.yaml 的绝对路径"""
    return _ROOT / rel_path


def resolve_shortcuts_file() -> Path:
    """返回 shortcuts.json 的绝对路径"""
    return SHORTCUTS_FILE


def resolve_theme_file() -> Path:
    """返回 theme.json 的绝对路径"""
    return THEME_FILE


def load_classes_from_yaml(yaml_path: Path) -> list[str]:
    """
    从 data.yaml 读取类别名称列表。

    data.yaml 中的 names 为 {0: 'person', 1: 'car', ...} 格式，
    按 key 升序返回 ['person', 'car', ...]。
    文件不存在或格式异常时返回空列表。
    """
    if not yaml_path.exists():
        return []
    try:
        with open(yaml_path, encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        names = data.get('names', {})
        if not names:
            return []
        sorted_keys = sorted(names, key=lambda k: int(k) if isinstance(k, str) else k)
        return [names[k] for k in sorted_keys]
    except Exception:
        return []


def save_classes_to_yaml(yaml_path: Path, names: list[str]) -> None:
    """
    将类别名称列表写入 data.yaml。

    写入格式：
      names:
        0: person
        1: car
        ...
      nc: <数目>
    保留 data.yaml 中其他已有字段。
    """
    data = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}

    data['names'] = {i: name for i, name in enumerate(names)}
    data['nc'] = len(names)

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True,
                  default_flow_style=False, sort_keys=False, indent=2)


def reload_cfg(cfg_module_path: str = 'src.config') -> dict:
    """
    重载 config 模块的 cfg 单例，返回更新后的 cfg。
    """
    import importlib
    mod = importlib.import_module(cfg_module_path)
    mod = importlib.reload(mod)
    return mod.cfg


def override_classes_in_cfg(cfg: dict, yaml_path: Path) -> dict:
    """
    用 data.yaml 中的 names 覆盖 cfg['project']['classes']。
    返回被修改的 cfg（原地修改）。
    """
    if not yaml_path.exists():
        return cfg
    try:
        with open(yaml_path, encoding='utf-8') as f:
            dy = yaml.safe_load(f) or {}
        dy_names = dy.get('names', {})
        if dy_names:
            sorted_keys = sorted(dy_names, key=lambda k: int(k) if isinstance(k, str) else k)
            cfg['project']['classes'] = [dy_names[k] for k in sorted_keys]
            cfg['project']['names'] = {i: n for i, n in enumerate(cfg['project']['classes'])}
    except Exception:
        pass
    return cfg


# ══════════════════════════════════════════════════════════════
# Shortcuts 操作
# ══════════════════════════════════════════════════════════════

def load_shortcuts(filepath: Path) -> dict[str, str]:
    """
    从 shortcuts.json 读取快捷键映射。
    文件不存在或解析失败时返回空字典。
    """
    if not filepath.exists():
        return {}
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_shortcuts(filepath: Path, data: dict[str, str]) -> None:
    """将快捷键映射写入 shortcuts.json，自动创建父目录。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
# Theme 操作
# ══════════════════════════════════════════════════════════════

def load_theme_dark(filepath: Path) -> bool:
    """
    从 theme.json 读取深色主题状态。
    文件不存在或解析失败时返回 False。
    """
    if not filepath.exists():
        return False
    try:
        return bool(json.loads(filepath.read_text(encoding='utf-8')).get('dark', False))
    except Exception:
        return False


def save_theme_dark(filepath: Path, dark: bool) -> None:
    """将深色主题状态写入 theme.json，自动创建父目录。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps({'dark': dark}), encoding='utf-8')


# ══════════════════════════════════════════════════════════════
# 跨 Tab 通知
# ══════════════════════════════════════════════════════════════

def notify_label_tab_shortcuts(studio, shortcuts_data: dict[str, str]) -> None:
    """
    通知 LabelTab 刷新快捷键映射。

    参数:
        studio: 主窗口实例（含 _tabs 字典）
        shortcuts_data: 完整的快捷键映射
    """
    label_tab = getattr(studio, '_tabs', {}).get('label')
    if label_tab is None:
        return

    # 更新内部快捷映射
    if hasattr(label_tab, '_shortcut_keys'):
        label_tab._shortcut_keys.clear()
        label_tab._shortcut_keys.update(shortcuts_data)

    # 同步界面输入框
    if hasattr(label_tab, '_shortcut_inputs'):
        for key, w in label_tab._shortcut_inputs.items():
            if key in shortcuts_data:
                w.setText(shortcuts_data[key])


def rebuild_label_tab_classes(studio) -> None:
    """通知 LabelTab 重建类别按钮。"""
    label_tab = getattr(studio, '_tabs', {}).get('label')
    if label_tab and hasattr(label_tab, 'rebuild_class_buttons'):
        label_tab.rebuild_class_buttons()


# ══════════════════════════════════════════════════════════════
# 项目结构初始化
# ══════════════════════════════════════════════════════════════

def init_project_structure(base_path=None) -> list:
    """在指定路径下创建项目目录结构，返回创建的项目列表"""
    from pathlib import Path
    from main.core.base import ROOT
    root = Path(base_path) if base_path else ROOT
    dirs = [
        'original', 'original/after', 'original/before', 'original/label',
        'datasets', 'datasets/images', 'datasets/labels',
        'models', 'runs', 'output',
    ]
    created = []
    for d in dirs:
        p = root / d
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(d)
    dy = root / 'data.yaml'
    if not dy.exists():
        try:
            dy.write_text('names:\n  0: object\nnc: 1\n', encoding='utf-8')
            created.append('data.yaml')
        except:
            pass
    return created


# ══════════════════════════════════════════════════════════════
# 目录配置持久化
# ══════════════════════════════════════════════════════════════

def save_paths(paths: dict) -> None:
    """保存目录配置到 paths.json"""
    PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PATHS_FILE.write_text(json.dumps(paths, ensure_ascii=False, indent=2), 'utf-8')
