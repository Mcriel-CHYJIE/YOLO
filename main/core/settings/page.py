# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""设置页 UI — 加载 .ui 文件，业务逻辑委托给 settings_service"""

from PyQt5 import uic
from main.core.base import *
from main.config import cfg, THEME_FILE, SHORTCUTS_FILE, load_paths
from main.core.settings import service as svc

# ── 路径常量 ──
ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_YAML_REL = cfg['project'].get('data_yaml', 'datasets/config.yaml')
DATA_YAML = Path(str(_DATA_YAML_REL))
if not DATA_YAML.is_absolute():
    DATA_YAML = ROOT / DATA_YAML

DARK_QSS = """\
QMainWindow,QWidget{background:#1e1e1e;}
QStackedWidget{background:#2d2d2d;}
QGroupBox{font-weight:600;font-size:10px;color:#e0e0e0;border:1px solid #3d3d3d;
    border-radius:6px;margin-top:8px;padding:10px 8px 8px;background:#2d2d2d;}
QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 5px;
    background:#2d2d2d;color:#666;}
QPushButton{border-radius:4px;padding:4px 14px;border:1px solid #3d3d3d;
    background:#2d2d2d;color:#e0e0e0;min-height:22px;font-size:11px;}
QPushButton:hover{background:#383838;}
QPushButton#pri{background:#07C160;color:#fff;border:none;padding:5px 18px;min-height:26px;font-size:12px;font-weight:600;border-radius:4px;}
QPushButton#pri:hover{background:#06ad56;}
QPushButton#pri:disabled{background:#a5d6a5;}
QPushButton#danger{background:#ef4444;color:#fff;border:none;padding:5px 18px;min-height:26px;border-radius:4px;}
QPushButton#danger:hover{background:#dc2626;}
QPushButton#danger:disabled{background:#fca5a5;}
QPushButton#sec{background:#2d2d2d;color:#07C160;border:1px solid #07C160;min-height:22px;font-size:11px;}
QPushButton#sec:hover{background:#1a3d2d;}
QPushButton#warn{background:#2d2d2d;color:#f59e0b;border:1px solid #f59e0b;min-height:22px;font-size:11px;}
QPushButton#warn:hover{background:#3d3010;}
QComboBox{border:1px solid #3d3d3d;border-radius:4px;padding:2px 6px;
    background:#2d2d2d;min-height:22px;color:#e0e0e0;font-size:11px;}
QComboBox:focus{border-color:#07C160;}
QComboBox::drop-down{border:none;width:16px;}
QSpinBox,QDoubleSpinBox{border:1px solid #3d3d3d;border-radius:4px;padding:2px 6px;
    background:#2d2d2d;min-height:22px;color:#e0e0e0;font-size:11px;}
QSpinBox:focus,QDoubleSpinBox:focus{border-color:#07C160;}
QSpinBox::up-button,QDoubleSpinBox::up-button{width:0;padding:0;border:none;}
QSpinBox::down-button,QDoubleSpinBox::down-button{width:0;padding:0;border:none;}
QProgressBar{border:none;border-radius:1px;height:3px;background:#3d3d3d;text-align:center;}
QProgressBar::chunk{background:#07C160;border-radius:1px;}
QTextEdit{background:#111;color:#d4d4d4;border:none;border-radius:5px;
    padding:8px;font-family:Consolas,Courier New;font-size:11px;}
QCheckBox{spacing:5px;font-size:11px;color:#e0e0e0;}
QScrollBar:vertical{width:6px;background:transparent;}
QScrollBar::handle:vertical{background:#555;border-radius:3px;min-height:30px;}
QScrollBar::handle:vertical:hover{background:#777;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
QScrollBar:horizontal{height:6px;background:transparent;}
QScrollBar::handle:horizontal{background:#555;border-radius:3px;min-width:30px;}
QScrollBar::handle:horizontal:hover{background:#777;}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{width:0;}
QLabel{color:#e0e0e0;}
QLineEdit{background:#2d2d2d;color:#e0e0e0;border:1px solid #3d3d3d;}
QPlainTextEdit{background:#2d2d2d;color:#e0e0e0;border:1px solid #3d3d3d;}
QTabWidget::pane{background:#2d2d2d;border:1px solid #3d3d3d;}
QTabBar::tab{background:#252526;color:#999;padding:6px 14px;border:1px solid #3d3d3d;}
QTabBar::tab:selected{background:#2d2d2d;color:#e0e0e0;}
"""


class _ToggleSwitch(QPushButton):
    """左右滑动开关按钮（iOS 风格 toggle）—— 纯 UI 组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(44, 22)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet('border:none;background:transparent;padding:0;min-height:0;')

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        checked = self.isChecked()
        r = h // 2
        qp.setBrush(QColor('#07C160') if checked else QColor('#bbbbbb'))
        qp.setPen(Qt.NoPen)
        qp.drawRoundedRect(0, 0, w, h, r, r)
        margin = 2
        knob_sz = h - margin * 2
        knob_x = w - knob_sz - margin if checked else margin
        qp.setBrush(QColor('#ffffff'))
        qp.drawEllipse(knob_x, margin, knob_sz, knob_sz)


class SettingsTab(QWidget):
    """设置页 — 加载 .ui 构建界面，委托业务逻辑给 service 层"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._shortcut_widgets = {}
        self._path_widgets = {}
        self._load_ui()
        self._post_process_ui()
        self._connect_signals()
        self._load()

    # ═══════════════════════════════════════════
    # UI 加载（从 .ui 文件）
    # ═══════════════════════════════════════════

    _DIR_FIELDS = [
        ('train_output', 'Training output'),
        ('predict_output', 'Predict output'),
        ('dataset_dir', 'Dataset dir'),
        ('preproc_dir', 'Preprocess dir'),
        ('label_dir', 'Label dir'),
        ('export_dir', 'Export dir'),
        ('models_dir', 'Models dir'),
    ]

    def _load_ui(self):
        """加载 Qt Designer .ui 文件中的静态布局"""
        ui_path = Path(__file__).resolve().parent / 'settings.ui'
        uic.loadUi(str(ui_path), self)
        # 通过 objectName 获得：classEditor, saveBtn, statusLabel, hintLabel,
        #   shortcutPrev/Next/DelBox/DelImg, shortcutSaveBtn, shortcutStatus,
        #   classShortcutsContainer, ssToggleHost, themeToggleHost,
        #   initBtn, initStatus

    def _post_process_ui(self):
        """替换占位 widget 为真实组件，安装事件过滤器"""
        nav_keys = [('shortcutPrev', 'prev'), ('shortcutNext', 'next'),
                     ('shortcutDelBox', 'delete_box'), ('shortcutDelImg', 'delete_img')]
        for obj_name, key in nav_keys:
            inp = getattr(self, obj_name)
            inp.setProperty('shortcut_key', key)
            inp.installEventFilter(self)
            self._shortcut_widgets[key] = inp

        # ── 替换屏保/主题占位 QWidget 为 _ToggleSwitch ──
        self._ss_toggle = self._replace_widget_with_toggle('ssToggleHost')
        self._theme_toggle = self._replace_widget_with_toggle('themeToggleHost')

        # ── 三列等宽（忽略内容宽度，完全按 stretch 比例） ──
        self.colsLo.setStretch(0, 1)
        self.colsLo.setStretch(1, 1)
        self.colsLo.setStretch(2, 1)
        for g in (self.classGroup, self.shortcutGroup, self.toggleGroup, self.initGroup, self.dirGroup):
            g.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.col3Lo.setStretch(0, 0)   # toggleGroup — 紧凑两行
        self.col3Lo.setStretch(1, 0)   # initGroup — 紧凑
        self.col3Lo.setStretch(2, 1)   # dirGroup — 占满剩余空间

        # ── 创建类快捷键容器（占满剩余空间） ──
        self.classShortcutsContainer = QWidget()
        self.shortcutGroupLo.insertWidget(3, self.classShortcutsContainer, 1)

        # ── 构建目录配置行 ──
        self._build_path_rows()

    def _replace_widget_with_toggle(self, host_name):
        """将 .ui 中的占位 QWidget 替换为 _ToggleSwitch，保持在其原始布局行内"""
        host = getattr(self, host_name)
        parent = host.parent()
        top = parent.layout()
        # 递归查找包含 host 的子布局（因为 host 可能在子 layout 中）
        target_layout = top
        target_idx = top.indexOf(host)
        if target_idx < 0:
            # 不在顶层布局，遍历子布局
            for i in range(top.count()):
                item = top.itemAt(i)
                if item and item.layout():
                    idx = item.layout().indexOf(host)
                    if idx >= 0:
                        target_layout = item.layout()
                        target_idx = idx
                        break
        toggle = _ToggleSwitch(parent)
        target_layout.insertWidget(target_idx, toggle)
        host.setParent(None)
        host.deleteLater()
        return toggle

    def _connect_signals(self):
        """连接信号"""
        self.saveBtn.clicked.connect(self._save)
        self.shortcutSaveBtn.clicked.connect(self._save_shortcuts)
        self._ss_toggle.toggled.connect(self._on_ss_toggle)
        self._theme_toggle.toggled.connect(self._on_theme_toggle)
        self.initBtn.clicked.connect(self._init_project)
        self.pathsSaveBtn.clicked.connect(self._save_paths)

    # ═══════════════════════════════════════════
    # 数据加载（委托 service 层）
    # ═══════════════════════════════════════════

    def _load(self):
        """从 service 层读取类别数据，填充 UI"""
        try:
            classes = svc.load_classes_from_yaml(DATA_YAML)
            if not classes:
                classes = cfg['project'].get('classes', [])
        except Exception:
            classes = cfg['project'].get('classes', [])

        self.classEditor.setPlainText('\n'.join(classes))
        self._rebuild_cls_shortcuts(classes)
        self._load_shortcuts()
        self._load_theme()
        self._load_paths()

    # ═══════════════════════════════════════════
    # 数据保存（委托 service 层）
    # ═══════════════════════════════════════════

    def _save(self):
        """保存类别名称（委托 service 层写入文件 + 重载配置）"""
        text = self.classEditor.toPlainText().strip()
        names = [s.strip() for s in text.split('\n') if s.strip()]
        if not names:
            self.statusLabel.setStyleSheet(f'font-size:10px;color:{RED};')
            self.statusLabel.setText('Error: at least one class name required')
            return

        try:
            # ── 业务逻辑 ──
            svc.save_classes_to_yaml(DATA_YAML, names)
            cfg_mod = svc.reload_cfg('src.config')
            svc.override_classes_in_cfg(cfg_mod, DATA_YAML)

            # ── UI 响应 ──
            import importlib
            import main.core.base as base_mod
            base_mod = importlib.reload(base_mod)
            from main.core.base import TITLE
            self.studio.setWindowTitle(TITLE)

            svc.rebuild_label_tab_classes(self.studio)

            self._rebuild_cls_shortcuts(names)
            self._load_shortcuts()

            self.statusLabel.setStyleSheet(f'font-size:10px;color:{GREEN};')
            self.statusLabel.setText(f'Saved — {len(names)} categories (datasets/config.yaml)')
        except Exception as e:
            self.statusLabel.setStyleSheet(f'font-size:10px;color:{RED};')
            self.statusLabel.setText(f'Save failed: {e}')

    # ═══════════════════════════════════════════
    # 屏保 / 主题
    # ═══════════════════════════════════════════

    def _on_ss_toggle(self, checked):
        """切换屏保开关（纯 UI 状态操作）"""
        if hasattr(self.studio, '_ss_enabled'):
            self.studio._ss_enabled = checked
            if not checked and hasattr(self.studio, '_ss_active'):
                self.studio._ss_active = False
                self.studio._ss_overlay.hide_overlay()

    def _on_theme_toggle(self, checked):
        """切换黑白主题 —— 持久化委托 service，样式刷新为 UI 操作"""
        svc.save_theme_dark(THEME_FILE, checked)

        import importlib
        import main.core.base as base_mod
        base_mod = importlib.reload(base_mod)
        for _n in ('BG','CARD','BORDER','TEXT','TEXT2','TEXT3','PRI','PRI_H',
                   'GREEN','RED','AMBER','CON','CON_T',
                   'SIDE_BG','SIDE_HOVER','SIDE_ACTIVE','TOP_BG','BOT_BG',
                   'BTN_HOVER','SEC_HOVER','WARN_HOVER','SCROLL_H','SCROLL_HH'):
            if hasattr(base_mod, _n):
                globals()[_n] = getattr(base_mod, _n)
        self.studio.setStyleSheet(base_mod.STYLE)
        QApplication.instance().setStyleSheet(DARK_QSS if checked else '')
        self.studio.refresh_theme()
        self._refresh_inline_styles()
        c = base_mod.AMBER if checked else base_mod.GREEN
        self.shortcutStatus.setStyleSheet(f'font-size:9px;color:{c};')
        self.shortcutStatus.setText('需重启应用以完全应用深色主题' if checked else '需重启应用以完全恢复浅色主题')
        QTimer.singleShot(5000, lambda: self.shortcutStatus.setText(''))

    # ═══════════════════════════════════════════
    # 样式刷新（纯 UI）
    # ═══════════════════════════════════════════

    def _refresh_structural_styles(self):
        """刷新顶栏/底栏/侧边栏以匹配当前主题"""
        studio = self.studio
        if hasattr(studio, '_top_bar'):
            studio._top_bar.setStyleSheet(f'background:{TOP_BG};border-bottom:1px solid {BORDER};')
        if hasattr(studio, '_tab_label'):
            studio._tab_label.setStyleSheet(
                f'font-size:12px;font-weight:400;color:{TEXT2};background:transparent;margin-left:2px;')
        if hasattr(studio, '_sidebar'):
            studio._sidebar.setStyleSheet(f'background:{SIDE_BG};')
        if hasattr(studio, '_nav_btns'):
            for btn, _ in studio._nav_btns:
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
        if hasattr(studio, '_bottom_bar'):
            studio._bottom_bar.setStyleSheet(f'background:{BOT_BG};')

    def _refresh_inline_styles(self):
        """重新应用本页所有内联样式以匹配当前主题"""
        for key, w in self._shortcut_widgets.items():
            is_cls = key.startswith('cls_')
            w.setStyleSheet(
                f'QLineEdit{{background:{BG};border:1px solid {BORDER};border-radius:3px;'
                f'color:{TEXT};font-size:10px;'
                f'{"min-width:36px;max-width:36px;" if is_cls else ""}}}'
                f'QLineEdit:focus{{border-color:{PRI};}}')

    # ═══════════════════════════════════════════
    # 快捷键 UI 构建（纯 UI）
    # ═══════════════════════════════════════════

    def _rebuild_cls_shortcuts(self, classes):
        """重建类别快捷键行"""
        container = self.classShortcutsContainer
        lo = container.layout()
        if lo is None:
            lo = QVBoxLayout(container)
            lo.setContentsMargins(0, 0, 0, 0)
            lo.setSpacing(3)
        else:
            while lo.count():
                w = lo.takeAt(0).widget()
                if w: w.deleteLater()

        remove_keys = [k for k in self._shortcut_widgets if k.startswith('cls_')]
        for k in remove_keys:
            self._shortcut_widgets.pop(k, None)

        defaults = ['1','2','3','4','5','6','7','8','9','0',
                    'Q','W','E','R','T','Y','U','I','O','P',
                    'A','S','D','F','G','H','J','K','L','Z',
                    'X','C','V','B','N','M']
        for i, cls_name in enumerate(classes):
            row = QHBoxLayout()
            row.setContentsMargins(4, 2, 4, 2)
            row.setSpacing(4)
            row.setAlignment(Qt.AlignTop)
            lbl = QLabel(cls_name)
            lbl.setStyleSheet(f'font-size:9px;color:{TEXT};')
            inp = QLineEdit()
            inp.setMinimumHeight(22)
            inp.setAlignment(Qt.AlignCenter)
            inp.setReadOnly(True)
            inp.setProperty('shortcut_key', f'cls_{i}')
            inp.setStyleSheet(
                f'QLineEdit{{background:{BG};border:1px solid {BORDER};border-radius:3px;'
                f'color:{TEXT};font-size:10px;min-width:36px;max-width:36px;}}'
                f'QLineEdit:focus{{border-color:{PRI};}}')
            inp.setText(defaults[i] if i < len(defaults) else '')
            inp.installEventFilter(self)
            row.addWidget(lbl, 1)
            row.addWidget(inp)
            lo.addLayout(row)
            self._shortcut_widgets[f'cls_{i}'] = inp

    # ═══════════════════════════════════════════
    # 快捷键读取/写入（委托 service 层）
    # ═══════════════════════════════════════════

    def _load_shortcuts(self):
        """从 service 层读取快捷键映射，填充 UI"""
        data = svc.load_shortcuts(SHORTCUTS_FILE)
        for key, value in data.items():
            if key in self._shortcut_widgets:
                self._shortcut_widgets[key].setText(value)

    def _load_theme(self):
        """从 service 层读取主题状态，恢复 UI toggle"""
        dark = svc.load_theme_dark(THEME_FILE)
        self._theme_toggle.setChecked(dark)

    def _init_project(self):
        """选择目录后创建项目结构，并自动更新目录配置"""
        folder = QFileDialog.getExistingDirectory(self, 'Select Project Root')
        if not folder:
            return
        created = svc.init_project_structure(folder)
        # 自动填充目录配置为所选根目录下的默认子目录
        root = Path(folder)
        subdirs = {
            'train_output': 'runs',
            'predict_output': 'output',
            'dataset_dir': 'datasets',
            'preproc_dir': 'original',
            'label_dir': 'original',
            'export_dir': 'output',
            'models_dir': 'models',
        }
        for key, sub in subdirs.items():
            if key in self._path_widgets:
                self._path_widgets[key].setText(str(root / sub))
        self._save_paths()
        if created:
            self.initStatus.setStyleSheet(f'font-size:10px;color:{GREEN};padding:0;')
            self.initStatus.setText(f'Created ({len(created)}): ' + ', '.join(created))
        else:
            self.initStatus.setStyleSheet(f'font-size:10px;color:{TEXT2};padding:0;')
            self.initStatus.setText('All directories already exist.')
        QTimer.singleShot(5000, lambda: self.initStatus.setText(''))

    def _build_path_rows(self):
        """在 pathsContainer 中构建标签+路径+设置按钮行"""
        lo = self.pathsContainerLo
        for key, label in self._DIR_FIELDS:
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl = QLabel(label)
            lbl.setStyleSheet(f'font-size:9px;color:{TEXT};')
            lbl.setFixedWidth(80)
            inp = QLineEdit()
            inp.setReadOnly(True)
            inp.setMinimumHeight(22)
            inp.setMaximumWidth(220)
            inp.setStyleSheet(
                f'QLineEdit{{background:{BG};border:1px solid {BORDER};'
                f'border-radius:3px;padding:1px 6px;font-size:10px;color:{TEXT3};}}'
                f'QLineEdit:focus{{border-color:{PRI};}}')
            btn = QPushButton('...')
            btn.setFixedSize(30, 24)
            btn.setStyleSheet(
                f'QPushButton{{background:{CARD};border:1px solid {BORDER};'
                f'border-radius:3px;font-size:12px;color:{TEXT};padding:0;}}'
                f'QPushButton:hover{{background:{PRI}20;border-color:{PRI};}}')
            btn.clicked.connect(lambda checked, k=key: self._browse_path(k))
            row.addWidget(lbl)
            row.addWidget(inp)
            row.addWidget(btn)
            lo.addLayout(row)
            self._path_widgets[key] = inp

    def _load_paths(self):
        """从 paths.json 加载目录配置"""
        paths = load_paths()
        for key, inp in self._path_widgets.items():
            inp.setText(paths.get(key, ''))

    def _save_paths(self):
        """收集目录配置并保存到 paths.json"""
        paths = {}
        for key, inp in self._path_widgets.items():
            paths[key] = inp.text()
        svc.save_paths(paths)
        self.pathsStatus.setStyleSheet(f'font-size:9px;color:{GREEN};')
        self.pathsStatus.setText('Directories saved')
        QTimer.singleShot(2000, lambda: self.pathsStatus.setText(''))

    def _browse_path(self, key):
        """弹出目录选择器"""
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory',
            self._path_widgets[key].text() or str(ROOT))
        if folder:
            self._path_widgets[key].setText(folder)

    def _save_shortcuts(self):
        """收集 UI 快捷键数据，委托 service 层写入文件 + 通知其他 tab"""
        try:
            data = {}
            for key, w in self._shortcut_widgets.items():
                data[key] = w.text()

            svc.save_shortcuts(SHORTCUTS_FILE, data)
            svc.notify_label_tab_shortcuts(self.studio, data)

            self.shortcutStatus.setStyleSheet(f'font-size:9px;color:{GREEN};')
            self.shortcutStatus.setText('Shortcuts saved')
            QTimer.singleShot(2000, lambda: self.shortcutStatus.setText(''))
        except Exception as e:
            self.shortcutStatus.setStyleSheet(f'font-size:9px;color:{RED};')
            self.shortcutStatus.setText(f'Save failed: {e}')

    # ═══════════════════════════════════════════
    # 事件处理（纯 UI）
    # ═══════════════════════════════════════════

    def eventFilter(self, obj, event):
        """捕获按键设置快捷键（含冲突检查）"""
        if event.type() == QEvent.MouseButtonPress and isinstance(obj, QLineEdit):
            obj.setFocus()
            obj.selectAll()
            return True
        if event.type() == QEvent.KeyPress and isinstance(obj, QLineEdit):
            key = event.key()
            if key in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta,
                       Qt.Key_CapsLock, Qt.Key_Tab, Qt.Key_Escape, Qt.Key_Return,
                       Qt.Key_Enter, Qt.Key_Backspace):
                return True
            key_map = {
                Qt.Key_Left: 'Left', Qt.Key_Right: 'Right', Qt.Key_Up: 'Up', Qt.Key_Down: 'Down',
                Qt.Key_Delete: 'Delete', Qt.Key_Space: 'Space',
                Qt.Key_F1: 'F1', Qt.Key_F2: 'F2', Qt.Key_F3: 'F3', Qt.Key_F4: 'F4',
                Qt.Key_F5: 'F5', Qt.Key_F6: 'F6', Qt.Key_F7: 'F7', Qt.Key_F8: 'F8',
                Qt.Key_F9: 'F9', Qt.Key_F10: 'F10', Qt.Key_F11: 'F11', Qt.Key_F12: 'F12',
            }
            if key in key_map:
                name = key_map[key]
            elif Qt.Key_0 <= key <= Qt.Key_9:
                name = chr(key)
            elif Qt.Key_A <= key <= Qt.Key_Z:
                name = chr(key)
            else:
                return True
            # 冲突检查：清除冲突键
            for k, w in self._shortcut_widgets.items():
                if w is not obj and w.text() == name:
                    w.clear()
            obj.setText(name)
            return True
        return super().eventFilter(obj, event)
