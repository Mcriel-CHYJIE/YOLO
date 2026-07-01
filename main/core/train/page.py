# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""训练标签页"""
from main.core.base import *
from main.config import load_paths
from PyQt5 import uic
from .service import Trainer
from .service import build_train_config

PROG_RE = re.compile(r'\d+%\s+\d+/\d+')
HTML_RE = re.compile(r'<[^>]+>')

# 预设目录
PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / 'profiles'


class TrainTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.trainer = None
        self._build_ui()
        self._build_preset_ui()
        self._init_widgets()
        self._refresh_preset_list()
        self._connect()

    # ═══════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════
    def _build_ui(self):
        ui_path = Path(__file__).resolve().parent / 'train.ui'
        uic.loadUi(str(ui_path), self)
        # ── 标题 ──
        self.titleLabel.setStyleSheet(f'font-size:18px;font-weight:700;color:{TEXT};padding:0;margin:0;')
        self.titleLabel.setFixedHeight(24)
        # 按钮样式直接设置（规避全局 STYLE 传播时机问题）
        self.bs1.setStyleSheet(
            "QPushButton{background:#07C160;color:#fff;border:none;padding:3px 6px;min-height:16px;font-size:10px;font-weight:600;border-radius:3px;}QPushButton:hover{background:#06ad56;}QPushButton:disabled{background:#a5d6a5;}"
        )
        self.bs2.setStyleSheet(
            "QPushButton{background:#ef4444;color:#fff;border:none;padding:3px 6px;min-height:16px;font-size:10px;border-radius:3px;}QPushButton:hover{background:#dc2626;}QPushButton:disabled{background:#fca5a5;}"
        )
        self.bs2.setEnabled(False)
        if not hasattr(self, '_amp'):
            from main.core.base import ToggleSwitch
            self._amp = ToggleSwitch(checked=True)
            self._amp.setFixedWidth(50)
        if not hasattr(self, '_cache'):
            from main.core.base import ToggleSwitch
            self._cache = ToggleSwitch(checked=False)
            self._cache.setFixedWidth(50)
        if not hasattr(self, 'ms'):
            from main.core.base import ToggleSwitch
            self.ms = ToggleSwitch(checked=True)
            self.ms.setFixedWidth(50)

        # ── Loss: configure items/range (widgets from .ui) ──
        self.clsLossCb.addItems(['BCE', 'Focal', 'ASL'])
        self.clsLossCb.setToolTip('Classification loss function')
        self.focalGammaSb.setRange(0.0, 10.0)
        self.focalGammaSb.setSingleStep(0.1)
        self.focalGammaSb.setValue(2.0)
        self.focalGammaSb.setToolTip('Focal γ / ASL γ- (higher = fewer FNs)')
        self.focalAlphaSb.setRange(0.0, 1.0)
        self.focalAlphaSb.setSingleStep(0.05)
        self.focalAlphaSb.setDecimals(2)
        self.focalAlphaSb.setValue(0.75)
        self.focalAlphaSb.setToolTip('Focal α / ASL γ+ (balance pos/neg)')
        self.iouLossCb.addItems(['CIoU', 'WIoU', 'Focal-EIoU'])
        self.iouLossCb.setToolTip('Box regression IoU loss function')

        # ── Add toggle pairs at row 5 of configGrid ──
        _ls_t = 'font-size:9px;color:#78716c;font-weight:500;'
        def _tog_pair(lbl, w):
            c = QWidget()
            lo = QHBoxLayout(c)
            lo.setContentsMargins(0,0,0,0); lo.setSpacing(2)
            l = QLabel(lbl); l.setStyleSheet(_ls_t)
            lo.addWidget(l); lo.addWidget(w, 1)
            return c
        if hasattr(self, '_amp'):
            self.configGrid.addWidget(_tog_pair('AMP', self._amp), 5, 0)
        if hasattr(self, '_cache'):
            self.configGrid.addWidget(_tog_pair('Cache', self._cache), 5, 1)
        if hasattr(self, 'ms'):
            self.configGrid.addWidget(_tog_pair('Multi-Scale', self.ms), 5, 2)

        # ── Column stretch: 4 equal columns ──
        for g in ('configGrid', 'algoGrid', 'lossGrid'):
            if hasattr(self, g):
                grid = getattr(self, g)
                for col in range(4):
                    grid.setColumnMinimumWidth(col, 0)
                    grid.setColumnStretch(col, 1)
                # Force all pair containers to expand to fill cells
                for i in range(grid.count()):
                    item = grid.itemAt(i)
                    if item and item.widget():
                        c = item.widget()
                        c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                        # Unify label and widget widths inside each container
                        lo = c.layout()
                        if lo:
                            for j in range(lo.count()):
                                li = lo.itemAt(j)
                                if li and li.widget():
                                    w = li.widget()
                                    if isinstance(w, QLabel):
                                        w.setFixedWidth(45)
                                    elif isinstance(w, (QComboBox, QDoubleSpinBox, QSpinBox)):
                                        w.setMinimumWidth(50)
        # ── 将 attnGroup 与 LoRA GroupBox 放在同一行 ──
        if hasattr(self, 'attnGroup'):
            idx = self.leftLo.indexOf(self.attnGroup)
            self.leftLo.removeWidget(self.attnGroup)
            self.otherGroup = QGroupBox('LoRA')
            other_lo = QHBoxLayout(self.otherGroup)
            other_lo.setContentsMargins(6, 2, 6, 2)
            other_lo.setSpacing(4)
            _ls = 'font-size:9px;color:#78716c;font-weight:500;'
            self.loraLabel = QLabel('LoRA Rank')
            self.loraLabel.setStyleSheet(_ls)
            self.loraSb = QSpinBox()
            self.loraSb.setRange(0, 32)
            self.loraSb.setValue(0)
            self.loraSb.setToolTip('0 = disable LoRA')
            other_lo.addWidget(self.loraLabel)
            other_lo.addWidget(self.loraSb, 1)

            row = QWidget()
            row_lo = QHBoxLayout(row)
            row_lo.setContentsMargins(0, 0, 0, 0)
            row_lo.setSpacing(4)
            row_lo.addWidget(self.attnGroup, 1)
            row_lo.addWidget(self.otherGroup, 1)

            row_lo.addWidget(self.otherGroup, 1)

            self.leftLo.insertWidget(idx, row)
            self.attnRow = row
            idx_after = self.leftLo.indexOf(row) if hasattr(self, 'leftLo') else self.leftLo.count() - 1


        self.fusionGroup = QGroupBox('Fusion')
        fusion_lo = QHBoxLayout(self.fusionGroup)
        self.fusionGrid = fusion_lo
        fusion_lo.setContentsMargins(6, 2, 6, 2)
        fusion_lo.setSpacing(4)

        self.fusionCb = QComboBox()
        self.fusionCb.addItems(['None', 'ASFF', 'BiFPN'])
        self.fusionCb.setToolTip('Multi-scale feature fusion')
        lbl_fusion = QLabel('Neck')
        lbl_fusion.setStyleSheet(_ls)
        fusion_lo.addWidget(lbl_fusion)
        fusion_lo.addWidget(self.fusionCb, 1)
        self.attnRow.layout().addWidget(self.fusionGroup, 1)


        # ── 将指标从左侧 controlGroup 移到右侧日志旁 ──
        # 1. 移除 controlGroup 中的 metricRow
        if hasattr(self, 'metricRow'):
            while self.metricRow.count():
                item = self.metricRow.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.controlLo.removeItem(self.metricRow)

        # 2. control 组件只保留按钮
        if hasattr(self, 'btnRow'):
            self.controlLo.removeItem(self.btnRow)
            self.controlLo.removeWidget(self.pg)
            row = QWidget()
            rlo = QHBoxLayout(row)
            rlo.setContentsMargins(0, 0, 0, 0)
            rlo.setSpacing(4)
            self.bs1.setFixedHeight(22)
            self.bs1.setText('▶ Start')
            rlo.addWidget(self.bs1, 1)
            self.bs2.setFixedHeight(22)
            self.bs2.setText('■ Stop')
            rlo.addWidget(self.bs2, 1)
            self.controlLo.addWidget(row)
            self.controlLo.setContentsMargins(8, 6, 8, 4)
            self.controlLo.setSpacing(4)

        # 2. 创建统计栏（日志区下方，各项水平排列）
        self._statsGroup = QGroupBox('Stats')
        sLo = QHBoxLayout(self._statsGroup)
        sLo.setContentsMargins(8, 12, 8, 6)
        sLo.setSpacing(8)

        _ls = 'font-size:10px;color:#57534e;font-weight:500;'

        def _add_stat(label, attr):
            val = QLabel(f'{label}: —')
            val.setStyleSheet(_ls)
            setattr(self, attr, val)
            sLo.addWidget(val, 1)

        _add_stat('Epoch',  '_me')
        _add_stat('mAP@0.5','_mm')
        _add_stat('Best',   '_mb')
        _add_stat('Loss',   '_ml')
        _add_stat('mAP50:95','_m95')
        _add_stat('Prec',   '_mp')
        _add_stat('Recall', '_mr')

        self._me.setText('Epoch: 0')

        # 删除进度条
        self.pg.deleteLater()
        self.pg = None
        if hasattr(self, 'controlGroup'):
            self.leftLo.removeWidget(self.controlGroup)
            self.controlGroup.setTitle('Control')
            self.controlGroup.setStyleSheet('')
            self.controlLo.setContentsMargins(8, 6, 8, 4)

        # 3. 替换 rightLo 中 log_panel，插入 stats + control 行
        #    统计面板放在日志区上面一行
        lp_idx = self.rightLo.indexOf(self.log_panel)
        self.rightLo.removeWidget(self.log_panel)
        # stats + control 同一行
        row = QWidget()
        rlo = QHBoxLayout(row)
        rlo.setContentsMargins(0, 0, 0, 0)
        rlo.setSpacing(4)
        rlo.addWidget(self._statsGroup, 1)
        if self.controlGroup:
            self.controlGroup.setFixedWidth(200)
            rlo.addWidget(self.controlGroup, 0)
        self.rightLo.insertWidget(lp_idx, row, 0)
        self.rightLo.addWidget(self.log_panel, 1)

        # 调整初始比例：图表区 80%，日志区 20%，统计区自适应
        self.rightLo.setStretch(0, 4)  # chartRow
        self.rightLo.setStretch(2, 1)  # log

        # ── 添加 Precision/Recall + LR 图表 ──
        from main.core.base import PrChart, LrChart
        self._pc = PrChart()
        self._lrc = LrChart()
        # 创建容器
        self.pcContainer = QGroupBox('● P/R')
        pLo = QVBoxLayout(self.pcContainer)
        pLo.setContentsMargins(8, 12, 8, 6); pLo.setSpacing(0)
        pLo.addWidget(self._pc)
        self.lrcContainer = QGroupBox('● LR')
        lrLo = QVBoxLayout(self.lrcContainer)
        lrLo.setContentsMargins(8, 12, 8, 6); lrLo.setSpacing(0)
        lrLo.addWidget(self._lrc)
        # 将 chartRow 从 2 列水平布局改为 2×2 网格
        # 找到 chartRow 在 rightLo 中的索引
        cr_idx = -1
        for i in range(self.rightLo.count()):
            item = self.rightLo.itemAt(i)
            if item and item.layout() == self.chartRow:
                cr_idx = i
                break
        if cr_idx >= 0:
            self.rightLo.takeAt(cr_idx)  # 移除旧 chartRow，不删除 widget
            self.chartRow.deleteLater()  # 只删 layout 对象，widget 保留
        # 新建 2×2 容器
        chart_widget = QWidget()
        cl = QVBoxLayout(chart_widget)
        cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(4)
        r1 = QHBoxLayout(); r1.setSpacing(4)
        r1.addWidget(self.lcContainer, 1)
        r1.addWidget(self.pcContainer, 1)
        cl.addLayout(r1, 1)
        r2 = QHBoxLayout(); r2.setSpacing(4)
        r2.addWidget(self.mcContainer, 1)
        r2.addWidget(self.lrcContainer, 1)
        cl.addLayout(r2, 1)
        self.rightLo.insertWidget(cr_idx if cr_idx >= 0 else 0, chart_widget, 3)

    def _build_preset_ui(self):
        """在 configGroup 上方插入预设选择栏"""
        self.presetGroup = QGroupBox('Preset')
        self.presetGroup.setStyleSheet("QGroupBox{font-size:10px;font-weight:600;padding-top:4px;margin-top:4px;}")
        lo = QHBoxLayout(self.presetGroup)
        lo.setContentsMargins(6, 10, 6, 4)
        lo.setSpacing(4)

        self.presetCombo = QComboBox()
        self.presetCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.presetCombo.setMaxVisibleItems(20)

        self.loadBtn = QPushButton('Load')
        self.saveBtn = QPushButton('Save')
        self.delBtn = QPushButton('Delete')
        for btn in (self.loadBtn, self.saveBtn, self.delBtn):
            btn.setFixedHeight(24)
            btn.setStyleSheet(
                "QPushButton{background:#07C160;color:#fff;border:none;padding:2px 8px;border-radius:3px;font-size:10px;font-weight:500;}"
                "QPushButton:hover{background:#06ad56;}"
            )

        lo.addWidget(self.presetCombo, 1)
        lo.addWidget(self.loadBtn)
        lo.addWidget(self.saveBtn)
        lo.addWidget(self.delBtn)

        # 插入到 leftLo 最顶部（index 0）
        idx = self.leftLo.indexOf(self.configGroup)
        self.leftLo.insertWidget(idx, self.presetGroup)

    def _refresh_preset_list(self):
        """扫描 profiles/ 目录刷新预设列表"""
        current = self.presetCombo.currentText()
        self.presetCombo.clear()
        self.presetCombo.addItem('— Default (project.yaml) —')
        if PRESETS_DIR.exists():
            for f in sorted(PRESETS_DIR.glob('*.yaml')):
                try:
                    import yaml
                    data = yaml.safe_load(f.read_text('utf-8'))
                    name = data.get('name', f.stem)
                    self.presetCombo.addItem(f'{name}  ({f.stem})', f)
                except:
                    self.presetCombo.addItem(f.stem, f)
        # 恢复之前选中的
        idx = self.presetCombo.findText(current)
        if idx >= 0:
            self.presetCombo.setCurrentIndex(idx)

    # ── 预设读取 ──
    def _on_load_click(self):
        """Load 按钮: 先刷新列表再加载当前项"""
        self._refresh_preset_list()
        if self.presetCombo.currentIndex() > 0:
            self._load_preset()

    def _load_preset(self):
        """将选中的预设值写入所有控件"""
        if self.presetCombo.currentIndex() <= 0:
            # "Default" 选项：从 project.yaml 恢复
            self._load_from_cfg()
            return

        f = self.presetCombo.currentData()
        if not f or not f.exists():
            return

        import yaml
        try:
            data = yaml.safe_load(f.read_text('utf-8'))
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to load preset:\n{e}')
            return

        t = data.get('training', {})
        if not t:
            return

        # 逐个控件写入
        self._set_combo_text(self.m, t.get('model', ''))
        self.ep.setValue(int(t.get('epochs', 300)))
        self.bs.setValue(int(t.get('batch', 32)))
        self._set_combo_text(self.sz, str(t.get('imgsz', 640)))
        self._set_combo_text(self.opt, t.get('optimizer', 'AdamW'))
        self._set_combo_text(self.dev, 'GPU' if t.get('device', 'auto') in ('auto', '0', 'GPU') else 'CPU')
        self._set_combo_text(self.sch, t.get('scheduler', 'Cosine').capitalize())
        self.pt.setValue(int(t.get('patience', 40)))
        self.lr0.setValue(float(t.get('lr0', 0.001)))
        self.lrf.setValue(float(t.get('lrf', 0.01)))
        self.wu.setValue(int(t.get('warmup_epochs', 3)))
        self.wk.setValue(int(t.get('workers', 8)))
        self.iou_thresh.setValue(float(t.get('iou', 0.7)))
        self.cm.setValue(int(t.get('close_mosaic', 15)))
        self.cp.setValue(float(t.get('copy_paste', 0)))
        self.dg.setValue(float(t.get('degrees', 0)))
        self.ms.setChecked(bool(t.get('multi_scale', False)))
        self.momentum.setValue(float(t.get('momentum', 0.937)))
        self.wd.setValue(float(t.get('weight_decay', 0.0005)))
        self.hsv_h.setValue(float(t.get('hsv_h', 0.015)))
        self.hsv_s.setValue(float(t.get('hsv_s', 0.7)))
        self.hsv_v.setValue(float(t.get('hsv_v', 0.4)))
        self.translate.setValue(float(t.get('translate', 0.15)))
        self.scale.setValue(float(t.get('scale', 0.6)))
        self.cls_pw.setValue(float(t.get('cls_pw', 0.75)))
        self.mosaic_sb.setValue(float(t.get('mosaic', 1.0)))
        self.mixup_sb.setValue(float(t.get('mixup', 0.2)))
        self._flip_lr.setValue(float(t.get('flip_lr', 0.5)))
        self._flipud.setValue(float(t.get('flipud', 0.0)))
        self._shear.setValue(float(t.get('shear', 0.0)))
        self._persp.setValue(float(t.get('perspective', 0.0)))
        self._dropout.setValue(float(t.get('dropout', 0.0)))
        self._wmom.setValue(float(t.get('warmup_momentum', 0.8)))
        self._amp.setChecked(bool(t.get('amp', True)))
        self._cache.setChecked(bool(t.get('cache', False)))
        self.loraSb.setValue(int(t.get('lora_rank', 0)))

        # ── Loss: map to combo items ──
        cls_loss_val = t.get('cls_loss', 'bce').lower()
        cls_map = {'bce': 'BCE', 'focal': 'Focal', 'asl': 'ASL'}
        self.clsLossCb.setCurrentText(cls_map.get(cls_loss_val, 'BCE'))
        self.focalGammaSb.setValue(float(t.get('focal_gamma', 2.0)))
        self.focalAlphaSb.setValue(float(t.get('focal_alpha', 0.75)))
        iou_loss_val = t.get('iou_loss', 'ciou').lower()
        iou_map = {'ciou': 'CIoU', 'wiou': 'WIoU', 'focaleiou': 'Focal-EIoU'}
        self.iouLossCb.setCurrentText(iou_map.get(iou_loss_val, 'CIoU'))
        fusion_val = t.get('fusion', 'none').lower()
        fusion_map = {'none': 'None', 'asff': 'ASFF', 'bifpn': 'BiFPN'}
        self.fusionCb.setCurrentText(fusion_map.get(fusion_val, 'None'))

        # ASL specific params (from preset if available)
        asl_gp = t.get('asl_gamma_pos', None)
        asl_gn = t.get('asl_gamma_neg', None)
        if asl_gp is not None and self.clsLossCb.currentText() == 'ASL':
            self.focalAlphaSb.setValue(float(asl_gp))
        if asl_gn is not None and self.clsLossCb.currentText() == 'ASL':
            self.focalGammaSb.setValue(float(asl_gn))

        # 注意力模块
        attn = t.get('attention', 'none')
        attn_map = {'none': 'None', 'se': 'SE', 'cbam': 'CBAM', 'ca': 'CA',
                    'eca': 'ECA', 'simam': 'SimAM', 'ema': 'EMA', 'gam': 'GAM'}
        self.attn_type.setCurrentText(attn_map.get(attn.lower(), 'None'))
        # 更新 algoGroup 的 tip（如果有）
        tip = data.get('tip', '')
        if tip and hasattr(self, 'algoGroup'):
            self.algoGroup.setTitle(f'Algorithm  |  {tip}')

        QApplication.processEvents()

        # 日志
        preset_name = f.stem if f else 'Default'
        self.studio.log_operation('Training', f'加载预设 · {preset_name}')

    def _load_from_cfg(self):
        """从 project.yaml 恢复默认值"""
        t = cfg['training']
        self._set_combo_text(self.m, t.get('model', 'yolo11n.pt'))
        self.ep.setValue(t.get('epochs', 500))
        self.bs.setValue(t.get('batch', 32))
        self._set_combo_text(self.sz, str(t.get('imgsz', 640)))
        self._set_combo_text(self.opt, t.get('optimizer', 'AdamW'))
        self._set_combo_text(self.dev, 'GPU' if t.get('device', 'auto') in ('auto', '0', 'GPU') else 'CPU')
        self._set_combo_text(self.sch, t.get('scheduler', 'Cosine').capitalize())
        self.pt.setValue(t.get('patience', 40))
        self.lr0.setValue(t.get('lr0', 0.0005))
        self.lrf.setValue(t.get('lrf', 0.01))
        self.wu.setValue(t.get('warmup_epochs', 10))
        self.wk.setValue(min(t.get('workers', 8), self.studio.cpu_count))
        self.iou_thresh.setValue(t.get('iou', 0.8))
        self.cm.setValue(t.get('close_mosaic', 25))
        self.cp.setValue(t.get('copy_paste', 0))
        self.dg.setValue(t.get('degrees', 0))
        self.ms.setChecked(t.get('multi_scale', False))
        self.momentum.setValue(t.get('momentum', 0.937))
        self.wd.setValue(t.get('weight_decay', 0.0005))
        self.hsv_h.setValue(t.get('hsv_h', 0.015))
        self.hsv_s.setValue(t.get('hsv_s', 0.7))
        self.hsv_v.setValue(t.get('hsv_v', 0.4))
        self.translate.setValue(t.get('translate', 0.15))
        self.scale.setValue(t.get('scale', 0.6))
        self.cls_pw.setValue(t.get('cls_pw', 0.75))
        self.mosaic_sb.setValue(t.get('mosaic', 1.0))
        self.mixup_sb.setValue(t.get('mixup', 0.2))
        self._flip_lr.setValue(t.get('flip_lr', 0.5))
        self._flipud.setValue(t.get('flipud', 0.0))
        self._shear.setValue(t.get('shear', 0.0))
        self._persp.setValue(t.get('perspective', 0.0))
        self._dropout.setValue(t.get('dropout', 0.0))
        self._wmom.setValue(t.get('warmup_momentum', 0.8))
        self._amp.setChecked(t.get('amp', True))
        self._cache.setChecked(t.get('cache', False))
        self.loraSb.setValue(t.get('lora_rank', 0))

        # ── Loss: map to combo items ──
        cls_map = {'bce': 'BCE', 'focal': 'Focal', 'asl': 'ASL'}
        self.clsLossCb.setCurrentText(cls_map.get(t.get('cls_loss', 'bce').lower(), 'BCE'))
        self.focalGammaSb.setValue(float(t.get('focal_gamma', 2.0)))
        self.focalAlphaSb.setValue(float(t.get('focal_alpha', 0.75)))
        iou_map = {'ciou': 'CIoU', 'wiou': 'WIoU', 'focaleiou': 'Focal-EIoU'}
        self.iouLossCb.setCurrentText(iou_map.get(t.get('iou_loss', 'ciou').lower(), 'CIoU'))
        fusion_map = {'none': 'None', 'asff': 'ASFF', 'bifpn': 'BiFPN'}
        self.fusionCb.setCurrentText(fusion_map.get(t.get('fusion', 'none').lower(), 'None'))

        # 注意力模块重置为 None
        self.attn_type.setCurrentIndex(0)

    @staticmethod
    def _set_combo_text(cb, text):
        """安全设置 QComboBox，找不到就不改"""
        idx = cb.findText(text)
        if idx >= 0:
            cb.setCurrentIndex(idx)

    # ── 预设保存 ──
    def _save_preset(self):
        """将当前 UI 值保存为预设"""
        if self.trainer and self.trainer.isRunning():
            QMessageBox.warning(self, 'Warning', 'Cannot save preset during training!')
            return

        # 收集当前 UI 值
        data = self._collect_ui_values()

        # 如果是选中了已有预设则直接覆写，否则弹出命名对话框
        if self.presetCombo.currentIndex() > 0:
            f = self.presetCombo.currentData()
            if f and f.exists():
                name = data.get('name', f.stem)
                ret = QMessageBox.question(
                    self, 'Save Preset',
                    f'Overwrite preset "{name}"?',
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                if ret == QMessageBox.Cancel:
                    return
                if ret == QMessageBox.Yes:
                    self._write_preset_file(f, data)
                    self._refresh_preset_list()
                    # 重新选中
                    idx = self.presetCombo.findText(f'{data["name"]}  ({f.stem})')
                    if idx >= 0:
                        self.presetCombo.setCurrentIndex(idx)
                    self.studio.log_operation('Training', f'保存预设 · {data["name"]}')
                    return

        # 另存为新预设
        self._save_as_preset(data)

    def _save_as_preset(self, data=None):
        """弹出对话框另存为新预设"""
        if data is None:
            data = self._collect_ui_values()

        name, ok = QInputDialog.getText(self, 'Save Preset As', 'Preset name:', text=data.get('name', ''))
        if not ok or not name.strip():
            return
        name = name.strip()

        # 生成文件名：拼音化/英文化的文件名
        import re
        stem = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '-').lower()
        if not stem:
            stem = 'custom'
        f = PRESETS_DIR / f'{stem}.yaml'

        # 如果文件已存在，询问是否覆盖
        if f.exists():
            ret = QMessageBox.question(
                self, 'File Exists',
                f'"{f.name}" already exists. Overwrite?',
                QMessageBox.Yes | QMessageBox.Cancel
            )
            if ret != QMessageBox.Yes:
                return

        data['name'] = name
        self._write_preset_file(f, data)
        self._refresh_preset_list()
        # 选中新预设
        idx = self.presetCombo.findText(f'{name}  ({stem})')
        if idx >= 0:
            self.presetCombo.setCurrentIndex(idx)
        self.studio.log_operation('Training', f'另存预设 · {name}')

    def _collect_ui_values(self):
        """读取当前所有控件值"""
        t = {
            'model': self.m.currentText(),
            'epochs': self.ep.value(),
            'batch': self.bs.value(),
            'imgsz': int(self.sz.currentText()),
            'optimizer': self.opt.currentText(),
            'device': 'GPU' if self.dev.currentIndex() == 0 else 'CPU',
            'scheduler': self.sch.currentText(),
            'patience': self.pt.value(),
            'lr0': self.lr0.value(),
            'lrf': self.lrf.value(),
            'warmup_epochs': self.wu.value(),
            'workers': self.wk.value(),
            'iou': self.iou_thresh.value(),
            'close_mosaic': self.cm.value(),
            'copy_paste': self.cp.value(),
            'degrees': self.dg.value(),
            'multi_scale': self.ms.isChecked(),
            'momentum': self.momentum.value(),
            'weight_decay': self.wd.value(),
            'hsv_h': self.hsv_h.value(),
            'hsv_s': self.hsv_s.value(),
            'hsv_v': self.hsv_v.value(),
            'translate': self.translate.value(),
            'scale': self.scale.value(),
            'cls_pw': self.cls_pw.value(),
            'mosaic': self.mosaic_sb.value(),
            'mixup': self.mixup_sb.value(),
            'flip_lr': self._flip_lr.value(),
            'flipud': self._flipud.value(),
            'shear': self._shear.value(),
            'perspective': self._persp.value(),
            'dropout': self._dropout.value(),
            'warmup_momentum': self._wmom.value(),
            'amp': self._amp.isChecked(),
            'cache': self._cache.isChecked(),
            'lora_rank': self.loraSb.value(),
            'cls_loss': self.clsLossCb.currentText().lower(),
            'focal_gamma': self.focalGammaSb.value(),
            'focal_alpha': self.focalAlphaSb.value(),
            'iou_loss': self.iouLossCb.currentText().lower(),
            'fusion': self.fusionCb.currentText().lower(),
            'attention': self.attn_type.currentText(),
        }
        return {'name': '', 'tip': '', 'training': t}

    def _write_preset_file(self, f, data):
        """写入 YAML 预设文件"""
        import yaml
        try:
            # 确保 training 键顺序友好
            content = {
                'name': data['name'],
                'tip': data.get('tip', ''),
                'training': data['training'],
            }
            f.write_text(yaml.dump(content, allow_unicode=True, default_flow_style=False, sort_keys=False), 'utf-8')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to save preset:\n{e}')

    # ── 预设删除 ──
    def _delete_preset(self):
        if self.presetCombo.currentIndex() <= 0:
            QMessageBox.information(self, 'Info', 'Cannot delete the default preset.')
            return
        f = self.presetCombo.currentData()
        if not f or not f.exists():
            return
        name = self.presetCombo.currentText()
        ret = QMessageBox.question(
            self, 'Delete Preset',
            f'Delete preset "{name}"?',
            QMessageBox.Yes | QMessageBox.Cancel
        )
        if ret == QMessageBox.Yes:
            f.unlink()
            self._refresh_preset_list()
            self.studio.log_operation('Training', f'删除预设 · {name}')

    def _init_widgets(self):
        t = cfg['training']

        # 模型下拉框：已下载的 .pt + 可选下载列表 + .yaml 架构（从零训练）
        models_dir = Path(load_paths().get('models_dir', str(ROOT / 'models')))
        self.m.clear()
        yaml_archs = ['yolo11n-p2.yaml', 'yolo11s-p2.yaml', 'yolov8n-p2.yaml', 'yolov8s-p2.yaml']
        for ya in yaml_archs:
            self.m.addItem(ya)
        # 合并已下载的 .pt 和 model_options，去重后显示
        seen = set()
        for f in sorted(models_dir.glob('*.pt')):
            seen.add(f.name)
            self.m.addItem(f.name)
        for mo in t['model_options']:
            if mo not in seen:
                seen.add(mo)
                self.m.addItem(mo)
        if t['model'] in [self.m.itemText(i) for i in range(self.m.count())]:
            self.m.setCurrentText(t['model'])

        self.sz.addItems([str(v) for v in t['imgsz_options']])
        self.sz.setCurrentText(str(t['imgsz']))
        self.opt.addItems(t['optimizer_options'])
        self.opt.setCurrentText(t['optimizer'])
        self.dev.addItems(['GPU', 'CPU'])
        self.sch.addItems(t['scheduler_options'])
        self.sch.setCurrentText(t['scheduler'].capitalize())
        if not self.studio.gpu_ok:
            self.dev.setCurrentIndex(1)
        else:
            self.dev.setCurrentText('GPU' if t['device'] in ('auto', '0', 'GPU') else 'CPU')

        for cb in (self.m, self.sz, self.opt, self.dev, self.sch, self.attn_type):
            from PyQt5.QtWidgets import QListView
            _lv = QListView()
            _lv.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            cb.setView(_lv)
            cb.setMaxVisibleItems(10)

        # 无需 MetricCard 创建，_build_ui 中已直接建好 _me/_mm/_mb 标签

        # 各控件从 project.yaml 读取默认值
        self.ep.setValue(t['epochs'])
        self.bs.setValue(t['batch'])
        self.pt.setValue(t['patience'])
        self.lr0.setValue(t['lr0'])
        self.lrf.setValue(t['lrf'])
        self.wu.setValue(t['warmup_epochs'])
        self.wk.setValue(min(t.get('workers', 8), self.studio.cpu_count))
        self.iou_thresh.setValue(t['iou'])
        self.cm.setValue(t['close_mosaic'])
        self.cp.setValue(t.get('copy_paste', 0))
        self.dg.setValue(t.get('degrees', 0))
        self.ms.setChecked(t.get('multi_scale', False))
        self.momentum.setValue(t.get('momentum', 0.937))
        self.wd.setValue(t.get('weight_decay', 0.0005))
        self.hsv_h.setValue(t.get('hsv_h', 0.015))
        self.hsv_s.setValue(t.get('hsv_s', 0.7))
        self.hsv_v.setValue(t.get('hsv_v', 0.4))
        self.translate.setValue(t.get('translate', 0.15))
        self.scale.setValue(t.get('scale', 0.6))
        self.cls_pw.setValue(t.get('cls_pw', 0.75))
        self.mosaic_sb.setValue(t.get('mosaic', 1.0))
        self.mixup_sb.setValue(t.get('mixup', 0.2))
        self._flip_lr.setValue(t.get('flip_lr', 0.5))
        self._flipud.setValue(t.get('flipud', 0.0))
        self._shear.setValue(t.get('shear', 0.0))
        self._persp.setValue(t.get('perspective', 0.0))
        self._dropout.setValue(t.get('dropout', 0.0))
        self._wmom.setValue(t.get('warmup_momentum', 0.8))
        self._amp.setChecked(t.get('amp', True))
        self._cache.setChecked(t.get('cache', False))
        self.loraSb.setValue(t.get('lora_rank', 0))

        self._params = [self.m, self.ep, self.bs, self.sz, self.opt, self.dev,
            self.sch, self.pt, self.lr0, self.lrf, self.wu, self.wk,
            self.iou_thresh, self.cm, self.cp, self.dg, self.ms,
            self.momentum, self.wd, self.hsv_h, self.hsv_s,
            self.hsv_v, self.translate, self.scale, self.cls_pw,
            self.mosaic_sb, self.mixup_sb,
            self._flip_lr, self._flipud, self._shear, self._persp,
            self._dropout, self._wmom, self._amp, self._cache,
            self.attn_type, self.loraSb,
            self.clsLossCb, self.focalGammaSb, self.focalAlphaSb,
            self.iouLossCb, self.fusionCb]


    # Signal Wiring
    # ═══════════════════════════════════════════
    def _connect(self):
        self.bs1.setObjectName('pri')
        self.bs1.clicked.connect(self._s)
        self.bs2.setObjectName('danger')
        self.bs2.setEnabled(False)
        self.bs2.clicked.connect(self._st)

        # 预设信号
        self.loadBtn.clicked.connect(self._on_load_click)
        self.saveBtn.clicked.connect(self._save_preset)
        self.delBtn.clicked.connect(self._delete_preset)
        self.presetCombo.currentIndexChanged.connect(self._load_preset)

        # Loss selection tooltip update
        self.clsLossCb.currentTextChanged.connect(self._update_loss_tooltips)



    def _update_loss_tooltips(self, text):
        """Update gamma/alpha tooltips based on loss type."""
        if text == 'Focal':
            self.focalGammaSb.setToolTip("Focal gamma (higher = focus on hard examples)")
            self.focalAlphaSb.setToolTip("Focal alpha (0.25-0.75, balance pos/neg)")
        elif text == 'ASL':
            self.focalGammaSb.setToolTip("ASL gamma- (higher = fewer FNs)")
            self.focalAlphaSb.setToolTip("ASL gamma+ (higher = fewer FPs)")
        else:
            self.focalGammaSb.setToolTip("Not used for BCE")
            self.focalAlphaSb.setToolTip("Not used for BCE")

    def _config(self):
        _ts = lambda: __import__('datetime').datetime.now().strftime('%H:%M:%S.%f')
        print(f'[{_ts()}] _config: m={self.m.currentText()}', flush=True)
        r = build_train_config(
            [self.m.itemText(i) for i in range(self.m.count())],
            self.m.currentText(), self.studio.gpu_ok, self.studio.cpu_count,
            self.ep.value(), self.bs.value(), self.sz.currentText(), self.opt.currentText(),
            self.dev.currentIndex(), self.sch.currentIndex(),
            self.pt.value(), self.lr0.value(), self.lrf.value(), self.wu.value(), self.wk.value(),
            self.iou_thresh.value(), self.cm.value(), self.cp.value(), self.dg.value(),
            self.ms.isChecked(),
            self.momentum.value(), self.wd.value(), self.hsv_h.value(), self.hsv_s.value(),
            self.hsv_v.value(), self.translate.value(), self.scale.value(), self.cls_pw.value(),
            self.mosaic_sb.value(), self.mixup_sb.value(),
            self._flip_lr.value(), self._flipud.value(), self._shear.value(), self._persp.value(),
            self._dropout.value(), self._wmom.value(),
            self._amp.isChecked(), self._cache.isChecked(),
            self.loraSb.value(),
            self.clsLossCb.currentText().lower(),
            self.focalGammaSb.value(), self.focalAlphaSb.value(),
            # ASL gamma_pos: maps from alpha when ASL, else 0
            self.focalAlphaSb.value() if self.clsLossCb.currentText() == 'ASL' else 0.0,
            # ASL gamma_neg: maps from gamma when ASL, else 4
            self.focalGammaSb.value() if self.clsLossCb.currentText() == 'ASL' else 4.0,
            self.iouLossCb.currentText().lower(),
            self.fusionCb.currentText().lower())
        print(f'[{_ts()}] _config done', flush=True)
        return r

    def _s(self):
        _ts = lambda: __import__('datetime').datetime.now().strftime('%H:%M:%S.%f')
        print(f'[{_ts()}] _s called', flush=True)
        if self.trainer and self.trainer.isRunning():
            QMessageBox.warning(self, 'Warning',
                'Training is in progress, please stop the current training first!'); return
        self.trainer = None
        print(f'[{_ts()}] disabling UI...', flush=True)
        self.bs1.setEnabled(False); self.bs2.setEnabled(True); self._enable_params(False)
        self.log_panel.clear()
        self._me.setText('Epoch: 0'); self._mm.setText('mAP@0.5: —'); self._mb.setText('Best: —')
        self._ml.setText('Loss: —'); self._m95.setText('mAP50:95: —'); self._mp.setText('Prec: —'); self._mr.setText('Recall: —')
        print(f'[{_ts()}] clearing charts...', flush=True)
        try:
            self._lc.upd({})
        except Exception as _e:
            print(f'[{_ts()}] _lc.upd error: {_e}', flush=True)
        try:
            self._mc.upd({})
        except Exception as _e:
            print(f'[{_ts()}] _mc.upd error: {_e}', flush=True)
        try:
            self._pc.upd({})
        except Exception as _e:
            print(f'[{_ts()}] _pc.upd error: {_e}', flush=True)
        try:
            self._lrc.upd({})
        except Exception as _e:
            print(f'[{_ts()}] _lrc.upd error: {_e}', flush=True)
        print(f'[{_ts()}] building config...', flush=True)
        cfg = self._config()
        # 传递预设标识给训练服务，用于输出文件夹命名
        if self.presetCombo.currentIndex() > 0:
            pf = self.presetCombo.currentData()
            if pf:
                cfg['preset_name'] = pf.stem
        self._log(f' {cfg["model"]} | {cfg["epochs"]}ep | batch={cfg["batch"]}')
        print(f'[{_ts()}] creating Trainer...', flush=True)
        self.trainer = Trainer(cfg)
        self.trainer.log.connect(self._log)
        self.trainer.status.connect(self._update_stats)
        self.trainer.chart.connect(
            lambda: (self.trainer and (self._lc.upd(self.trainer.history), self._mc.upd(self.trainer.history),
                                       self._pc.upd(self.trainer.history), self._lrc.upd(self.trainer.history))))
        self.trainer.done.connect(self._dn)
        self.trainer.start()
        self.studio.log_operation('Training', f'训练开始 · {cfg.get("model","?")} | {cfg.get("epochs","?")}ep | batch={cfg.get("batch","?")}')

    def _st(self):
        if self.trainer and self.trainer.isRunning():
            self.trainer.stop(); self.bs2.setEnabled(False); self.bs2.setText('Stopping…')
            self._log('  Stopping after current epoch…')
            self.studio.log_operation('Training', '训练停止中（等待当前 epoch 完成）')

    def _dn(self, ok, msg):
        self._log(msg); self.bs1.setEnabled(True); self.bs2.setEnabled(True); self.bs2.setText('Stop')
        self._enable_params(True); self.trainer = None
        if ok:
            self.studio.log_operation('Training', '训练完成 ✓')
        else:
            self.studio.log_operation('Training', f'训练结束 · {msg}')

    def _update_stats(self, t, p, b, c, loss):
        self._me.setText(f'Epoch: {t.split("/")[0].replace("Epoch ", "")}')
        self._mm.setText(f'mAP@0.5: {c:.4f}' if c > 0 else 'mAP@0.5: —')
        self._mb.setText(f'Best: {b:.4f}' if b > 0 else 'Best: —')
        self._ml.setText(f'Loss: {loss:.4f}')
        # 从 history 读取扩展指标
        if self.trainer:
            h = self.trainer.history
            self._m95.setText(f'mAP50:95: {h.get("mAP50_95", [None])[-1]:.4f}' if h.get('mAP50_95') else 'mAP50:95: —')
            self._mp.setText(f'Prec: {h.get("precision", [None])[-1]:.4f}' if h.get('precision') else 'Prec: —')
            self._mr.setText(f'Recall: {h.get("recall", [None])[-1]:.4f}' if h.get('recall') else 'Recall: —')

    def _enable_params(self, on):
        for w in self._params: w.setEnabled(on)
        if on and not self.studio.gpu_ok: self.dev.setCurrentIndex(1)

    def _log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        is_prog = bool(PROG_RE.search(msg))
        if is_prog and self.log_panel._log_lines:
            last = self.log_panel._log_lines[-1]
            text = HTML_RE.sub('', last)
            if PROG_RE.search(text):
                self.log_panel.replace_last(format_log(ts, msg))
                return
        self.log_panel.append(format_log(ts, msg))
