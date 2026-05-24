# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""训练标签页"""
from main.core.base import *
from main.config import ATTENTION_FILE, load_paths
from PyQt5 import uic
from .service import Trainer
from .service import build_train_config
ATTENTION_DEFAULTS = {'type': 'none'}

PROG_RE = re.compile(r'\d+%\s+\d+/\d+')
HTML_RE = re.compile(r'<[^>]+>')


class TrainTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.trainer = None
        self._build_ui()
        self._init_widgets()
        self._connect()

    # ═══════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════
    def _build_ui(self):
        ui_path = Path(__file__).resolve().parent / 'train.ui'
        uic.loadUi(str(ui_path), self)
        # 按钮样式直接设置（规避全局 STYLE 传播时机问题）
        self.bs1.setStyleSheet(
            "QPushButton{background:#07C160;color:#fff;border:none;padding:5px 18px;min-height:26px;font-size:12px;font-weight:600;border-radius:4px;}QPushButton:hover{background:#06ad56;}QPushButton:disabled{background:#a5d6a5;}"
        )
        self.bs2.setStyleSheet(
            "QPushButton{background:#ef4444;color:#fff;border:none;padding:5px 18px;min-height:26px;border-radius:4px;}QPushButton:hover{background:#dc2626;}QPushButton:disabled{background:#fca5a5;}"
        )
        self.bs2.setEnabled(False)
        if hasattr(self, 'configGrid'):
            self.configGrid.setColumnStretch(0, 0)
            self.configGrid.setColumnStretch(1, 1)
            self.configGrid.setColumnStretch(2, 0)
            self.configGrid.setColumnStretch(3, 1)
        if hasattr(self, 'algoGrid'):
            self.algoGrid.setColumnStretch(0, 0)
            self.algoGrid.setColumnStretch(1, 1)
            self.algoGrid.setColumnStretch(2, 0)
            self.algoGrid.setColumnStretch(3, 1)

    def _init_widgets(self):
        t = cfg['training']
        self.configGrid.setColumnStretch(0, 0)
        self.configGrid.setColumnStretch(1, 1)
        self.configGrid.setColumnStretch(2, 0)
        self.configGrid.setColumnStretch(3, 1)

        # 模型下拉框：已下载的 .pt + 可选下载列表 + .yaml 架构（从零训练）
        models_dir = Path(load_paths().get('models_dir', str(ROOT / 'models')))
        self.m.clear()
        yaml_archs = ['yolov8n.yaml', 'yolo11n.yaml']
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

        # MetricCard 占位替换
        for placeholder, label, color, default, attr_val, attr_card in [
            (self.epochCard, 'Epoch', TEXT, '0', '_me', '_me_card'),
            (self.mapCard, 'mAP@0.5', GREEN, '—', '_mm', '_mm_card'),
            (self.bestCard, 'Best', PRI, '—', '_mb', '_mb_card'),
        ]:
            card = MetricCard(label, color, default)
            idx = self.metricRow.indexOf(placeholder)
            self.metricRow.removeWidget(placeholder)
            placeholder.deleteLater()
            self.metricRow.insertWidget(idx, card, 1)
            setattr(self, attr_val, card.value_label)
            setattr(self, attr_card, card)

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

        # 注意力模块配置
        import json
        attn_cfg = ATTENTION_DEFAULTS.copy()
        if ATTENTION_FILE.exists():
            try:
                attn_cfg.update(json.loads(ATTENTION_FILE.read_text('utf-8')))
            except: pass
        idx = self.attn_type.findText(attn_cfg.get('type', 'none').upper() if attn_cfg.get('type', 'none') != 'none' else 'None')
        self.attn_type.setCurrentIndex(idx if idx >= 0 else 0)

        self._params = [self.m, self.ep, self.bs, self.sz, self.opt, self.dev,
            self.sch, self.pt, self.lr0, self.lrf, self.wu, self.wk,
            self.iou_thresh, self.cm, self.cp, self.dg, self.ms,
            self.momentum, self.wd, self.hsv_h, self.hsv_s,
            self.hsv_v, self.translate, self.scale, self.cls_pw,
            self.attn_type]

    # ═══════════════════════════════════════════
    # Signal Wiring
    # ═══════════════════════════════════════════
    def _connect(self):
        self.bs1.setObjectName('pri')
        self.bs1.clicked.connect(self._s)
        self.bs2.setObjectName('danger')
        self.bs2.setEnabled(False)
        self.bs2.clicked.connect(self._st)
        self.attn_type.currentTextChanged.connect(self._save_attn_config)

    def _save_attn_config(self):
        """将注意力类型保存到 config/attention.json"""
        import json
        t = self.attn_type.currentText().lower()
        ATTENTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        ATTENTION_FILE.write_text(json.dumps({'type': t}, ensure_ascii=False, indent=2), 'utf-8')

    def _config(self):
        return build_train_config(
            [self.m.itemText(i) for i in range(self.m.count())],
            self.m.currentText(), self.studio.gpu_ok, self.studio.cpu_count,
            self.ep.value(), self.bs.value(), self.sz.currentText(), self.opt.currentText(),
            self.dev.currentIndex(), self.sch.currentIndex(),
            self.pt.value(), self.lr0.value(), self.lrf.value(), self.wu.value(), self.wk.value(),
            self.iou_thresh.value(), self.cm.value(), self.cp.value(), self.dg.value(),
            self.ms.isChecked(),
            self.momentum.value(), self.wd.value(), self.hsv_h.value(), self.hsv_s.value(),
            self.hsv_v.value(), self.translate.value(), self.scale.value(), self.cls_pw.value())

    def _s(self):
        if self.trainer and self.trainer.isRunning():
            QMessageBox.warning(self, 'Warning',
                'Training is in progress, please stop the current training first!'); return
        self.trainer = None
        self.bs1.setEnabled(False); self.bs2.setEnabled(True); self._enable_params(False)
        self.log_panel.clear(); self.pg.setValue(0)
        self._me.setText('0'); self._mm.setText('—'); self._mb.setText('—')
        self._lc.upd({}); self._mc.upd({})
        cfg = self._config()
        self._log(f' {cfg["model"]} | {cfg["epochs"]}ep | batch={cfg["batch"]}')
        self.trainer = Trainer(cfg)
        self.trainer.log.connect(self._log)
        self.trainer.status.connect(lambda t, p, b, c, loss: (
            self._me.setText(t.split('/')[0].replace('Epoch ', '')),
            self._mm.setText(f'{c:.4f}' if c > 0 else '—'),
            self._mb.setText(f'{b:.4f}' if b > 0 else '—'),
            self.pg.setValue(int(p * 100)))),
        self.trainer.chart.connect(
            lambda: (self.trainer and (self._lc.upd(self.trainer.history), self._mc.upd(self.trainer.history))))
        self.trainer.done.connect(self._dn)
        self.trainer.start()

    def _st(self):
        if self.trainer and self.trainer.isRunning():
            self.trainer.stop(); self.bs2.setEnabled(False); self.bs2.setText('Stopping…')
            self._log('  Stopping after current epoch…')

    def _dn(self, ok, msg):
        self._log(msg); self.bs1.setEnabled(True); self.bs2.setEnabled(True); self.bs2.setText('Stop')
        self._enable_params(True); self.trainer = None
        if ok: self.pg.setValue(100)

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
