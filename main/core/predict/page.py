# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""推理标签页 — 融合实时视频检测 + 批量图片预测 + 热力图/特征图可视化"""
from main.core.base import *
from main.config import load_paths
from PyQt5 import uic
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QTextEdit
from .service import Detector
from .service import run_batch_inference


class PredictTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._model_path = None
        self._source_path = None
        self._worker = None
        self._worker_run_id = 0
        self._mode = None
        self._image_results = []
        self._heatmaps = []
        self._featuremaps = []
        self._fm_layers_all = []
        self._current_image_idx = 0
        self._cumulative_stats = {}
        self._seeking = False
        self._build_ui()
        self._init_widgets()
        self._connect()
        self._load_latest()

    def _build_ui(self):
        ui_path = Path(__file__).resolve().parent / 'predict.ui'
        uic.loadUi(str(ui_path), self)
        # ── 标题 ──
        self.titleLabel.setStyleSheet(f'font-size:18px;font-weight:700;color:{TEXT};padding:0;margin:0;')
        self.titleLabel.setFixedHeight(24)
        self.btn_start.setStyleSheet(
            "QPushButton{background:#07C160;color:#fff;border:none;padding:5px 18px;min-height:26px;font-size:12px;font-weight:600;border-radius:4px;}QPushButton:hover{background:#06ad56;}QPushButton:disabled{background:#a5d6a5;}"
        )
        self.btn_stop.setStyleSheet(
            "QPushButton{background:#ef4444;color:#fff;border:none;padding:5px 18px;min-height:26px;border-radius:4px;}QPushButton:hover{background:#dc2626;}QPushButton:disabled{background:#fca5a5;}"
        )
        self.btn_stop.setEnabled(False)

    def _init_widgets(self):
        # 移除 statsRowP 中的空占位控件
        for w in [self.imgCardP, self.detCardP]:
            self.statsRowP.removeWidget(w)
            w.deleteLater()

        # ── 热力图开关（预先创建，供 Export 行和 Control 共用）──
        self.btn_hm = ToggleSwitch(checked=True)
        self.btn_hm.toggled.connect(self._toggle_hm)

        # ── 累计 + 当前帧分左右（样式一致）──
        stats_lo = self.statsGroupP.layout()
        cl_lo = QHBoxLayout()
        cl_lo.setContentsMargins(0, 0, 0, 0); cl_lo.setSpacing(4)
        # lbl_cls 从 stats_lo 中取出，放入 cl_lo 左列
        for i in range(stats_lo.count()):
            item = stats_lo.itemAt(i)
            if item.widget() is self.lbl_cls:
                stats_lo.removeWidget(self.lbl_cls)
                self.lbl_cls.setParent(None)
                break
        self.lbl_cls.setStyleSheet(
            f'font-size:9px;color:{TEXT3};padding:1px 4px;'
            f'background:{BG};border-radius:3px;')
        cl_lo.addWidget(self.lbl_cls, 1)

        self._frame_cls_label = QLabel('—')
        self._frame_cls_label.setWordWrap(True)
        self._frame_cls_label.setStyleSheet(
            f'font-size:9px;color:{TEXT3};padding:1px 4px;'
            f'background:{BG};border-radius:3px;')
        cl_lo.addWidget(self._frame_cls_label, 1)
        stats_lo.addLayout(cl_lo)

        # ── 帧跳转输入 ──
        jump_row = QWidget()
        jump_row.setFixedHeight(22)
        jl = QHBoxLayout(jump_row)
        jl.setContentsMargins(0, 0, 0, 0); jl.setSpacing(4)
        jlbl = QLabel('Frame:')
        jlbl.setStyleSheet(f'font-size:9px;color:{TEXT3};font-weight:500;background:transparent;')
        jlbl.setFixedWidth(36)
        jl.addWidget(jlbl)
        self._spin_frame = QSpinBox()
        self._spin_frame.setRange(0, 999999)
        self._spin_frame.setFixedHeight(18)
        self._spin_frame.setStyleSheet(
            f'QSpinBox{{font-size:9px;padding:0 2px;border:1px solid {BORDER};border-radius:2px;'
            f'background:{CARD};color:{TEXT};}}'
            f'QSpinBox::up-button,QSpinBox::down-button{{width:0px;}}')
        jl.addWidget(self._spin_frame, 1)
        self._btn_frame_go = QPushButton('Go')
        self._btn_frame_go.setFixedHeight(18)
        self._btn_frame_go.setStyleSheet(
            f'QPushButton{{font-size:9px;padding:0 6px;background:{PRI};color:#fff;'
            f'border:none;border-radius:2px;}}'
            f'QPushButton:hover{{background:{PRI_H};}}')
        self._btn_frame_go.clicked.connect(self._on_frame_go)
        # 注意：不连 editingFinished，避免焦点丢失时误触发 seek 到 frame 0
        jl.addWidget(self._btn_frame_go)
        stats_lo.addWidget(jump_row)

        # 让 statistics 组件撑满左侧剩余高度
        self.leftPanel.layout().setStretchFactor(self.statsGroupP, 1)

        # ── 每帧检测详情日志（类 + 置信度）──
        self._details_log = QTextEdit()
        self._details_log.setReadOnly(True)
        self._details_log.setStyleSheet(
            f'QTextEdit{{background:{CON};color:{CON_T};border:none;border-radius:4px;'
            f'padding:2px 6px;font-family:Consolas;font-size:12px;}}')
        self._details_log.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._details_log.document().setMaximumBlockCount(0)
        stats_lo.addWidget(self._details_log, 1)

        self.cb_export = ToggleSwitch(checked=True)
        idx = self.exportRowLo.indexOf(self.exportToggleHolder)
        self.exportRowLo.removeWidget(self.exportToggleHolder)
        self.exportToggleHolder.deleteLater()
        self.exportRowLo.insertWidget(idx, self.cb_export)
        self.cb_export.toggled.connect(lambda on: self.exportPath.setText('') if not on else None)

        # 热力图开关放在 Export 右边
        hm_lbl = QLabel('Heatmap')
        hm_lbl.setStyleSheet(
            f'font-size:9px;color:{TEXT3};font-weight:500;background:transparent;')
        self.exportRowLo.addWidget(hm_lbl)
        self.exportRowLo.addWidget(self.btn_hm)

        # 进度条高度修正在这里
        self.progress_bar_p.setFixedHeight(6)
        self.progress_bar_p.setStyleSheet(
            'QProgressBar{border:none;border-radius:1px;background:%s;text-align:center;}'
            'QProgressBar::chunk{background:%s;border-radius:1px;}' % (BORDER, PRI))

        # ── Pause + Run Batch 放同一行 ──
        ctrl_lo = self.controlGroupP.layout()
        # 找到 btn_pause 和 btn_run 的位置，移除后放入新行
        pause_idx = -1; run_idx = -1
        for i in range(ctrl_lo.count()):
            w = ctrl_lo.itemAt(i).widget()
            if w is self.btn_pause: pause_idx = i
            if w is self.btn_run: run_idx = i
        if pause_idx >= 0 and run_idx >= 0:
            ctrl_lo.removeWidget(self.btn_pause)
            ctrl_lo.removeWidget(self.btn_run)
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)
            self.btn_pause.setStyleSheet('padding:4px 10px;font-size:10px;')
            rl.addWidget(self.btn_pause)
            self.btn_run.setStyleSheet('padding:4px 10px;font-size:10px;')
            rl.addWidget(self.btn_run)
            ctrl_lo.insertWidget(min(pause_idx, run_idx), row)

        # ── 循环播放开关 ──
        loop_row = QWidget()
        lr_lo = QHBoxLayout(loop_row)
        lr_lo.setContentsMargins(0, 0, 0, 0); lr_lo.setSpacing(4)
        loop_lbl = QLabel('Loop')
        loop_lbl.setStyleSheet(f'font-size:9px;color:{TEXT3};font-weight:500;background:transparent;')
        self.cb_loop = ToggleSwitch(checked=False)
        lr_lo.addWidget(loop_lbl)
        lr_lo.addStretch()
        lr_lo.addWidget(self.cb_loop)
        ctrl_lo.addWidget(loop_row)

        # ── 右侧
        self._stack_layout_idx = None
        main_lo = self.layout()
        # stack 现在在 contentLo 嵌套布局中（titleLabel 添加后）
        content_lo = None
        for i in range(main_lo.count()):
            item = main_lo.itemAt(i)
            if item and item.layout() and item.layout().objectName() == 'contentLo':
                content_lo = item.layout()
                break
        if content_lo is None:
            content_lo = main_lo  # fallback
        for i in range(content_lo.count()):
            if content_lo.itemAt(i).widget() is self.stack:
                self._stack_layout_idx = i
                break

        self.preview_container = QWidget()
        pc_lo = QVBoxLayout(self.preview_container)
        pc_lo.setContentsMargins(0, 0, 0, 0)

        content_lo.removeWidget(self.stack)
        self.stack.setParent(self.preview_container)
        pc_lo.addWidget(self.stack, 1)
        content_lo.insertWidget(self._stack_layout_idx, self.preview_container, 1)

        # ── 日志移到预览区下方，右侧加 FPS/Detections/Current ──
        left_lo = self.leftPanel.layout()
        for i in range(left_lo.count()):
            w = left_lo.itemAt(i).widget()
            if w is self.log:
                left_lo.removeWidget(self.log)
                self.log.setParent(None)
                break
        self._status_bar = QWidget()
        self._status_bar.setStyleSheet(
            f'background:{CON};color:{CON_T};border:none;border-radius:5px;'
            f'font-family:Consolas;font-size:11px;')
        self._status_bar.setFixedHeight(24)
        sb_lo = QHBoxLayout(self._status_bar)
        sb_lo.setContentsMargins(8, 0, 8, 0); sb_lo.setSpacing(12)
        self.log.setStyleSheet('background:transparent;color:' + CON_T + ';border:none;padding:0;')
        self.log.setFixedHeight(24)
        sb_lo.addWidget(self.log, 1)

        sb_lo.addStretch()

        self._sb_fps = QLabel('FPS: 0')
        self._sb_fps.setStyleSheet('background:transparent;color:' + CON_T + ';border:none;padding:0;font-weight:600;')
        sb_lo.addWidget(self._sb_fps)

        self._sb_dets = QLabel('Detections: 0')
        self._sb_dets.setStyleSheet('background:transparent;color:' + CON_T + ';border:none;padding:0;font-weight:600;')
        sb_lo.addWidget(self._sb_dets)

        self._sb_curr = QLabel('Current: 0')
        self._sb_curr.setStyleSheet('background:transparent;color:' + CON_T + ';border:none;padding:0;font-weight:600;')
        sb_lo.addWidget(self._sb_curr)

        self._progress_preview = QSlider(Qt.Horizontal)
        self._progress_preview.setFixedHeight(20)
        self._progress_preview.setRange(0, 100)
        self._progress_preview.setValue(0)
        self._progress_preview.setStyleSheet(
            'QSlider{background:transparent;}'
            'QSlider::groove{background:%s;height:6px;border-radius:3px;}'
            'QSlider::handle{background:%s;width:20px;height:20px;margin:-7px 0;border-radius:10px;}'
            'QSlider::sub-page{background:%s;border-radius:3px;}' % (BORDER, PRI, PRI))
        self._progress_preview.sliderPressed.connect(self._on_seek_start)
        self._progress_preview.sliderReleased.connect(self._on_seek_end)
        self._progress_preview.sliderMoved.connect(self._on_seek_moved)
        pc_lo.addWidget(self._progress_preview)

        pc_lo.addWidget(self._status_bar)

        # ── 0: 视频视图 ──
        self.video_view = QLabel('Load model & source then start')
        self.video_view.setAlignment(Qt.AlignCenter)
        self.video_view.setScaledContents(True)
        self.video_view.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:7px;font-size:14px;')
        self.stack.addWidget(self.video_view)

        # ── 1: 单图预览 + 选项卡 ──
        self.preview_page = QWidget()
        pp_lo = QVBoxLayout(self.preview_page)
        pp_lo.setContentsMargins(0, 0, 0, 0); pp_lo.setSpacing(2)

        self.viz_tabs = QTabBar()
        self.viz_tabs.addTab('Detection')
        self.viz_tabs.addTab('Heatmap')
        self.viz_tabs.addTab('Feature Map')
        self.viz_tabs.setExpanding(True)
        self.viz_tabs.setStyleSheet(f"""
            QTabBar{{background:{BG};border:none;font-size:10px;}}
            QTabBar::tab{{
                background:{CARD};color:{TEXT3};border:1px solid {BORDER};
                border-bottom:none;border-top-left-radius:4px;border-top-right-radius:4px;
                padding:3px 12px;min-height:18px;font-weight:500;
            }}
            QTabBar::tab:selected{{background:{CARD};color:{PRI};border-bottom:2px solid {PRI};}}
            QTabBar::tab:hover:!selected{{background:{BTN_HOVER};color:{TEXT};}}
        """)
        self.viz_tabs.currentChanged.connect(self._on_viz_tab)
        pp_lo.addWidget(self.viz_tabs)

        self.preview_stack = QStackedWidget()
        # page 0: Detection / Heatmap
        self.single_view = QLabel('No image to display')
        self.single_view.setAlignment(Qt.AlignCenter)
        self.single_view.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:7px;font-size:14px;')
        self.single_view.setMinimumHeight(350)
        self.preview_stack.addWidget(self.single_view)
        # page 1: 特征图三层预览
        self.fm_scroll = QScrollArea()
        self.fm_scroll.setWidgetResizable(True)
        self.fm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.fm_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.fm_scroll.setStyleSheet(f'QScrollArea{{background:{CON};border:none;border-radius:7px;}}')

        self.fm_container = QWidget()
        fmc_lo = QVBoxLayout(self.fm_container)
        fmc_lo.setContentsMargins(4, 4, 4, 4)
        fmc_lo.setSpacing(6)
        self.fm_labels = []
        for i in range(3):
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            title = QLabel(f'Layer {i+1}')
            title.setStyleSheet(
                f'font-size:9px;color:{TEXT3};font-weight:600;background:{CARD};'
                f'padding:2px 8px;border-radius:3px;')
            title.setFixedHeight(16)
            cl.addWidget(title)
            lbl = QLabel('—')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setScaledContents(True)
            lbl.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:4px;font-size:11px;')
            cl.addWidget(lbl, 1)
            fmc_lo.addWidget(cell, 1)
            self.fm_labels.append((title, lbl))

        fmc_lo.addStretch()
        self.fm_scroll.setWidget(self.fm_container)
        self.preview_stack.addWidget(self.fm_scroll)

        pp_lo.addWidget(self.preview_stack, 1)

        # 导航栏
        nav = QWidget()
        nav.setStyleSheet('background:transparent;')
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(4, 0, 4, 0); nl.setSpacing(8)
        self.btn_prev = QPushButton('◀ Previous')
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._prev_image)
        self.btn_prev.setStyleSheet('padding:4px 10px;font-size:10px;')
        self.btn_next = QPushButton('Next ▶')
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._next_image)
        self.btn_next.setStyleSheet('padding:4px 10px;font-size:10px;')
        self.lbl_image_index = QLabel('0 / 0')
        self.lbl_image_index.setStyleSheet(f'font-size:11px;font-weight:600;color:{TEXT};min-width:80px;')
        self.lbl_image_index.setAlignment(Qt.AlignCenter)
        nl.addWidget(self.btn_prev)
        nl.addWidget(self.lbl_image_index, 1)
        nl.addWidget(self.btn_next)
        pp_lo.addWidget(nav)

        self.stack.addWidget(self.preview_page)

        # ── 2: 日志视图 ──
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            f'QTextEdit{{background:{CON};color:{CON_T};border:none;border-radius:7px;padding:10px 12px;font-family:"Consolas","Courier New",monospace;font-size:13px;}}')
        self.stack.addWidget(self.log_view)

        self.stack.setCurrentIndex(0)

    # ═══════════════════════════════════════════
    # Signal Wiring
    # ═══════════════════════════════════════════
    def _connect(self):
        self.modelBrowseBtn.clicked.connect(self._browse_model)
        self.srcBrowseBtn.clicked.connect(self._browse_src)
        self.btn_start.setObjectName('pri')
        self.btn_start.clicked.connect(self._toggle_start)
        self.btn_stop.setObjectName('danger')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_run)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_run.setObjectName('pri')
        self.btn_run.clicked.connect(self._run_batch)

    def _load_latest(self):
        w = find_latest_best()
        if w: self.lbl_model.setText(f'[{Path(w).name}]'); self._model_path = Path(w)

    def _browse_model(self):
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        p, _ = QFileDialog.getOpenFileName(self, 'Select Model', 'runs', MODEL_FILTER, options=opts)
        if p:
            self._model_path = Path(p)
            self.lbl_model.setText(f'[{self._model_path.name}]')
            self.studio.log_operation('Predict', f'选择模型 · {self._model_path.name}')

    def _browse_src(self):
        menu = QMenu(self)
        va = menu.addAction('Video File'); ia = menu.addAction('Image File'); fa = menu.addAction('Image Folder')
        act = menu.exec_(QCursor.pos())
        if act == va:
            p, _ = QFileDialog.getOpenFileName(self, 'Select Video', '', 'Video Files (*.mp4 *.avi *.mov *.mkv)')
            if p: self.lbl_src.setText(p); self._set_mode(p); self.studio.log_operation('Predict', f'选择源 · 视频 {Path(p).name}')
        elif act == ia:
            p, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.jpg *.png *.jpeg)')
            if p: self.lbl_src.setText(p); self._set_mode(p); self.studio.log_operation('Predict', f'选择源 · 图片 {Path(p).name}')
        elif act == fa:
            d = QFileDialog.getExistingDirectory(self, 'Select Image Folder')
            if d: self.lbl_src.setText(d); self._set_mode(d); self.studio.log_operation('Predict', f'选择源 · 文件夹 {Path(d).name}')

    def _set_mode(self, p):
        if not p: return
        pp = Path(p)
        if pp.is_file() and pp.suffix.lower() in VIDEO_EXTS:
            self._mode = 'video'; self._source_path = pp; self._show_video_controls()
        elif pp.is_file() and pp.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            self._mode = 'image'; self._source_path = pp; self._show_batch_controls()
        elif pp.is_dir():
            self._mode = 'folder'; self._source_path = pp; self._show_batch_controls()

    def _show_video_controls(self):
        self.stack.setCurrentIndex(0)
        for w in [self.btn_start, self.btn_stop, self.btn_pause]: w.setVisible(True)
        self.btn_run.setVisible(False)
        self.progress_bar_p.setValue(0); self.progress_bar_p.setVisible(True)

    def _show_batch_controls(self):
        self.stack.setCurrentIndex(1)
        for w in [self.btn_start, self.btn_stop, self.btn_pause]: w.setVisible(False)
        self.btn_run.setVisible(True)
        self.progress_bar_p.setVisible(False)

    def _toggle_start(self):
        if not self._model_path or not self._model_path.exists():
            QMessageBox.warning(self, 'Warning', 'Please select a model first'); return
        if not self._source_path or not self._source_path.exists():
            QMessageBox.warning(self, 'Warning', 'Please select a video first'); return
        if self._worker and self._worker.isRunning():
            self._worker.toggle_pause(); self.btn_pause.setText('Resume' if self._worker._pause_event.is_set() else 'Pause'); return
        export_path = None
        if self.cb_export.isChecked():
            d = Path(load_paths().get('predict_output', '')); d.mkdir(parents=True, exist_ok=True) if d else None
            fn = str(d / f"detected_{self._source_path.stem}_{datetime.now().strftime('%m%d_%H%M%S')}.mp4")
            export_path, _ = QFileDialog.getSaveFileName(self, 'Save Detected Video', fn, 'Video Files (*.mp4 *.avi)')
            if not export_path: return
            self.exportPath.setText(export_path)
        else:
            self.exportPath.clear()

        self._worker = Detector(
            self._model_path, self._source_path,
            self.sp_conf.value(), self.sp_iou.value(),
            show_heatmap=self.btn_hm.isChecked(),
            loop=self.cb_loop.isChecked())
        self._worker_run_id += 1
        _rid = self._worker_run_id
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.fps_updated.connect(lambda v: self._sb_fps.setText(f'FPS: {v:.0f}'))
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.details_ready.connect(self._on_details)
        self._worker.finished.connect(lambda rid=_rid: self._on_finish(rid))
        self._worker.log_signal.connect(lambda m: self.log.setText(m))
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_pause.setEnabled(True)
        self.btn_pause.setText('Pause')
        self._cumulative_stats.clear()
        self.lbl_cls.setText('Detecting...'); self._details_log.clear()
        self._sb_fps.setText('FPS: 0'); self._sb_dets.setText('Detections: 0'); self._sb_curr.setText('Current: 0')
        self._spin_frame.setValue(0)
        self._worker.export_path = export_path
        self._worker.start()
        self.studio.log_operation('Predict', f'视频检测开始 · {self._source_path.name}')

    def _toggle_pause(self):
        if self._worker and self._worker.isRunning():
            paused = self._worker._pause_event.is_set()
            self._worker.toggle_pause()
            self.btn_pause.setText('Resume' if paused else 'Pause')
            self.studio.log_operation('Predict', f'视频检测已{"暂停" if paused else "恢复"}')

    def _toggle_hm(self, on):
        """视频运行时切换热力图"""
        if self._worker and self._worker.isRunning():
            self._worker.set_heatmap(on)
            self.studio.log_operation('Predict', f'热力图 {"ON" if on else "OFF"}')

    def _stop_run(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop(); self.log.setText('Stopped')
            self.progress_bar_p.setValue(0)
            self.studio.log_operation('Predict', '视频检测已停止')

    def _on_seek_start(self):
        """开始拖拽进度条 — 标记 seeking 屏蔽 _on_frame 覆盖"""
        if self._worker and self._worker.isRunning():
            self._seeking = True

    def _on_seek_moved(self, pct):
        """用户拖拽/点击凹槽时跳转（sliderMoved 仅用户操作触发，不响应 setValue）"""
        if self._worker and self._worker.isRunning() and hasattr(self._worker, 'seek'):
            total = self._worker._total_frames if hasattr(self._worker, '_total_frames') else 0
            if total > 0:
                target = int(pct / 100 * total)
                self._worker.seek(target)

    def _on_seek_end(self):
        """释放进度条 — 结束 seeking"""
        self._seeking = False

    def _on_frame_go(self):
        """输入帧数跳转 — 仅跳转不播放"""
        if self._worker and self._worker.isRunning() and hasattr(self._worker, 'seek'):
            target = self._spin_frame.value()
            if target < 0:
                return
            self._worker.seek(target)
            total = self._worker._total_frames if hasattr(self._worker, '_total_frames') else 0
            if total > 0:
                self._progress_preview.setValue(int(target / total * 100))

    def _on_frame(self, frame, idx, total):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, frame.shape[1], frame.shape[0],
                    frame.strides[0], QImage.Format_RGB888)
        self.video_view.setPixmap(QPixmap.fromImage(qi))
        self.video_view.setStyleSheet('')
        if total > 0:
            pct = int(idx / total * 100)
            self.progress_bar_p.setValue(pct)
            if not self._seeking:
                self._progress_preview.setValue(pct)

    def _on_stats(self, stats):
        for k, v in stats.items():
            self._cumulative_stats[k] = self._cumulative_stats.get(k, 0) + v
        t = sum(self._cumulative_stats.values())
        self._sb_dets.setText(f'Detections: {t}')
        # 当前帧检测标签
        if stats:
            t_f = sum(stats.values())
            self._sb_curr.setText(f'Current: {t_f}')
            fm_rows = []
            for name, count in sorted(stats.items(), key=lambda x: -x[1]):
                pct = count / t_f * 100
                fm_rows.append(
                    f'<tr><td style="padding:1px 4px;white-space:nowrap;">● {name}</td>'
                    f'<td align="right" style="padding:1px 4px;font-weight:600;">{count}</td>'
                    f'<td align="right" style="padding:1px 4px;color:{TEXT3};">{pct:.0f}%</td></tr>')
            self._frame_cls_label.setText(
                '<table style="width:100%%;font-size:9px;color:%s;border-collapse:collapse;">'
                % TEXT + ''.join(fm_rows) + '</table>')
        else:
            self._frame_cls_label.setText('—')
        # 累计统计明细
        if not self._cumulative_stats:
            self.lbl_cls.setText('No detections'); return
        rows = []
        for name, count in sorted(self._cumulative_stats.items(), key=lambda x: -x[1]):
            pct = count / t * 100
            bar = max(1, int(pct * 0.5))
            rows.append(
                f'<tr><td style="padding:1px 4px;white-space:nowrap;">● {name}</td>'
                f'<td align="right" style="padding:1px 4px;font-weight:600;">{count}</td>'
                f'<td align="right" style="padding:1px 4px;color:#78716c;">{pct:.0f}%</td></tr>'
            )
        self.lbl_cls.setText(
            '<table style="width:100%;font-size:9px;color:#1c1917;border-collapse:collapse;">'
            + ''.join(rows) + '</table>')

    def _on_details(self, details):
        if details:
            for line in details:
                self._details_log.append(line)
            self._details_log.verticalScrollBar().setValue(
                self._details_log.verticalScrollBar().maximum())

    def _on_finish(self, rid):
        if rid != self._worker_run_id:
            return
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_pause.setEnabled(False)
        self.btn_start.setText('Restart'); self._worker = None
        self.studio.log_operation('Predict', '视频检测完成 ✓')

    def _run_batch(self):
        w = self._model_path; src = self._source_path
        if not w or not w.exists(): QMessageBox.warning(self, 'Warning', 'Please select a model first'); return
        if not src or not src.exists(): QMessageBox.warning(self, 'Warning', 'Please select images/folder first'); return
        self.log_view.clear(); self.btn_run.setEnabled(False)
        self.log_view.append('Running batch inference...'); QApplication.processEvents()
        self.studio.log_operation('Predict', f'批量检测开始 · {src.name if src.is_file() else src.stem}')

        try:
            result = run_batch_inference(
                w, src, self.sp_conf.value(), self.sp_iou.value(),
                self.studio.gpu_ok, self.cb_export.isChecked())
            self._image_results = result['results']
            self._heatmaps = result.get('heatmaps', [])
            self._featuremaps = result.get('featuremaps', [])
            self._fm_layers_all = result.get('fm_layers_all', [])
            self._current_image_idx = 0
            
            if result['cls_counts']:
                t = sum(result['cls_counts'].values())
                rows = []
                for name, count in sorted(result['cls_counts'].items(), key=lambda x: -x[1]):
                    pct = count / t * 100
                    bar = max(1, int(pct * 0.5))
                    rows.append(f'<tr><td style="padding:1px 4px;white-space:nowrap;">● {name}</td><td align="right" style="padding:1px 4px;font-weight:600;">{count}</td><td align="right" style="padding:1px 4px;color:#78716c;">{pct:.0f}%</td></tr>')
                self.lbl_cls.setText('<table style="width:100%;font-size:9px;color:#1c1917;border-collapse:collapse;">' + ''.join(rows) + '</table>')
            else:
                self.lbl_cls.setText('No detections')
            self.log_view.append(f'Done! {result["total_imgs"]} sources, {result["total_dets"]} detections')
            if result['save_dir']:
                self.log_view.append(f'   Saved: {result["save_dir"]}')
                self.exportPath.setText(result['save_dir'])
            else:
                self.exportPath.clear()
            sv = result.get('saved_views_dir', '')
            if sv:
                self.log_view.append(f'   4-views: {sv}')
            if result['total_imgs'] > 0: self._show_image_preview(0)
            self.studio.log_operation('Predict', f'批量检测完成 · {result["total_imgs"]} 源, {result["total_dets"]} 检测')
        except Exception as e:
            import traceback; traceback.print_exc(); self.log_view.append(f'Error: {e}')
            self.studio.log_operation('Predict', f'批量检测失败 · {e}')
        finally: self.btn_run.setEnabled(True)

    def _show_image_preview(self, idx):
        self._current_image_idx = idx
        self.stack.setCurrentIndex(1)
        # 当前图片检测标签
        r = self._image_results[idx]
        if r.boxes is not None and len(r.boxes):
            from collections import Counter
            cnt = Counter()
            for cid in r.boxes.cls:
                nm = r.names.get(int(cid), f'cls_{int(cid)}')
                cnt[nm] += 1
            t_f = sum(cnt.values())
            fm_rows = []
            for name, count in sorted(cnt.items(), key=lambda x: -x[1]):
                pct = count / t_f * 100
                fm_rows.append(
                    f'<tr><td style="padding:1px 4px;white-space:nowrap;">● {name}</td>'
                    f'<td align="right" style="padding:1px 4px;font-weight:600;">{count}</td>'
                    f'<td align="right" style="padding:1px 4px;color:{TEXT3};">{pct:.0f}%</td></tr>')
            self._frame_cls_label.setText(
                '<table style="width:100%%;font-size:9px;color:%s;border-collapse:collapse;">'
                % TEXT + ''.join(fm_rows) + '</table>')
        else:
            self._frame_cls_label.setText('—')
        self._render_current_tab()

    def _render_current_tab(self):
        idx = self._current_image_idx
        if idx < 0 or idx >= len(self._image_results):
            return
        tab = self.viz_tabs.currentIndex()
        r = self._image_results[idx]

        if tab == 0:  # Detection
            det = r.plot() if hasattr(r, 'plot') else None
            if det is not None:
                self._display_numpy_on_label(det, self.single_view)
                self.preview_stack.setCurrentIndex(0)
            else:
                self.single_view.setText('Detection N/A')
                self.single_view.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:7px;font-size:14px;')
        elif tab == 1:  # Heatmap
            if self._heatmaps and idx < len(self._heatmaps) and self._heatmaps[idx] is not None:
                self._display_numpy_on_label(self._heatmaps[idx], self.single_view)
                self.preview_stack.setCurrentIndex(0)
            else:
                self.single_view.setText('Heatmap —')
                self.single_view.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:7px;font-size:14px;')
                self.preview_stack.setCurrentIndex(0)
        elif tab == 2:  # Feature Map
            layers = self._fm_layers_all[idx] if idx < len(self._fm_layers_all) else []
            if layers:
                self.preview_stack.setCurrentIndex(1)
                for i in range(3):
                    title, lbl = self.fm_labels[i]
                    if i < len(layers):
                        name, grid = layers[i]
                        title.setText(name)
                        h, w = grid.shape[:2]
                        rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
                        qi = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
                        lbl.setPixmap(QPixmap.fromImage(qi))
                        lbl.setStyleSheet('')
                    else:
                        title.setText(f'Layer {i+1}')
                        lbl.setText('—')
                        lbl.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:4px;font-size:11px;')
            else:
                self.preview_stack.setCurrentIndex(1)
                for _, lbl in self.fm_labels:
                    lbl.setText('—')
                    lbl.setStyleSheet(f'background:{CON};color:{TEXT3};border-radius:4px;font-size:11px;')

    def _on_viz_tab(self, tab_idx):
        self._render_current_tab()

    def _update_nav(self, idx):
        self._current_image_idx = idx
        n = len(self._image_results)
        self.lbl_image_index.setText(f'{idx + 1} / {n}')
        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < n - 1)

    def _display_numpy_on_label(self, img, label):
        h, w = img.shape[:2]
        lw = label.width() - 10
        lh = label.height() - 10
        if lw > 0 and lh > 0:
            sc = min(lw / max(w, 1), lh / max(h, 1), 1.0)
            if sc < 1.0:
                img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, img.shape[1], img.shape[0], rgb.strides[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qi))
        label.setStyleSheet('')

    def _prev_image(self):
        if self._current_image_idx > 0: self._show_image_preview(self._current_image_idx - 1)

    def _next_image(self):
        if self._current_image_idx < len(self._image_results) - 1:
            self._show_image_preview(self._current_image_idx + 1)
