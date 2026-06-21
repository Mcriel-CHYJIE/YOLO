# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""标注标签页 UI — 加载 .ui 文件，业务逻辑委托给 label_service"""

import math
from PyQt5 import uic
from main.core.base import *
from . import service as lbsvc
from .service import AutoLabelWorker, ExportWorker

# ═══════════════════════ 常量 ═══════════════════════
CLASS_COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444',
                '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']
CANVAS_SIZE = 640


# ═══════════════════════ AnnotationCanvas ═══════════════════════

class AnnotationCanvas(QWidget):
    """标注画布 — 固定640×640，图片居中按比例缩放（纯 UI 组件）"""

    annotation_changed = pyqtSignal()
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setStyleSheet("background:#1a1a1a;border-radius:8px;")
        self._image = None
        self._img_h = 0
        self._img_w = 0
        self._annotations = []
        self._selected_idx = -1
        self._class_id = 0
        self._drawing = False
        self._drag_start = None
        self._drag_end = None
        self._drag_idx = -1
        self._drag_mode = ""
        self._resize_corner = -1
        self._scale = 1.0
        self._ox = 0
        self._oy = 0
        self._disp_w = 0
        self._disp_h = 0
        self._pixmap = None
        self._zoom = 1.0          # 缩放倍率 (1.0=适应窗口)
        self._pan_x = 0            # 画布平移偏移 (像素)
        self._pan_y = 0
        self._cursor = None
        self._color_timer = QTimer(self)
        self._color_timer.timeout.connect(self._update_drawing_color)
        self._color_index = 0
        self._drawing_colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444',
                                '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']
        self._current_drawing_color = '#10b981'

    @property
    def annotations(self):
        return self._annotations

    @property
    def selected_idx(self):
        return self._selected_idx

    @selected_idx.setter
    def selected_idx(self, val):
        self._selected_idx = val

    def set_image(self, img: np.ndarray):
        self._image = img
        self._img_h, self._img_w = img.shape[:2]
        self._annotations = []
        self._selected_idx = -1
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._calc_display()
        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qi = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qi)
        self.update()

    def set_annotations(self, anns):
        self._annotations = list(anns)
        self._selected_idx = -1
        self.annotation_changed.emit()
        self.update()

    def set_current_class(self, cls_id: int):
        self._class_id = cls_id

    def _calc_display(self):
        if self._image is None:
            return
        cw, ch = self.width(), self.height()
        if cw <= 0 or ch <= 0:
            return
        fit_scale = min(cw / self._img_w, ch / self._img_h)
        self._scale = fit_scale * self._zoom
        self._disp_w = int(self._img_w * self._scale)
        self._disp_h = int(self._img_h * self._scale)
        base_ox = (cw - self._disp_w) // 2
        base_oy = (ch - self._disp_h) // 2
        self._ox = base_ox + self._pan_x
        self._oy = base_oy + self._pan_y

    def _img2can(self, x, y):
        return (x * self._img_w * self._scale + self._ox,
                y * self._img_h * self._scale + self._oy)

    def _can2img(self, cx, cy):
        ix = (cx - self._ox) / (self._scale * self._img_w)
        iy = (cy - self._oy) / (self._scale * self._img_h)
        return ix, iy

    def _ann_rect(self, ann):
        x1, y1 = self._img2can(ann.xc - ann.w / 2, ann.yc - ann.h / 2)
        x2, y2 = self._img2can(ann.xc + ann.w / 2, ann.yc + ann.h / 2)
        return x1, y1, x2, y2

    def _update_drawing_color(self):
        self._color_index = (self._color_index + 1) % len(self._drawing_colors)
        self._current_drawing_color = self._drawing_colors[self._color_index]
        if self._drawing:
            self.update()

    def paintEvent(self, event):
        if self._image is None or self._pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(self._ox, self._oy, self._disp_w, self._disp_h, self._pixmap)
        for i, ann in enumerate(self._annotations):
            self._draw_ann(painter, ann, i == self._selected_idx)
        if self._drawing and self._drag_start and self._drag_end:
            sx, sy = self._drag_start
            ex, ey = self._drag_end
            pen = QPen(QColor(self._current_drawing_color))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(int(min(sx, ex)), int(min(sy, ey)),
                             int(abs(ex - sx)), int(abs(ey - sy)))

    def _draw_ann(self, painter, ann, selected=False):
        x1, y1, x2, y2 = self._ann_rect(ann)
        # 防护：跳过无效框
        import math
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return
        if int(x2 - x1) <= 0 or int(y2 - y1) <= 0:
            return
        color = CLASS_COLORS[ann.class_id % len(CLASS_COLORS)]
        qc = QColor(color)
        painter.setBrush(Qt.NoBrush)
        pen = QPen(qc)
        pen.setWidth(3 if selected else 2)
        if selected:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        label = lbsvc.resolve_class_name(ann.class_id)
        painter.setPen(Qt.NoPen)
        painter.setBrush(qc)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 8
        th = fm.height() + 2
        painter.drawRect(int(x1), int(y1 - th), int(tw), int(th))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(int(x1) + 4, int(y1) - 3, label)
        if selected:
            handles = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for hx, hy in handles:
                painter.setBrush(QColor("#ffffff"))
                painter.setPen(qc)
                painter.drawRect(int(hx) - 3, int(hy) - 3, 7, 7)

    def _in_image(self, cx, cy):
        return (self._ox <= cx <= self._ox + self._disp_w and
                self._oy <= cy <= self._oy + self._disp_h)

    def _hit_test(self, cx, cy):
        for i, ann in enumerate(self._annotations):
            try:
                x1, y1, x2, y2 = self._ann_rect(ann)
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                    continue
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    return i
            except Exception:
                continue
        return -1

    def _hit_handle(self, cx, cy):
        for i, ann in enumerate(self._annotations):
            try:
                x1, y1, x2, y2 = self._ann_rect(ann)
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                    continue
                corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                for j, (hx, hy) in enumerate(corners):
                    if abs(cx - hx) <= 5 and abs(cy - hy) <= 5:
                        return i, j
            except Exception:
                continue
        return -1, -1

    def mousePressEvent(self, event):
        if self._image is None:
            return event.ignore()
        cx, cy = event.x(), event.y()
        # 中键平移 — 仅在缩放 > 1 时有效
        if event.button() == Qt.MidButton and self._zoom > 1.0:
            self._drag_mode = "pan"
            self._drag_start = (cx, cy)
            event.accept()
            return
        if not self._in_image(cx, cy):
            return event.ignore()
        event.accept()
        if event.button() == Qt.LeftButton:
            idx, corner = self._hit_handle(cx, cy)
            if idx >= 0 and corner >= 0:
                self._drag_mode = "resize"
                self._drag_idx = idx
                self._resize_corner = corner
                self._drag_start = (cx, cy)
                return
            idx = self._hit_test(cx, cy)
            if idx >= 0:
                self._selected_idx = idx
                self.annotation_changed.emit()
                self._drag_mode = "move"
                self._drag_idx = idx
                self._drag_start = (cx, cy)
                self.update()
                return
            self._selected_idx = -1
            self._drawing = True
            self._drag_mode = ""
            self._drag_start = (cx, cy)
            self._drag_end = (cx, cy)
            self._color_index = 0
            self._current_drawing_color = self._drawing_colors[0]
            self._color_timer.start(150)
            self.annotation_changed.emit()
            self.update()
        elif event.button() == Qt.RightButton:
            idx = self._hit_test(cx, cy)
            if idx >= 0:
                self._annotations.pop(idx)
                self._selected_idx = -1
                self.annotation_changed.emit()
                self.update()

    def mouseMoveEvent(self, event):
        event.accept()
        cx, cy = event.x(), event.y()
        if self._drawing and self._drag_start:
            self._drag_end = (cx, cy)
            self.update()
            return
        if self._drag_mode == "pan":
            dx = cx - self._drag_start[0]
            dy = cy - self._drag_start[1]
            self._pan_x += dx
            self._pan_y += dy
            self._ox += dx
            self._oy += dy
            self._drag_start = (cx, cy)
            self.update()
            return
        if self._drag_mode == "move" and self._drag_idx >= 0:
            if self._drag_idx >= len(self._annotations):
                self._drag_mode = None; self._drag_idx = -1; return
            dx = (cx - self._drag_start[0]) / (self._scale * self._img_w)
            dy = (cy - self._drag_start[1]) / (self._scale * self._img_h)
            ann = self._annotations[self._drag_idx]
            ann.xc = max(ann.w / 2, min(1 - ann.w / 2, ann.xc + dx))
            ann.yc = max(ann.h / 2, min(1 - ann.h / 2, ann.yc + dy))
            self._drag_start = (cx, cy)
            self.annotation_changed.emit()
            self.update()
            return
        if self._drag_mode == "resize" and self._drag_idx >= 0:
            if self._drag_idx >= len(self._annotations):
                self._drag_mode = None; self._drag_idx = -1; return
            ann = self._annotations[self._drag_idx]
            nx, ny = self._can2img(cx, cy)
            nx = max(0, min(1, nx))
            ny = max(0, min(1, ny))
            x1 = ann.xc - ann.w / 2
            y1 = ann.yc - ann.h / 2
            x2 = ann.xc + ann.w / 2
            y2 = ann.yc + ann.h / 2
            if self._resize_corner == 0:
                x1, y1 = nx, ny
            elif self._resize_corner == 1:
                x2, y1 = nx, ny
            elif self._resize_corner == 2:
                x1, y2 = nx, ny
            elif self._resize_corner == 3:
                x2, y2 = nx, ny
            if x2 > x1 and y2 > y1:
                ann.xc = (x1 + x2) / 2
                ann.yc = (y1 + y2) / 2
                ann.w = x2 - x1
                ann.h = y2 - y1
                self.annotation_changed.emit()
                self.update()
            return
        idx, _ = self._hit_handle(cx, cy)
        new_cursor = None
        if idx >= 0:
            new_cursor = Qt.SizeAllCursor
        elif self._hit_test(cx, cy) >= 0:
            new_cursor = Qt.PointingHandCursor
        else:
            new_cursor = Qt.CrossCursor
        if new_cursor != self._cursor:
            self._cursor = new_cursor
            self.setCursor(new_cursor)

    def mouseReleaseEvent(self, event):
        event.accept()
        if event.button() == Qt.MidButton:
            if self._drag_mode == "pan":
                self._drag_mode = ""
            return
        if event.button() != Qt.LeftButton:
            return
        if self._drawing and self._drag_start and self._drag_end:
            self._drawing = False
            self._color_timer.stop()
            sx, sy = self._drag_start
            ex, ey = self._drag_end
            drag_distance = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            if drag_distance < 5:
                self._drag_start = None
                self._drag_end = None
                self.update()
                return
            nsx, nsy = self._can2img(sx, sy)
            nex, ney = self._can2img(ex, ey)
            x1 = max(0, min(nsx, nex))
            y1 = max(0, min(nsy, ney))
            x2 = min(1, max(nsx, nex))
            y2 = min(1, max(nsy, ney))
            w, h = x2 - x1, y2 - y1
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            if w > 0.005 and h > 0.005:
                self._annotations.append(
                    lbsvc.Annotation(self._class_id, xc, yc, w, h))
                self.annotation_changed.emit()
            self._drag_start = None
            self._drag_end = None
            self.update()
            return
        self._drag_mode = ""
        self._drag_idx = -1

    def resizeEvent(self, event):
        self._calc_display()
        self.update()

    def wheelEvent(self, event):
        """鼠标滚轮缩放 — 以光标位置为中心缩放"""
        if self._image is None:
            return
        cx, cy = event.pos().x(), event.pos().y()
        if not self._in_image(cx, cy):
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = self._zoom * factor
        new_zoom = max(1.0, min(20.0, new_zoom))
        if abs(new_zoom - self._zoom) < 0.001:
            return
        # 缩放前光标下的图像归一化坐标
        ix = (cx - self._ox) / (self._scale * self._img_w)
        iy = (cy - self._oy) / (self._scale * self._img_h)
        self._zoom = new_zoom
        self._calc_display()
        # 调整平移使光标下的图像点保持不动
        new_cx, new_cy = self._img2can(ix, iy)
        self._pan_x += int(cx - new_cx)
        self._pan_y += int(cy - new_cy)
        self._ox += int(cx - new_cx)
        self._oy += int(cy - new_cy)
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        """双击重置缩放为适应窗口"""
        if self._image is None:
            return
        if event.button() == Qt.LeftButton and self._zoom > 1.0:
            self._zoom = 1.0
            self._pan_x = 0
            self._pan_y = 0
            self._calc_display()
            self.update()
            event.accept()

    def delete_selected(self):
        if 0 <= self._selected_idx < len(self._annotations):
            self._annotations.pop(self._selected_idx)
            self._selected_idx = -1
            self.annotation_changed.emit()
            self.update()

    def clear_annotations(self):
        self._annotations.clear()
        self._selected_idx = -1
        self.annotation_changed.emit()
        self.update()

    def clear_image(self):
        self._image = None
        self._pixmap = None
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._annotations.clear()
        self._selected_idx = -1
        self._drawing = False
        self._drag_start = None
        self._drag_end = None
        self._drag_idx = -1
        self._drag_mode = ""
        self._resize_corner = -1
        self.update()


# Workers imported from .service: AutoLabelWorker, ExportWorker


# ═══════════════════════ LabelTab ═══════════════════════

class LabelTab(QWidget):
    """标注标签页 — 加载 .ui 构建界面，委托业务逻辑给 label_service"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._init_state()
        self._load_ui()
        self._post_process_ui()
        self._register_widgets()
        self._connect_signals()
        self._refresh_source_folders()

    def _init_state(self):
        self._image_paths = []
        self._current_idx = 0
        self._annotations = {}           # {str(img_path): [Annotation, ...]}
        self._current_image = None
        self._session_file = None
        self._auto_anns = {}             # {str(img_path): [dict, ...]}
        self._is_auto_mode = False
        self._has_unsaved = False
        self._pending_delete_path = None
        self._worker = None
        self._export_worker = None
        self._class_ids = []
        self._class_btns = []
        self._shortcut_keys = {}
        self._image_cache = {}  # {str(path): np.ndarray} 图像像素缓存

    # ═══════════════ UI LOADING ═══════════════

    def _load_ui(self):
        ui_path = Path(__file__).resolve().parent / 'label.ui'
        uic.loadUi(str(ui_path), self)

    def _post_process_ui(self):
        """替换占位 widget + 初始化动态区域"""
        # ── 标题 ──
        self.titleLabel.setStyleSheet(f'font-size:18px;font-weight:700;color:{TEXT};padding:0;margin:0;')
        self.titleLabel.setFixedHeight(24)

        # ── 替换 canvasPlaceholder 为 AnnotationCanvas ──
        self.canvas = AnnotationCanvas(self.centerPanel)
        self.canvas.annotation_changed.connect(self._on_annotation_changed)
        self.centerLo.addWidget(self.canvas, 0, 0)  # row 0, col 0
        # Remove placeholder
        self.canvasPlaceholder.setParent(None)
        self.canvasPlaceholder.deleteLater()

        # ── Delete confirm styles ──
        self.centerLo.setAlignment(self.deleteConfirm, Qt.AlignBottom)
        self.deleteConfirm.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.deleteConfirm.setMaximumHeight(48)
        self.deleteLo.setStretch(0, 1)   # confirmMsg 填满
        self.deleteLo.setStretch(1, 0)   # confirmYesBtn 紧凑
        self.deleteLo.setStretch(2, 0)   # confirmNoBtn 紧凑
        self.deleteConfirm.setStyleSheet(
            "background:#fff3cd;border:1px solid #ffc107;border-radius:6px;")
        self.confirmMsg.setStyleSheet(
            "font-size:12px;font-weight:500;color:#856404;background:transparent;")
        self.confirmYesBtn.setStyleSheet(
            "QPushButton{background:#dc3545;color:#fff;border:none;border-radius:4px;"
            "padding:6px 16px;font-size:11px;font-weight:600;}"
            "QPushButton:hover{background:#c82333;}")
        self.confirmNoBtn.setStyleSheet(
            "QPushButton{background:#fff;color:#6c757d;border:1px solid #ced4da;"
            "border-radius:4px;padding:6px 16px;font-size:11px;font-weight:500;}"
            "QPushButton:hover{background:#f8f9fa;}")

        # ── exportStatus 放到进度条下方 ──
        self.exportStatus.setParent(None)
        self.exportStatus.setWordWrap(True)
        self.exportStatus.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # 从 exportStatusRow 中移除，插入到 modelLo 中 exportStatusRow 之后
        idx = self.modelLo.indexOf(self.exportStatusRow)
        self.modelLo.insertWidget(idx + 1, self.exportStatus)

        # ── Style the toggle buttons ──
        ss_on = "background:#07C160;color:#fff;"
        ss_off = "background:#f5f5f5;color:#78716c;"
        self.modeManual.setStyleSheet(
            "QPushButton{font-size:10px;font-weight:600;padding:4px 0;border:none;"
            "border-radius:4px 0 0 4px;" + ss_off + "}"
            "QPushButton:checked{" + ss_on + "}"
            "QPushButton:hover:!checked{background:#e7e5e4;}")
        self.modeAuto.setStyleSheet(
            "QPushButton{font-size:10px;font-weight:600;padding:4px 0;border:none;"
            "border-radius:0 4px 4px 0;" + ss_off + "}"
            "QPushButton:checked{" + ss_on + "}"
            "QPushButton:hover:!checked{background:#e7e5e4;}")
        self.modeContainer.setStyleSheet(
            "QFrame{background:transparent;border:1px solid #d4d4d4;border-radius:5px;}")

        # ── Build class buttons ──
        self._build_class_buttons()

        # ── Load shortcut keys ──
        self._shortcut_keys = lbsvc.load_shortcut_keys()

        # ── Set focus policy ──
        self.setFocusPolicy(Qt.StrongFocus)

        # ── Style individual widgets referenced by code ──
        self._apply_widget_styles()

    def _apply_widget_styles(self):
        """应用需匹配当前主题的样式（从 base 模块读取常量）"""
        # ── 左右面板 QGroupBox 固定自然高度，spacer 吸收剩余空间 ──
        for g in (self.sourceGroup, self.modelGroup, self.statsGroup,
                  self.annGroup, self.classGroup):
            g.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Source section labels
        self.countLabel.setStyleSheet(f"font-size:10px;color:{TEXT3};font-weight:500;")
        self.annotatedLabel.setStyleSheet(f"font-size:10px;color:{GREEN};font-weight:600;")
        self.srcPathLabel.setStyleSheet(
            f"font-size:9px;color:{TEXT3};padding:4px 6px;background:{BG};"
            f"border-radius:4px;border:1px solid {BORDER};")
        self.srcPathLabel.setWordWrap(True)
        self.srcPathLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.srcCombo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        from PyQt5.QtWidgets import QListView
        _lv = QListView()
        _lv.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.srcCombo.setView(_lv)
        self.srcCombo.setMaxVisibleItems(10)
        self.filterStatsLabel.setStyleSheet(
            f"font-size:9px;color:{TEXT2};font-weight:500;padding:3px 5px;"
            f"background:{BG};border-radius:3px;border:1px solid {BORDER};")
        self.filterStatsLabel.setWordWrap(True)
        self.filterStatsLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        # Separators
        for sep in (self.sourceSep, self.exportSep, self.classSep3):
            sep.setStyleSheet(f"background:{BORDER};")
        # Stats header
        for hdr in (self.statsHeaderClass, self.statsHeaderCount,
                    self.statsHeaderPct, self.statsHeaderDist):
            hdr.setStyleSheet(f"font-size:9px;color:{TEXT2};font-weight:500;")
        self.statsHeaderClass.setMinimumWidth(60)
        self.statsHeaderCount.setMinimumWidth(40)
        self.statsHeaderPct.setMinimumWidth(36)
        self.statsSummary.setStyleSheet(
            f"font-size:9px;color:{TEXT3};padding:4px 6px;background:{BG};"
            f"border-radius:4px;border:1px solid {BORDER};")
        # Delete/Clear buttons
        common_btn_style = (
            f"QPushButton{{background:{BG};border:1px solid {BORDER};border-radius:5px;"
            f"color:{RED};font-size:10px;font-weight:500;padding:4px 8px;text-align:left;}}"
            f"QPushButton:hover{{background:{PRI}20;border-color:{PRI};}}")
        self.delBtn.setStyleSheet(common_btn_style)
        self.delBtn.setMinimumHeight(30)
        self.clearBtn.setStyleSheet(common_btn_style)
        self.clearBtn.setMinimumHeight(30)
        # Navigation buttons
        for nav_btn in (self.prevBtn, self.nextBtn):
            nav_btn.setMinimumSize(30, 28)
            nav_btn.setStyleSheet(
                f"QPushButton{{background:{BG};border:1px solid {BORDER};border-radius:5px;"
                f"color:{TEXT};font-size:13px;font-weight:600;}}"
                f"QPushButton:hover{{background:{PRI}20;border-color:{PRI};}}"
                f"QPushButton:disabled{{color:{TEXT3};border-color:{BORDER};}}")
        # Source refresh button
        self.srcRefreshBtn.setStyleSheet(
            f"QPushButton{{background:{BG};border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT};font-size:13px;font-weight:500;}}"
            f"QPushButton:hover{{background:{PRI}20;border-color:{PRI};}}"
            f"QPushButton:pressed{{background:{PRI}40;}}")
        # Browse model button
        self.browseModelBtn.setStyleSheet(
            f"QPushButton{{background:{BG};border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT};font-size:13px;font-weight:500;}}"
            f"QPushButton:hover{{background:{PRI}20;border-color:{PRI};}}"
            f"QPushButton:pressed{{background:{PRI}40;}}")
        # Auto button — green
        self.autoBtn.setStyleSheet(
            f"QPushButton{{background:{PRI};color:#fff;border:none;border-radius:4px;"
            f"padding:6px 16px;font-size:11px;font-weight:600;}}"
            f"QPushButton:hover{{background:{PRI_H};}}"
            f"QPushButton:disabled{{background:{PRI};color:#fff;}}")
        # Export button — amber
        self.exportBtn.setStyleSheet(
            f"QPushButton{{background:{AMBER};color:#fff;border:none;border-radius:4px;"
            f"padding:6px 16px;font-size:11px;font-weight:600;}}"
            f"QPushButton:hover{{background:{PRI_H};}}"
            f"QPushButton:disabled{{background:{AMBER}80;color:#ffffffaa;}}")
        # Export status
        self.exportStatus.setStyleSheet(
            f"font-size:8px;color:{TEXT3};padding:2px 4px;background:{BG};"
            f"border-radius:3px;border:1px solid {BORDER};")
        # Config labels
        for lbl in (self.confLabel, self.iouLabel, self.trainLabel, self.valLabel2):
            lbl.setStyleSheet(f"font-size:9px;color:{TEXT2};")
        from PyQt5.QtWidgets import QListView
        _lv = QListView()
        _lv.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.modelCombo.setView(_lv)
        self.modelCombo.setMaxVisibleItems(10)
        self.confLabel.setMinimumWidth(32)
        self.iouLabel.setMinimumWidth(28)
        self.trainLabel.setMinimumWidth(32)
        self.valLabel2.setMinimumWidth(24)
        self.valLabel.setStyleSheet(f"font-size:10px;color:{TEXT3};font-weight:600;")
        self.imgNameLabel.setStyleSheet(f"font-size:10px;font-weight:500;color:{TEXT};")
        self.imgNameLabel.setWordWrap(True)
        self.imgNameLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.annCountLabel.setStyleSheet(f"font-size:9px;color:{TEXT2};")

        # Annotation list
        self.annList.setStyleSheet(
            f"QListWidget{{background:{BG};border:1px solid {BORDER};border-radius:5px;"
            f"font-size:10px;color:{TEXT};padding:3px;min-height:120px;}}"
            f"QListWidget::item{{padding:4px 8px;border-radius:3px;"
            f"border-bottom:1px solid {BORDER};}}"
            f"QListWidget::item:selected{{background:{PRI}25;color:{TEXT};border:none;}}")

        # Center panel
        self.centerPanel.setStyleSheet(
            f"background:#f0f0f0;border:1px solid {BORDER};border-radius:6px;")

    def _register_widgets(self):
        """用 objectName 注册代码中引用的 widget 别名（兼容旧代码引用方式）"""
        # Source section already has objectName access via self.<name>
        pass

    def _connect_signals(self):
        """连接信号"""
        # Source
        self.srcCombo.currentIndexChanged.connect(self._on_folder_selected)
        self.srcRefreshBtn.clicked.connect(self._refresh_source_folders)
        self.filterBtn.clicked.connect(self._random_filter_dataset)

        # Model & Export
        self.modeManual.toggled.connect(self._on_mode_changed)
        self.modeAuto.toggled.connect(self._on_mode_changed)
        self.browseModelBtn.clicked.connect(self._browse_model)
        self.autoBtn.clicked.connect(self._start_auto)
        self.trainRatio.valueChanged.connect(
            lambda v: self.valLabel.setText(f"{100 - v}%"))
        self.exportBtn.clicked.connect(self._export)

        # Actions
        self.delBtn.clicked.connect(self._delete_selected)
        self.clearBtn.clicked.connect(self._clear_annotations)
        self.annList.currentRowChanged.connect(self._on_ann_list_select)

        # Navigation
        self.prevBtn.clicked.connect(lambda: self._navigate(-1))
        self.nextBtn.clicked.connect(lambda: self._navigate(1))
        self.idxInput.valueChanged.connect(self._goto_idx)

        # Delete confirm
        self.confirmYesBtn.clicked.connect(self._confirm_delete_image)
        self.confirmNoBtn.clicked.connect(
            lambda: self.deleteConfirm.setVisible(False))

        # Create button group for mode
        self._mode_group = QButtonGroup()
        self._mode_group.addButton(self.modeManual)
        self._mode_group.addButton(self.modeAuto)
        self._mode_group.setExclusive(True)

    # ═══════════════ CLASS BUTTONS ═══════════════

    def _build_class_buttons(self):
        """重建类别按钮网格"""
        import main.core.base as base
        for btn in self._class_btns:
            self.classBtnsGrid.removeWidget(btn)
            btn.deleteLater()
        self._class_btns.clear()

        # Remove existing widgets from grid
        while self.classBtnsGrid.count():
            item = self.classBtnsGrid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._class_ids = sorted(int(k) for k in base.CLASS_NAMES.keys())

        for i, class_id in enumerate(self._class_ids):
            row, col = i // 2, i % 2
            name = lbsvc.resolve_class_name(class_id)
            btn = QPushButton(name)
            btn.setMinimumHeight(32)
            btn.setCheckable(True)
            btn.setStyleSheet(
                f"QPushButton{{background:{BG};border:1px solid {BORDER};border-radius:5px;"
                f"color:{TEXT};font-size:11px;font-weight:500;padding:4px 8px;text-align:left;}}"
                f"QPushButton:checked{{background:{PRI};color:#fff;border:1px solid {PRI};}}"
                f"QPushButton:hover{{background:{PRI}20;border-color:{PRI};}}")
            btn.clicked.connect(lambda checked, cid=class_id: self._on_class_selected(cid))
            if i == 0:
                btn.setChecked(True)
                self.canvas.set_current_class(class_id)
            self.classBtnsGrid.addWidget(btn, row, col)
            self._class_btns.append(btn)

    def rebuild_class_buttons(self):
        """供外部调用（Settings 保存后），基于最新的 CLASS_NAMES 重建"""
        self._build_class_buttons()

    # ═══════════════ EVENTS ═══════════════

    def _on_class_selected(self, class_id):
        btn_idx = self._class_ids.index(class_id) if class_id in self._class_ids else -1
        for i, btn in enumerate(self._class_btns):
            btn.setChecked(i == btn_idx)
        self.canvas.set_current_class(class_id)
        if self.canvas.selected_idx >= 0:
            self.canvas.annotations[self.canvas.selected_idx].class_id = class_id
            self.canvas.annotation_changed.emit()
            self.canvas.update()

    def _on_mode_changed(self):
        is_auto = self.modeAuto.isChecked()
        self.autoPanel.setEnabled(True)  # 父容器永远启用，子控件各自控制自己的状态
        self.autoBtn.setEnabled(is_auto)
        self.modelCombo.setEnabled(is_auto)
        self.browseModelBtn.setEnabled(is_auto)
        self.alConf.setEnabled(is_auto)
        self.alIou.setEnabled(is_auto)
        self._is_auto_mode = is_auto

    def _on_annotation_changed(self):
        self._has_unsaved = True
        self.saveIndicator.setText("● Unsaved")
        self.saveIndicator.setStyleSheet(f"font-size:10px;color:{AMBER};font-weight:600;")
        self._update_ann_list()
        self._update_stats()
        if getattr(self, '_canvas_save', True):
            self._save_session()

    def _on_ann_list_select(self, row):
        if 0 <= row < len(self.canvas.annotations) and self.canvas.selected_idx != row:
            self.canvas.selected_idx = row
            self.canvas.update()

    def _on_folder_selected(self):
        folder = self.srcCombo.currentText()
        if not folder:
            return
        src = lbsvc.after_root() / folder
        self.srcPathLabel.setText(f" {src}")
        self._load_images(src)
        self.studio.log_operation('Label', f'加载标注文件夹 · {folder} · {len(self._image_paths)} 张图')

    # ═══════════════ IMAGE LOADING ═══════════════

    def _load_images(self, folder: Path):
        self._image_paths = sorted(folder.rglob("*.jpg"))
        self._current_idx = 0
        self._annotations = {}
        self._auto_anns = {}
        self._image_cache.clear()
        self._session_file = folder / "_annotations.json"
        if self._session_file.exists():
            try:
                saved_anns, curr_idx = lbsvc.load_session(self._session_file)
                self._annotations = saved_anns
                if self._image_paths:
                    self._current_idx = min(curr_idx, len(self._image_paths) - 1)
            except Exception as e:
                print(f"Load annotations error: {e}")
        self._update_counts()
        self._update_stats()
        if self._image_paths:
            self._show_image(self._current_idx, save=False)
        else:
            self.countLabel.setText("0 images")

    # ═══════════════ NAVIGATION ═══════════════

    def _navigate(self, delta):
        if not self._image_paths:
            return
        self._save_current_annotations()
        new_idx = max(0, min(len(self._image_paths) - 1, self._current_idx + delta))
        if new_idx != self._current_idx:
            self._current_idx = new_idx
            self._show_image(self._current_idx)

    def _goto_idx(self, val):
        if not self._image_paths:
            return
        idx = val - 1
        if 0 <= idx < len(self._image_paths):
            self._save_current_annotations()
            if idx != self._current_idx:
                self._current_idx = idx
                self._show_image(idx)

    def _show_image(self, idx, save=True):
        if idx < 0 or idx >= len(self._image_paths):
            return
        img_path = self._image_paths[idx]
        key = str(img_path)
        if key in self._image_cache:
            self._current_image = self._image_cache[key]
        else:
            self._current_image = cv2.imread(str(img_path))
            if self._current_image is not None:
                self._image_cache[key] = self._current_image
        if self._current_image is None:
            return
        self._canvas_save = save
        self.canvas.set_image(self._current_image)
        key = str(img_path)
        if key in self._annotations:
            self.canvas.set_annotations(self._annotations[key])
        elif key in self._auto_anns:
            anns = [lbsvc.Annotation.from_dict(d) for d in self._auto_anns[key]]
            self.canvas.set_annotations(anns)
        self._current_idx = idx
        self.idxInput.blockSignals(True)
        self.idxInput.setValue(idx + 1)
        self.idxInput.blockSignals(False)
        self.totalLabel.setText(f"/ {len(self._image_paths)}")
        self.imgNameLabel.setText(f"{img_path.parent.name}/{img_path.name}")
        self._update_ann_list()
        self._update_nav_buttons()

    def keyPressEvent(self, event):
        if hasattr(self, 'deleteConfirm') and self.deleteConfirm.isVisible():
            return event.ignore()

        key = event.key()
        prev_key = self._shortcut_keys.get('prev', 'A')
        next_key = self._shortcut_keys.get('next', 'D')
        delete_box_key = self._shortcut_keys.get('delete_box', 'W')
        delete_img_key = self._shortcut_keys.get('delete_img', 'S')

        key_map = {
            Qt.Key_Left: "Left", Qt.Key_Right: "Right",
            Qt.Key_Delete: "Delete", Qt.Key_Space: "Space",
            Qt.Key_Backspace: "Backspace", Qt.Key_Escape: "Esc",
            Qt.Key_F1: "F1", Qt.Key_F2: "F2", Qt.Key_F3: "F3",
            Qt.Key_F4: "F4", Qt.Key_F5: "F5", Qt.Key_F6: "F6",
            Qt.Key_F7: "F7", Qt.Key_F8: "F8", Qt.Key_F9: "F9",
            Qt.Key_F10: "F10", Qt.Key_F11: "F11", Qt.Key_F12: "F12",
        }
        if key in key_map:
            current_key_name = key_map[key]
        elif Qt.Key_0 <= key <= Qt.Key_9:
            current_key_name = chr(key)
        elif Qt.Key_A <= key <= Qt.Key_Z:
            current_key_name = chr(key)
        else:
            current_key_name = None

        if current_key_name:
            if current_key_name == prev_key:
                self._navigate(-1); return
            elif current_key_name == next_key:
                self._navigate(1); return
            elif current_key_name == delete_box_key:
                self._delete_selected(); return
            elif current_key_name == delete_img_key:
                self._delete_image(); return
            else:
                cls_shortcuts = lbsvc.get_cls_shortcuts()
                if current_key_name in cls_shortcuts:
                    class_id = cls_shortcuts[current_key_name]
                    if 0 <= class_id < len(self._class_ids):
                        self._on_class_selected(self._class_ids[class_id])
                        return
        super().keyPressEvent(event)

    def on_theme_changed(self):
        from main.core.base import CON
        if hasattr(self, 'canvas'):
            self.canvas.setStyleSheet(f'background:{CON};border-radius:8px;')

    def _img_key(self, path):
        return str(path)

    def _save_current_annotations(self):
        if not self._image_paths or self._current_image is None:
            return
        key = str(self._image_paths[self._current_idx])
        if self.canvas.annotations:
            self._annotations[key] = list(self.canvas.annotations)
        elif key in self._annotations:
            del self._annotations[key]

    # ═══════════════ SESSION ═══════════════

    def _save_session(self):
        if not self._session_file:
            return
        self._save_current_annotations()
        result = lbsvc.save_session_file(
            self._session_file, self._current_idx, self._annotations)
        if result["ok"]:
            self._has_unsaved = False
            self.saveIndicator.setText("✓ Saved")
            self.saveIndicator.setStyleSheet(f"font-size:10px;color:{GREEN};font-weight:600;")
        else:
            self.saveIndicator.setText("✕ Save failed")
            self.saveIndicator.setStyleSheet(f"font-size:10px;color:{RED};")

    # ═══════════════ ACTIONS ═══════════════

    def _delete_selected(self):
        self.canvas.delete_selected()
        self._update_stats()

    def _delete_image(self):
        if not self._image_paths or self._current_idx < 0:
            return
        current_path = self._image_paths[self._current_idx]
        boxes = len(self.canvas.annotations)
        self._pending_delete_path = current_path
        self.confirmMsg.setText(
            f"Delete <b>{current_path.name}</b>?"
            f" ({boxes} box{'es' if boxes != 1 else ''})")
        self.deleteConfirm.setVisible(True)
        self.deleteConfirm.raise_()

    def _confirm_delete_image(self):
        self.deleteConfirm.setVisible(False)
        current_path = self._pending_delete_path
        if current_path is None:
            return
        try:
            lbsvc.delete_image_file(current_path)
            lbsvc.delete_label_file(current_path, lbsvc.label_root())
            key = str(current_path)
            self._annotations.pop(key, None)
            self._auto_anns.pop(key, None)
            self._image_paths.pop(self._current_idx)
            if self._current_idx >= len(self._image_paths):
                self._current_idx = max(0, len(self._image_paths) - 1)
            if self._image_paths:
                self._show_image(self._current_idx)
            else:
                self.canvas.clear_image()
                self.countLabel.setText("0 images")
                self.annotatedLabel.setText("0 annotated")
                self.imgNameLabel.setText("—")
                self.annCountLabel.setText("0 boxes")
            self._update_counts()
            self._update_stats()
            self._save_session()
            self.studio.log_operation('Label', f'删除图片 · {current_path.name}')
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", f"Failed to delete image:\n{str(e)}")

    def _clear_annotations(self):
        self.canvas.clear_annotations()
        self._update_stats()

    def _random_filter_dataset(self):
        if not self._image_paths:
            QMessageBox.warning(self, "Warning", "No images to filter")
            return
        total = len(self._image_paths)
        if total < 3:
            QMessageBox.warning(self, "Warning", f"Need at least 3 images (current: {total})")
            return
        remaining = (total + 2) // 3
        deleted = total - remaining
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Random Filter Dataset")
        msg.setText("Are you sure you want to randomly filter the dataset?")
        msg.setInformativeText(
            f"Current: {total} images\nAfter filter: {remaining} images\n"
            f"Will delete: {deleted} images\n\nThis action cannot be undone!")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        if msg.exec_() != QMessageBox.Yes:
            return
        try:
            self._save_current_annotations()
            to_keep, to_delete = lbsvc.random_filter_paths(self._image_paths)
            for img_path in to_delete:
                lbsvc.delete_image_file(img_path)
                lbsvc.delete_label_file(img_path, lbsvc.label_root())
                key = str(img_path)
                self._annotations.pop(key, None)
                self._auto_anns.pop(key, None)
            self._image_paths = sorted(to_keep)
            if self._current_idx >= len(self._image_paths):
                self._current_idx = max(0, len(self._image_paths) - 1)
            if self._image_paths:
                self._show_image(self._current_idx)
            else:
                self.canvas.clear_image()
                self.countLabel.setText("0 images")
                self.annotatedLabel.setText("0 annotated")
                self.imgNameLabel.setText("—")
                self.annCountLabel.setText("0 boxes")
            self._update_counts()
            self._update_stats()
            self._save_session()
            QMessageBox.information(
                self, "Filter Complete",
                f"Dataset filtered successfully!\n\n"
                f"Deleted: {len(to_delete)} images\n"
                f"Remaining: {len(self._image_paths)} images")
            self.studio.log_operation('Label', f'随机筛选 · {total}→{len(self._image_paths)} 张')
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", f"Failed to filter dataset:\n{str(e)}")

    # ═══════════════ UI UPDATES ═══════════════

    def _update_ann_list(self):
        self.annList.blockSignals(True)
        self.annList.clear()
        for i, ann in enumerate(self.canvas.annotations):
            name = lbsvc.resolve_class_name(ann.class_id)
            try:
                self.annList.addItem(
                    f"[{i}] {name}  ({ann.xc:.3f}, {ann.yc:.3f})  {ann.w:.3f}x{ann.h:.3f}")
            except Exception:
                self.annList.addItem(f"[{i}] {name}  (invalid)")
            if i == self.canvas.selected_idx:
                self.annList.setCurrentRow(i)
        self.annList.blockSignals(False)
        self.annCountLabel.setText(f"{len(self.canvas.annotations)} boxes")

    def _update_nav_buttons(self):
        self.prevBtn.setEnabled(self._current_idx > 0)
        self.nextBtn.setEnabled(self._current_idx < len(self._image_paths) - 1)

    def _update_counts(self):
        n = len(self._image_paths)
        annotated = len([k for k in self._annotations if self._annotations[k]])
        self.countLabel.setText(f"{n} images")
        self.annotatedLabel.setText(f"{annotated} annotated")
        remaining = (n + 2) // 3 if n >= 3 else n
        self.filterStatsLabel.setText(f"{n} → {remaining} images")
        self._update_stats()

    def _update_stats(self):
        if not hasattr(self, 'statsContainer'):
            return
        lo = self.statsContainerLo
        while lo.count():
            item = lo.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        class_counts = {}
        total_boxes = 0
        total_images = len(self._annotations)
        for anns in self._annotations.values():
            for ann in anns:
                class_counts[ann.class_id] = class_counts.get(ann.class_id, 0) + 1
                total_boxes += 1

        self.statsSummary.setText(f"images {total_images} · boxes {total_boxes}")

        if class_counts:
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                name = lbsvc.resolve_class_name(class_id)
                color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
                percentage = (count / total_boxes * 100) if total_boxes > 0 else 0

                row_w = QWidget()
                rl = QHBoxLayout(row_w)
                rl.setSpacing(6)
                rl.setContentsMargins(0, 0, 0, 0)

                nl = QHBoxLayout()
                nl.setSpacing(4)
                dot = QLabel("●")
                dot.setStyleSheet(f"color:{color};font-size:12px;")
                nl.addWidget(dot)
                nlbl = QLabel(name)
                nlbl.setStyleSheet(f"font-size:10px;color:{TEXT};font-weight:500;")
                nl.addWidget(nlbl)
                nl.addStretch()
                rl.addLayout(nl, 1)

                cl = QLabel(str(count))
                cl.setStyleSheet(f"font-size:10px;color:{TEXT};font-weight:500;min-width:36px;")
                rl.addWidget(cl)

                pl = QLabel(f"{percentage:.1f}%")
                pl.setStyleSheet(f"font-size:10px;color:{TEXT2};min-width:40px;")
                rl.addWidget(pl)

                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(int(percentage))
                bar.setTextVisible(False)
                bar.setFixedHeight(8)
                bar.setStyleSheet(
                    f"QProgressBar{{background:#f0f0f0;border-radius:4px;border:none;}}"
                    f"QProgressBar::chunk{{background:{color};border-radius:4px;}}")
                rl.addWidget(bar, 1)
                lo.addWidget(row_w)
        else:
            nd = QLabel("No annotations yet")
            nd.setStyleSheet(f"font-size:10px;color:{TEXT3};padding:8px;")
            nd.setAlignment(Qt.AlignCenter)
            lo.addWidget(nd)

    # ═══════════════ SOURCE ═══════════════

    def _refresh_source_folders(self):
        self.srcCombo.blockSignals(True)
        self.srcCombo.clear()
        ar = lbsvc.after_root()
        if ar.exists():
            dirs = sorted([d.name for d in ar.iterdir() if d.is_dir()])
            if dirs:
                self.srcCombo.addItems(dirs)
                self.srcCombo.blockSignals(False)
                self._on_folder_selected()
                return
        self.srcCombo.blockSignals(False)
        self.srcPathLabel.setText(f" No subfolders in {ar}")

    # ═══════════════ AUTO LABEL ═══════════════

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Model", "runs", "PyTorch (*.pt)")
        if p:
            existing = [self.modelCombo.itemText(i) for i in range(self.modelCombo.count())]
            rel = Path(p)
            display = rel.name
            if display not in existing:
                self.modelCombo.addItem(display, p)
            self.modelCombo.setCurrentText(display)
            self.studio.log_operation('Label', f'选择自动标注模型 · {display}')

    def _scan_models(self):
        self.modelCombo.clear()
        models = sorted(ROOT.rglob("weights/best5.20.pt"))
        for m in models:
            display = str(m.relative_to(ROOT))
            self.modelCombo.addItem(display, str(m))
        for m in sorted(ROOT.glob("*.pt")):
            if m.name not in [self.modelCombo.itemText(i) for i in range(self.modelCombo.count())]:
                self.modelCombo.addItem(m.name, str(m))

    def _start_auto(self):
        if not self._image_paths:
            QMessageBox.warning(self, "Warning", "Please select a folder first")
            return
        model_path = self.modelCombo.currentData()
        if not model_path or not Path(model_path).exists():
            self._scan_models()
            model_path = self.modelCombo.currentData()
            if not model_path or not Path(model_path).exists():
                QMessageBox.warning(self, "Warning", "Please select a model file")
                return
        self._auto_anns = {}
        self.autoBtn.setEnabled(False)
        self.autoBtn.setText("  Working…")
        self.exportBar.setValue(0)
        self._worker = AutoLabelWorker(
            model_path=model_path,
            image_paths=[str(p) for p in self._image_paths],
            conf=self.alConf.value(),
            iou=self.alIou.value(),
        )
        self._worker.progress.connect(lambda c, t: self.exportBar.setValue(int(c / t * 100)))
        self._worker.image_done.connect(self._on_auto_image_done)
        self._worker.log.connect(lambda m: self.exportStatus.setText(m))
        self._worker.done.connect(self._on_auto_done)
        self._worker.start()
        self.studio.log_operation('Label', f'自动标注开始 · {len(self._image_paths)} 张图 · 模型 {Path(model_path).name}')

    def _on_auto_image_done(self, img_path, anns):
        self._auto_anns[img_path] = anns

    def _on_auto_done(self, ok, msg):
        self.autoBtn.setEnabled(True)
        self.autoBtn.setText("▶ Auto")
        if ok and self._auto_anns:
            for img_path, ann_dicts in self._auto_anns.items():
                if ann_dicts:
                    self._annotations[img_path] = [lbsvc.Annotation.from_dict(d) for d in ann_dicts]
            key = str(self._image_paths[self._current_idx])
            if key in self._annotations:
                self.canvas.set_annotations(self._annotations[key])
            self._save_session()
            self._update_counts()
            self._update_stats()
        self.exportBar.setValue(100 if ok else 0)
        ann_count = sum(1 for v in self._auto_anns.values() if v)
        self.studio.log_operation('Label', f'自动标注{"完成" if ok else "失败"} · {ann_count} 张已标 · {msg}')

    # ═══════════════ EXPORT ═══════════════

    def _export(self):
        self._save_current_annotations()
        for p in self._image_paths:
            key = str(p)
            if key not in self._annotations:
                self._annotations[key] = []
        total = len(self._annotations)
        if total == 0:
            QMessageBox.warning(self, "Warning", "No images to export")
            return
        labeled = {k: v for k, v in self._annotations.items() if v}
        unlabeled = {k: v for k, v in self._annotations.items() if not v}
        unlabeled_count = len(unlabeled)
        if unlabeled_count > 0:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Export Dataset")
            msg.setText(f"Ready to export {total} images.")
            msg.setInformativeText(
                f"With annotations: {len(labeled)}\n"
                f"Empty annotations (negative samples): {unlabeled_count}\n\n"
                f"Images with empty annotations will be exported\n"
                f"(with empty .txt files for negative samples). Continue?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.Yes)
            if msg.exec_() != QMessageBox.Yes:
                return
        self.exportBtn.setEnabled(False)
        self.exportBar.setValue(0)
        self._export_worker = ExportWorker(
            self._annotations, lbsvc.label_root(), self.trainRatio.value(), ROOT)
        self._export_worker.progress.connect(lambda v: self.exportBar.setValue(v))
        self._export_worker.log.connect(lambda m: self.exportStatus.setText(m))
        self._export_worker.done.connect(self._on_export_done)
        self._export_worker.start()
        self.studio.log_operation('Label', f'开始导出数据集 · {total} 张图 (标注 {len(labeled)} / 空 {unlabeled_count})')

    def _on_export_done(self, ok, msg):
        self.exportBtn.setEnabled(True)
        if ok:
            QMessageBox.information(self, "Export Complete", msg)
            self.studio.log_operation('Label', '数据集导出成功 ✓')
        else:
            self.studio.log_operation('Label', f'导出失败 · {msg}')
        self._export_worker = None

    # ═══════════════ EVENT FILTER ═══════════════

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and isinstance(obj, QLineEdit):
            key = event.key()
            modifiers = event.modifiers()
            if key in (Qt.Key_Control, Qt.Key_Alt, Qt.Key_Shift, Qt.Key_Meta):
                return True
            parts = []
            if modifiers & Qt.ControlModifier:
                parts.append("Ctrl")
            if modifiers & Qt.AltModifier:
                parts.append("Alt")
            if modifiers & Qt.ShiftModifier:
                parts.append("Shift")
            key_name = None
            if Qt.Key_0 <= key <= Qt.Key_9:
                key_name = str(key - Qt.Key_0)
            elif Qt.Key_A <= key <= Qt.Key_Z:
                key_name = chr(key)
            elif key == Qt.Key_Space:
                key_name = "Space"
            elif key in (Qt.Key_Enter, Qt.Key_Return):
                key_name = "Enter"
            elif key == Qt.Key_Delete:
                key_name = "Delete"
            elif key == Qt.Key_Backspace:
                key_name = "Backspace"
            elif key == Qt.Key_Left:
                key_name = "Left"
            elif key == Qt.Key_Right:
                key_name = "Right"
            elif key == Qt.Key_Up:
                key_name = "Up"
            elif key == Qt.Key_Down:
                key_name = "Down"
            elif key == Qt.Key_F1:
                key_name = "F1"
            elif key == Qt.Key_F2:
                key_name = "F2"
            elif key == Qt.Key_F3:
                key_name = "F3"
            elif key == Qt.Key_F4:
                key_name = "F4"
            elif key == Qt.Key_F5:
                key_name = "F5"
            elif key == Qt.Key_F6:
                key_name = "F6"
            elif key == Qt.Key_F7:
                key_name = "F7"
            elif key == Qt.Key_F8:
                key_name = "F8"
            elif key == Qt.Key_F9:
                key_name = "F9"
            elif key == Qt.Key_F10:
                key_name = "F10"
            elif key == Qt.Key_F11:
                key_name = "F11"
            elif key == Qt.Key_F12:
                key_name = "F12"
            elif key == Qt.Key_Escape:
                key_name = "Esc"
            elif key == Qt.Key_Tab:
                key_name = "Tab"
            if key_name:
                parts.append(key_name)
                obj.setText("+".join(parts))
            return True
        return super().eventFilter(obj, event)
