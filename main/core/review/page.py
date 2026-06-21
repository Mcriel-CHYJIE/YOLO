# =============================================================================
# YOLO Training Studio — 重新标注标签页 UI
# 直接从 dataset 的 images/labels 目录读取，修改后直接覆盖 .txt 文件
# 界面同 Label 页
# =============================================================================

"""重新标注标签页 UI — 读取 dataset images/labels，直接覆写 .txt"""

from PyQt5 import uic
from main.core.base import *
from . import service as rsvc
from .service import Annotation
from main.core.review.import_service import count_importable, import_dataset

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
        self._drag_idx = -1
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
        color = CLASS_COLORS[ann.class_id % len(CLASS_COLORS)]
        qc = QColor(color)
        painter.setBrush(Qt.NoBrush)
        pen = QPen(qc)
        pen.setWidth(3 if selected else 2)
        if selected:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        label = rsvc.resolve_class_name(ann.class_id)
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
            x1, y1, x2, y2 = self._ann_rect(ann)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return i
        return -1

    def _hit_handle(self, cx, cy):
        for i, ann in enumerate(self._annotations):
            x1, y1, x2, y2 = self._ann_rect(ann)
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for j, (hx, hy) in enumerate(corners):
                if abs(cx - hx) <= 5 and abs(cy - hy) <= 5:
                    return i, j
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
                self._drag_idx = -1
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
        if self._drag_mode == "move" and 0 <= self._drag_idx < len(self._annotations):
            dx = (cx - self._drag_start[0]) / (self._scale * self._img_w)
            dy = (cy - self._drag_start[1]) / (self._scale * self._img_h)
            ann = self._annotations[self._drag_idx]
            ann.xc = max(ann.w / 2, min(1 - ann.w / 2, ann.xc + dx))
            ann.yc = max(ann.h / 2, min(1 - ann.h / 2, ann.yc + dy))
            self._drag_start = (cx, cy)
            self.annotation_changed.emit()
            self.update()
            return
        if self._drag_mode == "resize" and 0 <= self._drag_idx < len(self._annotations):
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
                    rsvc.Annotation(self._class_id, xc, yc, w, h))
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
            self._drag_idx = -1
            self.annotation_changed.emit()
            self.update()

    def clear_annotations(self):
        self._annotations.clear()
        self._selected_idx = -1
        self._drag_idx = -1
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


# ═══════════════════════ RelabelTab ═══════════════════════

class RelabelTab(QWidget):
    """重新标注标签页 — 直接读写 dataset images/labels 的 .txt 文件"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._init_state()
        self._load_ui()
        self._post_process_ui()
        self._connect_signals()
        self._refresh_source_folders()

    def _init_state(self):
        self._image_paths = []
        self._current_idx = 0
        self._current_image = None
        self._current_split = ''
        self._is_auto_mode = False
        self._pending_delete_path = None
        self._worker = None
        self._class_ids = []
        self._class_btns = []
        self._shortcut_keys = {}
        self._ann_counts = {}  # {img_path_str: instance_count}
        self._ann_data = {}    # {img_path_str: [Annotation, ...]} 惰性缓存
        self._canvas_save = True  # 防导航时重复保存
        self._has_unsaved = False

    def _load_ui(self):
        ui_path = Path(__file__).resolve().parent / 'relabel.ui'
        uic.loadUi(str(ui_path), self)

    def _post_process_ui(self):
        """替换占位 widget + 初始化动态区域"""
        # ── 标题 ──
        self.titleLabel.setStyleSheet(f'font-size:18px;font-weight:700;color:{TEXT};padding:0;margin:0;')
        self.titleLabel.setFixedHeight(24)

        # ── 替换 canvasPlaceholder 为 AnnotationCanvas ──
        self.canvas = AnnotationCanvas(self.centerPanel)
        self.canvas.annotation_changed.connect(self._on_annotation_changed)
        self.centerLo.addWidget(self.canvas, 0, 0)
        # Remove placeholder
        self.canvasPlaceholder.setParent(None)
        self.canvasPlaceholder.deleteLater()

        # ── Delete confirm styles ──
        self.centerLo.setAlignment(self.deleteConfirm, Qt.AlignBottom)
        self.deleteConfirm.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.deleteConfirm.setMaximumHeight(48)
        self.deleteLo.setStretch(0, 1)
        self.deleteLo.setStretch(1, 0)
        self.deleteLo.setStretch(2, 0)
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

        # ── Build class buttons ──
        self._build_class_buttons()

        # ── Load shortcut keys ──
        self._shortcut_keys = rsvc.load_shortcut_keys()

        # ── Set focus policy ──
        self.setFocusPolicy(Qt.StrongFocus)

        # ── Apply widget styles ──
        self._apply_widget_styles()

    def _apply_widget_styles(self):
        for g in (self.sourceGroup,
                  self.annGroup, self.classGroup):
            g.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.countLabel.setStyleSheet(f"font-size:10px;color:{TEXT3};font-weight:500;")
        self.instLabel.setStyleSheet(f"font-size:10px;color:{TEXT2};font-weight:500;")
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
        for sep in (self.sourceSep, self.classSep3):
            sep.setStyleSheet(f"background:{BORDER};")
        common_btn_style = (
            f"QPushButton{{background:{BG};border:1px solid {BORDER};border-radius:5px;"
            f"color:{RED};font-size:10px;font-weight:500;padding:4px 8px;text-align:left;}}"
            f"QPushButton:hover{{background:{PRI}20;border-color:{PRI};}}")
        self.delBtn.setStyleSheet(common_btn_style)
        self.delBtn.setMinimumHeight(30)
        self.clearBtn.setStyleSheet(common_btn_style)
        self.clearBtn.setMinimumHeight(30)
        for nav_btn in (self.prevBtn, self.nextBtn):
            nav_btn.setMinimumSize(30, 28)
            nav_btn.setStyleSheet(
                f"QPushButton{{background:{BG};border:1px solid {BORDER};border-radius:5px;"
                f"color:{TEXT};font-size:13px;font-weight:600;}}"
                f"QPushButton:hover{{background:{PRI}20;border-color:{PRI};}}"
                f"QPushButton:disabled{{color:{TEXT3};border-color:{BORDER};}}")
        self.srcRefreshBtn.setStyleSheet(
            f"QPushButton{{background:{BG};border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT};font-size:13px;font-weight:500;}}"
            f"QPushButton:hover{{background:{PRI}20;border-color:{PRI};}}"
            f"QPushButton:pressed{{background:{PRI}40;}}")
        self.srcImportBtn.setStyleSheet(
            "QPushButton{background:#ffffff;color:#07C160;border:1px solid #07C160;"
            "border-radius:4px;font-size:11px;font-weight:600;}"
            "QPushButton:hover{background:#e8f5e9;}"
            "QPushButton:pressed{background:#d0ebd0;}")
        self.imgNameLabel.setStyleSheet(f"font-size:10px;font-weight:500;color:{TEXT};")
        self.imgNameLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.annCountLabel.setStyleSheet(f"font-size:9px;color:{TEXT2};")
        self.annList.setStyleSheet(
            f"QListWidget{{background:{BG};border:1px solid {BORDER};border-radius:5px;"
            f"font-size:10px;color:{TEXT};padding:3px;min-height:120px;}}"
            f"QListWidget::item{{padding:4px 8px;border-radius:3px;"
            f"border-bottom:1px solid {BORDER};}}"
            f"QListWidget::item:selected{{background:{PRI}25;color:{TEXT};border:none;}}")
        self.centerPanel.setStyleSheet(
            f"background:#f0f0f0;border:1px solid {BORDER};border-radius:6px;")

    def _connect_signals(self):
        self.srcCombo.currentIndexChanged.connect(self._on_folder_selected)
        self.srcRefreshBtn.clicked.connect(self._refresh_source_folders)
        self.srcImportBtn.clicked.connect(self._import_dataset)

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

    # ═══════════════ CLASS BUTTONS ═══════════════

    def _build_class_buttons(self):
        import main.core.base as base
        for btn in self._class_btns:
            self.classBtnsGrid.removeWidget(btn)
            btn.deleteLater()
        self._class_btns.clear()
        while self.classBtnsGrid.count():
            item = self.classBtnsGrid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._class_ids = sorted(int(k) for k in base.CLASS_NAMES.keys())
        for i, class_id in enumerate(self._class_ids):
            row, col = i // 2, i % 2
            name = rsvc.resolve_class_name(class_id)
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

    def _on_annotation_changed(self):
        if not getattr(self, '_canvas_save', True):
            return
        self._has_unsaved = True
        self.saveIndicator.setText("● Unsaved")
        self.saveIndicator.setStyleSheet(f"font-size:10px;color:{AMBER};font-weight:600;")
        # 更新当前图片的实例缓存
        if self._image_paths and self._current_idx < len(self._image_paths):
            key = str(self._image_paths[self._current_idx])
            self._ann_data[key] = list(self.canvas.annotations)
            self._ann_counts[key] = len(self.canvas.annotations)
        # 自动保存到 .txt
        self._save_current()
        self._update_ann_list()
        self._update_stats()

    def _on_ann_list_select(self, row):
        if 0 <= row < len(self.canvas.annotations) and self.canvas.selected_idx != row:
            self.canvas.selected_idx = row
            self.canvas.update()

    def _on_folder_selected(self):
        split = self.srcCombo.currentText()
        if not split:
            return
        self._current_split = split
        img_root = rsvc.dataset_images_root() / split
        self.srcPathLabel.setText(f" {img_root}")
        self._load_images(img_root)
        self.studio.log_operation('Review', f'加载数据集 · {split}/{Path(img_root).name} · {len(self._image_paths)} 张图')

    # ═══════════════ IMAGE LOADING ═══════════════

    def _load_images(self, folder: Path):
        self._image_paths = sorted(folder.rglob("*.jpg"))
        self._current_idx = 0
        self._ann_counts = {}       # 惰性填充
        self._ann_data = {}         # {path: [Annotation, ...]} 缓存已解析的标注
        self._update_counts()
        self._update_stats()
        if self._image_paths:
            self._show_image(self._current_idx)
        else:
            self.countLabel.setText("0 images")
            self.instLabel.setText("0 instances")

    def _show_image(self, idx):
        if idx < 0 or idx >= len(self._image_paths):
            return
        img_path = self._image_paths[idx]
        self._current_image = cv2.imread(str(img_path))
        if self._current_image is None:
            return
        self.canvas.set_image(self._current_image)
        # 从缓存或 disk 加载标注
        key = str(img_path)
        anns = self._ann_data.get(key)
        if anns is None:
            anns = rsvc.load_labels_for_image(img_path, self._current_split)
            self._ann_data[key] = anns
            self._ann_counts[key] = len(anns)
        self._canvas_save = False
        try:
            if anns:
                self.canvas.set_annotations(anns)
        finally:
            self._canvas_save = True
        self._current_idx = idx
        self.idxInput.blockSignals(True)
        self.idxInput.setValue(idx + 1)
        self.idxInput.blockSignals(False)
        self.totalLabel.setText(f"/ {len(self._image_paths)}")
        self.imgNameLabel.setText(f"{img_path.parent.name}/{img_path.name}")
        self._update_ann_list()
        self._update_nav_buttons()

    # ═══════════════ NAVIGATION ═══════════════

    def _navigate(self, delta):
        if not self._image_paths:
            return
        self._save_current()
        new_idx = max(0, min(len(self._image_paths) - 1, self._current_idx + delta))
        if new_idx != self._current_idx:
            self._current_idx = new_idx
            self._show_image(self._current_idx)

    def _goto_idx(self, val):
        if not self._image_paths:
            return
        idx = val - 1
        if 0 <= idx < len(self._image_paths):
            self._save_current()
            if idx != self._current_idx:
                self._current_idx = idx
                self._show_image(idx)

    # ═══════════════ SAVE ═══════════════

    def _save_current(self):
        """将当前标注直接写入 .txt 文件"""
        if not self._image_paths or self._current_image is None:
            return
        img_path = self._image_paths[self._current_idx]
        anns = self.canvas.annotations
        rsvc.save_labels_for_image(img_path, self._current_split, anns)

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
            # 删除图片和对应的 label .txt
            current_path.unlink()
            lbl_root = rsvc.dataset_labels_root()
            lbl_file = lbl_root / self._current_split / f"{current_path.stem}.txt"
            if lbl_file.exists():
                lbl_file.unlink()
            self._image_paths.pop(self._current_idx)
            self._ann_counts.pop(str(current_path), None)
            self._ann_data.pop(str(current_path), None)
            if self._current_idx >= len(self._image_paths):
                self._current_idx = max(0, len(self._image_paths) - 1)
            if self._image_paths:
                self._show_image(self._current_idx)
            else:
                self.canvas.clear_image()
                self.countLabel.setText("0 images")
                self.instLabel.setText("0 instances")
                self.imgNameLabel.setText("—")
                self.annCountLabel.setText("0 boxes")
            self._update_counts()
            self._update_stats()
            self.studio.log_operation('Review', f'删除图片 · {current_path.name}')
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
            QMessageBox.warning(self, "Warning", "Need at least 3 images to filter")
            return
        from .service import random_filter_paths as rfp
        to_keep, to_delete = rfp(self._image_paths)
        # 删除被筛选掉的图片和对应 label
        for p in to_delete:
            try:
                p.unlink()
                lbl_root = rsvc.dataset_labels_root()
                lbl = lbl_root / self._current_split / f"{p.stem}.txt"
                if lbl.exists():
                    lbl.unlink()
            except Exception:
                pass
        kept = len(to_keep)
        removed = total - kept
        self._image_paths = sorted(to_keep)
        self._current_idx = 0
        if self._image_paths:
            self._show_image(0)
        else:
            self.canvas.clear_image()
        self._update_counts()
        self.annotatedLabel.setText(f"{len(self._image_paths)} images")
        self.studio.log_operation('Review', f'随机筛选 · {total}→{kept} 张 · 删除 {removed} 张')

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
                cls_shortcuts = rsvc.get_cls_shortcuts()
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

    # ═══════════════ REFRESH ═══════════════

    def _refresh_source_folders(self):
        """扫描 dataset_dir/images/ 下的子目录"""
        current = self.srcCombo.currentText()
        self.srcCombo.clear()
        folders = rsvc.get_split_folders()
        if not folders:
            self.srcPathLabel.setText(" No dataset configured. Go to Settings → Dataset dir.")
        for f in folders:
            self.srcCombo.addItem(f)
        idx = self.srcCombo.findText(current)
        if idx >= 0:
            self.srcCombo.setCurrentIndex(idx)

    def _import_dataset(self):
        """从 original/label 导入数据集到 datasets/"""
        from PyQt5.QtWidgets import QMessageBox
        train_c, val_c, total, err = count_importable()
        if err:
            QMessageBox.warning(self, 'Error', err)
            return
        if total == 0:
            QMessageBox.information(self, 'Nothing to Import',
                'No images found in original/label/images/')
            return
        reply = QMessageBox.question(self, 'Confirm Import',
            f'Import {total} images from original/label?\n'
            f'  Train: {train_c}  |  Val: {val_c}\n\n'
            'This will copy all labeled images into datasets/',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        copied, total, err = import_dataset()
        if err:
            if 'not found' in err:
                QMessageBox.warning(self, 'Error', err)
            elif copied == 0:
                QMessageBox.warning(self, 'Warning', err)
            else:
                QMessageBox.warning(self, 'Error', err)
            return
        QMessageBox.information(self, 'Import Complete',
            f'Successfully imported {copied} images from original/label')
        self._refresh_source_folders()
        if self.srcCombo.count():
            self._on_folder_selected()
        self.studio.log_operation('Review', f'导入数据集 · {copied} 张图')

    # ═══════════════ UI UPDATE ═══════════════

    def _update_counts(self):
        total = len(self._image_paths)
        self.countLabel.setText(f"{total} images")

    def _update_ann_list(self):
        self.annList.blockSignals(True)
        self.annList.clear()
        for i, ann in enumerate(self.canvas.annotations):
            name = rsvc.resolve_class_name(ann.class_id)
            item = QListWidgetItem(
                f"[{ann.class_id}] {name}: "
                f"xc={ann.xc:.3f} yc={ann.yc:.3f} w={ann.w:.3f} h={ann.h:.3f}")
            item.setData(Qt.UserRole, i)
            item.setSelected(i == self.canvas.selected_idx)
            self.annList.addItem(item)
        self.annList.blockSignals(False)
        self.annCountLabel.setText(f"{len(self.canvas.annotations)} boxes")

    def _update_stats(self):
        # 清除旧统计行
        while self.statsContainerLo.count():
            item = self.statsContainerLo.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # 惰性加载所有标注并聚合类别
        from collections import Counter
        counter = Counter()
        total = 0
        for img_path in self._image_paths:
            key = str(img_path)
            if key in self._ann_data:
                anns = self._ann_data[key]
            else:
                anns = rsvc.load_labels_for_image(img_path, self._current_split)
                self._ann_data[key] = anns
            self._ann_counts[key] = len(anns)
            total += len(anns)
            for a in anns:
                counter[a.class_id] += 1

        self.instLabel.setText(f"{total} instances")
        if total == 0:
            return
        for class_id in sorted(counter.keys()):
            count = counter[class_id]
            pct = count / total * 100
            name = rsvc.resolve_class_name(class_id)
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            row = QWidget()
            row_lo = QHBoxLayout(row)
            row_lo.setContentsMargins(0, 0, 0, 0)
            row_lo.setSpacing(4)
            cls_lbl = QLabel(name)
            cls_lbl.setStyleSheet(f"font-size:9px;color:{TEXT};font-weight:500;")
            cls_lbl.setMinimumWidth(60)
            cnt_lbl = QLabel(str(count))
            cnt_lbl.setStyleSheet(f"font-size:9px;color:{TEXT2};")
            cnt_lbl.setFixedWidth(36)
            pct_lbl = QLabel(f"{pct:.0f}%")
            pct_lbl.setStyleSheet(f"font-size:9px;color:{TEXT2};")
            pct_lbl.setFixedWidth(32)
            bar = QWidget()
            bar.setFixedHeight(10)
            bar.setStyleSheet(
                f"background:{color};border-radius:3px;")
            bar.setMinimumWidth(4)
            row_lo.addWidget(cls_lbl)
            row_lo.addWidget(cnt_lbl)
            row_lo.addWidget(pct_lbl)
            row_lo.addWidget(bar, int(pct))
            self.statsContainerLo.addWidget(row)

    def _update_nav_buttons(self):
        self.prevBtn.setEnabled(self._current_idx > 0)
        self.nextBtn.setEnabled(self._current_idx < len(self._image_paths) - 1)
