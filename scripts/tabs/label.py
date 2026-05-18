"""标注标签页 — 手动/自动标注 → 审核 → 导出YOLO格式数据集"""
from scripts.tabs.base import *
import cv2, json, random, shutil, hashlib

# ═══════════════════════ 常量 ═══════════════════════

CLASS_COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444',
                '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']

CANVAS_SIZE = 640


def resolve_class_name(class_id):
    """从配置的 names 映射中获取类名（模块级函数，与 LabelTab 解耦）"""
    name = CLASS_NAMES.get(class_id, f'cls{class_id}')
    return str(list(name.values())[0]) if isinstance(name, dict) else name


# ═══════════════════════ Annotation ═══════════════════════

class Annotation:
    """单个标注框 (YOLO format: class_id xc yc w h 归一化)"""

    __slots__ = ('class_id', 'xc', 'yc', 'w', 'h')

    def __init__(self, class_id, xc, yc, w, h):
        self.class_id = class_id
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h

    def to_yolo(self):
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"

    def to_dict(self):
        return {'class_id': self.class_id, 'xc': self.xc, 'yc': self.yc, 'w': self.w, 'h': self.h}

    @classmethod
    def from_dict(cls, d):
        return cls(d['class_id'], d['xc'], d['yc'], d['w'], d['h'])

    @classmethod
    def from_yolo(cls, line):
        parts = line.strip().split()
        if len(parts) >= 5:
            return cls(int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        return None
# ═══════════════════════ AnnotationCanvas ═══════════════════════

class AnnotationCanvas(QWidget):
    """标注画布 — 固定640×640，图片居中按比例缩放"""

    annotation_changed = pyqtSignal()
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)
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
        self._cursor = None
        # 动态彩色虚线
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
        scale = min(CANVAS_SIZE / self._img_w, CANVAS_SIZE / self._img_h)
        self._scale = scale
        self._disp_w = int(self._img_w * scale)
        self._disp_h = int(self._img_h * scale)
        self._ox = (CANVAS_SIZE - self._disp_w) // 2
        self._oy = (CANVAS_SIZE - self._disp_h) // 2

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
        """更新绘制框的颜色"""
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
        label = resolve_class_name(ann.class_id)
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
            event.ignore()
            return
        cx, cy = event.x(), event.y()
        if not self._in_image(cx, cy):
            event.ignore()
            return
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
            self._color_timer.start(150)  # 150ms切换一次颜色
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
        if self._drag_mode == "move" and self._drag_idx >= 0:
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
        if event.button() != Qt.LeftButton:
            return
        if self._drawing and self._drag_start and self._drag_end:
            self._drawing = False
            self._color_timer.stop()
            sx, sy = self._drag_start
            ex, ey = self._drag_end
            
            # 计算拖拽距离（像素）
            drag_distance = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            
            # 最小拖拽距离阈值（5像素），避免误触
            if drag_distance < 5:
                # 距离太小，取消绘制
                self._drag_start = None
                self._drag_end = None
                self.update()
                return
            
            nsx, nsy = self._can2img(sx, sy)
            nex, ney = self._can2img(ex, ey)
            xc, yc = (nsx + nex) / 2, (nsy + ney) / 2
            w, h = abs(nex - nsx), abs(ney - nsy)
            xc = max(0, min(1, xc))
            yc = max(0, min(1, yc))
            w = max(0.01, min(1, w))
            h = max(0.01, min(1, h))
            if w > 0.005 and h > 0.005:
                self._annotations.append(Annotation(self._class_id, xc, yc, w, h))
                self.annotation_changed.emit()
            self._drag_start = None
            self._drag_end = None
            self.update()
            return
        self._drag_mode = ""
        self._drag_idx = -1

    def resizeEvent(self, event):
        self.update()

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
        """清空画布上的图片"""
        self._image = None
        self._pixmap = None
        self._annotations.clear()
        self._selected_idx = -1
        self._drawing = False
        self._drag_start = None
        self._drag_end = None
        self._drag_idx = -1
        self._drag_mode = ""
        self._resize_corner = -1
        self.update()
# ═══════════════════════ AutoLabelWorker ═══════════════════════

class AutoLabelWorker(QThread):
    progress = pyqtSignal(int, int)
    image_done = pyqtSignal(str, list)
    done = pyqtSignal(bool, str)
    log = pyqtSignal(str)

    def __init__(self, model_path, image_paths, conf=0.25, iou=0.45):
        super().__init__()
        self._model_path = model_path
        self._image_paths = image_paths
        self._conf = conf
        self._iou = iou
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            from ultralytics import YOLO
            self.log.emit(f"🚀 加载模型: {Path(self._model_path).name}")
            model = YOLO(self._model_path)
            n = len(self._image_paths)
            self.log.emit(f"📸 开始自动标注 {n} 张图片...")
            for i, img_path in enumerate(self._image_paths):
                if self._stop_flag:
                    self.done.emit(False, "Stopped")
                    return
                results = model(img_path, conf=self._conf, iou=self._iou, verbose=False)[0]
                anns = []
                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        xywhn = box.xywhn[0].tolist()
                        if len(xywhn) == 4:
                            anns.append(dict(class_id=cls_id, xc=xywhn[0], yc=xywhn[1],
                                             w=xywhn[2], h=xywhn[3]))
                self.image_done.emit(str(img_path), anns)
                self.progress.emit(i + 1, n)
            self.log.emit("✅ 自动标注完成")
            self.done.emit(True, "Auto-label complete")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log.emit(f"❌ {e}")
            self.done.emit(False, str(e))


# ═══════════════════════ ExportWorker ═══════════════════════

class ExportWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, annotations, out_root, train_ratio, project_root):
        super().__init__()
        self._annotations = annotations
        self._out_root = out_root
        self._train_ratio = train_ratio
        self._project_root = project_root

    def run(self):
        try:
            items = list(self._annotations.items())
            items.sort(key=lambda x: hashlib.md5(x[0].encode("utf-8")).hexdigest())
            split_idx = max(1, int(len(items) * self._train_ratio / 100.0))
            train_items = items[:split_idx]
            val_items = items[split_idx:]
            for sub in ["images/train", "images/val", "labels/train", "labels/val", "visualize"]:
                (self._out_root / sub).mkdir(parents=True, exist_ok=True)
            
            total = len(items)
            exported_count = 0
            skipped_count = 0
            
            for split_name, split_items in [("train", train_items), ("val", val_items)]:
                for orig_path, anns in split_items:
                    src = Path(orig_path)
                    if not src.exists():
                        continue
                    
                    # 使用原始文件名（不含扩展名）
                    base_name = src.stem
                    dst_img = self._out_root / "images" / split_name / f"{base_name}.jpg"
                    dst_lbl = self._out_root / "labels" / split_name / f"{base_name}.txt"
                    dst_viz = self._out_root / "visualize" / f"{base_name}.jpg"
                    
                    # 检查文件是否已存在，避免覆盖
                    if dst_img.exists() or dst_lbl.exists():
                        skipped_count += 1
                        continue
                    
                    shutil.copy2(str(src), str(dst_img))
                    lines = [a.to_yolo() for a in anns]
                    dst_lbl.write_text("\n".join(lines) + "\n", "utf-8")
                    img = cv2.imread(str(src))
                    if img is not None:
                        h, w = img.shape[:2]
                        for a in anns:
                            x1 = int((a.xc - a.w / 2) * w)
                            y1 = int((a.yc - a.h / 2) * h)
                            x2 = int((a.xc + a.w / 2) * w)
                            y2 = int((a.yc + a.h / 2) * h)
                            color_hex = CLASS_COLORS[a.class_id % len(CLASS_COLORS)]
                            color_bgr = tuple(int(color_hex[j:j + 2], 16) for j in (5, 3, 1))
                            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
                            cv2.putText(img, resolve_class_name(a.class_id), (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                        cv2.imwrite(str(dst_viz), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    exported_count += 1
                    self.progress.emit(int(exported_count / total * 100))
            rel_label = self._out_root.relative_to(self._project_root)
            names_dict = cfg["project"].get("names", {i: name for i, name in enumerate(CLASSES)})
            names_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(names_dict.items()))
            yaml_content = (
                f"# Auto-generated by LabelTab\n"
                f"path: {rel_label.as_posix()}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"nc: {len(CLASSES)}\n"
                f"names:\n"
                f"{names_lines}\n"
            )
            (self._out_root / "data.yaml").write_text(yaml_content, "utf-8")
            self.log.emit(f"✅ Export complete: {exported_count} images (train {len(train_items)}/val {len(val_items)})")
            msg = f"Total: {exported_count} images | Train: {len(train_items)} | Val: {len(val_items)}"
            if skipped_count > 0:
                msg += f" | Skipped: {skipped_count} (already exists)"
            self.done.emit(True, msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log.emit(f"❌ Export failed: {e}")
            self.done.emit(False, str(e))
# ═══════════════════════ LabelTab ═══════════════════════

class LabelTab(QWidget):
    """标注标签页"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._init_state()
        self._build_ui()
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

    @property
    def _after_root(self):
        return ROOT / "original" / "after"

    @property
    def _label_root(self):
        return ROOT / "original" / "label"

    # ═══════════════ BUILD ═══════════════

    def _build_ui(self):
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(0)
        self.canvas = AnnotationCanvas()
        self.canvas.annotation_changed.connect(self._on_annotation_changed)
        # 使用固定布局
        main_lo = QHBoxLayout()
        main_lo.setContentsMargins(0, 0, 0, 0)
        main_lo.setSpacing(0)
        
        # 左侧面板
        left = self._build_left_panel()
        main_lo.addWidget(left)
        
        # 中间画布
        mid = self._build_center()
        main_lo.addWidget(mid, 1)
        
        # 右侧面板
        right = self._build_right_panel()
        main_lo.addWidget(right)
        
        lo.addLayout(main_lo)
        
        # 设置焦点策略，确保键盘事件可以被捕获
        self.setFocusPolicy(Qt.StrongFocus)

    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        panel = QWidget()
        panel.setFixedWidth(280)
        lo = QVBoxLayout(panel)
        lo.setSpacing(6)
        lo.setContentsMargins(4, 4, 4, 4)
        lo.addWidget(self._build_source_section())
        lo.addWidget(self._build_model_export_section())
        lo.addWidget(self._build_stats_section())
        lo.addStretch()
        scroll.setWidget(panel)
        scroll.setFixedWidth(280)
        return scroll

    def _build_center(self):
        mid = QWidget()
        mid.setStyleSheet(f"background:#f0f0f0;border:1px solid {BORDER};border-radius:6px;")
        mid.setMinimumSize(640 + 8, 640 + 8)
        lo = QVBoxLayout(mid)
        lo.setContentsMargins(4, 4, 4, 4)
        lo.setSpacing(4)
        lo.addWidget(self.canvas, alignment=Qt.AlignCenter)

        # 删除确认栏（位于画布下方）
        self._delete_confirm = QWidget()
        self._delete_confirm.setStyleSheet(f"background:#fff3cd;border:1px solid #ffc107;border-radius:6px;")
        self._delete_confirm.setVisible(False)
        confirm_lo = QHBoxLayout(self._delete_confirm)
        confirm_lo.setContentsMargins(12, 8, 12, 8)
        confirm_lo.setSpacing(8)

        self._confirm_msg = QLabel()
        self._confirm_msg.setStyleSheet("font-size:12px;font-weight:500;color:#856404;background:transparent;")
        self._confirm_msg.setWordWrap(True)
        confirm_lo.addWidget(self._confirm_msg, 1)

        yes_btn = QPushButton("Yes, Delete")
        yes_btn.setStyleSheet(
            "QPushButton { background:#dc3545;color:#fff;border:none;border-radius:4px;"
            "padding:6px 16px;font-size:11px;font-weight:600; }"
            "QPushButton:hover { background:#c82333; }")
        yes_btn.clicked.connect(self._confirm_delete_image)
        confirm_lo.addWidget(yes_btn)

        no_btn = QPushButton("Cancel")
        no_btn.setStyleSheet(
            "QPushButton { background:#fff;color:#6c757d;border:1px solid #ced4da;border-radius:4px;"
            "padding:6px 16px;font-size:11px;font-weight:500; }"
            "QPushButton:hover { background:#f8f9fa; }")
        no_btn.clicked.connect(lambda: self._delete_confirm.setVisible(False))
        confirm_lo.addWidget(no_btn)

        lo.addWidget(self._delete_confirm)
        return mid

    def _build_right_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        panel = QWidget()
        panel.setFixedWidth(280)
        lo = QVBoxLayout(panel)
        lo.setSpacing(6)
        lo.setContentsMargins(4, 4, 4, 4)
        lo.addWidget(self._build_annotation_section())
        lo.addWidget(self._build_class_nav_section())
        lo.addStretch()
        scroll.setWidget(panel)
        scroll.setFixedWidth(280)
        return scroll

    def _build_source_section(self):
        g = QGroupBox("Source")
        lo = QVBoxLayout(g)
        lo.setSpacing(5)
        lo.setContentsMargins(8, 8, 8, 8)
        row = QHBoxLayout()
        row.setSpacing(4)
        self._src_combo = QComboBox()
        self._src_combo.setMinimumHeight(28)
        self._src_combo.currentIndexChanged.connect(self._on_folder_selected)
        row.addWidget(self._src_combo, 2)
        row.addWidget(self._make_tool_btn("↻", self._refresh_source_folders), 1)
        lo.addLayout(row)
        self._src_path_label = QLabel()
        self._src_path_label.setStyleSheet(
            f"font-size:9px;color:{TEXT3};padding:4px 6px;background:{BG};"
            f"border-radius:4px;border:1px solid {BORDER};")
        self._src_path_label.setWordWrap(True)
        lo.addWidget(self._src_path_label)
        sr = QHBoxLayout()
        sr.setSpacing(6)
        self._count_label = QLabel("0 images")
        self._count_label.setStyleSheet(f"font-size:10px;color:{TEXT3};font-weight:500;")
        sr.addWidget(self._count_label)
        self._annotated_label = QLabel("0 annotated")
        self._annotated_label.setStyleSheet(f"font-size:10px;color:{GREEN};font-weight:600;")
        sr.addWidget(self._annotated_label)
        sr.addStretch()
        lo.addLayout(sr)
        
        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet(f"background:{BORDER};")
        lo.addWidget(sep)
        
        # Filter 统计信息
        self._filter_stats_label = QLabel("523 → 175 images")
        self._filter_stats_label.setStyleSheet(
            f"font-size:9px;color:{TEXT2};font-weight:500;padding:3px 5px;"
            f"background:{BG};border-radius:3px;border:1px solid {BORDER};")
        self._filter_stats_label.setAlignment(Qt.AlignCenter)
        lo.addWidget(self._filter_stats_label)
        
        # Filter 按钮
        filter_btn = QPushButton("🎲 Random Filter")
        filter_btn.setObjectName("pri")
        filter_btn.setMinimumHeight(26)
        filter_btn.clicked.connect(self._random_filter_dataset)
        lo.addWidget(filter_btn)
        
        return g

    def _build_class_nav_section(self):
        g = QGroupBox("Classes & Shortcuts")
        lo = QVBoxLayout(g)
        lo.setSpacing(6)
        lo.setContentsMargins(10, 10, 10, 10)
        self._class_layout = lo
        self._class_btns = []
        self._build_class_buttons()
        
        # 分隔线
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        sep1.setStyleSheet(f"background:{BORDER};")
        lo.addWidget(sep1)
        
        # 快捷键配置（两列布局）
        if not hasattr(self, '_shortcut_inputs'):
            self._shortcut_inputs = {}
        
        shortcuts = [
            ("Prev", "prev", "A"),
            ("Next", "next", "D"),
            ("Del Box", "delete_box", "W"),
            ("Del Img", "delete_img", "S"),
        ]
        
        grid = QGridLayout()
        grid.setSpacing(4)
        
        for idx, (label, key, default) in enumerate(shortcuts):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"font-size:9px;color:{TEXT2};font-weight:500;")
            grid.addWidget(lbl, 0, idx * 2)
            
            input_w = QLineEdit()
            input_w.setMinimumHeight(24)
            input_w.setAlignment(Qt.AlignCenter)
            input_w.setStyleSheet(
                f"QLineEdit {{ background:{BG}; border:1px solid {BORDER}; border-radius:4px;"
                f"color:{TEXT}; font-size:10px; padding:2px; }}"
                f"QLineEdit:focus {{ border-color:{PRI}; }}")
            
            input_w.installEventFilter(self)
            input_w.setProperty("shortcut_key", key)
            input_w.setReadOnly(True)
            input_w.setText(default)
            
            grid.addWidget(input_w, 0, idx * 2 + 1)
            self._shortcut_inputs[key] = input_w
        
        lo.addLayout(grid)
        
        # 分隔线
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        sep2.setStyleSheet(f"background:{BORDER};")
        lo.addWidget(sep2)
        
        # 按钮两列布局
        btn_row = QHBoxLayout()
        btn_row.setSpacing(5)
        
        del_btn = QPushButton("Delete")
        del_btn.setMinimumHeight(30)
        del_btn.setStyleSheet(
            f"QPushButton {{ background:{BG}; border:1px solid {BORDER}; border-radius:5px;"
            f"color:{RED}; font-size:10px; font-weight:500; padding:4px 8px; text-align:left; }}"
            f"QPushButton:hover {{ background:{PRI}20; border-color:{PRI}; }}")
        del_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(del_btn, 1)
        
        clear_btn = QPushButton("Clear")
        clear_btn.setMinimumHeight(30)
        clear_btn.setStyleSheet(
            f"QPushButton {{ background:{BG}; border:1px solid {BORDER}; border-radius:5px;"
            f"color:{RED}; font-size:10px; font-weight:500; padding:4px 8px; text-align:left; }}"
            f"QPushButton:hover {{ background:{PRI}20; border-color:{PRI}; }}")
        clear_btn.clicked.connect(self._clear_annotations)
        btn_row.addWidget(clear_btn, 1)
        
        lo.addLayout(btn_row)
        
        # 提示信息
        hint = QLabel("Click input to set key | Click buttons to select class")
        hint.setStyleSheet(f"font-size:8px;color:{TEXT3};font-style:italic;")
        hint.setAlignment(Qt.AlignCenter)
        lo.addWidget(hint)
        
        # 分隔线
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setFrameShadow(QFrame.Sunken)
        sep3.setStyleSheet(f"background:{BORDER};")
        lo.addWidget(sep3)
        
        # Navigation 部分
        # 导航控件行
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        self._prev_btn = self._make_icon_btn("◀")
        self._prev_btn.clicked.connect(lambda: self._navigate(-1))
        nav_row.addWidget(self._prev_btn, 1)
        self._next_btn = self._make_icon_btn("▶")
        self._next_btn.clicked.connect(lambda: self._navigate(1))
        nav_row.addWidget(self._next_btn, 1)
        
        self._idx_input = QSpinBox()
        self._idx_input.setMinimum(1)
        self._idx_input.setMaximum(99999)
        self._idx_input.setMinimumHeight(24)
        self._idx_input.valueChanged.connect(self._goto_idx)
        nav_row.addWidget(self._idx_input, 1)
        
        self._total_label = QLabel("/ 0")
        self._total_label.setStyleSheet(f"font-size:10px;color:{TEXT3};")
        nav_row.addWidget(self._total_label, 1)
        lo.addLayout(nav_row)
        
        # 信息行
        info_row = QHBoxLayout()
        info_row.setSpacing(6)
        self._img_name_label = QLabel("—")
        self._img_name_label.setStyleSheet(f"font-size:10px;font-weight:500;color:{TEXT};")
        info_row.addWidget(self._img_name_label, 1)
        
        self._ann_count_label = QLabel("0 boxes")
        self._ann_count_label.setStyleSheet(f"font-size:9px;color:{TEXT2};")
        info_row.addWidget(self._ann_count_label)
        
        self._save_indicator = QLabel("")
        self._save_indicator.setStyleSheet(f"font-size:9px;font-weight:600;")
        info_row.addWidget(self._save_indicator)
        lo.addLayout(info_row)
        
        return g

    def _build_model_export_section(self):
        g = QGroupBox("Model & Export")
        lo = QVBoxLayout(g)
        lo.setSpacing(5)
        lo.setContentsMargins(8, 8, 8, 8)
        
        # 模式选择
        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        self._mode_manual = QRadioButton("Manual")
        self._mode_manual.setStyleSheet(f"font-size:10px;font-weight:500;color:{TEXT};")
        self._mode_auto = QRadioButton("Auto")
        self._mode_auto.setStyleSheet(f"font-size:10px;font-weight:500;color:{TEXT};")
        self._mode_manual.setChecked(True)
        mode_row.addWidget(self._mode_manual)
        mode_row.addWidget(self._mode_auto)
        mode_row.addStretch()
        lo.addLayout(mode_row)
        
        # Auto 配置面板
        self._auto_panel = QWidget()
        alo = QVBoxLayout(self._auto_panel)
        alo.setContentsMargins(0, 4, 0, 0)
        alo.setSpacing(4)
        
        # 模型选择和Auto Label按钮在同一行
        ml2 = QHBoxLayout()
        ml2.setSpacing(4)
        self._model_combo = QComboBox()
        self._model_combo.setMinimumHeight(24)
        ml2.addWidget(self._model_combo, 1)
        ml2.addWidget(self._make_tool_btn("📁", self._browse_model))
        self._auto_btn = QPushButton("▶")
        self._auto_btn.setObjectName("pri")
        self._auto_btn.setMinimumHeight(24)
        self._auto_btn.setFixedWidth(32)
        self._auto_btn.clicked.connect(self._start_auto)
        ml2.addWidget(self._auto_btn)
        alo.addLayout(ml2)
        
        # Conf 和 IoU
        pr = QHBoxLayout()
        pr.setSpacing(4)
        self._al_conf = QDoubleSpinBox()
        self._al_conf.setRange(0.01, 0.99)
        self._al_conf.setValue(0.25)
        self._al_conf.setSingleStep(0.05)
        self._al_conf.setDecimals(2)
        self._al_conf.setMinimumHeight(24)
        pr.addWidget(QLabel("Conf", styleSheet=f"font-size:9px;color:{TEXT2};min-width:32px;"))
        pr.addWidget(self._al_conf, 1)
        self._al_iou = QDoubleSpinBox()
        self._al_iou.setRange(0.01, 0.99)
        self._al_iou.setValue(0.45)
        self._al_iou.setSingleStep(0.05)
        self._al_iou.setDecimals(2)
        self._al_iou.setMinimumHeight(24)
        pr.addWidget(QLabel("IoU", styleSheet=f"font-size:9px;color:{TEXT2};min-width:28px;"))
        pr.addWidget(self._al_iou, 1)
        alo.addLayout(pr)
        
        self._auto_panel.setEnabled(False)
        lo.addWidget(self._auto_panel)
        
        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet(f"background:{BORDER};")
        lo.addWidget(sep)
        
        # Export 配置和按钮在同一行
        export_row = QHBoxLayout()
        export_row.setSpacing(6)
        
        # Train/Val 比例
        sr = QHBoxLayout()
        sr.setSpacing(4)
        sr.addWidget(QLabel("Train", styleSheet=f"font-size:9px;color:{TEXT2};min-width:32px;"))
        self._train_ratio = QSpinBox()
        self._train_ratio.setRange(50, 99)
        self._train_ratio.setValue(90)
        self._train_ratio.setSuffix(" %")
        self._train_ratio.setMinimumHeight(24)
        sr.addWidget(self._train_ratio, 1)
        sr.addWidget(QLabel("Val", styleSheet=f"font-size:9px;color:{TEXT3};min-width:24px;"))
        self._val_label = QLabel("10%")
        self._val_label.setStyleSheet(f"font-size:10px;color:{TEXT3};font-weight:600;")
        sr.addWidget(self._val_label)
        self._train_ratio.valueChanged.connect(lambda v: self._val_label.setText(f"{100 - v}%"))
        export_row.addLayout(sr, 1)
        
        # 导出按钮
        self._export_btn = QPushButton("📦 Export")
        self._export_btn.setObjectName("pri")
        self._export_btn.setMinimumHeight(24)
        self._export_btn.clicked.connect(self._export)
        export_row.addWidget(self._export_btn)
        
        lo.addLayout(export_row)
        
        # 进度条和状态
        status_row = QHBoxLayout()
        status_row.setSpacing(4)
        self._export_bar = QProgressBar()
        self._export_bar.setMinimumHeight(4)
        self._export_bar.setTextVisible(False)
        status_row.addWidget(self._export_bar, 1)
        self._export_status = QLabel("Ready")
        self._export_status.setStyleSheet(
            f"font-size:8px;color:{TEXT3};padding:2px 4px;background:{BG};"
            f"border-radius:3px;border:1px solid {BORDER};")
        status_row.addWidget(self._export_status, 0)
        lo.addLayout(status_row)
        
        self._mode_manual.toggled.connect(self._on_mode_changed)
        return g

    def _build_stats_section(self):
        g = QGroupBox("Class Distribution")
        lo = QVBoxLayout(g)
        lo.setSpacing(6)
        lo.setContentsMargins(8, 8, 8, 8)
        
        # 表头
        header = QHBoxLayout()
        header.setSpacing(4)
        header.addWidget(QLabel("Class", styleSheet=f"font-size:9px;color:{TEXT2};font-weight:500;min-width:60px;"))
        header.addWidget(QLabel("Count", styleSheet=f"font-size:9px;color:{TEXT2};font-weight:500;min-width:40px;"))
        header.addWidget(QLabel("%", styleSheet=f"font-size:9px;color:{TEXT2};font-weight:500;min-width:36px;"))
        header.addWidget(QLabel("Distribution", styleSheet=f"font-size:9px;color:{TEXT2};font-weight:500;"), 1)
        lo.addLayout(header)
        
        # 总计信息
        self._stats_summary = QLabel("0 images · 0 instances")
        self._stats_summary.setStyleSheet(
            f"font-size:9px;color:{TEXT3};padding:4px 6px;background:{BG};"
            f"border-radius:4px;border:1px solid {BORDER};")
        lo.addWidget(self._stats_summary)
        
        # 类别统计列表
        self._stats_list = QWidget()
        self._stats_layout = QVBoxLayout(self._stats_list)
        self._stats_layout.setSpacing(4)
        self._stats_layout.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(self._stats_list, 1)
        
        return g

    def _build_annotation_section(self):
        g = QGroupBox("Annotations")
        lo = QVBoxLayout(g)
        lo.setSpacing(6)
        lo.setContentsMargins(10, 10, 10, 10)
        self._ann_list = QListWidget()
        self._ann_list.setStyleSheet(
            f"QListWidget {{ background:{BG}; border:1px solid {BORDER}; border-radius:5px;"
            f"font-size:10px; color:{TEXT}; padding:3px; min-height:120px; }}"
            f"QListWidget::item {{ padding:4px 8px; border-radius:3px;"
            f"border-bottom:1px solid {BORDER}; }}"
            f"QListWidget::item:selected {{ background:{PRI}25; color:{TEXT}; border:none; }}")
        self._ann_list.currentRowChanged.connect(self._on_ann_list_select)
        lo.addWidget(self._ann_list)
        return g

    def eventFilter(self, obj, event):
        """捕获快捷键设置框的按键输入"""
        if event.type() == QEvent.KeyPress and isinstance(obj, QLineEdit):
            key = event.key()
            modifiers = event.modifiers()
            
            # 忽略特殊键（Control, Alt, Shift等修饰键）
            if key in (Qt.Key_Control, Qt.Key_Alt, Qt.Key_Shift, Qt.Key_Meta):
                return True
            
            # 构建快捷键字符串
            parts = []
            if modifiers & Qt.ControlModifier:
                parts.append("Ctrl")
            if modifiers & Qt.AltModifier:
                parts.append("Alt")
            if modifiers & Qt.ShiftModifier:
                parts.append("Shift")
            
            # 获取按键名称
            key_name = None
            if Qt.Key_0 <= key <= Qt.Key_9:
                key_name = str(key - Qt.Key_0)
            elif Qt.Key_A <= key <= Qt.Key_Z:
                key_name = chr(key)
            elif key == Qt.Key_Space:
                key_name = "Space"
            elif key == Qt.Key_Enter or key == Qt.Key_Return:
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
            elif key == Qt.Key_Insert:
                key_name = "Insert"
            elif key == Qt.Key_Home:
                key_name = "Home"
            elif key == Qt.Key_End:
                key_name = "End"
            elif key == Qt.Key_PageUp:
                key_name = "PgUp"
            elif key == Qt.Key_PageDown:
                key_name = "PgDn"
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
                shortcut_str = "+".join(parts)
                obj.setText(shortcut_str)
            
            return True
        
        return super().eventFilter(obj, event)

    def _make_tool_btn(self, text, callback, w=36, h=28):
        btn = QPushButton(text)
        btn.setMinimumSize(w, h)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setStyleSheet(
            f"QPushButton {{ background:{BG}; border:1px solid {BORDER}; border-radius:4px;"
            f"color:{TEXT}; font-size:13px; font-weight:500; }}"
            f"QPushButton:hover {{ background:{PRI}20; border-color:{PRI}; }}"
            f"QPushButton:pressed {{ background:{PRI}40; }}")
        btn.clicked.connect(callback)
        return btn

    def _make_icon_btn(self, text):
        btn = QPushButton(text)
        btn.setMinimumSize(30, 28)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setStyleSheet(
            f"QPushButton {{ background:{BG}; border:1px solid {BORDER}; border-radius:5px;"
            f"color:{TEXT}; font-size:13px; font-weight:600; }}"
            f"QPushButton:hover {{ background:{PRI}20; border-color:{PRI}; }}"
            f"QPushButton:disabled {{ color:{TEXT3}; border-color:{BORDER}; }}")
        return btn

    def _build_class_buttons(self):
        for btn in self._class_btns:
            self._class_layout.removeWidget(btn)
            btn.deleteLater()
        self._class_btns.clear()
        while self._class_layout.count() > 1:
            item = self._class_layout.itemAt(0)
            if item is not None:
                self._class_layout.removeItem(item)
        self._class_ids = sorted(int(k) for k in CLASS_NAMES.keys())
        
        # 使用QGridLayout实现两列布局
        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)
        
        for i, class_id in enumerate(self._class_ids):
            row = i // 2  # 每行2个
            col = i % 2   # 0或1
            
            name = resolve_class_name(class_id)
            btn = QPushButton(name)
            btn.setMinimumHeight(32)
            btn.setCheckable(True)
            btn.setStyleSheet(
                f"QPushButton {{ background:{BG}; border:1px solid {BORDER}; border-radius:5px;"
                f"color:{TEXT}; font-size:11px; font-weight:500; padding:4px 8px; text-align:left; }}"
                f"QPushButton:checked {{ background:{PRI}; color:#fff; border:1px solid {PRI}; }}"
                f"QPushButton:hover {{ background:{PRI}20; border-color:{PRI}; }}")
            btn.clicked.connect(lambda checked, cid=class_id: self._on_class_selected(cid))
            if i == 0:
                btn.setChecked(True)
                self.canvas.set_current_class(class_id)
            grid.addWidget(btn, row, col)
            self._class_btns.append(btn)
        
        self._class_layout.insertLayout(0, grid)
    # ═══════════════ EVENTS ═══════════════

    def _on_class_selected(self, class_id):
        # 同步更新Classes区域的按钮
        btn_idx = self._class_ids.index(class_id) if class_id in self._class_ids else -1
        for i, btn in enumerate(self._class_btns):
            btn.setChecked(i == btn_idx)
        
        self.canvas.set_current_class(class_id)
        if self.canvas.selected_idx >= 0:
            self.canvas.annotations[self.canvas.selected_idx].class_id = class_id
            self.canvas.annotation_changed.emit()
            self.canvas.update()

    def _on_mode_changed(self):
        is_auto = self._mode_auto.isChecked()
        # 启用/禁用auto面板中的组件，而不是隐藏
        self._auto_panel.setEnabled(is_auto)
        self._is_auto_mode = is_auto

    def _on_annotation_changed(self):
        self._has_unsaved = True
        self._save_indicator.setText("● Unsaved")
        self._save_indicator.setStyleSheet(f"font-size:10px;color:{AMBER};font-weight:600;")
        self._update_ann_list()
        self._update_stats()
        self._save_session()

    def _on_ann_list_select(self, row):
        if 0 <= row < len(self.canvas.annotations) and self.canvas.selected_idx != row:
            self.canvas.selected_idx = row
            self.canvas.update()

    def _on_folder_selected(self):
        folder = self._src_combo.currentText()
        if not folder:
            return
        src = self._after_root / folder
        self._src_path_label.setText(f"📂 {src}")
        self._load_images(src)

    # ═══════════════ IMAGE LOADING ═══════════════

    def _load_images(self, folder: Path):
        self._image_paths = sorted(folder.rglob("*.jpg"))
        self._current_idx = 0
        self._annotations = {}
        self._auto_anns = {}
        self._session_file = folder / "_annotations.json"
        if self._session_file.exists():
            try:
                data = json.loads(self._session_file.read_text("utf-8"))
                saved = data.get("annotations", {})
                curr = data.get("current_idx", 0)
                for k, v in saved.items():
                    self._annotations[k] = [Annotation.from_dict(a) for a in v]
                if self._image_paths:
                    self._current_idx = min(curr, len(self._image_paths) - 1)
            except Exception as e:
                print(f"Load annotations error: {e}")
        self._update_counts()
        self._update_stats()
        if self._image_paths:
            self._show_image(self._current_idx)
        else:
            self._count_label.setText("0 images")

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

    def _show_image(self, idx):
        if idx < 0 or idx >= len(self._image_paths):
            return
        img_path = self._image_paths[idx]
        self._current_image = cv2.imread(str(img_path))
        if self._current_image is None:
            return
        self.canvas.set_image(self._current_image)
        key = self._img_key(img_path)
        if key in self._annotations:
            self.canvas.set_annotations(self._annotations[key])
        elif key in self._auto_anns:
            anns = [Annotation.from_dict(d) for d in self._auto_anns[key]]
            self.canvas.set_annotations(anns)
        self._current_idx = idx
        self._idx_input.blockSignals(True)
        self._idx_input.setValue(idx + 1)
        self._idx_input.blockSignals(False)
        self._total_label.setText(f"/ {len(self._image_paths)}")
        self._img_name_label.setText(f"{img_path.parent.name}/{img_path.name}")
        self._update_ann_list()
        self._update_nav_buttons()

    def keyPressEvent(self, event):
        """处理键盘快捷键"""
        # 当删除确认框可见时，阻止所有快捷键操作
        if hasattr(self, '_delete_confirm') and self._delete_confirm.isVisible():
            event.ignore()
            return
        
        # 检查是否有自定义快捷键配置
        if not hasattr(self, '_shortcut_inputs') or not self._shortcut_inputs:
            super().keyPressEvent(event)
            return
        
        key = event.key()
        
        # 获取当前设置的快捷键（避免创建临时对象）
        prev_input = self._shortcut_inputs.get('prev')
        next_input = self._shortcut_inputs.get('next')
        delete_box_input = self._shortcut_inputs.get('delete_box')
        delete_img_input = self._shortcut_inputs.get('delete_img')
        
        if not all([prev_input, next_input, delete_box_input, delete_img_input]):
            super().keyPressEvent(event)
            return
        
        prev_key = prev_input.text()
        next_key = next_input.text()
        delete_box_key = delete_box_input.text()
        delete_img_key = delete_img_input.text()
        
        # 构建按键名称映射
        key_map = {
            Qt.Key_Left: "Left",
            Qt.Key_Right: "Right",
            Qt.Key_Up: "Up",
            Qt.Key_Down: "Down",
            Qt.Key_Delete: "Delete",
            Qt.Key_A: "A",
            Qt.Key_D: "D",
            Qt.Key_W: "W",
            Qt.Key_S: "S",
            Qt.Key_1: "1",
            Qt.Key_2: "2",
            Qt.Key_3: "3",
            Qt.Key_4: "4",
        }
        
        current_key_name = key_map.get(key)
        
        # 只有当当前按键有对应的名称时才比较
        if current_key_name:
            if current_key_name == prev_key:
                self._navigate(-1)
                return
            elif current_key_name == next_key:
                self._navigate(1)
                return
            elif current_key_name == delete_box_key:
                self._delete_selected()
                return
            elif current_key_name == delete_img_key:
                self._delete_image()
                return
            # 处理类别快捷键（1-4）
            elif current_key_name in ["1", "2", "3", "4"]:
                class_id = int(current_key_name) - 1
                if 0 <= class_id < len(self._class_ids):
                    self._on_class_selected(self._class_ids[class_id])
                    return
        
        super().keyPressEvent(event)

    def _img_key(self, path):
        return str(path)

    def _save_current_annotations(self):
        if not self._image_paths or self._current_image is None:
            return
        key = self._img_key(self._image_paths[self._current_idx])
        if self.canvas.annotations:
            self._annotations[key] = list(self.canvas.annotations)
        elif key in self._annotations:
            del self._annotations[key]

    # ═══════════════ SESSION ═══════════════

    def _save_session(self):
        if not self._session_file:
            return
        self._save_current_annotations()
        data = {
            "current_idx": self._current_idx,
            "annotations": {k: [a.to_dict() for a in v] for k, v in self._annotations.items()}
        }
        try:
            self._session_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
            self._has_unsaved = False
            self._save_indicator.setText("✓ Saved")
            self._save_indicator.setStyleSheet(f"font-size:10px;color:{GREEN};font-weight:600;")
        except Exception:
            self._save_indicator.setText("✕ Save failed")
            self._save_indicator.setStyleSheet(f"font-size:10px;color:{RED};")

    # ═══════════════ ACTIONS ═══════════════

    def _delete_selected(self):
        self.canvas.delete_selected()
        self._update_stats()

    def _delete_image(self):
        """删除当前图片及其标注"""
        if not self._image_paths or self._current_idx < 0:
            return
        
        current_path = self._image_paths[self._current_idx]
        boxes = len(self.canvas.annotations)
        self._pending_delete_path = current_path
        self._confirm_msg.setText(
            f"Delete this image and{' its' if boxes else ''} data? "
            f"<b>{current_path.name}</b> "
            f"({boxes} box{'es' if boxes != 1 else ''}) — This cannot be undone."
        )
        self._delete_confirm.setVisible(True)

    def _confirm_delete_image(self):
        """执行删除图片操作（确认后）"""
        self._delete_confirm.setVisible(False)
        current_path = self._pending_delete_path
        if current_path is None:
            return

        try:
            # 删除图片文件
            if current_path.exists():
                current_path.unlink()
                print(f"Deleted image: {current_path.name}")
            
            # 删除标注文件（如果存在）
            label_path = self._label_root / current_path.parent.name / f"{current_path.stem}.txt"
            if label_path.exists():
                label_path.unlink()
                print(f"Deleted label: {label_path.name}")
            
            # 从内存中移除标注
            key = self._img_key(current_path)
            if key in self._annotations:
                del self._annotations[key]
            if key in self._auto_anns:
                del self._auto_anns[key]
            
            # 从图片列表中移除
            self._image_paths.pop(self._current_idx)
            
            # 调整索引
            if self._current_idx >= len(self._image_paths):
                self._current_idx = max(0, len(self._image_paths) - 1)
            
            # 更新显示
            if self._image_paths:
                self._show_image(self._current_idx)
            else:
                self.canvas.clear_image()
                self._count_label.setText("0 images")
                self._annotated_label.setText("0 annotated")
                self._img_name_label.setText("—")
                self._ann_count_label.setText("0 boxes")
            
            # 更新计数
            self._update_counts()
            self._update_stats()
            
            # 保存session
            self._save_session()
            
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", f"Failed to delete image:\n{str(e)}")
            print(f"Delete image error: {e}")

    def _clear_annotations(self):
        self.canvas.clear_annotations()
        self._update_stats()

    def _random_filter_dataset(self):
        """数据集随机筛选：每3张图片为一组，随机删除2张，保留1张"""
        if not self._image_paths:
            QMessageBox.warning(self, "Warning", "No images to filter")
            return
        
        total = len(self._image_paths)
        if total < 3:
            QMessageBox.warning(self, "Warning", f"Need at least 3 images (current: {total})")
            return
        
        # 计算筛选后的数量
        remaining = (total + 2) // 3  # 向上取整
        deleted = total - remaining
        
        # 确认对话框
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Random Filter Dataset")
        msg.setText(f"Are you sure you want to randomly filter the dataset?")
        msg.setInformativeText(
            f"Current: {total} images\n"
            f"After filter: {remaining} images\n"
            f"Will delete: {deleted} images\n\n"
            f"This action cannot be undone!"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        reply = msg.exec_()
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # 保存当前标注
            self._save_current_annotations()
            
            import random
            
            # 按原顺序每3张一组，每组随机保留1张，删除2张
            paths_to_delete = []
            paths_to_keep = []
            
            for i in range(0, len(self._image_paths), 3):
                group = list(self._image_paths[i:i+3])
                if group:
                    keep = random.choice(group)
                    paths_to_keep.append(keep)
                    for p in group:
                        if p != keep:
                            paths_to_delete.append(p)
            
            # 执行删除
            deleted_count = 0
            for img_path in paths_to_delete:
                try:
                    # 删除图片文件
                    if img_path.exists():
                        img_path.unlink()
                    
                    # 删除标注文件（如果存在）
                    label_path = self._label_root / img_path.parent.name / f"{img_path.stem}.txt"
                    if label_path.exists():
                        label_path.unlink()
                    
                    # 从内存中移除标注
                    key = self._img_key(img_path)
                    if key in self._annotations:
                        del self._annotations[key]
                    if key in self._auto_anns:
                        del self._auto_anns[key]
                    
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {img_path.name}: {e}")
            
            # 更新图片列表为保留的图片
            self._image_paths = sorted(paths_to_keep)
            
            # 调整当前索引
            if self._current_idx >= len(self._image_paths):
                self._current_idx = max(0, len(self._image_paths) - 1)
            
            # 重新加载当前图片
            if self._image_paths:
                self._show_image(self._current_idx)
            else:
                self.canvas.clear_image()
                self._count_label.setText("0 images")
                self._annotated_label.setText("0 annotated")
                self._img_name_label.setText("—")
                self._ann_count_label.setText("0 boxes")
            
            # 更新计数
            self._update_counts()
            self._update_stats()
            
            # 保存session
            self._save_session()
            
            # 显示结果
            QMessageBox.information(
                self, 
                "Filter Complete",
                f"Dataset filtered successfully!\n\n"
                f"Deleted: {deleted_count} images\n"
                f"Remaining: {len(self._image_paths)} images"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", f"Failed to filter dataset:\n{str(e)}")
            print(f"Filter dataset error: {e}")

    def _update_ann_list(self):
        self._ann_list.blockSignals(True)
        self._ann_list.clear()
        for i, ann in enumerate(self.canvas.annotations):
            name = resolve_class_name(ann.class_id)
            self._ann_list.addItem(f"[{i}] {name}  ({ann.xc:.3f}, {ann.yc:.3f})  {ann.w:.3f}x{ann.h:.3f}")
            if i == self.canvas.selected_idx:
                self._ann_list.setCurrentRow(i)
        self._ann_list.blockSignals(False)
        self._ann_count_label.setText(f"{len(self.canvas.annotations)} boxes")

    def _update_nav_buttons(self):
        self._prev_btn.setEnabled(self._current_idx > 0)
        self._next_btn.setEnabled(self._current_idx < len(self._image_paths) - 1)

    def _update_counts(self):
        n = len(self._image_paths)
        annotated = len([k for k in self._annotations if self._annotations[k]])
        self._count_label.setText(f"{n} images")
        self._annotated_label.setText(f"{annotated} annotated")
        
        if hasattr(self, '_filter_stats_label'):
            remaining = (n + 2) // 3 if n >= 3 else n
            self._filter_stats_label.setText(f"{n} → {remaining} images")
        
        self._update_stats()

    def _update_stats(self):
        """更新类别统计信息"""
        if not hasattr(self, '_stats_layout'):
            return
        
        # 清除旧控件
        while self._stats_layout.count():
            item = self._stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 统计所有标注中的类别
        class_counts = {}
        total_boxes = 0
        total_images = len(self._annotations)
        
        for anns in self._annotations.values():
            for ann in anns:
                class_id = ann.class_id
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total_boxes += 1
        
        # 更新总计信息
        self._stats_summary.setText(f"{total_images} images · {total_boxes} instances")
        
        # 显示每个类别的统计
        if class_counts:
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                name = resolve_class_name(class_id)
                color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
                percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
                
                # 创建行容器
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setSpacing(6)
                row_layout.setContentsMargins(0, 0, 0, 0)
                
                # 颜色点 + 类别名
                name_layout = QHBoxLayout()
                name_layout.setSpacing(4)
                dot = QLabel("●")
                dot.setStyleSheet(f"color:{color};font-size:12px;")
                name_layout.addWidget(dot)
                name_label = QLabel(name)
                name_label.setStyleSheet(f"font-size:10px;color:{TEXT};font-weight:500;")
                name_layout.addWidget(name_label)
                name_layout.addStretch()
                row_layout.addLayout(name_layout, 1)
                
                # 数量
                count_label = QLabel(str(count))
                count_label.setStyleSheet(f"font-size:10px;color:{TEXT};font-weight:500;min-width:36px;")
                row_layout.addWidget(count_label)
                
                # 百分比
                pct_label = QLabel(f"{percentage:.1f}%")
                pct_label.setStyleSheet(f"font-size:10px;color:{TEXT2};min-width:40px;")
                row_layout.addWidget(pct_label)
                
                # 进度条
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(int(percentage))
                bar.setTextVisible(False)
                bar.setFixedHeight(8)
                bar.setStyleSheet(
                    f"QProgressBar {{ background:#f0f0f0; border-radius:4px; border:none; }}"
                    f"QProgressBar::chunk {{ background:{color}; border-radius:4px; }}")
                row_layout.addWidget(bar, 1)
                
                self._stats_layout.addWidget(row_widget)
        else:
            no_data = QLabel("No annotations yet")
            no_data.setStyleSheet(f"font-size:10px;color:{TEXT3};padding:8px;text-align:center;")
            no_data.setAlignment(Qt.AlignCenter)
            self._stats_layout.addWidget(no_data)

    # ═══════════════ SOURCE ═══════════════

    def _refresh_source_folders(self):
        self._src_combo.blockSignals(True)
        self._src_combo.clear()
        if self._after_root.exists():
            dirs = sorted([d.name for d in self._after_root.iterdir() if d.is_dir()])
            if dirs:
                self._src_combo.addItems(dirs)
                self._src_combo.blockSignals(False)
                self._on_folder_selected()
                return
        self._src_combo.blockSignals(False)
        self._src_path_label.setText(f"⚠️ No subfolders in {self._after_root}")

    # ═══════════════ AUTO LABEL ═══════════════

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Model", "runs", "PyTorch (*.pt)")
        if p:
            existing = [self._model_combo.itemText(i) for i in range(self._model_combo.count())]
            rel = Path(p)
            display = rel.name
            if display not in existing:
                self._model_combo.addItem(display, p)
            self._model_combo.setCurrentText(display)

    def _scan_models(self):
        self._model_combo.clear()
        models = sorted(ROOT.rglob("weights/best.pt"))
        for m in models:
            display = str(m.relative_to(ROOT))
            self._model_combo.addItem(display, str(m))
        for m in sorted(ROOT.glob("*.pt")):
            if m.name not in [self._model_combo.itemText(i) for i in range(self._model_combo.count())]:
                self._model_combo.addItem(m.name, str(m))

    def _start_auto(self):
        if not self._image_paths:
            QMessageBox.warning(self, "Warning", "Please select a folder first")
            return
        model_path = self._model_combo.currentData()
        if not model_path or not Path(model_path).exists():
            self._scan_models()
            model_path = self._model_combo.currentData()
            if not model_path or not Path(model_path).exists():
                QMessageBox.warning(self, "Warning", "Please select a model file")
                return
        self._auto_anns = {}
        self._auto_btn.setEnabled(False)
        self._auto_btn.setText("⏳")
        self._export_bar.setValue(0)
        self._worker = AutoLabelWorker(
            model_path=model_path,
            image_paths=[str(p) for p in self._image_paths],
            conf=self._al_conf.value(),
            iou=self._al_iou.value(),
        )
        self._worker.progress.connect(lambda c, t: self._export_bar.setValue(int(c / t * 100)))
        self._worker.image_done.connect(self._on_auto_image_done)
        self._worker.log.connect(lambda m: self._export_status.setText(m))
        self._worker.done.connect(self._on_auto_done)
        self._worker.start()

    def _on_auto_image_done(self, img_path, anns):
        self._auto_anns[img_path] = anns

    def _on_auto_done(self, ok, msg):
        self._auto_btn.setEnabled(True)
        self._auto_btn.setText("▶")
        if ok and self._auto_anns:
            for img_path, ann_dicts in self._auto_anns.items():
                if ann_dicts:
                    self._annotations[img_path] = [Annotation.from_dict(d) for d in ann_dicts]
            key = self._img_key(self._image_paths[self._current_idx])
            if key in self._annotations:
                self.canvas.set_annotations(self._annotations[key])
            self._save_session()
            self._update_counts()
            self._update_stats()
        self._export_bar.setValue(100 if ok else 0)

    # ═══════════════ EXPORT ═══════════════

    def _export(self):
        self._save_current_annotations()
        
        # 确保所有已加载的图片在 _annotations 中都有条目（未标注的初始化为空列表）
        for p in self._image_paths:
            key = self._img_key(p)
            if key not in self._annotations:
                self._annotations[key] = []
        
        # Check for images with empty annotations
        labeled = {k: v for k, v in self._annotations.items() if v}
        unlabeled = {k: v for k, v in self._annotations.items() if not v}
        
        total = len(self._annotations)
        labeled_count = len(labeled)
        unlabeled_count = len(unlabeled)
        
        if total == 0:
            QMessageBox.warning(self, "Warning", "No annotation data to export")
            return
        
        # Prompt confirmation if there are images with empty annotations
        if unlabeled_count > 0:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Export Dataset")
            msg.setText("Some images have empty annotations.")
            msg.setInformativeText(
                f"Total images: {total}\n"
                f"With annotations: {labeled_count}\n"
                f"Empty annotations: {unlabeled_count}\n\n"
                f"Images with empty annotations will still be exported\n"
                f"(with empty .txt files). Continue?"
            )
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.Yes)
            reply = msg.exec_()
            
            if reply != QMessageBox.Yes:
                return
        
        self._export_btn.setEnabled(False)
        self._export_bar.setValue(0)
        self._export_worker = ExportWorker(
            self._annotations, self._label_root, self._train_ratio.value(), ROOT)
        self._export_worker.progress.connect(lambda v: self._export_bar.setValue(v))
        self._export_worker.log.connect(lambda m: self._export_status.setText(m))
        self._export_worker.done.connect(self._on_export_done)
        self._export_worker.start()

    def _on_export_done(self, ok, msg):
        self._export_btn.setEnabled(True)
        if ok:
            QMessageBox.information(self, "Export Complete", msg)
        self._export_worker = None
