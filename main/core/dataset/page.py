# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""数据集预览标签页"""
from main.core.base import *
from PyQt5 import uic
import yaml, random, shutil
from collections import Counter
from PIL import Image
from .service import load_dataset_preview, import_dataset


class DatasetTab(QWidget):
    COLORS = ['#ef4444','#10b981','#3b82f6','#f59e0b',
              '#8b5cf6','#ec4899','#14b8a6','#f97316']

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._pixmaps = []; self._images = []; self._names = []
        self._build_ui()
        self._init_widgets()
        self._connect()

    def _build_ui(self):
        ui_path = Path(__file__).resolve().parent / 'dataset.ui'
        uic.loadUi(str(ui_path), self)

    def _init_widgets(self):
        self.leftPanel.setStyleSheet(f'background:transparent;')
        self.rightPanel.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:6px;')
        self.sa.setStyleSheet('QScrollArea{background:transparent;border:none;}')
        self.dp_grid.setStyleSheet('background:transparent;')
        
        # 移除统计标签的边框样式，确保透明背景
        for label_name in ['_st_total', '_st_instances', '_st_classes']:
            label = self.findChild(QLabel, label_name)
            if label:
                # 查找并移除可能包裹的容器边框
                parent = label.parentWidget()
                if parent and parent != self.statsGroup:
                    parent.setStyleSheet('background:transparent;border:none;')
                label.setStyleSheet('font-size:11px;font-weight:600;color:#1c1917;background:transparent;border:none;')
        
        self._cols = 3
        self.dp_status.setText('Ready - Click Refresh to load dataset preview')
        # 设置按钮最小高度
        for btn in [self.btn_train, self.btn_val, self.dp_rf, self.dp_import]:
            btn.setMinimumHeight(32)
        # 设置 split 按钮样式
        for btn in [self.btn_train, self.btn_val]:
            btn.setStyleSheet(f'''
                QPushButton {{
                    background: {BG};
                    border: 1px solid {BORDER};
                    border-radius: 4px;
                    color: {TEXT};
                    font-size: 11px;
                    font-weight: 500;
                    padding: 0 8px;
                }}
                QPushButton:checked {{
                    background: #6366f1;
                    border-color: #6366f1;
                    color: white;
                }}
                QPushButton:hover:!checked {{
                    background: {CARD};
                }}
            ''')
        # 互斥逻辑
        self.btn_train.clicked.connect(self._on_split_click)
        self.btn_val.clicked.connect(self._on_split_click)
        # 初始占位行
        self._show_placeholder_stats()

    def _on_split_click(self):
        """处理 split 按钮点击，实现互斥选中"""
        sender = self.sender()
        if sender.isChecked():
            if sender == self.btn_train:
                self.btn_val.setChecked(False)
            else:
                self.btn_train.setChecked(False)
        else:
            sender.setChecked(True)  # 至少保持一个选中
        self._dp_refresh()

    # ═══════════════════════════════════════════
    # Signal Wiring
    # ═══════════════════════════════════════════
    def _connect(self):
        self.dp_rf.setStyleSheet(
            "QPushButton{background:#07C160;color:#fff;border:none;padding:5px 18px;min-height:26px;font-size:12px;font-weight:600;border-radius:4px;}QPushButton:hover{background:#06ad56;}QPushButton:disabled{background:#a5d6a5;}"
        )
        self.dp_rf.clicked.connect(self._dp_refresh)
        self.dp_import.setStyleSheet(
            "QPushButton{background:#ffffff;color:#07C160;border:1px solid #07C160;min-height:24px;font-size:11px;border-radius:4px;}QPushButton:hover{background:#e8f5e9;}"
        )
        self.dp_import.clicked.connect(self._import_dataset)
    def on_theme_changed(self):
        """主题切换时刷新内联样式"""
        from main.core.base import CARD, BORDER, BG, TEXT
        if hasattr(self, 'rightPanel'):
            self.rightPanel.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:6px;')
        for btn in (self.btn_train, self.btn_val):
            btn.setStyleSheet(f'''
                QPushButton {{background:{BG};border:1px solid {BORDER};
                    border-radius:4px;color:{TEXT};font-size:11px;font-weight:500;padding:0 8px;}}
                QPushButton:checked {{background:#6366f1;border-color:#6366f1;color:white;}}
                QPushButton:hover:!checked {{background:{CARD};}}
            ''')
    def _dp_refresh(self):
        self.dp_status.setText('Loading...'); QApplication.processEvents()
        split = 'train' if self.btn_train.isChecked() else 'val'
        data = load_dataset_preview(split)
        if 'error' in data:
            self.dp_status.setText(data['error']); return
        self._images = data['preview']  # store preview for _make_card
        self._st_total.setText(f"Total: {data['total']}")
        self._st_instances.setText(f"Instances: {data['total_instances']}")
        self._st_classes.setText(f"Classes: {data['num_classes']}")
        self._update_class_stats(Counter(data['cls_counts']), data['labeled'])
        self._clear_grid()
        for idx, item in enumerate(data['preview']):
            r, c = divmod(idx, self._cols)
            self.dp_gl.addWidget(
                self._make_card(Path(item['img_path']), Path(item['lbl_path'])), r, c)
        status_parts = [f"{data['total']} images"]
        if data['unlabeled']:
            status_parts.append(f"{data['unlabeled']} unlabeled")
        if data['preview_count'] < data['total']:
            status_parts.append(f"{data['preview_count']} previewed")
        self.dp_status.setText(' | '.join(status_parts))

    def _update_class_stats(self, cls_counter, total_imgs):
        self._clear_layout(self._cs_grid)
        for i, (cid, cnt) in enumerate(sorted(cls_counter.items())):
            color = self.COLORS[cid % len(self.COLORS)]
            name = CLASSES[cid] if cid < len(CLASSES) else f'cls_{cid}'
            pct = cnt / total_imgs * 100 if total_imgs > 0 else 0
            dot = QLabel('●'); dot.setStyleSheet(f'color:{color};font-size:10px;')
            self._cs_grid.addWidget(dot, i, 0)
            nm = QLabel(name); nm.setStyleSheet(f'font-size:10px;color:{TEXT};font-weight:500;')
            self._cs_grid.addWidget(nm, i, 1)
            cv = QLabel(str(cnt)); cv.setStyleSheet(f'font-size:10px;color:{TEXT2};')
            cv.setAlignment(Qt.AlignRight); self._cs_grid.addWidget(cv, i, 2)
            pb = QProgressBar(); pb.setRange(0, 100); pb.setValue(int(pct))
            pb.setTextVisible(False); pb.setFixedHeight(4)
            pb.setStyleSheet(f'QProgressBar{{border:none;background:{BORDER};height:4px;border-radius:2px;}}'
                             f'QProgressBar::chunk{{background:{color};border-radius:2px;}}')
            self._cs_grid.addWidget(pb, i, 3)

    def _show_placeholder_stats(self):
        """预览前显示5个灰色占位类别行，计数器显示-"""
        self._clear_layout(self._cs_grid)
        self._st_total.setText('Total: -')
        self._st_instances.setText('Instances: -')
        self._st_classes.setText('Classes: -')
        gray = '#d0d0d0'
        for i in range(10):
            name = 'abcdefghijklmnopqrstuvwxyz'[i]
            dot = QLabel('●'); dot.setStyleSheet(f'color:{gray};font-size:10px;')
            self._cs_grid.addWidget(dot, i, 0)
            nm = QLabel(name); nm.setStyleSheet(f'font-size:10px;color:{gray};font-weight:500;')
            self._cs_grid.addWidget(nm, i, 1)
            cv = QLabel('—'); cv.setStyleSheet(f'font-size:10px;color:{gray};')
            cv.setAlignment(Qt.AlignRight); self._cs_grid.addWidget(cv, i, 2)
            pb = QProgressBar(); pb.setRange(0, 100); pb.setValue(0)
            pb.setTextVisible(False); pb.setFixedHeight(4)
            pb.setStyleSheet(f'QProgressBar{{border:none;background:#eee;height:4px;border-radius:2px;}}'
                             f'QProgressBar::chunk{{background:{gray};border-radius:2px;}}')
            self._cs_grid.addWidget(pb, i, 3)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            elif item.layout(): self._clear_layout(item.layout())

    def _clear_grid(self):
        self._clear_layout(self.dp_gl)

    def _make_card(self, img_path, lbl_path):
        card = QWidget(); card.setStyleSheet(f'background:{BG};border-radius:6px;')
        hl = QHBoxLayout(card); hl.setContentsMargins(4,4,4,4); hl.setSpacing(6)
        has_lbl = lbl_path.exists() and lbl_path.stat().st_size > 0

        # ── 左侧：图片(含标注框) + 文件名 ──
        left = QWidget(); left.setStyleSheet('background:transparent;border:none;')
        ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0); ll.setSpacing(2)

        thumb = QLabel(); thumb.setAlignment(Qt.AlignCenter)
        thumb.setFixedSize(180, 180); thumb.setStyleSheet('background:transparent;')
        try:
            pil_img = Image.open(str(img_path)).convert('RGB')
            ow, oh = pil_img.size
            scale = min(178 / ow, 178 / oh)
            nw, nh = int(ow * scale), int(oh * scale)
            pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS)

            if has_lbl:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(pil_img)
                lbl_lines = [l.strip() for l in lbl_path.read_text().strip().split('\n') if l.strip()]
                for line in lbl_lines:
                    parts = line.split()
                    cid = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = int((xc - w / 2) * nw)
                    y1 = int((yc - h / 2) * nh)
                    x2 = int((xc + w / 2) * nw)
                    y2 = int((yc + h / 2) * nh)
                    color = self.COLORS[cid % len(self.COLORS)]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    nm = CLASSES[cid] if cid < len(CLASSES) else f'cls{cid}'
                    bb = draw.textbbox((0, 0), nm, font=None) if hasattr(draw, 'textbbox') else (0, 0, len(nm) * 7, 10)
                    tw, th = bb[2] - bb[0], bb[3] - bb[1]
                    draw.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=color)
                    draw.text((x1 + 2, y1 - th - 1), nm, fill='#fff')

            canvas = Image.new('RGB', (180, 180), '#f5f5f4')
            cx, cy = (180 - nw) // 2, (180 - nh) // 2
            canvas.paste(pil_img, (cx, cy))
            from PyQt5.QtGui import QPixmap
            from io import BytesIO
            buffer = BytesIO(); canvas.save(buffer, format='PNG')
            pixmap = QPixmap(); pixmap.loadFromData(buffer.getvalue())
            thumb.setPixmap(pixmap)
        except:
            thumb.setText(img_path.name)
            thumb.setStyleSheet(f'color:{TEXT3};font-size:9px;')
        ll.addWidget(thumb)

        # 文件名过长时缩略中间
        name = img_path.name
        if len(name) > 25:
            half = 10
            name = name[:half] + '...' + name[-half:]
        fn = QLabel(name); fn.setStyleSheet(f'font-size:8px;color:{TEXT2};')
        fn.setAlignment(Qt.AlignCenter)
        ll.addWidget(fn)
        hl.addWidget(left)

        # ── 右侧：类别列表 ──
        right = QWidget(); right.setStyleSheet('background:transparent;border:none;')
        rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(2)

        if has_lbl:
            cids = [int(l.strip().split()[0]) for l in lbl_path.read_text().strip().split('\n') if l.strip()]
            for cid, cnt in sorted(Counter(cids).items()):
                color = self.COLORS[cid % len(self.COLORS)]
                nm = CLASSES[cid] if cid < len(CLASSES) else f'cls{cid}'
                cl = QLabel(f'{nm} x{cnt}')
                cl.setStyleSheet(f'font-size:8px;color:{color};font-weight:500;background:transparent;')
                rl.addWidget(cl)

        rl.addStretch()
        hl.addWidget(right, 1)
        return card

    def _import_dataset(self):
        """自动从 original/label 文件夹导入数据集"""
        copied, total, err = import_dataset()
        if err:
            if 'not found' in err:
                QMessageBox.warning(self, 'Error', err)
            elif copied == 0:
                QMessageBox.warning(self, 'Warning', err)
            else:
                QMessageBox.warning(self, 'Error', err)
            return
        self.dp_status.setText(f'Imported {copied} images from original/label')
        QMessageBox.information(self, 'Import Complete',
            f'Successfully imported {copied} images from original/label')
        self._dp_refresh()
