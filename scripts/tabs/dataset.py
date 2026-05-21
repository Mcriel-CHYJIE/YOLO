"""数据集预览标签页"""
from scripts.tabs.base import *
from PyQt5 import uic
import yaml, random, shutil
from collections import Counter
from PIL import Image


class DatasetTab(QWidget):
    COLORS = ['#ef4444','#10b981','#3b82f6','#f59e0b',
              '#8b5cf6','#ec4899','#14b8a6','#f97316']

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._pixmaps = []; self._images = []; self._names = []
        self._build()
        self._init_widgets()
        self._connect_signals()

    def _build(self):
        ui_path = Path(__file__).resolve().parent.parent / 'ui' / 'dataset.ui'
        uic.loadUi(str(ui_path), self)

    def _init_widgets(self):
        self.leftPanel.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:6px;')
        self.rightPanel.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:6px;')
        self.sa.setStyleSheet('QScrollArea{background:transparent;border:none;}')
        self.dp_grid.setStyleSheet('background:transparent;')
        self._cols = 3
        self.dp_status.setText('Ready - Click Refresh to load dataset preview')
        # 设置按钮最小高度
        for btn in [self.btn_train, self.btn_val, self.dp_rf, self.dp_import, self.dp_fix_yaml]:
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
        # 设置按钮互斥逻辑
        self.btn_train.clicked.connect(self._on_split_click)
        self.btn_val.clicked.connect(self._on_split_click)

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

    def _connect_signals(self):
        self.dp_rf.setObjectName('pri')
        self.dp_rf.clicked.connect(self._dp_refresh)
        self.dp_import.setObjectName('sec')
        self.dp_import.clicked.connect(self._import_dataset)
        self.dp_fix_yaml.setObjectName('warn')
        self.dp_fix_yaml.clicked.connect(self._fix_yaml_config)

    def _dp_refresh(self):
        self.dp_status.setText('Loading...'); QApplication.processEvents()
        # 根据按钮选中状态确定 split
        split = 'train' if self.btn_train.isChecked() else 'val'
        img_dir = ROOT / 'datasets' / 'images' / split
        lbl_dir = ROOT / 'datasets' / 'labels' / split
        if not img_dir.exists() or not lbl_dir.exists():
            self.dp_status.setText(f'Split "{split}" not found in datasets/'); return
        imgs = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')) + \
               sorted(img_dir.glob('*.jpeg')) + sorted(img_dir.glob('*.webp'))
        if not imgs: self.dp_status.setText(f'No images in {split}'); return
        self._images = []; labeled = 0; cls_counter = Counter()
        for img_path in imgs:
            lbl_path = lbl_dir / f'{img_path.stem}.txt'
            has_lbl = lbl_path.exists() and lbl_path.stat().st_size > 0
            if has_lbl: labeled += 1
            self._images.append((img_path, lbl_path))
            if has_lbl:
                for line in lbl_path.read_text().strip().split('\n'):
                    if line.strip(): cls_counter[int(line.strip().split()[0])] += 1
        self._st_total_v.setText(f'{len(imgs)}')
        self._st_lbl_v.setText(f'{labeled}')
        unlabeled = len(imgs) - labeled
        total_instances = sum(cls_counter.values())
        self._st_cls_v.setText(f'{total_instances}')
        self._st_cls_text.setText('Instances')
        self._st_total_lbl.setText('Total')
        self._st_lbl_text.setText('Labeled')
        self._update_class_stats(cls_counter, labeled)
        self._clear_grid()
        # 随机选9张预览，少于9张则全显示
        preview = random.sample(self._images, min(9, len(self._images)))
        for idx, (img_path, lbl_path) in enumerate(preview):
            r, c = divmod(idx, self._cols)
            self.dp_gl.addWidget(self._make_card(img_path, lbl_path), r, c)
        shown = len(preview)
        status_parts = [f'{len(imgs)} images']
        if unlabeled:
            status_parts.append(f'{unlabeled} unlabeled')
        if shown < len(imgs):
            status_parts.append(f'{shown} previewed')
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

    def _fix_yaml_config(self):
        try:
            data_dir = ROOT / 'datasets'; yaml_path = data_dir / 'data.yaml'
            img_dir = data_dir / 'images'
            if not img_dir.exists(): QMessageBox.warning(self, 'Error', f'images dir not found: {img_dir}'); return
            splits = [d.name for d in img_dir.iterdir() if d.is_dir()]
            if not splits: QMessageBox.warning(self, 'Error', 'No split directories found'); return
            lbl_dir = data_dir / 'labels'; all_cls = set()
            for split in splits:
                ls = lbl_dir / split
                if ls.exists():
                    for f in ls.glob('*.txt'):
                        if f.stat().st_size > 0:
                            for line in f.read_text().strip().split('\n'):
                                if line.strip(): all_cls.add(int(line.strip().split()[0]))
            if not all_cls: all_cls = set(range(len(CLASSES)))
            nc = max(all_cls) + 1
            data = {'path': str(data_dir).replace('\\', '/'), 'train': 'images/train', 'val': 'images/val',
                    'nc': nc, 'names': {i: CLASSES[i] if i < len(CLASSES) else f'class_{i}' for i in range(nc)}}
            with open(yaml_path, 'w') as f: yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            self.dp_status.setText(f'data.yaml generated: {nc} classes')
            QMessageBox.information(self, 'YAML Fixed', f'{yaml_path}\n{nc} classes, {len(splits)} splits')
        except Exception as e: QMessageBox.critical(self, 'Error', f'Failed: {e}')

    def _import_dataset(self):
        """自动从 original/label 文件夹导入数据集"""
        src = ROOT / 'original' / 'label'
        if not src.exists():
            QMessageBox.warning(self, 'Error', f'Label directory not found: {src}')
            return
        
        # 检查是否有 images 和 labels 子目录
        img_src = src / 'images'
        lbl_src = src / 'labels'
        if not img_src.exists() or not lbl_src.exists():
            QMessageBox.warning(self, 'Error', 
                f'Invalid dataset structure. Expected:\n{src}/images/train, images/val\n{src}/labels/train, labels/val')
            return
        
        # 统计将要导入的文件数量
        total_images = 0
        for split in ['train', 'val']:
            si = img_src / split
            if si.exists():
                total_images += len([f for f in si.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg', '.webp')])
        
        if total_images == 0:
            QMessageBox.warning(self, 'Warning', 'No images found in original/label')
            return
        
        # 显示确认对话框
        reply = QMessageBox.question(
            self,
            'Confirm Import',
            f'Found {total_images} images in:\n{src}\n\n'
            f'This will copy images and labels to datasets/ folder.\n'
            f'Duplicate files will be skipped.\n\n'
            f'Continue with import?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        dst = ROOT / 'datasets'
        copied = 0
        
        for split in ['train', 'val']:
            si = img_src / split
            sl = lbl_src / split
            di = dst / 'images' / split
            dl = dst / 'labels' / split
            
            if not si.exists():
                continue
            
            di.mkdir(parents=True, exist_ok=True)
            dl.mkdir(parents=True, exist_ok=True)
            
            for f in si.iterdir():
                if f.suffix.lower() in ('.jpg', '.png', '.jpeg', '.webp'):
                    if not (di / f.name).exists():
                        shutil.copy2(f, di / f.name)
                        lbl = sl / f'{f.stem}.txt'
                        if lbl.exists():
                            shutil.copy2(lbl, dl / f'{lbl.name}')
                        copied += 1
        
        if copied > 0:
            self.dp_status.setText(f'Imported {copied} images from original/label')
            QMessageBox.information(self, 'Import Complete', 
                f'Successfully imported {copied} images from:\n{src}')
            self._dp_refresh()
        else:
            QMessageBox.warning(self, 'Warning', 'No new images to import (all files already exist)')
