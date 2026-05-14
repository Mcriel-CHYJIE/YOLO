"""数据集预览标签页 — 优化排布"""
from scripts.tabs.base import *
import yaml, random, shutil
from collections import Counter
from PIL import Image, ImageDraw
from io import BytesIO


class DatasetTab(QWidget):
    """数据集预览：统计摘要 + 控制栏 + 缩略图网格"""

    COLORS = ['#ef4444','#10b981','#3b82f6','#f59e0b',
              '#8b5cf6','#ec4899','#14b8a6','#f97316']

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._pixmaps = []      # 保持QPixmap引用
        self._images = []       # 当前所有标注图片
        self._names = []        # 类别名
        self._build()
        # 默认不加载，用户点击 Refresh 按钮时再加载
        self.dp_status.setText('Ready - Click Refresh to load dataset preview')

    # ═══════════════ BUILD ═══════════════

    def _build(self):
        lo = QHBoxLayout(self); lo.setContentsMargins(8,8,8,8); lo.setSpacing(8)

        # ── Left Panel: Stats + Controls + Class Distribution ──
        left_panel = QWidget()
        left_panel.setFixedWidth(260)
        left_panel.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:6px;')
        left_lo = QVBoxLayout(left_panel); left_lo.setContentsMargins(12,12,12,12); left_lo.setSpacing(8)

        # Title
        title_lbl = QLabel('📊 Dataset Preview')
        title_lbl.setStyleSheet(f'font-size:14px;font-weight:600;color:{TEXT};')
        left_lo.addWidget(title_lbl)

        # Stats blocks
        stats_widget = QWidget(); stats_widget.setStyleSheet('background:transparent;border:none;')
        stats_lo = QVBoxLayout(stats_widget); stats_lo.setContentsMargins(0,8,0,8); stats_lo.setSpacing(6)
        
        self._st_total, self._st_total_v = self._stat_block('Total', '—')
        stats_lo.addWidget(self._st_total)
        
        self._st_lbl, self._st_lbl_v = self._stat_block('Labeled', '—')
        stats_lo.addWidget(self._st_lbl)
        
        self._st_cls, self._st_cls_v = self._stat_block('Classes', '—')
        stats_lo.addWidget(self._st_cls)
        
        left_lo.addWidget(stats_widget)

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'background:{BORDER};border:none;height:1px;')
        left_lo.addWidget(sep)

        # Controls
        controls_widget = QWidget(); controls_widget.setStyleSheet('background:transparent;border:none;')
        controls_lo = QVBoxLayout(controls_widget); controls_lo.setContentsMargins(0,4,0,4); controls_lo.setSpacing(6)

        # Split selector
        split_row = QWidget(); split_row.setStyleSheet('background:transparent;border:none;')
        split_lo = QHBoxLayout(split_row); split_lo.setContentsMargins(0,0,0,0); split_lo.setSpacing(8)
        
        split_label = QLabel('Split')
        split_label.setStyleSheet(f'font-size:11px;color:{TEXT2};font-weight:500;')
        split_lo.addWidget(split_label)
        
        self.dp_split = QComboBox()
        self.dp_split.addItems(['train', 'val'])
        self.dp_split.setStyleSheet(f'''
            QComboBox {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                color: {TEXT};
            }}
            QComboBox:hover {{
                border-color: {PRI};
            }}
            QComboBox:focus {{
                border-color: {PRI};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        ''')
        self.dp_split.currentIndexChanged.connect(lambda: self._dp_refresh())
        split_lo.addWidget(self.dp_split)
        split_lo.addStretch()
        
        controls_lo.addWidget(split_row)

        # Buttons row
        btn_row = QWidget(); btn_row.setStyleSheet('background:transparent;border:none;')
        btn_lo = QHBoxLayout(btn_row); btn_lo.setContentsMargins(0,0,0,0); btn_lo.setSpacing(6)
        
        self.dp_rf = QPushButton('🔄 Refresh')
        self.dp_rf.setObjectName('pri')
        self.dp_rf.setStyleSheet(f'''
            QPushButton {{
                background: {PRI};
                color: #fff;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {PRI_H};
            }}
            QPushButton:pressed {{
                background: #4338ca;
            }}
            QPushButton:disabled {{
                background: #a5b4fc;
            }}
        ''')
        self.dp_rf.clicked.connect(self._dp_refresh)
        btn_lo.addWidget(self.dp_rf)
        
        self.dp_import = QPushButton('📥 Import')
        self.dp_import.setObjectName('pri')
        self.dp_import.setStyleSheet(f'''
            QPushButton {{
                background: #10b981;
                color: #fff;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: #059669;
            }}
            QPushButton:pressed {{
                background: #047857;
            }}
            QPushButton:disabled {{
                background: #6ee7b7;
            }}
        ''')
        self.dp_import.clicked.connect(self._import_dataset)
        btn_lo.addWidget(self.dp_import)
        
        self.dp_fix_yaml = QPushButton('🔧 Fix YAML')
        self.dp_fix_yaml.setObjectName('pri')
        self.dp_fix_yaml.setStyleSheet(f'''
            QPushButton {{
                background: #f59e0b;
                color: #fff;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: #d97706;
            }}
            QPushButton:pressed {{
                background: #b45309;
            }}
            QPushButton:disabled {{
                background: #fcd34d;
            }}
        ''')
        self.dp_fix_yaml.clicked.connect(self._fix_yaml_config)
        btn_lo.addWidget(self.dp_fix_yaml)
        
        controls_lo.addWidget(btn_row)
        left_lo.addWidget(controls_widget)

        # Separator
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f'background:{BORDER};border:none;height:1px;')
        left_lo.addWidget(sep2)

        # Class Distribution
        cs_header = QLabel('📊 Class Distribution')
        cs_header.setStyleSheet(f'font-size:12px;font-weight:600;color:{TEXT};')
        left_lo.addWidget(cs_header)

        self._cs_grid = QGridLayout()
        self._cs_grid.setSpacing(4)
        self._cs_grid.setColumnStretch(0, 0)
        self._cs_grid.setColumnStretch(1, 0)
        self._cs_grid.setColumnStretch(2, 0)
        self._cs_grid.setColumnStretch(3, 1)
        left_lo.addLayout(self._cs_grid)

        left_lo.addStretch()

        # Status at bottom of left panel
        self.dp_status = QLabel('Ready - Click Refresh to load dataset preview')
        self.dp_status.setStyleSheet(f'font-size:9px;color:{TEXT2};padding:4px 0;')
        left_lo.addWidget(self.dp_status)

        lo.addWidget(left_panel)

        # ── Right Panel: Image Grid ──
        right_panel = QWidget()
        right_panel.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:6px;')
        right_lo = QVBoxLayout(right_panel); right_lo.setContentsMargins(8,8,8,8); right_lo.setSpacing(0)

        self._cols = 3
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setStyleSheet(f'QScrollArea{{background:transparent;border:none;}}')
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.dp_grid = QWidget()
        self.dp_grid.setStyleSheet('background:transparent;')
        self.dp_gl = QGridLayout(self.dp_grid)
        self.dp_gl.setSpacing(8)
        self.dp_gl.setAlignment(Qt.AlignTop)
        self.dp_gl.setContentsMargins(0, 0, 0, 0)
        
        sa.setWidget(self.dp_grid)
        right_lo.addWidget(sa)

        lo.addWidget(right_panel, 1)

    def _stat_block(self, title, value, is_title=False):
        """创建统计块，同一行显示数值和标签"""
        w = QWidget(); w.setStyleSheet('background:transparent;border:none;')
        l = QHBoxLayout(w); l.setContentsMargins(0,0,0,0); l.setSpacing(6)
        
        if is_title:
            lbl = QLabel(title)
            lbl.setStyleSheet(f'font-size:13px;font-weight:600;color:{TEXT};')
            l.addWidget(lbl)
            l.addStretch()
            l.setAlignment(Qt.AlignVCenter)
            return w
        else:
            # 数值 - 左对齐
            v = QLabel(value)
            v.setStyleSheet(f'font-size:18px;font-weight:700;color:{TEXT};')
            v.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            l.addWidget(v)
            
            # 标签 - 左对齐，垂直居中
            lbl = QLabel(title)
            lbl.setStyleSheet(f'font-size:9px;color:{TEXT3};')
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            l.addWidget(lbl)
            
            l.addStretch()
            l.setAlignment(Qt.AlignVCenter)
            return w, v

    # ═══════════════ REFRESH ═══════════════

    def _dp_refresh(self):
        self.dp_status.setText('⏳ Loading...'); self.dp_rf.setEnabled(False)
        QApplication.processEvents()

        try:
            # ── Resolve paths ──
            dp = Path(DATA_YAML) if Path(DATA_YAML).is_absolute() else ROOT / DATA_YAML
            if not dp.exists():
                self.dp_status.setText(f'❌ data.yaml not found: {dp}')
                self.dp_rf.setEnabled(True); return
            with open(dp, encoding='utf-8') as f:
                meta = yaml.safe_load(f)

            base = Path(meta['path']) if Path(meta['path']).is_absolute() else ROOT / meta['path']
            split = self.dp_split.currentText()
            img_dir = base / 'images' / split
            lbl_dir = base / 'labels' / split
            self._names = meta.get('names', [f'cls{i}' for i in range(meta.get('nc',0))])

            if not img_dir.exists():
                self.dp_status.setText(f'❌ Image dir not found: {img_dir}')
                self.dp_rf.setEnabled(True); return

            # ── Stats summary ──
            total = sum(1 for f in img_dir.iterdir()
                        if f.suffix.lower() in ('.jpg','.jpeg','.png'))
            all_imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            cls_counter = Counter()
            for f in all_imgs:
                lf = lbl_dir / (f.stem+'.txt')
                if lf.exists():
                    for line in open(lf, encoding='utf-8'):
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_counter[int(parts[0])] += 1
            labeled = sum(cls_counter.values())
            labeled_imgs = sum(1 for f in all_imgs
                               if (lbl_dir / (f.stem+'.txt')).exists())
            self._update_stats(total, labeled_imgs, labeled, cls_counter)

            # ── Image list ──
            self._images = []
            for f in all_imgs:
                lbl_path = lbl_dir / (f.stem+'.txt')
                if lbl_path.exists():
                    self._images.append((str(f), str(lbl_path)))

            if not self._images:
                self.dp_status.setText(f'❌ No labeled images in {split}')
                self.dp_rf.setEnabled(True); return

            cnt = min(9, len(self._images))
            selected = random.sample(self._images, cnt)

            # ── Clear grid ──
            self._clear_grid()
            self._pixmaps = []
            error_count = 0

            # ── Render cards ──
            for idx, (img_path, lbl_path) in enumerate(selected):
                try:
                    card = self._make_card(img_path, lbl_path)
                    self.dp_gl.addWidget(card, idx // self._cols, idx % self._cols)
                except Exception as e:
                    error_count += 1
                    import traceback; traceback.print_exc()
                    print(f'Error: {img_path} → {e}')

            # ── Status ──
            shown = len(selected) - error_count
            if error_count > 0:
                self.dp_status.setText(f'⚠️ Showing {shown}/{len(selected)} ({error_count} errors)')
            else:
                self.dp_status.setText(f'✅ Showing {shown}/{len(self._images)} from {split}')

        except ImportError:
            self.dp_status.setText('❌ pip install Pillow PyYAML')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.dp_status.setText(f'❌ {e}')
        finally:
            self.dp_rf.setEnabled(True)

    # ═══════════════ CARD ═══════════════

    def _make_card(self, img_path, lbl_path):
        img = Image.open(img_path).convert('RGB')
        wi, hi = img.size
        draw = ImageDraw.Draw(img)

        # Parse annotations
        annos = []
        with open(lbl_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                x1 = (xc - bw/2) * wi
                y1 = (yc - bh/2) * hi
                x2 = (xc + bw/2) * wi
                y2 = (yc + bh/2) * hi
                color = self.COLORS[cid % len(self.COLORS)]
                cn = self._names[cid] if cid < len(self._names) else f'cls{cid}'
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                bb = draw.textbbox((x1, y1-16), cn)
                draw.rectangle(bb, fill=color)
                draw.text((x1, y1-16), cn, fill='white')
                annos.append((cid, cn, color))

        # Thumbnail → QPixmap (bytes roundtrip avoids PIL.ImageQt fragility)
        img.thumbnail((260, 260), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format='PNG')
        pix = QPixmap()
        pix.loadFromData(buf.getvalue())
        self._pixmaps.append(pix)

        # Card widget
        card = QWidget()
        card.setStyleSheet(f'''
            DatasetCard{{background:{CARD};border:1px solid {BORDER};border-radius:6px;}}
            DatasetCard:hover{{border:2px solid {PRI};}}
        ''')

        # Workaround: no direct hover on QWidget in Qt style
        cv = QVBoxLayout(card); cv.setContentsMargins(4,4,4,4); cv.setSpacing(2)

        # Image label
        lbl = QLabel(); lbl.setPixmap(pix)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(pix.width(), pix.height())
        cv.addWidget(lbl)

        # Class tags
        tag_row = QWidget(); tag_row.setStyleSheet('background:transparent;border:none;')
        tl = QHBoxLayout(tag_row); tl.setContentsMargins(2,0,2,0); tl.setSpacing(3)
        seen = set()
        for cid, cn, color in annos:
            if cn in seen: continue
            seen.add(cn)
            tag = QLabel(cn)
            tag.setStyleSheet(f'background:{color};color:#fff;border-radius:3px;padding:1px 5px;'
                              f'font-size:8px;font-weight:600;')
            tl.addWidget(tag)
        tl.addStretch()
        cv.addWidget(tag_row)

        # File info
        fn = QLabel(f'{Path(img_path).name}  ·  {len(annos)} anns')
        fn.setStyleSheet(f'font-size:8px;color:{TEXT3};border:none;'
                         f'qproperty-alignment:AlignCenter;')
        cv.addWidget(fn)

        return card

    # ═══════════════ HELPERS ═══════════════

    def _update_stats(self, total, labeled_imgs, labeled, cls_counter):
        """更新摘要统计行 + 类分布"""
        self._st_total_v.setText(str(total))
        self._st_lbl_v.setText(str(labeled_imgs))
        cls_count = len(self._names)
        self._st_cls_v.setText(str(cls_count))
        if self._names:
            cls_str = '  ' + ' '.join(str(n) for n in self._names)
            self._st_cls.setToolTip(f'Classes: {cls_str}')
        self._update_class_stats(cls_counter, total)

    def _update_class_stats(self, cls_counter, total_imgs):
        """更新各类别详细统计"""
        for i in reversed(range(self._cs_grid.count())):
            w = self._cs_grid.itemAt(i).widget()
            if w: w.deleteLater()

        total_instances = sum(cls_counter.values()) if cls_counter else 0

        if not cls_counter or not self._names:
            placeholder = QLabel('Click Refresh to load class distribution')
            placeholder.setStyleSheet(f'font-size:10px;color:{TEXT3};padding:10px 0;')
            self._cs_grid.addWidget(placeholder, 0, 0, 1, 4)
            return

        # Headers
        for c, txt in enumerate(['Class', 'Count', '%', 'Distribution']):
            lbl = QLabel(txt)
            lbl.setStyleSheet(f'font-size:9px;color:{TEXT3};font-weight:600;padding:2px 0;')
            self._cs_grid.addWidget(lbl, 0, c)

        # Image count info row
        info = QLabel(f'{total_imgs} images  ·  {total_instances} instances')
        info.setStyleSheet(f'font-size:9px;color:{TEXT2};padding:0 0 2px 0;')
        self._cs_grid.addWidget(info, 1, 0, 1, 4)

        for r, (cls_id, count) in enumerate(sorted(cls_counter.items()), start=2):
            cls_name = self._names[cls_id] if cls_id < len(self._names) else f'cls{cls_id}'
            pct = (count / total_instances * 100) if total_instances > 0 else 0
            color = self.COLORS[cls_id % len(self.COLORS)]

            # Name with colored dot
            name_w = QWidget(); name_w.setStyleSheet('background:transparent;')
            nl = QHBoxLayout(name_w); nl.setContentsMargins(0,0,0,0); nl.setSpacing(4)
            dot = QLabel('●'); dot.setStyleSheet(f'color:{color};font-size:10px;')
            nl.addWidget(dot)
            nl.addWidget(QLabel(cls_name, styleSheet=f'font-size:10px;color:{TEXT};font-weight:500;'))
            nl.addStretch()
            self._cs_grid.addWidget(name_w, r, 0)

            # Count
            cl = QLabel(str(count))
            cl.setStyleSheet(f'font-size:10px;color:{TEXT};')
            self._cs_grid.addWidget(cl, r, 1)

            # Percentage
            pl = QLabel(f'{pct:.1f}%')
            pl.setStyleSheet(f'font-size:10px;color:{TEXT2};')
            self._cs_grid.addWidget(pl, r, 2)

            # Bar
            bar = QProgressBar()
            bar.setFixedHeight(8); bar.setMinimum(0); bar.setMaximum(100)
            bar.setValue(int(round(pct))); bar.setTextVisible(False)
            bar.setStyleSheet(f'''
                QProgressBar {{
                    border:none; border-radius:4px; background:{BORDER};
                    height:8px;
                }}
                QProgressBar::chunk {{
                    background:{color}; border-radius:4px;
                }}
            ''')
            self._cs_grid.addWidget(bar, r, 3)

    def _clear_grid(self):
        for i in reversed(range(self.dp_gl.count())):
            w = self.dp_gl.itemAt(i).widget()
            if w: w.deleteLater()

    # ═══════════════ IMPORT ═══════════════

    def _fix_yaml_config(self):
        """修复 data.yaml 配置，将 path 指向当前 datasets 目录"""
        yaml_path = ROOT / DATA_YAML if not Path(DATA_YAML).is_absolute() else Path(DATA_YAML)
        
        if not yaml_path.exists():
            QMessageBox.warning(self, "Fix Error", f"data.yaml not found:\n{yaml_path}")
            return
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)
            
            old_path = yaml_content.get('path', '')
            
            # 计算相对于 YAML 文件的正确路径
            # YAML 在 datasets/ 目录下，所以 path 应该是 '.' (指向 datasets 自身)
            # 或者使用绝对路径
            yaml_dir = yaml_path.parent
            images_dir = yaml_dir / 'images'
            labels_dir = yaml_dir / 'labels'
            
            # 检查目录是否存在
            if not images_dir.exists() or not labels_dir.exists():
                QMessageBox.warning(
                    self,
                    "Directory Not Found",
                    f"Dataset directories not found in:\n{yaml_dir}\n\n"
                    f"Missing: {'images' if not images_dir.exists() else ''} {'labels' if not labels_dir.exists() else ''}"
                )
                return
            
            # 检查是否需要修复
            if old_path == str(yaml_dir.resolve()):
                QMessageBox.information(
                    self,
                    "Already Fixed",
                    f"data.yaml is already configured correctly!\n\nCurrent path:\n{yaml_dir}"
                )
                return
            
            # 更新 path 为绝对路径（指向 datasets 目录）
            yaml_content['path'] = str(yaml_dir.resolve())
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)
            
            QMessageBox.information(
                self,
                "YAML Fixed",
                f"data.yaml has been updated!\n\n"
                f"Old path: {old_path}\n"
                f"New path: .\n\n"
                f"Images: {images_dir}\n"
                f"Labels: {labels_dir}\n\n"
                f"Now it will read from the datasets directory."
            )
            
            # 自动刷新预览
            self._dp_refresh()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Fix Error", f"Failed to fix data.yaml:\n{str(e)}")

    def _import_dataset(self):
        """从 original/label 导入数据集到 datasets 目录"""
        source_dir = ROOT / "original" / "label"
        target_dir = ROOT / "datasets"
        
        # 检查源目录是否存在
        if not source_dir.exists():
            QMessageBox.warning(self, "Import Error", f"Source directory not found:\n{source_dir}")
            return
        
        # 统计源文件数量
        src_images = list((source_dir / "images" / "train").glob("*")) + \
                     list((source_dir / "images" / "val").glob("*")) if (source_dir / "images").exists() else []
        src_labels = list((source_dir / "labels" / "train").glob("*")) + \
                     list((source_dir / "labels" / "val").glob("*")) if (source_dir / "labels").exists() else []
        
        total_images = len(src_images)
        total_labels = len(src_labels)
        
        # 确认对话框
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Import Dataset")
        msg.setText(f"Import dataset from original/label to datasets?")
        msg.setInformativeText(
            f"Source: {source_dir}\n"
            f"Target: {target_dir}\n\n"
            f"Images: {total_images} files\n"
            f"Labels: {total_labels} files\n\n"
            f"This will overwrite existing files in the target directory."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        reply = msg.exec_()
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            self.dp_import.setEnabled(False)
            self.dp_import.setText("⏳ Importing...")
            QApplication.processEvents()
            
            imported_images = 0
            imported_labels = 0
            
            # 创建目标目录结构
            for sub_dir in ["images/train", "images/val", "labels/train", "labels/val"]:
                (target_dir / sub_dir).mkdir(parents=True, exist_ok=True)
            
            # 复制图片
            for split in ["train", "val"]:
                src_img_dir = source_dir / "images" / split
                tgt_img_dir = target_dir / "images" / split
                
                if src_img_dir.exists():
                    for img_file in src_img_dir.iterdir():
                        if img_file.is_file():
                            shutil.copy2(str(img_file), str(tgt_img_dir / img_file.name))
                            imported_images += 1
            
            # 复制标签
            for split in ["train", "val"]:
                src_lbl_dir = source_dir / "labels" / split
                tgt_lbl_dir = target_dir / "labels" / split
                
                if src_lbl_dir.exists():
                    for lbl_file in src_lbl_dir.iterdir():
                        if lbl_file.is_file():
                            shutil.copy2(str(lbl_file), str(tgt_lbl_dir / lbl_file.name))
                            imported_labels += 1
            
            # 复制 data.yaml 并更新 path
            src_yaml = source_dir / "data.yaml"
            tgt_yaml = target_dir / "data.yaml"
            if src_yaml.exists():
                # 读取原始 yaml
                with open(src_yaml, 'r', encoding='utf-8') as f:
                    yaml_content = yaml.safe_load(f)
                
                # 更新 path 为绝对路径（指向 datasets 目录）
                yaml_content['path'] = str(target_dir.resolve())
                
                # 写回更新后的 yaml
                with open(tgt_yaml, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)
            
            # 更新状态
            self.dp_status.setText(f"✅ Imported: {imported_images} images, {imported_labels} labels")
            
            QMessageBox.information(
                self,
                "Import Complete",
                f"Dataset imported successfully!\n\n"
                f"Images: {imported_images}\n"
                f"Labels: {imported_labels}"
            )
            
            # 自动刷新预览
            self._dp_refresh()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.dp_status.setText(f"❌ Import failed: {e}")
            QMessageBox.critical(self, "Import Error", f"Failed to import dataset:\n{str(e)}")
        finally:
            self.dp_import.setEnabled(True)
            self.dp_import.setText(" Import")
