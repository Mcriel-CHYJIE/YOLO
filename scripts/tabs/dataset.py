"""数据集预览标签页 — 优化排布"""
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
        self.dp_split.addItems(['train', 'val'])
        self._cols = 3
        self.dp_status.setText('Ready - Click Refresh to load dataset preview')

    def _connect_signals(self):
        self.dp_split.currentIndexChanged.connect(lambda: self._dp_refresh())
        self.dp_rf.setObjectName('pri')
        self.dp_rf.clicked.connect(self._dp_refresh)
        self.dp_import.clicked.connect(self._import_dataset)
        self.dp_fix_yaml.clicked.connect(self._fix_yaml_config)

    def _dp_refresh(self):
        self.dp_status.setText('Loading...'); QApplication.processEvents()
        split = self.dp_split.currentText()
        img_dir = ROOT / 'datasets' / 'images' / split
        lbl_dir = ROOT / 'datasets' / 'labels' / split
        if not img_dir.exists() or not lbl_dir.exists():
            self.dp_status.setText(f'❌ Split "{split}" not found in datasets/'); return
        imgs = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')) + \
               sorted(img_dir.glob('*.jpeg')) + sorted(img_dir.glob('*.webp'))
        if not imgs: self.dp_status.setText(f'⚠ No images in {split}'); return
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
        self._st_cls_v.setText(f'{len(cls_counter)}')
        self._update_class_stats(cls_counter, labeled)
        self._clear_grid()
        for idx, (img_path, lbl_path) in enumerate(self._images):
            r, c = divmod(idx, self._cols)
            self.dp_gl.addWidget(self._make_card(img_path, lbl_path), r, c)
        self.dp_status.setText(f'✅ {len(imgs)} images ({len(self._images)} total)')

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
        cv = QVBoxLayout(card); cv.setContentsMargins(4,4,4,4); cv.setSpacing(2)
        has_lbl = lbl_path.exists() and lbl_path.stat().st_size > 0
        thumb = QLabel(); thumb.setAlignment(Qt.AlignCenter)
        thumb.setFixedSize(200, 150); thumb.setStyleSheet('background:transparent;')
        try:
            pil_img = Image.open(img_path); pil_img.thumbnail((198, 148), Image.LANCZOS)
            from PIL.ImageQt import toqpixmap
            thumb.setPixmap(toqpixmap(pil_img))
        except: thumb.setText(img_path.name); thumb.setStyleSheet(f'color:{TEXT3};font-size:9px;')
        cv.addWidget(thumb)
        tr = QWidget(); tr.setStyleSheet('background:transparent;border:none;')
        tl = QHBoxLayout(tr); tl.setContentsMargins(2,0,2,0); tl.setSpacing(3)
        fn = QLabel(img_path.name); fn.setStyleSheet(f'font-size:8px;color:{TEXT2};')
        fn.setWordWrap(True); tl.addWidget(fn, 1)
        st = QLabel('✅' if has_lbl else '⏳'); st.setStyleSheet('font-size:10px;'); st.setFixedWidth(18)
        tl.addWidget(st); cv.addWidget(tr)
        if has_lbl:
            ci = QLabel(); ci.setStyleSheet(f'font-size:7px;color:{TEXT3};')
            try:
                cids = [int(l.strip().split()[0]) for l in lbl_path.read_text().strip().split('\n') if l.strip()]
                parts = []
                for cid, cnt in Counter(cids).items():
                    nm = CLASSES[cid] if cid < len(CLASSES) else f'cls_{cid}'
                    parts.append(f'<span style="color:{self.COLORS[cid % len(self.COLORS)]}">{nm}:{cnt}</span>')
                ci.setText(' '.join(parts))
            except: pass
            cv.addWidget(ci)
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
            self.dp_status.setText(f'✅ data.yaml generated: {nc} classes')
            QMessageBox.information(self, 'YAML Fixed', f'{yaml_path}\n{nc} classes, {len(splits)} splits')
        except Exception as e: QMessageBox.critical(self, 'Error', f'Failed: {e}')

    def _import_dataset(self):
        src = Path(QFileDialog.getExistingDirectory(self, 'Select Dataset Folder (with images+labels)'))
        if not src or not src.exists(): return
        img_dirs = [d for d in src.iterdir() if d.is_dir() and d.name in ('train', 'val')]
        if not img_dirs: QMessageBox.warning(self, 'Error', 'Dataset must have train/val subdirectories'); return
        dst = ROOT / 'datasets'; copied = 0
        for split in ['train', 'val']:
            si = src / split / 'images'; sl = src / split / 'labels'
            di = dst / 'images' / split; dl = dst / 'labels' / split
            if not si.exists(): continue
            di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
            for f in si.iterdir():
                if f.suffix.lower() in ('.jpg', '.png', '.jpeg', '.webp'):
                    if not (di / f.name).exists():
                        shutil.copy2(f, di / f.name)
                        lbl = sl / f'{f.stem}.txt'
                        if lbl.exists(): shutil.copy2(lbl, dl / f'{lbl.name}')
                        copied += 1
        self.dp_status.setText(f'✅ Imported {copied} images'); self._dp_refresh()
