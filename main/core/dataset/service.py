"""数据集业务逻辑 — 加载预览数据 + 导入数据集"""
from pathlib import Path
from collections import Counter
import random, shutil
from main.core.base import ROOT, CLASSES
from main.config import load_paths


def _ds_path():
    return load_paths().get('dataset_dir', '')


def _lbl_path():
    return load_paths().get('label_dir', '')


def load_dataset_preview(split: str) -> dict:
    """加载数据集预览数据，返回结构体"""
    _ds = _ds_path()
    img_dir = Path(_ds) / 'images' / split
    lbl_dir = Path(_ds) / 'labels' / split
    if not img_dir.exists() or not lbl_dir.exists():
        return {'error': f'Split "{split}" not found in datasets/', 'images': [], 'labeled': 0}

    imgs = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')) + \
           sorted(img_dir.glob('*.jpeg')) + sorted(img_dir.glob('*.webp'))
    if not imgs:
        return {'error': f'No images in {split}', 'images': []}

    images = []
    labeled = 0
    cls_counter = Counter()
    for img_path in imgs:
        lbl_path = lbl_dir / f'{img_path.stem}.txt'
        has_lbl = lbl_path.exists() and lbl_path.stat().st_size > 0
        if has_lbl:
            labeled += 1
        images.append({'img_path': str(img_path), 'lbl_path': str(lbl_path), 'has_lbl': has_lbl})
        if has_lbl:
            for line in lbl_path.read_text().strip().split('\n'):
                if line.strip():
                    cls_counter[int(line.strip().split()[0])] += 1

    # Select 9 random images for preview
    preview_indices = random.sample(range(len(images)), min(9, len(images)))
    previews = [images[i] for i in preview_indices]

    return {
        'total': len(imgs),
        'labeled': labeled,
        'unlabeled': len(imgs) - labeled,
        'cls_counts': dict(cls_counter),
        'total_instances': sum(cls_counter.values()),
        'num_classes': len(cls_counter),
        'preview': previews,
        'preview_count': len(previews),
        'split': split,
    }


def import_dataset() -> tuple:
    """从 original/label 导入数据集到 datasets/，返回 (copied, total, error_msg)"""
    src = Path(_lbl_path()) / 'label'
    if not src.exists():
        return 0, 0, f'Label directory not found: {src}'

    img_src = src / 'images'
    lbl_src = src / 'labels'
    if not img_src.exists() or not lbl_src.exists():
        return 0, 0, f'Invalid dataset structure. Expected: {src}/images/train, images/val\n{src}/labels/train, labels/val'

    total_images = 0
    for split in ['train', 'val']:
        si = img_src / split
        if si.exists():
            total_images += len([f for f in si.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg', '.webp')])

    dst = Path(_ds_path())
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

    return copied, total_images, '' if copied > 0 else 'No new images to import (all files already exist)'
