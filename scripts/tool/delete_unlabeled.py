"""删除文件夹内没有标注的图片，标注从 JSON 文件读取。

支持两种模式：
  1. COCO JSON (--coco) — 一个 JSON 含 images + annotations，删无标注的
  2. 逐文件 JSON (--json-dir) — 每张图对应一个同名的 .json，删不存在或空标注的

用法:
  # COCO 模式
  python scripts/tool/delete_unlabeled.py D:/dataset/images --coco D:/dataset/annotations.json

  # 逐文件 JSON 模式
  python scripts/tool/delete_unlabeled.py D:/dataset/images --json-dir D:/dataset/labels

  # 仅预览不删除
  python scripts/tool/delete_unlabeled.py D:/dataset/images --coco ann.json --dry-run
"""
import argparse, json, sys
from pathlib import Path

# ── 支持的图片后缀 ──
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def collect_images(img_dir: Path):
    """扫描目录下所有图片，返回 {stem: path} 映射"""
    images = {}
    for f in img_dir.rglob('*'):
        if f.suffix.lower() in IMG_EXTS:
            images[f.stem] = f
    return images


def _no_ann(fname: str) -> bool:
    """检查单文件 JSON 是否无有效标注"""
    try:
        data = json.loads(Path(fname).read_text(encoding='utf-8'))
    except Exception:
        return True  # 无法解析视为无效
    # LabelMe 格式: shapes 数组
    if isinstance(data, dict):
        shapes = data.get('shapes', data.get('objects', data.get('annotations', [])))
        if isinstance(shapes, list) and len(shapes) > 0:
            return False
        # 也可能是 COCO 子集含 annotations
        anns = data.get('annotations', [])
        if isinstance(anns, list) and len(anns) > 0:
            return False
        return True
    # 顶层数组
    if isinstance(data, list) and len(data) > 0:
        return False
    return True


def mode_per_json(images: dict, json_dir: Path, dry_run: bool):
    """逐文件 JSON 模式：每张图对应一个同名 .json"""
    if not json_dir.is_dir():
        print(f'❌ JSON 目录不存在: {json_dir}'); return

    to_del = []
    for stem, img_path in images.items():
        json_path = json_dir / f'{stem}.json'
        if not json_path.exists() or _no_ann(json_path):
            to_del.append(img_path)

    _delete_images(to_del, dry_run, 'json 缺失或空标注')


def mode_coco(images: dict, coco_path: Path, dry_run: bool):
    """简单列表 JSON 模式：annotations 数组存有标注的文件名"""
    if not coco_path.exists():
        print(f'❌ COCO JSON 不存在: {coco_path}'); return

    try:
        data = json.loads(coco_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'❌ 解析 JSON 失败: {e}'); return

    # annotations 是 dict(key=路径)/list(文件名)/标准COCO?
    ann_raw = data.get('annotations', [])
    if not ann_raw:
        print('⚠ annotations 为空'); return

    # dict 模式: {路径: [标注...]}
    if isinstance(ann_raw, dict):
        # key 可能是完整路径或纯文件名
        annotated_stems = set()
        for k in ann_raw:
            stem = Path(k).stem
            if stem:
                annotated_stems.add(stem)
        to_del = [p for stem, p in images.items() if stem not in annotated_stems]
        _delete_images(to_del, dry_run, f'JSON 中无标注（共 {len(annotated_stems)} 张有标注）')
        return

    # list 模式: 字符串列表 → 简单文件名匹配
    if isinstance(ann_raw[0], str):
        annotated_stems = set()
        for fname in ann_raw:
            stem = Path(fname).stem
            if stem:
                annotated_stems.add(stem)
        to_del = [p for stem, p in images.items() if stem not in annotated_stems]
        _delete_images(to_del, dry_run, f'JSON 中无标注（共 {len(annotated_stems)} 张有标注）')
        return

    # 标准 COCO 格式：dict 列表 → image_id 模式
    ann_img_ids = set()
    for ann in ann_raw:
        iid = ann.get('image_id')
        if iid is not None:
            ann_img_ids.add(iid)

    id_to_name = {}
    for img in data.get('images', []):
        iid = img.get('id')
        fname = img.get('file_name', '')
        if iid is not None and fname:
            id_to_name[iid] = Path(fname).stem

    annotated_stems = {s for iid in ann_img_ids if (s := id_to_name.get(iid))}
    to_del = [p for stem, p in images.items() if stem not in annotated_stems]
    _delete_images(to_del, dry_run, 'COCO 中无标注')


def _delete_images(paths: list, dry_run: bool, reason: str):
    if not paths:
        print('✓ 所有图片都有标注，无需删除')
        return

    print(f'{"[DRY-RUN] " if dry_run else ""}找到 {len(paths)} 张无标注图片（原因: {reason}）:')
    for p in paths:
        print(f'  {p}')
    if not dry_run:
        for p in paths:
            try:
                p.unlink()
                print(f'  ✗ 已删除: {p.name}')
            except Exception as e:
                print(f'  ⚠ 删除失败 {p.name}: {e}')
        print(f'✓ 共删除 {len(paths)} 张')
    else:
        print(f'  (dry-run，未实际删除)')


def main():
    ap = argparse.ArgumentParser(description='删除无标注图片')
    ap.add_argument('img_dir', help='图片文件夹路径')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--coco', metavar='JSON', help='COCO JSON 标注文件')
    grp.add_argument('--json-dir', metavar='DIR', help='逐文件 JSON 标注目录')
    ap.add_argument('--dry-run', action='store_true', help='仅预览，不删除')
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        print(f'❌ 图片目录不存在: {img_dir}'); sys.exit(1)

    images = collect_images(img_dir)
    if not images:
        print(f'❌ 目录中未找到图片: {img_dir}'); sys.exit(1)
    print(f'📷 扫描到 {len(images)} 张图片')

    if args.coco:
        mode_coco(images, Path(args.coco), args.dry_run)
    else:
        mode_per_json(images, Path(args.json_dir), args.dry_run)


if __name__ == '__main__':
    main()
