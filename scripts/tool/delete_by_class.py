"""删除标注全为某类的图片。

对于标注全部为指定类别（如 "person"）的图片，删除图片。
常用于清理负样本干扰，例如训练摔倒检测时删除仅含 "person" 类（无人摔倒）的图片。

标注支持三种格式（参照 delete_unlabeled.py）：
  1. COCO JSON (--coco) — 标准 COCO 格式或 dict-of-lists 格式
  2. 逐文件 JSON (--json-dir) — 每张图对应一个同名的 .json

用法:
  # COCO JSON 模式（标准 COCO 或 dict 格式）
  python scripts/tool/delete_by_class.py D:/Projects/11_fall/original/after/other7 --coco D:/Projects/11_fall/original/after/other7/_annotations.json --class 0

  # 逐文件 JSON 模式
  python scripts/tool/delete_by_class.py D:/Projects/11_fall/original/after/other7 --json-dir D:/Projects/11_fall/original/label/train/labels --class 0

  # 仅预览不删除
  python scripts/tool/delete_by_class.py D:/Projects/11_fall/original/after/other7 --coco _annotations.json --class 0 --dry-run
"""
import argparse, json, sys
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def collect_images(img_dir):
    """扫描目录下所有图片，返回 {stem: path} 映射。"""
    images = {}
    for f in img_dir.rglob('*'):
        if f.suffix.lower() in IMG_EXTS:
            images[f.stem] = f
    return images


# ── COCO JSON 模式（支持标准COCO + dict-of-lists 两种） ──

def _all_class_in_dict_anns(ann_list, target_cid):
    """dict-of-lists 格式: {"path.jpg": [{"class_id": N, ...}, ...]}"""
    if not ann_list:
        return False
    cids = []
    for ann in ann_list:
        cid = ann.get('class_id', ann.get('category_id'))
        if cid is not None:
            cids.append(int(cid))
    if not cids:
        return False
    return all(c == target_cid for c in cids)


def _all_class_in_coco(annotations, image_id, target_cid):
    """标准 COCO list-of-dicts 格式，按 image_id 分组。"""
    cids = [int(a['category_id']) for a in annotations
            if a.get('image_id') == image_id and 'category_id' in a]
    if not cids:
        return False
    return all(c == target_cid for c in cids)


def mode_coco(images, coco_path, target_cid, dry_run):
    if not coco_path.exists():
        print(f'  COCO JSON not found: {coco_path}'); return
    try:
        data = json.loads(coco_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'  Parse failed: {e}'); return

    raw_anns = data.get('annotations', data) if isinstance(data, dict) else data

    # case 1: dict-of-lists — {"path.jpg": [{class_id}, ...], ...}
    if isinstance(raw_anns, dict):
        to_del = []
        for fpath_str, ann_list in raw_anns.items():
            stem = Path(fpath_str).stem
            img_path = images.get(stem)
            if not img_path:
                continue
            if _all_class_in_dict_anns(ann_list, target_cid):
                to_del.append((img_path, len(ann_list)))
        _do_delete(to_del, dry_run, f'dict-of-lists JSON (class={target_cid})')
        return

    # case 2: standard COCO list-of-dicts
    if isinstance(raw_anns, list):
        # 构建 stem → image_id 映射
        stem_to_id = {}
        for img in data.get('images', []):
            fname = img.get('file_name', '')
            iid = img.get('id')
            if iid is not None:
                stem_to_id[Path(fname).stem] = iid

        to_del = []
        for stem, img_path in images.items():
            iid = stem_to_id.get(stem)
            if iid is None:
                continue
            if _all_class_in_coco(raw_anns, iid, target_cid):
                img_anns = [a for a in raw_anns if a.get('image_id') == iid]
                to_del.append((img_path, len(img_anns)))
        _do_delete(to_del, dry_run, f'COCO JSON (class={target_cid})')
        return

    print(f'  Unrecognized annotations format: {type(raw_anns).__name__}')


# ── 逐文件 JSON 模式 ──

def _get_per_json_cids(json_path):
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
    except Exception:
        return []
    cids = []
    if isinstance(data, dict):
        for shapes_key in ('shapes', 'objects', 'annotations'):
            items = data.get(shapes_key, [])
            if isinstance(items, list) and items:
                for item in items:
                    cid = item.get('label', item.get('category_id', item.get('class_id')))
                    if isinstance(cid, str):
                        try: cids.append(int(cid))
                        except ValueError: pass
                    elif isinstance(cid, (int, float)):
                        cids.append(int(cid))
                if cids:
                    break
        return cids
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                cid = item.get('label', item.get('category_id', item.get('class_id')))
                if isinstance(cid, str):
                    try: cids.append(int(cid))
                    except ValueError: pass
                elif isinstance(cid, (int, float)):
                    cids.append(int(cid))
            elif isinstance(item, (int, float)):
                cids.append(int(item))
        return cids
    return cids


def mode_per_json(images, json_dir, target_cid, dry_run):
    if not json_dir.is_dir():
        print(f'  JSON dir not found: {json_dir}'); return
    to_del = []
    for stem, img_path in sorted(images.items()):
        json_path = json_dir / f'{stem}.json'
        if not json_path.exists():
            continue
        cids = _get_per_json_cids(json_path)
        if not cids:
            continue
        if all(c == target_cid for c in cids):
            to_del.append((img_path, len(cids)))
    _do_delete(to_del, dry_run, f'per-file JSON (class={target_cid})')


# ── 执行删除 ──

def _do_delete(to_del, dry_run, source):
    if not to_del:
        print('  No images to delete (none have all labels matching the target class)')
        return
    print(f'  Found {len(to_del)} images where all labels are class:')
    for img_path, count in to_del:
        if dry_run:
            print(f'    [DRY-RUN] {img_path.name} ({count} boxes)')
        else:
            try:
                img_path.unlink()
                print(f'    Deleted: {img_path.name} ({count} boxes)')
            except Exception as e:
                print(f'    Failed: {img_path.name} - {e}')


def main():
    ap = argparse.ArgumentParser(
        description='删除标注全为某类的图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('img_dir', help='图片文件夹路径')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--coco', metavar='JSON', help='COCO JSON 标注文件（标准COCO或dict-of-lists格式）')
    src.add_argument('--json-dir', metavar='DIR', help='逐文件 JSON 标注目录')
    ap.add_argument('--class', '-c', dest='class_arg', required=True,
                    help='目标类别 ID（数字）')
    ap.add_argument('--dry-run', action='store_true', help='仅预览不删除')
    args = ap.parse_args()

    try:
        target_cid = int(args.class_arg)
    except ValueError:
        print('Error: --class must be a numeric class ID'); sys.exit(1)

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        print(f'Error: image dir not found: {img_dir}'); sys.exit(1)

    images = collect_images(img_dir)
    if not images:
        print(f'No images found in: {img_dir}'); sys.exit(1)
    print(f'Scanned {len(images)} images in {img_dir}')

    if args.coco:
        mode_coco(images, Path(args.coco), target_cid, args.dry_run)
    else:
        mode_per_json(images, Path(args.json_dir), target_cid, args.dry_run)


if __name__ == '__main__':
    main()
