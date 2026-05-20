"""删除文件夹内内容重复的图片（基于感知哈希对比）。

用法:
  # 默认阈值 5，仅删除精确/近似重复
  python scripts/tool/delete_duplicates.py D:/dataset/images

  # 紧匹配（只删像素级完全相同）
  python scripts/tool/delete_duplicates.py D:/dataset/images --threshold 0

  # 宽松匹配
  python scripts/tool/delete_duplicates.py D:/dataset/images --threshold 10

  # 仅预览
  python scripts/tool/delete_duplicates.py D:/dataset/images --threshold 5 --dry-run

依赖: pip install Pillow imagehash
"""
import argparse, sys
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def main():
    ap = argparse.ArgumentParser(description='删除内容重复的图片')
    ap.add_argument('img_dir', help='图片文件夹路径')
    ap.add_argument('--threshold', type=int, default=5,
                    help='汉明距离阈值 (0=精确重复, 默认5, 越大越宽松)')
    ap.add_argument('--dry-run', action='store_true', help='仅预览，不删除')
    ap.add_argument('--algo', choices=['phash', 'dhash', 'ahash', 'whash'],
                    default='phash', help='哈希算法 (默认 phash，兼顾速度与准确)')
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        print(f'❌ 目录不存在: {img_dir}'); sys.exit(1)

    # ── 扫描图片 ──
    images = sorted(p for p in img_dir.rglob('*') if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f'❌ 未找到图片: {img_dir}'); sys.exit(1)
    print(f'📷 扫描到 {len(images)} 张图片')

    # ── 计算哈希 ──
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        print('❌ 缺少依赖，请安装: pip install Pillow imagehash')
        sys.exit(1)

    hash_func = getattr(imagehash, args.algo)
    hashes = []  # [(hash, path), ...]
    errors = 0
    for i, p in enumerate(images):
        try:
            img = Image.open(p).convert('RGB')
            h = hash_func(img)
            hashes.append((h, p))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f'  ⚠ 跳过 {p.name}: {e}')
        if (i + 1) % 1000 == 0:
            print(f'  ⏳ 处理 {i+1}/{len(images)}...')

    print(f'  ✓ 计算完成: {len(hashes)} 张, 跳过 {errors} 张')

    if len(hashes) < 2:
        print('⚠ 图片不足 2 张，无需去重'); return

    # ── 查重 ──
    T = args.threshold
    keep = set()        # 保留的图片 (hash 值)
    groups = []         # [(保留path, [重复path...]), ...]
    seen = {}           # hash_val → kept_path
    bucket = defaultdict(list)  # hash_val → [paths...]

    if T == 0:
        # ── 精确匹配：直接按 hash 分组 ──
        for h, p in hashes:
            bucket[h].append(p)
        for h, paths in bucket.items():
            if len(paths) > 1:
                groups.append((paths[0], paths[1:]))
    else:
        # ── 模糊匹配：线性扫描 + 阈值 ──
        kept_hashes = []  # [(hash, path)]
        for h, p in hashes:
            found = False
            for kh, kp in kept_hashes:
                if h - kh <= T:
                    # 找到重复组
                    found_grp = False
                    for g_idx, (kp2, dups) in enumerate(groups):
                        if kp2 == kp:
                            dups.append(p)
                            found_grp = True
                            break
                    if not found_grp:
                        groups.append((kp, [p]))
                    found = True
                    break
            if not found:
                kept_hashes.append((h, p))

    # ── 统计 ──
    total_dup = sum(len(dups) for _, dups in groups)
    if not groups:
        print('✓ 未发现重复图片')
        return

    print(f'\n🔁 发现 {len(groups)} 组重复，共 {total_dup} 张可删除:\n')
    for kept, dups in groups:
        print(f'  保留: {kept.name}')
        for d in dups:
            print(f'    删除: {d.name}')

    # ── 执行删除 ──
    if args.dry_run:
        print(f'\n  (dry-run, 未实际删除)')
        return

    deleted = 0
    failed = 0
    for kept, dups in groups:
        for d in dups:
            try:
                d.unlink()
                deleted += 1
            except Exception as e:
                print(f'  ⚠ 删除失败 {d.name}: {e}')
                failed += 1

    print(f'\n✓ 共删除 {deleted} 张重复图片' + (f', {failed} 张失败' if failed else ''))


if __name__ == '__main__':
    main()
