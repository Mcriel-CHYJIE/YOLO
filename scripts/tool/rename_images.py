"""
将指定文件夹下的图片按顺序重命名为从 0000 开始的连续编号。

用法：
    python rename_images.py D:\Dataset\images                # 重命名为 0000.jpg, 0001.jpg...
    python rename_images.py D:\Dataset\images --prefix zero  # 重命名为 zero_0000.jpg, zero_0001.jpg...
    python rename_images.py D:\Dataset\images --digits 5     # 使用 5 位数字: 00000.jpg...
    python rename_images.py D:\Dataset\images --dry-run      # 仅预览
    python rename_images.py D:\Dataset\images --ext png      # 输出为 PNG 格式
"""

import os
import sys
from pathlib import Path
import argparse

# ─ 支持的图片扩展名 ──
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def collect_images(root: Path) -> list[Path]:
    """遍历 root，返回所有图片文件的绝对路径（按文件名排序）"""
    images = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return sorted(images)


def build_new_name(index: int, ext: str, prefix: str = '', digits: int = 4) -> str:
    """
    构建新文件名
    
    Args:
        index: 当前索引
        ext: 扩展名（包含点，如 '.jpg'）
        prefix: 可选前缀
        digits: 数字位数
    
    Returns:
        新文件名
    """
    num_str = str(index).zfill(digits)
    
    if prefix:
        return f'{prefix}_{num_str}{ext}'
    else:
        return f'{num_str}{ext}'


def main():
    parser = argparse.ArgumentParser(description='将图片按顺序重命名为连续编号')
    parser.add_argument('folder', help='目标文件夹路径')
    parser.add_argument('--prefix', default='',
                        help='文件名前缀（可选，如 "zero"）')
    parser.add_argument('--digits', type=int, default=6,
                        help='编号位数（默认 4 位，即 0000-9999）')
    parser.add_argument('--ext', default=None,
                        help='输出格式扩展名（可选，如 "png"；不指定则保持原格式）')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅预览，不实际重命名')
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f' 目录不存在: {folder}')
        sys.exit(1)

    # ── 收集图片 ──
    print(f' 扫描目录: {folder}')
    images = collect_images(folder)

    if not images:
        print('  未找到任何图片文件')
        sys.exit(0)

    print(f'🔍 找到 {len(images)} 张图片')

    # ── 确定输出扩展名 ──
    output_ext = None
    if args.ext:
        ext_lower = args.ext.lower()
        if not ext_lower.startswith('.'):
            ext_lower = f'.{ext_lower}'
        if ext_lower not in IMAGE_EXTS:
            print(f' 不支持的格式: {ext_lower}')
            sys.exit(1)
        output_ext = ext_lower

    # ── 检查冲突 ──
    dest_map = {}
    for idx, img_path in enumerate(images):
        if output_ext:
            ext = output_ext
        else:
            ext = img_path.suffix
        
        new_name = build_new_name(idx, ext, args.prefix, args.digits)
        dest_map[idx] = (img_path, new_name)

        if not args.dry_run and img_path.name == new_name:
            continue  # 文件名未变，跳过

    # ── 预览 ──
    print(f'\n📋 预览（前 20 个）:')
    for idx, (src_path, new_name) in list(dest_map.items())[:20]:
        if src_path.name == new_name:
            print(f'  {src_path.name}  →  (no change)')
        else:
            print(f'  {src_path.name}')
            print(f'    → {new_name}')
    if len(images) > 20:
        print(f'  ... 还有 {len(images) - 20} 个')

    if args.dry_run:
        print('\n Dry-run 完成，未执行任何操作')
        return

    # ── 执行重命名 ─
    renamed = 0
    skipped = 0
    errors = 0

    for idx, (src_path, new_name) in dest_map.items():
        if src_path.name == new_name:
            skipped += 1
            continue

        dest_path = src_path.parent / new_name

        if dest_path.exists():
            print(f'  跳过，目标已存在: {new_name}')
            skipped += 1
            continue

        try:
            src_path.rename(dest_path)
            print(f'  {src_path.name}  →  {new_name}')
            renamed += 1
        except Exception as e:
            print(f' 失败: {src_path.name} → {e}')
            errors += 1

    print(f'\n 完成: 重命名 {renamed} 个, 跳过 {skipped} 个')
    if errors:
        print(f'  失败: {errors} 个')


if __name__ == '__main__':
    main()
