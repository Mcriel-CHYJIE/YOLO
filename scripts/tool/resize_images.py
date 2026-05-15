#!/usr/bin/env python3
"""
图片批量 Resize 到 640×640 工具

用法:
  # 处理单个图片
  python resize_images.py input.jpg -o output.jpg

  # 处理整个目录（默认 letterbox 模式）
  python resize_images.py D:/Dataset/images -o D:/Dataset/640x640

  # 直接拉伸（不保留宽高比）
  python resize_images.py D:/Dataset/images --mode stretch

  # 裁切到 640×640
  python resize_images.py D:/Dataset/images --mode crop

依赖: pip install Pillow
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("请先安装 Pillow: pip install Pillow")
    sys.exit(1)

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def resize_image(img_path: Path, out_path: Path, size: tuple[int, int], mode: str, quality: int):
    """Resize 单张图片"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    target_w, target_h = size

    if mode == 'stretch':
        # 直接拉伸到目标尺寸
        resized = img.resize(size, Image.LANCZOS)

    elif mode == 'crop':
        # 等比缩放使短边填满，再居中裁切
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        resized = img.crop((left, top, left + target_w, top + target_h))

    else:  # letterbox (默认)
        # 等比缩放使长边符合目标尺寸，短边黑边填充
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        resized = Image.new('RGB', size, (0, 0, 0))
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2
        resized.paste(img, (left, top))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(out_path, quality=quality)
    return (w, h), resized.size


def main():
    parser = argparse.ArgumentParser(description='图片批量 Resize 到 640×640')
    parser.add_argument('input', help='输入图片路径或目录')
    parser.add_argument('-o', '--out', default='./output_640x640',
                        help='输出路径（默认 ./output_640x640）')
    parser.add_argument('-s', '--size', default='640,640',
                        help='目标尺寸: Width,Height（默认 640,640）')
    parser.add_argument('-m', '--mode', choices=['letterbox', 'stretch', 'crop'],
                        default='letterbox',
                        help='缩放模式: letterbox(黑边填充) / stretch(拉伸) / crop(裁切)')
    parser.add_argument('-q', '--quality', type=int, default=95,
                        help='JPEG 质量 1-100（默认 95）')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅预览效果，不实际处理')
    args = parser.parse_args()

    size = tuple(int(x) for x in args.size.split(','))
    in_path = Path(args.input)

    # --- 收集待处理图片 ---
    if in_path.is_file():
        files = [in_path]
    elif in_path.is_dir():
        files = [p for p in sorted(in_path.rglob('*'))
                 if p.suffix.lower() in SUPPORTED_EXTS]
        print(f'找到 {len(files)} 张图片')
    else:
        print(f'错误: 路径不存在 — {in_path}')
        sys.exit(1)

    out_dir = Path(args.out)

    # --- 处理 ---
    ok = fail = 0
    for src in files:
        # 计算输出路径（保持原始目录结构）
        if in_path.is_file():
            rel = src.name
        else:
            rel = src.relative_to(in_path)
        dst = out_dir / rel

        orig_size_text = f'{src.name}  ({src.stat().st_size / 1024:.0f} KB)'
        if args.dry_run:
            print(f'[DRY]  {orig_size_text}  →  {dst}')
            ok += 1
            continue

        try:
            orig_size, new_size = resize_image(src, dst, size, args.mode, args.quality)
            print(f'  ✓  {orig_size_text}  {orig_size[0]}×{orig_size[1]} → {new_size[0]}×{new_size[1]}')
            ok += 1
        except Exception as e:
            print(f'  ✗  {orig_size_text}  错误: {e}')
            fail += 1

    # --- 汇总 ---
    print(f'\n完成: {ok} OK', end='')
    if fail:
        print(f', {fail} FAIL', end='')
    print(f'  →  {out_dir.resolve()}')


if __name__ == '__main__':
    main()
