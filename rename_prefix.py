"""
重命名文件，将文件名开头的单数字前缀补零。
例如: 5.12_test1.mp4 → 05.12_test1.mp4

匹配规则：文件名开头若为 `数字.` 且数字是 1 位（1-9），则补成 2 位（01-09）。

用法：
    python rename_prefix.py D:\Dataset\_all_videos          # 处理该目录下所有文件
    python rename_prefix.py D:\Dataset\_all_videos --dry-run # 预览
    python rename_prefix.py D:\Dataset\Real                 # 指定其他文件夹
"""

import os
import sys
import re
from pathlib import Path
import argparse


def zero_pad_prefix(filename: str) -> tuple[str, bool]:
    """
    将文件名开头的单数字补零。
    例如: '5.12_test1.mp4' → '05.12_test1.mp4'
          '1.13_test2.mp4' → '01.13_test2.mp4'
          '5.1_test3.mp4'  → '05.1_test3.mp4'
          'Real_1.13_test.mp4' → 不变（非数字开头）
    返回 (新文件名, 是否修改)
    """
    # 匹配开头: 1-2位数字 + 点 + 数字
    m = re.match(r'^(\d{1,2})(\.\d)', filename)
    if not m:
        return filename, False

    num = m.group(1)
    rest = m.group(2) + filename[m.end():]

    if len(num) == 1:
        return f'0{num}{rest}', True
    else:
        return filename, False


def main():
    parser = argparse.ArgumentParser(description='将文件名开头的单数字补零')
    parser.add_argument('folder', help='目标文件夹路径')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅预览，不实际重命名')
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f'❌ 目录不存在: {folder}')
        sys.exit(1)

    files = sorted(f for f in folder.iterdir() if f.is_file())
    if not files:
        print(f'⚠️  目录为空: {folder}')
        sys.exit(0)

    renamed = 0
    skipped = 0

    for f in files:
        new_name, changed = zero_pad_prefix(f.name)
        if not changed:
            skipped += 1
            continue

        new_path = f.parent / new_name

        if new_path.exists():
            print(f'⚠️  跳过，目标已存在: {new_path.name}')
            continue

        if args.dry_run:
            print(f'  {f.name}  →  {new_name}')
        else:
            f.rename(new_path)
            print(f'  {f.name}  →  {new_name}')

        renamed += 1

    print(f'\n{"📋 预览: " if args.dry_run else "✅ 完成: "}'
          f'重命名 {renamed} 个, 跳过 {skipped} 个')

    if renamed == 0:
        print('所有文件名前缀格式已正确，无需修改')


if __name__ == '__main__':
    main()
