"""
提取指定文件夹下各子文件夹中的视频到统一目录，
自动以父文件夹路径为前缀避免同名冲突。

用法：
    python extract_videos.py D:\Dataset\Real
    python extract_videos.py D:\Dataset\Real --out D:\Dataset\videos
    python extract_videos.py D:\Dataset\Real --dry-run
    python extract_videos.py D:\Dataset\Real --move
"""

import os
import sys
import shutil
from pathlib import Path
import argparse

# ── 视频扩展名 ──
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}

# ── 排除的文件夹名 ──
EXCLUDE_DIRS = {'_all_videos', '__pycache__', '.omc', 'detection_results'}


def collect_videos(root: Path, exclude_dirs: set) -> list[tuple[Path, str]]:
    """遍历 root，返回 (视频文件绝对路径, 相对于 root 的路径字符串)"""
    videos = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        cur = Path(dirpath)
        for f in filenames:
            if Path(f).suffix.lower() in VIDEO_EXTS:
                rel = cur.relative_to(root)
                videos.append((cur / f, str(rel)))
    return videos


def build_dest_name(rel_path: str, filename: str) -> str:
    """用相对路径构建唯一前缀。例如: 3.2/上边/video1.mp4 → 3.2_上边_video1.mp4"""
    prefix = rel_path.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
    return f"{prefix}_{filename}"


def main():
    parser = argparse.ArgumentParser(description='提取子文件夹视频到统一目录')
    parser.add_argument('root', help='要扫描的文件夹路径')
    parser.add_argument('--out', default=None,
                        help='输出目录 (默认 <root>\\_all_videos)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅预览，不实际复制')
    parser.add_argument('--move', action='store_true',
                        help='移动模式 (而非复制)')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f' 目录不存在: {root}')
        sys.exit(1)

    out_dir = Path(args.out) if args.out else (root / '_all_videos')
    out_dir = out_dir.resolve()

    # ── 收集视频 ──
    print(f' 扫描目录: {root}')
    videos = collect_videos(root, EXCLUDE_DIRS)

    if not videos:
        print('  未找到任何视频文件')
        sys.exit(0)

    print(f'🔍 找到 {len(videos)} 个视频文件')

    # ── 检查命名冲突（加前缀后仍有同名的极低概率，但防一手） ──
    dest_map: dict[str, list[Path]] = {}
    for src_path, rel in videos:
        dest_name = build_dest_name(rel, src_path.name)
        dest_map.setdefault(dest_name, []).append(src_path)

    collisions = {k: v for k, v in dest_map.items() if len(v) > 1}
    if collisions:
        print(f'  发现 {len(collisions)} 个冲突，即使加前缀仍同名：')
        for name, sources in collisions.items():
            print(f'  "{name}" 来自:')
            for s in sources:
                print(f'    - {s}')
        print('请手动处理冲突后再运行')
        sys.exit(1)

    # ── 预览 ──
    print(f'\n📋 预览（前 20 个）:')
    for src_path, rel in videos[:20]:
        dest_name = build_dest_name(rel, src_path.name)
        action = 'MOVE' if args.move else 'COPY'
        print(f'  {src_path.parent.name}/{src_path.name}')
        print(f'    → {action} to {out_dir / dest_name}')
    if len(videos) > 20:
        print(f'  ... 还有 {len(videos) - 20} 个')

    if args.dry_run:
        print('\n Dry-run 完成，未执行任何操作')
        return

    # ── 执行 ──
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    errors = 0

    for src_path, rel in videos:
        dest_name = build_dest_name(rel, src_path.name)
        dest_path = out_dir / dest_name

        try:
            if args.move:
                shutil.move(str(src_path), str(dest_path))
            else:
                shutil.copy2(str(src_path), str(dest_path))
            copied += 1
        except Exception as e:
            print(f' 失败: {src_path} → {e}')
            errors += 1

    action = '移动' if args.move else '复制'
    print(f'\n 完成: {action}了 {copied} 个视频到 {out_dir}')
    if errors:
        print(f'  失败: {errors} 个')


if __name__ == '__main__':
    main()
