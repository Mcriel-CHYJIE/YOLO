#!/usr/bin/env python
# =============================================================================
# 模型分析工具 — 误报/漏检分析 + F1 曲线 + debug 截图
# 用法：
#   python tools/analyze.py                          # 用默认 val 集
#   python tools/analyze.py --source test             # 用 test 集
#   python tools/analyze.py --model runs/train/exp/weights/best.pt
#   python tools/analyze.py --conf 0.5                # 指定推理阈值
# =============================================================================
import sys, os, argparse, time
from pathlib import Path
from collections import Counter
import cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 确保在项目根目录运行 ──
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import ops
import torch


def parse_args():
    p = argparse.ArgumentParser(description='YOLO 模型误报/漏检分析')
    p.add_argument('--model', default='', help='模型路径，默认自动找最新 best.pt')
    p.add_argument('--source', default='val', help='数据 split: val / test')
    p.add_argument('--conf', type=float, default=0.0, help='推理置信度阈值（0=自动从 F1 曲线选最佳）')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--output', default='runs/analyze', help='输出目录')
    return p.parse_args()


def find_best_model():
    """自动找最新的 best.pt"""
    runs = ROOT / 'runs' / 'train'
    if not runs.exists():
        print('⚠  No training runs found in runs/train/')
        return None
    exps = sorted(runs.iterdir(), key=lambda p: p.stat().st_mtime)
    for exp in reversed(exps):
        w = exp / 'weights' / 'best.pt'
        if w.exists():
            return w
    return None


def load_data_yaml():
    import yaml
    py = ROOT / 'main' / 'project.yaml'
    if py.exists():
        with open(py, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        dy_path = cfg.get('project', {}).get('data_yaml', '')
        if dy_path:
            dyp = ROOT / dy_path
            if dyp.exists():
                return yaml.safe_load(dyp.read_text(encoding='utf-8'))
    pj = ROOT / 'main' / 'config' / 'paths.json'
    if pj.exists():
        import json
        paths = json.loads(pj.read_text(encoding='utf-8'))
        dd = Path(paths.get('dataset_dir', ''))
        if dd.exists():
            dyp = dd / 'data.yaml'
            if dyp.exists():
                return yaml.safe_load(dyp.read_text(encoding='utf-8'))
    dyp = ROOT / 'datasets' / 'data.yaml'
    if dyp.exists():
        return yaml.safe_load(dyp.read_text(encoding='utf-8'))
    dyp = ROOT / 'datasets.yaml'
    if dyp.exists():
        return yaml.safe_load(dyp.read_text(encoding='utf-8'))
    return None


def get_img_paths(data_yaml, split):
    """获取 split 下所有图片路径"""
    path = Path(data_yaml.get('path') or data_yaml.get('path', '.'))
    if not path.exists():
        path = Path('.')
    img_dir = path / 'images' / split
    if not img_dir.exists():
        print(f'⚠  {img_dir} 不存在')
        return []
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def load_labels(img_path, data_yaml):
    """加载 YOLO 格式标注"""
    path = Path(data_yaml.get('path') or data_yaml.get('path', '.'))
    if not path.exists():
        path = Path('.')
    lbl_dir = path / 'labels' / img_path.parent.name.replace('images', 'labels')
    lbl_file = lbl_dir / f'{img_path.stem}.txt'
    if not lbl_file.exists():
        return np.empty((0, 5))
    boxes = []
    with open(lbl_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append([float(x) for x in parts[:5]])
    return np.array(boxes) if boxes else np.empty((0, 5))


def yolo_to_xyxy(boxes, w, h):
    """YOLO 格式 [cls,xc,yc,w,h] → [x1,y1,x2,y2] 像素坐标"""
    if len(boxes) == 0:
        return np.empty((0, 4))
    xyxy = boxes[:, 1:].copy()
    xyxy[:, 0] = (boxes[:, 1] - boxes[:, 3] / 2) * w  # x1
    xyxy[:, 1] = (boxes[:, 2] - boxes[:, 4] / 2) * h  # y1
    xyxy[:, 2] = (boxes[:, 1] + boxes[:, 3] / 2) * w  # x2
    xyxy[:, 3] = (boxes[:, 2] + boxes[:, 4] / 2) * h  # y2
    return xyxy


def iou(box1, box2):
    """两个框 [x1,y1,x2,y2] 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def match_detections(gt_boxes, det_boxes, iou_thresh=0.5):
    """
    匹配检测框与标注框。
    返回 (tp, fp, fn, fp_indices, fn_indices)
    """
    tp = 0
    matched_gt = set()
    fp_indices = []
    # 对每个检测框，找匹配的 GT
    for di, det in enumerate(det_boxes):
        best_iou = 0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            i = iou(det[:4], gt[:4])
            if i > best_iou:
                best_iou = i
                best_gi = gi
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gi)
        else:
            fp_indices.append(di)

    fp = len(fp_indices)
    fn = len(gt_boxes) - len(matched_gt)
    fn_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return tp, fp, fn, fp_indices, fn_indices


def draw_debug(img_path, det_boxes, gt_boxes, fp_idx, fn_idx, output_path, names):
    """绘制 debug 图：绿色=TP，红色=FP，蓝色=FN"""
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]

    # GT 框（蓝色虚线）
    for gi, gt in enumerate(gt_boxes):
        cls_id = int(gt[0])
        x1, y1, x2, y2 = gt[1:5].astype(int)
        label = names.get(cls_id, f'cls_{cls_id}')
        if gi in fn_idx:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝 = 漏检
            cv2.putText(img, f'FN:{label}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 检测框
    for di, det in enumerate(det_boxes):
        x1, y1, x2, y2 = det[:4].astype(int)
        conf = det[4]
        cls_id = int(det[5])
        label = names.get(cls_id, f'cls_{cls_id}')
        if di in fp_idx:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红 = 误报
            cv2.putText(img, f'FP:{label}', (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿 = TP
            cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(str(output_path), img)


def compute_f1(confidences, matches, n_gt, num_points=101):
    """计算不同 conf 阈值下的 Precision/Recall/F1"""
    thresholds = np.linspace(0, 1, num_points)
    precisions = []
    recalls = []
    f1s = []

    for thresh in thresholds:
        tp = sum(1 for c, m in zip(confidences, matches) if c >= thresh and m)
        fp = sum(1 for c, m in zip(confidences, matches) if c >= thresh and not m)
        fn = n_gt - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return thresholds, precisions, recalls, f1s


def main():
    args = parse_args()
    print(f'[DEBUG] ROOT={ROOT}')
    print(f'[DEBUG] source={args.source}, model={args.model}, conf={args.conf}')

    # ── 数据配置 ──
    data_yaml = load_data_yaml()
    if data_yaml is None:
        print('✗  Cannot find data.yaml')
        print('  Tried:')
        for p in [ROOT/'main'/'project.yaml', ROOT/'datasets'/'data.yaml', ROOT/'datasets.yaml']:
            print(f'    {p}  exists={p.exists()}')
        return
    names = data_yaml.get('names', {})
    print(f'[DEBUG] data_yaml path={data_yaml.get("path")}')
    print(f'[DEBUG] data_yaml keys={list(data_yaml.keys())}')
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    print(f'  Classes: {names}')
    print(f'  Split:   {args.source}')

    # ── 加载模型 ──
    model_path = Path(args.model) if args.model else find_best_model()
    print(f'[DEBUG] model_path={model_path}')
    if model_path is None or not model_path.exists():
        print(f'✗  Model not found: {model_path}')
        if not args.model:
            print('  No --model specified, tried auto-detect in runs/train/')
        return
    print(f'  Model:   {model_path}')
    model = YOLO(str(model_path))

    # ── 图片列表 ──
    img_paths = get_img_paths(data_yaml, args.source)
    print(f'[DEBUG] get_img_paths returned {len(img_paths)} images')
    if not img_paths:
        print(f'⚠  No images found in images/{args.source}/')
        print(f'   Place test images in: {Path(data_yaml.get("path", ".")) / "images" / args.source}')
        return
    print(f'  Images:  {len(img_paths)}')
    print(f'[PROGRESS] 开始推理 0/{len(img_paths)}', flush=True)

    # ── 推理 ──
    out_dir = Path(args.output) / f'{args.source}_{datetime.now().strftime("%m%d_%H%M")}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[PRINT] 开始批量推理: {len(img_paths)} 张, device=0, imgsz={args.imgsz}', flush=True)
    t0 = time.time()
    results = model.predict(
        source=[str(p) for p in img_paths],
        conf=args.conf if args.conf > 0 else 0.001,
        iou=0.5,
        imgsz=args.imgsz,
        device='0',
        verbose=False,
    )
    t1 = time.time()
    print(f'[PRINT] 批量推理完成: {len(results)} 结果, 耗时 {t1-t0:.1f}s', flush=True)

    print(f'[PROGRESS] 推理完成，开始分析', flush=True)

    # ── 逐图分析 ──
    all_confidences = []
    all_matches = []     # 每个检测框是否匹配 GT
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_class = {}       # {cls_id: {'tp':0, 'fp':0, 'fn':0}}

    print(f'\n{"─"*60}')
    print(f'  分析中...')
    total_imgs = len(img_paths)

    for idx, r in enumerate(results):
        print(f'[PROGRESS] {idx+1}/{total_imgs}', flush=True)
        img_path = img_paths[idx]
        gt_raw = load_labels(img_path, data_yaml)
        gt_boxes = yolo_to_xyxy(gt_raw, r.orig_img.shape[1], r.orig_img.shape[0]) if len(gt_raw) else np.empty((0, 4))
        # 拼接 cls_id
        gt_full = np.column_stack([gt_raw[:, 0], gt_boxes]) if len(gt_raw) else np.empty((0, 5))

        # 检测框
        dets = r.boxes
        det_full = np.empty((0, 6))
        if dets is not None and len(dets):
            xyxy = dets.xyxy.cpu().numpy()
            conf = dets.conf.cpu().numpy().reshape(-1, 1)
            cls = dets.cls.cpu().numpy().reshape(-1, 1)
            det_full = np.column_stack([xyxy, conf, cls])

        tp, fp, fn, fp_idx, fn_idx = match_detections(gt_full, det_full, iou_thresh=0.5)

        # 收集 F1 数据
        for di, det in enumerate(det_full):
            all_confidences.append(float(det[4]))
            all_matches.append(di not in fp_idx)

        # 按类别统计
        for gi in range(len(gt_full)):
            cid = int(gt_full[gi, 0])
            per_class.setdefault(cid, {'tp': 0, 'fp': 0, 'fn': 0})
            if gi in fn_idx:
                per_class[cid]['fn'] += 1
        for di in range(len(det_full)):
            cid = int(det_full[di, 5])
            per_class.setdefault(cid, {'tp': 0, 'fp': 0, 'fn': 0})
            if di in fp_idx:
                per_class[cid]['fp'] += 1
            elif di not in fp_idx:
                per_class[cid]['tp'] += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 跳过逐图 debug 截图，只输出最终统计和 F1 曲线

    # ── 汇总 ──
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f'\n{"="*60}')
    print(f'  总览')
    print(f'{"="*60}')
    print(f'  TP: {total_tp}   FP: {total_fp}   FN: {total_fn}')
    print(f'  Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}')
    print()

    # ── 各类别统计 ──
    print(f'{"─"*60}')
    print(f'  按类别')
    print(f'{"─"*60}')
    for cid in sorted(per_class.keys()):
        st = per_class[cid]
        p = st['tp'] / (st['tp'] + st['fp']) if (st['tp'] + st['fp']) > 0 else 0
        r = st['tp'] / (st['tp'] + st['fn']) if (st['tp'] + st['fn']) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        name = names.get(cid, f'cls_{cid}')
        print(f'  {name:15s}  TP:{st["tp"]:3d}  FP:{st["fp"]:3d}  FN:{st["fn"]:3d}  '
              f'P:{p:.3f}  R:{r:.3f}  F1:{f:.3f}')

    # ── F1-Confidence 曲线 ──
    n_gt = total_tp + total_fn
    if n_gt > 0 and len(all_confidences) > 0:
        thresholds, precisions, recalls, f1s = compute_f1(all_confidences, all_matches, n_gt)

        best_idx = np.argmax(f1s)
        best_conf = thresholds[best_idx]
        best_f1 = f1s[best_idx]
        best_p = precisions[best_idx]
        best_r = recalls[best_idx]

        print(f'\n{"─"*60}')
        print(f'  最佳推理阈值')
        print(f'{"─"*60}')
        print(f'  conf={best_conf:.3f} → P={best_p:.3f}  R={best_r:.3f}  F1={best_f1:.3f}')

        # 画曲线
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(thresholds, precisions, label='Precision', lw=1.5)
        ax.plot(thresholds, recalls, label='Recall', lw=1.5)
        ax.plot(thresholds, f1s, label='F1', lw=2)
        ax.axvline(best_conf, color='gray', ls='--', alpha=0.5)
        ax.scatter([best_conf], [best_f1], color='red', zorder=5)
        ax.annotate(f'conf={best_conf:.2f}\nF1={best_f1:.3f}',
                     xy=(best_conf, best_f1),
                     xytext=(best_conf + 0.15, best_f1 - 0.1),
                     arrowprops=dict(arrowstyle='->'), fontsize=9)
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'F1-Confidence Curve (best F1={best_f1:.3f} @ conf={best_conf:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        curve_path = out_dir / 'f1_curve.png'
        fig.savefig(str(curve_path), dpi=150)
        plt.close()
        print(f'  曲线: {curve_path}')

    print(f'\n  输出: {out_dir}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    from datetime import datetime
    main()
