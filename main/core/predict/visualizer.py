"""
Predict可视化 — 热力图 + 特征图生成模块
适用于 Ultralytics YOLO (v8/v11) 模型
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple


# ══════════════════════════════════════════════════════════════
# 热力图生成（基于 backbone 最后特征图的激活响应）
# ══════════════════════════════════════════════════════════════

def draw_boxes(
    image_bgr: np.ndarray,
    results,
    names: dict,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """
    在图像上绘制 YOLO 检测框（不修改原始图像）。

    Args:
        image_bgr: BGR 图像
        results: YOLO predict 返回的单张结果对象（有 boxes 属性）
        names: class id → name 映射（model.names）
        conf_threshold: 仅显示置信度 ≥ 此值的框

    Returns:
        绘制了框的 BGR 图像副本
    """
    img = image_bgr.copy()
    if results.boxes is None or len(results.boxes) == 0:
        return img

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

    CLASS_COLORS = [
        (16, 185, 129), (59, 130, 246), (245, 158, 11), (239, 68, 68),
        (139, 92, 246), (236, 72, 153), (20, 184, 166), (249, 115, 22),
        (34, 197, 94), (168, 85, 247), (244, 63, 94), (14, 165, 233),
        (234, 179, 8), (99, 102, 241), (236, 72, 153), (20, 184, 166),
    ]

    for i in range(len(boxes)):
        conf = float(confs[i])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, boxes[i])
        cls_id = cls_ids[i]
        label = names.get(cls_id, f'cls_{cls_id}')
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        # 画框（线宽固定 1px 保持清晰）
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        # 标签背景
        text = f'{label} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def _get_internal_model(model):
    """获取 YOLO 内部 nn.Sequential 层序列"""
    m = model.model
    # YOLO 的 .model 可能是 DetectionModel（有 .model 属性指向 nn.Sequential）
    if hasattr(m, 'model') and isinstance(m.model, torch.nn.Sequential):
        return m.model
    return m

def _find_last_feature_layer(model):
    """
    找到 backbone 最后一个特征提取层（Detect 之前的层）。
    返回 (layer_index, layer_module)。
    """
    from ultralytics.nn.modules import Detect
    seq = _get_internal_model(model)
    last_idx = len(seq) - 1
    for i in range(len(seq) - 1, -1, -1):
        if not isinstance(seq[i], Detect):
            last_idx = i
            break
    return last_idx, seq[last_idx]


class _FeatureHook:
    """注册 forward hook 收集层输出"""
    def __init__(self):
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output.detach()


def _preprocess_img(model, image_bgr, device=None):
    """手动预处理图像：letterbox + BGR→RGB + CHW + /255 → Tensor"""
    h, w = image_bgr.shape[:2]
    # 获取模型期望的输入尺寸
    if isinstance(model.model.args, dict):
        imgsz = model.model.args.get('imgsz', 640)
    elif hasattr(model.model.args, 'imgsz'):
        imgsz = model.model.args.imgsz
    else:
        imgsz = 640
    if isinstance(imgsz, (list, tuple)):
        imgsz = imgsz[0]
    # letterbox resize
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # pad
    dh, dw = imgsz - nh, imgsz - nw
    top, left = dh // 2, dw // 2
    img = cv2.copyMakeBorder(img, top, dh - top, left, dw - left,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # BGR→RGB, HWC→CHW, /255
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def compute_heatmap(
    model,
    image_bgr: np.ndarray,
    alpha: float = 0.4,
    cmap: int = cv2.COLORMAP_JET,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """
    CAM 热力图 — backbone 最后特征图通道平均 + padding 裁剪对齐。
    """
    h_orig, w_orig = image_bgr.shape[:2]
    imgsz = _get_imgsz(model)
    r = min(imgsz / h_orig, imgsz / w_orig)
    nh, nw = int(round(h_orig * r)), int(round(w_orig * r))
    dh, dw = imgsz - nh, imgsz - nw
    top, left = dh // 2, dw // 2

    return _simple_heatmap(model, image_bgr, alpha, cmap,
                           h_orig, w_orig, imgsz, nh, nw, top, left)


def _get_imgsz(model):
    if isinstance(model.model.args, dict):
        imgsz = model.model.args.get('imgsz', 640)
    elif hasattr(model.model.args, 'imgsz'):
        imgsz = model.model.args.imgsz
    else:
        imgsz = 640
    return imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz


def _crop_padding(heat, imgsz, nh, nw, top, left, w_orig, h_orig):
    """去掉 letterbox padding 后 resize 到原图尺寸"""
    fm_h, fm_w = heat.shape
    x_scale = fm_w / imgsz
    y_scale = fm_h / imgsz
    margin = 2
    x1 = max(0, int(left * x_scale) - margin)
    y1 = max(0, int(top * y_scale) - margin)
    x2 = min(fm_w, int((left + nw) * x_scale) + margin)
    y2 = min(fm_h, int((top + nh) * y_scale) + margin)
    if x2 > x1 and y2 > y1:
        heat = heat[y1:y2, x1:x2]
    return cv2.resize(heat, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)


def _render_heatmap(heat, image_bgr, alpha, cmap):
    """归一化 + 伪彩色 + 叠加"""
    if heat.max() > 0:
        heat = heat / heat.max()
    heat = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cmap)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heat_color, alpha, 0)


def _simple_heatmap(model, image_bgr, alpha, cmap,
                     h_orig, w_orig, imgsz, nh, nw, top, left):
    """普通激活热力图（通道平均）"""
    idx, layer = _find_last_feature_layer(model)
    hook = _FeatureHook()
    handle = layer.register_forward_hook(hook.hook_fn)

    orig_mode = model.model.training
    model.model.eval()
    with torch.no_grad():
        device = next(model.model.parameters()).device
        tensor = _preprocess_img(model, image_bgr, device)
        _ = model.model(tensor)
    handle.remove()
    model.model.train(orig_mode)

    if hook.features is None:
        return image_bgr.copy()

    heat = hook.features[0].mean(dim=0).cpu().numpy().astype(np.float32)
    heat = np.maximum(heat, 0)
    heat = _crop_padding(heat, imgsz, nh, nw, top, left, w_orig, h_orig)
    return _render_heatmap(heat, image_bgr, alpha, cmap)


# ══════════════════════════════════════════════════════════════
# 特征图可视化（中间层激活网格）
# ══════════════════════════════════════════════════════════════

def _find_layer_groups(model):
    """
    按位置选取 3 个特征层（~25%, ~50%, ~75% 位置），
    兼容所有 YOLO 变体（v8/v11/CBAM/LoRA）。
    返回 [(name, layer_module), ...]。
    """
    from ultralytics.nn.modules import Detect
    seq = _get_internal_model(model)

    # 先找到 Detect 前的有效层范围
    valid_end = len(seq)
    for i in range(len(seq) - 1, -1, -1):
        if isinstance(seq[i], Detect):
            valid_end = i
        else:
            break
    if valid_end == 0:
        valid_end = len(seq)

    # 按位置均分取 3 层
    if valid_end >= 3:
        indices = [int(valid_end * 0.2), int(valid_end * 0.5), int(valid_end * 0.8)]
        # 去重
        indices = sorted(set(i for i in indices if i < valid_end))
    else:
        indices = list(range(valid_end))

    groups = []
    for i in indices:
        layer = seq[i]
        name = f'layer_{i}_{type(layer).__name__}'
        groups.append((name, layer))

    return groups


def extract_feature_maps(
    model,
    image_bgr: np.ndarray,
    max_channels: int = 16,
    grid_cols: int = 8,
    target_width: int = 300,
) -> List[np.ndarray]:
    """
    提取模型中间层的特征图（通过 forward hook 捕获，支持 skip-connection）。
    动态调整 cell 大小使整个网格适合 target_width。
    返回 [(layer_name, grid_image_bgr), ...]。
    """
    h_orig, w_orig = image_bgr.shape[:2]

    # 找目标层
    layers = _find_layer_groups(model)
    if not layers:
        return []

    hooks = []
    orig_mode = model.model.training
    model.model.eval()

    for name, layer in layers:
        hook = _FeatureHook()
        handle = layer.register_forward_hook(hook.hook_fn)
        hooks.append((name, hook, handle))

    # 完整 forward（skip-connection 依赖完整图）
    with torch.no_grad():
        device = next(model.model.parameters()).device
        tensor = _preprocess_img(model, image_bgr, device)
        _ = model.model(tensor)

    model.model.train(orig_mode)

    # 渲染每个特征图
    results = []
    for name, hook, handle in hooks:
        handle.remove()
        if hook.features is None:
            continue

        feats = hook.features[0]  # [C, H, W]
        C, H, W = feats.shape

        n_show = min(C, max_channels)
        if n_show <= 0:
            continue

        stride = max(1, C // n_show)
        indices = list(range(0, C, stride))[:n_show]
        if indices and indices[-1] != C - 1 and len(indices) < max_channels:
            indices[-1] = C - 1

        n_cols = min(n_show, grid_cols)
        cell_w = max(16, min(120, int(target_width / n_cols)))
        cell_h = max(16, int(cell_w * H / W))
        n_rows = (n_show + grid_cols - 1) // grid_cols

        grid_w = n_cols * cell_w + (n_cols - 1) * 1 + 2
        grid_h = n_rows * cell_h + (n_rows - 1) * 1 + 2
        grid = np.ones((grid_h, grid_w), dtype=np.uint8) * 240

        for inner_idx, ch_idx in enumerate(indices):
            row = inner_idx // grid_cols
            col = inner_idx % grid_cols

            fm = feats[ch_idx].cpu().numpy().astype(np.float32)
            fm = fm - fm.min()
            if fm.max() > 0:
                fm = fm / fm.max()
            fm_8u = (fm * 255).astype(np.uint8)
            fm_resized = cv2.resize(fm_8u, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)

            y0 = row * (cell_h + 1) + 1
            x0 = col * (cell_w + 1) + 1
            grid[y0:y0 + cell_h, x0:x0 + cell_w] = fm_resized

        grid_color = cv2.applyColorMap(grid, cv2.COLORMAP_JET)
        results.append((name, grid_color, (n_cols, n_rows)))

    return results


def render_feature_map_grid(
    feature_map_list: List[Tuple[str, np.ndarray, Tuple[int, int]]],
    image_bgr: np.ndarray,
    max_width: int = 800,
) -> np.ndarray:
    """
    将多个层的特征网格合并为一张垂直布局的大图。

    Args:
        feature_map_list: extract_feature_maps 的返回值
        image_bgr: 原图（用于显示输入缩略图）
        max_width: 输出图最大宽度

    Returns:
        BGR 格式的完整可视化图像
    """
    panels = []

    # 第一个 panel: 输入原图
    ih, iw = image_bgr.shape[:2]
    if max(iw, ih) > 200:
        sc = 200 / max(iw, ih)
        thumb = cv2.resize(image_bgr, (int(iw * sc), int(ih * sc)))
    else:
        thumb = image_bgr
    # 加白边框
    thumb_bordered = cv2.copyMakeBorder(thumb, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h_label = 18
    label_img = np.ones((h_label, thumb_bordered.shape[1], 3), dtype=np.uint8) * 245
    cv2.putText(label_img, 'Input', (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
    panels.append(np.vstack([label_img, thumb_bordered]))

    # 每个层一个 panel
    for name, grid, (n_cols, n_rows) in feature_map_list:
        h_label = 18
        label_img = np.ones((h_label, grid.shape[1], 3), dtype=np.uint8) * 245
        cv2.putText(label_img, f'Layer: {name}  ({n_cols}x{n_rows} grid)', (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        panels.append(np.vstack([label_img, grid]))

    # 垂直拼接
    if not panels:
        return image_bgr.copy()

    # 统一宽度
    widths = [p.shape[1] for p in panels]
    target_w = min(max(widths), max_width)

    resized = []
    for p in panels:
        pw = p.shape[1]
        if pw != target_w:
            sc = target_w / pw
            new_h = int(p.shape[0] * sc)
            p = cv2.resize(p, (target_w, new_h), interpolation=cv2.INTER_NEAREST)
        resized.append(p)

    return np.vstack(resized)
