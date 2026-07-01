"""
Custom Loss Functions for YOLO Training Studio.
Targets: reduce false positives (FP) and false negatives (FN) in fire detection.

Classification losses (replace BCE):
  - FocalLoss: down-weights easy negatives → fewer FNs
  - ASLLoss: independent γ_pos/γ_neg → tunable FP/FN trade-off

Box losses (replace CIoU):
  - WIoU v3: dynamic non-monotonic focusing, better for fuzzy boundaries
  - Focal-EIoU: separate w/h penalty, better for extreme aspect ratios

Usage:
  from main.core.train.loss import patch_yolo_loss, restore_original_loss
  patch_yolo_loss(model, loss_config)
  restore_original_loss()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Classification Losses — replace nn.BCEWithLogitsLoss in criterion.bce
# Return per-element losses (no reduction) to match ultralytics convention.
# ======================================================================


class FocalLoss(nn.Module):
    """Focal Loss — reduces easy-negative contribution, focuses on hard positives.

    L_focal = -α_t · (1-p_t)^γ · log(p_t)

    Higher γ  → more aggressive down-weighting of easy examples → fewer FNs.
    α balances positive/negative class weights.
    """
    def __init__(self, gamma=2.0, alpha=0.75, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: raw logits (B, N_anchor, N_cls)  target: one-hot (B, N_anchor, N_cls)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce.detach())                  # probability of correct
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = focal_weight * alpha_t
        loss = focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ASLLoss(nn.Module):
    """Asymmetric Loss — independent γ for positive and negative samples.

    Set γ_neg > γ_pos → aggressively down-weight easy negatives → fewer FNs.
    Set γ_pos > γ_neg → aggressively down-weight easy positives → fewer FPs.
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, reduction='none'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, pred, target):
        prob = torch.sigmoid(pred).clamp(self.clip, 1 - self.clip)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pos_mask = (target > 0.5).float()
        neg_mask = 1.0 - pos_mask
        pt = pos_mask * prob + neg_mask * (1 - prob)
        gamma = pos_mask * self.gamma_pos + neg_mask * self.gamma_neg
        loss = loss * ((1 - pt) ** gamma)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ======================================================================
# Box Regression Losses — replace bbox_iou in ultralytics.utils.loss
# Called by BboxLoss.forward() with signature:
#   fn(pred_boxes, target_boxes, xywh=True, CIoU=False) → IoU tensor
# Return values are 0~1 (higher = better), loss is computed as 1 - IoU.
# ======================================================================


def _xywh2xyxy(boxes):
    """Convert (x, y, w, h) to (x1, y1, x2, y2)."""
    x, y, w, h = boxes.unbind(dim=-1)
    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1)


def _box_areas(boxes):
    """Compute area from xyxy boxes."""
    return (boxes[..., 2] - boxes[..., 0]).clamp(0) * (boxes[..., 3] - boxes[..., 1]).clamp(0)


def _intersection(box1, box2):
    """Intersection area of two xyxy box tensors."""
    return (torch.min(box1[..., 2:], box2[..., 2:])
            - torch.max(box1[..., :2], box2[..., :2])).clamp(0).prod(dim=-1)


def wiou_loss(box1, box2, xywh=True, CIoU=False):
    """Wise-IoU v3 with dynamic non-monotonic focusing mechanism.

    - R_WIoU: distance-attention penalty (center distance / enclosing diagonal)
    - r: non-monotonic focusing coefficient based on outlier degree
    - Low-quality anchors get smaller gradients, high-quality get larger ones.

    Reference: Tong et al., 2023 (https://arxiv.org/abs/2301.10051)
    """
    if xywh:
        box1 = _xywh2xyxy(box1)
        box2 = _xywh2xyxy(box2)

    inter = _intersection(box1, box2)
    area1 = _box_areas(box1)
    area2 = _box_areas(box2)
    union = area1 + area2 - inter
    iou = inter / (union + 1e-7)

    # Enclosing box
    enclose = (torch.max(box1[..., 2:], box2[..., 2:])
               - torch.min(box1[..., :2], box2[..., :2])).clamp(0)
    c2 = enclose.pow(2).sum(dim=-1) + 1e-7

    # Center distance
    c1 = (box1[..., :2] + box1[..., 2:]) / 2
    c2_pt = (box2[..., :2] + box2[..., 2:]) / 2
    rho2 = (c1 - c2_pt).pow(2).sum(dim=-1)

    # Distance-attention IoU
    R_wiou = torch.exp(rho2 / c2)

    # WIoU v3 — dynamic non-monotonic focusing
    iou_loss = 1 - iou
    with torch.no_grad():
        beta = iou_loss / (iou_loss.mean() + 1e-7)       # outlier degree
        delta, alpha = 0.5, 1.5
        r = torch.clamp(beta / (delta * alpha ** (beta - 1)), 0, 1.5)

    # Return adjusted IoU (lower than plain IoU for low-quality matches)
    return 1 - r * R_wiou * (1 - iou)


def focal_eiou_loss(box1, box2, xywh=True, CIoU=False):
    """Focal-EIoU Loss — focal weighting + separate width/height penalty.

    Standard CIoU couples w/h ratio, which struggles with extreme aspect ratios.
    EIoU separates w/h penalties → better for tall flames / wide smoke.
    Focal weighting focuses gradient on high-quality boxes.

    Reference: Zhang et al., 2022 (https://arxiv.org/abs/2101.08158)
    """
    if xywh:
        box1 = _xywh2xyxy(box1)
        box2 = _xywh2xyxy(box2)

    inter = _intersection(box1, box2)
    area1 = _box_areas(box1)
    area2 = _box_areas(box2)
    union = area1 + area2 - inter
    iou = inter / (union + 1e-7)

    # Center distance
    c1 = (box1[..., :2] + box1[..., 2:]) / 2
    c2_pt = (box2[..., :2] + box2[..., 2:]) / 2
    rho2 = (c1 - c2_pt).pow(2).sum(dim=-1)
    enclose = (torch.max(box1[..., 2:], box2[..., 2:])
               - torch.min(box1[..., :2], box2[..., :2])).clamp(0)
    c2 = enclose.pow(2).sum(dim=-1) + 1e-7

    # Width / height differences
    w1, h1 = box1[..., 2] - box1[..., 0], box1[..., 3] - box1[..., 1]
    w2, h2 = box2[..., 2] - box2[..., 0], box2[..., 3] - box2[..., 1]

    # EIoU = overlap + center + width + height
    eiou = ((1 - iou) + rho2 / c2
            + (w1 - w2).pow(2) / (enclose[..., 0].pow(2) + 1e-7)
            + (h1 - h2).pow(2) / (enclose[..., 1].pow(2) + 1e-7))

    # Focal focusing: weight high-IoU boxes more
    gamma = 0.5
    focal_weight = iou.detach() ** gamma

    return 1 - focal_weight * eiou


# ======================================================================
# Patching Infrastructure
# ======================================================================

# Registry: key → (class_or_fn, display_name)
# 'bce' / 'ciou' = use ultralytics default (None)
CLS_LOSS_REGISTRY = {
    'bce':   (None, 'BCE'),
    'focal': (FocalLoss, 'Focal'),
    'asl':   (ASLLoss, 'ASL'),
}

IOU_LOSS_REGISTRY = {
    'ciou':      (None, 'CIoU'),
    'wiou':      (wiou_loss, 'WIoU'),
    'focaleiou': (focal_eiou_loss, 'Focal-EIoU'),
}

# Saved original for restoration
_original_bbox_iou = None


def patch_yolo_loss(model, loss_config):
    """Monkey-patch the model's criterion with custom loss functions.

    Args:
        model: YOLO model instance (after YOLO() but before model.train())
        loss_config: dict with keys:
            cls_loss: 'bce' | 'focal' | 'asl'
            focal_gamma: float (default 2.0)
            focal_alpha: float (default 0.75)
            asl_gamma_pos: float (default 0.0)
            asl_gamma_neg: float (default 4.0)
            iou_loss: 'ciou' | 'wiou' | 'focaleiou'

    Returns:
        list of patched component names (for logging)
    """
    # pylint: disable=global-statement
    global _original_bbox_iou

    criterion = getattr(model, 'criterion', None)
    if criterion is None:
        return []

    patched = []

    # ── 1. Classification loss ───────────────────────────────────
    cls_type = loss_config.get('cls_loss', 'bce')
    if cls_type != 'bce' and cls_type in CLS_LOSS_REGISTRY:
        cls_class, _ = CLS_LOSS_REGISTRY[cls_type]
        if cls_class is not None and hasattr(criterion, 'bce'):
            if cls_type == 'focal':
                g = loss_config.get('focal_gamma', 2.0)
                a = loss_config.get('focal_alpha', 0.75)
                criterion.bce = cls_class(gamma=g, alpha=a)
                patched.append(f'Focal(γ={g}, α={a})')
            elif cls_type == 'asl':
                gp = loss_config.get('asl_gamma_pos', 0.0)
                gn = loss_config.get('asl_gamma_neg', 4.0)
                criterion.bce = cls_class(gamma_pos=gp, gamma_neg=gn)
                patched.append(f'ASL(γ⁺={gp}, γ⁻={gn})')

    # ── 2. Box regression loss (IoU) ─────────────────────────────
    iou_type = loss_config.get('iou_loss', 'ciou')
    if iou_type != 'ciou' and iou_type in IOU_LOSS_REGISTRY:
        iou_fn, disp = IOU_LOSS_REGISTRY[iou_type]
        if iou_fn is not None:
            # Save original once
            if _original_bbox_iou is None:
                # pylint: disable=import-outside-toplevel
                import ultralytics.utils.loss as _loss_mod
                _original_bbox_iou = _loss_mod.bbox_iou
            import ultralytics.utils.loss as _loss_mod
            _loss_mod.bbox_iou = iou_fn
            patched.append(disp)

    return patched


def restore_original_loss():
    """Restore the original bbox_iou after training ends."""
    # pylint: disable=global-statement
    global _original_bbox_iou
    if _original_bbox_iou is not None:
        import ultralytics.utils.loss as _loss_mod  # pylint: disable=import-outside-toplevel
        _loss_mod.bbox_iou = _original_bbox_iou
        _original_bbox_iou = None
        return True
    return False


__all__ = [
    'FocalLoss', 'ASLLoss',
    'wiou_loss', 'focal_eiou_loss',
    'patch_yolo_loss', 'restore_original_loss',
    'CLS_LOSS_REGISTRY', 'IOU_LOSS_REGISTRY',
]
