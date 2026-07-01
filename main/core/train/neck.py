"""
Multi-Scale Feature Fusion modules for YOLO.

Currently implemented:
  - ASFF (Adaptively Spatial Feature Fusion, https://arxiv.org/abs/1911.09516)
  - BiFPN (Weighted Feature Fusion, https://arxiv.org/abs/1911.09070)
    Learnable spatial weighting of multi-scale features before the detection head.
    Each output scale is a weighted combination of all input scales, with
    spatial weights predicted from the features themselves.

Relevance to fire detection:
  - Fire varies enormously in scale (tiny flame to full-frame)
  - ASFF helps the model simultaneously detect small and large fires
  - Learnable fusion adapts to input content

Usage:
  from main.core.train.neck import inject_multiscale_fusion
  result = inject_multiscale_fusion(model, 'asff')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# ASFF — Adaptively Spatial Feature Fusion
# ======================================================================

class ASFF(nn.Module):
    """ASFF for a single output scale level.

    For output level l, takes all 3 input scales (P3/P4/P5), resizes them
    to the spatial size of level l, and computes a per-pixel weighted sum
    with learnable softmax weights.
    """
    def __init__(self, level, channels):
        super().__init__()
        self.level = level
        self.num_levels = len(channels)
        self.channels = channels[level]

        # Channel alignment convs (1x1) to project all scales to the same channel dim
        self.align_convs = nn.ModuleList()
        for i in range(self.num_levels):
            if i != level and channels[i] != channels[level]:
                self.align_convs.append(nn.Conv2d(channels[i], channels[level], 1, bias=False))
            else:
                self.align_convs.append(nn.Identity() if i == level else nn.Identity())

        # Weight predictor: concat of aligned features -> 3 weight maps
        in_c = channels[level] * self.num_levels
        self.weight_conv = nn.Sequential(
            nn.Conv2d(in_c, self.num_levels, 1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs):
        """inputs: list of [N, C_i, H_i, W_i] at different scales"""
        level = self.level
        target_h, target_w = inputs[level].shape[2:]

        resized = []
        for i, feat in enumerate(inputs):
            if i == level:
                resized.append(feat)
            elif i < level:
                # Smaller scale -> upsample
                r = F.interpolate(feat, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
                r = self.align_convs[i](r)
                resized.append(r)
            else:
                # Larger scale -> downsample
                r = F.adaptive_avg_pool2d(feat, (target_h, target_w))
                r = self.align_convs[i](r)
                resized.append(r)

        # Concatenate and predict spatial weight maps
        concat = torch.cat(resized, dim=1)
        weights = self.weight_conv(concat)  # [N, 3, H, W]

        # Weighted sum
        output = 0
        for i in range(self.num_levels):
            output += weights[:, i:i+1] * resized[i]
        return output


class ASFFWrapper(nn.Module):
    """Full ASFF module producing 3 fused output scales.

    Takes [P3, P4, P5] from the neck and returns [P3', P4', P5'].
    Each output is a spatial-weighted combination of all 3 input scales.
    """
    def __init__(self, channels):
        super().__init__()
        self.asff_0 = ASFF(0, channels)  # P3 output (high-res)
        self.asff_1 = ASFF(1, channels)  # P4 output (mid)
        self.asff_2 = ASFF(2, channels)  # P5 output (low-res)

    def forward(self, features):
        return [
            self.asff_0(features),
            self.asff_1(features),
            self.asff_2(features),
        ]


# ======================================================================
# Model Wrapper — applies fusion after original backbone+neck
# ======================================================================

class WeightedFeatureFusion(nn.Module):
    """BiFPN-style weighted fusion with learnable scalar weights.

    Each output scale is a softmax-normalized weighted sum of all input
    scales. Features are resized and channel-aligned before summation.
    The weights are global scalars (one per input-output level pair).

    Reference: Tan et al., EfficientDet (CVPR 2020)
    """
    def __init__(self, channels):
        super().__init__()
        self.num_levels = len(channels)
        self.weights = nn.Parameter(torch.ones(self.num_levels, self.num_levels))
        self.align = nn.ModuleList()
        for i in range(self.num_levels):
            level_list = nn.ModuleList()
            for j in range(self.num_levels):
                if channels[j] != channels[i]:
                    level_list.append(nn.Conv2d(channels[j], channels[i], 1, bias=False))
                else:
                    level_list.append(nn.Identity())
            self.align.append(level_list)

    def forward(self, features):
        """features: [P3, P4, P5] -> [P3", P4", P5"]"""
        outs = []
        for i in range(self.num_levels):
            ih, iw = features[i].shape[2:]
            w = torch.softmax(self.weights[i], dim=0)
            fused = 0.0
            for j in range(self.num_levels):
                if i == j:
                    feat = features[j]
                elif j < i:
                    feat = F.interpolate(features[j], size=(ih, iw),
                                         mode='bilinear', align_corners=False)
                else:
                    feat = F.adaptive_avg_pool2d(features[j], (ih, iw))
                feat = self.align[i][j](feat)
                fused = fused + w[j] * feat
            outs.append(fused)
        return outs

class FusedSequential(nn.Module):
    """Wraps model.model (nn.Sequential) to add fusion after the neck.

    Implements __getitem__/__len__/__iter__ so external code that indexes
    into model.model (e.g. for layer access) still works.
    """
    def __init__(self, original, fusion):
        super().__init__()
        self.original = original
        self.fusion = fusion

    def forward(self, x):
        features = self.original(x)   # [P3, P4, P5] from backbone+neck
        return self.fusion(features)  # [P3', P4', P5'] fused

    def __getitem__(self, idx):
        return self.original[idx]

    def __len__(self):
        return len(self.original)

    def __iter__(self):
        return iter(self.original)


# ======================================================================
# Registry + Injection
# ======================================================================

FUSION_REGISTRY = {
    'none': None,
    'asff': ASFFWrapper,
    'bifpn': WeightedFeatureFusion,
}


def _detect_channels(model):
    """Auto-detect [c3, c4, c5] channel dimensions from model."""
    try:
        seq = model.model if hasattr(model, 'model') else model
        # Walk backward, find the last 3 distinct Conv2d output channels
        conv_channels = []
        seen = set()
        for module in reversed(list(seq.modules())):
            if isinstance(module, nn.Conv2d) and module.out_channels not in seen:
                conv_channels.append(module.out_channels)
                seen.add(module.out_channels)
                if len(conv_channels) == 3:
                    break
        if len(conv_channels) == 3:
            return list(reversed(conv_channels))
    except Exception:
        pass
    # Fallback: typical YOLO11n values
    return [256, 256, 512]


def inject_multiscale_fusion(model, fusion_type, channels=None):
    """Inject multi-scale feature fusion into model.model.

    Args:
        model: YOLO model instance (post __init__)
        fusion_type: str - 'asff' or 'none'
        channels: [c3, c4, c5] or None for auto-detect

    Returns:
        str description of injected module, or '' if none
    """
    if not fusion_type or fusion_type == 'none':
        return ''

    if fusion_type not in FUSION_REGISTRY:
        return ''

    fusion_cls = FUSION_REGISTRY[fusion_type]
    if fusion_cls is None:
        return ''

    if channels is None:
        channels = _detect_channels(model)

    fusion_module = fusion_cls(channels)
    original = model.model
    model.model = FusedSequential(original, fusion_module)
    return f'{fusion_type.upper()}({channels[0]},{channels[1]},{channels[2]})'


__all__ = [
    'ASFF', 'ASFFWrapper',
    'WeightedFeatureFusion',
    'inject_multiscale_fusion',
    '_detect_channels',
]
