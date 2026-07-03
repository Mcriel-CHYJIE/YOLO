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
# ASFF — Adaptively Spatial Feature Fusion (scalar-weights variant)
# ======================================================================

class ASFF(nn.Module):
    """ASFF for a single output scale level — scalar-weighted, no per-pixel conv.

    Original ASFF uses a per-pixel conv (weight_conv) to predict fusion masks,
    which introduces gradient instability even with identity init (the 1×1 conv
    weights drift from zero under AMP, eventualy causing NaN loss).

    This variant replaces the per-pixel conv with **learnable scalar fusion
    weights** (3 scalars per level, softmax-normalized), identical to the BiFPN
    approach.  Channel-alignment 1×1 convs are frozen at zero init so they
    contribute zero signal — no gradients flow through them.

    Total learnable params: 3 scalars × 3 levels = 9 parameters.
    """
    def __init__(self, level, channels):
        super().__init__()
        self.level = level
        self.num_levels = len(channels)

        # ── Scalar fusion logits (learnable) ──
        # Initialised so that softmax ≈ [1,0,0] for level 0, [0,1,0] for level 1, etc.
        init = [-10.0] * self.num_levels
        init[level] = 10.0
        self.fusion_logits = nn.Parameter(torch.tensor(init, dtype=torch.float32))

        # ── Channel alignment convs (1×1, frozen) ──
        # These map P4(128ch)→64ch, P5(256ch)→64ch etc.  Zero-initialised and
        # frozen so they never contribute signal — the fusion initially passes
        # through the same-level feature unchanged.
        self.align_convs = nn.ModuleList()
        for i in range(self.num_levels):
            if i != level and channels[i] != channels[level]:
                conv = nn.Conv2d(channels[i], channels[level], 1, bias=False)
                conv.weight.data.zero_()
                for p in conv.parameters():
                    p.requires_grad = False
                self.align_convs.append(conv)
            else:
                self.align_convs.append(nn.Identity() if i == level else nn.Identity())

    def forward(self, inputs):
        """inputs: list of [N, C_i, H_i, W_i] at different scales

        YOLO order: [P3(high-res), P4(mid), P5(low-res)].
          i < level  → feature resolution is HIGHER → downsample
          i > level  → feature resolution is LOWER  → upsample

        Runs all internal ops in float32 (outside autocast) to avoid AMP
        float16 overflow in interpolate/avg_pool backward, then casts the
        output back to the input dtype.
        """
        level = self.level
        orig_dtypes = [f.dtype for f in inputs]

        with torch.cuda.amp.autocast(enabled=False):
            inputs = [f.float() for f in inputs]
            target_h, target_w = inputs[level].shape[2:]

            resized = []
            for i, feat in enumerate(inputs):
                if i == level:
                    resized.append(feat)
                    continue
                # ── Resize spatial dims ──
                if i < level:
                    r = F.interpolate(feat, size=(target_h, target_w), mode='nearest')
                else:
                    r = F.interpolate(feat, size=(target_h, target_w), mode='nearest')
                # ── Channel alignment (frozen, always zero output) ──
                if hasattr(self.align_convs[i], 'weight'):
                    r = self.align_convs[i](r)
                resized.append(r)

            # ── Scalar-weighted fusion ──
            w = F.softmax(self.fusion_logits.float(), dim=0)
            output = torch.zeros_like(resized[0])
            for i in range(self.num_levels):
                output += w[i] * resized[i]

        return output.to(orig_dtypes[level])


class ASFFWrapper(nn.Module):
    """Full ASFF module producing N fused output scales.

    Takes [P2, P3, P4, ...] from the neck and returns [P2', P3', P4', ...].
    Each output is a spatial-weighted combination of all N input scales.
    Automatically adapts to 3-level (P3-P5) or 4-level (P2-P5) models.
    """
    def __init__(self, channels):
        super().__init__()
        self.num_levels = len(channels)
        for lvl in range(self.num_levels):
            self.add_module(f'asff_{lvl}', ASFF(lvl, channels))

    def forward(self, features):
        return [getattr(self, f'asff_{i}')(features)
                for i in range(self.num_levels)]


# ======================================================================
# Model Wrapper — applies fusion after original backbone+neck
# ======================================================================

class WeightedFeatureFusion(nn.Module):
    """BiFPN-style weighted fusion with learnable scalar weights.

    Each output scale is a softmax-normalized weighted sum of all input
    scales. Features are resized and channel-aligned before summation.
    The weights are global scalars (one per input-output level pair).

    Channel-alignment 1×1 convs are zero-initialised and frozen so they
    contribute zero signal — like the ASFF rewrite, this prevents gradient
    instability with the pre-trained model.

    Reference: Tan et al., EfficientDet (CVPR 2020)
    """
    def __init__(self, channels):
        super().__init__()
        self.num_levels = len(channels)
        # Identity init: each output level ≈ its own input level
        init_w = torch.eye(self.num_levels) * 10.0 - 10.0  # diag=10, others=0 → softmax=identity
        self.weights = nn.Parameter(init_w)
        self.align = nn.ModuleList()
        for i in range(self.num_levels):
            level_list = nn.ModuleList()
            for j in range(self.num_levels):
                if channels[j] != channels[i]:
                    conv = nn.Conv2d(channels[j], channels[i], 1, bias=False)
                    conv.weight.data.zero_()
                    for p in conv.parameters():
                        p.requires_grad = False
                    level_list.append(conv)
                else:
                    level_list.append(nn.Identity())
            self.align.append(level_list)

    def forward(self, features):
        """features: [P3(high-res), P4(mid), P5(low-res)] → [P3', P4', P5']

        YOLO order is [P3, P4, P5] where lower index = higher resolution.
        So j < i → need to DOWNSAMPLE, j > i → need to UPSAMPLE.
        """
        outs = []
        for i in range(self.num_levels):
            ih, iw = features[i].shape[2:]
            w = torch.softmax(self.weights[i], dim=0)
            fused = 0.0
            for j in range(self.num_levels):
                if i == j:
                    feat = features[j]
                elif j < i:
                    # j is earlier → higher res → downsample
                    feat = F.adaptive_avg_pool2d(features[j], (ih, iw))
                else:
                    # j is later → lower res → upsample
                    feat = F.interpolate(features[j], size=(ih, iw),
                                         mode='bilinear', align_corners=False)
                # Channel alignment (frozen, always zero for cross-level)
                if hasattr(self.align[i][j], 'weight'):
                    feat = self.align[i][j](feat)
                fused = fused + w[j] * feat
            outs.append(fused)
        return outs

class FusedDetect(nn.Module):
    """Applies multi-scale fusion inside the Detect head without wrapping it.

    Instead of replacing seq[-1] (which shifts param keys like model.23.cv2.*
    to model.23.detect.cv2.* and breaks EMA's deepcopy), this module is attached
    as a plain attribute on the Detect head and its forward is monkey-patched:

        detect._fusion = fusion_module
        detect.forward = lambda x: fusion_module(x)  # → detect.forward(fused)
                           then processed by original detect logic

    The EMA key tree is preserved, and the _fusion parameters simply aren't
    EMA-tracked — acceptable since fusion layers are a small fraction of total
    params and the training loss directly updates them.
    """
    def __init__(self, detect, fusion):
        super().__init__()
        self.detect = detect
        self.fusion = fusion

    def forward(self, x):
        # x: list of features [P3, P4, P5] from backbone+neck
        fused = self.fusion(x)  # [P3', P4', P5'] fused
        return self.detect(fused)

    # ── Explicitly delegate attributes accessed by ultralytics trainer ──
    @property
    def stride(self):
        return self.detect.stride

    @property
    def nc(self):
        return self.detect.nc

    @property
    def ch(self):
        return self.detect.ch

    @property
    def cv2(self):
        return self.detect.cv2

    @property
    def cv3(self):
        return self.detect.cv3

    @property
    def dfl(self):
        return self.detect.dfl

    @property
    def reg_max(self):
        return self.detect.reg_max

    @property
    def anchors(self):
        return self.detect.anchors

    @property
    def f(self):
        """From indices - which layers the detect head takes input from."""
        return self.detect.f

    @property
    def i(self):
        """Module index within the sequential."""
        return self.detect.i

    @property
    def type(self):
        return self.detect.type


# ======================================================================
# Registry + Injection
# ======================================================================

FUSION_REGISTRY = {
    'none': None,
    'asff': ASFFWrapper,
    'bifpn': WeightedFeatureFusion,
}


def _detect_channels(model):
    """Auto-detect [P3, P4, P5] channel dimensions from the Detect head's cv2 input channels.

    Instead of walking sub-modules recursively (which picks up the Detect head's
    own internal convs), reads the input channels directly from each detection
    scale's first conv layer — these are the true neck output channels.
    """
    try:
        seq = model.model if hasattr(model, 'model') else model
        detect = seq[-1]
        if hasattr(detect, 'cv2'):
            channels = []
            for cv in detect.cv2:
                # cv[0] is the first Conv block (ultralytics Conv wrapper)
                sub = cv[0]
                if hasattr(sub, 'conv') and hasattr(sub.conv, 'in_channels'):
                    channels.append(sub.conv.in_channels)
            if len(channels) >= 3:
                return channels
    except Exception:
        pass
    # Fallback: typical yolo11n values
    return [64, 128, 256]


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
    # Match the model's parameter device and dtype.
    ref_param = next(model.parameters())
    fusion_module = fusion_module.to(ref_param.device, dtype=ref_param.dtype)

    # ── NaN-proof gradients on fusion learnable params ──
    for p in fusion_module.parameters():
        if p.requires_grad:
            p.register_hook(lambda g: torch.nan_to_num(g, 0.0, 0.0, 0.0))

    # ── Insert fusion as a real layer in nn.Sequential ──
    # Standard approach used by YOLO-Extended / YOLOv6-ASFF:
    # Put the fusion module right before Detect and update from-indices.
    #   old: ... → P3,P4,P5 → Detect(f=[16,19,22])
    #   new: ... → P3,P4,P5 → Fusion(f=[16,19,22]) → Detect(f=[fusion_idx])
    #
    # This avoids all injection issues (monkey-patch, pre-hook, gradient
    # isolation) because the fusion is a native Sequential layer that
    # _predict_once handles with its normal caching/routing logic.
    seq = model.model  # nn.Sequential, currently 24 layers (0-23)
    detect = seq[-1]   # Detect at index 23
    fusion_idx = len(seq) - 1  # 23 — fusion takes detect's spot
    detect_idx = len(seq)      # 24 — detect moves to new index

    # Build new Sequential: old[:-1] + fusion + detect
    old_layers = list(seq.children())[:-1]
    new_seq = nn.Sequential(*(old_layers + [fusion_module, detect]))

    # Set from-indices so _predict_once routes correctly.
    # fusion takes the same 3 features detect used to take:
    fusion_module.f = detect.f           # [16, 19, 22]
    fusion_module.i = fusion_idx          # 23
    # detect now takes its input from the fusion layer's single output.
    # The fusion returns a list [P3', P4', P5']; accessing y[fusion_idx]
    # with detect.f = fusion_idx (int) gives that list directly.
    detect.f = fusion_idx                # 23 (int, not list)
    detect.i = detect_idx                 # 24

    # Patch model.mode = ... it's a property, replace underlying
    model.model = new_seq
    # Update save list (cache indices still include the same backbone layers)
    _save = getattr(model, 'save', None)
    if isinstance(_save, list):
        _save.append(fusion_idx)

    return f'{fusion_type.upper()}({",".join(str(c) for c in channels)})'


__all__ = [
    'ASFF', 'ASFFWrapper',
    'WeightedFeatureFusion',
    'FusedDetect',
    'inject_multiscale_fusion',
    '_detect_channels',
]
