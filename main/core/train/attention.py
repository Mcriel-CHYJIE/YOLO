"""注意力模块集合 — SE / CBAM / CA / ECA / SimAM / EMA / GAM
所有模块接受 (b,c,h,w) 输入，输出同 shape。

LoRA 模块 — LoRAConv2d + inject_lora，卷积低秩适配训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════
# SE — Squeeze-and-Excitation
# ═══════════════════════════════════════════
class SE(nn.Module):
    """Squeeze-and-Excitation 通道注意力，轻量级"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        c = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, c, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


# ═══════════════════════════════════════════
# CBAM — Convolutional Block Attention Module
# ═══════════════════════════════════════════
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        c = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, c, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, channels, 1),
        )

    def forward(self, x):
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        max_ = self.mlp(torch.max(x, dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0])
        return torch.sigmoid(avg + max_)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        return torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    """通道 + 空间注意力"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)


# ═══════════════════════════════════════════
# CA — Coordinate Attention (CVPR 2021)
# 对小目标和细长物体友好
# ═══════════════════════════════════════════
class CA(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        c = max(channels // reduction, 4)
        self.conv1 = nn.Conv2d(channels, c, 1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv_h = nn.Conv2d(c, channels, 1)
        self.conv_w = nn.Conv2d(c, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = F.relu(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


# ═══════════════════════════════════════════
# ECA — Efficient Channel Attention (CVPR 2020)
# 用 1D 卷积替代 SE 的全连接层，极轻量
# 对火焰颜色特征敏感的通道选择友好
# ═══════════════════════════════════════════
class ECA(nn.Module):
    """Efficient Channel Attention — 1D conv 自适应 kernel"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        import math
        t = int(abs((math.log2(channels) + b) / gamma))
        k = max(t if t % 2 else t + 1, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ═══════════════════════════════════════════
# SimAM — Simple Attention Module (ICML 2021)
# 基于能量函数的无参数注意力
# 对小目标、遮挡场景友好，无需额外参数
# ═══════════════════════════════════════════
class SimAM(nn.Module):
    """Simple Attention — 能量函数，无额外可学习参数"""
    def __init__(self, channels=None, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        x_minus_mu = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = x_minus_mu / (4 * (x_minus_mu.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


# ═══════════════════════════════════════════
# EMA — Efficient Multi-Scale Attention (2023)
# 多尺度并行分支 + 跨空间信息融合
# 对人体/火焰的多尺度特征提取效果好
# ═══════════════════════════════════════════
class EMA(nn.Module):
    """Efficient Multi-Scale Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        c = max(channels // reduction, 4)
        self.conv1 = nn.Conv2d(channels, c, 1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.conv3 = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.conv4 = nn.Conv2d(c * 2, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shortcut = x
        # 压缩通道
        x = self.conv1(x)
        b, c, h, w = x.shape
        # 并行分支
        x1 = self.conv2(x)  # 3x3 局部
        x2 = self.conv3(x)  # 3x3 局部
        # 全局上下文融合
        x1_g = x1.mean(dim=(2, 3), keepdim=True).expand_as(x1)
        x2_g = x2.mean(dim=(2, 3), keepdim=True).expand_as(x2)
        # 拼接 + 还原通道
        x = torch.cat([x1 * x1_g, x2 * x2_g], dim=1)
        x = self.conv4(x)
        return shortcut * self.sigmoid(x)


# ═══════════════════════════════════════════
# GAM — Global Attention Mechanism (2022)
# 全维度注意力：保留空间 + 通道精细结构
# 对火焰边缘、人体轮廓等细节敏感
# ═══════════════════════════════════════════
class GAM(nn.Module):
    """Global Attention — reduce→conv→expand 全维度"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        c = max(channels // reduction, 4)
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.Conv2d(channels, c, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, channels, 1),
        )
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, c, 7, padding=3, groups=min(channels, 4)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, channels, 7, padding=3, groups=min(channels, 4)),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道门控
        c_map = self.channel_attn(x)
        x = x * self.sigmoid(c_map)
        # 空间门控
        s_map = self.spatial_attn(x)
        return x * self.sigmoid(s_map)


# ═══════════════════════════════════════════
# Attention Wrapper — 用于注入到 YOLO 模型中
# ═══════════════════════════════════════════
ATTENTION_MAP = {
    'se': SE,
    'cbam': CBAM,
    'ca': CA,
    'eca': ECA,
    'simam': SimAM,
    'ema': EMA,
    'gam': GAM,
}


class AttentionWrapper(nn.Module):
    """包裹一个现有层，在其输出后应用注意力"""
    def __init__(self, module, attn_type, channels):
        super().__init__()
        self.module = module
        cls = ATTENTION_MAP.get(attn_type)
        if cls:
            self.attn = cls(channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        x = self.module(x)
        return self.attn(x)


def inject_attention(model, attn_type):
    """
    将注意力模块注入到 YOLO 模型中。
    遍历 model.model 中的所有模块，将 C2f / C3k2 等替换为 AttentionWrapper。
    """
    if attn_type == 'none' or not attn_type:
        return

    try:
        from ultralytics.nn.modules import C2f, C3k2, Bottleneck, C3
    except ImportError:
        return

    TARGETS = (C2f, C3k2, C3, Bottleneck)
    replaced = 0
    seq = model.model

    for i in range(len(seq)):
        module = seq[i]
        for target in TARGETS:
            if isinstance(module, target):
                c = None
                if hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):
                    c = module.cv2.conv.out_channels
                elif hasattr(module, 'cv3') and hasattr(module.cv3, 'conv'):
                    c = module.cv3.conv.out_channels
                elif hasattr(module, 'cv1') and hasattr(module.cv1, 'conv'):
                    c = module.cv1.conv.out_channels

                if c and c > 0:
                    seq[i] = AttentionWrapper(module, attn_type, c)
                    replaced += 1
                break

    return replaced


# ═══════════════════════════════════════════
# LoRA — Low-Rank Adaptation for Conv2d
# ═══════════════════════════════════════════

class LoRAConv2d(nn.Module):
    """冻结原 Conv2d，附加可训练的 1×1 低秩旁路"""
    def __init__(self, original_conv, rank=4):
        super().__init__()
        self.conv = original_conv
        for p in self.conv.parameters():
            p.requires_grad = False  # 冻结原权重

        c_in = original_conv.in_channels
        c_out = original_conv.out_channels
        self.lora_down = nn.Conv2d(c_in, rank, 1, bias=False)
        self.lora_up = nn.Conv2d(rank, c_out, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.conv(x) + self.lora_up(self.lora_down(x))


def inject_lora(model, rank):
    """
    遍历 YOLO model.model，将 3×3 Conv2d（stride=1）替换为 LoRAConv2d。
    递归搜索所有子模块，替换内部 Bottleneck 等深层 Conv2d。
    跳过 Detect head 和前 3 层（stem / 浅层特征）。
    """
    if rank <= 0:
        return 0

    replaced = 0
    seq = model.model if hasattr(model, 'model') else model

    def _walk(module, depth=0):
        nonlocal replaced
        for name, child in module.named_children():
            # 跳过 Detect head
            if hasattr(child, 'proj') or (hasattr(child, 'cv2') and not hasattr(child, 'cv1')):
                continue
            if isinstance(child, nn.Conv2d):
                ks = child.kernel_size
                s = child.stride
                if ks[0] == ks[1] and ks[0] >= 3 and s[0] == 1 and s[1] == 1:
                    setattr(module, name, LoRAConv2d(child, rank))
                    replaced += 1
            else:
                _walk(child, depth + 1)

    # 跳过前 3 层（stem）
    for i in range(len(seq)):
        if i < 3:
            continue
        _walk(seq[i])

    return replaced
