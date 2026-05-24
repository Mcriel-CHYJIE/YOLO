"""
注意力模块集合 — SE / CBAM / CA（Coordinate Attention）
所有模块接受 (b,c,h,w) 输入，输出同 shape。
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
# 对小目标和细长物体友好，适合火灾/烟雾检测
# ═══════════════════════════════════════════
class CA(nn.Module):
    """Coordinate Attention — 将位置编码编码到通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        c = max(channels // reduction, 4)
        self.conv1 = nn.Conv2d(channels, c, 1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv_h = nn.Conv2d(c, channels, 1)
        self.conv_w = nn.Conv2d(c, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # X方向池化 (b,c,1,w) + Y方向池化 (b,c,h,1)
        x_h = torch.mean(x, dim=3, keepdim=True)      # b,c,h,1
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)  # b,c,1,w → b,c,w,1
        # 拼接 + 卷积
        y = torch.cat([x_h, x_w], dim=2)  # b,c,h+w,1
        y = F.relu(self.bn1(self.conv1(y)))
        # 分离
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # b,c,1,w
        # 激活
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


# ═══════════════════════════════════════════
# Attention Wrapper — 用于注入到 YOLO 模型中
# ═══════════════════════════════════════════
class AttentionWrapper(nn.Module):
    """包裹一个现有层，在其输出后应用注意力"""
    def __init__(self, module, attn_type, channels):
        super().__init__()
        self.module = module
        if attn_type == 'se':
            self.attn = SE(channels)
        elif attn_type == 'cbam':
            self.attn = CBAM(channels)
        elif attn_type == 'ca':
            self.attn = CA(channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        x = self.module(x)
        return self.attn(x)


def inject_attention(model, attn_type):
    """
    将注意力模块注入到 YOLO 模型中。
    遍历 model.model 中的所有模块，将 C2f / C3k2 替换为 AttentionWrapper。
    """
    if attn_type == 'none' or not attn_type:
        return

    # 从 ultralytics 找到 C2f 类
    try:
        from ultralytics.nn.modules import C2f, C3k2, Bottleneck, C3
    except ImportError:
        return  # 无法导入，跳过

    TARGETS = (C2f, C3k2, C3, Bottleneck)
    replaced = 0
    seq = model.model  # nn.Sequential

    for i in range(len(seq)):
        module = seq[i]
        for target in TARGETS:
            if isinstance(module, target):
                # 获取输出通道数
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
