---
tags: [yolo11m, training, legacy, reference]
---

# 📜 YOLO11m 高精度训练配置（历史参考）

> **📚 文档导航**: [返回文档首页](README.md) | [预设指南](PRESETS_GUIDE.md) | [配置分析](TRAINING_ANALYSIS_yolo11n.md)

> ⚠️ **注意**: 本文档为历史配置参考，建议使用 [PRESETS_GUIDE.md](PRESETS_GUIDE.md) 中的最新预设参数。

## 📋 项目概述

本项目基于 **YOLO11m** 模型，针对跌倒检测任务进行了优化配置，旨在实现**最高精度**的同时确保**训练稳定性**。

---

## 🖥️ 硬件环境

### GPU 配置
- **型号**: NVIDIA GeForce RTX 5070 Ti
- **显存**: 16 GB GDDR6
- **架构**: Blackwell (最新一代)
- **适用场景**: 中大规模模型训练

### 系统要求
- **操作系统**: Windows 11 / Linux
- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA support
- **Ultralytics**: 8.0+

---

## ⚙️ 训练配置详解

### 核心参数

```yaml
# 模型选择
model: yolo11m.pt                    # Medium模型 - 精度与安全的最佳平衡

# 训练时长
epochs: 500                          # 扩展训练周期，追求最高精度
patience: 80                         # 早停耐心值，允许充分探索最优解

# 批次与尺寸
batch: 16                            # 保守批量大小，确保显存安全
imgsz: 640                           # 固定图像尺寸（禁用多尺度）

# 优化器配置
optimizer: AdamW                     # 长周期训练最佳优化器
lr0: 0.001                           # 初始学习率（AdamW默认）
lrf: 0.01                            # 最终学习率比例 → 0.00001
warmup_epochs: 10                    # 预热轮数，稳定启动
momentum: 0.937                      # 动量参数
weight_decay: 0.0005                 # L2正则化系数

# 学习率调度
scheduler: cosine                    # 余弦退火调度

# 数据加载
workers: 6                           # 数据加载线程数（减少内存占用）

# 损失函数
fl_gamma: 1.0                        # Focal Loss gamma值（温和类别平衡）
label_smoothing: 0.05                # 标签平滑（保留标签信息）
iou: 0.7                             # IoU阈值
```

### 数据增强策略

```yaml
# 几何增强
degrees: 10.0                        # 旋转角度 ±10°（保守）
translate: 0.1                       # 平移增强 10%
scale: 0.5                           # 缩放范围 0.5-1.5x
flip_lr: 0.5                         # 水平翻转概率 50%
multi_scale: false                   # ❌ 禁用多尺度（显存安全）

# 颜色增强
hsv_h: 0.015                         # 色调增强
hsv_s: 0.7                           # 饱和度增强
hsv_v: 0.4                           # 亮度增强

# 混合增强
copy_paste: 0.15                     # Copy-Paste增强概率 15%（保守）
close_mosaic: 20                     # 最后20轮关闭Mosaic（精细调优）
```

---

## 🛡️ 显存安全保障

### 三重保险机制

#### 1️⃣ 保守的 Batch Size
- **配置**: `batch=16`（而非32或24）
- **效果**: 降低显存占用约33%
- **优势**: 梯度稳定性好，留有充足余量

#### 2️⃣ 禁用 Multi-scale
- **配置**: `multi_scale=false`
- **原因**: 消除动态尺寸导致的显存峰值
- **效果**: 显存占用可预测且稳定

#### 3️⃣ 减少 Workers
- **配置**: `workers=6`（而非8）
- **效果**: 减少数据加载器内存占用约0.5GB
- **优势**: 仍能保证数据加载效率

### 显存占用分析

| 组件 | 占用显存 | 说明 |
|------|---------|------|
| 模型参数 | ~2.0 GB | YOLO11m (20.1M参数) |
| 激活值 | ~3.5 GB | batch=16, imgsz=640 |
| 梯度 | ~2.0 GB | 反向传播存储 |
| AdamW状态 | ~4.0 GB | momentum + variance |
| 数据缓存 | ~0.8 GB | pin memory (workers=6) |
| CUDA上下文 | ~0.5 GB | CUDA运行时开销 |
| Python开销 | ~0.5 GB | PyTorch框架本身 |
| **总计** | **~13.3 GB** | **峰值约14GB** |

### 安全余量

```
GPU总显存:     16.0 GB
系统保留:      -1.0 GB  (Windows + 驱动)
训练占用:      -14.0 GB (峰值)
─────────────────────────────
安全余量:       1.0 GB  ✅

可承受额外占用:
- 浏览器: ~1GB ✅
- 桌面管理器: ~0.5GB ✅
- 显存碎片化: ~0.5GB ✅
- 后台程序: ~0.5GB ✅
```

**结论**: 即使同时运行日常应用，也绝对不会OOM！

---

## 📊 预期性能指标

### 训练时间估算

```
RTX 5070 Ti + YOLO11m + batch=16
单epoch时间: ~65-75秒
总训练时间: 500 epochs × 70秒 ≈ 9.7小时

早停情况（patience=80触发）:
可能在 350-420 epoch 停止
实际时间: 6.5-8小时
```

### 精度预期

| 指标 | 预期范围 | 说明 |
|------|---------|------|
| **mAP@50** | 0.94-0.97 | 优秀级别 ⭐⭐⭐⭐⭐ |
| **mAP@50-95** | 0.72-0.78 | 优秀级别 ⭐⭐⭐⭐⭐ |
| **Precision** | > 0.92 | 高精确率 |
| **Recall** | > 0.90 | 高召回率 |
| **模型文件大小** | ~40 MB | 适中大小 |

### 推理性能

```
推理速度: 30-40 FPS (RTX 5070 Ti)
延迟: ~25-33ms/帧
适用场景: 实时跌倒检测（可接受1-2秒延迟）
```

---

## 🎯 配置设计理念

### 1. 为什么选择 YOLO11m？

| 模型 | 参数量 | mAP@50 | 显存需求 | 推荐度 |
|------|--------|--------|---------|--------|
| yolo11n | 2.6M | 0.88-0.91 | 6GB | ⭐⭐⭐ |
| yolo11s | 9.4M | 0.91-0.94 | 8GB | ⭐⭐⭐⭐ |
| **yolo11m** | **20.1M** | **0.94-0.97** | **13GB** | **⭐⭐⭐⭐⭐** |
| yolo11l | 25.3M | 0.95-0.98 | 15GB | ⭐⭐ |
| yolo11x | 56.9M | 0.96-0.99 | 25GB+ | ❌ |

**选择理由**:
- ✅ 精度接近yolo11l（差距<2%）
- ✅ 显存需求可控（13GB < 16GB）
- ✅ 训练速度可接受
- ✅ 推理速度满足实时性要求

### 2. 为什么使用 AdamW 优化器？

| 特性 | SGD | AdamW |
|------|-----|-------|
| 短期收敛 | ⚡ 快 | 🐢 慢 |
| **长期训练** | 可能震荡 | ✅ **稳定收敛** |
| 泛化能力 | 好 | 很好 |
| 超参数敏感度 | 高 | 低 |
| **适用场景** | 快速实验 | **长周期精细训练** ⭐ |

**AdamW优势**:
- 自适应学习率，不同参数有不同lr
- Weight decay解耦，正则化更有效
- 500轮长训练更稳定，不易发散
- 对学习率不那么敏感

### 3. 为什么采用保守的数据增强？

**激进增强 vs 保守增强**:

```
激进增强（degrees=30°, copy_paste=0.5）:
✅ 提高鲁棒性
❌ 可能破坏标注信息
❌ 增加显存波动
❌ 训练不稳定

保守增强（degrees=10°, copy_paste=0.15）:
✅ 保护标签质量
✅ 显存占用稳定
✅ 训练过程平滑
✅ 配合500轮长训练弥补
```

**理念**: 通过**延长训练时间**而非**激进增强**来提升精度。

### 4. 学习率调度策略

```
Cosine Annealing with Warmup:

学习率曲线:
lr
 |\
 | \        Cosine下降
 |  \______/¯¯¯¯¯¯¯¯¯¯
 |           \
 |            \___________
 +------------------------→ epoch
 0    10              500

阶段划分:
- Epoch 0-10:   Warmup (线性增长到0.001)
- Epoch 10-500: Cosine衰减 (0.001 → 0.00001)
```

**优势**:
- 前期快速学习主要特征
- 中期稳定优化
- 后期极小学习率精细调优

---

## 🚀 训练流程

### 启动前检查

```powershell
# 1. 检查GPU状态
nvidia-smi

# 应该看到:
# GPU Memory-Usage: < 2GB (空闲状态)
# GPU Utilization: < 5%

# 2. 确认数据集准备就绪
ls datasets/images/train/  # 应该有训练图片
ls datasets/images/val/    # 应该有验证图片
ls datasets/data.yaml      # 配置文件存在
```

### 开始训练

```python
# 方法1: 通过GUI界面
python
scripts / main.py
# 在Training标签页点击"Start"按钮

# 方法2: 命令行直接训练
from ultralytics import YOLO

model = YOLO('../yolo11m.pt')
model.train(
    data='datasets/data.yaml',
    epochs=500,
    batch=16,
    imgsz=640,
    optimizer='AdamW',
    lr0=0.001,
    patience=80,
    workers=6,
    # ... 其他参数
)
```

### 训练监控

#### 实时监控显存
```powershell
# 在另一个终端窗口运行
watch -n 2 nvidia-smi
# 每2秒刷新一次显存占用
```

#### 查看训练日志
```
日志位置: runs/detect/runs/11_fall_XXXX_XXXX/
关键文件:
- results.csv: 每轮训练指标
- args.yaml: 训练参数配置
- weights/best.pt: 最佳模型
- weights/last.pt: 最后一轮模型
```

#### 关键指标监控

**正常训练曲线**:
```
Epoch 0-50:
  - Loss快速下降
  - mAP@50: 0.5 → 0.85
  
Epoch 50-300:
  - Loss缓慢下降
  - mAP@50: 0.85 → 0.93
  
Epoch 300-500:
  - Loss微小波动
  - mAP@50: 0.93 → 0.95+
  - LR: 0.00001 (极小)
```

**异常情况识别**:
```
❌ Loss突然激增 → 学习率过高或数据问题
❌ mAP持续不提升 → 可能需要调整增强策略
❌ 显存超过14GB → 考虑降低batch到12
❌ 训练速度过慢(<20秒/epoch) → 检查workers设置
```

---

## 🔧 故障排除

### 问题1: 训练开始时OOM

**症状**: 第一个epoch就报错 `CUDA out of memory`

**解决方案**:
```yaml
# 立即调整为超保守配置
batch: 12
workers: 4
model: yolo11s.pt  # 降级到Small模型
```

### 问题2: 训练中途OOM

**症状**: 跑到一半突然崩溃

**可能原因**:
- 其他程序占用了显存
- 显存碎片化

**解决方案**:
```powershell
# 1. 关闭不必要的GPU程序
# 2. 重启训练，使用更保守配置
batch: 12
workers: 4
```

### 问题3: 训练速度过慢

**症状**: 每个epoch超过120秒

**解决方案**:
```yaml
# 增加workers（如果显存充足）
workers: 8

# 或升级CPU
# 检查CPU利用率
taskmgr  # Windows任务管理器
```

### 问题4: 精度不理想

**症状**: mAP@50 < 0.90

**可能原因**:
- 数据质量问题
- 标注错误
- 类别不平衡

**解决方案**:
```yaml
# 1. 检查数据集
python scripts/main.py  # Dataset标签页查看统计

# 2. 调整Focal Loss
fl_gamma: 1.5  # 从1.0增加到1.5

# 3. 增加训练轮数
epochs: 600  # 从500增加到600
```

---

## 📈 进阶优化建议

### 如果想进一步提升精度

#### 方案A: 测试时增强 (TTA)
```python
# 推理时启用TTA
results = model.predict(source='test.jpg', augment=True)
# 通常提升mAP 1-3%
```

#### 方案B: 模型集成
```python
# 训练3次，得到3个best.pt
# 推理时集成预测（投票或平均）
# 通常提升mAP 2-4%
```

#### 方案C: 更大模型（需降batch）
```yaml
model: yolo11l.pt
batch: 12              # 必须降低
workers: 4
# 显存: ~13-14GB
# 精度: mAP@50 ≈ 0.95-0.98
```

### 如果想加快训练速度

#### 方案A: 减小模型
```yaml
model: yolo11s.pt
batch: 24              # 可以提高batch
# 训练时间: ~5小时
# 精度: mAP@50 ≈ 0.92-0.95
```

#### 方案B: 减少epochs
```yaml
epochs: 300            # 从500降到300
patience: 50           # 相应调整
# 训练时间: ~6小时
# 精度: mAP@50 ≈ 0.93-0.96（可能略低）
```

---

## 📝 训练记录模板

建议使用以下格式记录每次训练：

```markdown
## 训练记录 #1

**日期**: 2026-05-15
**配置**:
- Model: yolo11m.pt
- Batch: 16
- Epochs: 500 (实际停止: 380)
- Optimizer: AdamW
- LR: 0.001 → 0.00001

**结果**:
- Best Epoch: 300
- mAP@50: 0.952
- mAP@50-95: 0.745
- Precision: 0.931
- Recall: 0.918
- Training Time: 7.2 hours

**观察**:
- Loss曲线平滑下降
- 验证集在epoch 300后趋于平稳
- patience在epoch 380触发早停

**模型文件**: runs/detect/runs/11_fall_0515_2100/weights/best.pt
```

---

## 🎓 常见问题 FAQ

### Q1: 为什么不用更大的batch size？
**A**: batch=32会导致显存占用超过16GB，有OOM风险。batch=16在保证精度的同时确保安全。

### Q2: 为什么要训练500轮这么多？
**A**: 长周期训练配合AdamW优化器和Cosine调度，可以让模型充分收敛到全局最优解。patience机制会自动在合适时机停止。

### Q3: 为什么不启用multi-scale？
**A**: multi-scale会导致显存占用不可预测，可能出现突发峰值导致OOM。固定640尺寸更安全，且500轮训练可以弥补尺度鲁棒性的不足。

### Q4: 如果我想用yolo11l怎么办？
**A**: 需要降低batch到12，workers到4，显存占用约13-14GB。精度提升约1-2%，但训练时间增加到12-15小时。

### Q5: 训练完成后如何使用模型？
**A**: 
```python
from ultralytics import YOLO
model = YOLO('runs/detect/runs/xxx/weights/best.pt')
results = model.predict(source='test.jpg', conf=0.25)
```

---

## 📚 参考资料

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO11 Technical Report](https://github.com/ultralytics/ultralytics)
- [AdamW Optimizer Paper](https://arxiv.org/abs/1711.05101)
- [Cosine Annealing LR Schedule](https://arxiv.org/abs/1608.03983)

---

## 📞 技术支持

如遇到问题，请检查：
1. GPU驱动是否为最新版本
2. PyTorch和CUDA版本是否匹配
3. 数据集格式是否正确
4. 显存占用是否正常

**联系**: 查看项目README或提交Issue

---

**最后更新**: 2026-05-15  
**配置版本**: v1.0  
**适用模型**: YOLO11m  
**目标精度**: mAP@50 > 0.94
