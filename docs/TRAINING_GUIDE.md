# 📚 YOLO Training Platform - Complete Training Guide

> **Last Updated**: 2026-05-19 | **Version**: v2.0 | **Device**: RTX 5070 Ti (16GB VRAM)

---

## 📖 Documentation Navigation

This guide consolidates all training-related configurations, analyses, and best practices into the following chapters:

1. **[Quick Start](#-quick-start)** - Beginner's introduction guide
2. **[Preset Parameters](#-preset-parameters-configuration)** - Detailed explanation of 4 optimized presets
3. **[In-depth Analysis](#-in-depth-configuration-analysis)** - Parameter design principles and theory
4. **[Historical Configuration](#-historical-configuration-reference)** - YOLO11m legacy configuration (reference)
5. **[Monitoring & Tuning](#-training-monitoring-and-tuning)** - Real-time monitoring and troubleshooting
6. **[Feature Updates](#-latest-feature-updates)** - Recent feature improvement records

---

## 🚀 快速开始

### 第一次训练?

**推荐阅读顺序**:
1. 阅读 **[预设参数](#-预设参数配置)** 了解4组预设
2. 选择 **均衡模式**(⭐推荐新手)
3. 修改 `project.yaml` 中的 `training` 配置
4. 运行 `python scripts/main.py` 开始训练

### 想要最高精度?

**推荐阅读顺序**:
1. 阅读 **[预设参数](#-预设参数配置)** 了解高精度模式
2. 阅读 **[深度分析](#-深度配置分析)** 深入理解参数
3. 选择 **高精度模式** 或 **大模型高精度模式**
4. 准备充足训练时间(15-35小时)

### 预设参数速查表

| 预设 | epochs | batch | 时长 | mAP@0.5 | 推荐度 |
|------|--------|-------|------|---------|--------|
| **均衡模式** ⭐ | 300 | 32 | 6-8h | 90-93% | ⭐⭐⭐⭐⭐ |
| **高精度模式** | 800 | 32 | 15-20h | 92-95% | ⭐⭐⭐⭐ |
| **快速实验** | 100 | 48 | 1-2h | 85-88% | ⭐⭐⭐ |
| **大模型高精度** | 600 | 24 | 25-35h | 95-98% | ⭐⭐⭐ |

**完整预设配置**: [PRESETS.yaml](../PRESETS.yaml)

---

## ✨ 最新功能更新 (2026-05-19)

### 🎬 Predict 标签页 - 视频检测优化
- **固定帧率播放**: 视频检测现在以 **24 FPS** 稳定播放(电影标准帧率)
- **不再追求速度**: 确保检测结果清晰可见,便于人工审核
- **导出支持**: 可选择保存带检测框的视频文件

### 🏷️ Label 标签页 - 标注功能增强
- **修复导航bug**: 点击上一张/下一张时正确保存标注状态
- **支持0样本导出**: 无标注图片可作为负样本导出(生成空.txt文件)
- **负样本训练**: 帮助模型学习"什么不是目标",降低误检率
- **友好提示**: 导出时明确显示正样本和负样本数量

---

## 🎯 预设参数配置

本项目提供 **4组优化预设参数**,适用于不同训练场景。所有预设均针对 **RTX 5070 Ti (16GB显存)** 设备优化。

### 预设对比总览

| 预设名称 | 适用场景 | 训练时长 | 预期mAP@0.5 | 推荐度 |
|---------|---------|---------|-------------|--------|
| **高精度模式** | 最终产品模型、研究 | 15-20小时 | 92-95% | ⭐⭐⭐⭐ |
| **均衡模式** ⭐ | 日常训练、迭代 | 6-8小时 | 90-93% | ⭐⭐⭐⭐⭐ |
| **快速实验** | 快速验证、调试 | 1-2小时 | 85-88% | ⭐⭐ |
| **大模型高精度** | 超越yolo11n极限 | 25-35小时 | 95-98% | ⭐⭐⭐⭐ |

---

### 1️⃣ 高精度模式 (Maximum Accuracy)

#### 核心配置
```yaml
model: yolo11n.pt
epochs: 800
batch: 32
imgsz: 640
optimizer: AdamW
scheduler: cosine
patience: 100
lr0: 0.002
warmup_epochs: 15
```

#### 数据增强策略(激进型)
```yaml
fl_gamma: 1.5              # 强焦点损失,关注难例
label_smoothing: 0.1       # 中度平滑,防止过拟合
copy_paste: 0.2            # 激进复制粘贴增强
degrees: 15.0              # ±15°旋转增强
translate: 0.15            # ±15%平移
scale: 0.6                 # 尺度范围[0.4, 1.6]x
close_mosaic: 30           # 最后30轮关闭mosaic精细调优
```

#### ✅ 优势
- 充分挖掘模型潜力,达到理论最优性能
- 强数据增强提升泛化能力
- 长训练周期+长patience不错过后期突破
- 适合对精度要求极高的场景

#### ⚠️ 注意事项
- 训练时间长,需要耐心等待
- 前200轮快速提升,200-600轮稳步提升,600-800轮精细微调
- 建议在数据质量高、标注准确时使用
- 显存占用:9-11GB

#### 📊 学习率曲线
```
LR
↑
|     /\
|    /  \
|   /    \___________
|  /                 \
| /                   \______
|/
+--------------------------------→ Epoch
  0   15          400         800
  |   |            |           |
 Warmup  Cosine Annealing  Final LR
```

---

### 2️⃣ 均衡模式 (Balanced) ⭐ **推荐**

#### 核心配置
```yaml
model: yolo11n.pt
epochs: 300
batch: 32
imgsz: 640
optimizer: AdamW
scheduler: cosine
patience: 50
lr0: 0.001
warmup_epochs: 5
```

#### 数据增强策略(适中型)
```yaml
fl_gamma: 1.0              # 轻度焦点损失
label_smoothing: 0.05      # 轻度平滑
copy_paste: 0.1            # 适度复制粘贴
degrees: 10.0              # ±10°旋转
translate: 0.1             # ±10%平移
scale: 0.5                 # 尺度范围[0.5, 1.5]x
close_mosaic: 15           # 最后15轮关闭mosaic
```

#### ✅ 优势
- **性价比最高**,精度与时间的最佳平衡
- 参数保守稳定,不易出现过拟合
- 适合日常迭代和模型更新
- 显存占用:9-11GB

#### 适用场景
- 数据集更新后的重新训练
- 验证新的数据增强策略
- 模型版本迭代
- 日常维护和优化

#### 📊 训练进度预期
```
Epoch 1-50:   快速学习基础特征 (mAP 0→0.60)
Epoch 50-150: 稳步提升识别能力 (mAP 0.60→0.85)
Epoch 150-250: 精细调优 (mAP 0.85→0.90)
Epoch 250-300: 收敛稳定 (mAP 0.90→0.93)
```

---

### 3️⃣ 快速实验模式 (Fast Experiment)

#### 核心配置
```yaml
model: yolo11n.pt
epochs: 100
batch: 48                    # 更高batch加速训练
imgsz: 640
optimizer: Adam              # Adam收敛更快
scheduler: linear            # 线性调度器
patience: 20
lr0: 0.001
warmup_epochs: 3
```

#### 数据增强策略(轻量型)
```yaml
fl_gamma: 0.0                # 无焦点损失
label_smoothing: 0.0         # 无平滑
copy_paste: 0.0              # 无复制粘贴
degrees: 0.0                 # 无旋转
translate: 0.1               # 仅平移
scale: 0.5                   # 尺度范围[0.5, 1.5]x
close_mosaic: 5              # 仅最后5轮关闭mosaic
```

#### ✅ 优势
- **训练速度最快**,1-2小时完成
- 适合快速验证想法和调试
- 减少数据增强,更容易定位问题
- 显存占用:10-12GB(batch较大)

#### 适用场景
- 测试新添加的训练数据
- 验证数据标注质量
- 调试训练流程和参数
- 快速对比不同模型版本
- 超参数搜索的初始阶段

#### ⚠️ 局限性
- 精度较低,不适合最终部署
- 缺乏数据增强,泛化能力有限
- 仅作为验证和调试工具

---

### 4️⃣ 大模型高精度模式 (Large Model Precision)

#### 核心配置
```yaml
model: yolo11s.pt                    # 或 yolo11m.pt
epochs: 600
batch: 24                            # 降低batch适应大模型
imgsz: 640
optimizer: AdamW
scheduler: cosine
patience: 80
lr0: 0.001
warmup_epochs: 10
workers: 6                           # 减少workers保显存安全
```

#### 数据增强策略(强化型)
```yaml
fl_gamma: 1.5                        # 强焦点损失
label_smoothing: 0.1                 # 中度平滑防过拟合
copy_paste: 0.15                     # 适度复制粘贴
degrees: 15.0                        # ±15°旋转
translate: 0.15                      # ±15%平移
scale: 0.6                           # 尺度范围[0.4, 1.6]x
close_mosaic: 25                     # 最后25轮关闭mosaic
```

#### ✅ 优势
- **精度最高**,超越yolo11n的理论上限
- 大模型容量更强,能学习更复杂的特征
- 适合对精度要求极高的场景
- yolo11s: 95-97% mAP, yolo11m: 96-98% mAP

#### ⚠️ 注意事项
- **训练时间最长**:yolo11s约25-30小时,yolo11m约30-35小时
- **显存占用高**:yolo11s约12-13GB,yolo11m约13-14GB
- **必须降低batch**:从32降至24,yolo11m可能需要降至16
- **减少workers**:从8降至6,避免显存峰值
- 仅在yolo11n精度不满足需求时使用

#### 📊 不同模型显存占用预估
```
yolo11n (2.6M params):  ~9-11 GB @ batch=32
yolo11s (9.4M params):  ~12-13 GB @ batch=24
yolo11m (20.1M params): ~13-14 GB @ batch=16-24
yolo11l (25.3M params): ~14-15 GB @ batch=8-12  ⚠️ 警戒
yolo11x (56.9M params): >16 GB @ batch=4-8     ❌ 不推荐
```

#### 适用场景
- yolo11n精度已达瓶颈(95%+)
- 需要极致的检测精度
- 有足够的训练时间和GPU资源
- 数据质量高且数量充足(>10000张)

#### 📊 训练进度预期(yolo11s)
```
Epoch 1-100:  快速学习基础特征 (mAP 0→0.70)
Epoch 100-300: 稳步提升复杂特征 (mAP 0.70→0.90)
Epoch 300-500: 精细调优难例 (mAP 0.90→0.95)
Epoch 500-600: 收敛稳定 (mAP 0.95→0.97)
```

---

## 🔧 使用方法

### 方法1:直接修改 project.yaml(推荐)

打开 [project.yaml](../project.yaml),将 `training:` 部分替换为对应预设配置(见上方各模式的核心配置)。

### 方法2:GUI界面手动调整

1. 启动训练界面:`python scripts/main.py`
2. 切换到 **Training** 标签页
3. 根据预设参数逐个调整
4. 点击 **Start** 开始训练

### 方法3:使用预设模板文件

已创建 [PRESETS.yaml](../PRESETS.yaml) 包含所有预设配置,可直接复制使用。

---

## 🔬 深度配置分析

### 为什么选择这些参数?

#### 1. 模型选择:yolo11n而非yolo11m/l/x

**yolo11n优势**:
- ✅ 参数量少(~2.6M),训练速度快
- ✅ 显存占用低(预计8-10GB @ batch=32)
- ✅ 更适合长时间精细训练
- ✅ 通过增强数据和长周期弥补容量不足

**权衡**:
- ❌ 理论上限低于大模型
- ✅ 但通过800 epochs + 强增强可接近最优

#### 2. Batch Size = 32的考量

**显存估算**:
```
yolo11n @ 640 input:
- 模型权重: ~10 MB
- 激活值: ~3-4 GB (batch=32)
- 梯度 + 优化器状态: ~2-3 GB
- 预留空间: ~2 GB
- 总计: ~8-10 GB < 16 GB ✓
```

**训练稳定性**: batch=32提供足够的梯度统计稳定性  
**GPU利用率**: 充分利用RTX 5070 Ti的计算能力

#### 3. 800 Epochs的必要性

**收敛曲线预期**:
```
mAP
↑
|        *********
|      **         ***
|    **              ***
|   *                   ****
|  *                        *****
| *                              ******
|*                                    *****
+------------------------------------------→ Epoch
 0   100  200  300  400  500  600  700  800
     |____| |________| |____________|
     Rapid  Steady    Fine-tuning
     Growth Improvement Plateau
```

- **前200轮**: 快速学习基础特征
- **200-600轮**: 稳步提升,学习复杂模式
- **600-800轮**: 微调阶段,挖掘最后1-2%性能

#### 4. 激进数据增强的理由

**目标**: 通过数据多样性弥补小模型容量限制

- **Copy-Paste 0.2**: 特别针对摔倒(squatting/fallen)等稀有类别
- **Rotation 15°**: 人体姿态多变,需要旋转鲁棒性
- **Scale 0.6**: 适应不同拍摄距离和人物大小
- **HSV增强**: 应对光照变化(早晚、室内外)

#### 5. Cosine Scheduler + Long Patience

- **Cosine退火**: 避免陷入局部最优,周期性"重启"探索
- **Patience=100**: 允许模型在 plateau 期继续寻找突破
- **预期**: 可能在epoch 600-750之间出现二次提升

---

### 优化器选择:AdamW vs SGD

| 特性 | SGD | AdamW |
|------|-----|-------|
| 短期收敛 | ⚡ 快 | 🐢 慢 |
| **长期训练** | 可能震荡 | ✅ **稳定收敛** |
| 泛化能力 | 好 | 很好 |
| 超参数敏感度 | 高 | 低 |
| **适用场景** | 快速实验 | **长周期精细训练** ⭐ |

**AdamW优势**:
- 自适应学习率,不同参数有不同lr
- Weight decay解耦,正则化更有效
- 500轮长训练更稳定,不易发散
- 对学习率不那么敏感

---

### 学习率调度策略

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

## ⚠️ 潜在风险与监控要点

### 1. 显存监控

```bash
# 训练时实时监控
nvidia-smi -l 1
```

**显存安全阈值**:
- **正常范围**: 8-12GB
- **警戒线**: >14GB
- **危险线**: >15GB

**OOM处理策略**:
1. **立即降低batch size**: 32 → 24 → 16
2. **减少workers**: 8 → 6 → 4
3. **禁用数据增强**: copy_paste → 0
4. **降低图像尺寸**: 640 → 512

**显存优化建议**:
```yaml
# 如果遇到显存不足,依次尝试:
batch: 32     → 24      → 16
workers: 8    → 6       → 4
copy_paste: 0.2 → 0.1   → 0.0
multi_scale: false        # 保持禁用
```

---

### 2. 过拟合预警信号

| 指标 | 正常 | 警告 | 危险 |
|------|------|------|------|
| train_loss ↓ val_loss ↑ | ✓ | 差距>20% | 差距>50% |
| mAP@0.5 (train-val) | <5% | 5-10% | >10% |
| 连续50轮无提升 | - | ✓ | 考虑early stop |

**应对策略**:
- 若出现过拟合:增大 `label_smoothing` 至 0.15,减小 `copy_paste` 至 0.1
- 若欠拟合:增大 `fl_gamma` 至 2.0,延长训练至 1000 epochs

---

### 3. 训练停滞检测

- **正常现象**: epoch 400-600 可能进入平台期
- **异常停滞**: 连续100轮 mAP 波动 <0.5%
- **解决**: 检查学习率是否过低,必要时手动调整 lr0

---

## 📊 预期训练结果

### 性能预估(基于yolo11n特性)

| 指标 | 保守估计 | 乐观估计 | 备注 |
|------|---------|---------|------|
| **mAP@0.5** | 0.85-0.88 | 0.90-0.93 | 取决于数据质量 |
| **mAP@0.5:0.95** | 0.65-0.70 | 0.72-0.78 | 严格IoU标准 |
| **Precision** | 0.88-0.92 | 0.93-0.96 | 查准率 |
| **Recall** | 0.82-0.86 | 0.87-0.91 | 查全率 |
| **训练时长** | 12-18小时 | - | RTX 5070 Ti估算 |
| **最佳epoch** | 600-750 | 700-800 | 后期突破可能性 |

### 各类别预期表现

| 类别 | 难度 | 预期mAP | 说明 |
|------|------|---------|------|
| **standing** | 低 | 0.92-0.95 | 常见姿态,易识别 |
| **sitting** | 中 | 0.88-0.92 | 与standing有相似性 |
| **squatting** | 中高 | 0.82-0.88 | 可能与fallen混淆 |
| **fallen** | 高 | 0.80-0.86 | 稀有类别,依赖增强 |

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
python scripts/main.py
# 在Training标签页点击"Start"按钮

# 方法2: 命令行直接训练
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(
    data='datasets/data.yaml',
    epochs=800,
    batch=32,
    imgsz=640,
    optimizer='AdamW',
    lr0=0.002,
    patience=100,
    workers=8,
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
  - mAP@0.5: 0.5 → 0.85
  
Epoch 50-300:
  - Loss缓慢下降
  - mAP@0.5: 0.85 → 0.93
  
Epoch 300-500:
  - Loss微小波动
  - mAP@0.5: 0.93 → 0.95+
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
# 2. 重启训练,使用更保守配置
batch: 12
workers: 4
```

### 问题3: 训练速度过慢

**症状**: 每个epoch超过120秒

**解决方案**:
```yaml
# 增加workers(如果显存充足)
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
# 训练3次,得到3个best.pt
# 推理时集成预测(投票或平均)
# 通常提升mAP 2-4%
```

#### 方案C: 更大模型(需降batch)
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
# 精度: mAP@50 ≈ 0.93-0.96(可能略低)
```

### 混合训练策略
```
Step 1: 快速实验 (100 epochs)  → 验证数据
Step 2: 均衡模式 (300 epochs)  → 获得可用模型
Step 3: 高精度模式 (800 epochs) → 最终优化
```

### 类别不平衡处理
如果某些类别(如fallen)数量较少:
```yaml
# 针对性增强
copy_paste: 0.25          # 增加复制粘贴概率
degrees: 20.0             # 增加旋转多样性
```

### 迁移学习策略
如果有预训练模型:
```yaml
# 使用之前训练的最佳模型作为起点
model: runs/detect/xxx/weights/best.pt
lr0: 0.0005               # 降低学习率
warmup_epochs: 5          # 缩短热身期
```

---

## 📜 历史配置参考 (YOLO11m)

> ⚠️ **注意**: 本节为历史配置参考,建议使用上方的最新预设参数。

### YOLO11m 核心配置

```yaml
# 模型选择
model: yolo11m.pt                    # Medium模型 - 精度与安全的最佳平衡

# 训练时长
epochs: 500                          # 扩展训练周期,追求最高精度
patience: 80                         # 早停耐心值,允许充分探索最优解

# 批次与尺寸
batch: 16                            # 保守批量大小,确保显存安全
imgsz: 640                           # 固定图像尺寸(禁用多尺度)

# 优化器配置
optimizer: AdamW                     # 长周期训练最佳优化器
lr0: 0.001                           # 初始学习率(AdamW默认)
lrf: 0.01                            # 最终学习率比例 → 0.00001
warmup_epochs: 10                    # 预热轮数,稳定启动
momentum: 0.937                      # 动量参数
weight_decay: 0.0005                 # L2正则化系数

# 学习率调度
scheduler: cosine                    # 余弦退火调度

# 数据加载
workers: 6                           # 数据加载线程数(减少内存占用)

# 损失函数
fl_gamma: 1.0                        # Focal Loss gamma值(温和类别平衡)
label_smoothing: 0.05                # 标签平滑(保留标签信息)
iou: 0.7                             # IoU阈值

# 数据增强
degrees: 10.0                        # 旋转角度 ±10°(保守)
translate: 0.1                       # 平移增强 10%
scale: 0.5                           # 缩放范围 0.5-1.5x
flip_lr: 0.5                         # 水平翻转概率 50%
multi_scale: false                   # ❌ 禁用多尺度(显存安全)
copy_paste: 0.15                     # Copy-Paste增强概率 15%(保守)
close_mosaic: 20                     # 最后20轮关闭Mosaic(精细调优)
```

### 显存安全保障

**三重保险机制**:

1. **保守的 Batch Size**: `batch=16`(而非32或24)
   - 效果: 降低显存占用约33%
   - 优势: 梯度稳定性好,留有充足余量

2. **禁用 Multi-scale**: `multi_scale=false`
   - 原因: 消除动态尺寸导致的显存峰值
   - 效果: 显存占用可预测且稳定

3. **减少 Workers**: `workers=6`(而非8)
   - 效果: 减少数据加载器内存占用约0.5GB
   - 优势: 仍能保证数据加载效率

**显存占用分析**:

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

**安全余量**:
```
GPU总显存:     16.0 GB
系统保留:      -1.0 GB  (Windows + 驱动)
训练占用:      -14.0 GB (峰值)
─────────────────────────────
安全余量:       1.0 GB  ✅
```

### YOLO11m 预期性能

| 指标 | 预期范围 | 说明 |
|------|---------|------|
| **mAP@50** | 0.94-0.97 | 优秀级别 ⭐⭐⭐⭐⭐ |
| **mAP@50-95** | 0.72-0.78 | 优秀级别 ⭐⭐⭐⭐⭐ |
| **Precision** | > 0.92 | 高精确率 |
| **Recall** | > 0.90 | 高召回率 |
| **训练时长** | 9.7小时 | RTX 5070 Ti + batch=16 |
| **模型文件大小** | ~40 MB | 适中大小 |

**Inference Performance**:
```
Inference Speed: 30-40 FPS (RTX 5070 Ti)
Latency: ~25-33ms/frame
Application: Real-time object detection (acceptable 1-2 second delay)
```

---

## 📝 训练记录模板

建议每次训练后记录:

```markdown
### 训练记录

**日期**: ___________
**预设**: 高精度 / 均衡 / 快速实验 / 大模型
**实际参数**:
- epochs: ___
- batch: ___
- lr0: ___
- 数据增强: ___

**结果**:
- 最佳Epoch: ___
- mAP@0.5: ___
- mAP@0.5:0.95: ___
- Precision: ___
- Recall: ___

**各类别表现**:
- standing: mAP=___
- sitting: mAP=___
- squatting: mAP=___
- fallen: mAP=___

**问题与改进**:
_________________________________
```

---

## 📞 常见问题 FAQ

### Q: 应该选择哪个预设?
A: 
- 日常训练 → **均衡模式**(6-8小时,90-93% mAP)
- 最终部署 → **高精度模式**(15-20小时,92-95% mAP)
- 快速验证 → **快速实验**(1-2小时,85-88% mAP)
- yolo11n不够 → **大模型高精度**(25-35小时,95-98% mAP)

### Q: 训练时显存不足怎么办?
A: 依次尝试:
1. 降低 batch size: 32 → 24 → 16
2. 减少 workers: 8 → 6 → 4
3. 禁用数据增强: copy_paste → 0
4. 详见:[显存监控与调整](#1-显存监控)

### Q: 如何修改训练参数?
A: 
1. 打开 [project.yaml](../project.yaml)
2. 找到 `training:` 部分
3. 替换为预设配置(见上方各模式配置)
4. 保存后重启训练界面

### Q: 训练日志在哪里查看?
A: 
- 实时日志:训练界面 Console 面板
- 历史日志:`runs/training_logs/`
- 模型权重:`runs/detect/`

### Q: 为什么不用更大的batch size?
A: batch=32会导致显存占用超过16GB,有OOM风险。batch=16在保证精度的同时确保安全。

### Q: 为什么要训练500/800轮这么多?
A: 长周期训练配合AdamW优化器和Cosine调度,可以让模型充分收敛到全局最优解。patience机制会自动在合适时机停止。

### Q: 为什么不启用multi-scale?
A: multi-scale会导致显存占用不可预测,可能出现突发峰值导致OOM。固定640尺寸更安全,且长周期训练可以弥补尺度鲁棒性的不足。

### Q: 训练完成后如何使用模型?
A: 
```python
from ultralytics import YOLO
model = YOLO('runs/detect/runs/xxx/weights/best.pt')
results = model.predict(source='test.jpg', conf=0.25)
```

---

## 📂 项目结构

```
项目根目录/
├── project.yaml                       ← 当前激活配置
├── PRESETS.yaml                       ← 所有预设参数模板
├── datasets/
│   ├── images/
│   │   ├── train/                     ← 训练图片
│   │   └── val/                       ← 验证图片
│   ├── labels/
│   │   ├── train/                     ← 训练标注
│   │   └── val/                       ← 验证标注
│   └── data.yaml                      ← 数据集配置
├── docs/
│   ├── README.md                      ← 本文档
│   ├── TRAINING_ANALYSIS_yolo11n.md   ← yolo11n深度分析(历史)
│   └── TRAINING_CONFIG.md             ← yolo11m历史配置(参考)
├── models/                            ← 预训练模型
├── output/                            ← 推理输出
├── runs/
│   ├── detect/                        ← 训练输出
│   └── training_logs/                 ← 训练日志
└── scripts/
    ├── main.py                        ← 主程序入口
    └── tabs/
        ├── train.py                   ← 训练模块
        ├── predict.py                 ← 推理模块
        └── label.py                   ← 标注模块
```

---

## 📈 Project Information

- **Project**: YOLO Training Platform (Universal Object Detection)
- **Current Task**: Fall Detection (standing, sitting, squatting, fallen)
- **Device**: RTX 5070 Ti (16GB VRAM)
- **Framework**: Ultralytics YOLO11
- **Dataset**: ~10000 training images, 1259 validation images

---

## 🔄 文档更新记录

| 日期 | 更新内容 | 版本 |
|------|----------|------|
| 2026-05-19 | 合并4个文档为统一指南,添加最新功能更新 | v2.0 |
| 2026-05-19 | 添加Predict视频24FPS播放、Label负样本导出功能 | v1.2 |
| 2026-05-19 | 修复Label导航保存bug、支持0样本导出 | v1.1 |
| 2026-05-16 | 创建文档索引和导航 | v1.0 |
| 2026-05-16 | 添加三组预设参数文档 | v1.0 |

---

## 🎯 快速参考卡片

```
┌─────────────────────────────────────────────────────┐
│  日常训练用哪个?    → 均衡模式 (6-8h, 90-93%)     │
│  最终部署用哪个?    → 高精度模式 (15-20h, 92-95%)  │
│  快速验证用哪个?    → 快速实验 (1-2h, 85-88%)      │
│  需要更高精度?      → 大模型模式 (25-35h, 95-98%) │
│  显存不足怎么办?    → batch: 32→24→16             │
│  过拟合怎么办?      → label_smoothing: 0.05→0.1   │
│  欠拟合怎么办?      → copy_paste: 0.1→0.2         │
│  视频播放太快?      → 已固定24FPS稳定播放          │
│  负样本怎么导出?    → Label页支持0样本导出         │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 开始训练

1. 选择适合的预设模式
2. 修改 `project.yaml` 中的 `training` 配置
3. 启动训练:`python scripts/main.py`
4. 监控训练过程(参考上方监控要点)
5. 记录训练结果

**祝您训练顺利!** 🎉

---

## 📚 参考资料

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO11 Technical Report](https://github.com/ultralytics/ultralytics)
- [AdamW Optimizer Paper](https://arxiv.org/abs/1711.05101)
- [Cosine Annealing LR Schedule](https://arxiv.org/abs/1608.03983)
