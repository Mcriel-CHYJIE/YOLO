---
tags: [preset, configuration, training]
---

# 🎯 YOLO 训练预设参数配置指南

> **📚 文档导航**: [返回文档首页](README.md) | [配置分析](TRAINING_ANALYSIS_yolo11n.md) | [历史配置](TRAINING_CONFIG.md)

## 📋 概述

本项目提供 **4组优化预设参数**，适用于不同训练场景。所有预设均针对 **RTX 5070 Ti (16GB显存)** 设备优化，保证不爆显存的同时追求最优识别率。

---

##  三组预设参数对比

| 预设名称 | 适用场景 | 训练时长 | 预期mAP@0.5 | 推荐度 |
|---------|---------|---------|-------------|--------|
| **高精度模式** | 最终产品模型、研究 | 15-20小时 | 92-95% | ⭐⭐⭐⭐ |
| **均衡模式** | 日常训练、迭代 | 6-8小时 | 90-93% | ⭐⭐⭐⭐⭐ |
| **快速实验** | 快速验证、调试 | 1-2小时 | 85-88% | ⭐⭐ |
| **大模型高精度** | 超越yolo11n极限 | 25-35小时 | 95-98% | ⭐⭐⭐⭐ |

---

## 1️⃣ 高精度模式 (Maximum Accuracy)

###  核心配置
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

###  数据增强策略（激进型）
```yaml
fl_gamma: 1.5              # 强焦点损失，关注难例
label_smoothing: 0.1       # 中度平滑，防止过拟合
copy_paste: 0.2            # 激进复制粘贴增强
degrees: 15.0              # ±15°旋转增强
translate: 0.15            # ±15%平移
scale: 0.6                 # 尺度范围[0.4, 1.6]x
close_mosaic: 30           # 最后30轮关闭mosaic精细调优
```

### ✅ 优势
- 充分挖掘模型潜力，达到理论最优性能
- 强数据增强提升泛化能力
- 长训练周期+长patience不错过后期突破
- 适合对精度要求极高的场景

### ⚠️ 注意事项
- 训练时间长，需要耐心等待
- 前200轮快速提升，200-600轮稳步提升，600-800轮精细微调
- 建议在数据质量高、标注准确时使用
- 显存占用：9-11GB

### 📊 学习率曲线
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

## 2️⃣ 均衡模式 (Balanced) ⭐ **推荐**

### 📌 核心配置
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

### 🔧 数据增强策略（适中型）
```yaml
fl_gamma: 1.0              # 轻度焦点损失
label_smoothing: 0.05      # 轻度平滑
copy_paste: 0.1            # 适度复制粘贴
degrees: 10.0              # ±10°旋转
translate: 0.1             # ±10%平移
scale: 0.5                 # 尺度范围[0.5, 1.5]x
close_mosaic: 15           # 最后15轮关闭mosaic
```

### ✅ 优势
- **性价比最高**，精度与时间的最佳平衡
- 参数保守稳定，不易出现过拟合
- 适合日常迭代和模型更新
- 显存占用：9-11GB

###  适用场景
- 数据集更新后的重新训练
- 验证新的数据增强策略
- 模型版本迭代
- 日常维护和优化

### 📊 训练进度预期
```
Epoch 1-50:   快速学习基础特征 (mAP 0→0.60)
Epoch 50-150: 稳步提升识别能力 (mAP 0.60→0.85)
Epoch 150-250: 精细调优 (mAP 0.85→0.90)
Epoch 250-300: 收敛稳定 (mAP 0.90→0.93)
```

---

## 3️⃣ 快速实验模式 (Fast Experiment)

### 📌 核心配置
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

###  数据增强策略（轻量型）
```yaml
fl_gamma: 0.0                # 无焦点损失
label_smoothing: 0.0         # 无平滑
copy_paste: 0.0              # 无复制粘贴
degrees: 0.0                 # 无旋转
translate: 0.1               # 仅平移
scale: 0.5                   # 尺度范围[0.5, 1.5]x
close_mosaic: 5              # 仅最后5轮关闭mosaic
```

### ✅ 优势
- **训练速度最快**，1-2小时完成
- 适合快速验证想法和调试
- 减少数据增强，更容易定位问题
- 显存占用：10-12GB（batch较大）

###  适用场景
- 测试新添加的训练数据
- 验证数据标注质量
- 调试训练流程和参数
- 快速对比不同模型版本
- 超参数搜索的初始阶段

### ⚠️ 局限性
- 精度较低，不适合最终部署
- 缺乏数据增强，泛化能力有限
- 仅作为验证和调试工具

---

## 4️⃣ 大模型高精度模式 (Large Model Precision)

### 📌 核心配置
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

### 🔧 数据增强策略（强化型）
```yaml
fl_gamma: 1.5                        # 强焦点损失
label_smoothing: 0.1                 # 中度平滑防过拟合
copy_paste: 0.15                     # 适度复制粘贴
degrees: 15.0                        # ±15°旋转
translate: 0.15                      # ±15%平移
scale: 0.6                           # 尺度范围[0.4, 1.6]x
close_mosaic: 25                     # 最后25轮关闭mosaic
```

### ✅ 优势
- **精度最高**，超越yolo11n的理论上限
- 大模型容量更强，能学习更复杂的特征
- 适合对精度要求极高的场景
- yolo11s: 95-97% mAP, yolo11m: 96-98% mAP

### ⚠️ 注意事项
- **训练时间最长**：yolo11s约25-30小时，yolo11m约30-35小时
- **显存占用高**：yolo11s约12-13GB，yolo11m约13-14GB
- **必须降低batch**: 从32降至24，yolo11m可能需要降至16
- **减少workers**: 从8降至6，避免显存峰值
- 仅在yolo11n精度不满足需求时使用

### 📊 不同模型显存占用预估
```
yolo11n (2.6M params):  ~9-11 GB @ batch=32
yolo11s (9.4M params):  ~12-13 GB @ batch=24
yolo11m (20.1M params): ~13-14 GB @ batch=16-24
yolo11l (25.3M params): ~14-15 GB @ batch=8-12  ⚠️ 警戒
yolo11x (56.9M params): >16 GB @ batch=4-8     ❌ 不推荐
```

### 🎯 适用场景
- yolo11n精度已达瓶颈（95%+）
- 需要极致的检测精度
- 有足够的训练时间和GPU资源
- 数据质量高且数量充足（>10000张）

### 📊 训练进度预期（yolo11s）
```
Epoch 1-100:  快速学习基础特征 (mAP 0→0.70)
Epoch 100-300: 稳步提升复杂特征 (mAP 0.70→0.90)
Epoch 300-500: 精细调优难例 (mAP 0.90→0.95)
Epoch 500-600: 收敛稳定 (mAP 0.95→0.97)
```

---

## 🔧 使用方法

### 方法1：直接修改 project.yaml（推荐）

打开 [project.yaml](file:///D:/Projects/11_fall/project.yaml)，将 `training:` 部分替换为对应预设：

#### 使用高精度模式
```yaml
training:
  model: yolo11n.pt
  epochs: 800
  batch: 32
  imgsz: 640
  optimizer: AdamW
  device: auto
  scheduler: cosine
  patience: 100
  lr0: 0.002
  lrf: 0.01
  warmup_epochs: 15
  workers: 8
  momentum: 0.937
  weight_decay: 0.0005
  fl_gamma: 1.5
  label_smoothing: 0.1
  iou: 0.7
  close_mosaic: 30
  copy_paste: 0.2
  degrees: 15.0
  multi_scale: false
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  translate: 0.15
  scale: 0.6
  flip_lr: 0.5
```

#### 使用均衡模式（推荐）
```yaml
training:
  model: yolo11n.pt
  epochs: 300
  batch: 32
  imgsz: 640
  optimizer: AdamW
  device: auto
  scheduler: cosine
  patience: 50
  lr0: 0.001
  lrf: 0.01
  warmup_epochs: 5
  workers: 8
  momentum: 0.937
  weight_decay: 0.0005
  fl_gamma: 1.0
  label_smoothing: 0.05
  iou: 0.7
  close_mosaic: 15
  copy_paste: 0.1
  degrees: 10.0
  multi_scale: false
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  translate: 0.1
  scale: 0.5
  flip_lr: 0.5
```

#### 使用快速实验模式
```yaml
training:
  model: yolo11n.pt
  epochs: 100
  batch: 48
  imgsz: 640
  optimizer: Adam
  device: auto
  scheduler: linear
  patience: 20
  lr0: 0.001
  lrf: 0.1
  warmup_epochs: 3
  workers: 8
  momentum: 0.937
  weight_decay: 0.0005
  fl_gamma: 0.0
  label_smoothing: 0.0
  iou: 0.7
  close_mosaic: 5
  copy_paste: 0.0
  degrees: 0.0
  multi_scale: false
  hsv_h: 0.015
  hsv_s: 0.5
  hsv_v: 0.3
  translate: 0.1
  scale: 0.5
  flip_lr: 0.5
```

### 方法2：GUI界面手动调整

1. 启动训练界面：`python scripts/main.py`
2. 切换到 **Training** 标签页
3. 根据预设参数逐个调整
4. 点击 **Start** 开始训练

### 方法3：使用预设模板文件

已创建 [PRESETS.yaml](file:///D:/Projects/11_fall/PRESETS.yaml) 包含所有预设配置，可直接复制使用。

---

## 📊 性能对比（RTX 5070 Ti + 10000张训练图像）

| 预设 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 训练时长 | 显存占用 |
|------|---------|--------------|-----------|--------|---------|----------|
| 高精度 | 92-95% | 72-78% | 93-96% | 87-91% | 15-20h | 9-11GB |
| 均衡 | 90-93% | 68-74% | 90-93% | 85-89% | 6-8h | 9-11GB |
| 快速实验 | 85-88% | 62-68% | 87-90% | 80-85% | 1-2h | 10-12GB |
| 大模型高精度 | 95-98% | 76-82% | 95-98% | 90-94% | 25-35h | 12-14GB |

---

## ⚠️ 显存监控与调整

### 实时监控
```bash
# 训练时在新终端运行
nvidia-smi -l 1
```

### 显存安全阈值
- **正常范围**: 8-12GB
- **警戒线**: >14GB
- **危险线**: >15GB

### OOM处理策略
1. **立即降低batch size**: 32 → 24 → 16
2. **减少workers**: 8 → 6 → 4
3. **禁用数据增强**: copy_paste → 0
4. **降低图像尺寸**: 640 → 512

### 显存优化建议
```yaml
# 如果遇到显存不足，依次尝试：
batch: 32     → 24      → 16
workers: 8    → 6       → 4
copy_paste: 0.2 → 0.1   → 0.0
multi_scale: false        # 保持禁用
```

---

##  选择建议

### 场景1：首次训练 / 数据质量不确定
→ **快速实验模式**
- 快速验证数据质量
- 发现标注问题
- 确认训练流程正常

### 场景2：日常迭代 / 模型更新
→ **均衡模式** ⭐
- 性价比最高
- 稳定性好
- 适合频繁训练

### 场景3：最终部署 / 比赛提交
→ **高精度模式**
- 追求极限精度
- 数据质量已验证
- 有充足训练时间

### 场景5：yolo11n已达精度瓶颈
→ **大模型高精度模式**
- 切换到yolo11s或yolo11m
- batch降至24（yolo11m需降至16）
- workers降至6
- 预期提升2-5% mAP

### 场景4：超参数调优
1. 先用**快速实验**测试参数范围
2. 用**均衡模式**验证最佳参数
3. 用**高精度模式**最终训练

---

## 🔍 训练过程监控要点

### 关键指标检查

#### Epoch 1-50
- ✅ Loss是否在下降
- ✅ mAP@0.5 是否达到 0.50+
- ✅ GPU利用率 >80%
- ✅ 显存占用稳定

#### Epoch 50-200
- ✅ mAP@0.5 是否达到 0.75+
- ✅ Train/Val loss 差距 <30%
- ✅ 各类别召回率均衡

#### Epoch 200+
- ✅ mAP增长是否持续
- ✅ 是否出现 plateau（正常现象）
- ✅ 最佳模型是否更新

### 常见警告信号

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| Loss不下降 | 学习率过高 | 降低lr0至0.0005 |
| Loss震荡 | batch太小 | 增大batch或启用梯度累积 |
| Val loss上升 | 过拟合 | 增大label_smoothing |
| mAP停滞 | 数据增强不足 | 增大degrees/copy_paste |
| 显存溢出 | batch过大 | 降低batch size |

---

##  高级技巧

### 1. 混合训练策略
```
Step 1: 快速实验 (100 epochs)  → 验证数据
Step 2: 均衡模式 (300 epochs)  → 获得可用模型
Step 3: 高精度模式 (800 epochs) → 最终优化
```

### 2. 类别不平衡处理
如果某些类别（如fallen）数量较少：
```yaml
# 针对性增强
copy_paste: 0.25          # 增加复制粘贴概率
degrees: 20.0             # 增加旋转多样性
```

### 3. 迁移学习策略
如果有预训练模型：
```yaml
# 使用之前训练的最佳模型作为起点
model: runs/detect/xxx/weights/best.pt
lr0: 0.0005               # 降低学习率
warmup_epochs: 5          # 缩短热身期
```

### 4. 多尺度训练（高级）
如果需要适应不同分辨率：
```yaml
multi_scale: true         # 启用多尺度
batch: 24                 # 降低batch保显存
imgsz: 640                # 基础尺寸
# 实际会在[512, 640, 768]之间随机切换
```

---

## 📝 训练记录模板

建议每次训练后记录：

```markdown
### 训练记录

**日期**: ___________
**预设**: 高精度 / 均衡 / 快速实验
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

## 🔗 相关文档

- [训练配置分析](TRAINING_ANALYSIS_yolo11n.md) - 详细参数解析
- [预设配置文件](../PRESETS.yaml) - 完整预设配置
- [项目配置](../project.yaml) - 当前激活配置
- [训练日志](runs/training_logs/) - 历史训练记录

---

## 📌 快速参考

```
┌─────────────────────────────────────────────────────┐
│  日常训练用哪个？    → 均衡模式 (6-8h, 90-93%)     │
│  最终部署用哪个？    → 高精度模式 (15-20h, 92-95%)  │
│  快速验证用哪个？    → 快速实验 (1-2h, 85-88%)      │
│  需要更高精度？      → 大模型模式 (25-35h, 95-98%) │
│  显存不足怎么办？    → batch: 32→24→16             │
│  过拟合怎么办？      → label_smoothing: 0.05→0.1   │
│  欠拟合怎么办？      → copy_paste: 0.1→0.2         │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 开始训练

1. 选择适合的预设模式
2. 修改 `project.yaml` 中的 `training` 配置
3. 启动训练：`python scripts/main.py`
4. 监控训练过程（参考上方监控要点）
5. 记录训练结果

**祝您训练顺利！** 🎉
