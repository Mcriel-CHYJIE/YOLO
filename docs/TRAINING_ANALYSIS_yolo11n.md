---
tags: [yolo11n, training, analysis, configuration]
---

# 🔬 YOLO11n 训练配置深度分析

> **📚 文档导航**: [返回文档首页](README.md) | [预设指南](PRESETS_GUIDE.md) | [历史配置](TRAINING_CONFIG.md)

## 📋 配置概览

**生成时间**: 2026-05-16  
**目标设备**: RTX 5070 Ti (16GB VRAM)  
**基底模型**: yolo11n.pt  
**输入尺寸**: 640×640  
**优化目标**: 识别率最优化（不考虑训练时间）

---

## 🎯 核心参数配置

### 1. 基础训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **model** | yolo11n.pt | Nano模型，参数量最小，适合长时间精细训练 |
| **epochs** | 800 | 超长训练周期，充分挖掘模型潜力 |
| **batch** | 32 | 针对yolo11n在16GB显存的最优批次大小 |
| **imgsz** | 640 | 固定输入尺寸，保证训练一致性 |
| **device** | GPU | 使用RTX 5070 Ti加速训练 |
| **workers** | 8 | 提升数据加载效率，充分利用CPU多核 |

### 2. 学习率策略

| 参数 | 值 | 设计理由 |
|------|-----|----------|
| **optimizer** | AdamW | 自适应学习率，适合长周期训练，收敛稳定 |
| **lr0** | 0.002 | 稍高初始LR，增强前期探索能力 |
| **lrf** | 0.01 | 最终LR=0.00002，精细微调阶段 |
| **warmup_epochs** | 15 | 较长热身期，避免初期梯度爆炸 |
| **scheduler** | cosine | 余弦退火，平滑过渡到最优解 |
| **patience** | 100 | 延长等待窗口，不错过后期突破 |

**学习率曲线示意**:
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

### 3. 正则化与损失函数

| 参数 | 值 | 作用 |
|------|-----|------|
| **fl_gamma** | 1.5 | 焦点损失强度，强化难例学习 |
| **label_smoothing** | 0.1 | 标签平滑，防止过拟合，提升泛化 |
| **weight_decay** | 0.0005 | L2正则化，控制模型复杂度 |
| **momentum** | 0.937 | 动量项，加速收敛并减少震荡 |
| **iou** | 0.7 | IoU阈值，平衡召回率与精确率 |

### 4. 数据增强策略（激进型）

| 参数 | 值 | 增强效果 |
|------|-----|----------|
| **copy_paste** | 0.2 | 20%概率复制粘贴，增加小目标样本 |
| **degrees** | 15.0° | ±15度旋转，提升姿态鲁棒性 |
| **translate** | 0.15 | ±15%平移，增强位置不变性 |
| **scale** | 0.6 | 尺度范围[0.4, 1.6]x，适应不同距离 |
| **flip_lr** | 0.5 | 50%水平翻转，左右对称增强 |
| **hsv_h** | 0.015 | 色调扰动±1.5% |
| **hsv_s** | 0.7 | 饱和度扰动±70% |
| **hsv_v** | 0.4 | 亮度扰动±40% |
| **multi_scale** | false | 禁用多尺度，保持640一致性 |
| **close_mosaic** | 30 | 最后30轮关闭mosaic，精细调优 |

**数据增强时间线**:
```
Epoch:  0 ────────────── 770 ────── 800
        |                  |          |
        |   Mosaic + All   |  Fine-tuning
        |   Augmentations  |  (No Mosaic)
        |                  |
        Strong Augmentation Phase
```

---

## 💡 设计思路解析

### 为什么选择这些参数？

#### 1. **模型选择：yolo11n而非yolo11m/l/x**
- ✅ **优势**: 
  - 参数量少（~2.6M），训练速度快
  - 显存占用低（预计8-10GB @ batch=32）
  - 更适合长时间精细训练
  - 通过增强数据和长周期弥补容量不足
- ❌ **权衡**: 
  - 理论上限低于大模型
  - 但通过800 epochs + 强增强可接近最优

#### 2. **Batch Size = 32的考量**
- **显存估算**:
  ```
  yolo11n @ 640 input:
  - 模型权重: ~10 MB
  - 激活值: ~3-4 GB (batch=32)
  - 梯度 + 优化器状态: ~2-3 GB
  - 预留空间: ~2 GB
  - 总计: ~8-10 GB < 16 GB ✓
  ```
- **训练稳定性**: batch=32提供足够的梯度统计稳定性
- **GPU利用率**: 充分利用RTX 5070 Ti的计算能力

#### 3. **800 Epochs的必要性**
- **收敛曲线预期**:
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
- **200-600轮**: 稳步提升，学习复杂模式
- **600-800轮**: 微调阶段，挖掘最后1-2%性能

#### 4. **激进数据增强的理由**
- **目标**: 通过数据多样性弥补小模型容量限制
- **Copy-Paste 0.2**: 特别针对摔倒(squatting/fallen)等稀有类别
- **Rotation 15°**: 人体姿态多变，需要旋转鲁棒性
- **Scale 0.6**: 适应不同拍摄距离和人物大小
- **HSV增强**: 应对光照变化（早晚、室内外）

#### 5. **Cosine Scheduler + Long Patience**
- **Cosine退火**: 避免陷入局部最优，周期性"重启"探索
- **Patience=100**: 允许模型在 plateau 期继续寻找突破
- **预期**: 可能在epoch 600-750之间出现二次提升

---

## ⚠️ 潜在风险与监控要点

### 1. 显存监控
```bash
# 训练时实时监控
nvidia-smi -l 1
```
- **警戒线**: >14GB 需降低batch至24
- **正常范围**: 8-12GB
- **异常处理**: 如遇OOM，立即调整 `batch: 24` 或 `workers: 6`

### 2. 过拟合预警信号
| 指标 | 正常 | 警告 | 危险 |
|------|------|------|------|
| train_loss ↓ val_loss ↑ | ✓ | 差距>20% | 差距>50% |
| mAP@0.5 (train-val) | <5% | 5-10% | >10% |
| 连续50轮无提升 | - | ✓ | 考虑early stop |

**应对策略**:
- 若出现过拟合：增大 `label_smoothing` 至 0.15，减小 `copy_paste` 至 0.1
- 若欠拟合：增大 `fl_gamma` 至 2.0，延长训练至 1000 epochs

### 3. 训练停滞检测
- **正常现象**: epoch 400-600 可能进入平台期
- **异常停滞**: 连续100轮 mAP 波动 <0.5%
- **解决**: 检查学习率是否过低，必要时手动调整 lr0

---

## 📊 预期训练结果

### 性能预估（基于yolo11n特性）

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
| **standing** | 低 | 0.92-0.95 | 常见姿态，易识别 |
| **sitting** | 中 | 0.88-0.92 | 与standing有相似性 |
| **squatting** | 中高 | 0.82-0.88 | 可能与fallen混淆 |
| **fallen** | 高 | 0.80-0.86 | 稀有类别，依赖增强 |

---

## 🔧 训练启动命令

### 方式1: GUI界面（推荐）
```bash
python scripts/main.py
```
1. 切换到 **Training** 标签页
2. 确认参数已加载（Model: yolo11n, Epochs: 800, Batch: 32）
3. 点击 **Start** 开始训练

### 方式2: 命令行直接训练
```bash
python scripts/tabs/train.py --model yolo11n.pt --epochs 800 --batch 32 --imgsz 640 --device 0
```

### 方式3: Python脚本

```python
from ultralytics import YOLO

model = YOLO('../yolo11n.pt')
results = model.train(
    data='datasets/data.yaml',
    epochs=800,
    batch=32,
    imgsz=640,
    device=0,
    optimizer='AdamW',
    lr0=0.002,
    lrf=0.01,
    warmup_epochs=15,
    patience=100,
    cos_lr=True,
    workers=8,
    # 数据增强
    degrees=15.0,
    translate=0.15,
    scale=0.6,
    copy_paste=0.2,
    flip_lr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    # 损失函数
    fl_gamma=1.5,
    label_smoothing=0.1,
    close_mosaic=30,
    # 其他
    verbose=True,
    project='runs/detect',
    name='yolo11n_optimized_800ep'
)
```

---

## 📈 训练过程监控清单

### 每日检查点

#### Epoch 1-50: 初期验证
- [ ] 训练是否正常启动，无CUDA错误
- [ ] Loss是否在下降（允许波动）
- [ ] 显存占用是否稳定在8-12GB
- [ ] GPU利用率是否>80%

#### Epoch 50-200: 快速成长期
- [ ] mAP@0.5 是否达到 0.60+
- [ ] Train/Val loss 差距是否<30%
- [ ] 各类别召回率是否均衡

#### Epoch 200-500: 稳定提升期
- [ ] mAP@0.5 是否达到 0.75+
- [ ] 验证集曲线是否平滑上升
- [ ] 是否出现明显过拟合迹象

#### Epoch 500-700: 平台期观察
- [ ] 接受mAP增长放缓（正常现象）
- [ ] 关注是否有小幅波动中的上升趋势
- [ ] 记录当前最佳模型的epoch

#### Epoch 700-800: 最终冲刺
- [ ] 最后30轮是否关闭mosaic（日志确认）
- [ ] 最佳模型是否在后期更新
- [ ] 保存最终模型和训练历史

### 关键日志关键词
```bash
# 正常训练
✅ "Epoch" / "loss" / "mAP"
✅ "Saving weights to runs/detect/..."
✅ "Validation" / "Results saved to"

# 警告信号
⚠️ "CUDA out of memory" → 降低batch
⚠️ "NaN detected" → 检查学习率
⚠️ "EarlyStopping" → patience触发

# 错误信号
❌ "AssertionError" → 数据标注问题
❌ "RuntimeError" → 环境/依赖问题
```

---

## 🎓 后续优化建议

### 如果训练完成后效果不理想

#### 方案A: 进一步提升精度
1. **切换至yolo11s.pt**（参数量翻倍）
   - 保持其他参数不变
   - batch降至24以保显存安全
   - 预期提升: mAP +2-4%

2. **引入知识蒸馏**
   - 用当前训练的yolo11n作为教师模型
   - 蒸馏到新的yolo11n学生模型
   - 参考 `scripts/tabs/distill.py`

3. **测试时增强(TTA)**
   - 推理时使用多尺度+翻转
   - 无需重新训练
   - 预期提升: mAP +1-2%

#### 方案B: 加速训练（如需迭代）
1. **混合精度训练**
   ```yaml
   amp: true  # 启用FP16
   batch: 48  # 可提升至48
   ```
   - 训练速度提升30-50%
   - 显存占用降低40%

2. **减少epochs至500**
   - 大部分收益在前500轮获得
   - 节省40%训练时间

#### 方案C: 针对性调优
1. **类别不平衡处理**
   - 检查 `datasets/labels/` 中各类别数量
   - 对稀少类别（如fallen）增加采样权重
   ```python
   # 在data.yaml中添加
   train_weights: [1.0, 1.0, 1.5, 2.0]  # standing,sitting,squatting,fallen
   ```

2. **困难样本挖掘**
   - 分析验证集误检案例
   - 针对性收集类似场景数据
   - 重新标注后加入训练集

---

## 📝 训练记录模板

建议在每次训练后填写：

```markdown
### 训练记录 #1

**日期**: 2026-05-16  
**配置**: yolo11n, 800ep, batch=32, imgsz=640  
**实际运行时间**: ___ 小时  
**最佳Epoch**: ___  
**最终指标**:
- mAP@0.5: ___
- mAP@0.5:0.95: ___
- Precision: ___
- Recall: ___

**各类别表现**:
- standing: mAP=___
- sitting: mAP=___
- squatting: mAP=___
- fallen: mAP=___

**遇到的问题**:
- [ ] 显存溢出
- [ ] 训练停滞
- [ ] 过拟合
- [ ] 其他: _________

**改进方向**:
_________________________________
```

---

## 🔗 相关资源

- **Ultralytics官方文档**: https://docs.ultralytics.com/
- **YOLOv11技术报告**: https://github.com/ultralytics/ultralytics
- **本项目配置**: `project.yaml` (training section)
- **训练历史**: `runs/training_logs/`
- **模型输出**: `runs/detect/yolo11n_optimized_800ep/`

---

## ✨ 总结

本配置针对 **RTX 5070 Ti (16GB)** 设备进行了深度优化：

1. **安全性**: batch=32 + multi_scale=false 确保显存不溢出
2. **准确性**: 800 epochs + 激进增强最大化模型潜力
3. **稳定性**: AdamW + Cosine + Long patience 保证收敛质量
4. **泛化性**: 强数据增强 + Label smoothing 防止过拟合

**预期成果**: 在yolo11n的限制下，达到该模型的理论与实际最优性能边界。

祝训练顺利！🚀
