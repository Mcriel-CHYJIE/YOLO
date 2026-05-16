---
tags: [synthetic-data, blender, pipeline, augmentation]
---

# 🎨 合成数据生成管线 — 室内摔倒检测

> **📚 文档导航**: [返回文档首页](README.md) | [预设指南](PRESETS_GUIDE.md) | [配置分析](TRAINING_ANALYSIS_yolo11n.md)

## 概述

用 Blender + Mixamo 生成 3D 人体渲染图，合成到真实室内背景图片，补充摔倒检测训练集。

**目的**：增加 fallen 类的场景多样性（不同房间、不同摔倒方向、不同视角、部分遮挡等场景），提升模型在真实监控场景下的泛化能力。

**管线流程**：

```
装环境 → 生成人模 → 下载动画 → Blender 批量渲人 → Python 合背景 → 训练/验证
```

---

## 一、环境准备

### 1.1 安装 Blender（Windows）

1. 去 [blender.org](https://www.blender.org/download/) 下载最新版（推荐 4.x）
2. 安装到 `D:\Blender`（或默认路径）
3. 确认命令行可用，在 CMD/PowerShell 执行：
   ```
   "D:\Blender\blender.exe" --version
   ```
4. 确保能调用 `blender -b`（后台模式），无需启动 GUI

### 1.2 安装 MB-Lab 插件

MB-Lab 是一个免费开源的人体建模插件，用于生成不同体型的人。

1. 下载 MB-Lab: https://github.com/animate1978/MB-Lab/releases
2. 安装到 Blender：
   - 打开 Blender → Edit → Preferences → Add-ons → Install...
   - 选择下载的 ZIP 文件
   - 勾选启用
3. 或者直接解压到 `D:\Blender\version\scripts\addons\` 目录

### 1.3 准备 Python 依赖

合成阶段需要 OpenCV 和 NumPy：

```bash
pip install opencv-python numpy pillow
```

---

## 二、生成人体模型

### 2.1 MB-Lab 导出人模

1. 打开 Blender，进入 MB-Lab 面板（N 键调出侧栏）
2. 选择体型参数：身高、体重、肌肉量等（做 3-5 种不同体型）
3. 选择性别和基本体态
4. 选择衣服类型（宽松/普通/紧身）
5. 导出为 FBX（File → Export → FBX）
   - 路径：`D:\Projects\11_fall\synthetic_data\humans\male_avg.fbx`
   - 勾选 "Selected Objects" 和 "Armature"

**建议制作 3-5 个不同人模**：
- `male_avg.fbx`（普通成年男性）
- `male_thin.fbx`（偏瘦男性）
- `female_avg.fbx`（普通成年女性）
- `female_heavy.fbx`（偏胖女性）
- `teen.fbx`（青少年体型）

---

## 三、下载 Mixamo 动画

### 3.1 操作步骤

1. 打开 [mixamo.com](https://www.mixamo.com)
2. 点击 "Upload Character" → 上传你的 MB-Lab FBX 人模
3. 系统自动绑定骨骼（Rigging 阶段等待 ~10 秒）
4. 下载动画：
   - 在搜索栏输入动作关键词
   - 选中结果 → 调整动画参数（一般不调）→ Download
   - 格式：**FBX Binary**，帧率：**30 FPS**
   - 下载后放入 `D:\Projects\11_fall\synthetic_data\animations\`

### 3.2 建议下载的动画

| 搜索关键词 | 动画数 | 说明 | 推荐帧数 |
|-----------|--------|------|---------|
| `fall forward` | 5-8 种 | 向前摔倒 | 每次渲染 10 帧 |
| `fall backward` | 3-5 种 | 向后摔倒 | 每次渲染 10 帧 |
| `fall side` | 3-5 种 | 侧向摔倒 | 每次渲染 10 帧 |
| `fall down` | 10+ 种 | 通用摔倒（跪倒/趴倒/晕倒） | 每次渲染 10 帧 |
| `lying` / `laying` | 5+ 种 | 躺在地上各种姿势 | 每次渲染 5 帧 |
| `sit down` | 10+ 种 | 坐到椅子上/沙发/地面 | 每次渲染 8 帧 |
| `squat` | 5+ 种 | 蹲下 | 每次渲染 5 帧 |
| `standing` / `idle` | 20+ 种 | 站立/闲逛 | 每次渲染 3 帧 |
| `get up` | 5+ 种 | 从地上爬起来 | 每次渲染 8 帧 |

**目标**：下载约 15-20 个动画文件，每个动画覆盖不同摔倒类型。

---

## 四、批量渲染管线（脚本自动化）

### 4.1 脚本逻辑

`scripts/render_humans.py` — 在 Blender 中批量渲染人模到透明背景 PNG

```
渲染脚本做的事：
1. 清除场景
2. 导入人模 FBX
3. 加载 Mixamo 动画（按帧播放）
4. 设置相机位置（等轴测或透视角度）
5. 设置光照（HDRi 或区域光）
6. 渲染当前帧为 RGBA PNG（透明背景）
7. 保存 bbox 标注（使用 Blender API 算 2D 投影框）
8. 换下一帧 → 重复
```

### 4.2 调用方式

```bash
# 后台批量渲染
"D:\Blender\blender.exe" -b -P scripts/render_humans.py -- \
    --human synthetic_data/humans/male_avg.fbx \
    --anim-dir synthetic_data/animations/ \
    --output synthetic_data/renders/ \
    --count 200 \
    --camera-angles 4 \
    --renderer eevee
```

### 4.3 渲染参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 图像尺寸 | 640×640 | 与训练集一致 |
| 渲染器 | Eevee | GPU 实时渲染，快 |
| 透明背景 | 开启 RGBA | 便于合成 |
| 抗锯齿 | 4x | 平衡质量与速度 |
| 相机视角 | 3-4 个 | 正面/侧面/俯拍/斜上 |

### 4.4 输出结构

```
synthetic_data/renders/
├── images/           ← RGBA PNG 透明底人
│   ├── render_0000.png
│   ├── render_0001.png
│   └── ...
└── labels/           ← YOLO 格式标注（归一化坐标）
    ├── render_0000.txt
    ├── render_0001.txt
    └── ...
```

---

## 五、背景合成（Python 脚本）

### 5.1 脚本逻辑

`scripts/composite_to_background.py` — 把渲染好的透明人贴到真实背景图

```
合成脚本做的事：
1. 加载渲染好的 RGBA PNG（人）
2. 随机选一张室内背景图（真实照片）
3. 随机缩放人（模拟近远距离）
4. 随机放置到背景中的合理位置
5. Alpha 混合合成
6. 根据放置位置重新算 bbox
7. 保存最终图 + YOLO 标注
```

### 5.2 背景图来源

| 来源 | 数量 | 路径 |
|------|------|------|
| 监控视频帧（用户已有数据） | 大量 | `D:\Dataset\检测\` 的视频文件 |
| 监控视频抽帧脚本 | 任意 | `scripts/extract_videos.py` |
| 网络下载室内图（可选） | 按需 | 手动收集 |

### 5.3 调用方式

```bash
python scripts/composite_to_background.py \
    --render-dir synthetic_data/renders/ \
    --bg-dir D:/Dataset/室内背景/ \
    --output synthetic_data/composited/ \
    --count 2000
```

### 5.4 输出结构

```

synthetic_data/composited/
├── images/           ← 合成图（JPG）
│   ├── comp_0000.jpg
│   ├── comp_0001.jpg
│   └── ...
├── labels/           ← YOLO 格式标注
│   ├── comp_0000.txt
│   ├── comp_0001.txt
│   └── ...
└── visualize/         ← 带标注框的预览图（可选）
```

---

## 六、域随机化策略（关键）

域随机化是合成数据泛化到真实场景的核心。参数范围越大，模型越不会过拟合到渲染风格。

### 6.1 人体域随机化

| 参数 | 范围 | 说明 |
|------|------|------|
| 人模体型 | 3-5 种 | 胖/瘦/标准/高/矮 |
| 衣服颜色 | 每种人模 5-8 种 | MB-Lab 材质调色 |
| 动画帧 | 每个动画抽 5-15 个关键帧 | 覆盖摔倒全过程 |
| 人体旋转 | 0-360° 水平旋转 | 面对不同方向倒地 |
| 人体缩放 | 0.3-0.8（相对背景） | 不同距离的视觉大小 |

### 6.2 相机域随机化

| 参数 | 范围 | 说明 |
|------|------|------|
| 俯仰角 | -30° 到 +15° | 模拟高/中/低监控位 |
| 水平角 | 0-360° | 绕 Y 轴旋转 |
| 距离 | 3-10 米 | 人近/远不同比例 |

### 6.3 光照域随机化

| 参数 | 范围 | 说明 |
|------|------|------|
| 主光强度 | 0.5-3.0 | 模拟暗/正常/亮 |
| 主光色温 | 3000K-6500K | 暖光到冷白光 |
| 环境光强度 | 0.1-0.5 | 补光 |
| HDRi 换图 | 3-5 种 | 不同室内环境贴图 |

### 6.4 图像后处理（抹平"CG感"）

合成图出 Blender 后，在 `composite_to_background.py` 中添加：

| 处理 | 参数范围 | 目的 |
|------|---------|------|
| 随机高斯模糊 | 0-2px | 模拟镜头对焦 |
| 随机 JPEG 压缩 | 质量 65-95% | 模拟编码失真 |
| 随机亮度偏移 | ±15% | 模拟测光差异 |
| 随机对比度偏移 | ±15% | 模拟不同画风 |
| 随机色调偏移 | ±5% | 模拟白平衡差异 |
| 人边缘羽化 | 1-2px | 消除"CG锐边" |

---

## 七、训练策略

### 7.1 混合训练方法

合成数据不是直接替换真实数据，而是**混合增强**。

```python
# 训练时 data.yaml 指向混合数据集
data.yaml
  -> train:  # 真实数据 + 合成数据混在一起
      - synthetic_data/composited/images/
      - datasets/images/train/
  -> val: datasets/images/val/   # 只用真实数据验证
```

### 7.2 推荐合成比例

| 合成图占比 | 效果 | 风险 |
|-----------|------|------|
| < 15% | 安全，几乎无副作用 | 提升可能不明显 |
| **15-30%** | **推荐——场景多样性显著提升** | 需要做域随机化对抗 |
| > 40% | 可能压过真实数据 | 模型可能学上 CG 特征 |

**推荐首批合成 1500-2000 张**，占总训练集 ~15%。

### 7.3 验证方法

```bash
# 先跑一个 20 轮的小训练对比
# baseline: 只用真实数据
python scripts/train.py --epochs 20  # 基准

# test: 真实 + 合成数据
# 手动合并 data.yaml，改 train 路径
python scripts/train.py --epochs 20  # 测试
```

比较两个模型的 val mAP50：
- **合成数据有效**：fallen 类的 mAP 提升 > 2-3%
- **无效/负效果**：全部类 mAP 下降 → 检查域随机化是否足够

---

## 八、完整脚本结构（待实现）

```
scripts/
├── render_humans.py           # Blender 批量渲染人（透明底）
├── composite_to_background.py  # 合成到真实背景
├── synthetic_dataset_stats.py  # 合成数据统计
└── synthetic_data_pipeline.md  # 本文档

synthetic_data/
├── humans/                     # MB-Lab 人模 FBX
│   ├── male_avg.fbx
│   ├── male_thin.fbx
│   └── female_avg.fbx
├── animations/                 # Mixamo 下载的动画 FBX
│   ├── fall_forward.fbx
│   ├── fall_backward.fbx
│   ├── fall_side.fbx
│   ├── sit_down.fbx
│   └── ...
├── renders/                    # 渲染出的透明底 PNG
│   ├── images/
│   └── labels/
├── backgrounds/               # 真实室内背景照片
│   └── ...
└── composited/                 # 合成后的最终训练数据
    ├── images/
    └── labels/
```

---

## 九、分步执行计划

### Phase 1：环境搭建（~1 小时）
| # | 任务 | 预计时间 |
|---|------|---------|
| 1 | 装 Blender 4.x | 10 min |
| 2 | 装 MB-Lab 插件 | 5 min |
| 3 | 用 MB-Lab 生成 3 种人模并导出 FBX | 20 min |
| 4 | Mixamo 上传人模，下载 10-15 个动画 | 15 min |

### Phase 2：脚本开发（我已写或将写）
| # | 任务 | 说明 |
|---|------|------|
| 5 | `render_humans.py` | Blender 批量渲人脚本 |
| 6 | `composite_to_background.py` | 背景合成脚本 |
| 7 | 准备真实背景图 | 从监控视频抽帧或收集网图 |

### Phase 3：首批生成（可过夜跑）
| # | 任务 | 说明 |
|---|------|------|
| 8 | 渲染 2000 张透明人 | blender -b -P render_humans.py（约 10 分钟） |
| 9 | 合成到背景 → 2000 张图 | python composite_to_background.py（约 5 分钟） |
| 10 | 检查合成图质量 | 目测 50 张确认质量 |

### Phase 4：训练验证（~1 天）
| # | 任务 | 说明 |
|---|------|------|
| 11 | 合并 data.yaml | 加入合成数据路径 |
| 12 | 跑 20 轮对比训练 | baseline vs with-synthetic |
| 13 | 对比 mAP 判断有效性 | fallen 类涨点则继续 |

---

## 十、检查清单

- [ ] Blender 4.x 安装完成，`blender -b` 可用
- [ ] MB-Lab 插件安装并成功导出至少 1 个人模 FBX
- [ ] Mixamo 下载至少 10 个不同动画
- [ ] `render_humans.py` 脚本就绪
- [ ] Eevee 渲染器测试通过（单帧透明 PNG 导出）
- [ ] `composite_to_background.py` 脚本就绪
- [ ] 准备 100+ 张真实室内背景图
- [ ] 首批 500 张合成图目测无明显问题
- [ ] 合成数据 + 真实数据混合训练 20 轮验证有效
