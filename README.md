# 🚀 YOLO Training Studio - 通用目标检测训练平台

基于 **YOLO11** 的通用目标检测训练系统，支持自定义数据集和多种姿态/物体检测任务，专为边缘设备部署优化。

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLO](https://img.shields.io/badge/YOLO-11-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

## ✨ 特性

- 🎯 **通用检测框架**: 支持任意自定义类别的目标检测任务
- 🖥️ **集成GUI**: Qt一体化界面，包含训练、验证、预测、数据集管理等9大功能模块
- 📊 **实时监控**: Loss和mAP曲线实时显示，系统资源监控（CPU/GPU/内存）
- 🔄 **智能日志**: 进度条去重，彩色输出，训练历史自动保存
- 💾 **自动保存**: 训练状态自动同步到JSON文件，支持断点续训
- 🚀 **边缘优化**: 支持ONNX、NCNN、TensorRT等多种部署格式
- 📱 **多平台**: Windows/Linux兼容，支持GPU/CPU训练
- ⚙️ **配置驱动**: 通过project.yaml快速切换不同项目

## 📁 项目结构

```
11_fall/
├── 📄 README.md                    # 项目说明（本文件）
├── 📄 project.yaml                 # ⭐ 项目配置文件（核心）
├── 📄 requirements.txt             # Python依赖
├── 📄 run.bat                      # Windows启动脚本
├── 📄 training_reminder.html       # 训练提醒页面
│
├── 📂 datasets/                    # 数据集
│   ├── data.yaml                   # 数据集配置
│   ├── images/                     # 图片数据
│   │   ├── train/ (9,942张)
│   │   └── val/ (1,159张)
│   └── labels/                     # 标注数据
│       ├── train/ (10,000个)
│       └── val/ (1,159个)
│
├── 📂 scripts/                     # 脚本工具
│   ├── 🎯 main.py                  # ⭐ 主程序入口（集成GUI）
│   └── tabs/                       # GUI功能模块
│       ├── train.py                # 训练模块
│       ├── val.py                  # 验证模块
│       ├── predict.py              # 预测模块
│       ├── dataset.py              # 数据集管理
│       ├── preprocess.py           # 数据预处理
│       ├── label.py                # 标注工具
│       ├── distill.py              # 知识蒸馏
│       ├── export.py               # 模型导出
│       └── guide.py                # 使用指南
│
├── 📂 models/                      # 预训练模型
│   ├── best5.11.pt
│   ├── best5.12.pt
│   └── best5.15.pt
│
├── 📂 runs/                        # 运行输出
│   ├── detect/runs/                # 训练结果
│   │   ├── 11_fall_0509_1611/
│   │   ├── 11_fall_0513_1802/
│   │   ├── 11_fall_0514_1303/
│   │   └── 11_fall_0515_1706/
│   └── training_logs/              # 训练日志
│
└── 📦 yolo*.pt                     # YOLO官方预训练模型
```

## 📊 示例项目：人体姿态检测

> 💡 本项目以人体姿态检测为例展示系统使用方法，您可以轻松替换为其他检测任务（如火灾检测、交通标志、工业缺陷等）

### 类别定义（示例）

| ID | 类别 | 英文 | 说明 |
|----|------|------|------|
| 0 | standing | 站立 | 正常站立姿态 |
| 1 | sitting | 坐姿 | 坐着的状态 |
| 2 | squatting | 蹲姿 | 蹲下或半蹲姿态 |
| 3 | fallen | 摔倒 | 摔倒或躺卧状态 |

### 数据统计（示例）

| 数据集 | 图片数量 | 标注数量 |
|--------|---------|----------|
| 训练集 | 9,942 | 10,000 |
| 验证集 | 1,159 | 1,159 |
| **总计** | **11,101** | **11,159** |

### 数据特点（示例）

- ✅ 高质量标注
- ✅ 多样化场景
- ✅ 类别平衡良好
- ✅ YOLO标准格式

---

## 🔧 如何创建新项目

### 步骤1: 准备数据集

按照YOLO格式组织您的数据集：
```
your_project/
├── datasets/
│   ├── data.yaml          # 数据集配置
│   ├── images/
│   │   ├── train/         # 训练图片
│   │   └── val/           # 验证图片
│   └── labels/
│       ├── train/         # 训练标注
│       └── val/           # 验证标注
```

### 步骤2: 配置project.yaml

复制并修改 `project.yaml` 文件：
```yaml
project:
  name: Your Project Name    # 项目名称
  task: detect               # 任务类型
  classes: ['class1', 'class2', ...]  # 您的类别
  names:
    0: class1                # 类别0
    1: class2                # 类别1
  data_yaml: datasets/data.yaml  # 数据集配置文件路径
```

### 步骤3: 启动训练

```bash
python scripts/main.py
```

在GUI中配置训练参数并开始训练即可！

## 🚀 快速开始

### 1️⃣ 环境安装

```bash
# 克隆或下载项目
cd 11_fall

# 创建虚拟环境（推荐）
conda create -n fall_detection python=3.10
conda activate fall_detection

# 安装依赖
pip install -r requirements.txt

# GPU版本（如需）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2️⃣ 启动集成GUI

```bash
# 方式1: 使用批处理文件
run.bat

# 方式2: 直接运行主程序
python scripts/main.py
```

启动后将打开集成界面，包含以下模块：
- **Training**: 模型训练，实时监控Loss/mAP曲线
- **Validate**: 模型验证，性能评估
- **Predict**: 图片/视频预测推理
- **Dataset**: 数据集管理与分析
- **Preprocess**: 数据预处理工具
- **Label**: 标注工具
- **Distill**: 知识蒸馏
- **Export**: 模型导出
- **Guide**: 使用指南

### 3️⃣ 配置项目参数

编辑 `project.yaml` 文件配置项目参数：

```yaml
project:
  name: Fall Detection Studio
  task: detect
  classes: ['standing', 'sitting', 'squatting', 'fallen']
  data_yaml: datasets/data.yaml

training:
  model: yolo11n.pt
  epochs: 150
  batch: 32
  imgsz: 640
  optimizer: SGD
  lr0: 0.0003
  patience: 20
```

### 4️⃣ 模型训练

#### 使用GUI（推荐）
1. 运行 `python scripts/main.py`
2. 切换到 "Training" 标签页
3. 配置参数（模型、epochs、batch等）
4. 点击 "Start" 开始训练
5. 实时监控Loss和mAP曲线及系统资源

#### 使用命令行
```bash
# 基础训练（需自行编写训练脚本）
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').train(data='datasets/data.yaml', epochs=150, batch=32)"

# 从断点恢复
python -c "from ultralytics import YOLO; YOLO('runs/detect/runs/11_fall_0515_1706/weights/last.pt').resume()"
```

### 5️⃣ 模型验证

在GUI中：
1. 切换到 "Validate" 标签页
2. 选择权重文件（如 `models/best5.15.pt`）
3. 点击 "Run"
4. 查看性能指标和混淆矩阵

或使用Python代码：
```python
from ultralytics import YOLO
model = YOLO('models/best5.15.pt')
metrics = model.val(data='datasets/data.yaml')
print(f'mAP@0.5: {metrics.box.map50:.3f}')
print(f'mAP@0.5:0.95: {metrics.box.map:.3f}')
```

### 6️⃣ 模型推理

在GUI中：
1. 切换到 "Predict" 标签页
2. 选择权重文件和输入源（图片/视频/摄像头）
3. 调整置信度阈值
4. 点击 "Run" 开始推理

或使用Python代码：
```python
from ultralytics import YOLO
model = YOLO('models/best5.15.pt')

# 单张图片
results = model.predict('test.jpg', conf=0.25)
results[0].show()

# 视频文件
results = model.predict('video.mp4', conf=0.25, save=True)

# 摄像头实时检测
results = model.predict(0, conf=0.25, show=True)
```

### 7️⃣ 模型导出

在GUI中：
1. 切换到 "Export" 标签页
2. 选择权重文件和导出格式
3. 配置导出参数（imgsz、量化等）
4. 点击 "Export"

或使用Python代码：
```python
from ultralytics import YOLO
model = YOLO('models/best5.15.pt')

# 导出为ONNX（通用格式）
model.export(format='onnx', imgsz=640)

# 导出为NCNN（ARM设备推荐）
model.export(format='ncnn', imgsz=640)

# INT8量化（更小更快）
model.export(format='onnx', imgsz=640, int8=True)

# TensorRT（NVIDIA GPU）
model.export(format='engine', imgsz=640, half=True)
```

## 🖥️ GUI功能展示

### 集成界面架构

```
┌─────────────────────────────────────────────────────────────┐
│ ◈ YOLO Training Studio  |  Your Project  |  GPU: RTX...    │
├─────────────────────────────────────────────────────────────┤
│ [Training] [Validate] [Predict] [Dataset] [Preprocess]      │
│ [Label] [Distill] [Export] [Guide]                          │
├──────────────┬──────────────────────┬──────────────────────┤
│ Configuration│                      │ ● Console            │
│ Model: [...] │   Loss Chart         │ [08:28:33] 🚀 ...    │
│ Epochs: 150  │                      │ [08:28:34] Epoch 1.. │
│ Batch: 32    ├──────────────────────┤                      │
│ ImgSz: 640   │                      │                      │
│              │   mAP Chart          │                      │
│ Optimizer:.. │                      │                      │
│ Device: GPU  │                      │                      │
├──────────────┤                      │                      │
│ Algorithm    │                      │                      │
│ Focal γ: 2.0 │                      │                      │
│ Smoothing:0.2│                      │                      │
├──────────────┴──────────────────────┴──────────────────────┤
│ [Start] [Stop]  ████████░░░░ 60%                           │
│ Epoch: 90  |  mAP@0.5: 0.85  |  Best: 0.87               │
└─────────────────────────────────────────────────────────────┘
```

### 功能模块说明

| 模块 | 功能 | 主要特性 |
|------|------|----------|
| **Training** | 模型训练 | 实时曲线、系统监控、断点续训 |
| **Validate** | 模型验证 | 性能评估、混淆矩阵、指标分析 |
| **Predict** | 推理预测 | 图片/视频/摄像头、结果可视化 |
| **Dataset** | 数据管理 | 统计分析、可视化预览、数据检查 |
| **Preprocess** | 数据预处理 | 图像增强、格式转换、数据清洗 |
| **Label** | 标注工具 | YOLO格式标注、批量处理、标注检查 |
| **Distill** | 知识蒸馏 | 教师-学生模型、模型压缩 |
| **Export** | 模型导出 | 多格式支持、量化优化 |
| **Guide** | 使用指南 | 快速入门、常见问题、最佳实践 |

**核心特性**：
- 📊 实时Loss和mAP曲线
- 🎨 彩色日志输出
- 🔄 进度条智能去重
- 📈 指标卡片显示
- 💾 自动保存训练历史到JSON
- 🔍 系统资源监控（CPU/GPU/内存/显存）

### 验证界面

- 权重文件选择
- 性能指标表格（mAP@0.5, mAP@0.5:0.95, Precision, Recall）
- 混淆矩阵可视化
- 各类别详细分析（per-class metrics）
- PR曲线和F1曲线

## 📱 部署方案

### 支持的导出格式

| 格式 | 框架 | 速度 | 适用场景 |
|------|------|------|----------|
| **ONNX** | ONNX Runtime | ⭐⭐⭐⭐ | 通用跨平台，PC/服务器 |
| **NCNN** | ncnn | ⭐⭐⭐⭐ | ARM设备（Orange Pi/Raspberry Pi） |
| **TensorRT** | TensorRT | ⭐⭐⭐⭐⭐ | NVIDIA GPU（Jetson系列） |
| **OpenVINO** | OpenVINO | ⭐⭐⭐⭐ | Intel CPU/VPU |
| **TFLite** | TensorFlow Lite | ⭐⭐⭐ | 移动设备/嵌入式 |
| **CoreML** | CoreML | ⭐⭐⭐⭐ | Apple设备（iOS/macOS） |
| **TorchScript** | PyTorch | ⭐⭐⭐ | PyTorch环境部署 |

### Orange Pi 部署流程

1. **导出模型**
   ```python
   from ultralytics import YOLO
   model = YOLO('models/best5.15.pt')
   model.export(format='ncnn', imgsz=640, half=True)
   ```

2. **传输到设备**
   ```bash
   scp runs/export/*.param orangepi@192.168.1.100:~/
   scp runs/export/*.bin orangepi@192.168.1.100:~/
   ```

3. **在Orange Pi上运行**
   ```bash
   # 安装ncnn
   sudo apt install libncnn-dev
   
   # 编译并运行推理程序
   g++ -o detect detect.cpp -lncnn -lopencv_core -lopencv_imgproc
   ./detect model.param model.bin input.jpg
   ```

### 性能优化建议

| 优化项 | 说明 | 效果 |
|--------|------|------|
| **输入尺寸** | 320x320 vs 640x640 | 速度提升2倍 |
| **INT8量化** | FP32 → INT8 | 体积缩小4倍，速度提升2-3倍 |
| **模型选择** | YOLO11n vs YOLO11m | n模型速度快3倍 |
| **多线程** | 设置workers | CPU利用率提升 |

## 🛠️ 功能模块详解

### 训练模块 (Training Tab)

**配置项**：
- 模型选择：yolo11n/s/m/l/x
- 训练参数：epochs, batch size, image size
- 优化器：SGD/Adam/AdamW
- 学习率调度：Cosine/Linear
- 数据增强：Mosaic, Mixup, Copy-Paste, Rotation等

**监控功能**：
- 实时Loss曲线（box loss, cls loss, dfl loss）
- 实时mAP曲线（mAP@0.5, mAP@0.5:0.95）
- 系统资源监控（CPU, GPU, 内存, 显存）
- 训练进度和预计剩余时间
- 最佳模型自动保存

### 验证模块 (Validate Tab)

**功能**：
- 模型性能评估
- 混淆矩阵生成
- PR曲线和F1曲线
- 各类别详细指标
- 推理速度测试

### 预测模块 (Predict Tab)

**支持输入**：
- 单张图片
- 图片文件夹
- 视频文件
- 摄像头实时流

**输出选项**：
- 可视化结果展示
- 保存标注图片
- 导出检测结果

### 数据集模块 (Dataset Tab)

**功能**：
- 数据集统计（类别分布、图片数量）
- 标注可视化预览
- 数据质量检查
- 训练/验证集划分

### 预处理模块 (Preprocess Tab)

**工具**：
- 图像尺寸调整
- 格式转换（JPG↔PNG）
- 数据增强预览
- 异常数据清理

### 标注模块 (Label Tab)

**功能**：
- YOLO格式标注编辑器
- 批量标注处理
- 标注质量检查
- 标注可视化

### 蒸馏模块 (Distill Tab)

**功能**：
- 教师-学生模型蒸馏
- 模型压缩与加速
- 知识迁移训练

### 导出模块 (Export Tab)

**支持格式**：
- ONNX, NCNN, TensorRT
- OpenVINO, TFLite, CoreML
- TorchScript, EdgeTPU

**优化选项**：
- FP16半精度量化
- INT8整数量化
- NMS融合

### 指南模块 (Guide Tab)

**内容**：
- 快速入门教程
- 常见问题解答
- 最佳实践建议
- 故障排除指南

## 📚 配置说明

### project.yaml - 项目配置文件（核心）

这是系统的核心配置文件，通过修改此文件可以快速切换到不同的检测任务。

#### 当前示例配置（人体姿态检测）

```yaml
# 项目身份
project:
  name: Fall Detection Studio    # 窗口标题
  task: detect                   # YOLO任务类型 (detect/segment/classify)
  classes: ['standing', 'sitting', 'squatting', 'fallen']  # 类别列表
  names:                         # 类别定义
    0: standing                  # 站立
    1: sitting                   # 坐姿
    2: squatting                 # 蹲姿
    3: fallen                    # 摔倒/躺卧
  data_yaml: datasets/data.yaml  # 数据集配置文件路径
  tip: "🦶 Balanced → LS+SIoU+Rot15"  # 算法提示

# 训练默认参数
training:
  model: yolo11n.pt              # 默认模型
  model_options:                 # 可选模型列表
    - yolo11n.pt
    - yolo11s.pt
    - yolo11m.pt
    - yolo11l.pt
    - yolo11x.pt
  epochs: 150                    # 训练轮数
  batch: 32                      # 批次大小
  imgsz: 640                     # 图像尺寸
  optimizer: SGD                 # 优化器
  lr0: 0.0003                    # 初始学习率
  patience: 20                   # 早停耐心值
  fl_gamma: 2.0                  # Focal Loss参数
  label_smoothing: 0.2           # 标签平滑
  
# 验证默认参数
validation:
  conf: 0.25                     # 置信度阈值
  iou: 0.45                      # IoU阈值
  
# 导出默认参数
export:
  format: onnx                   # 导出格式
  imgsz: 640                     # 导出图像尺寸
  half: false                    # FP16量化
  int8: false                    # INT8量化
```

#### 创建新项目的配置示例

**示例1: 火灾检测**
```yaml
project:
  name: Fire Detection Studio
  task: detect
  classes: ['fire', 'smoke']
  names:
    0: fire
    1: smoke
  data_yaml: datasets/fire_data.yaml
```

**示例2: 交通标志检测**
```yaml
project:
  name: Traffic Sign Detection
  task: detect
  classes: ['stop', 'speed_limit', 'no_entry', 'parking']
  names:
    0: stop
    1: speed_limit
    2: no_entry
    3: parking
  data_yaml: datasets/traffic_data.yaml
```

**示例3: 工业缺陷检测**
```yaml
project:
  name: Industrial Defect Detection
  task: detect
  classes: ['scratch', 'dent', 'crack', 'discoloration']
  names:
    0: scratch
    1: dent
    2: crack
    3: discoloration
  data_yaml: datasets/defect_data.yaml
```

修改 `project.yaml` 后重启GUI即可应用新配置，无需修改代码！

## 💡 训练技巧

### 数据准备

1. **检查数据质量**
   - 在GUI中使用 "Dataset" 模块预览数据
   - 使用 "Preprocess" 模块清理异常数据
   - 确保标注文件格式正确（YOLO格式）

2. **分析数据分布**
   - 查看各类别样本数量是否平衡
   - 检查训练集和验证集比例（建议 8:2 或 9:1）
   - 确认图片质量和标注准确性

### 训练调优

| 场景 | 建议配置 |
|------|----------|
| **小数据集** (< 5K) | epochs: 300-500, batch: 8-16, lr0: 0.0001, patience: 50 |
| **中等数据集** (5K-20K) | epochs: 150-300, batch: 16-32, lr0: 0.0003, patience: 20 |
| **大数据集** (> 20K) | epochs: 100-200, batch: 32-64, lr0: 0.001, patience: 10 |

### 当前示例项目配置（已优化）

本示例项目针对11K+图片的四分类姿态检测任务，采用以下优化配置：
- **优化器**: SGD（比Adam泛化性更好）
- **学习率**: 0.0003（较低的学习率，稳定训练）
- **标签平滑**: 0.2（防止过拟合）
- **Focal Loss**: γ=2.0（处理类别不平衡）
- **数据增强**: Mosaic + Copy-Paste + Rotation 30°
- **早停耐心值**: 20 epochs

> 💡 这些参数可根据您的具体任务调整，建议在project.yaml中修改

### 常见问题解决

#### ❌ 显存不足 (OOM)
```yaml
解决方案:
- 减小 batch size (32 → 16 → 8)
- 减小 imgsz (640 → 512 → 416)
- 使用更小的模型 (m → s → n)
- 设置 workers=0 或减少workers数量
```

#### ❌ 训练不收敛
```yaml
解决方案:
- 降低学习率 (lr0: 0.001 → 0.0001)
- 增加 warmup_epochs (3 → 10)
- 检查数据标注质量
- 增加训练轮数
- 切换优化器 (Adam → SGD)
```

#### ❌ 过拟合
```yaml
解决方案:
- 增强数据增强 (mixup, copy_paste, rotation)
- 增加 label_smoothing (0.1 → 0.2)
- 增加 weight_decay
- 使用早停 (patience=20)
- 减少模型复杂度 (m → s → n)
```

#### ❌ 训练速度慢
```yaml
解决方案:
- 增加 batch size（如果显存允许）
- 增加 workers (CPU模式，建议设为CPU核心数的一半)
- 使用GPU训练而非CPU
- 启用混合精度训练 (amp=True，默认开启)
- 使用更快的存储（SSD）
```

### 最佳实践

1. **训练前**
   - ✅ 使用Dataset模块检查数据集质量和分布
   - ✅ 预览标注确保正确性
   - ✅ 根据硬件选择合适的模型大小（n/s/m/l/x）
   - ✅ 配置合理的batch size（不超过显存限制）
   - ✅ 备份原始数据和配置文件

2. **训练中**
   - ✅ 监控Loss和mAP曲线是否正常下降/上升
   - ✅ 观察控制台日志是否有警告或错误
   - ✅ 不要随意中断训练过程
   - ✅ 关注系统资源使用情况
   - ✅ 定期检查保存的checkpoint

3. **训练后**
   - ✅ 使用Validate模块验证模型性能
   - ✅ 分析混淆矩阵找出薄弱类别
   - ✅ 在Predict模块测试实际效果
   - ✅ 导出适合部署的格式（ONNX/NCNN等）
   - ✅ 备份最佳模型和训练日志

## 📊 训练历史

### 最近的训练记录（示例项目）

| 训练ID | 日期 | 状态 | 备注 |
|--------|------|------|------|
| 11_fall_0515_1706 | 2026-05-15 | ✅ 完成 | 最新训练 |
| 11_fall_0514_1303 | 2026-05-14 | ✅ 完成 | - |
| 11_fall_0513_1802 | 2026-05-13 | ✅ 完成 | - |
| 11_fall_0509_1611 | 2026-05-09 | ✅ 完成 | - |

> 💡 每次新项目的训练记录会保存在各自的目录中

### 训练日志位置

所有训练记录保存在 `runs/training_logs/`：

```json
{
  "timestamp": "2026-05-15 17:06:00",
  "status": "success",
  "config": {
    "model": "yolo11n.pt",
    "epochs": 150,
    "batch": 32,
    "imgsz": 640
  },
  "results": {
    "best_mAP50": 0.87,
    "best_mAP50_95": 0.65,
    "epochs_trained": 150
  },
  "history": {...}
}
```

### 训练结果位置

每次训练的完整结果保存在 `runs/detect/runs/<训练ID>/`：
- `weights/best.pt` - 最佳模型权重
- `weights/last.pt` - 最后epoch的权重
- `results.csv` - 训练指标CSV文件
- `args.yaml` - 训练参数配置
- `*.jpg` - 训练批次可视化和结果图

## 🔐 注意事项

1. ⚠️ 训练时确保有足够的磁盘空间（至少20GB，用于保存checkpoints和日志）
2. ⚠️ GPU训练需要安装CUDA和cuDNN（建议使用PyTorch自带版本）
3. ⚠️ 大batch size需要更多显存，根据显卡调整（RTX 3060建议batch≤32）
4. ⚠️ 定期备份重要的模型权重（models/目录）
5. ⚠️ 验证时使用与训练相同的图像尺寸
6. ⚠️ 不要随意修改datasets/data.yaml中的路径配置
7. ⚠️ 修改project.yaml后需要重启GUI才能生效
8. ⚠️ 训练过程中避免强制关闭程序，使用Stop按钮正常停止
9. ⚠️ 导出模型前先验证模型性能达标
10. ⚠️ 部署到边缘设备时注意模型格式兼容性

## 📞 技术支持

遇到问题请：
1. 查看 `runs/training_logs/` 中的训练日志
2. 检查GUI控制台的完整错误信息
3. 参考 [Ultralytics官方文档](https://docs.ultralytics.com/)
4. 检查数据集格式是否符合YOLO标准
5. 确认project.yaml配置是否正确
6. 查看Guide模块中的常见问题解答

### 常见资源

- 📘 [Ultralytics YOLO文档](https://docs.ultralytics.com/)
- 🎓 [YOLO训练教程](https://docs.ultralytics.com/guides/train-custom-dataset/)
- 🔧 [模型导出指南](https://docs.ultralytics.com/modes/export/)
- 📊 [评估指标说明](https://docs.ultralytics.com/guides/yolo-performance-metrics/)

## 📄 License

MIT License

---

<div align="center">

**Made with ❤️ for Universal Object Detection**

一个通用的YOLO训练平台，适用于任何目标检测任务

如有问题，请查阅文档或提交Issue

</div>
