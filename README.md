# YOLO Training Studio

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的通用目标检测训练平台，提供桌面级 GUI 操作界面。

## 功能

| 模块 | 功能 |
|---|---|
| **Training** | 模型训练，支持 YOLOv8/v11，注意力模块注入（SE/CBAM/CA），实时 loss/mAP 图表 |
| **Predict** | 图片/视频推理，实时帧检测，FPS 监控，结果导出 |
| **Dataset** | 数据集管理与预览 |
| **Preprocess** | 视频帧提取、图片预处理 |
| **Label** | 手动标注（YOLO 格式），自动标注辅助，数据集导出 |
| **Distill** | 知识蒸馏训练（Teacher-Student），自定义蒸馏损失 |
| **Validate** | 模型验证，mAP 指标评估 |
| **Export** | 导出为 ONNX / TensorRT / OpenVINO / TFLite 等格式 |
| **Settings** | 类别名称编辑、快捷键配置、黑白主题切换、工作目录配置、项目结构初始化 |
| **AI Agent** | AI 助手面板 |

## 环境要求

- Python 3.10+
- Windows / Linux（建议 NVIDIA GPU，CUDA 12+）
- 推荐：RTX 30/40/50 系列 + 16GB+ 显存

### 核心依赖

```
ultralytics>=8.0.0
PyQt5
torch>=2.0.0
opencv-python
numpy
matplotlib
pyyaml
```

## 快速开始

### 1. 安装依赖

```bash
pip install ultralytics PyQt5 torch opencv-python numpy matplotlib pyyaml
```

或使用 Conda：

```bash
conda install -c pytorch pytorch torchvision
conda install pyqt opencv matplotlib pyyaml
pip install ultralytics
```

### 2. 启动

```bash
python run.py
```

首次启动会提示配置工作目录。在 **Settings → Init** 选择项目根目录即可自动创建目录结构。

## 项目结构

```
YOLO/
├── run.py              # 启动入口，自动添加根目录到 sys.path
├── main/
│   ├── config/         # 配置模块（包）
│   │   ├── __init__.py # cfg 单例 + 路径常量 + load_paths()
│   │   ├── project.yaml# 项目配置文件（唯一数据源）
│   │   ├── paths.json  # 工作目录配置（运行时读写）
│   │   ├── attention.json # 注意力模块选择（运行时读写）
│   │   ├── theme.json  # 主题状态（运行时生成）
│   │   └── shortcuts.json # 快捷键映射（运行时生成）
│   ├── core/
│   │   ├── base.py     # 共享组件（颜色常量、MetricCard、LogPanel、Chart 等）
│   │   ├── main.py     # 程序入口（Studio 主窗口）
│   │   ├── train/      # 训练页
│   │   │   ├── page.py     # UI 事件
│   │   │   ├── service.py  # 配置构建 + Trainer 工作线程
│   │   │   └── attention.py # 注意力模块（SE/CBAM/CA）
│   │   ├── predict/    # 推理页
│   │   ├── dataset/    # 数据集页
│   │   ├── preprocess/ # 预处理页
│   │   ├── label/      # 标注页
│   │   ├── distill/    # 蒸馏页
│   │   ├── validate/   # 验证页
│   │   ├── export/     # 导出页
│   │   ├── settings/   # 设置页（目录配置、快捷键、主题）
│   │   ├── guide/      # 使用引导页
│   │   └── agent/      # AI 助手页
│   └── project.yaml    # 项目配置
├── assets/             # 图标资源
├── README.md
└── LICENSE
```

### 模块说明

每个功能页遵循三层分离架构：

```
page.py     — UI 事件处理、信号连接、动态控件构建
service.py  — 业务逻辑函数 + 后台工作线程（Trainer/Distiller/Detector）
[name].ui   — Qt Designer 静态布局
```

共享模块：

```
main/config/__init__.py  — cfg 单例：读取 project.yaml 作为唯一数据源
main/core/base.py        — 颜色常量（WeChat 风格）、MetricCard、LogPanel、Chart
```

## 配置

### project.yaml

所有默认参数通过 `main/project.yaml` 配置：

```yaml
project:
  name: YOLO Training Studio
  task: detect
  data_yaml: datasets/config.yaml

training:
  model: yolov8n.pt
  epochs: 300
  ...
```

完整默认值见 `main/config/__init__.py` 中的 `_FALLBACK`。

### 工作目录配置（paths.json）

启动会检查路径是否完整，未配置时弹出提示。可在 **Settings → Directories** 中配置：

| 配置项 | 说明 |
|---|---|
| Training output | 训练日志和权重输出目录 |
| Predict output | 推理结果输出目录 |
| Dataset dir | 数据集目录 |
| Preprocess dir | 预处理原始数据目录 |
| Label dir | 标注数据目录 |
| Export dir | 模型导出目录 |
| Models dir | 预训练权重下载和读取目录 |

点击 **Init Project Structure** 按钮，选择根目录后自动创建目录结构并填充配置。

### JSON 配置文件

| 文件 | 路径 | 说明 |
|---|---|---|
| `paths.json` | `main/config/` | 工作目录配置（运行时读写） |
| `attention.json` | `main/config/` | 注意力模块选择 |
| `theme.json` | `main/config/` | 黑白主题状态 |
| `shortcuts.json` | `main/config/` | 快捷键映射 |

## 训练

1. 在 Training 页选择预训练权重（`yolov8n.pt` / `yolov11n.pt`）或从零训练（`.yaml` 架构）
2. 模型自动从 ultralytics 下载到配置的 `models_dir`
3. 配置超参数：epochs、batch size、learning rate 等
4. 可选注入注意力模块（Settings → Other → Attention 选择）
5. 点击训练，实时监控 loss 和 mAP 图表

### 注意力模块

支持三种注意力机制，通过训练页的 attention 下拉框选择，训练时自动注入到 C2f 模块：

- **SE** — Squeeze-and-Excitation，通道注意力
- **CBAM** — 通道+空间注意力
- **CA** — Coordinate Attention，坐标注意力

### 知识蒸馏

Distill 页支持 Teacher-Student 知识蒸馏：

- Teacher 模型冻结，Student 模型学习
- 可调节蒸馏权重 alpha
- 自定义蒸馏损失（特征图 MSE）

## 窗口自适应

窗口缩放时全局字体按比例自动调整（参考宽度 1400px，缩放范围 0.7～1.5 倍）。

## 许可证

MIT License — 仅限学习交流，禁止商用倒卖。

Copyright (c) 2025 Mcriel-CHYJIE
