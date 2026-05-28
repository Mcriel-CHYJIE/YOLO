<p align="center">
  <img src="assets/YOLO.png" width="120" alt="YOLO Training Studio" />
</p>

<h1 align="center">YOLO Training Studio</h1>

<p align="center">
  <strong>Desktop GUI for Ultralytics YOLO — Train, Evaluate, Deploy.</strong>
  <br />
  <strong>桌面级 YOLO 训练与管理工具 — 训练、评估、部署一体化。</strong>
  <br />
  WeChat-style sidebar, dark/light theme, real-time monitoring.
  <br />
  微信风格侧边栏，黑暗/明亮主题，实时监控。
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/framework-PyQt5-green" alt="PyQt5" />
  <img src="https://img.shields.io/badge/YOLO-v8%20%7C%20v11-orange" alt="YOLO v8/v11" />
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="MIT License" />
</p>

---

## Overview / 概述

**EN** | YOLO Training Studio is a full-featured desktop application for object detection workflows built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). It provides a complete pipeline from data preparation and annotation to model training, evaluation, distillation, and deployment — all through an intuitive graphical interface.

**中文** | YOLO Training Studio 是一款基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的全功能桌面端目标检测工具。提供从数据准备、标注到模型训练、评估、蒸馏和部署的完整工作流，全部通过直观的图形界面完成。

The application is packaged as a standalone Windows installer for production use, while remaining fully runnable from source for development and customization.

支持打包为独立的 Windows 安装程序用于生产环境，同时保留从源码运行的能力便于开发和定制。

---

## Features / 功能特性

| Module / 模块 | Description / 说明 |
|---|---|
| **Training / 训练** | YOLOv8/v11 training with SE / CBAM / CA attention injection, real-time loss & mAP charts, hyperparameter tuning, auto-NMS. / 支持注意力模块注入，实时损失与 mAP 曲线，超参数调优。 |
| **Predict / 推理** | Image & video inference with live frame detection, FPS monitoring, result export. / 图片与视频推理，实时帧检测，FPS 监控，结果导出。 |
| **Dataset / 数据集** | Per-split preview, class distribution, annotation coverage stats. / 各分片预览，类别分布，标注覆盖率统计。 |
| **Preprocess / 预处理** | Video frame extraction, image resizing, batch preprocessing. / 视频帧提取，图像缩放，批量预处理。 |
| **Label / 标注** | Manual YOLO-format annotation, keyboard shortcuts, auto-labeling assist, dataset export. / 手动 YOLO 格式标注，快捷键，辅助自动标注，数据集导出。 |
| **Distill / 蒸馏** | Teacher-Student knowledge distillation with configurable alpha and feature-map MSE loss. / 师生知识蒸馏，可配置蒸馏权重与特征图 MSE 损失。 |
| **Export / 导出** | ONNX, TensorRT, OpenVINO, NCNN, TFLite, CoreML, TorchScript, EdgeTPU. |
| **Validate / 验证** | mAP metrics, confusion matrix analysis. / mAP 指标与混淆矩阵分析。 |
| **Settings / 设置** | Workspace management, class name editor, theme switch, shortcut config. / 工作区管理，类别名编辑，主题切换，快捷键配置。 |
| **AI Agent / AI 助手** | Integrated LLM assistant with YOLO-domain expertise. / 集成大模型助手，提供 YOLO 领域知识支持。 |
| **Tools / 工具** | GPU benchmark, speed test, integrity check, batch rename. / GPU 基准测试，速度测试，完整性检查，批量重命名。 |

---

## Architecture / 架构

### Module Layout / 模块布局

```
YOLO/
├── run.py                      # Entry point / 入口
├── main/
│   ├── config/                 # Configuration manager / 配置管理
│   │   ├── __init__.py         # cfg singleton + path constants
│   │   ├── project.yaml        # Defaults (单数据源)
│   │   ├── paths.json          # Workspace dirs (runtime)
│   │   ├── attention.json      # Attention module (runtime)
│   │   ├── theme.json          # Theme state (runtime)
│   │   └── shortcuts.json      # Shortcut mappings (runtime)
│   ├── core/
│   │   ├── base.py             # Shared UI components / 共享组件
│   │   ├── train/              # Model training / 训练
│   │   ├── predict/            # Inference engine / 推理
│   │   ├── dataset/            # Dataset browser / 数据集浏览
│   │   ├── preprocess/         # Preprocessing / 预处理
│   │   ├── label/              # Annotation tool / 标注
│   │   ├── distill/            # Knowledge distillation / 蒸馏
│   │   ├── export/             # Format converter / 导出
│   │   ├── settings/           # Configuration / 设置
│   │   ├── guide/              # Walkthrough / 引导
│   │   ├── agent/              # AI assistant / AI 助手
│   │   └── tools/              # Utility suite / 工具集
│   └── project.yaml            # Deployed config
├── assets/                     # Icons, images / 图标、图片
├── pack.py                     # Build script / 打包脚本
├── rthook_torch.py             # PyTorch DLL hook for PyInstaller
├── installer.iss               # Inno Setup definition
├── README.md
└── LICENSE
```

### Design Pattern / 设计模式

Each tab follows a **three-layer separation** (每个标签页采用三层分离):

```
page.py      — UI event wiring, signal/slot connections / 事件绑定
service.py   — Business logic + background worker threads / 业务逻辑 + 后台线程
[name].ui    — Qt Designer static layout / 静态布局
```

Shared infrastructure in `main/core/base.py` includes (共享基础设施): WeChat-style green accent `#07C160`, **MetricCard** (KPI card), **LogPanel** (scrolling log viewer), **Chart** (matplotlib embedded).

---

## Technology Stack / 技术栈

| Category / 类别 | Technology / 技术 |
|---|---|
| **GUI Framework** | PyQt5 |
| **CV / ML Runtime** | Ultralytics YOLO v8/v11, PyTorch 2.x, OpenCV |
| **Inference Backend** | ONNX Runtime, TensorRT, OpenVINO, NCNN, TFLite |
| **Visualization** | matplotlib (training curves), OpenCV (frame rendering) |
| **Configuration** | YAML (defaults), JSON (runtime state) |
| **Packaging** | PyInstaller + Inno Setup |

---

## Getting Started / 快速开始

### Prerequisites / 环境要求

- **Python** 3.10+
- **OS** Windows 10/11 (Linux supported for development / 开发环境支持 Linux)
- **GPU** Recommended: NVIDIA with CUDA 12+ and 8GB+ VRAM (推荐)

### Run from Source / 源码运行

```bash
# Clone
git clone https://github.com/Mcriel-CHYJIE/YOLO.git
cd YOLO

# Install dependencies / 安装依赖
pip install ultralytics PyQt5 torch torchvision opencv-python numpy matplotlib pyyaml psutil scipy

# Launch / 启动
python run.py
```

On first launch, navigate to **Settings → Init** to select a root directory — the required folder structure will be created automatically.
首次启动后进入 **设置 → 初始化** 选择根目录，自动创建所需文件夹结构。

### Build Standalone Installer / 打包安装程序

```bash
# Quick build (PyInstaller only) / 仅打包 EXE
python pack.py

# Full build with Inno Setup installer / 打包安装程序
python pack.py --installer
```

Output: `Output/YOLO_Training_Studio_Setup_v1.exe`

---

## Configuration / 配置

### project.yaml (Defaults / 默认配置)

All defaults are in a single YAML file at `main/project.yaml` — the single source of truth.
所有默认值定义在 `main/project.yaml`，是唯一的默认数据源。

```yaml
project:
  name: YOLO Training Studio
  task: detect
  data_yaml: datasets/data.yaml

training:
  model: yolo11n.pt
  epochs: 500
  batch: 32
  optimizer: AdamW
  lr0: 0.0005
  ...
```

### Workspace Layout / 工作区结构

| Directory / 目录 | Purpose / 用途 |
|---|---|
| `train_output` | Training logs & weights / 训练日志与权重 |
| `predict_output` | Inference results / 推理结果 |
| `dataset_dir` | Dataset images & labels / 数据集图片与标签 |
| `preproc_dir` | Raw footage for frame extraction / 原始视频素材 |
| `label_dir` | Active annotation workspace / 标注工作区 |
| `export_dir` | Exported model files / 导出模型文件 |
| `models_dir` | Pretrained weights cache / 预训练权重缓存 |

---

## Training / 训练

### Standard Workflow / 标准流程

1. **Select model** — Pretrained weight or `.yaml` architecture / 选择预训练权重或架构文件
2. **Configure hyperparameters** — epochs, batch, lr, optimizer, augmentation, NMS / 配置超参数
3. **(Optional) Inject attention** — SE / CBAM / CA via dropdown / 可选注入注意力模块
4. **Start training** — Monitor real-time loss & mAP curves / 开始训练，实时监控曲线
5. **Review results** — Access logs & weights from output directory / 查看结果

### Attention Module Injection / 注意力模块注入

| Module / 模块 | Type / 类型 | Description / 说明 |
|---|---|---|
| **SE** | Channel | Squeeze-and-Excitation — 全局池化 + 全连接层 |
| **CBAM** | Channel + Spatial | 通道与空间注意力串联 |
| **CA** | Coordinate | 坐标注意力，编码位置信息 |

### Knowledge Distillation / 知识蒸馏

Teacher model frozen; Student learns from both labels and teacher feature maps. Adjustable `alpha` weight.
教师模型冻结，学生模型同时从真实标签和教师特征图学习，支持可调蒸馏权重。

### Real-time Monitoring / 实时监控

- **Training curves** — Live loss & mAP charts / 实时损失与 mAP 曲线
- **System monitor** — CPU, memory, disk, GPU, VRAM / 系统监控栏
- **FPS counter** — Inference preview speed / 推理 FPS 计数

---

## Export / 导出

```
ONNX → TensorRT → OpenVINO → NCNN → TFLite → CoreML → TorchScript → EdgeTPU
```

Parameters configurable via Export tab UI (format, input size, half-precision, INT8, NMS inclusion).
通过导出标签页配置格式、输入尺寸、半精度、INT8 量化等参数。

---

## Packaging / 打包

1. **PyInstaller** — Bundles Python + dependencies + resources into a single directory / 打包为单目录
2. **Inno Setup** — Wraps output into installer with shortcut, uninstall, license / 封装为安装程序

### Runtime Hooks / 运行时钩子

- `rthook_torch.py` — Pre-loads VC++ runtime DLLs to prevent `OSError 1114` on c10.dll / 预加载 VC++ 运行时 DLL，防止 c10.dll 加载失败
- Config files stored in `%APPDATA%\YOLO Training Studio\` to survive reinstallation / 配置文件存于 APPDATA，重装不丢失

### System Requirements / 系统要求

- **OS**: Windows 10/11 (64-bit)
- **Disk**: ~4 GB (extracted / 解压后)
- **RAM**: 8 GB minimum, 16 GB+ recommended / 建议 16 GB+
- **GPU**: NVIDIA with CUDA 12+ and 8 GB+ VRAM for training (CPU-only inference works but is slow) / 训练建议，纯 CPU 推理可用但慢

---

## License / 许可证

MIT License — for educational and research purposes only. Commercial redistribution and resale are prohibited.
MIT 许可证 — 仅限学习与研究用途，禁止商业倒卖。

Copyright &copy; 2025 Mcriel-CHYJIE

---

<p align="center">
  <sub>Built with PyQt5 &middot; Ultralytics YOLO &middot; PyTorch</sub>
</p>
