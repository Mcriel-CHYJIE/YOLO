# 🚀 Fall Detection YOLO - 跌倒检测系统

基于 **YOLO11** 的跌倒检测模型，专为边缘设备部署优化。

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLO](https://img.shields.io/badge/YOLO-11-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

## ✨ 特性

- 🎯 **高精度检测**: 基于YOLO11架构，准确识别跌倒行为
- 🖥️ **图形界面**: Qt GUI工具，可视化训练和监控
- 📊 **实时监控**: Loss和mAP曲线实时显示
- 🔄 **智能日志**: 进度条去重，彩色输出
- 💾 **自动保存**: 训练历史自动记录
- 🚀 **边缘优化**: 支持ONNX、NCNN等部署格式
- 📱 **多平台**: Windows/Linux兼容

## 📁 项目结构

```
11_fall/
├── 📄 README.md                    # 项目说明（本文件）
├──  PROJECT_STRUCTURE.md         # 详细项目架构
── 📄 CONFIG_GUIDE.md              # 配置指南
├── 📄 requirements.txt             # Python依赖
├── 📄 run.bat                      # Windows启动脚本
├──  training_reminder.html       # 训练提醒页面
├── 📄 project.yaml                 # ⭐ GUI系统配置
│
├── 📂 configs/                     # 配置文件（参考）
│   └── fall_detection.yaml         # 命令行训练配置示例
│
├──  datasets/                    # 数据集
│   ├── data.yaml                   # 数据集配置
│   ├── images/                     # 图片数据
│   │   ├── train/ (9,870张)
│   │   └── val/ (2,064张)
│   └── labels/                     # 标注数据
│       ├── train/ (10,000个)
│       └── val/ (2,064个)
│
├── 📂 scripts/                     # 脚本工具
│   ├── 🎯 train_qt.py             # ⭐ Qt训练GUI（主程序）
│   ├── train.py                    # 命令行训练
│   ├── val.py                      # 模型验证
│   ├── predict.py                  # 预测推理
│   ├── export.py                   # 模型导出
│   ├── video_detect_gui.py        # 视频检测GUI
│   └── ...                         # 其他工具
│
├── 📂 runs/                        # 运行输出
│   ├── detect/                     # 检测结果
│   └── training_logs/              # 训练日志
│
└── 📦 yolo*.pt                     # 预训练模型
```

📖 查看完整架构：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 📊 数据集

### 类别定义

| ID | 类别 | 说明 |
|----|------|------|
| 0 | fall | 跌倒/摔倒的人 |
| 1 | person | 人员（站立/行走等） |

### 数据统计

| 数据集 | 图片数量 | 标注数量 |
|--------|---------|----------|
| 训练集 | 9,870 | 10,000 |
| 验证集 | 2,064 | 2,064 |
| **总计** | **11,934** | **12,064** |

### 数据特点

- ✅ 高质量标注
- ✅ 多样化场景
- ✅ 类别平衡良好
- ✅ YOLO标准格式

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

### 2️⃣ 启动训练GUI

```bash
# 方式1: 使用批处理文件
run.bat

# 方式2: 直接运行
python scripts/model_train_qt.py
```

### 3️⃣ 数据集分析

```bash
# 统计数据集信息
python scripts/dataset_stats.py

# 可视化预览
python scripts/dataset_preview.py
```

### 4️⃣ 模型训练

#### 使用GUI（推荐）
1. 打开 `train_qt.py`
2. 配置参数（模型、epochs、batch等）
3. 点击 "Start" 开始训练
4. 实时监控Loss和mAP曲线

#### 使用命令行
```bash
# 基础训练
python scripts/model_train.py

# 自定义参数
python scripts/model_train.py \
    --model yolo11n.pt \
    --epochs 300 \
    --batch 32 \
    --imgsz 640

# 从断点恢复
python scripts/model_train.py --resume
```

### 5️⃣ 模型验证

```bash
# 验证最佳模型
python scripts/model_val.py --weights runs/train/best.pt

# 自定义阈值
python scripts/model_val.py \
    --weights runs/train/best.pt \
    --conf 0.25 \
    --iou 0.45
```

或在GUI中：
1. 切换到 "Validate" 标签页
2. 选择权重文件
3. 点击 "Run"
4. 查看性能指标和混淆矩阵

### 6️⃣ 模型推理

```bash
# 单张图片
python scripts/model_predict.py --source test.jpg

# 视频文件
python scripts/model_predict.py --source video.mp4

# 文件夹批量处理
python scripts/model_predict.py --source ./images/

# 摄像头实时检测
python scripts/model_predict.py --source 0 --show
```

或使用视频检测GUI：
```bash
python scripts/model_detect_qt.py
```

### 7️⃣ 模型导出

```bash
# 导出为ONNX（通用格式）
python scripts/model_export.py --format onnx

# 导出为NCNN（ARM设备推荐）
python scripts/model_export.py --format ncnn

# INT8量化（更小更快）
python scripts/model_export.py --format onnx --int8

# 导出所有格式
python scripts/model_export.py --format all
```

## 🖥️ GUI功能展示

### 训练界面

```
┌─────────────────────────────────────────────────────────────┐
│ ◈ YOLO Training Studio  |  Fall Detection  |  GPU: RTX...  │
├──────────────┬──────────────────────┬──────────────────────┤
│ Configuration│                      │ ● Console            │
│ Model: [...] │   Loss Chart         │ [08:28:33] 🚀 ...    │
│ Epochs: 300  │                      │ [08:28:34] Epoch 1.. │
│ Batch: 32    ├──────────────────────┤                      │
│ ImgSz: 640   │                      │                      │
│              │   mAP Chart          │                      │
│ Optimizer:.. │                      │                      │
│ Device: GPU  │                      │                      │
├──────────────┤                      │                      │
│ Algorithm    │                      │                      │
│ Focal γ: 0.0 │                      │                      │
│ Smoothing:.. │                      │                      │
├──────────────┴──────────────────────┴──────────────────────┤
│ [Start] [Stop]  ████████░░░░ 60%                           │
│ Epoch: 180  |  mAP@0.5: 0.85  |  Best: 0.87               │
└─────────────────────────────────────────────────────────────┘
```

**特性**：
- 📊 实时Loss和mAP曲线
- 🎨 彩色日志输出
- 🔄 进度条智能去重
- 📈 指标卡片显示
- 💾 自动保存训练历史

### 验证界面

- 权重文件选择
- 性能指标表格
- 混淆矩阵可视化
- 各类别详细分析

## 📱 部署方案

| 部署方式 | 推理框架 | 速度 | 备注 |
|---------|---------|------|------|
| ONNX + CPU | ONNX Runtime | ⭐⭐⭐ | 通用方案，安装方便 |
| NCNN | ncnn | ⭐⭐⭐⭐ | ARM优化，Orange Pi推荐 |
| RKNN | Rockchip NPU | ⭐⭐⭐⭐⭐ | 需Orange Pi 5 (RK3588) |

### 支持的平台

| 平台 | 框架 | 速度 | 推荐场景 |
|------|------|------|----------|
| **PC (x86)** | ONNX Runtime | ⭐⭐⭐⭐ | 开发测试 |
| **ARM (Orange Pi)** | NCNN | ⭐⭐⭐⭐ | 边缘设备 |
| **Rockchip NPU** | RKNN | ⭐⭐⭐⭐⭐ | Orange Pi 5 |
| **NVIDIA Jetson** | TensorRT | ⭐⭐⭐⭐⭐ | Jetson系列 |
| **Android** | MNN/TNN | ⭐⭐⭐ | 移动设备 |

### Orange Pi 部署流程

1. **导出模型**
   ```bash
   python scripts/model_export.py --format ncnn --half
   ```

2. **传输到设备**
   ```bash
   scp runs/export/model.ncnn orangepi@192.168.1.100:~/
   ```

3. **在Orange Pi上运行**
   ```bash
   # 安装ncnn
   sudo apt install libncnn-dev
   
   # 运行推理
   ./detect_model model.ncnn input.jpg
   ```

### 性能优化建议

| 优化项 | 说明 | 效果 |
|--------|------|------|
| **输入尺寸** | 320x320 vs 640x640 | 速度提升2倍 |
| **INT8量化** | FP32 → INT8 | 体积缩小4倍，速度提升2-3倍 |
| **模型选择** | YOLO11n vs YOLO11m | n模型速度快3倍 |
| **多线程** | 设置workers | CPU利用率提升 |

## 🛠️ 工具脚本

### 数据处理

| 脚本 | 功能 | 用法 |
|------|------|------|
| `merge_datasets.py` | 合并多个数据集 | `python scripts/merge_datasets.py` |
| `merge_person_data.py` | Person数据合并 | `python scripts/merge_person_data.py` |
| `cleanup_corrupt_labels.py` | 清理损坏标注 | `python scripts/cleanup_corrupt_labels.py` |
| `dataset_stats.py` | 数据集统计 | `python scripts/dataset_stats.py` |
| `preview_dataset.py` | 数据集预览 | `python scripts/preview_dataset.py` |

### 训练与推理

| 脚本 | 功能 | 用法 |
|------|------|------|
| `train_qt.py` | ⭐ Qt训练GUI | `python scripts/train_qt.py` |
| `train.py` | 命令行训练 | `python scripts/train.py` |
| `val.py` | 模型验证 | `python scripts/val.py` |
| `predict.py` | 预测推理 | `python scripts/predict.py` |
| `export.py` | 模型导出 | `python scripts/export.py` |

### GUI工具

| 脚本 | 功能 | 用法 |
|------|------|------|
| `video_detect_gui.py` | 视频检测GUI | `python scripts/video_detect_gui.py` |
| `distill_gui_qt.py` | 知识蒸馏GUI | `python scripts/distill_gui_qt.py` |

## 📚 文档

- 📖 [项目架构详解](PROJECT_STRUCTURE.md)
- ⚙️ [配置指南](CONFIG_GUIDE.md)
- 📝 [训练技巧](#-训练技巧)

## 💡 训练技巧

### 数据准备

1. **检查数据质量**
   ```bash
   python scripts/dataset_preview.py  # 可视化检查
   python scripts/cleanup_corrupt_labels.py  # 清理错误标注
   ```

2. **分析数据分布**
   ```bash
   python scripts/dataset_stats.py  # 查看类别平衡
   ```

### 训练调优

| 场景 | 建议配置 |
|------|----------|
| **小数据集** (< 5K) | epochs: 500-1000, batch: 8-16, lr: 0.0001 |
| **中等数据集** (5K-20K) | epochs: 200-400, batch: 16-32, lr: 0.001 |
| **大数据集** (> 20K) | epochs: 100-200, batch: 32-64, lr: 0.01 |

### 常见问题解决

#### ❌ 显存不足 (OOM)
```yaml
解决方案:
- 减小 batch size (32 → 16 → 8)
- 减小 imgsz (640 → 512 → 416)
- 使用更小的模型 (m → s → n)
- 设置 workers=0
```

#### ❌ 训练不收敛
```yaml
解决方案:
- 降低学习率 (lr0: 0.001 → 0.0001)
- 增加 warmup_epochs (3 → 10)
- 检查数据标注质量
- 增加训练轮数
```

#### ❌ 过拟合
```yaml
解决方案:
- 增强数据增强 (mixup, copy_paste)
- 添加 label_smoothing (0.1)
- 增加 weight_decay
- 使用早停 (patience=50)
```

#### ❌ 训练速度慢
```yaml
解决方案:
- 增加 batch size
- 增加 workers (CPU模式)
- 使用GPU训练
- 启用混合精度 (amp=True)
```

### 最佳实践

1. **训练前**
   - ✅ 检查数据集质量和分布
   - ✅ 预览标注确保正确
   - ✅ 选择合适的模型大小
   - ✅ 配置合理的batch size

2. **训练中**
   - ✅ 监控Loss和mAP曲线
   - ✅ 观察控制台日志
   - ✅ 不要中断训练过程
   - ✅ 定期保存checkpoint

3. **训练后**
   - ✅ 验证模型性能
   - ✅ 分析混淆矩阵
   - ✅ 导出部署格式
   - ✅ 备份最佳模型

## 📊 训练日志

所有训练记录保存在 `runs/training_logs/`：

```json
{
  "timestamp": "2026-05-08 10:54:43",
  "status": "success",
  "config": {...},
  "results": {
    "best_mAP50": 0.87,
    "best_mAP50_95": 0.65,
    "epochs_trained": 300
  },
  "history": {...}
}
```

## 🔐 注意事项

1. ⚠️ 训练时确保有足够的磁盘空间（至少10GB）
2. ⚠️ GPU训练需要安装CUDA和cuDNN
3. ⚠️ 大batch size需要更多显存
4. ⚠️ 定期备份重要的模型权重
5. ⚠️ 验证时使用与训练相同的图像尺寸
6. ⚠️ 不要随意修改数据集路径

## 📞 技术支持

遇到问题请：
1. 查看 `runs/training_logs/` 中的错误日志
2. 检查控制台的完整错误信息
3. 参考 [Ultralytics官方文档](https://docs.ultralytics.com/)
4. 检查数据集格式是否符合YOLO标准

## 📄 License

MIT License

---

<div align="center">

**Made with ❤️ for Fall Detection**

如有问题，请查阅文档或提交Issue

</div>
