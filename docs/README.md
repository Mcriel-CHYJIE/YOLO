# 📚 YOLO Training Platform - Project Documentation

> ⚠️ **Important**: This documentation has been consolidated into a single comprehensive guide. Please read **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for complete training instructions.

Welcome to the YOLO Training Platform documentation center. All training configurations, preset parameters, and technical specifications have been unified into **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**.

---

## 📖 快速导航

### 🌟 主要文档

| 文档 | 说明 | 推荐阅读 |
|------|------|----------|
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | **📚 完整训练指南** - 包含所有训练配置、预设参数、深度分析和故障排除 | ⭐⭐⭐⭐⭐ **必读** |

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

## 🎯 快速开始

### 立即开始训练

**只需3步**:
1. 阅读 **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** 的[快速开始章节](TRAINING_GUIDE.md#-快速开始)
2. 选择适合的[预设模式](TRAINING_GUIDE.md#-预设参数配置)(推荐新手使用**均衡模式**)
3. 运行 `python scripts/main.py` 开始训练

### 查找特定信息

如需深入了解某个主题,请参考 [TRAINING_GUIDE.md](TRAINING_GUIDE.md) 的对应章节:

- **预设参数详解** → [第2章: 预设参数配置](TRAINING_GUIDE.md#-预设参数配置)
- **参数设计原理** → [第3章: 深度配置分析](TRAINING_GUIDE.md#-深度配置分析)
- **历史配置参考** → [第4章: 历史配置参考](TRAINING_GUIDE.md#-历史配置参考-yolo11m)
- **故障排除** → [第5章: 训练监控与调优](TRAINING_GUIDE.md#-训练监控与调优)
- **最新功能** → [第6章: 最新功能更新](TRAINING_GUIDE.md#-最新功能更新)









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
| 2026-05-19 | **重大更新**: 合并4个文档为统一综合指南 [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | v2.0 |
| 2026-05-19 | 添加Predict视频24FPS播放、Label负样本导出功能 | v1.2 |
| 2026-05-19 | 修复Label导航保存bug、支持0样本导出 | v1.1 |
| 2026-05-16 | 创建文档索引和导航 | v1.0 |
| 2026-05-16 | 添加三组预设参数文档 | v1.0 |

---

**祝您训练顺利！** 🚀

如有疑问，请查阅相关文档或检查训练日志。
