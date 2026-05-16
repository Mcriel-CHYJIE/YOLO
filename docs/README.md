# 📚 YOLO Fall Detection - 项目文档

欢迎来到跌倒检测项目的文档中心。本文件夹包含所有训练配置、预设参数和技术说明。

---

## 📖 文档导航

### 🚀 入门必读

| 文档 | 说明 | 推荐阅读 |
|------|------|----------|
| [PRESETS_GUIDE.md](PRESETS_GUIDE.md) | **三组预设参数配置指南** | ⭐⭐⭐⭐⭐ 所有用户必读 |
| [TRAINING_ANALYSIS_yolo11n.md](TRAINING_ANALYSIS_yolo11n.md) | **yolo11n 深度配置分析** | ⭐⭐⭐⭐ 推荐 |

### 📊 训练配置

| 文档 | 说明 | 适用场景 |
|------|------|----------|
| [TRAINING_CONFIG.md](TRAINING_CONFIG.md) | YOLO11m 高精度训练配置（旧版） | 参考历史配置 |

### 🛠️ 高级技术

| 文档 | 说明 | 适用场景 |
|------|------|----------|
| [synthetic_data_pipeline.md](synthetic_data_pipeline.md) | 合成数据生成管线（Blender+Mixamo） | 数据增强 |

---

## 🎯 快速开始

### 1️⃣ 第一次训练？

**推荐阅读顺序**：
1. [PRESETS_GUIDE.md](PRESETS_GUIDE.md) - 了解三组预设参数
2. 选择 **均衡模式**（推荐新手）
3. 修改 `project.yaml` 中的 `training` 配置
4. 运行 `python scripts/main.py` 开始训练

### 2️⃣ 想要最高精度？

**推荐阅读顺序**：
1. [PRESETS_GUIDE.md](PRESETS_GUIDE.md) - 了解高精度模式
2. [TRAINING_ANALYSIS_yolo11n.md](TRAINING_ANALYSIS_yolo11n.md) - 深入理解参数
3. 选择 **高精度模式** 或 **大模型高精度模式**
4. 准备充足训练时间（15-35小时）

### 3️ 需要合成数据？

**推荐阅读**：
- [synthetic_data_pipeline.md](synthetic_data_pipeline.md) - 完整合成数据管线说明

---

## 📋 预设参数速查

| 预设 | epochs | batch | 时长 | mAP@0.5 | 推荐度 |
|------|--------|-------|------|---------|--------|
| **均衡模式** ⭐ | 300 | 32 | 6-8h | 90-93% | ⭐⭐⭐⭐⭐ |
| **高精度模式** | 800 | 32 | 15-20h | 92-95% | ⭐⭐⭐⭐ |
| **快速实验** | 100 | 48 | 1-2h | 85-88% | ⭐⭐⭐ |
| **大模型高精度** | 600 | 24 | 25-35h | 95-98% | ⭐⭐⭐ |

**完整预设配置**：[PRESETS.yaml](../PRESETS.yaml)

---

## 📂 文档结构

```
docs/
├── README.md                          ← 本文档（文档导航）
├── PRESETS_GUIDE.md                   ← 预设参数使用指南（推荐）
├── TRAINING_ANALYSIS_yolo11n.md       ← yolo11n 深度分析
├── TRAINING_CONFIG.md                 ← yolo11m 历史配置（参考）
└── synthetic_data_pipeline.md         ← 合成数据管线
```

---

## 🔧 配置文件位置

```
项目根目录/
├── project.yaml                       ← 当前激活配置
├── PRESETS.yaml                       ← 所有预设参数模板
├── datasets/data.yaml                 ← 数据集配置
└── docs/                              ← 本文档文件夹
```

---

## 📞 常见问题

### Q: 应该选择哪个预设？
A: 
- 日常训练 → **均衡模式**（6-8小时，90-93% mAP）
- 最终部署 → **高精度模式**（15-20小时，92-95% mAP）
- 快速验证 → **快速实验**（1-2小时，85-88% mAP）
- yolo11n不够 → **大模型高精度**（25-35小时，95-98% mAP）

### Q: 训练时显存不足怎么办？
A: 依次尝试：
1. 降低 batch size: 32 → 24 → 16
2. 减少 workers: 8 → 6 → 4
3. 禁用数据增强: copy_paste → 0
4. 详见：[PRESETS_GUIDE.md](PRESETS_GUIDE.md#显存监控与调整)

### Q: 如何修改训练参数？
A: 
1. 打开 [project.yaml](../project.yaml)
2. 找到 `training:` 部分
3. 替换为预设配置（见 [PRESETS_GUIDE.md](PRESETS_GUIDE.md)）
4. 保存后重启训练界面

### Q: 训练日志在哪里查看？
A: 
- 实时日志：训练界面 Console 面板
- 历史日志：`runs/training_logs/`
- 模型权重：`runs/detect/`

---

## 📈 项目信息

- **项目**: YOLO Fall Detection（跌倒检测）
- **类别**: standing, sitting, squatting, fallen
- **设备**: RTX 5070 Ti (16GB VRAM)
- **框架**: Ultralytics YOLO11
- **数据集**: ~10000 张训练图像，1259 张验证图像

---

## 🔄 文档更新记录

| 日期 | 更新内容 | 版本 |
|------|----------|------|
| 2026-05-16 | 创建文档索引和导航 | v1.0 |
| 2026-05-16 | 添加三组预设参数文档 | v1.0 |

---

**祝您训练顺利！** 🚀

如有疑问，请查阅相关文档或检查训练日志。
