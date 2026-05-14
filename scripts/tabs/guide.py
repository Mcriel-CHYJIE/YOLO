"""引导标签页 — 介绍各面板功能"""
from scripts.tabs.base import *

TABS = [
    ("Training", "配置并启动 YOLO 模型训练", "Model / Epochs / Batch / ImgSz / Optimizer / Scheduler 等超参数设置，支持 GPU/CPU 切换，实时显示 Loss、mAP 曲线及系统资源监控", PRI),
    ("Validate", "验证已训练模型的性能", "加载训练好的权重，在验证集上计算 mAP@0.5 / mAP@0.5:0.95 / Precision / Recall 等指标", GREEN),
    ("Predict", "对图片或视频进行实时目标检测", "加载模型并推理，实时显示检测结果、FPS、各类别统计信息", AMBER),
    ("Dataset", "数据集管理与预处理", "数据集结构概览、类别分布统计，方便检查标注质量与数据平衡", "#8b5cf6"),
    ("Preprocess", "视频预处理：重命名 + 缩放 + 抽帧", "将原始视频统一重命名、保持宽高比缩放到指定尺寸、每秒随机抽取一帧保存为 JPG 图片", "#14b8a6"),
    ("Label", "手动/自动标注与审核", "支持手动绘制标注框、移动/调整/删除框，集成 YOLO 自动标注，审核后导出为 YOLO 格式数据集", "#ec4899"),
    ("Distill", "知识蒸馏：Teacher → Student", "以大模型为教师、小模型为学生进行知识蒸馏训练，在保持精度的同时获得更轻量的模型", "#f97316"),
    ("Export", "导出训练好的模型", "支持 ONNX / TorchScript / NCNN / OpenVINO / TensorRT / TFLite / EdgeTPU / CoreML 等多种格式", RED),
]


class GuideTab(QWidget):
    """引导标签页"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._build()

    def _build(self):
        lo = QVBoxLayout(self)
        lo.setContentsMargins(20, 20, 20, 20)
        lo.setSpacing(14)

        # 标题
        title = QLabel("YOLO Training Studio — 功能导览")
        title.setStyleSheet(f"font-size:24px;font-weight:700;color:{TEXT};border:none;")
        lo.addWidget(title)

        subtitle = QLabel("选择一个标签页开始你的深度学习工作流")
        subtitle.setStyleSheet(f"font-size:14px;color:{TEXT3};border:none;margin-bottom:4px;")
        lo.addWidget(subtitle)

        # 卡片网格
        grid = QGridLayout()
        grid.setSpacing(10)

        for i, (name, brief, detail, color) in enumerate(TABS):
            row, col = divmod(i, 4)

            card = QWidget()
            card.setStyleSheet(f"background:{CARD};border:1px solid {BORDER};border-radius:8px;")
            card.setMinimumHeight(160)

            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 14, 14, 14)
            cl.setSpacing(8)

            # 圆点 + 标题
            h = QHBoxLayout()
            h.setSpacing(8)
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color};font-size:14px;border:none;")
            h.addWidget(dot)

            lbl = QLabel(name)
            lbl.setStyleSheet(f"font-size:18px;font-weight:700;color:{TEXT};border:none;")
            h.addWidget(lbl)
            h.addStretch()
            cl.addLayout(h)

            # 简短说明
            brief_lbl = QLabel(brief)
            brief_lbl.setStyleSheet(f"font-size:13px;color:{TEXT2};border:none;font-weight:500;")
            brief_lbl.setWordWrap(True)
            cl.addWidget(brief_lbl)

            # 详细说明
            detail_lbl = QLabel(detail)
            detail_lbl.setStyleSheet(f"font-size:12px;color:{TEXT3};border:none;line-height:1.4;")
            detail_lbl.setWordWrap(True)
            cl.addWidget(detail_lbl, 1)

            grid.addWidget(card, row, col)

        lo.addLayout(grid, 1)

        # 底部信息栏
        footer = QWidget()
        footer.setStyleSheet(f"background:{CARD};border:1px solid {BORDER};border-radius:6px;")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(12, 8, 12, 8)
        fl.addWidget(QLabel(f"项目: {TITLE}", styleSheet=f"font-size:12px;color:{TEXT2};border:none;"))
        fl.addStretch()
        fl.addWidget(QLabel(f"类别数: {len(CLASSES)}", styleSheet=f"font-size:12px;color:{TEXT2};border:none;"))
        fl.addWidget(QLabel("|", styleSheet=f"font-size:12px;color:{BORDER};border:none;"))
        fl.addWidget(QLabel("快捷键: A 上一张 | D 下一张 | W 删除框 | S 删除图片",
                            styleSheet=f"font-size:12px;color:{TEXT3};border:none;"))
        lo.addWidget(footer)
