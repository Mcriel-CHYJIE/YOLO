# =============================================================================
# YOLO Training Studio — 基于 Ultralytics YOLO 的通用目标检测训练平台
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# SPDX-License-Identifier: MIT | See <ROOT>/LICENSE for full text
# =============================================================================

"""引导标签页 — 程序介绍 + 每个标签页的功能使用指引"""
from main.core.base import *


TAB_GUIDES = [
    {
        'name': 'Training',
        'icon': '🎯',
        'brief': '配置超参数并启动 YOLO 模型训练，实时查看 Loss / mAP 曲线和日志',
        'color': PRI,
        'steps': [
            ('Model Selection', '从下拉框选择预训练权重（.pt），自动扫描 models/ 目录。续训时选 previous best.pt'),
            ('Set Hyperparameters', '配置 Epochs/Batch/ImgSz/Optimizer/Device。推荐：Epochs=800, Batch=32, ImgSz=640, AdamW'),
            ('Learning Rate', 'LR=0.001-0.002, LRF=0.01, Warmup=15。续训时 LR 降 10 倍至 0.0001, Warmup=1'),
            ('Scheduler', 'Cosine（余弦退火，平滑衰减） / Linear（线性衰减）。续训推荐 Linear'),
            ('Augmentation', 'Rotation=15°, IoU=0.7, Close Mosaic=30, Copy-Paste=0.3, cls_pw=0.75'),
            ('Start Training', '点击 Start Training 按钮。参数自动锁定，停止/中断后恢复'),
            ('Monitor', '训练中实时查看 Loss Chart（红色曲线）和 mAP Chart（绿色曲线）'),
            ('Auto-Save', '每轮自动保存 best.pt / last.pt，best.pt 保留 mAP 最高的 epoch'),
        ],
    },
    {
        'name': 'Validate',
        'icon': '✅',
        'brief': '在验证集上评估已训练模型的检测性能',
        'color': GREEN,
        'steps': [
            ('Model Selection', '自动搜索 runs/*/weights/best.pt 选取最新模型，也可手动选择 .pt 文件'),
            ('Parameters', '配置验证 ImgSz（640）、Batch（48）、Conf Threshold（0.35）、IoU（0.5）'),
            ('Metrics', '输出 mAP@0.5 / mAP@0.5:0.95 / Precision / Recall 四项核心指标'),
            ('Per-Class Table', '每类的 AP、Precision、Recall、F1 详细数值，帮助定位弱势类别'),
            ('Confusion Matrix', '混淆矩阵可视化，分析类别间误检情况（例如 fallen 误检为 sitting）'),
            ('Output', '验证结果（图表 + 指标明细）保存到 runs/val/ 目录'),
        ],
    },
    {
        'name': 'Predict',
        'icon': '👁️',
        'brief': '对单张图片或视频进行实时目标检测推理',
        'color': AMBER,
        'steps': [
            ('Source', '选择图片（.jpg/.png）或视频（.mp4/.avi/.mov）文件'),
            ('Model', '选择已训练的 .pt 模型文件，推荐使用 best.pt'),
            ('Thresholds', 'Confidence=0.35（过滤低分框）, IoU=0.5（NMS 去重）'),
            ('Run', '点击 Start 开始推理，检测框实时叠加在画面上'),
            ('Results', '每帧显示 FPS、各类别检测数量统计、推理耗时'),
            ('Save', '预测结果图片/视频保存到 runs/predict/ 目录'),
        ],
    },
    {
        'name': 'Dataset',
        'icon': '📁',
        'brief': '查看数据集结构与类别分布统计',
        'color': '#8b5cf6',
        'steps': [
            ('Data Info', '显示总图片数、总标注数、总实例数等概览统计'),
            ('Distribution', '类别分布直方图，直观展示各类别实例数量是否均衡'),
            ('Split', '支持切换 Train / Val 查看不同分片的统计信息'),
            ('Class Balance', '每类标注框数量统计，快速定位类别不平衡问题'),
        ],
    },
    {
        'name': 'Preprocess',
        'icon': '🎞️',
        'brief': '视频预处理流水线：集中 → 重命名 → 缩放 → 抽帧',
        'color': '#14b8a6',
        'steps': [
            ('Step 1 — Collect', '将散落在子文件夹中的视频统一收集到指定目录'),
            ('Step 2 — Rename', '按文件名排序统一重命名为 00.ext, 01.ext ...'),
            ('Step 3 — Resize', '保持宽高比 Letterbox 缩放至目标尺寸（默认 640×640），黑边填充'),
            ('Step 4 — Extract Frames', '按目标 FPS（推荐 1-2 fps）均匀抽取帧，JPEG 质量 95'),
            ('Output Format', '输出命名：{视频源}-{编号}-{秒数:04d}.jpg'),
            ('Tips', '推荐采样率 1-2 fps 以避免时序相邻帧过于相似。连续帧间差异过大会降低标注效率'),
        ],
    },
    {
        'name': 'Label',
        'icon': '🏷️',
        'brief': '手动标注 / 自动标注 / 审核导出全流程',
        'color': '#ec4899',
        'steps': [
            ('Select Source', '选择待标注的图片目录（或使用 Preprocess 输出的帧目录）'),
            ('Manual Labeling', '选择类别 → 在图片上拖拽绘制矩形框。单击框可移动，四角手柄调整尺寸，右键删除'),
            ('Auto-Label', '选择检测模型对全部图片自动预测生成标注 → 再逐张审核修正'),
            ('Navigation', '← → 或 Prev/Next 按钮浏览图片，快捷键可配置'),
            ('Save Progress', '标注自动保存到内存，切换图片时持久化。避免意外丢失'),
            ('Export Dataset', '按比例随机切分 Train/Val → 生成 YOLO 格式 .txt 标注文件和 config.yaml'),
        ],
    },
    {
        'name': 'Distill',
        'icon': '🔬',
        'brief': '知识蒸馏：大模型（Teacher）教小模型（Student），提升小模型精度',
        'color': '#f97316',
        'steps': [
            ('Concept', 'Teacher 模型冻结权重，Student 同时学习 One-Hot 标签和 Teacher 的 Soft Prediction'),
            ('Alpha', '蒸馏权重 0.5（50% 检测 Loss + 50% 蒸馏 Loss），控制两个目标的平衡'),
            ('Teacher', '自动搜索 runs/ 下最新的 best.pt，也可手动指定更大模型（如 yolo11m）'),
            ('Student', '通常选择小模型（yolo11n/s），蒸馏后推理速度不变但精度接近 Teacher'),
            ('Parameters', 'LR=0.002（略高于普通训练加速知识迁移）, Batch=24（Teacher 冻结可更大）'),
            ('Output', 'Student 权重保存到 runs/distill/，可直接用于 Predict / Export'),
            ('Benefit', '小模型蒸馏后 mAP50 可提升 6-11%，推理速度保持与 Student 一致'),
        ],
    },
    {
        'name': 'Export',
        'icon': '📦',
        'brief': '将训练好的 .pt 模型导出为部署格式',
        'color': RED,
        'steps': [
            ('Select Model', '选择需要导出的 .pt 文件（best.pt 或蒸馏输出）'),
            ('Choose Format', 'ONNX（通用）/ TorchScript / NCNN / OpenVINO / TensorRT / TFLite / CoreML'),
            ('Set Options', 'ImgSz（默认 640）、Half（FP16 缩小体积）、NMS（内嵌非极大抑制）'),
            ('Export', '点击 Export 按钮，转换进度显示在控制台'),
            ('Output', '导出文件保存到模型同级目录，文件名带格式后缀'),
        ],
    },
    {
        'name': 'Settings',
        'icon': '⚙️',
        'brief': '编辑项目配置文件中的类别名称（仅此一项功能）',
        'color': '#666',
        'steps': [
            ('Edit Names', '每行一个类别名称，顺序决定索引（第 1 行 = 0, 第 2 行 = 1...）'),
            ('Save', '保存后自动更新 project.yaml 的 classes 列表和 names 映射'),
            ('Hot Reload', '配置立即生效，窗口标题、类别名称等全局常量自动更新'),
        ],
    },
    ]


def _make_tab_item(name, icon, brief, steps, color):
    """创建一个 tab 使用指引组件"""
    section = QWidget()
    section.setStyleSheet(f"""
        background:{CARD};
        border:1px solid {BORDER};
        border-left:4px solid {color};
        border-radius:0 7px 7px 0;
    """)
    layout = QVBoxLayout(section)
    layout.setContentsMargins(16, 12, 16, 12)
    layout.setSpacing(6)

    # 标题行
    h = QHBoxLayout()
    h.setSpacing(8)
    dot = QLabel(icon)
    dot.setStyleSheet(f"font-size:16px;border:none;")
    h.addWidget(dot)
    lbl = QLabel(name)
    lbl.setStyleSheet(f"font-size:16px;font-weight:700;color:{TEXT};border:none;")
    h.addWidget(lbl)
    h.addStretch()
    layout.addLayout(h)

    # 简短说明
    brief_lbl = QLabel(brief)
    brief_lbl.setStyleSheet(f"font-size:12px;color:{TEXT2};border:none;font-weight:500;")
    brief_lbl.setWordWrap(True)
    layout.addWidget(brief_lbl)

    # 步骤列表
    for step_name, step_desc in steps:
        w = QWidget()
        w.setStyleSheet("background:transparent;border:none;")
        wl = QVBoxLayout(w)
        wl.setContentsMargins(4, 3, 0, 3)
        wl.setSpacing(1)

        # 步骤标题
        sh = QHBoxLayout()
        sh.setSpacing(6)
        bullet = QLabel("▸")
        bullet.setStyleSheet(f"font-size:11px;color:{color};border:none;font-weight:700;")
        bullet.setFixedWidth(14)
        sh.addWidget(bullet)
        sn = QLabel(step_name)
        sn.setStyleSheet(f"font-size:12px;font-weight:600;color:{TEXT};border:none;")
        sh.addWidget(sn)
        sh.addStretch()
        wl.addLayout(sh)

        # 步骤描述
        sd = QLabel(step_desc)
        sd.setStyleSheet(f"font-size:11px;color:{TEXT3};border:none;padding-left:20px;")
        sd.setWordWrap(True)
        wl.addWidget(sd)

        layout.addWidget(w)

    return section


class GuideTab(QWidget):
    """引导标签页 — 程序介绍 + 每个标签页的功能使用指引"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._build_ui()

    def _make_intro(self):
        """创建程序介绍组件（动态读取当前配置）"""
        w = QWidget()
        w.setStyleSheet(f"background:{CARD};border:1px solid {BORDER};border-radius:7px;")
        lo = QVBoxLayout(w)
        lo.setContentsMargins(20, 16, 20, 16)
        lo.setSpacing(10)

        # 标题
        h = QHBoxLayout()
        h.setSpacing(8)
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{PRI};font-size:14px;border:none;")
        h.addWidget(dot)
        lbl = QLabel("Program Introduction")
        lbl.setStyleSheet(f"font-size:18px;font-weight:700;color:{TEXT};border:none;")
        h.addWidget(lbl)
        h.addStretch()
        lo.addLayout(h)

        # 简短说明
        brief = QLabel("YOLO Training Studio — 基于 YOLOv8/YOLO11 的桌面端训练与部署工作室")
        brief.setStyleSheet(f"font-size:12px;color:{TEXT2};border:none;font-weight:500;")
        brief.setWordWrap(True)
        lo.addWidget(brief)

        # 段落正文
        paras = [
            'YOLO Training Studio 是一个基于 PyQt5 的桌面应用程序，专为 YOLO 系列模型的训练、验证、预测、标注和部署提供一站式工作流。核心训练引擎基于 Ultralytics YOLO，支持 YOLOv8/YOLO11 全部模型变体。',
            '界面采用 WeChat 风格侧边栏导航 + QStackedWidget 多标签页布局，左侧固定 130px 窄边栏，右侧内容自适应。支持 Training / Predict / Dataset / Preprocess / Label / Distill / Validate / Export / Guide / Settings 共 10 个功能模块。',
        ]
        for text in paras:
            p = QLabel(text)
            p.setStyleSheet(f"font-size:11px;color:{TEXT3};border:none;line-height:1.6;")
            p.setWordWrap(True)
            lo.addWidget(p)

        # 动态规格表格
        lo.addSpacing(4)

        gpu_text = (f'{self.studio.gpu_name} ({self.studio.gpu_mem})'
                    if self.studio.gpu_ok else 'CPU (no GPU detected)')

        specs = [
            ('Tech Stack', 'PyQt5 · Ultralytics YOLO · PyTorch · CUDA'),
            ('Architecture', 'WeChat 风格侧边栏 + QStackedWidget · WebUI 训练界面'),
            ('Project', TITLE),
            ('Classes', f'{", ".join(CLASSES)} ({len(CLASSES)} 类)'),
            ('Data Config', DATA_YAML),
            ('GPU', gpu_text),
            ('Output', 'runs/train/ · runs/val/ · runs/predict/ · runs/distill/'),
        ]
        for label, value in specs:
            row = QWidget()
            row.setStyleSheet("background:transparent;border:none;")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(4, 1, 4, 1)
            rl.setSpacing(12)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"font-size:11px;font-weight:600;color:{TEXT};border:none;min-width:90px;")
            rl.addWidget(lbl)
            val = QLabel(value)
            val.setStyleSheet(f"font-size:11px;color:{TEXT3};border:none;")
            rl.addWidget(val, 1)
            lo.addWidget(row)

        return w

    # ═══════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════
    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:transparent;}}")

        inner = QWidget()
        inner.setStyleSheet("background:transparent;")
        lo = QVBoxLayout(inner)
        lo.setContentsMargins(24, 20, 24, 20)
        lo.setSpacing(12)

        # ── 标题 ──
        title = QLabel("YOLO Training Studio")
        title.setStyleSheet(f"font-size:26px;font-weight:700;color:{TEXT};border:none;")
        lo.addWidget(title)

        subtitle = QLabel("使用指南 — 程序介绍与各标签页详细操作指引")
        subtitle.setStyleSheet(f"font-size:13px;color:{TEXT3};border:none;margin-bottom:2px;")
        lo.addWidget(subtitle)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"border:none;border-top:1px solid {BORDER};margin:4px 0 6px;")
        lo.addWidget(sep)

        # ── 程序介绍 ──
        lo.addWidget(self._make_intro())

        # ── 分隔线 ──
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"border:none;border-top:1px solid {BORDER};margin:6px 0;")
        lo.addWidget(sep2)

        # 标签页使用指引
        section_title = QLabel("Tab Usage Guide")
        section_title.setStyleSheet(f"font-size:16px;font-weight:700;color:{TEXT};border:none;")
        lo.addWidget(section_title)

        section_desc = QLabel("每个标签页的功能说明与逐步操作指引")
        section_desc.setStyleSheet(f"font-size:12px;color:{TEXT3};border:none;")
        lo.addWidget(section_desc)

        for guide in TAB_GUIDES:
            lo.addWidget(_make_tab_item(
                guide['name'], guide['icon'], guide['brief'],
                guide['steps'], guide['color']
            ))

        lo.addStretch()

        # ── 底部信息栏 ──
        footer = QWidget()
        footer.setStyleSheet(f"background:{CARD};border:1px solid {BORDER};border-radius:6px;")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(12, 8, 12, 8)
        fl.addWidget(QLabel(f"Project: {TITLE}",
                            styleSheet=f"font-size:12px;color:{TEXT2};border:none;"))
        fl.addStretch()
        fl.addWidget(QLabel(f"Classes: {', '.join(CLASSES)}",
                            styleSheet=f"font-size:12px;color:{TEXT2};border:none;"))
        fl.addWidget(QLabel("|", styleSheet=f"font-size:12px;color:{BORDER};border:none;"))
        fl.addWidget(QLabel(f"{len(CLASSES)} classes · ultralytics based",
                            styleSheet=f"font-size:12px;color:{TEXT3};border:none;"))
        lo.addWidget(footer)

        scroll.setWidget(inner)

        main_lo = QVBoxLayout(self)
        main_lo.setContentsMargins(0, 0, 0, 0)
        main_lo.addWidget(scroll)
