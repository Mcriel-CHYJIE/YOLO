"""引导标签页 — 功能导览（文档式分段布局）"""
from scripts.tabs.base import *


SECTIONS = [
    {
        'name': 'Training',
        'brief': '配置并启动 YOLO 模型训练，实时监控收敛过程',
        'color': PRI,
        'items': [
            'Model — 从 models/ 文件夹中选择预训练权重（yolo11/yolov8 系列），下拉框自动扫描',
            'Epochs — 总训练轮数，摔倒项目推荐 800 轮（早停机制自动控制实际轮数）',
            'Batch — 每次迭代的图片数，RTX 5070 Ti 16GB 推荐 32，大数据集可降至 24',
            'ImgSz — 输入分辨率 640/800，640 为推荐值（精度与显存的平衡点）',
            'Optimizer — AdamW（推荐）/ Adam / SGD，AdamW 收敛更稳定',
            'Device — GPU/CPU 自动检测，无 GPU 时默认 CPU',
            'Scheduler — Cosine（余弦退火，平滑衰减）/ Linear（线性衰减）',
            'LR / LR Final — 初始学习率 0.002、最终学习率比例 0.01（最终 LR = LR × LRF）',
            'Patience — 早停耐心值 100 轮，验证 mAP 不提升即停止',
            'Warmup — 前 15 轮线性升温，防止初期梯度震荡',
            'Workers — 数据加载线程数，默认 CPU 核数的一半',
        ],
    },
    {
        'name': 'Algorithm',
        'brief': '数据增强与训练策略参数',
        'color': '#8b5cf6',
        'items': [
            'Focal γ — Focal Loss 系数，难分样本权重（默认 0.0 = 关闭）',
            'Label Smoothing — 标签平滑，0.10 推荐值，抑制过拟合提升泛化',
            'IoU — NMS IoU 阈值 0.7，控制检测框去重严格度',
            'Close Mosaic — 最后 N 轮关闭 Mosaic 增强做微调，默认 30 轮',
            'Copy-Paste — 目标复制粘贴增强，0.3 可提升目标密度（多目标场景）',
            'Rotation — 随机旋转 ±15°，增强多角度检测能力',
            'Multi-Scale — 训练时随机缩放输入尺寸，提升尺度鲁棒性',
        ],
    },
    {
        'name': 'Validate',
        'brief': '验证已训练模型的检测性能',
        'color': GREEN,
        'items': [
            '加载 runs/*/weights/best.pt 自动发现最新训练结果',
            '在验证集上计算 mAP@0.5 / mAP@0.5:0.95 / Precision / Recall',
            '显示混淆矩阵（Confusion Matrix），直观分析类别间的误检情况',
            '验证结果保存到 runs/val/ 目录，包含曲线图和详细指标',
        ],
    },
    {
        'name': 'Predict',
        'brief': '对图片或视频进行实时目标检测推理',
        'color': AMBER,
        'items': [
            '支持图片（jpg/png）和视频（mp4/avi/mov）输入',
            '实时显示检测结果，带类别标签和置信度',
            '显示 FPS、各类别检测数量统计',
            '可调整置信度阈值和 IoU 阈值控制检测敏感度',
        ],
    },
    {
        'name': 'Dataset',
        'brief': '数据集结构与分布概览',
        'color': '#8b5cf6',
        'items': [
            '显示数据集总图片数、已标注标注数、总实例数',
            '类别分布直方图，检查各类别是否平衡',
            '每类标注框数量统计，快速发现类别不平衡问题',
            '支持 train / val 切换查看',
        ],
    },
    {
        'name': 'Preprocess',
        'brief': '视频预处理：统一命名 + 缩放 + 抽帧',
        'color': '#14b8a6',
        'items': [
            '第一步：视频集中 — 将散落子文件夹的视频统一收集到同一目录',
            '第二步：重命名 — 按文件名排序统一命名为 00.ext, 01.ext...',
            '第三步：缩放 — 保持宽高比 letterbox 缩放至指定尺寸，黑边填充',
            '第四步：抽帧 — 按目标 FPS（推荐 1-2 fps）均匀抽取帧，JPEG 95 质量',
            '输出命名格式：{视频源}-{编号}-{秒数:04d}.jpg',
        ],
    },
    {
        'name': 'Label',
        'brief': '手动/自动标注与审核导出',
        'color': '#ec4899',
        'items': [
            '手动标注 — 选择类别后在图片上拖拽绘制矩形框',
            '自动标注 — 选择 YOLO 模型对全部图片自动检测标注，再人工审核修正',
            '标注操作 — 单击选框移动、四角手柄调整、右键删除、类别按钮切换',
            '导出数据集 — 按比例随机切分 train/val，生成 YOLO 格式 .txt 和 data.yaml',
            '支持多轮标注会话自动保存，避免意外丢失',
        ],
    },
    {
        'name': 'Distill',
        'brief': '知识蒸馏：大模型（Teacher）教小模型（Student）',
        'color': '#f97316',
        'items': [
            '核心原理 — Teacher 模型冻结权重，Student 同时学习真实标签和 Teacher 的 soft prediction',
            'Alpha — 蒸馏权重比例 0.5（50% 检测 loss + 50% 蒸馏 loss），控制两个目标的平衡',
            'Teacher — 自动搜索 runs/ 下最新的 best.pt，也可手动指定更大模型（如 yolo11m）',
            'Student — 通常选小模型（yolo11n），蒸馏后保持推理速度但精度接近大模型',
            'LR — 蒸馏学习率 0.002，略高于普通训练以加速知识迁移',
            'Batch — 蒸馏时可使用更大 batch（Teacher 冻结无梯度），推荐 24',
            'Warmup / Weight Decay / Momentum — 与普通训练一致，预热 5 轮',
            '收益 — yolo11n 蒸馏后可提升 mAP50 约 6-11%，推理速度不变',
        ],
    },
    {
        'name': 'Export',
        'brief': '导出训练好的模型到部署格式',
        'color': RED,
        'items': [
            'ONNX — 通用格式，适用于 ONNX Runtime，边缘设备首选',
            'TorchScript — PyTorch 原生部署格式',
            'NCNN — ARM 平台优化（树莓派/Orange Pi），INT8 量化后体积压缩 4 倍',
            'OpenVINO — Intel 平台加速推理',
            'TensorRT — NVIDIA GPU 加速，适合 Jetson 系列',
            'TFLite / EdgeTPU — 移动端/嵌入式 Google Coral 部署',
            'CoreML — Apple 设备（iPhone/Mac）部署',
        ],
    },
]


def _make_section(name, brief, items, color):
    """创建文档式分段组件"""
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
    dot = QLabel("●")
    dot.setStyleSheet(f"color:{color};font-size:12px;border:none;")
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

    # 详细项列表
    for item in items:
        w = QWidget()
        w.setStyleSheet("background:transparent;border:none;")
        wl = QHBoxLayout(w)
        wl.setContentsMargins(4, 1, 0, 1)
        wl.setSpacing(6)
        bullet = QLabel("·")
        bullet.setStyleSheet(f"font-size:12px;color:{color};border:none;font-weight:700;")
        bullet.setFixedWidth(10)
        wl.addWidget(bullet)
        text = QLabel(item)
        text.setStyleSheet(f"font-size:11px;color:{TEXT3};border:none;")
        text.setWordWrap(True)
        wl.addWidget(text, 1)
        layout.addWidget(w)

    return section


class GuideTab(QWidget):
    """引导标签页 — 文档式分段布局"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._build()

    def _build(self):
        # 外层 ScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none;background:transparent;}}")

        inner = QWidget()
        inner.setStyleSheet("background:transparent;")
        lo = QVBoxLayout(inner)
        lo.setContentsMargins(24, 20, 24, 20)
        lo.setSpacing(10)

        # 标题
        title = QLabel("YOLO Training Studio")
        title.setStyleSheet(f"font-size:26px;font-weight:700;color:{TEXT};border:none;")
        lo.addWidget(title)

        subtitle = QLabel("功能导览 — 每个标签页的作用与参数说明")
        subtitle.setStyleSheet(f"font-size:13px;color:{TEXT3};border:none;margin-bottom:2px;")
        lo.addWidget(subtitle)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"border:none;border-top:1px solid {BORDER};margin:4px 0 6px;")
        lo.addWidget(sep)

        # 文档式分段
        for s in SECTIONS:
            section = _make_section(s['name'], s['brief'], s['items'], s['color'])
            lo.addWidget(section)

        lo.addStretch()

        # 底部信息栏
        footer = QWidget()
        footer.setStyleSheet(f"background:{CARD};border:1px solid {BORDER};border-radius:6px;")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(12, 8, 12, 8)
        fl.addWidget(QLabel(f"项目: {TITLE}", styleSheet=f"font-size:12px;color:{TEXT2};border:none;"))
        fl.addStretch()
        fl.addWidget(QLabel(f"类别: {', '.join(CLASSES)}", styleSheet=f"font-size:12px;color:{TEXT2};border:none;"))
        fl.addWidget(QLabel("|", styleSheet=f"font-size:12px;color:{BORDER};border:none;"))
        fl.addWidget(QLabel(f"模型: models/*.pt ({len(CLASSES)} 类)",
                            styleSheet=f"font-size:12px;color:{TEXT3};border:none;"))
        lo.addWidget(footer)

        scroll.setWidget(inner)

        # 主布局
        main_lo = QVBoxLayout(self)
        main_lo.setContentsMargins(0, 0, 0, 0)
        main_lo.addWidget(scroll)
