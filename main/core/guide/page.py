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
        'brief': '配置超参数并启动 YOLO 模型训练，实时查看 Loss / mAP 曲线和日志\n    LoRA 低秩适配 + 7 种注意力模块 + 20 套参数预设',
        'color': PRI,
        'steps': [
            ('Model & Data',
             'Model: 选预训练 .pt 或 .yaml 架构，自动扫描 models/\n'
             'Epochs: 总轮数，推荐 200-500\n'
             'Batch: 每批样本数，yolo11n 可 48，大模型 32'),
            ('Input & Compute',
             'ImgSz: 416/512/640/800，默认 640\n'
             'Workers: 加载线程，推荐 CPU 核数减半，Win 4-8\n'
             'Optimizer: AdamW(推荐) / Adam / SGD'),
            ('Schedule & Device & LR',
             'Schedule: Cosine(余弦退火) / Linear(续训推荐)\n'
             'Device: GPU(自动) / CPU(调试)\n'
             'LR: 初始学习率 0.0005-0.002，CBAM 注入减半'),
            ('LR Final & Warmup & Patience',
             'LR Final: 最终 = LR × LRF，默认 0.01\n'
             'Warmup: 预热轮数，CBAM 推荐 10-15\n'
             'Patience: 早停，mAP 连续 N 轮不升即停'),
            ('WarmM & Momentum & Weight Decay',
             'WarmM: 预热期动量 0.8\n'
             'Momentum: 默认 0.937，一般不动\n'
             'Weight Decay: L2 权重衰减 0.0005'),
            ('Dropout & cls_pw & IoU',
             'Dropout: 0.0(关) / 0.1-0.3(过拟合时)\n'
             'cls_pw: <1 降误报，>1 提召回\n'
             'IoU: 训练 NMS 阈值 0.7-0.8'),
            ('Close Mosaic & Toggles',
             'Close Mosaic: 最后 N 轮关 Mosaic，15-30\n'
             'AMP: 混合精度加速，减 VRAM ~30%\n'
             'Cache: 数据集缓存到 RAM，加速加载\n'
             'Multi-Scale: 每轮随机缩放 ±25%，增泛化'),
            ('Augmentation',
             'Rotation/Scale/Translate/Shear/Persp: 几何变换\n'
             'Mosaic/MixUp/Copy-Paste: 混合增强\n'
             'flip_lr/flipud: 翻转(摔倒禁 flipud)\n'
             'HSV_H/S/V: 色彩扰动'),
            ('Attention',
             'SE/CBAM/CA/ECA/SimAM/EMA/GAM 七种可选\n'
             'CBAM 精度最稳，ECA 极轻量，SimAM 无参数\n'
             '训练后导出不含注意力模块，零推理成本'),
            ('LoRA',
             'LoRA Rank: 0=关闭，1-32 有效\n'
             '低秩适配微调，大幅减少可训练参数量'),
            ('Presets — 20 Profiles',
             '4 基底模型 × 5 场景 = 20 套预设（profiles/ 目录）\n'
             '场景: 均衡型 | +CBAM 高精度 | +LoRA 高精度 | +EMA 高召回 | 鲁棒泛化\n'
             '每个场景针对 4 种模型 (11n/8n/11s/8s) 独立调参\n'
             '加载自动填入: 全部超参数 + 注意力类型 + LoRA Rank'),
            ('Monitor',
             'Loss/mAP 实时曲线 + Stats 信息面板\n'
             'Epoch / mAP@0.5 / Best / Loss / mAP50:95 / Prec / Recall\n'
             '自动保存 best.pt + last.pt'),
        ],
    },
    {
        'name': 'Predict',
        'icon': '👁️',
        'brief': '对图片或视频进行实时目标检测推理，支持热力图和特征图可视化',
        'color': AMBER,
        'steps': [
            ('Source',
             '选图片 .jpg/.png 或视频 .mp4/.avi/.mov，也支持摄像头'),
            ('Model',
             '选 .pt 模型，推荐 best.pt，自动扫描 models/'),
            ('Thresholds',
             'Conf: 0.55(过滤低分框) | IoU: 0.5(NMS 去重)'),
            ('Visualization',
             'Detection: 检测框+类别+置信度\n'
             'Heatmap: 热力图叠加，可选透明度\n'
             'Feature Map: 各层特征图格子排列'),
            ('Run',
             'Start 开始推理，检测框实时叠加画面'),
            ('Results',
             '每帧 FPS + 各类别数量 + 推理耗时'),
            ('Save',
             '结果保存到 predict_output 目录'),
        ],
        },
    {
        'name': 'Review',
        'icon': '🔁',
        'brief': '审查数据集标注：直接在 dataset 目录读写 .txt，实时修正标注',
        'color': '#8b5cf6',
        'steps': [
            ('Source',
             '选 data split (train/val)，自动加载对应图片和 .txt 标注'),
            ('Edit',
             '选类 → 画框/拖动/调角。标注变更直接写入 .txt，无 JSON 中间态'),
            ('Navigate',
             '← → 翻图，支持跳转到指定序号'),
            ('Stats',
             '每张图标注实例数 + 按类分布占比条，实时更新'),
            ('Differences from Label',
             'Review 直接从 dataset 目录读写，无 session、无自动标注、无导出'),
        ],
    },
    {
        'name': 'Preprocess',
        'icon': '🎞️',
        'brief': '视频预处理：收集 → 重命名 → 缩放 → 抽帧',
        'color': '#14b8a6',
        'steps': [
            ('Collect',
             '将子文件夹中的视频集中到指定目录'),
            ('Rename',
             '按文件名排序重命名为 00.ext, 01.ext...'),
            ('Resize',
             'Letterbox 缩放至目标尺寸(默认 640)，黑边填充'),
            ('Extract',
             '按目标 FPS(推荐 1-2)均匀抽帧，JPEG 质量 95'),
            ('Naming',
             '输出: {视频源}-{编号}-{秒数:04d}.jpg'),
            ('Tips',
             '1-2 fps 避免相邻帧过相似，提高标注效率'),
        ],
    },
    {
        'name': 'Label',
        'icon': '🏷️',
        'brief': '手动标注 / 自动标注 / 审核导出全流程',
        'color': '#ec4899',
        'steps': [
            ('Source',
             '选图片目录(或 Preprocess 输出的帧目录)'),
            ('Manual',
             '选类 → 拖拽画框。单击框移动，四角调大小，右键删除'),
            ('Auto-Label',
             '选检测模型自动预测生成标注 → 逐张审核修正'),
            ('Navigate',
             '← → 或 Prev/Next 翻图，快捷键可设置'),
            ('Export',
             '按比例随机切分 Train/Val，生成 YOLO .txt + data.yaml'),
        ],
    },
    {
        'name': 'Distill',
        'icon': '🔬',
        'brief': '知识蒸馏：大模型(Teacher)教小模型(Student)',
        'color': '#f97316',
        'steps': [
            ('Concept',
             'Teacher 冻结，Student 同时学 One-Hot 标签 + Teacher Soft Prediction'),
            ('Alpha',
             '蒸馏权重 0.5(50% 检测 Loss + 50% 蒸馏 Loss)'),
            ('Teacher',
             '自动选 runs/ 下最新 best.pt，也可手动指定大模型'),
            ('Student',
             '选小模型 yolo11n/s，推理速度不变，精度接近 Teacher'),
            ('Params',
             'LR: 0.002(加速迁移) | Batch: 24 | Epochs: 150'),
            ('Output',
             'Student 权重 → runs/distill/，可直接 Predict/Export'),
            ('Benefit',
             'mAP50 提升 6-11%，推理速度保持 Student 水平'),
        ],
    },
    {
        'name': 'MIRO',
        'icon': '🤖',
        'brief': '内置 LLM 助手，提供 YOLO 领域知识和参数建议，仅回答 YOLO 相关问题',
        'color': '#3b82f6',
        'steps': [
            ('Chat',
             '在输入框提问，Agent 基于 YOLO 知识库回答，只回应 YOLO 相关话题'),
            ('Configuration',
             '点击 ⚙ 配置 API 地址、API Key 和模型名\n'
             '支持任意 OpenAI 兼容 API（如 DeepSeek、OpenAI、本地 Ollama）'),
            ('Test Connection',
             '配置对话框中点击「测试连接」验证 API 连通性，结果弹窗提示'),
            ('Clear Chat',
             '点击清空按钮重置对话历史'),
        ],
    },
    {
        'name': 'Tools',
        'icon': '🛠️',
        'brief': '视频导入 + 标注导出/导入 + 图片爬虫 + 模型导出 + 模型分析',
        'color': '#f59e0b',
        'steps': [
            ('Preproc Import',
             '将视频文件夹导入预处理目录，自动复制'),
            ('Label Export/Import',
             '选中子文件夹导出标注 .zip，或导入 .zip/.rar/.7z 文件'),
            ('Image Crawler',
             '输入关键词 → 选保存目录 → Start，爬取百度图片'),
            ('Model Export',
             '选 .pt 权重 → 选择导出格式(ONNX/TensorRT/NCNN 等)\n'
             '配置 ImgSz、FP16、INT8、NMS → Export\n'
             '输出保存到 export 目录'),
            ('Model Analysis',
             '选 .pt 模型 → 选 Split(val/test) → 选 Conf 阈值\n'
             '自动推理并输出 TP/FP/FN + 按类 P/R/F1 + F1-Confidence 曲线\n'
             '结果保存到 export_dir'),
        ],
    },
    {
        'name': 'Settings',
        'icon': '⚙️',
        'brief': '工作目录配置 + 类别名编辑 + 主题/快捷键',
        'color': '#666',
        'steps': [
            ('Directories',
             '配置各模块工作目录(train_output / dataset_dir / label_dir 等)'),
            ('Init Wizard',
             '选根目录自动创建完整文件夹结构'),
            ('Classes',
             '编辑类别名称，每行一个，顺序决定索引(第1行=0)'),
            ('Theme',
             '切换 Dark / Light 主题，立即生效'),
            ('Shortcuts',
             '自定义键盘快捷键映射'),
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
    brief_lbl.setStyleSheet(f"font-size:14px;color:{TEXT2};border:none;font-weight:500;")
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
        bullet.setStyleSheet(f"font-size:14px;color:{color};border:none;font-weight:700;")
        bullet.setFixedWidth(14)
        sh.addWidget(bullet)
        sn = QLabel(step_name)
        sn.setStyleSheet(f"font-size:14px;font-weight:600;color:{TEXT};border:none;")
        sh.addWidget(sn)
        sh.addStretch()
        wl.addLayout(sh)

        # 步骤描述
        sd = QLabel(step_desc)
        sd.setStyleSheet(f"font-size:13px;color:{TEXT3};border:none;padding-left:20px;")
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
            '界面采用 WeChat 风格侧边栏导航 + QStackedWidget 多标签页布局，左侧固定 110px 窄边栏，右侧内容自适应。支持 Training / Predict / Preprocess / Label / Review / Distill / AI Agent / Tools / Settings 共 9 个功能模块 + Guide 使用指引。',
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
            ('Output', 'runs/train/ · runs/distill/ · predict_output/ · export_dir/'),
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
