# =============================================================================
# YOLO Training Studio — 引导页数据
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# =============================================================================
"""引导页 — 配置数据，与 UI 分离"""

# ── 各标签页指引数据 ──
TAB_GUIDES = [
    {
        'name': 'Training',
        'icon': '🎯',
        'brief': '配置超参数并启动 YOLO 模型训练，实时查看 Loss / mAP 曲线和日志',
        'color': '#07C160',
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
        'name': 'Predict',
        'icon': '👁️',
        'brief': '对单张图片或视频进行实时目标检测推理',
        'color': '#f59e0b',
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
            ('Tips', '推荐采样率 1-2 fps 以避免时序相邻帧过于相似'),
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
            ('Save Progress', '标注自动保存到内存，切换图片时持久化，避免意外丢失'),
            ('Export Dataset', '按比例随机切分 Train/Val → 生成 YOLO 格式 .txt 标注文件和 data.yaml'),
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
        'color': '#ef4444',
        'steps': [
            ('Select Model', '选择需要导出的 .pt 文件（best.pt 或蒸馏输出）'),
            ('Choose Format', 'ONNX（通用）/ TorchScript / NCNN / OpenVINO / TensorRT / TFLite / CoreML'),
            ('Set Options', 'ImgSz（默认 640）、Half（FP16 缩小体积）、NMS（内嵌非极大抑制）'),
            ('Export', '点击 Export 按钮，转换进度显示在控制台'),
            ('Output', '导出文件保存到模型同级目录，文件名带格式后缀'),
        ],
    },
    {
        'name': 'Agent',
        'icon': '🤖',
        'brief': 'AI 助手 — 对话式 YOLO 专家，解答训练/推理/调参等问题',
        'color': '#3b82f6',
        'steps': [
            ('Configure', '点击 ⚙ 配置 API 地址和 Key（支持 OpenAI 兼容接口）'),
            ('Ask', '在聊天框输入 YOLO 相关问题，支持流式输出'),
            ('Topics', '训练调参、模型选型、数据增强、错误排查、功能指引'),
        ],
    },
    {
        'name': 'Settings',
        'icon': '⚙️',
        'brief': '类别管理、快捷键、主题切换、工作目录配置',
        'color': '#666',
        'steps': [
            ('Classes', '编辑类别名称列表，保存后全局自动更新'),
            ('Shortcuts', '自定义快捷键映射，立即生效'),
            ('Theme', '黑白主题一键切换'),
            ('Directories', '配置 7 个工作目录，点击 Init 自动创建目录结构'),
        ],
    },
]
