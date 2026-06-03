# =============================================================================
# YOLO Training Studio — AI Agent 配置管理 & API 调用
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# =============================================================================
"""AI Agent — 大模型接口配置持久化 + OpenAI 兼容 API 调用"""

import json, urllib.request, urllib.error
from pathlib import Path

# ── 配置路径 ──
_CONFIG_FILE = Path(__file__).resolve().parent.parent.parent / 'config' / 'agent_config.json'

# ── 默认配置 ──
DEFAULT_CONFIG = {
    'base_url': 'https://api.openai.com/v1',
    'api_key': '',
    'model': 'gpt-4',
}

# ── YOLO 专家系统提示词 ──
SYSTEM_PROMPT = """你是 MIRO，一个精通 YOLO 全系列目标检测的 AI 助手。你所在的应用是 YOLO Training Studio，一个基于 PyQt5 的桌面端训练与部署平台。

你的能力与知识范围：

1. YOLO 系列原理
   - 了解从 v1 到 v11 的架构演进，包括 DarkNet / CSPNet / ELAN 等 backbone 变化
   - 熟悉 Anchor-Based 与 Anchor-Free 检测头区别、Decoupled Head、Task-Specific Head
   - 理解 loss 构成：Bbox Loss（CIoU / WIoU / MPDIoU）、Class Loss（BCE / Softmax）、DFL Loss
   - 熟悉 NMS / Soft-NMS 原理及 IoU 阈值影响

2. 训练调参建议
   - 根据用户的数据集规模（几百到几十万张）、场景需求（精度优先/速度优先/小目标），给出合理的超参数建议
   - epochs: 小数据集(<500张) 300-500，中等(500-5000) 200-300，大数据集 100-200
   - batch: 越大越稳但吃显存，yolo11n/b 可 48-64，yolo11s/m 32，yolo11l/x 16-24
   - lr0(初始学习率): AdamW 推荐 0.0005-0.002，SGD 0.01，注意力模块注入时应减半
   - lrf(最终LR=lr0*lrf): 默认 0.01
   - optimizer: AdamW(推荐) / Adam / SGD
   - momentum: 默认 0.937
   - warmup_epochs: 预热轮数，CBAM 推荐 10-15
   - warmup_momentum: 预热期动量 0.8
   - weight_decay: L2 权重衰减 0.0005
   - patience: 早停轮数，mAP 连续 N 轮不升即停，默认 40
   - dropout: 0.0(关) / 0.1-0.3(过拟合时)
   - cls_pw: <1 降误报，>1 提召回，默认 0.75
   - iou: 训练 NMS 阈值 0.7-0.8
   - close_mosaic: 最后 N 轮关 Mosaic，15-30
   - multi_scale: 每轮随机缩放 ±25%，增泛化
   - cache: 数据集缓存到 RAM 加速加载
   - amp: 混合精度加速，减 VRAM ~30%，默认开启
   - 几何增强: degrees(旋转0)、translate(平移0.15)、scale(缩放0.6)、shear(剪切0)、perspective(透视0)
   - 翻转: flip_lr(水平0.5)、flipud(垂直0.0，摔倒检测禁 flipud)
   - HSV: hsv_h(色调0.015)、hsv_s(饱和度0.7)、hsv_v(明度0.4)
   - Mosaic: 默认 1.0，小数据集可开
   - MixUp: 默认 0.2
   - Copy-Paste: 默认 0.0，小目标数据可开 0.3-0.5
   - 数据增强：Mosaic/MixUp/CopyPaste 对小数据集提升大但训练初期可关，CloseMosaic 最后 10-30 轮

3. 项目功能指引
   - Training：配置超参数启动训练，支持 LoRA 低秩适配、7 种注意力模块（SE/CBAM/CA/ECA/SimAM/EMA/GAM）、20 套参数预设
   - Predict：图片/视频/摄像头推理，支持 Detection / Heatmap / Feature Map 三种可视化
   - Preprocess：视频导入、抽帧、缩放
   - Label：手动标注 + 自动标注 + 导出 YOLO 格式
   - Review：审查现有标注，直接读写 dataset 目录的 .txt
   - Distill：知识蒸馏，Teacher 教 Student
   - AI Agent：当前对话机器人
   - Tools：视频导入、标注导入导出、百度爬虫、模型导出（ONNX/TensorRT/NCNN）、模型分析（F1 曲线）
   - Settings：工作目录、类别名、主题、快捷键

4. 问题排查
   - Loss 不下降：lr 太小 / 数据增强过强 / batch 太小 / 梯度爆炸
   - mAP 低但 loss 低：过拟合 / 验证集分布不一致 / NMS 阈值不合适
   - 过拟合：减少 epochs / 增强数据增强 / 加 dropout / 减小模型
   - 欠拟合：增大模型 / 加注意力 / 加 epochs / 检查 lr
   - CUDA OOM：降 batch / 降 imgsz / 开 AMP / 用梯度累积
   - 小目标检测差：加 P2 小目标检测头 / 提高 imgsz / mosaic 增强 / 避免过多下采样

回答要求：
- 使用中文回答，关键术语保留英文（如 mAP, NMS, IoU, Loss, batch）
- 简洁务实，直击要点，每个建议附上理由
- 给出具体数值范围而非模糊建议
- 不要使用 ## ** ``` 等格式符号，用纯文字表达
- 代码或命令用一行文字说明即可，不要使用代码块
- 只回答与 YOLO 目标检测、训练调参、模型部署、数据标注直接相关的问题，遇到不相关的问题仅输出「抱歉，您说的问题不相关，不作回答」，不多解释"""


# ═══════════════════════════════════════════════
# 配置管理
# ═══════════════════════════════════════════════

def load_config() -> dict:
    """读取 Agent API 配置，不存在则返回默认值"""
    if not _CONFIG_FILE.exists():
        return dict(DEFAULT_CONFIG)
    try:
        data = json.loads(_CONFIG_FILE.read_text(encoding='utf-8'))
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        return merged
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    """保存 Agent API 配置到 JSON 文件"""
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    _CONFIG_FILE.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


# ═══════════════════════════════════════════════
# OpenAI 兼容 API 调用
# ═══════════════════════════════════════════════

def _build_request(messages: list, cfg: dict, stream: bool = False):
    """构建 HTTP POST 请求"""
    api_key = cfg.get('api_key', '').strip()
    if not api_key:
        raise ValueError('API Key 未配置')
    base_url = cfg.get('base_url', '').strip().rstrip('/')
    if not base_url:
        raise ValueError('API 地址未配置')

    payload = json.dumps({
        'model': cfg.get('model', 'gpt-4'),
        'messages': messages,
        'temperature': 0.7,
        'max_tokens': 16384,
        'stream': stream,
    }).encode('utf-8')

    req = urllib.request.Request(
        f'{base_url}/chat/completions',
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    return req


def send_message(messages: list, cfg: dict) -> str:
    """调用 OpenAI 兼容 API，返回模型回复文本（非流式）。"""
    try:
        with urllib.request.urlopen(_build_request(messages, cfg), timeout=60) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        detail = ''
        try:
            detail = e.read().decode('utf-8')
        except Exception:
            detail = str(e)
        raise RuntimeError(f'API 请求失败 ({e.code}): {detail}')
    except urllib.error.URLError as e:
        raise ConnectionError(f'网络连接失败: {e.reason}')
    except json.JSONDecodeError as e:
        raise RuntimeError(f'API 返回格式异常: {e}')

    choices = body.get('choices', [])
    if not choices:
        raise RuntimeError(f'API 返回异常（无 choices）: {json.dumps(body, ensure_ascii=False)[:200]}')
    content = choices[0].get('message', {}).get('content', '')
    return content if content else '(模型未返回有效内容)'


def send_message_stream(messages: list, cfg: dict, on_chunk, on_done, on_error):
    """
    流式调用 OpenAI 兼容 API，通过回调返回结果。
    
    Args:
        messages: 对话历史
        cfg: API 配置
        on_chunk: 收到文本块时回调 on_chunk(text: str)
        on_done: 完成时回调 on_done(full_text: str)
        on_error: 出错时回调 on_error(err_msg: str)
    """
    try:
        req = _build_request(messages, cfg, stream=True)
        resp = urllib.request.urlopen(req, timeout=120)
        full = []
        buffer = ''
        while True:
            chunk = resp.read(1).decode('utf-8', errors='replace')
            if not chunk:
                break
            buffer += chunk
            if buffer.endswith('\n'):
                line = buffer.strip()
                buffer = ''
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        ev = json.loads(data)
                        delta = ev.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            full.append(content)
                            on_chunk(content)
                    except json.JSONDecodeError:
                        pass
        resp.close()
        full_text = ''.join(full)
        on_done(full_text if full_text else '(模型未返回有效内容)')
    except urllib.error.HTTPError as e:
        detail = ''
        try:
            detail = e.read().decode('utf-8')
        except Exception:
            detail = str(e)
        on_error(f'API 请求失败 ({e.code}): {detail[:200]}')
    except urllib.error.URLError as e:
        on_error(f'网络连接失败: {e.reason}')
    except Exception as e:
        on_error(str(e))
