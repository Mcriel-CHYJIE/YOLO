# =============================================================================
# YOLO Training Studio — AI Agent 配置管理 & API 调用
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# =============================================================================
"""AI Agent — 大模型接口配置持久化 + OpenAI 兼容 API 调用"""

import json, urllib.request, urllib.error
from pathlib import Path

# ── 配置路径 ──
_CONFIG_DIR = Path(__file__).resolve().parent
_CONFIG_FILE = _CONFIG_DIR / 'agent_config.json'

# ── 默认配置 ──
DEFAULT_CONFIG = {
    'base_url': 'https://api.openai.com/v1',
    'api_key': '',
    'model': 'gpt-4',
    'temperature': 0.7,
    'max_tokens': 4096,
}

# ── YOLO 专家系统提示词 ──
SYSTEM_PROMPT = """你是 YOLO 训练大师，一个精通 YOLO 系列目标检测的 AI 助手。

## 你的能力
1. **YOLO 原理讲解** — 从 v1 到 v11 的架构演进、核心思想（网格划分、锚框、NMS、损失函数）
2. **项目功能指引** — 帮助用户理解当前 YOLO Training Studio 各页面的功能和操作流程
3. **训练调参建议** — 根据用户的数据集规模、场景需求，给出合理的超参数建议（epochs, batch, lr, optimizer, 数据增强等）
4. **需求解读** — 把用户的工作需求转化为具体的技术方案和训练策略
5. **问题排查** — 分析训练曲线、指标异常、过拟合/欠拟合、CUDA 报错等常见问题

## 回答风格
- 专业但易懂：用中文回答，关键术语保留英文
- 简洁务实：直击要点，避免空洞的套话
- 给出具体数值建议时附上理由
- 涉及代码/命令时用适当的格式展示"""


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
        'temperature': cfg.get('temperature', 0.7),
        'max_tokens': cfg.get('max_tokens', 4096),
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
