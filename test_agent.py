# =============================================================================
# YOLO Training Studio — AI Agent 连接测试脚本
# 独立于项目，直接测试大模型 API 连通性
# =============================================================================
"""用法: D:\Anaconda3\envs\Projects\python.exe D:\test_agent.py"""

import json
import urllib.request
import urllib.error
import sys
from pathlib import Path

# ── 配置路径 ──
CONFIG_FILE = Path(r'D:\Projects\YOLO\main\core\agent\agent_config.json')
TIMEOUT = 30

# GBK 终端无法显示 emoji，改用纯文本标记
OK = '[OK]'
FAIL = '[FAIL]'
WARN = '[WARN]'

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        print(f'\n{FAIL} 配置文件不存在: {CONFIG_FILE}')
        sys.exit(1)
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
        return data
    except Exception as e:
        print(f'[错误] 读取配置文件失败: {e}')
        sys.exit(1)


def mask_key(key: str) -> str:
    """隐藏 API Key 中间部分"""
    if len(key) <= 8:
        return key[:4] + '****'
    return key[:6] + '****' + key[-4:]


def test_connection(cfg: dict):
    print('=' * 60)
    print('  AI Agent 连接测试')
    print('=' * 60)

    # ── 显示配置信息 ──
    base_url = cfg.get('base_url', '').strip().rstrip('/')
    api_key = cfg.get('api_key', '').strip()
    model = cfg.get('model', '')
    temperature = cfg.get('temperature', 0.7)
    max_tokens = cfg.get('max_tokens', 4096)

    print(f'  API 地址:    {base_url}')
    print(f'  API Key:     {mask_key(api_key)}')
    print(f'  模型:        {model}')
    print(f'  温度:        {temperature}')
    print(f'  最大 Token:  {max_tokens}')

    if not api_key:
        print(f"\n{FAIL} API Key 为空")
        return
    if not base_url:
        print(f"\n{FAIL} API 地址为空")
        return

    # ── 构建请求 ──
    payload = json.dumps({
        'model': model,
        'messages': [
            {'role': 'user', 'content': 'Hello, reply OK to confirm connection.'}
        ],
        'temperature': temperature,
        'max_tokens': 64,
        'stream': False,
    }).encode('utf-8')

    url = f'{base_url}/chat/completions'
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )

    print(f'\n  请求 URL:    {url}')
    print(f'  请求体:      {payload.decode("utf-8")}')
    print(f'  超时:        {TIMEOUT}s')
    print('  ── 正在发送请求...')

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            status = resp.status
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        detail = ''
        try:
            detail = e.read().decode('utf-8')
        except Exception:
            detail = str(e)
        print(f"\n{FAIL} HTTP {e.code}")
        print(f'  详情: {detail[:500]}')
        return
    except urllib.error.URLError as e:
        print(f"\n{FAIL} 网络连接失败")
        print(f'  详情: {e.reason}')
        return
    except Exception as e:
        print(f"\n{FAIL} 未知错误")
        print(f'  详情: {str(e)[:500]}')
        return

    # ── 解析响应 ──
    print(f'\n  HTTP 状态:   {status}')
    print(f'  响应体:      {json.dumps(body, indent=2, ensure_ascii=False)[:800]}')

    choices = body.get('choices', [])
    if not choices:
        print(f"\n{FAIL} API 返回异常（无 choices）")
        return

    content = choices[0].get('message', {}).get('content', '')
    finish_reason = choices[0].get('finish_reason', 'unknown')

    print(f'\n  回复内容:    {content[:200]}')
    print(f'  结束原因:    {finish_reason}')

    if content and 'error' not in content.lower() and 'invalid' not in content.lower():
        print(f"\n{OK} 连接成功！")
    else:
        print(f"\n{WARN} 返回了内容，但可能存在问题，请检查上面的详情")


if __name__ == '__main__':
    cfg = load_config()
    test_connection(cfg)
