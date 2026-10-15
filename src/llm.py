"""API adapter nhỏ dùng urllib: timeout, retry, log lỗi, không log API key."""
import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from .config import ROOT
from .models import Usage


def call_llm(payload: dict, usage: Usage, log) -> str:
    key = os.getenv('OPENAI_API_KEY', '')
    if not key:
        raise RuntimeError('API mode cần OPENAI_API_KEY trong .env')
    body = json.dumps({'model': os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'),
        'messages': [{'role': 'system', 'content': (ROOT/'templates/agent.txt').read_text(encoding='utf-8')},
                     {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
        'response_format': {'type': 'json_object'}}).encode()
    url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/') + '/chat/completions'
    for attempt in range(3):
        usage.calls += 1  # Số request đã thử; request lỗi cũng được đếm.
        req = Request(url, data=body, headers={'Authorization': 'Bearer '+key,
                                             'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=30) as response:
                data = json.load(response)
            u = data.get('usage')
            if not u or 'prompt_tokens' not in u or 'completion_tokens' not in u:
                usage.usage_complete = False
            else:
                usage.prompt_tokens += u['prompt_tokens']
                usage.completion_tokens += u['completion_tokens']
                usage.max_prompt = max(usage.max_prompt, u['prompt_tokens'])
            return data['choices'][0]['message']['content']
        except (HTTPError, URLError, TimeoutError) as exc:
            usage.usage_complete = False  # Không biết số token tính phí ở request lỗi.
            code = exc.code if isinstance(exc, HTTPError) else None
            log('HTTP_ERROR', attempt=attempt+1, status=code, error=type(exc).__name__)
            if code and code not in (408, 429) and code < 500:
                raise RuntimeError(f'HTTP {code}; kiểm tra key/model, xem log') from None
            if attempt == 2:
                raise RuntimeError('API không phản hồi sau 3 lần thử') from None
            time.sleep(2 ** attempt)
    raise RuntimeError('API unavailable')
