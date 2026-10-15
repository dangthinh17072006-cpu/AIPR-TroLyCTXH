"""Gọi API (yêu cầu 2.4): có timeout, retry, log lỗi HTTP (không log API key); prompt nằm ở templates/agent.txt."""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import LLM_RETRIES, LLM_TIMEOUT_S, ROOT
from .models import Usage

PROMPT_FILE = ROOT / "templates" / "agent.txt"


def call_llm(payload: dict, usage: Usage, log) -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("API mode cần OPENAI_API_KEY trong .env (xem .env.example)")
    url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "temperature": 0,
        "messages": [{"role": "system", "content": PROMPT_FILE.read_text(encoding="utf-8")},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "response_format": {"type": "json_object"},
    }).encode()

    for attempt in range(1, LLM_RETRIES + 1):
        usage.calls += 1  # request lỗi cũng tính
        request = Request(url, data=body, headers={"Authorization": "Bearer " + key,
                                                   "Content-Type": "application/json"})
        start = time.time()
        try:
            with urlopen(request, timeout=LLM_TIMEOUT_S) as response:
                data = json.load(response)
            used = data.get("usage") or {}
            usage.prompt_tokens += used.get("prompt_tokens", 0)
            usage.completion_tokens += used.get("completion_tokens", 0)
            log("LLM_CALL", attempt=attempt, latency_s=round(time.time() - start, 3),
                prompt_tokens=used.get("prompt_tokens"), completion_tokens=used.get("completion_tokens"))
            return data["choices"][0]["message"]["content"]
        except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            status = exc.code if isinstance(exc, HTTPError) else None
            log("HTTP_ERROR", attempt=attempt, status=status, error=type(exc).__name__)
            # lỗi 4xx (sai key, sai model...) thì thử lại cũng vô ích, trừ 408/409/429
            if status is not None and status < 500 and status not in (408, 409, 429):
                raise RuntimeError(f"HTTP {status}: kiểm tra key/model/base_url") from None
            if attempt == LLM_RETRIES:
                raise RuntimeError(f"API không phản hồi sau {LLM_RETRIES} lần thử") from None
            time.sleep(2 ** (attempt - 1))
