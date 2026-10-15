import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_STEPS = 8
MCP_TIMEOUT_S = 15
LLM_TIMEOUT_S = 30
LLM_RETRIES = 3


def load_env():
    """Đọc file .env (nếu có). Biến môi trường đã đặt sẵn thì giữ nguyên."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
