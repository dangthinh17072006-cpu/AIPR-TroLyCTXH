import json
from datetime import datetime, timezone
from pathlib import Path


class RunLog:
    """Ghi log JSONL: mỗi dòng có timestamp, question_id, event và các trường kèm theo."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.question_id = ""

    def __call__(self, event: str, **fields):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "question_id": self.question_id, "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
