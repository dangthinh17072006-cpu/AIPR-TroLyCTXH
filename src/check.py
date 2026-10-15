"""Tự kiểm theo bảng yêu cầu của đề (02-Yeu-cau-Thuc-hien). Chạy: make check.
Chỉ ĐỌC repo, không sửa gì. Mục nào FAIL thì sửa trước khi nộp."""
import json
import re
import sys
from pathlib import Path
from .config import ROOT, MAX_STEPS


def jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def main() -> int:
    rows = []

    def check(name, ok, detail=''):
        rows.append((name, bool(ok), detail))

    src = ''.join(p.read_text(encoding='utf-8') for p in (ROOT / 'src').glob('*.py'))
    check('Python 3.11+', sys.version_info >= (3, 11), sys.version.split()[0])
    check('2.3 dataclass/TypedDict', '@dataclass' in src and 'TypedDict' in src)
    check('2.4 prompt tách file', (ROOT / 'templates/agent.txt').exists())
    check('2.4 retry + timeout + log HTTP', 'timeout=' in src and "'HTTP_ERROR'" in src and 'range(3)' in src)
    check('2.5 dense + hybrid BM25', "'dense'" in src and "'hybrid'" in src and 'BM25Okapi' in src)
    check('2.5 tái dùng ex2_1_hybrid_search.py',
          (ROOT / 'course/ex2_1_hybrid_search.py').exists()
          and 'ex2_1_hybrid_search' in (ROOT / 'src/retrieval.py').read_text(encoding='utf-8'))
    check('2.5 chunking có module riêng', (ROOT / 'src/chunking.py').exists())
    check('2.6 MAX_STEPS', MAX_STEPS >= 1 and 'MAX_STEPS' in src, f'MAX_STEPS={MAX_STEPS}')
    check('2.6 tool lỗi trả "Error:"', "return f'Error:" in src or "return 'Error:" in src)
    check('2.7 verify sau ghi', 'get_registration' in src and 'VERIFY' in src)
    check('2.8 MCP stdio, không print trong server',
          "transport='stdio'" in src and 'print(' not in (ROOT / 'src/server.py').read_text(encoding='utf-8'))
    check('2.8 client discover động', 'list_tools' in src)
    check('Không lộ key (.env bị ignore, có .env.example)',
          '.env' in (ROOT / '.gitignore').read_text(encoding='utf-8') and (ROOT / '.env.example').exists()
          and not (ROOT / '.env').exists())
    cases = jsonl(ROOT / 'tests/testset.jsonl')
    check('testset >= 12 câu', len(cases) >= 12, f'{len(cases)} câu')
    check('testset >= 4 no_answer', sum(c['no_answer'] for c in cases) >= 4,
          f"{sum(c['no_answer'] for c in cases)} câu")
    logs = sorted((ROOT / 'logs').glob('*.jsonl'))
    check('logs/ >= 3 file jsonl', len(logs) >= 3, f'{len(logs)} file')
    good = 0
    for p in logs:
        ev = jsonl(p)
        names = {e['event'] for e in ev}
        if {'QUESTION', 'TOOL', 'USAGE'} <= names and all('timestamp' in e for e in ev):
            good += 1
    check('log có timestamp + TOOL + USAGE', logs and good == len(logs), f'{good}/{len(logs)} file')
    check('log LLM thật (engine=api)', any(p.name.startswith('api-') for p in logs),
          'chưa có — cần chạy AIPR_MODE=api')
    docs = json.loads((ROOT / 'data/documents.json').read_text(encoding='utf-8'))
    check('metadata doc_id/title/date/version/owner',
          all({'doc_id', 'title', 'date', 'version', 'owner'} <= set(d) for d in docs))
    for f in ['report.docx', 'README.md', 'ai-usage.md', 'Makefile', '.env.example']:
        check(f'có {f}', (ROOT / f).exists())
    check('thư mục .git', (ROOT / '.git').exists())
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    check('README có make setup && make demo, make eval',
          'make setup && make demo' in readme and 'make eval' in readme)
    check('README giải thích chunking', re.search(r'chunking', readme, re.I) is not None)
    check('ai-usage.md khai báo theo file', 'src/agent.py' in (ROOT / 'ai-usage.md').read_text(encoding='utf-8'))
    width = max(len(r[0]) for r in rows)
    for name, ok, detail in rows:
        print(f"[{'OK' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    failed = [r for r in rows if not r[1]]
    print(f'\n{len(rows) - len(failed)}/{len(rows)} mục đạt.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
