"""make eval: so sánh dense và hybrid ở 2 tầng.

Tầng 1 (retrieval): Hit@4 và MRR, tức evidence có nằm trong 4 chunk đầu không.
Tầng 2 (agent): từng câu của tests/testset.jsonl đúng hay sai, và trích dẫn có đúng chunk không.
Kết quả: results/<engine>-<giờ>/summary.json; mỗi lượt chạy có 1 file log trong logs/.
"""
import json
import tempfile
import time
from pathlib import Path

from .agent import run_agent
from .config import MAX_STEPS, ROOT
from .logging_utils import RunLog
from .mcp_client import connect
from .retrieval import Retriever

MODES = ("dense", "hybrid")
K = 4


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def load_testset() -> list[dict]:
    path = ROOT / "tests" / "testset.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def contains_all(answer: str, needles: list[str]) -> bool:
    """Mỗi phần tử của needles phải có trong câu trả lời (dấu | nghĩa là một trong các cách viết)."""
    return all(any(norm(alt) in norm(answer) for alt in needle.split("|")) for needle in needles)


def retrieval_eval(retriever: Retriever, items: list[dict], mode: str) -> dict:
    graded = [it for it in items if not it["no_answer"] and it.get("evidence")]
    hits, rr = 0, 0.0
    for it in graded:
        ranked = retriever.search(it["question"], mode, K)
        rank = next((i for i, (c, _) in enumerate(ranked, 1) if norm(it["evidence"]) in norm(c.text)), None)
        if rank:
            hits += 1
            rr += 1 / rank
    n = len(graded)
    return {"hit_at_4": round(hits / n, 3), "mrr": round(rr / n, 3), "n": n}


def judge(it: dict, answer, obs, chunk_text: dict[str, str]) -> tuple[bool, bool | None]:
    """Trả (đạt hay không, trích dẫn đúng hay không). Trích dẫn là None nếu câu không cần."""
    wrote = any(o["tool"] == "record_activity" and not o["result"].startswith("Error:") for o in obs)
    if it["no_answer"]:
        return answer.status == "no_answer", None
    if it.get("expect_refusal"):
        return answer.status == "refused" and not wrote and contains_all(answer.text, it["must_contain"]), None
    if it.get("expect_write") is False and wrote:
        return False, None
    if it.get("expect_write"):
        return wrote and answer.verified is True and contains_all(answer.text, it["must_contain"]), None
    passed = answer.status == "answered" and contains_all(answer.text, it["must_contain"])
    cite_ok = None
    if it.get("evidence"):
        cite_ok = any(norm(it["evidence"]) in norm(chunk_text.get(s.chunk_id, "")) for s in answer.sources)
    return passed, cite_ok


async def run_eval(engine: str, repeats: int):
    items = load_testset()
    retriever = Retriever()
    chunk_text = {c.source.chunk_id: c.text for c in retriever.chunks}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / f"{engine}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"engine": engine, "embedding_backend": retriever.backend, "max_steps": MAX_STEPS,
               "n_questions": len(items), "repeats": repeats, "retrieval": {}, "agent": {}}

    for mode in MODES:
        summary["retrieval"][mode] = retrieval_eval(retriever, items, mode)
        print(f"[retrieval/{mode}]", summary["retrieval"][mode])

    for mode in MODES:
        results = []  # mỗi phần tử: một câu ở một lượt chạy
        for rep in range(1, repeats + 1):
            log = RunLog(ROOT / "logs" / f"eval-{engine}-{mode}-r{rep}-{stamp}.jsonl")
            log("RUN_START", engine=engine, retrieval=mode, repeat=rep, max_steps=MAX_STEPS,
                embedding=retriever.backend, n_questions=len(items))
            with tempfile.TemporaryDirectory() as tmp:  # DB tạm để eval không đụng DB demo
                async with connect(Path(tmp) / "eval.sqlite3") as (session, catalog):
                    for it in items:
                        log.question_id = it["id"]
                        answer, usage, obs = await run_agent(session, catalog, it["question"], mode, engine, log)
                        passed, cite_ok = judge(it, answer, obs, chunk_text)
                        results.append({"id": it["id"], "no_answer": it["no_answer"], "passed": passed,
                                        "cite_ok": cite_ok, "steps": answer.steps, "calls": usage.calls,
                                        "prompt_tokens": usage.prompt_tokens,
                                        "completion_tokens": usage.completion_tokens})
                        print(f"  {mode} r{rep} {it['id']} {'PASS' if passed else 'FAIL'} ({answer.status})")
            print(f"  log: {log.path.relative_to(ROOT)}")

        answerable = [r for r in results if not r["no_answer"]]
        refusals = [r for r in results if r["no_answer"]]
        cited = [r for r in answerable if r["cite_ok"] is not None]
        summary["agent"][mode] = {
            "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3),
            "answerable_pass_rate": round(sum(r["passed"] for r in answerable) / len(answerable), 3),
            "no_answer_refusal_rate": round(sum(r["passed"] for r in refusals) / len(refusals), 3),
            "citation_ok_rate": round(sum(r["cite_ok"] for r in cited) / len(cited), 3),
            "avg_steps": round(sum(r["steps"] for r in results) / len(results), 2),
            "llm_calls": sum(r["calls"] for r in results),
            "prompt_tokens": sum(r["prompt_tokens"] for r in results),
            "completion_tokens": sum(r["completion_tokens"] for r in results),
            "failed_ids": sorted({r["id"] for r in results if not r["passed"]}),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["agent"], ensure_ascii=False, indent=2))
    print(f"Kết quả: {out_dir.relative_to(ROOT)}/summary.json")
