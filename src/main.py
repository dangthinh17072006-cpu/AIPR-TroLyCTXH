"""python -m src.main demo|chat|eval   (hoặc make demo / make chat / make eval)"""
import argparse
import asyncio
import os
import time

from .agent import run_agent
from .config import MAX_STEPS, ROOT
from .logging_utils import RunLog
from .mcp_client import connect

DEMO_QUESTIONS = [
    "Trực tiếp hiến máu tình nguyện một lần, mã 001202, được bao nhiêu ngày CTXH?",
    "Sinh viên được làm đơn khiếu nại kết quả CTXH trong thời hạn bao lâu sau khi công bố kết quả?",
    "Ghi nhận giúp tôi: sinh viên 11110001 hiến máu trực tiếp mã 001202, 5 điểm, ngày 2026-10-10.",
]


def pick_engine(arg: str | None) -> str:
    engine = arg or os.getenv("AIPR_MODE", "local")
    if engine not in ("local", "api"):
        raise SystemExit("--engine phải là local hoặc api")
    if engine == "api" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("API mode cần OPENAI_API_KEY trong .env (copy từ .env.example). Không tự chuyển về local.")
    return engine


def show(question: str, answer, usage, obs):
    print(f"\n❯ {question}")
    for o in obs:
        args = ", ".join(f"{k}={v!r}" for k, v in o["arguments"].items())
        print(f"  [{o['step']}] {o['tool']}({args})"[:160])
    verified = f" | verified={answer.verified}" if answer.verified is not None else ""
    print(f"  → {answer.status}{verified}")
    print("  " + answer.text.replace("\n", "\n  "))
    print(f"  usage: calls={usage.calls} prompt={usage.prompt_tokens} completion={usage.completion_tokens}")


async def demo(engine: str, mode: str, questions: list[str], interactive: bool):
    kind = "chat" if interactive else "demo"
    log = RunLog(ROOT / "logs" / f"{kind}-{engine}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    log("RUN_START", engine=engine, retrieval=mode, max_steps=MAX_STEPS, kind=kind)
    async with connect() as (session, catalog):
        tool_names = [t["name"] for t in catalog]
        log("TOOLS_DISCOVERED", tools=tool_names)
        print(f"Engine: {engine} | retrieval: {mode} | tool từ MCP server: {', '.join(tool_names)}")
        if engine == "local":
            print("(engine local là bộ chọn theo luật, KHÔNG phải LLM; đặt AIPR_MODE=api + key để dùng LLM)")

        if interactive:
            print("Gõ câu hỏi, 'exit' để thoát.")
            n = 0
            while True:
                try:
                    question = input("\nCâu hỏi> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question.lower() in ("exit", "quit", ""):
                    break
                n += 1
                log.question_id = f"chat{n}"
                show(question, *await run_agent(session, catalog, question, mode, engine, log))
        else:
            for i, question in enumerate(questions, 1):
                log.question_id = f"demo{i}"
                show(question, *await run_agent(session, catalog, question, mode, engine, log))
    print(f"\nLog: {log.path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(prog="python -m src.main")
    parser.add_argument("action", choices=["demo", "chat", "eval"])
    parser.add_argument("--engine", choices=["local", "api"])
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--question", help="chạy demo với một câu hỏi tùy ý")
    args = parser.parse_args()
    engine = pick_engine(args.engine)

    if args.action == "eval":
        from .evaluate import run_eval
        asyncio.run(run_eval(engine, args.repeats or (3 if engine == "api" else 1)))
    else:
        questions = [args.question] if args.question else DEMO_QUESTIONS
        asyncio.run(demo(engine, args.retrieval, questions, args.action == "chat"))


if __name__ == "__main__":
    main()
