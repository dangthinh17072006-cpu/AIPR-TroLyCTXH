"""Vòng lặp agent tự viết (yêu cầu 2.2, 2.6, 2.7).

- Tối đa MAX_STEPS bước; tool lỗi trả chuỗi 'Error: ...' chứ không raise.
- parse_action không làm chết chương trình: JSON hỏng thì thành một observation lỗi.
- record_activity chỉ chạy khi người dùng yêu cầu ghi rõ ràng; sau khi ghi, agent TỰ gọi get_record để verify.
"""
import json
import re

from .activities import fmt
from .config import MAX_STEPS
from .llm import call_llm
from .mcp_client import call
from .models import Action, Answer, Observation, Source, Usage
from .planner import compose, explicit_write_request, plan

OBS_LIMIT = 3500  # số ký tự tối đa của kết quả tool đưa lại cho model


def parse_action(raw: str) -> Action | str:
    """Đọc JSON model trả về thành Action. Không hiểu được thì trả chuỗi 'Error: ...'."""
    if not isinstance(raw, str) or not raw.strip():
        return "Error: model trả về rỗng"
    start = raw.find("{")
    if start < 0:
        return "Error: không có JSON object trong phản hồi"
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[start:])  # bỏ qua chữ/``` thừa xung quanh
    except json.JSONDecodeError:
        return "Error: JSON không hợp lệ"
    if not isinstance(value, dict):
        return "Error: JSON phải là object"
    tool, args = value.get("tool"), value.get("arguments", {})
    if not isinstance(tool, str) or not tool:
        return "Error: thiếu trường 'tool' dạng chuỗi"
    if not isinstance(args, dict):
        return "Error: 'arguments' phải là object"
    return Action(tool, args)


def shorten(text: str) -> str:
    return text if len(text) <= OBS_LIMIT else text[:OBS_LIMIT] + "…[cắt]"


def same_record(written: dict, read_back: str) -> bool:
    """Bản ghi đọc lại có khớp bản vừa ghi không."""
    try:
        a, b = written["record"], json.loads(read_back)["record"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return all(a.get(k) == b.get(k) for k in ("record_id", "student_id", "activity_code", "points", "event_date"))


def answer_from_finish(args: dict, obs: list[Observation]) -> Answer | None:
    """Lấy câu trả lời từ tool 'finish'; bỏ các nguồn không hề xuất hiện trong kết quả tool."""
    text = args.get("answer")
    if not isinstance(text, str) or not text.strip():
        return None
    seen = {m for o in obs for m in re.findall(r"CTXH-[IV]+(?:-BANG)?:v\d+:\d+", o["result"])}
    sources = []
    for chunk_id in args.get("sources") or []:
        if chunk_id in seen:
            doc_id, version, _ = chunk_id.split(":")
            sources.append(Source(doc_id, chunk_id, int(version[1:])))
    status = args.get("status")
    if status not in ("answered", "no_answer", "refused"):
        status = "answered"
    return Answer(text.strip(), sources, status)


async def run_agent(session, catalog: list[dict], question: str, retrieval_mode: str, engine: str, log):
    usage = Usage()
    obs: list[Observation] = []
    names = {t["name"] for t in catalog}
    write_allowed = explicit_write_request(question)
    to_verify = None   # kết quả record_activity đang chờ verify
    verified = None
    answer = None
    log("QUESTION", text=question, retrieval_mode=retrieval_mode, engine=engine, max_steps=MAX_STEPS)

    for step in range(1, MAX_STEPS + 1):
        # 1) chọn hành động
        if to_verify:  # vừa ghi xong: bắt buộc đọc lại để verify
            record_id = to_verify["record"]["record_id"]
            action = Action("get_record", {"record_id": record_id})
            log("VERIFY_FORCED", step=step, record_id=record_id)
        elif engine == "local":
            action = plan(question, obs, retrieval_mode)
        else:
            history = [{k: o[k] for k in ("step", "tool", "arguments", "result")} for o in obs[-6:]]
            try:
                raw = call_llm({"question": question, "retrieval_mode": retrieval_mode,
                                "tools": catalog, "history": history}, usage, log)
            except RuntimeError as exc:
                log("LLM_FAILED", step=step, error=str(exc))
                return Answer(f"Không gọi được LLM: {exc}", status="error", steps=step), usage, obs
            action = parse_action(raw)
        log("ACTION", step=step, action=action.__dict__ if isinstance(action, Action) else action)

        # 2) thực hiện
        if isinstance(action, str):  # không đọc được JSON: báo lỗi lại cho model rồi đi tiếp
            log("ACTION_ERROR", step=step, error=action)
            obs.append({"step": step, "tool": "(parse)", "arguments": {}, "result": action})
            continue
        if action.tool == "finish":
            answer = answer_from_finish(action.arguments, obs)
            break
        if action.tool not in names:
            result = f"Error: tool '{action.tool}' không tồn tại; có: {sorted(names)}"
        elif action.tool == "record_activity" and not write_allowed:
            result = "Error: người dùng chưa yêu cầu ghi rõ ràng; không được thay đổi dữ liệu"
            log("GUARD_BLOCKED_WRITE", step=step)
        else:
            log("TOOL_CALL", step=step, tool=action.tool, arguments=action.arguments)
            result = await call(session, action.tool, action.arguments)
        result = shorten(result)
        log("TOOL_RESULT", step=step, tool=action.tool, result=result)
        obs.append({"step": step, "tool": action.tool, "arguments": action.arguments, "result": result})

        # 3) theo dõi bước ghi và verify
        if action.tool == "record_activity" and not result.startswith("Error:"):
            try:
                to_verify = json.loads(result)
            except json.JSONDecodeError:
                verified = False
        elif action.tool == "get_record" and to_verify:
            verified = not result.startswith("Error:") and same_record(to_verify, result)
            log("VERIFY", step=step, ok=verified, written=to_verify["record"], read_back=result[:300])
            to_verify = None
    else:
        log("MAX_STEPS_REACHED", max_steps=MAX_STEPS)
        answer = Answer("Đã đạt giới hạn số bước mà chưa kết luận được.", status="error")

    if answer is None:
        answer = compose(question, obs)
    return finish(answer, verified, obs, usage, log)


def finish(answer: Answer, verified, obs: list[Observation], usage: Usage, log):
    """Nếu có ghi dữ liệu thì câu trả lời cuối cùng phụ thuộc vào kết quả verify."""
    wrote = next((o for o in reversed(obs)
                  if o["tool"] == "record_activity" and not o["result"].startswith("Error:")), None)
    if wrote:
        data = json.loads(wrote["result"])
        rec = data["record"]
        answer.verified = bool(verified)
        if verified:
            action_text = "ghi nhận" if data["created"] else "tìm thấy bản ghi đã có"
            answer.text = (f"Đã {action_text} và xác minh bằng cách đọc lại: sinh viên {rec['student_id']}, "
                           f"mã {rec['activity_code']}, {fmt(rec['points'])} điểm, ngày {rec['event_date']} "
                           f"(record_id {rec['record_id']}).")
            answer.status = "answered"
            summary = next((o for o in reversed(obs)
                            if o["tool"] == "get_student_summary" and not o["result"].startswith("Error:")), None)
            if summary:
                s = json.loads(summary["result"])
                answer.text += (f" Tổng hiện có: {fmt(s['total_points'])} điểm = {fmt(s['days'])} ngày CTXH; "
                                f"còn thiếu {fmt(s['missing_days'])} ngày so với mức tối thiểu "
                                f"{s['required_days']} ngày (khóa từ 4 năm).")
        else:
            answer.text = "Đã gửi yêu cầu ghi nhưng KHÔNG xác minh được bằng cách đọc lại, nên không xác nhận thành công."
            answer.status = "error"

    answer.steps = len(obs)
    log("ANSWER", status=answer.status, verified=answer.verified, text=answer.text,
        sources=[s.chunk_id for s in answer.sources], steps=answer.steps)
    log("USAGE", calls=usage.calls, prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)
    return answer, usage, obs
