"""Agent loop viết tay. Local policy để demo; API mode dùng LLM chọn tool."""
import json
import re
from dataclasses import asdict
from .config import MAX_STEPS
from .models import Action, Answer, Claim, Source, Usage
from .retrieval import normalize
from .mcp_client import call
from .llm import call_llm


def parse_action(text: str):
    try:
        text = text.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
        obj = json.loads(text)  # Tuyệt đối không eval() chuỗi model trả về.
        if not isinstance(obj, dict) or not isinstance(obj.get('tool'), str):
            return 'Error: action cần tool dạng chuỗi'
        args = obj.get('arguments', {})
        if not isinstance(args, dict):
            return 'Error: arguments phải là JSON object'
        return Action(obj['tool'], args)
    except (ValueError, TypeError):
        return 'Error: JSON action không hợp lệ'


def question_info(question):
    q = normalize(question)
    codes = list(dict.fromkeys(re.findall(r'\b[a-z]\d{2}\b', q.upper(), re.I)))
    student = re.search(r'\b\d{8}\b', q)
    # Phủ định không phải ý định ghi. Đây là guard tối thiểu, chưa phải NLP tổng quát.
    wants_write = 'dang ky' in q and not any(t in q for t in ['khong dang ky', 'dung dang ky', 'cach dang ky'])
    return q, [c.upper() for c in codes], student.group() if student else None, wants_write


def plan_local(question, mode, evidence, calculation, write_result, observed):
    q, codes, student, wants_write = question_info(question)
    if not observed:
        return Action('search_documents', {'query': question, 'mode': mode})
    if any(t in q for t in ['hoc phi','hoc bong','lich thi','hoan thanh','diem thi']):
        return Action('finish')
    if observed[-1]['result'].startswith('Error:'):
        return Action('finish')
    if codes:
        for code in codes:
            if not any(d['facts'].get('code') == code for d in evidence.values()):
                # Chỉ thử bổ sung một lần cho mỗi mã; mã không tồn tại sẽ bị từ chối.
                if not any(o['arguments'].get('query') == code for o in observed):
                    return Action('search_documents', {'query': code, 'mode': mode})
                return Action('finish')
        if 'POLICY' not in evidence:
            return Action('search_documents', {'query': 'quy tắc quy đổi giờ ngày đăng ký CTXH', 'mode': mode})
        if calculation is None:
            return Action('calculate_days', {'activity_codes': codes})
        if wants_write and student and len(codes) == 1 and write_result is None:
            return Action('register_activity', {'student_id': student, 'activity_code': codes[0]})
        return Action('finish')
    if any(t in q for t in ['quy doi', 'bao nhieu gio', 'ban moi', 'phien ban']):
        if 'POLICY' not in evidence:
            return Action('search_documents', {'query': 'quy tắc quy đổi giờ ngày CTXH', 'mode': mode})
    return Action('finish')


def synthesize(question, evidence, calculation, write_result, verified, chunks_by_doc):
    q, codes, student, wants_write = question_info(question)
    if any(t in q for t in ['hoc phi','hoc bong','lich thi','hoan thanh','diem thi']):
        return Answer('no_answer', 'Không đủ dữ liệu trong corpus để trả lời câu hỏi này.')
    if wants_write and (not student or len(codes) != 1):
        return Answer('no_answer', 'Cần một mã hoạt động và mã sinh viên demo 8 chữ số để đăng ký.')
    if codes:
        if calculation is None:
            return Answer('no_answer', 'Không đủ dữ liệu để quy đổi hoặc hoạt động chưa có số giờ.')
        sources = [Source(**s) for s in calculation['sources']]
        claims = [Claim('days', calculation['days'], sources),
                  Claim('hours_per_day', calculation['hours_per_day'], sources[:1])]
        text = f"Tổng {calculation['hours']} giờ / {calculation['hours_per_day']} = {calculation['days']:g} ngày CTXH dự kiến."
        if wants_write:
            if not verified or write_result is None:
                return Answer('error', 'Chưa xác minh được đăng ký. Không thông báo thành công.', claims)
            text += ' Đã đọc lại đăng ký ' + write_result['registration']['registration_id'] + ' và xác minh khớp.'
        return Answer('answered', text, claims, verified)
    policy = evidence.get('POLICY')
    if policy and any(t in q for t in ['quy doi','bao nhieu gio','ban moi','phien ban']):
        value = policy['facts']['hours_per_day']
        cited = rate_chunk((chunks_by_doc or {}).get('POLICY', [policy]), value)
        return Answer('answered', f'Bản hiện hành v{policy["version"]}: {value} giờ = 1 ngày CTXH dự kiến.',
                      [Claim('hours_per_day', value, [Source(**cited['source'])])])
    return Answer('no_answer', 'Không đủ dữ liệu hoặc chưa nhận diện được yêu cầu. Hãy nêu mã hoạt động và điều cần tính.')


def rate_chunk(policy_chunks, rate):
    """Chọn chunk POLICY thật sự chứa câu quy đổi để trích dẫn (không trích chunk không liên quan)."""
    phrase = f'{rate} giờ bằng 1 ngày'
    return next((c for c in policy_chunks if phrase in c['text']), policy_chunks[0])


async def run_agent(session, catalog, question, retrieval_mode, engine, log):
    usage = Usage()
    evidence = {}
    chunks_by_doc = {}
    calculation = None
    write_result = None
    pending = None
    verified = False
    observed = []
    allowed_names = {t['name'] for t in catalog}
    q, codes, student, wants_write = question_info(question)
    log('DISCOVER', tools=catalog)
    try:
        for step in range(1, MAX_STEPS + 1):
            # Đọc ngay sau ghi, trước mọi LLM/action khác. Guard do host chịu trách nhiệm.
            if pending:
                action = Action('get_registration', {'registration_id': pending['registration_id']})
            elif engine == 'api':
                raw = call_llm({'question': question, 'retrieval_mode': retrieval_mode,
                                'tools': catalog, 'observations': observed}, usage, log)
                action = parse_action(raw)
                log('MODEL_ACTION', raw=raw)
            else:
                action = plan_local(question, retrieval_mode, evidence, calculation, write_result, observed)
            if isinstance(action, str):
                log('ACTION_ERROR', step=step, error=action)
                observed.append({'tool':'parse', 'arguments':{}, 'result':action})
                continue
            if action.tool == 'finish':
                answer = synthesize(question, evidence, calculation, write_result, verified, chunks_by_doc)
                log('FINAL', answer=asdict(answer))
                return answer, usage, observed
            error = None
            if action.tool not in allowed_names:
                error = 'Error: tool không có trong catalog đã discover'
            if action.tool == 'calculate_days':
                requested = action.arguments.get('activity_codes', [])
                found = {d['facts'].get('code') for d in evidence.values()}
                if set(requested) != set(codes) or not requested or 'POLICY' not in evidence or not set(codes) <= found:
                    error = 'Error: chưa tra đủ policy và các mã được hỏi'
            if action.tool == 'register_activity':
                matches = (wants_write and student and len(codes) == 1 and
                           action.arguments.get('student_id') == student and
                           action.arguments.get('activity_code') == codes[0])
                if not matches or calculation is None or step >= MAX_STEPS:
                    error = 'Error: không đủ điều kiện ghi hoặc không còn bước verify'
            if action.tool == 'search_documents':
                action.arguments['mode'] = retrieval_mode
            result = error or await call(session, action.tool, action.arguments)
            log('TOOL', step=step, tool=action.tool, arguments=action.arguments, result=result)
            observed.append({'tool':action.tool,'arguments':action.arguments,'result':result})
            if result.startswith('Error:'):
                if pending:
                    answer = Answer('error', 'Ghi có thể đã thành công nhưng đọc lại thất bại; chưa xác minh, không tự ghi lại.')
                    log('FINAL', answer=asdict(answer))
                    return answer, usage, observed
                continue
            try:
                data = json.loads(result)
            except ValueError:
                observed[-1]['result'] = 'Error: tool trả JSON không hợp lệ'
                continue
            if action.tool == 'search_documents':
                for doc in data['documents']:
                    evidence.setdefault(doc['doc_id'], doc)  # chunk điểm cao nhất của doc
                    chunks_by_doc.setdefault(doc['doc_id'], []).append(doc)
            elif action.tool == 'calculate_days':
                calculation = data
            elif action.tool == 'register_activity':
                write_result = data
                pending = data['registration']
            elif action.tool == 'get_registration' and pending:
                row = data.get('registration', {})
                verified = all(row.get(k) == pending[k] for k in ['registration_id','student_id','activity_code'])
                log('VERIFY', success=verified, expected=pending, actual=row)
                pending = None
                answer = synthesize(question, evidence, calculation, write_result, verified, chunks_by_doc)
                log('FINAL', answer=asdict(answer))
                return answer, usage, observed
        answer = Answer('error', f'Dừng an toàn vì đạt MAX_STEPS={MAX_STEPS}.')
    except Exception as exc:
        log('RUN_ERROR', error=type(exc).__name__, message=str(exc))
        answer = Answer('error', str(exc))
    finally:
        log('USAGE', **asdict(usage), engine=engine)
    log('FINAL', answer=asdict(answer))
    return answer, usage, observed
