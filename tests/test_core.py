import asyncio
import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from src.activities import load_activities, points_to_days, required_days
from src.agent import parse_action, run_agent
from src.chunking import load_chunks
from src.llm import call_llm
from src.logging_utils import RunLog
from src.mcp_client import call, connect
from src.models import Usage
from src.planner import extract
from src.retrieval import Retriever

ROOT = Path(__file__).resolve().parent.parent


class DataTests(unittest.TestCase):
    def test_table_parsed_with_forward_filled_category(self):
        table = load_activities()
        self.assertEqual(len(table), 55)
        self.assertEqual(table["001105"].category, "Mùa hè xanh")  # ô lĩnh vực trống ở dòng gốc
        self.assertEqual((table["001202"].points_min, table["001202"].days_max), (5, 0.5))
        self.assertEqual((table["006102"].days_min, table["006102"].days_max), (1.5, 3))

    def test_rounding_rule_2c(self):
        self.assertEqual([points_to_days(p) for p in (43, 45, 47, 10, 4, 6)], [4.0, 4.5, 5.0, 1.0, 0.0, 1.0])
        self.assertEqual((required_days(4), required_days(3)), (4, 2))

    def test_chunks_have_metadata_and_table_rows_are_atomic(self):
        chunks = load_chunks()
        self.assertTrue(all(c.document.owner and c.source.version >= 1 for c in chunks))
        rows = [c for c in chunks if c.kind == "table_row"]
        self.assertEqual(len(rows), 55)
        self.assertTrue(all(c.code in c.text and "Quy đổi:" in c.text for c in rows))
        self.assertEqual(len({c.source.chunk_id for c in chunks}), len(chunks))


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Retriever()

    def test_two_modes_find_evidence(self):
        for mode in ("dense", "hybrid"):
            hits = self.r.search("Sinh viên khiếu nại kết quả CTXH trong bao lâu?", mode, 3)
            self.assertTrue(any("Trong vòng 10 ngày" in c.text for c, _ in hits), mode)
        with self.assertRaises(ValueError):
            self.r.search("x", "bm25")

    def test_metadata_filter_applies_before_ranking(self):
        hits = self.r.search("hiến máu", "hybrid", 4, doc_id="CTXH-III")
        self.assertTrue(hits and all(c.source.doc_id == "CTXH-III" for c, _ in hits))
        with self.assertRaises(ValueError):
            self.r.search("x", "hybrid", 3, doc_id="NOPE")


class ParserTests(unittest.TestCase):
    def test_invalid_inputs_return_error_string(self):
        for raw in ["", "not json", "[]", '{"tool":4}', '{"tool":"x","arguments":[]}', "{"]:
            self.assertTrue(str(parse_action(raw)).startswith("Error:"), raw)

    def test_json_inside_code_fence_is_parsed(self):
        action = parse_action('```json\n{"tool":"finish","arguments":{"answer":"xin chào"}}\n```')
        self.assertEqual((action.tool, action.arguments["answer"]), ("finish", "xin chào"))

    def test_write_intent_rules(self):
        self.assertTrue(extract("Ghi nhận mã 001202 cho sinh viên 11110001").write)
        self.assertFalse(extract("Không ghi nhận, chỉ tính mã 001202").write)
        self.assertFalse(extract("Còn bao nhiêu chỗ đăng ký?").write)


class McpAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = RunLog(Path(self.tmp.name) / "t.jsonl")

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def db(self):
        return Path(self.tmp.name) / "db.sqlite3"

    async def test_discover_write_then_verify_and_idempotent(self):
        async with connect(self.db()) as (session, catalog):
            self.assertGreaterEqual(len(catalog), 2)
            self.assertIn("record_activity", [t["name"] for t in catalog])
            q = "Ghi nhận sinh viên 11110001 mã 001202, 5 điểm, ngày 2026-10-10."
            answer, _, obs = await run_agent(session, catalog, q, "hybrid", "local", self.log)
            names = [o["tool"] for o in obs]
            at = names.index("record_activity")
            self.assertEqual(names[at + 1], "get_record")
            self.assertTrue(answer.verified)
            first = json.loads(obs[at]["result"])["record"]["record_id"]
            again = json.loads(await call(session, "record_activity", {"student_id": "11110001", "activity_code": "001202", "points": 5, "event_date": "2026-10-10"}))
            self.assertFalse(again["created"])
            self.assertEqual(again["record"]["record_id"], first)

    async def test_tool_errors_are_strings_not_exceptions(self):
        async with connect(self.db()) as (session, _):
            cases = [("record_activity", {"student_id": "bad", "activity_code": "001202", "points": 5, "event_date": "2026-10-10"}),
                     ("record_activity", {"student_id": "11110001", "activity_code": "999999", "points": 5, "event_date": "2026-10-10"}),
                     ("record_activity", {"student_id": "11110001", "activity_code": "001202", "points": 9, "event_date": "2026-10-10"}),
                     ("record_activity", {"student_id": "11110001", "activity_code": "001202", "points": 5, "event_date": "2999-01-01"}),
                     ("record_activity", {"student_id": "11110001", "activity_code": "001202", "points": 5, "event_date": "2026-10-10", "was_paid": True}),
                     ("lookup_activity", {"activity_codes": ["000000"]}), ("points_to_days", {"total_points": -1}),
                     ("get_record", {"record_id": "none"}), ("search_regulations", {"query": " "}),
                     ("search_regulations", {"query": "x", "doc_id": "NOPE"}), ("not_a_tool", {})]
            for name, args in cases:
                result = await call(session, name, args)
                self.assertTrue(result.startswith("Error:"), (name, args, result))

    async def test_negated_request_never_writes(self):
        async with connect(self.db()) as (session, catalog):
            _, _, obs = await run_agent(session, catalog, "Không ghi nhận gì cả, chỉ tính mã 001202.", "hybrid", "local", self.log)
            self.assertFalse(any(o["tool"] == "record_activity" for o in obs))

    async def test_missing_fields_refuse_without_write(self):
        async with connect(self.db()) as (session, catalog):
            answer, _, obs = await run_agent(session, catalog, "Ghi nhận mã 001202 cho sinh viên 11110002.", "hybrid", "local", self.log)
            self.assertEqual(answer.status, "refused")
            self.assertFalse(any(o["tool"] == "record_activity" for o in obs))

    async def test_malformed_model_output_stops_at_max_steps(self):
        events = []
        with patch("src.agent.call_llm", return_value="{broken"):
            answer, _, _ = await run_agent(None, [], "mã 001202?", "hybrid", "api", lambda e, **kw: events.append(e))
        self.assertEqual(answer.status, "error")
        self.assertEqual(events.count("ACTION_ERROR"), 8)
        self.assertIn("MAX_STEPS_REACHED", events)

    async def test_model_cannot_force_unauthorized_write(self):
        raw = json.dumps({"tool": "record_activity", "arguments": {"student_id": "11110001", "activity_code": "001202", "points": 5, "event_date": "2026-10-10"}})
        with patch("src.agent.call_llm", return_value=raw), patch("src.agent.call") as invoke:
            await run_agent(None, [{"name": "record_activity"}], "Chỉ cho tôi xem mã 001202", "hybrid", "api", lambda *a, **k: None)
            invoke.assert_not_called()

    async def test_verify_failure_never_claims_success(self):
        from src.mcp_client import call as real_call

        async def faulty(session, name, args):
            return "Error: simulated read failure" if name == "get_record" else await real_call(session, name, args)

        async with connect(self.db()) as (session, catalog):
            with patch("src.agent.call", side_effect=faulty):
                answer, _, _ = await run_agent(session, catalog, "Ghi nhận sinh viên 11110001 mã 001202, 5 điểm, ngày 2026-10-10.", "hybrid", "local", self.log)
        self.assertFalse(answer.verified)
        self.assertEqual(answer.status, "error")

    async def test_api_mode_end_to_end_with_scripted_model(self):
        script = iter([json.dumps({"tool": "lookup_activity", "arguments": {"activity_codes": ["001202"]}}),
                       '{"tool":"finish","arguments":{"status":"answered","answer":"0,5 ngày","sources":["CTXH-II-BANG:v1:7","BIA-DAT"]}}'])
        with patch("src.agent.call_llm", side_effect=lambda *a, **k: next(script)):
            async with connect(self.db()) as (session, catalog):
                answer, _, _ = await run_agent(session, catalog, "mã 001202 được mấy ngày?", "hybrid", "api", self.log)
        self.assertEqual(answer.status, "answered")
        self.assertEqual([s.chunk_id for s in answer.sources], ["CTXH-II-BANG:v1:7"])  # nguồn bịa bị loại


class LlmAdapterTests(unittest.TestCase):
    def serve(self, statuses):
        hits = {"n": 0}

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                code = statuses[min(hits["n"], len(statuses) - 1)]
                hits["n"] += 1
                body = json.dumps({"choices": [{"message": {"content": "{\"tool\":\"finish\"}"}}],
                                   "usage": {"prompt_tokens": 11, "completion_tokens": 3}} if code == 200 else {"error": "x"}).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/v1", hits

    def test_retries_5xx_then_succeeds_and_counts_usage(self):
        url, hits = self.serve([500, 200])
        events, usage = [], Usage()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": url}), patch("src.llm.time.sleep"):
            out = call_llm({"question": "x"}, usage, lambda e, **kw: events.append((e, kw)))
        self.assertEqual(out, '{"tool":"finish"}')
        self.assertEqual((hits["n"], usage.calls, usage.prompt_tokens, usage.completion_tokens), (2, 2, 11, 3))
        self.assertEqual(events[0][0], "HTTP_ERROR")

    def test_4xx_is_not_retried_and_key_not_logged(self):
        url, hits = self.serve([401])
        events = []
        with patch.dict("os.environ", {"OPENAI_API_KEY": "SECRET-KEY-123", "OPENAI_BASE_URL": url}):
            with self.assertRaises(RuntimeError):
                call_llm({"question": "x"}, Usage(), lambda e, **kw: events.append(json.dumps(kw)))
        self.assertEqual(hits["n"], 1)
        self.assertNotIn("SECRET-KEY-123", "".join(events))

    def test_missing_key_raises_clear_error(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                call_llm({}, Usage(), lambda *a, **k: None)


class RepoHygieneTests(unittest.TestCase):
    def test_no_secret_in_tracked_text_files(self):
        pattern = re.compile(r"sk-[A-Za-z0-9_\-]{20,}|gsk_[A-Za-z0-9]{20,}")
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".txt", ".json", ".jsonl", ".example"} and ".venv" not in path.parts and ".git" not in path.parts:
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8", errors="ignore")), str(path))
        self.assertIn("OPENAI_API_KEY=\n", (ROOT / ".env.example").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
