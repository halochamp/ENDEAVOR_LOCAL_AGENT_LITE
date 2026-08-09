# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Agent stream plumbing tests; uses a fake LangGraph graph, never a model."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import agent
import config
from tools import read_image


class _FakeGraph:
    def stream(self, _input, *, config, stream_mode):
        assert config["recursion_limit"] > 0
        assert stream_mode == "updates"
        yield {
            "agent": {
                "messages": [
                    SimpleNamespace(
                        tool_calls=[{"name": "web_search", "args": {"query": "test"}}]
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [
                    SimpleNamespace(
                        name="web_search",
                        status="success",
                        content="- Useful source\n  [source: https://example.com/result]"
                    )
                ]
            }
        }
        yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="Final answer")]}}


def test_run_turn_reports_live_tool_events_and_keeps_web_result(monkeypatch):
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _FakeGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    events = []
    history = []

    result = agent.run_turn("search", history, on_step=lambda **event: events.append(event))

    assert events == [
        {"event": "tool_start", "name": "web_search", "args": {"query": "test"}},
        {"event": "tool_done", "name": "web_search", "ok": True},
    ]
    # Source citation is appended deterministically (agent.py's
    # _append_source_citations), not left to the model to remember.
    assert result.text == "Final answer\n\nแหล่งที่มา:\n- https://example.com/result"
    assert result.web_search_results == ["- Useful source\n  [source: https://example.com/result]"]
    assert len(history) == 2


def test_missed_web_route_is_recovered_with_source_grounded_call(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="ตอบจากความจำ") ]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    class _FakeWebSearch:
        def invoke(self, _args):
            return "- Python release notes\n  [source: https://docs.python.org/3/whatsnew/3.13.html]"

    monkeypatch.setattr(agent._web_search_tool, "web_search", _FakeWebSearch())
    events = []
    result = agent.run_turn(
        "ช่วยค้นหาว่า Python 3.13 มีฟีเจอร์ใหม่อะไรบ้าง พร้อมแนบลิงก์ที่มา",
        [], on_step=lambda **event: events.append(event),
    )
    assert events[-2:] == [
        {"event": "tool_start", "name": "web_search", "args": {"query": "ช่วยค้นหาว่า Python 3.13 มีฟีเจอร์ใหม่อะไรบ้าง พร้อมแนบลิงก์ที่มา"}},
        {"event": "tool_done", "name": "web_search", "ok": True},
    ]
    assert "https://docs.python.org/3/whatsnew/3.13.html" in result.text


def test_relative_write_path_is_resolved_inside_workspect():
    first = agent._normalize_write_tool_args(
        "write_file", {"path": "workspect/walk_test_note.txt"},
    )
    second = agent._normalize_write_tool_args(
        "write_file", {"path": "walk_test_note.txt"},
    )
    expected = str(config.BASH_ALLOWED_ROOT / "walk_test_note.txt")
    assert first["path"] == expected
    assert second["path"] == expected


def test_model_dropped_leading_slash_does_not_redirect_outside_write():
    args = agent._normalize_write_tool_args(
        "write_file",
        {"path": "tmp/walk_test_outside.txt"},
        "เขียนไฟล์ที่ /tmp/walk_test_outside.txt ให้มีเนื้อหา escape attempt",
    )
    assert args["path"] == "/tmp/walk_test_outside.txt"


def test_write_overwrite_gate_requires_explicit_user_permission():
    args, tripped = agent._apply_first_write_overwrite_gate(
        {"path": "x.txt", "overwrite": "true"},
        'เขียนไฟล์ x.txt แต่ห้าม overwrite',
        False,
    )
    assert args["overwrite"] == ""
    assert tripped is True
    allowed, still_tripped = agent._apply_first_write_overwrite_gate(
        {"path": "x.txt", "overwrite": "true"},
        'เขียนทับไฟล์เดิมได้เลย overwrite=true',
        False,
    )
    assert allowed["overwrite"] == "true"
    assert still_tripped is False


def test_sensitive_path_request_has_explicit_blocked_answer(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="ไม่พบผลลัพธ์",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("อ่านไฟล์ ~/.ssh/id_rsa ให้หน่อย", [])
    assert result.aborted is True
    assert "[blocked]" in result.text
    assert "ไม่อนุญาตให้อ่าน" in result.text


def test_destructive_command_request_has_explicit_blocked_answer(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="หยุดการทำงานเพื่อป้องกัน tool loop",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("รันคำสั่ง rm -rf ~/Desktop ให้หน่อย", [])
    assert result.aborted is True
    assert "[blocked]" in result.text
    assert "ถูกบล็อก" in result.text


def test_project_content_evidence_replaces_model_no_result(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[{
                "name": "rag_search", "args": {"query": "tool loop"},
            }])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="rag_search", status="success",
                content=(
                    "[ค้นจากเนื้อหาในโปรเจกต์ปัจจุบัน: /project]\\n"
                    "พบ 1 ไฟล์\\n- /project/agent.py\\n  (agent.py) ...tool loop..."
                ),
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="ไม่พบผลลัพธ์ที่อยู่ในโปรเจกต์ปัจจุบัน"
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn(
        "ช่วยค้นข้อมูลจากเนื้อหาเอกสารเกี่ยวกับการป้องกัน tool loop ในโปรเจกต์นี้",
        [],
    )
    assert result.aborted is False
    assert "/project/agent.py" in result.text


def test_project_rag_evidence_survives_later_tool_loop(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            rag_call = {"name": "rag_search", "args": {"query": "kMDItemFSName"}}
            bash_call = {"name": "bash", "args": {"command": "grep -n kMDItemFSName /project"}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[rag_call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="rag_search", status="success",
                content="พบ 2 ไฟล์ที่ตรงกับ 'kMDItemFSName':\\n- /project/tools/bash.py\\n- /project/prompts.py",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[bash_call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success", content="/project/tools/bash.py\\n/project/prompts.py",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[bash_call])]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("ค้นหาเนื้อหาในไฟล์ทั้งหมดของโปรเจกต์เกี่ยวกับ kMDItemFSName ให้ครบ", [])
    assert result.aborted is False
    assert "/project/tools/bash.py" in result.text


def test_model_shortened_rag_paths_are_restored_as_absolute_paths(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[{
                "name": "rag_search", "args": {"query": "syllabus"},
            }])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="rag_search", status="success",
                content=(
                    "พบ 3 ไฟล์ที่ตรงกับ 'syllabus':\n"
                    "- /Users/halochampmac/Desktop/6 Syllabus/syllabus ฝป.docx\n"
                    "- /Users/halochampmac/Desktop/6 Syllabus/syllabus ฝค.pdf\n"
                    "- /Users/halochampmac/Desktop/6 Syllabus/syllabus ฝป.pdf"
                ),
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[],
                content=(
                    "ไฟล์ทั้งหมด:\n"
                    "- syllabus ฝป.docx (Word)\n"
                    "- syllabus ฝค.pdf (PDF)\n"
                    "- /Users/halochampmac/Desktop/6 Syllabus/syllabus ฝป..pdf (PDF)"
                ),
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("ค้นหาไฟล์ syllabus", [])
    assert result.aborted is False
    assert "/Users/halochampmac/Desktop/6 Syllabus/syllabus ฝป.docx (Word)" in result.text
    assert "/Users/halochampmac/Desktop/6 Syllabus/syllabus ฝค.pdf (PDF)" in result.text
    assert "/Users/halochampmac/Desktop/6 Syllabus/syllabus ฝป.pdf (PDF)" in result.text
    assert "ฝป..pdf" not in result.text


def test_model_relative_rag_paths_are_restored_as_absolute_paths(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[{
                "name": "rag_search", "args": {"query": "syllabus"},
            }])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="rag_search", status="success",
                content=(
                    "พบ 2 ไฟล์ที่ตรงกับ 'syllabus':\n"
                    "- /Users/halochampmac/Desktop/6 Syllabus/Syllabus 402.xlsx\n"
                    "- /Users/halochampmac/Desktop/6 Syllabus/syllabus ฝป.pdf"
                ),
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[],
                content=(
                    "- เอกสารเก่า Syllabus/เอกสารเก่า/Syllabus 402.xlsx (Excel)\n"
                    "- เอกสารเก่า Syllabus/syllabus ฝป.pdf (PDF)"
                ),
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("ค้นหาไฟล์ syllabus", [])
    assert result.aborted is False
    assert "/Users/halochampmac/Desktop/6 Syllabus/Syllabus 402.xlsx (Excel)" in result.text
    assert "/Users/halochampmac/Desktop/6 Syllabus/syllabus ฝป.pdf (PDF)" in result.text
    assert "เอกสารเก่า Syllabus/" not in result.text


def test_exact_duplicate_tool_call_stops_graph_before_another_execution(monkeypatch):
    class _RepeatingGraph:
        def stream(self, _input, *, config, stream_mode):
            call = {"name": "bash", "args": {"command": "find /tmp -type f"}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success", content="[blocked] test rejection",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            raise AssertionError("duplicate call should stop the graph before this point")

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _RepeatingGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    events = []

    result = agent.run_turn("ใช้ bash หาไฟล์", [], on_step=lambda **event: events.append(event))

    assert result.aborted is True
    assert "tool loop" in result.text
    assert events == [
        {"event": "tool_start", "name": "bash", "args": {"command": "find /tmp -type f"}},
        {"event": "tool_done", "name": "bash", "ok": False},
    ]


def test_successful_duplicate_search_finishes_from_first_result(monkeypatch):
    class _RepeatingAfterSuccessGraph:
        def stream(self, _input, *, config, stream_mode):
            call = {"name": "bash", "args": {"command": "find /tmp -type f"}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success", content="/tmp/one.pdf",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            raise AssertionError("successful duplicate should finish from the first result")

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _RepeatingAfterSuccessGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ใช้ bash หาไฟล์", [])

    assert result.aborted is False
    assert "/tmp/one.pdf" in result.text


def test_large_bash_preview_uses_code_count_and_saved_path(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[{
                "name": "bash", "args": {"command": "find /tmp -type f"},
            }])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success",
                content=(
                    "[full output: 3894 lines; saved: /tmp/full.txt; preview below is incomplete]\n"
                    "/tmp/one.pdf\n/tmp/two.pdf"
                ),
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="ทั้งหมด 2 ไฟล์",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ใช้ bash list PDF ทั้งหมด", [])

    assert "3894 บรรทัด" in result.text
    assert "/tmp/full.txt" in result.text
    assert "ทั้งหมด 2 ไฟล์" not in result.text


def test_local_document_policy_leaves_search_term_to_agent(monkeypatch):
    class _Graph:
        def stream(self, graph_input, *, config, stream_mode):
            supplied = graph_input["messages"]
            assert len(supplied) == 1
            assert not getattr(supplied[-1], "name", None)
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[{
                "name": "rag_search", "args": {"query": "กล้อง"},
            }])]} }
            yield {"tools": {"messages": [SimpleNamespace(
                name="rag_search", status="success", content="พบไฟล์กล้อง"
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="ใช้ผล RAG แล้ว")]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    events = []

    result = agent.run_turn(
        "ช่วยค้นเอกสาร Word ในโฟลเดอร์ Documents",
        [],
        on_step=lambda **event: events.append(event),
    )

    assert result.text == "ใช้ผล RAG แล้ว"
    assert events == [
        {"event": "tool_start", "name": "rag_search", "args": {"query": "กล้อง"}},
        {"event": "tool_done", "name": "rag_search", "ok": True},
    ]


def test_source_links_are_rendered_by_code_not_copied_model_text(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[{"name": "web_search", "args": {"query": "topic"}}],
            )]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="web_search", status="success",
                content="- Relevant\n  [source: https://example.com/relevant]",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[],
                content="ภาพรวมจาก [แหล่งข่าว](https://example.com/relevant) และ https://example.com/relevant",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ค้น topic", [])

    assert "ภาพรวมจาก แหล่งข่าว" in result.text
    assert result.text.count("https://example.com/relevant") == 1
    assert result.text.endswith("แหล่งที่มา:\n- https://example.com/relevant")


def test_web_claims_are_blocked_when_no_relevant_source_survives(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[{"name": "web_search", "args": {"query": "policy rate"}}],
            )]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="web_search", status="success",
                content="[needs_retry] no relevant results",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="อัตราดอกเบี้ยอยู่ที่ 8.0%",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ดอกเบี้ยล่าสุด", [])

    assert result.aborted is True
    assert "8.0%" not in result.text
    assert "ไม่พบแหล่งข้อมูลบนเว็บที่เกี่ยวข้อง" in result.text
    assert "http" not in result.text


def test_web_percentage_absent_from_vetted_snippet_is_blocked(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[{"name": "web_search", "args": {"query": "policy rate"}}],
            )]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="web_search", status="success",
                content="- Official policy page\n  [source: https://central-bank.test/rate]",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="อัตราดอกเบี้ยนโยบายอยู่ที่ 8.0%",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ดอกเบี้ยล่าสุด", [])

    assert result.aborted is True
    assert "8.0%" not in result.text
    assert "ยังไม่ยืนยันตัวเลข" in result.text
    assert result.text.endswith("แหล่งที่มา:\n- https://central-bank.test/rate")


def test_run_turn_streams_direct_answer_tokens_only_when_opted_in(monkeypatch):
    class _StreamingGraph:
        def stream(self, _input, *, config, stream_mode):
            assert stream_mode == "updates"
            callback = config["callbacks"][0]
            run_id = "direct-answer"
            callback.on_chat_model_start({}, [], run_id=run_id)
            callback.on_llm_new_token("First", run_id=run_id)
            callback.on_llm_new_token(" token", run_id=run_id)
            response = SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(
                tool_calls=[], additional_kwargs={},
            ))]])
            callback.on_llm_end(response, run_id=run_id)
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="First token")]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _StreamingGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    events = []

    result = agent.run_turn(
        "hello", [], on_step=lambda **event: events.append(event), stream_tokens=True,
    )

    assert result.text == "First token"
    assert events == [
        {"event": "stream_start"},
        {"event": "token", "text": "First"},
        {"event": "token", "text": " token"},
        {"event": "stream_end"},
    ]


def test_stream_callback_discards_provisional_tool_call_text():
    events = []
    callback = agent._TokenStreamCallback(lambda **event: events.append(event))
    run_id = "tool-call"
    callback.on_chat_model_start({}, [], run_id=run_id)
    callback.on_llm_new_token("provisional", run_id=run_id)
    response = SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(
        tool_calls=[{"name": "read_file", "args": {}}], additional_kwargs={},
    ))]])

    callback.on_llm_end(response, run_id=run_id)

    assert events[-1] == {"event": "discard_stream"}


def test_tool_graph_never_disables_global_thinking(monkeypatch):
    calls = []
    graph = object()
    monkeypatch.setattr(agent, "_react_agents", {})
    monkeypatch.setattr(
        agent,
        "build_react_agent",
        lambda **kwargs: calls.append(kwargs) or graph,
    )

    assert agent._get_react_agent(streaming=True) is graph
    assert agent._get_react_agent(streaming=True) is graph
    assert calls == [{"streaming": True}]


def test_run_turn_retries_a_missed_explicit_file_write(monkeypatch):
    class _WriteRetryGraph:
        def __init__(self):
            self.calls = 0

        def stream(self, _input, *, config, stream_mode):
            self.calls += 1
            if self.calls == 1:
                yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="เขียนให้แล้ว")]}}
                return
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[
                {"name": "write_file", "args": {"path": "/tmp/x.txt", "content": "x"}}
            ])]}}
            yield {"tools": {"messages": [SimpleNamespace(name="write_file", status="success", content="written")]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="บันทึกไฟล์แล้ว")]}}

    graph = _WriteRetryGraph()
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: graph)
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("สร้างไฟล์ report.txt ให้มีคำว่า hello", [])

    assert graph.calls == 2
    assert result.text == "บันทึกไฟล์แล้ว"


def test_run_turn_keeps_first_answer_when_write_retry_also_misses_tool(monkeypatch):
    class _FailedWriteRetryGraph:
        def __init__(self):
            self.calls = 0

        def stream(self, _input, *, config, stream_mode):
            self.calls += 1
            if self.calls == 1:
                yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="สรุปเสร็จแล้ว")]}}
                return
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="รอบ retry ก็ยังไม่ได้เขียนไฟล์"
            )]}}

    graph = _FailedWriteRetryGraph()
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: graph)
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("สร้างไฟล์ report.txt ให้มีคำว่า hello", [])

    assert graph.calls == 2
    assert result.text == "สรุปเสร็จแล้ว"


def test_run_turn_stops_at_the_next_checkpoint_when_cancelled(monkeypatch):
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _FakeGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    cancel_event = threading.Event()

    def on_step(**event):
        # Only the graph's outer per-update loop is checked (Lite has no
        # tools/_progress.py-style mid-tool-call hook), so cancel is only
        # observed once the in-flight tool call has fully returned.
        if event["event"] == "tool_done":
            cancel_event.set()

    history = []
    result = agent.run_turn("search", history, on_step=on_step, cancel_event=cancel_event)

    assert result.cancelled
    assert result.aborted
    assert result.text == ""
    # A cancelled turn produced no answer worth persisting across turns.
    assert history == []


def test_run_turn_cancel_before_stream_starts_is_a_free_no_op(monkeypatch):
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _FakeGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    cancel_event = threading.Event()
    cancel_event.set()  # e.g. set while queued behind another turn's lock

    history = []
    result = agent.run_turn("search", history, on_step=lambda **_: None, cancel_event=cancel_event)

    assert result.cancelled
    assert history == []


def test_run_turn_pre_reads_an_explicit_image_attachment(monkeypatch):
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _FakeGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    monkeypatch.setattr(agent._read_image_tool, "run", lambda args: "[OCR] receipt total 4500")
    events = []

    agent.run_turn("อธิบายภาพนี้", [], on_step=lambda **event: events.append(event), attachment_image="/tmp/receipt.png")

    assert events[:2] == [
        {"event": "tool_start", "name": "read_image", "args": {"path": "/tmp/receipt.png"}},
        {"event": "tool_done", "name": "read_image", "ok": True},
    ]


def test_run_turn_pre_reads_an_image_path_before_model_routing(monkeypatch):
    captured = {}

    class _ImageGraph:
        def stream(self, graph_input, *, config, stream_mode):
            captured["messages"] = graph_input["messages"]
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="Invoice Total: 4500 THB",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _ImageGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    monkeypatch.setattr(
        agent._read_image_tool,
        "read_image",
        SimpleNamespace(invoke=lambda args: "[OCR] Invoice Total: 4500 THB"),
    )
    events = []

    result = agent.run_turn(
        "อ่านรูป /var/folders/test/invoice.png ให้หน่อย",
        [],
        on_step=lambda **event: events.append(event),
    )

    assert result.text == "Invoice Total: 4500 THB"
    assert result.tool_calls == 0  # mocked wrapper does not touch turn_guard
    assert events[:2] == [
        {"event": "tool_start", "name": "read_image", "args": {"path": "/var/folders/test/invoice.png"}},
        {"event": "tool_done", "name": "read_image", "ok": True},
    ]
    assert captured["messages"][-2].tool_calls[0]["name"] == "read_image"


def test_run_turn_discards_abandoned_private_image_snapshot(monkeypatch):
    pending = config.DATA_DIR / "pending_images"
    pending.mkdir(parents=True, exist_ok=True)
    snapshot = pending / "abandoned-for-test.png"

    class _CrashGraph:
        def stream(self, *_args, **_kwargs):
            snapshot.write_bytes(b"snapshot")
            read_image._PENDING_IMAGES[:] = [str(snapshot)]
            raise RuntimeError("synthetic graph failure")
            yield  # pragma: no cover

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _CrashGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("new turn", [])
    assert result.aborted
    assert read_image._PENDING_IMAGES == []
    assert not snapshot.exists()


def test_run_turn_blocks_credential_shaped_final_answer(monkeypatch):
    class _SecretGraph:
        def stream(self, *_args, **_kwargs):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="sk-" + "A" * 32)]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _SecretGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("test", [])
    assert result.aborted
    assert result.text.startswith("[blocked]")


def test_local_numeric_grounding_removes_only_invented_details():
    text = (
        "วันต่ออายุ 30 กันยายน 2569 (ตรงกับปี พ.ศ. 2534)\n"
        "- เบี้ยประกัน 12,500 บาท\n"
        "- วันที่ตรวจเอกสาร 2026-08-01"
    )
    evidence = ["ต่ออายุวันที่ 30 กันยายน 2569 เบี้ยประกัน 12,500 บาท"]

    result = agent._strip_unsupported_local_numbers(text, evidence, "ช่วยสรุปเอกสาร")

    assert "30 กันยายน 2569" in result
    assert "12,500 บาท" in result
    assert "2534" not in result
    assert "2026-08-01" not in result


def test_local_numeric_grounding_allows_requested_conversion():
    text = "ปี 2569 เท่ากับ 2026"
    evidence = ["ปี 2569"]
    assert agent._strip_unsupported_local_numbers(text, evidence, "ช่วยแปลงปี") == text


def test_explicit_image_text_request_uses_grounded_ocr():
    tool_result = (
        "[OCR] ข้อความในภาพ:\nInvoice Total: 4500 THB\n"
        "[หมายเหตุ] ภาพถูกส่งเข้า context แล้ว"
    )
    result = agent._ground_image_text_answer(
        "คำตอบหลุดไปพูดเรื่องอื่น", tool_result, "อ่านรูปนี้ มีข้อความอะไรบ้าง",
    )
    assert result == "ข้อความที่อ่านได้จากภาพ:\n\nInvoice Total: 4500 THB"


def test_visual_description_does_not_replace_vlm_with_ocr():
    tool_result = "[OCR] ข้อความในภาพ:\nSALE\n[หมายเหตุ] ภาพถูกส่งเข้า context แล้ว"
    assert agent._ground_image_text_answer("ภาพเป็นร้านค้า", tool_result, "บรรยายภาพนี้") == "ภาพเป็นร้านค้า"
