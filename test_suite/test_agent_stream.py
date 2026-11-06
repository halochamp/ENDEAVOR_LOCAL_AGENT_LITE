"""Agent stream plumbing tests; uses a fake LangGraph graph, never a model."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import agent
import config
from helpers import search_scope
from tools import read_image


def _rag_mcp_call(query: str, call_id: str = "rag-mcp") -> dict:
    return {
        "name": "mcp_call_tool",
        "id": call_id,
        "args": {
            "server": "lite-rag",
            "tool_name": "rag_search",
            "arguments_json": json.dumps({"query": query}, ensure_ascii=False),
        },
    }


def _rag_mcp_message(call_id: str, content: str, status: str = "success"):
    return SimpleNamespace(
        name="mcp_call_tool", tool_call_id=call_id, status=status, content=content,
    )


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


def test_run_turn_keeps_streaming_transport_for_non_token_callers(monkeypatch):
    seen = {}

    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="ok")]}}

    def fake_get_react_agent(**kwargs):
        seen.update(kwargs)
        return _NoToolGraph()

    monkeypatch.setattr(agent, "_get_react_agent", fake_get_react_agent)
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ตอบสั้น ๆ", [], stream_tokens=False)

    assert result.text == "ok"
    assert seen == {"streaming": True, "thinking_budget": config.THINKING_BUDGET}


def test_run_turn_selects_the_requested_effort_budget(monkeypatch):
    seen = {}

    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="ok")]}}

    def fake_get_react_agent(**kwargs):
        seen.update(kwargs)
        return _NoToolGraph()

    monkeypatch.setattr(agent, "_get_react_agent", fake_get_react_agent)
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ตอบสั้น ๆ", [], effort="high")

    assert result.text == "ok"
    assert seen == {"streaming": True, "thinking_budget": 512}


def test_boilerplate_fallback_surfaces_verified_tool_result():
    result = agent._fallback_summary(
        ["- bash(command=cat /tmp/one.py) -> สำเร็จ", "- cat(path=/tmp/missing.py) -> ล้มเหลว"],
        "bash",
        "/tmp/one.py\nprint('ok')",
    )

    assert "แสดงผลที่ยืนยันได้" in result
    assert "/tmp/one.py" in result
    assert "สรุปคำตอบไม่สำเร็จ" not in result


def test_mcp_call_shorthand_is_repaired_to_fixed_local_rag():
    assert agent._normalize_mcp_tool_args(
        "mcp_call_tool", {"arguments_json": '{"query":"ต่ออายุ"}'},
    ) == {
        "arguments_json": '{"query":"ต่ออายุ"}',
        "server": "lite-rag",
        "tool_name": "rag_search",
    }


def test_mcp_call_does_not_override_explicit_route():
    args = {
        "server": "other",
        "tool_name": "ping",
        "arguments_json": "{}",
    }
    assert agent._normalize_mcp_tool_args("mcp_call_tool", args) == args


def test_local_document_search_query_extracts_content_phrase():
    assert agent._local_document_search_query(
        "เอกสารไหนพูดถึงค่าปรับยกเลิกสัญญา และค่าปรับเท่าไร"
    ) == "ค่าปรับยกเลิกสัญญา"
    assert agent._local_document_search_query(
        "หาเอกสารที่มีคำว่า ยอดชำระ 4,500 บาท"
    ) == "ยอดชำระ 4,500 บาท"
    assert agent._rag_search_query(
        "ค้นหาข้อมูลใน rag ที่มีเนื้อหาเกี่ยวกับการ landing"
    ) == "การ landing"
    assert agent._rag_search_query("ค้นหาใน rag เรื่อง landing") == "landing"
    assert agent._rag_search_query("ค้นหาใน rag_document เรื่อง landing") == "landing"
    assert agent._rag_search_query("ค้นหาใน rag_document/md เรื่อง landing") == "landing"
    assert agent._rag_search_query(
        "ดึงข้อมูลจากฐานความรู้เรื่อง landing"
    ) == "landing"


def test_focused_web_queries_do_not_split_thai_about_phrase():
    assert agent._focused_web_queries(
        "ค้นหาข่าวล่าสุดเกี่ยวกับ OpenAI"
    ) == "ข่าว เกี่ยวกับ OpenAI 2026"
    assert agent._focused_web_queries(
        "สถานการณ์ล่าสุดเกี่ยวกับไทยกับกัมพูชา"
    ) == ["ไทย 2026", "กัมพูชา 2026"]


def test_local_document_scope_requires_path_as_a_word_not_a_substring():
    assert search_scope.is_local_document_query("latest pathology research") is False
    assert search_scope.is_local_document_query("look up multipath networking") is False
    assert agent._should_force_web_search("latest pathology research") is True
    assert agent._should_force_web_search("look up multipath networking") is True
    assert search_scope.is_local_document_query(
        "show path /Users/example/file.txt"
    ) is True


def test_rag_scope_requires_local_source_intent_not_bare_rag_mention():
    assert search_scope.is_knowledge_base_query("RAG คืออะไร") is False
    assert search_scope.is_knowledge_base_query("RAG ทำงานอย่างไร") is False
    assert search_scope.is_knowledge_base_explanation_query("RAG คืออะไร") is True
    assert search_scope.is_knowledge_base_query("ค้นหาใน RAG เรื่อง landing") is True
    assert search_scope.is_knowledge_base_query("ค้นหาใน rag_document เรื่อง landing") is True
    assert search_scope.is_knowledge_base_query(
        "ดึงข้อมูลจากฐานความรู้เรื่อง landing"
    ) is True
    assert search_scope.is_knowledge_base_query(
        "ค้นหาบนเว็บเกี่ยวกับ RAG"
    ) is False
    assert search_scope.is_outside_knowledge_base_query(
        "ค้นหานอก RAG เรื่อง landing"
    ) is True
    assert search_scope.is_knowledge_base_query(
        "ค้นหานอก RAG เรื่อง landing"
    ) is False
    assert search_scope.is_outside_knowledge_base_query("RAG คืออะไร") is False
    assert search_scope.is_outside_knowledge_base_query(
        "ค้นหาบนเว็บนอก RAG เรื่อง landing"
    ) is False
    assert agent._outside_rag_search_query(
        "ค้นหานอก RAG เรื่อง landing"
    ) == "landing"


def test_explanatory_rag_query_does_not_force_web_or_preload_rag(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="RAG คือ Retrieval-Augmented Generation",
            )]}}

    class _UnexpectedMcp:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            raise AssertionError("explanatory RAG query must not preload rag_search")

    class _UnexpectedWeb:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            raise AssertionError("explanatory RAG query must not force web_search")

    fake_mcp = _UnexpectedMcp()
    fake_web = _UnexpectedWeb()
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._mcp_tool, "mcp_call_tool", fake_mcp)
    monkeypatch.setattr(agent._web_search_tool, "web_search", fake_web)
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("RAG คืออะไร", [])

    assert result.text == "RAG คือ Retrieval-Augmented Generation"
    assert fake_mcp.calls == []
    assert fake_web.calls == []


def test_explicit_outside_rag_scope_forces_bash_before_graph(monkeypatch, tmp_path):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="สรุปจากผล bash",
            )]}}

    class _FakeBash:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return "/Users/test/Desktop/manual.txt\n"

    class _UnexpectedTool:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            raise AssertionError("outside-RAG scope must not use this tool")

    desktop = tmp_path / "Desktop"
    documents = tmp_path / "Documents"
    desktop.mkdir()
    documents.mkdir()
    fake_bash = _FakeBash()
    fake_mcp = _UnexpectedTool()
    fake_web = _UnexpectedTool()
    monkeypatch.setattr(config, "LOCAL_SEARCH_ROOTS", (desktop, documents))
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._bash_tool, "bash", fake_bash)
    monkeypatch.setattr(agent._mcp_tool, "mcp_call_tool", fake_mcp)
    monkeypatch.setattr(agent._web_search_tool, "web_search", fake_web)
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    events = []
    result = agent.run_turn(
        "ค้นหานอก RAG เรื่อง landing",
        [],
        on_step=lambda **event: events.append(event),
    )

    from tools import bash

    assert len(fake_bash.calls) == 1
    assert fake_bash.calls[0]["command"].startswith("grep -r -l -F landing ")
    assert bash._read_only_violation(fake_bash.calls[0]["command"]) is None
    assert fake_mcp.calls == []
    assert fake_web.calls == []
    assert events[0]["name"] == "bash"
    assert result.text == "สรุปจากผล bash"


def test_local_document_recovery_finishes_when_model_skips_tools(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="") ]}}

    class _FakeMcpCall:
        def invoke(self, args):
            assert args["server"] == "lite-rag"
            assert args["tool_name"] == "rag_search"
            assert json.loads(args["arguments_json"]) == {"query": "ค่าปรับยกเลิกสัญญา"}
            return "พบ 1 ไฟล์\n- /project/เงื่อนไขสัญญาเช่า.txt\n  ค่าปรับยกเลิกสัญญา 8,000 บาท"

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._mcp_tool, "mcp_call_tool", _FakeMcpCall())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    events = []
    result = agent.run_turn(
        "เอกสารไหนพูดถึงค่าปรับยกเลิกสัญญา และค่าปรับเท่าไร", [],
        on_step=lambda **event: events.append(event),
    )

    assert "8,000" in result.text
    assert any(event.get("name") == "mcp_call_tool" for event in events)
    assert not result.aborted


def test_generic_local_document_recovery_falls_back_to_desktop_documents(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="ไม่พบใน Knowledge Base"
            )]}}

    class _FakeMcpCall:
        def invoke(self, args):
            assert args["tool_name"] == "rag_search"
            return "ไม่พบเนื้อหาที่ตรงกับ 'อายุการใช้งานกล้อง' ใน Knowledge Base"

    class _FakeLocalSearch:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return (
                "พบ 1 ไฟล์ใน Desktop/Documents ที่มีเนื้อหาตรงกับ 'อายุการใช้งานกล้อง':\n"
                "- /Users/test/Documents/camera-notes.txt\n"
                "  (Documents) อายุการใช้งานกล้องขึ้นกับการดูแล"
            )

    fake_local = _FakeLocalSearch()
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._mcp_tool, "mcp_call_tool", _FakeMcpCall())
    monkeypatch.setattr(agent._local_search_tool, "local_search", fake_local)
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ค้นหาเนื้อหาไฟล์ที่มีเนื้อหาเกี่ยวกับอายุการใช้งานกล้อง", [])

    assert fake_local.calls == [{"query": "อายุการใช้งานกล้อง"}]
    assert "/Users/test/Documents/camera-notes.txt" in result.text
    assert "Knowledge Base ว่าง" not in result.text
    assert not result.aborted


def test_explicit_rag_scope_forces_mcp_before_web(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="ตอบจากความจำ",
            )]}}

    class _FakeMcpCall:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return (
                "พบ 1 ช่วงข้อความที่ตรงกับ 'การ landing':\n"
                "- ไฟล์: /Users/test/Knowledge/manual.docx\n"
                "  หน้า/ตำแหน่ง: page 7/20\n"
                "  [source: /Users/test/Knowledge/manual.docx]\n"
                "  ตัวอย่างเนื้อหา: ขั้นตอนการ landing"
            )

    class _UnexpectedWeb:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return "- web must not run\n  [source: https://unexpected.test]"

    fake_mcp = _FakeMcpCall()
    fake_web = _UnexpectedWeb()
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._mcp_tool, "mcp_call_tool", fake_mcp)
    monkeypatch.setattr(agent._web_search_tool, "web_search", fake_web)
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    events = []
    result = agent.run_turn(
        "ค้นหาข้อมูลใน rag ที่มีเนื้อหาเกี่ยวกับการ landing",
        [],
        on_step=lambda **event: events.append(event),
    )

    assert len(fake_mcp.calls) == 1
    assert json.loads(fake_mcp.calls[0]["arguments_json"]) == {"query": "การ landing"}
    assert fake_web.calls == []
    assert any(
        event.get("name") == "mcp_call_tool" for event in events
    )
    assert not any(event.get("name") == "web_search" for event in events)
    assert "แหล่งที่มา (Knowledge Base):" in result.text
    assert "page 7/20" in result.text
    assert "รายละเอียดตัวเลขจากข้อความในเอกสาร" not in result.text


def test_explicit_rag_scope_falls_back_to_bash_when_mcp_unavailable(monkeypatch, tmp_path):
    knowledge = tmp_path / "Knowledge"
    knowledge.mkdir()

    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="ไม่พบข้อมูล",
            )]}}

    class _UnavailableMcp:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return "[error] mcp_call_tool failed for 'lite-rag' (FileNotFoundError)"

    class _FakeBash:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return str(knowledge / "manual.txt")

    class _UnexpectedWeb:
        def invoke(self, _args):
            raise AssertionError("web_search must not replace local RAG data")

    fake_mcp = _UnavailableMcp()
    fake_bash = _FakeBash()
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._mcp_tool, "mcp_call_tool", fake_mcp)
    monkeypatch.setattr(agent._bash_tool, "bash", fake_bash)
    monkeypatch.setattr(agent._web_search_tool, "web_search", _UnexpectedWeb())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ค้นหาข้อมูลใน RAG เรื่อง landing", [])

    assert len(fake_mcp.calls) == 1
    assert len(fake_bash.calls) == 1
    assert str(knowledge) in fake_bash.calls[0]["command"]
    assert "landing" in fake_bash.calls[0]["command"]
    assert "ผลการค้นหาใน Knowledge Base ด้วย bash:" in result.text
    assert str(knowledge / "manual.txt") in result.text
    assert not result.aborted


def test_generic_local_search_uses_bash_when_rag_mcp_unavailable(monkeypatch, tmp_path):
    desktop = tmp_path / "Desktop"
    documents = tmp_path / "Documents"
    desktop.mkdir()
    documents.mkdir()

    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="ไม่พบข้อมูล",
            )]}}

    class _UnavailableMcp:
        def invoke(self, _args):
            return "[error] no MCP server named 'lite-rag' configured"

    class _FakeBash:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return str(documents / "camera.txt")

    class _UnexpectedLocalSearch:
        def invoke(self, _args):
            raise AssertionError("local_search must not replace the requested bash fallback")

    fake_bash = _FakeBash()
    monkeypatch.setattr(config, "LOCAL_SEARCH_ROOTS", (desktop, documents))
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._mcp_tool, "mcp_call_tool", _UnavailableMcp())
    monkeypatch.setattr(agent._bash_tool, "bash", fake_bash)
    monkeypatch.setattr(agent._local_search_tool, "local_search", _UnexpectedLocalSearch())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ค้นหาเนื้อหาไฟล์ที่มีเนื้อหาเกี่ยวกับอายุการใช้งานกล้อง", [])

    assert len(fake_bash.calls) == 1
    command = fake_bash.calls[0]["command"]
    assert str(desktop) in command and str(documents) in command
    assert "อายุการใช้งานกล้อง" in command
    assert "ผลการค้นหาใน Desktop/Documents ด้วย bash:" in result.text
    assert str(documents / "camera.txt") in result.text
    assert not result.aborted


def test_explicit_web_scope_about_rag_keeps_web_route(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="ตอบจากความจำ",
            )]}}

    class _FakeWebSearch:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return "- Web evidence\n  [source: https://example.test/rag]"

    fake_web = _FakeWebSearch()
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._web_search_tool, "web_search", fake_web)
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("ค้นหาบนเว็บเกี่ยวกับ rag implementation", [])

    assert fake_web.calls
    assert "https://example.test/rag" in result.text


def test_local_fact_recovery_restores_numeric_detail_dropped_by_model(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            args = _rag_mcp_call("ค่าปรับยกเลิกสัญญา", "rag-1")
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[args])]}}
            yield {"tools": {"messages": [_rag_mcp_message(
                "rag-1",
                "พบ 1 ไฟล์\n- /project/เงื่อนไขสัญญาเช่า.txt\n"
                "  (เงื่อนไขสัญญาเช่า.txt) ค่าปรับยกเลิกสัญญา 8,000 บาท",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="พบเอกสารเงื่อนไขสัญญาเช่า.txt",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn(
        "เอกสารไหนพูดถึงค่าปรับยกเลิกสัญญา และค่าปรับเท่าไร", [],
    )

    assert "8,000" in result.text
    assert not result.aborted


def test_web_boilerplate_fallback_uses_best_vetted_result(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            first = {"name": "web_search", "id": "web-1", "args": {"query": "rate"}}
            second = {"name": "web_search", "id": "web-2", "args": {"query": "rate tomorrow"}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[first])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="web_search", tool_call_id="web-1", status="success",
                content="- Official rate\n  [source: https://bot.example/rate]",
            )]}}
            agent.turn_guard.record(
                "web_search", {"query": "rate"},
                "- Official rate\n  [source: https://bot.example/rate]",
            )
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[second])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="web_search", tool_call_id="web-2", status="success",
                content="[needs_retry] refine query",
            )]}}
            agent.turn_guard.record("web_search", {"query": "rate tomorrow"}, "[needs_retry] refine query")
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="Sorry, need more steps to process this request.",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("ค้นหาอัตราดอกเบี้ยล่าสุด", [])

    assert "Official rate" in result.text
    assert "[needs_retry]" not in result.text
    assert result.text.endswith("แหล่งที่มา:\n- https://bot.example/rate")


def test_web_search_is_blocked_for_local_document_turn():
    from helpers import turn_guard
    from tools import web_search

    turn_guard.reset("เอกสารไหนพูดถึงค่าปรับยกเลิกสัญญา")
    result = web_search.web_search.invoke({"query": "ค่าปรับยกเลิกสัญญา"})

    assert result.startswith("[blocked]")
    assert "mcp_call_tool" in result


def test_web_search_is_blocked_for_explicit_rag_scope():
    from helpers import turn_guard
    from tools import web_search

    query = "ค้นหาข้อมูลใน rag ที่มีเนื้อหาเกี่ยวกับการ landing"
    turn_guard.reset(query)
    result = web_search.web_search.invoke({"query": "การ landing"})

    assert result.startswith("[blocked]")
    assert "Knowledge Base/RAG" in result
    assert "mcp_call_tool" in result


def test_mcp_rag_search_is_blocked_for_explanatory_rag_question():
    from helpers import turn_guard
    from tools import mcp_client

    turn_guard.reset("RAG คืออะไร")
    result = mcp_client.mcp_call_tool.invoke({
        "server": "lite-rag",
        "tool_name": "rag_search",
        "arguments_json": '{"query":"คืออะไร"}',
    })

    assert result.startswith("[blocked]")
    assert "ความหมายหรือการทำงานของ RAG" in result


def test_web_search_is_blocked_for_explanatory_rag_question():
    from helpers import turn_guard
    from tools import web_search

    turn_guard.reset("RAG คืออะไร")
    result = web_search.web_search.invoke({"query": "RAG คืออะไร"})

    assert result.startswith("[blocked]")
    assert "ไม่ใช่คำขอค้นเว็บ" in result


def test_mcp_and_web_are_blocked_for_explicit_outside_rag_scope():
    from helpers import turn_guard
    from tools import mcp_client, web_search

    query = "ค้นหานอก RAG เรื่อง landing"
    turn_guard.reset(query)
    rag_result = mcp_client.mcp_call_tool.invoke({
        "server": "lite-rag",
        "tool_name": "rag_search",
        "arguments_json": '{"query":"landing"}',
    })
    web_result = web_search.web_search.invoke({"query": "landing"})

    assert rag_result.startswith("[blocked]")
    assert "ต้องใช้ bash" in rag_result
    assert web_result.startswith("[blocked]")
    assert "ต้องใช้ bash" in web_result


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


def test_current_thai_question_forces_code_owned_web_search(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="ตอบจากความจำ")]}}

    class _FakeWebSearch:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return "- Evidence\n  [source: https://example.com/current]"

    fake_web = _FakeWebSearch()
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._web_search_tool, "web_search", fake_web)
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    query = "บรรยากาศสงครามและการลงทุนโลกตอนนี้เป็นอย่างไร"
    result = agent.run_turn(query, [])

    assert agent._should_force_web_search(query) is True
    assert fake_web.calls == [{"query": [
        f"ความขัดแย้งระหว่างประเทศ {agent.date.today().year}",
        f"การลงทุนโลก {agent.date.today().year}",
    ]}]
    assert "https://example.com/current" in result.text


def test_bare_search_followup_reuses_prior_subject(monkeypatch):
    class _NoToolGraph:
        def stream(self, _input, *, config, stream_mode):
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[], content="ตอบจากความจำ")]}}

    class _FakeWebSearch:
        def __init__(self):
            self.calls = []

        def invoke(self, args):
            self.calls.append(args)
            return "- Python release notes\n  [source: https://docs.python.org/3/whatsnew/3.13.html]"

    fake_web = _FakeWebSearch()
    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _NoToolGraph())
    monkeypatch.setattr(agent._web_search_tool, "web_search", fake_web)
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    original = "Python 3.13 release notes"
    history = [{"role": "user", "content": original}, {"role": "assistant", "content": "คำตอบเดิม"}]

    result = agent.run_turn("คุณไปค้นหาก่อนตอบสิ", history)

    assert agent._is_web_search_followup("คุณไปค้นหาก่อนตอบสิ") is True
    assert agent._web_search_subject("คุณไปค้นหาก่อนตอบสิ", history[:2]) == original
    assert fake_web.calls == [{"query": original}]
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
            call = _rag_mcp_call("tool loop", "rag-project")
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [_rag_mcp_message(
                "rag-project",
                "[ค้นจากเนื้อหาในโปรเจกต์ปัจจุบัน: /project]\\n"
                "พบ 1 ไฟล์\\n- /project/agent.py\\n  (agent.py) ...tool loop...",
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
            rag_call = _rag_mcp_call("kMDItemFSName", "rag-loop")
            bash_call = {"name": "bash", "args": {"command": "grep -n kMDItemFSName /project"}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[rag_call])]}}
            yield {"tools": {"messages": [_rag_mcp_message(
                "rag-loop",
                "พบ 2 ไฟล์ที่ตรงกับ 'kMDItemFSName':\n- /project/tools/bash.py\n- /project/prompts.py",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[bash_call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success", content="/project/tools/bash.py\n/project/prompts.py",
            )]}}
            # First exact duplicate reaches the wrapper and returns its hint.
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[bash_call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success",
                content="[error] bash ถูกเรียกด้วย command เดิมติดต่อกัน\n[hint] เปลี่ยน command จริง",
            )]}}
            # Ignoring the hint and emitting the same call once more hard-stops.
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[bash_call])]}}
            raise AssertionError("third identical call should stop before this point")

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("ค้นหาเนื้อหาในไฟล์ทั้งหมดของโปรเจกต์เกี่ยวกับ kMDItemFSName ให้ครบ", [])
    assert result.aborted is False
    assert "/project/tools/bash.py" in result.text


def test_model_shortened_rag_paths_are_restored_as_absolute_paths(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            call = _rag_mcp_call("syllabus", "rag-syllabus-1")
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [_rag_mcp_message(
                "rag-syllabus-1",
                content=(
                    "พบ 3 ไฟล์ที่ตรงกับ 'syllabus':\n"
                    "- /Users/test/Desktop/6 Syllabus/syllabus ฝป.docx\n"
                    "- /Users/test/Desktop/6 Syllabus/syllabus ฝค.pdf\n"
                    "- /Users/test/Desktop/6 Syllabus/syllabus ฝป.pdf"
                ),
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[],
                content=(
                    "ไฟล์ทั้งหมด:\n"
                    "- syllabus ฝป.docx (Word)\n"
                    "- syllabus ฝค.pdf (PDF)\n"
                    "- /Users/test/Desktop/6 Syllabus/syllabus ฝป..pdf (PDF)"
                ),
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    result = agent.run_turn("ค้นหาไฟล์ syllabus", [])
    assert result.aborted is False
    assert "/Users/test/Desktop/6 Syllabus/syllabus ฝป.docx (Word)" in result.text
    assert "/Users/test/Desktop/6 Syllabus/syllabus ฝค.pdf (PDF)" in result.text
    assert "/Users/test/Desktop/6 Syllabus/syllabus ฝป.pdf (PDF)" in result.text
    assert "ฝป..pdf" not in result.text


def test_model_relative_rag_paths_are_restored_as_absolute_paths(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            call = _rag_mcp_call("syllabus", "rag-syllabus-2")
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [_rag_mcp_message(
                "rag-syllabus-2",
                content=(
                    "พบ 2 ไฟล์ที่ตรงกับ 'syllabus':\n"
                    "- /Users/test/Desktop/6 Syllabus/Syllabus 402.xlsx\n"
                    "- /Users/test/Desktop/6 Syllabus/syllabus ฝป.pdf"
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
    assert "/Users/test/Desktop/6 Syllabus/Syllabus 402.xlsx (Excel)" in result.text
    assert "/Users/test/Desktop/6 Syllabus/syllabus ฝป.pdf (PDF)" in result.text
    assert "เอกสารเก่า Syllabus/" not in result.text


def test_exact_duplicate_bash_gets_one_wrapper_hint_before_hard_stop(monkeypatch):
    class _RepeatingGraph:
        def stream(self, _input, *, config, stream_mode):
            call = {"name": "bash", "args": {"command": "find /tmp -type f"}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success", content="[blocked] test rejection",
            )]}}
            # Second identical call must reach the wrapper so the model sees a hint.
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success",
                content="[error] bash ถูกเรียกด้วย command เดิมติดต่อกัน\n[hint] เปลี่ยน command จริง",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            raise AssertionError("third identical call should stop before this point")

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _RepeatingGraph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)
    events = []

    result = agent.run_turn("ใช้ bash หาไฟล์", [], on_step=lambda **event: events.append(event))

    assert result.aborted is True
    assert "tool loop" in result.text
    assert events == [
        {"event": "tool_start", "name": "bash", "args": {"command": "find /tmp -type f"}},
        {"event": "tool_done", "name": "bash", "ok": False},
        {"event": "tool_start", "name": "bash", "args": {"command": "find /tmp -type f"}},
        {"event": "tool_done", "name": "bash", "ok": False},
    ]


def test_successful_duplicate_search_recovers_first_result_after_ignored_hint(monkeypatch):
    class _RepeatingAfterSuccessGraph:
        def stream(self, _input, *, config, stream_mode):
            call = {"name": "bash", "args": {"command": "find /tmp -type f"}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success", content="/tmp/one.pdf",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [SimpleNamespace(
                name="bash", status="success",
                content="[error] bash ถูกเรียกด้วย command เดิมติดต่อกัน\n[hint] เปลี่ยน command จริง",
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            raise AssertionError("third identical call should stop before this point")

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
            call = _rag_mcp_call("กล้อง", "rag-local")
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]} }
            yield {"tools": {"messages": [_rag_mcp_message(
                "rag-local", "พบไฟล์กล้อง"
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
        {
            "event": "tool_start",
            "name": "mcp_call_tool",
            "args": {
                "server": "lite-rag",
                "tool_name": "rag_search",
                "arguments_json": json.dumps({"query": "กล้อง"}, ensure_ascii=False),
            },
        },
        {"event": "tool_done", "name": "mcp_call_tool", "ok": True},
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


def test_rag_source_paths_are_rendered_by_code_with_chunk_location(monkeypatch):
    class _Graph:
        def stream(self, _input, *, config, stream_mode):
            call = _rag_mcp_call("กล้อง", "rag-source-1")
            yield {"agent": {"messages": [SimpleNamespace(tool_calls=[call])]}}
            yield {"tools": {"messages": [_rag_mcp_message(
                "rag-source-1",
                (
                    "พบ 1 ช่วงข้อความที่ตรงกับ 'กล้อง':\n"
                    "- ไฟล์: /Users/test/Desktop/manual.pdf\n"
                    "  ชื่อไฟล์: manual.pdf\n"
                    "  หน้า/ตำแหน่ง: page 4/9\n"
                    "  [source: /Users/test/Desktop/manual.pdf]\n"
                    "  ตัวอย่างเนื้อหา: กล้องมีอายุการใช้งานห้าปี"
                ),
            )]}}
            yield {"agent": {"messages": [SimpleNamespace(
                tool_calls=[], content="กล้องมีข้อมูลในคู่มือครับ",
            )]}}

    monkeypatch.setattr(agent, "_get_react_agent", lambda **_kwargs: _Graph())
    monkeypatch.setattr(agent, "_log", lambda _event: None)

    result = agent.run_turn("เอกสารไหนพูดถึงกล้อง", [])

    assert "[source:" not in result.text
    assert result.text.endswith(
        "แหล่งที่มา (Knowledge Base):\n"
        "- /Users/test/Desktop/manual.pdf — page 4/9"
    )


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
    monkeypatch.setattr(
        agent._web_search_tool,
        "web_search",
        type("_NoEvidence", (), {"invoke": lambda _self, _args: "[needs_retry] no relevant results"})(),
    )

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
    snapshot = pending / "pending-image-abandoned-for-test.png"

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


def test_runtime_machine_hint_is_derived_from_current_process_account():
    import os
    import pwd
    from pathlib import Path
    import prompts

    account = pwd.getpwuid(os.getuid())
    assert config.RUNTIME_USER == account.pw_name
    assert config.RUNTIME_HOME == Path(account.pw_dir).resolve()
    assert str(config.RUNTIME_HOME) in prompts.SYSTEM_PROMPT
    assert str(config.RUNTIME_DESKTOP) in prompts.SYSTEM_PROMPT
    assert str(config.PROJECT_ROOT) in prompts.SYSTEM_PROMPT
    assert "ห้ามเดา username" in prompts.SYSTEM_PROMPT
    if config.RUNTIME_OS == "Darwin":
        assert "macOS/BSD" in prompts.SYSTEM_PROMPT


def test_default_tool_recovery_budgets_are_ten_without_env_override():
    import os

    if "LITE_MAX_TOOL_STEPS" not in os.environ:
        assert config.MAX_TOOL_STEPS == 10
    if "LITE_MAX_CALLS_PER_TOOL" not in os.environ:
        assert config.MAX_CALLS_PER_TOOL == 10
    assert agent._RECURSION_LIMIT == config.MAX_TOOL_STEPS * 2 + 2
