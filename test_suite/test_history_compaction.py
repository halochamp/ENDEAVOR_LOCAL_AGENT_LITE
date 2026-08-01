# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""No model needed — unit tests for agent.py's cross-turn history compaction.

Updated 2026-07-23 for the architecture switch to create_react_agent
(config.py's module docstring): tool_call_ok/format_tool_log_line moved to
helpers/turn_guard.py (shared per-turn state the tool wrappers check
directly); agent.py no longer inserts/tracks a system message in the
persisted history (react.py's dynamic_prompt always prepends it fresh), so
trim_history no longer has system-message-aware slicing; image handling
moved into react.py's dynamic_prompt (base64-encodes on the fly), so
agent._collect_history_images no longer exists.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
import config
from helpers import turn_guard


def test_compact_turn_drops_tool_output_keeps_log_and_answer():
    pair = agent._compact_turn(
        "หาไฟล์ PLAN.md",
        ["- rag_search(query=PLAN.md) -> สำเร็จ"],
        "เจอที่ /path/PLAN.md",
    )
    assert pair[0] == {"role": "user", "content": "หาไฟล์ PLAN.md"}
    assistant_content = pair[1]["content"]
    assert "rag_search" in assistant_content
    assert "เจอที่ /path/PLAN.md" in assistant_content
    assert "<tool_response>" not in assistant_content


def test_compact_turn_no_tool_log_when_no_tools_called():
    pair = agent._compact_turn("สวัสดี", [], "สวัสดีครับ")
    assert pair[1]["content"] == "สวัสดีครับ"


def test_trim_history_keeps_at_most_max_turns():
    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    agent.trim_history(messages)
    assert len(messages) == agent.MAX_HISTORY_TURNS * 2
    assert messages[0]["content"] == f"q{10 - agent.MAX_HISTORY_TURNS}"


def test_trim_history_enforces_char_cap():
    messages = []
    big = "x" * 9000
    for i in range(agent.MAX_HISTORY_TURNS):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": big})
    total_before = sum(len(m["content"]) for m in messages)
    assert total_before > agent.MAX_HISTORY_CHARS
    agent.trim_history(messages)
    total_after = sum(len(m["content"]) for m in messages)
    assert total_after <= agent.MAX_HISTORY_CHARS


def test_trim_history_char_cap_never_drops_below_one_turn():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "x" * 30_000},
    ]
    agent.trim_history(messages)
    assert len(messages) == 2


def test_trim_history_noop_on_short_history():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    before = list(messages)
    agent.trim_history(messages)
    assert messages == before


def test_to_lc_messages_round_trip():
    from langchain_core.messages import HumanMessage, AIMessage
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    lc = agent._to_lc_messages(messages)
    assert isinstance(lc[0], HumanMessage) and lc[0].content == "hi"
    assert isinstance(lc[1], AIMessage) and lc[1].content == "hello"


# ── turn_guard (moved from agent.py) ────────────────────────────────────
def test_turn_guard_check_allows_first_call():
    turn_guard.reset()
    assert turn_guard.check("bash", {"command": "ls"}) is None


def test_turn_guard_check_blocks_on_quota():
    turn_guard.reset()
    for _ in range(3):
        assert turn_guard.check("bash", {"command": f"echo {_}"}) is None
        turn_guard.record("bash", {"command": f"echo {_}"}, "ok")
    blocked = turn_guard.check("bash", {"command": "echo 4"})
    assert blocked is not None and "โควตา" in blocked


def test_turn_guard_check_warns_then_aborts_on_repeat():
    turn_guard.reset()
    args = {"command": "find / -name x"}
    assert turn_guard.check("bash", args) is None
    turn_guard.record("bash", args, "[error] timeout")
    warn = turn_guard.check("bash", args)
    assert warn is not None
    turn_guard.record("bash", args, "[error] timeout")
    abort = turn_guard.check("bash", args)
    assert abort is not None and "ห้ามเรียกซ้ำ" in abort


def test_turn_guard_record_computes_ok_status():
    turn_guard.reset()
    turn_guard.record("bash", {"command": "ls"}, "some output")
    log = turn_guard.get_tool_log()
    assert "สำเร็จ" in log[0]


def test_turn_guard_record_computes_failure_status():
    turn_guard.reset()
    turn_guard.record("read_file", {"path": "/x"}, "[error] not found")
    log = turn_guard.get_tool_log()
    assert "ล้มเหลว" in log[0]


def test_turn_guard_reset_clears_state():
    turn_guard.reset()
    turn_guard.record("bash", {"command": "ls"}, "ok")
    assert turn_guard.tool_call_count() == 1
    turn_guard.reset()
    assert turn_guard.tool_call_count() == 0
    assert turn_guard.get_tool_log() == []


def test_turn_guard_reserves_quota_before_parallel_tool_execution(monkeypatch):
    monkeypatch.setattr(config, "MAX_CALLS_PER_TOOL", 2)
    turn_guard.reset()
    args = [{"query": str(i)} for i in range(3)]
    allowed = [turn_guard.check("web_search", arg) for arg in args]
    assert allowed[:2] == [None, None]
    assert allowed[2].startswith("[error]")
    assert turn_guard.tool_call_count() == 2
    for arg in args[:2]:
        turn_guard.record("web_search", arg, "ok")
    assert turn_guard.tool_call_count() == 2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
