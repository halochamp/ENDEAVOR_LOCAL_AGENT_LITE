# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""No model needed — unit tests for the 2026-07-23 training-round fixes:
date injection, param-alias correction, degenerate-repetition detection."""
import sys, os
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.date_inject import resolve_relative_date, inject_datetime_context
from helpers.tool_parser import ToolCall, apply_param_aliases
from helpers.guardrails import is_degenerate_repetition

SCHEMAS = {
    "rag_search": {"parameters": {"required": ["query"]}},
    "bash": {"parameters": {"required": ["command"]}},
}


def test_resolve_relative_date_today():
    assert resolve_relative_date("วันนี้อากาศดี") == date.today().isoformat()


def test_resolve_relative_date_yesterday():
    expected = (date.today() - timedelta(days=1)).isoformat()
    assert resolve_relative_date("เมื่อวานฝนตก") == expected


def test_resolve_relative_date_none_when_absent():
    assert resolve_relative_date("แนะนำร้านกาแฟ") is None


def test_inject_datetime_context_adds_note_when_triggered():
    out = inject_datetime_context("พรุ่งนี้มีนัดไหม")
    assert "[DateTime context:" in out
    assert "พรุ่งนี้มีนัดไหม" in out


def test_inject_datetime_context_adds_note_even_without_trigger():
    text = "กองทุนรวมที่น่าสนใจตอนนี้มีอะไรบ้าง"
    out = inject_datetime_context(text)
    assert out != text
    assert "[DateTime context:" in out
    assert "resolved_date" not in out
    assert date.today().isoformat() in out


def test_apply_param_aliases_fixes_command_to_query():
    call = ToolCall(name="rag_search", args={"command": "find / -name x"})
    fixed = apply_param_aliases(call, SCHEMAS)
    assert fixed.args == {"query": "find / -name x"}


def test_apply_param_aliases_noop_when_correct_already():
    call = ToolCall(name="rag_search", args={"query": "trigram"})
    fixed = apply_param_aliases(call, SCHEMAS)
    assert fixed.args == {"query": "trigram"}


def test_apply_param_aliases_noop_for_unknown_tool():
    call = ToolCall(name="mystery_tool", args={"command": "x"})
    fixed = apply_param_aliases(call, SCHEMAS)
    assert fixed.args == {"command": "x"}


def test_is_degenerate_repetition_detects_repeating_chunk():
    text = "normal start. " + ("*.gz -o -name *.bz2 " * 10)
    assert is_degenerate_repetition(text) is True


def test_is_degenerate_repetition_false_on_normal_text():
    text = "สวัสดีครับ วันนี้อากาศดีมาก ผมช่วยอะไรได้บ้างครับ ลองบอกความต้องการมาได้เลยครับ"
    assert is_degenerate_repetition(text) is False


def test_is_degenerate_repetition_false_on_distinct_absolute_path_list():
    root = "/Users/example/Desktop/ENDEAVOR_AGENTIC/ENDEAVOR_AGENT_LITE/"
    text = "\n".join(f"{i}. `{root}module_{i:02d}.py`" for i in range(12))
    assert is_degenerate_repetition(text) is False


def test_is_degenerate_repetition_false_on_short_text():
    assert is_degenerate_repetition("สั้นๆ") is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
