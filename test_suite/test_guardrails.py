# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""No model needed — pure logic tests for helpers/guardrails.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.guardrails import RepeatCallGuard, StepBudget, truncate_output


def test_repeat_guard_allows_first_call():
    g = RepeatCallGuard(max_repeats=2)
    assert g.check("bash", {"command": "find / -name x"}) is None


def test_repeat_guard_warns_on_second_identical_call():
    g = RepeatCallGuard(max_repeats=2)
    g.check("bash", {"command": "find / -name x"})
    warning = g.check("bash", {"command": "find / -name x"})
    assert warning is not None
    assert "ห้ามรันซ้ำ" in warning


def test_repeat_guard_aborts_on_third_identical_call():
    g = RepeatCallGuard(max_repeats=2)
    g.check("bash", {"command": "find / -name x"})
    g.check("bash", {"command": "find / -name x"})
    result = g.check("bash", {"command": "find / -name x"})
    assert result == "__ABORT__"


def test_repeat_guard_different_args_not_flagged():
    g = RepeatCallGuard(max_repeats=2)
    g.check("bash", {"command": "find / -name x"})
    result = g.check("bash", {"command": "find . -name x"})
    assert result is None


def test_repeat_guard_reset():
    g = RepeatCallGuard(max_repeats=2)
    g.check("bash", {"command": "x"})
    g.reset()
    assert g.check("bash", {"command": "x"}) is None


def test_truncate_output_short_text_unchanged():
    assert truncate_output("hello", 100) == "hello"


def test_truncate_output_long_text_cut_with_note():
    text = "a" * 3000
    out = truncate_output(text, 2000)
    assert len(out) > 2000  # cut text + note
    assert out.startswith("a" * 2000)
    assert "ตัดที่ 2000 chars" in out


def test_truncate_output_includes_saved_path():
    out = truncate_output("a" * 100, 10, saved_path="/tmp/full.txt")
    assert "/tmp/full.txt" in out


def test_step_budget_allows_up_to_max():
    b = StepBudget(max_steps=3)
    assert b.take() is True
    assert b.take() is True
    assert b.take() is True
    assert b.take() is False


def test_step_budget_remaining():
    b = StepBudget(max_steps=3)
    b.take()
    assert b.remaining == 2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
