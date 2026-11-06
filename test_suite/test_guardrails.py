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


def test_bash_retry_after_changed_command_starts_new_sequence():
    g = RepeatCallGuard(max_repeats=2)
    assert g.check("bash", {"command": "find /tmp -name a"}) is None
    warning = g.check("bash", {"command": "find /tmp -name a"})
    assert warning is not None and "command" in warning
    assert g.check("bash", {"command": "find /tmp -name b"}) is None
    assert g.check("bash", {"command": "find /tmp -name a"}) is None


def test_web_search_retry_hint_requires_a_changed_query():
    g = RepeatCallGuard(max_repeats=2)
    assert g.check("web_search", {"query": "ข่าววันนี้"}) is None
    warning = g.check("web_search", {"query": "ข่าววันนี้"})
    assert warning is not None
    assert "query" in warning
    assert g.check("web_search", {"query": "ข่าววันนี้ กรุงเทพ"}) is None


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


def test_web_search_batch_retry_guard_hashes_list_queries_deterministically():
    g = RepeatCallGuard(max_repeats=2)
    batch = {"query": ["alpha revenue", "beta margin"]}
    assert g.check("web_search", batch) is None
    warning = g.check("web_search", batch)
    assert warning is not None and "query" in warning
    assert g.check("web_search", {"query": ["alpha revenue", "beta guidance"]}) is None


def test_outer_duplicate_policy_allows_one_bash_hint_then_hard_stops():
    import agent

    attempts = agent._ToolCallAttemptState()
    args = {"command": "ls -la /tmp"}
    assert agent._tool_call_attempt(attempts, "bash", args) == (1, False)
    # Second identical call reaches the wrapper so RepeatCallGuard can return
    # its actionable hint to the model.
    assert agent._tool_call_attempt(attempts, "bash", args) == (2, False)
    # If the model ignores that hint and emits the exact same call again, stop.
    assert agent._tool_call_attempt(attempts, "bash", args) == (3, True)


def test_outer_duplicate_policy_allows_ten_changed_bash_commands():
    import agent

    attempts = agent._ToolCallAttemptState()
    for index in range(10):
        count, stop = agent._tool_call_attempt(
            attempts, "bash", {"command": f"ls -la /tmp/changed-{index}"}
        )
        assert count == 1
        assert stop is False


def test_non_hint_tool_keeps_second_identical_outer_stop():
    import agent

    attempts = agent._ToolCallAttemptState()
    args = {"path": "/tmp/example.txt"}
    assert agent._tool_call_attempt(attempts, "read_file", args) == (1, False)
    assert agent._tool_call_attempt(attempts, "read_file", args) == (2, True)


def test_changed_bash_command_resets_exact_duplicate_streak():
    import agent

    attempts = agent._ToolCallAttemptState()
    a = {"command": "ls -la /tmp/a"}
    b = {"command": "ls -la /tmp/b"}
    assert agent._tool_call_attempt(attempts, "bash", a) == (1, False)
    assert agent._tool_call_attempt(attempts, "bash", a) == (2, False)
    assert agent._tool_call_attempt(attempts, "bash", b) == (1, False)
    # Returning to A after a materially changed command starts a fresh streak.
    assert agent._tool_call_attempt(attempts, "bash", a) == (1, False)


def test_turn_guard_second_duplicate_bash_returns_runtime_hint():
    import config
    from helpers import turn_guard

    args = {"command": "ls -la /tmp"}
    turn_guard.reset("inspect files")
    assert turn_guard.check("bash", args) is None
    turn_guard.record("bash", args, "ok")

    warning = turn_guard.check("bash", args)

    assert warning is not None
    assert warning.startswith("[error]")
    assert "command เดิม" in warning
    assert str(config.RUNTIME_HOME) in warning
    assert str(config.PROJECT_ROOT) in warning
