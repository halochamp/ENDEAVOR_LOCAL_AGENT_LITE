# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""turn_guard.py — per-turn guardrail state shared by the 6 tool wrappers.

Architecture switch (2026-07-23, config.py's module docstring): create_react_agent
runs its own internal ReAct loop (LangGraph graph), so agent.py no longer owns
a hand-rolled loop to check RepeatCallGuard/MAX_CALLS_PER_TOOL against before
dispatch. Reinserting the checks at the graph level (pre_model_hook/
post_model_hook, mutating AIMessage.tool_calls) is fragile — it fights
LangGraph's own routing logic for deciding tools-node vs end. Simpler and
just as effective: each tool's @tool wrapper (tools/*.py) checks this shared
per-turn state itself before calling its real run() — if blocked, it returns
the guard's warning text as the tool's own result (LangGraph wraps whatever a
tool returns into a ToolMessage automatically), so the model sees the same
kind of feedback it always did and the ReAct loop continues naturally.

reset() must be called once per turn, before the react agent's .invoke() —
see agent.py's run_turn().
"""
from __future__ import annotations

import json
import threading

import config
from helpers.guardrails import RepeatCallGuard

_guard = RepeatCallGuard(max_repeats=config.MAX_REPEAT_CALLS)
_tool_counts: dict[str, int] = {}
_tool_log: list[str] = []
_reservations: dict[tuple[str, str], int] = {}
_user_query = ""
_state_lock = threading.RLock()


def reset(user_query: str = "") -> None:
    global _guard, _tool_counts, _tool_log, _reservations, _user_query
    with _state_lock:
        _guard = RepeatCallGuard(max_repeats=config.MAX_REPEAT_CALLS)
        _tool_counts = {}
        _tool_log = []
        _reservations = {}
        _user_query = user_query


def _key(name: str, args: dict[str, str]) -> tuple[str, str]:
    return name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def check(name: str, args: dict[str, str]) -> str | None:
    """Call before executing a tool. Returns a block message (use as the
    tool's return value instead of running it) or None if it's fine to run.
    """
    # ToolNode may run several calls concurrently. Reserve the quota while
    # holding the lock, before the tool starts, so parallel calls cannot all
    # pass a stale count and exceed the per-turn cap.
    with _state_lock:
        if _tool_counts.get(name, 0) >= config.MAX_CALLS_PER_TOOL:
            return (
                f"[error] เรียก {name} ครบโควตาแล้ว ({config.MAX_CALLS_PER_TOOL} ครั้งต่อ turn) "
                f"ห้ามเรียกซ้ำอีก ใช้ข้อมูลที่มีอยู่แล้วตอบสรุปให้ผู้ใช้เลย"
            )
        guard_msg = _guard.check(name, args)
        if guard_msg == "__ABORT__":
            return "[error] ลองหลายวิธีกับ tool นี้แล้วยังไม่สำเร็จ ห้ามเรียกซ้ำอีก ให้สรุปบอกผู้ใช้ตรงๆ ว่าทำไม่สำเร็จ"
        if guard_msg:
            return f"[error] {guard_msg}"
        _tool_counts[name] = _tool_counts.get(name, 0) + 1
        key = _key(name, args)
        _reservations[key] = _reservations.get(key, 0) + 1
        return None


def record(name: str, args: dict[str, str], result: str) -> None:
    """Call after a tool actually executed (not when blocked by check())."""
    with _state_lock:
        key = _key(name, args)
        if _reservations.get(key, 0):
            _reservations[key] -= 1
            if not _reservations[key]:
                del _reservations[key]
        else:
            # Supports direct unit/tool use while preserving the wrapper's
            # check-then-record contract.
            _tool_counts[name] = _tool_counts.get(name, 0) + 1
        ok = not (result.startswith("[error]") or result.startswith("[blocked]"))
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        if len(args_str) > 150:
            args_str = args_str[:150] + "..."
        status = "สำเร็จ" if ok else "ล้มเหลว"
        _tool_log.append(f"- {name}({args_str}) -> {status}")


def get_tool_log() -> list[str]:
    with _state_lock:
        return list(_tool_log)


def tool_call_count() -> int:
    with _state_lock:
        return sum(_tool_counts.values())


def call_count(name: str) -> int:
    """Calls to `name` so far this turn, including the one just admitted by
    check() (it increments before returning). A wrapper can use this right
    after check() passes to detect "this is the first call this turn"."""
    with _state_lock:
        return _tool_counts.get(name, 0)


def user_query() -> str:
    with _state_lock:
        return _user_query
