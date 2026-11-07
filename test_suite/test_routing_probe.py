"""test_routing_probe.py — Phase 7.1 (PLAN.md §Phase 7): turn-1 routing probe.

Needs the real model loaded — not part of the LLM-free unit suite (run
directly: `python test_suite/test_routing_probe.py`, or with `--rounds N`).

Measures ONLY which tool the model picks on its very first generate() call
for each query — not full end-to-end task success (that's
test_agent_live.py's job). This is the L42 pattern from the training guide:
a routing probe is 5-10x faster than a full ReAct loop per query, so it's
cheap enough to run 3x and check for flakiness before trusting a result —
exactly the tool that was missing when prefill_step_size=128 was tried and
reverted (PLAN.md §Phase 7 background, ENDMEMEX checkpoint 284).

Each case's `expect` is either a tool name (must be called) or None (must
NOT call any tool). Cases without a clean single right answer use
expect="any" and are reported but never fail the pass bar — they're here to
watch for drift, not to gate on.
"""
from __future__ import annotations
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prompts
from tools import ALL_TOOLS
from helpers.date_inject import inject_datetime_context
from llm import build_llm
from langchain_core.messages import SystemMessage, HumanMessage

# Architecture switch 2026-07-23 (config.py's module docstring): probe now
# binds tools via LangChain's bind_tools() against the real mlx_vlm.server,
# reading the structured tool_calls field directly instead of regex-parsing
# XML text — the in-process llm.generate()/parse_tool_call() path this file
# used to test no longer exists.
_llm_with_tools = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = build_llm().bind_tools(ALL_TOOLS)
    return _llm_with_tools

CASES: list[tuple[str, str, str | None]] = [
    ("find_by_name_1", "PLAN.md อยู่ที่ไหนในโปรเจกต์ ENDEAVOR_LOCAL_AGENT_LITE (บอก path เต็ม) แล้วสรุปว่าไฟล์นี้พูดถึงอะไร", "mcp_call_tool"),
    ("find_by_name_2", "หาไฟล์ config.py อยู่ที่ไหน", "mcp_call_tool"),
    ("find_by_content", "มีไฟล์ไหนในโปรเจกต์นี้พูดถึง trigram tokenizer บ้าง", "mcp_call_tool"),
    ("web_search", "ช่วยค้นหาว่า Python 3.13 มีฟีเจอร์ใหม่อะไรบ้าง พร้อมแนบลิงก์ที่มา", "web_search"),
    ("chitchat", "สวัสดีครับ วันนี้เป็นไงบ้าง", None),
    ("read_image", "อ่านรูปนี้ให้หน่อย /tmp/agent_live_test_image.png มีข้อความอะไรบ้าง", "read_image"),
    ("bash_count_files", "ดูหน่อยว่ามีไฟล์ python กี่ไฟล์ในโฟลเดอร์ ENDMEMEX", "bash"),
    ("missing_file", "อ่านไฟล์ /nonexistent_absolutely_not_real_path_xyz/ghost.txt ให้หน่อย", "any"),
]


def probe_turn1(query: str) -> str | None:
    messages = [
        SystemMessage(content=prompts.SYSTEM_PROMPT),
        HumanMessage(content=inject_datetime_context(query)),
    ]
    result = _get_llm_with_tools().invoke(messages)
    tool_calls = getattr(result, "tool_calls", None) or []
    return tool_calls[0]["name"] if tool_calls else None


def run(rounds: int) -> dict[str, list[str | None]]:
    results: dict[str, list[str | None]] = {name: [] for name, _, _ in CASES}
    for round_n in range(1, rounds + 1):
        print(f"\n=== round {round_n}/{rounds} ===")
        for name, query, expect in CASES:
            got = probe_turn1(query)
            results[name].append(got)
            print(f"  {name:20} got={got!s:15} expect={expect!s:12}")
    return results


def summarize(results: dict[str, list[str | None]]) -> bool:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_stable = True
    for name, query, expect in CASES:
        got_list = results[name]
        stable = len(set(got_list)) == 1
        matches = expect == "any" or got_list[0] == expect
        status = "STABLE" if stable else "FLAKY"
        verdict = "" if expect == "any" else (" MATCH" if matches else " MISMATCH")
        if not stable or (expect != "any" and not matches):
            all_stable = False
        print(f"  [{status}{verdict}] {name:20} {got_list}")
    print(f"\nall cases stable & matching expectation: {all_stable}")
    return all_stable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    results = run(args.rounds)
    ok = summarize(results)
    raise SystemExit(0 if ok else 1)
