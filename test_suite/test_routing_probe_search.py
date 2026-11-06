"""test_routing_probe_search.py — turn-1 tool-selection walking test for
file-search (lite-rag MCP/bash/read_file) and internet-search (web_search)
routing, per agent_training_guide/sum_agent_training_guide.md's "Routing
probe pattern" (§ Validating a Prompt Change with an Executable Eval).

Why this shape, not full E2E (test_agent_live.py already covers 6 broad
scenarios end-to-end): a routing probe measures ONLY which tool turn 1
emits — one model call per case instead of a full multi-step turn, so the
whole set runs in seconds and is cheap enough to re-run after every prompt
edit. It measures routing divergence, not task success — a case can pick
the "wrong" turn-1 tool and still finish correctly downstream; that's a
routing finding, not a pass/fail verdict on the agent.

Cases are grouped by the exact distinctions prompts.py's SYSTEM_PROMPT
rules 2, 3, and 9 draw, since those are the boundaries most likely to
blur for a 2B model:
  - default RAG-first vs explicit-user-requested bash-first
  - lite-rag.rag_search via MCP then bash (list/count/full-content fallback trajectory)
  - lite-rag.rag_search via MCP + read_file vs bash (combined "find X and summarize" query)
  - web_search vs no-tool (real-time info vs general/opinion knowledge)
  - web_search vs lite-rag MCP (must not divert a local file request to the
    internet just because no RAG index exists yet)
  - no-tool distractors (chit-chat) that must not trigger ANY tool

Needs the real model server (mlx_vlm.server, config.MLX_BASE_URL). Not a
pytest-asserts suite for the same reason test_agent_live.py isn't one: a 2B
model's tool choice has real variance run-to-run. Run directly and read the
per-category score; use `--repeat N` to average out noise on borderline
cases before concluding a routing bug (training guide: "run 2-3 times
before concluding, normal variance").

Run: python test_suite/test_routing_probe_search.py
     python test_suite/test_routing_probe_search.py --repeat 3
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict

from langchain_core.messages import SystemMessage, HumanMessage

from llm import build_llm
from tools import ALL_TOOLS
import prompts

# (name, category, query, expected_tool_or_None, note)
# expected_tool=None means "no tool call at all" is correct.
CASES: list[tuple[str, str, str, str | None, str]] = [
    # Document-name lookup.
    ("doc-name-1", "document/name", "ใบเสร็จเดือนกรกฎาคม.pdf อยู่ที่ไหน", "mcp_call_tool",
     "exact PDF name"),
    ("doc-name-2", "document/name", "หาไฟล์สัญญาเช่า.docx ให้หน่อย", "mcp_call_tool",
     "casual exact Word filename"),
    ("doc-name-3", "document/name", "มีรายงานประจำปี.xlsx ในเครื่องไหม", "mcp_call_tool",
     "yes/no exact spreadsheet lookup"),

    # Combined locate + read/summarize still starts at RAG.
    ("doc-combo-1", "document/combo",
     "สัญญาเช่า.docx อยู่ที่ไหน แล้วสรุปเงื่อนไขสำคัญให้หน่อย",
     "mcp_call_tool", "locate then read document"),
    ("doc-combo-2", "document/combo",
     "เอกสารไหนพูดถึงค่าปรับยกเลิกสัญญา แล้วช่วยสรุปให้ด้วย",
     "mcp_call_tool", "content lookup then explain"),

    # Content lookup is RAG-first; bash is a later fallback.
    ("doc-content-1", "document/content", "มีเอกสารไหนพูดถึงค่าปรับยกเลิกสัญญาบ้าง", "mcp_call_tool",
     "document content phrase"),
    ("doc-content-2", "document/content", "หาเอกสารที่มีคำว่า ยอดชำระ 4,500 บาท", "mcp_call_tool",
     "invoice-like content phrase"),
    ("doc-content-3", "document/content", "เอกสารไหนกล่าวถึงวันครบกำหนดชำระเงิน", "mcp_call_tool",
     "semantic-looking but lexical document request"),
    ("doc-machine-1", "document/machine-content", "ค้นหาทั้งเครื่องที่มีคำว่า เลขที่กรมธรรม์", "mcp_call_tool",
     "whole-machine scope still begins with RAG"),
    ("doc-machine-2", "document/machine-content", "ค้นทุกไฟล์ใน Mac ที่พูดถึงผู้รับผลประโยชน์", "mcp_call_tool",
     "alternate whole-machine phrasing"),

    # List/count needs RAG first, then bash for completeness.
    ("doc-count-1", "document/count", "มีเอกสาร PDF กี่ไฟล์ใน Documents", "mcp_call_tool",
     "RAG first; bash later provides complete count"),
    ("doc-count-2", "document/count", "นับไฟล์ Excel ทั้งหมดใน Documents ให้หน่อย", "mcp_call_tool",
     "alternate count phrasing"),
    ("doc-list-1", "document/list", "list เอกสาร Word ทั้งหมดใน Documents", "mcp_call_tool",
     "RAG first; bash later provides complete list"),
    ("doc-list-2", "document/list", "ขอดูรายชื่อไฟล์ PowerPoint ใน Downloads ทั้งหมด", "mcp_call_tool",
     "natural-language list request"),

    # Explicit bash instruction is the only bash-first route.
    ("bash-explicit-1", "bash/explicit", "ใช้ bash ค้นหาทั้งเครื่องที่มีคำว่า เลขที่กรมธรรม์", "bash",
     "explicit whole-machine bash search"),
    ("bash-explicit-2", "bash/explicit", "ใช้ bash list เอกสาร PDF ใน Documents", "bash",
     "explicit PDF list"),
    ("bash-explicit-3", "bash/explicit", "รัน bash เพื่อนับไฟล์ Excel ใน Documents", "bash",
     "explicit spreadsheet count"),
    ("bash-explicit-4", "bash/explicit", "ใช้ bash ดูขนาดโฟลเดอร์ Documents", "bash",
     "explicit directory-size inspection"),
    ("bash-explicit-5", "bash/explicit", "ช่วยใช้ bash ดูไฟล์ PDF ใน Downloads", "bash",
     "explicit PDF listing in a second standard document root"),

    # A known document path goes directly to read_file.
    ("doc-readpath-1", "document/readpath",
     "อ่านไฟล์ /Users/example/Documents/รายงานประจำปี.pdf ให้หน่อย",
     "read_file", "path is already known"),

    # Current internet info remains web_search.
    ("web-realtime-1", "web/realtime", "วันนี้อากาศที่กรุงเทพเป็นยังไงบ้าง", "web_search",
     "live weather"),
    ("web-realtime-2", "web/realtime", "อัตราดอกเบี้ยนโยบายไทยล่าสุดเท่าไร พร้อมแหล่งที่มา", "web_search",
     "current sourced financial fact"),
    ("web-realtime-3", "web/realtime", "ข่าวเศรษฐกิจวันนี้มีอะไรสำคัญบ้าง", "web_search",
     "today's news"),

    # General knowledge/opinion requires no tool.
    ("web-noop-1", "web/no-tool", "คิดว่าจัดเอกสารตามปีอ่านง่ายไหม", None,
     "stable opinion with no external facts needed"),
    ("web-noop-2", "web/no-tool", "คิดว่า PDF กับ Word แบบไหนเหมาะส่งรายงานมากกว่า", None,
     "general comparison"),
    ("web-noop-3", "web/no-tool", "อธิบายดอกเบี้ยทบต้นคืออะไร", None,
     "stable concept"),

    # Local documents must never divert to the web.
    ("web-not-diverted-1", "web/not-diverted",
     "หาเอกสารในเครื่องที่เกี่ยวกับเงื่อนไขต่ออายุประกันให้หน่อย",
     "mcp_call_tool", "local content lookup, not internet research"),

    ("chitchat-1", "no-tool", "สวัสดีครับ วันนี้เป็นไงบ้าง", None, "greeting"),
    ("chitchat-2", "no-tool", "ขอบคุณมากครับ ช่วยได้เยอะเลย", None, "thanks"),
]


def probe(llm, query: str) -> tuple[str | None, list[str]]:
    resp = llm.invoke([SystemMessage(content=prompts.SYSTEM_PROMPT), HumanMessage(content=query)])
    calls = getattr(resp, "tool_calls", None) or []
    names = [c.get("name") for c in calls]
    return (names[0] if names else None), names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1, help="repeat each case N times, report majority vote")
    parser.add_argument(
        "--category",
        action="append",
        choices=sorted({case[1] for case in CASES}),
        help="run only this category (repeatable); keeps long stress runs below an external command timeout",
    )
    args = parser.parse_args()
    active_cases = [case for case in CASES if not args.category or case[1] in args.category]

    # Runtime policy: every request, including tool routing, uses the bounded
    # thinking budget configured in config.py.  Do not make this probe silently
    # test a non-thinking path that production never uses.
    llm = build_llm(temperature=0).bind_tools(ALL_TOOLS)

    results: list[tuple[str, str, bool, str, list[str]]] = []
    for name, category, query, expected, note in active_cases:
        votes: list[str | None] = []
        all_names: list[list[str]] = []
        for _ in range(args.repeat):
            got, names = probe(llm, query)
            votes.append(got)
            all_names.append(names)
        majority = max(set(votes), key=votes.count)
        ok = majority == expected
        results.append((name, category, ok, f"expected={expected} got={majority} votes={votes}", note))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:22s} ({category:16s}) "
              f"expected={str(expected):12s} got={str(majority):12s} votes={votes}", flush=True)
        if not ok:
            print(f"           note: {note}")

    print(f"\n{'='*78}\nSUMMARY BY CATEGORY\n{'='*78}", flush=True)
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for _, category, ok, _, _ in results:
        by_cat[category].append(ok)
    for category, oks in sorted(by_cat.items()):
        print(f"  {category:18s} {sum(oks)}/{len(oks)}")

    total_ok = sum(1 for _, _, ok, _, _ in results if ok)
    print(f"\nTOTAL: {total_ok}/{len(results)} passed", flush=True)
    if total_ok < len(results):
        print("\nFAILED CASES:")
        for name, category, ok, detail, note in results:
            if not ok:
                print(f"  - {name} ({category}): {detail}\n    note: {note}")


if __name__ == "__main__":
    main()
