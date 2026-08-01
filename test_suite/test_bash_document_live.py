# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Live trajectory eval for document discovery and bash argument quality.

The turn-1 routing probe proves which tool is selected first.  This script
tests the production ReAct loop as well: default document discovery must take
the rag_search -> bash path when RAG cannot produce a complete answer, while
an explicit user request may start with bash.  It also validates the actual
read-only command emitted by the 2B model.

Run:
    python test_suite/test_bash_document_live.py
    python test_suite/test_bash_document_live.py --repeat 3
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
import config
from tools import bash as bash_tool


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    expected_prefix: tuple[str, ...]
    command_check: Callable[[str], bool]
    requirement: str


def _document_root() -> Path:
    return next(
        (path for path in config.INDEX_ROOTS if path.name == "Documents"),
        config.INDEX_ROOTS[0] if config.INDEX_ROOTS else config.PROJECT_ROOT,
    ).resolve()


def _downloads_root() -> Path:
    return next(
        (path for path in config.INDEX_ROOTS if path.name == "Downloads"),
        Path.home() / "Downloads",
    ).resolve()


DOCUMENTS = str(_document_root())
DOWNLOADS = str(_downloads_root())


def _safe(command: str) -> bool:
    return bool(command) and bash_tool._is_denied(command) is None and bash_tool._read_only_violation(command) is None


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _is_find(command: str, root: str, suffix: str, *, count: bool) -> bool:
    tokens = _tokens(command)
    return (
        _safe(command)
        and bool(tokens)
        and tokens[0] == "find"
        and root in tokens
        and "-type" in tokens
        and "f" in tokens
        and any(suffix in token.casefold() for token in tokens)
        and (("wc" in tokens and "-l" in tokens) if count else True)
    )


def _is_mdfind(command: str) -> bool:
    tokens = _tokens(command)
    return _safe(command) and bool(tokens) and tokens[0] == "mdfind" and DOCUMENTS not in command


CASES = [
    Case(
        "explicit-whole-machine-content",
        "ใช้ bash ค้นหาทั้งเครื่องที่มีคำว่า เอกสารทดสอบซอร์กซิฟ-ไม่มีจริง-92841",
        ("bash",),
        _is_mdfind,
        "explicit bash starts with a whole-machine mdfind command",
    ),
    Case(
        "explicit-pdf-list",
        "ใช้ bash list เอกสาร PDF ใน Documents",
        ("bash",),
        lambda command: _is_find(command, DOCUMENTS, ".pdf", count=False),
        "explicit bash emits a safe absolute-path PDF find",
    ),
    Case(
        "explicit-excel-count",
        "รัน bash เพื่อนับไฟล์ Excel ใน Documents",
        ("bash",),
        lambda command: _is_find(command, DOCUMENTS, ".xls", count=True),
        "explicit bash emits a safe absolute-path spreadsheet count",
    ),
    Case(
        "rag-first-pdf-count",
        "มีเอกสาร PDF กี่ไฟล์ใน Documents",
        ("rag_search", "bash"),
        lambda command: _is_find(command, DOCUMENTS, ".pdf", count=True),
        "default count uses RAG first, then a complete bash count",
    ),
    Case(
        "rag-first-whole-machine-fallback",
        "ค้นหาทั้งเครื่องที่มีคำว่า เอกสารทดสอบซอร์กซิฟ-ไม่มีจริง-92841",
        ("rag_search", "bash"),
        _is_mdfind,
        "missing whole-machine content falls back from RAG to mdfind",
    ),
    Case(
        "explicit-downloads-list",
        "ช่วยใช้ bash ดูรายการเอกสารใน Downloads",
        ("bash",),
        lambda command: _safe(command)
        and bool(_tokens(command))
        and _tokens(command)[0] in {"find", "ls"}
        and DOWNLOADS in _tokens(command),
        "explicit directory listing uses the real absolute Downloads path",
    ),
]


def _run(case: Case) -> tuple[bool, str]:
    events: list[dict] = []
    result = agent.run_turn(
        case.query,
        [],
        on_step=lambda **event: events.append(event),
    )
    starts = [event for event in events if event.get("event") == "tool_start"]
    completions = [event for event in events if event.get("event") == "tool_done"]
    names = [str(event.get("name")) for event in starts]
    bash_commands = [
        str((event.get("args") or {}).get("command") or "")
        for event in starts
        if event.get("name") == "bash"
    ]
    trajectory_ok = tuple(names) == case.expected_prefix
    command_ok = bool(bash_commands) and case.command_check(bash_commands[0])
    final_bash_ok = bool(completions) and completions[-1].get("name") == "bash" and completions[-1].get("ok") is True
    ok = trajectory_ok and command_ok and final_bash_ok and not result.aborted
    detail = (
        f"tools={names} completions={[(event.get('name'), event.get('ok')) for event in completions]} "
        f"command={bash_commands[0] if bash_commands else None!r} "
        f"aborted={result.aborted}; {case.requirement}"
    )
    return ok, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    passed = 0
    total = len(CASES) * args.repeat
    for round_number in range(1, args.repeat + 1):
        for case in CASES:
            ok, detail = _run(case)
            passed += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] round={round_number} {case.name}: {detail}")
    print(f"\nTOTAL: {passed}/{total} passed")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
