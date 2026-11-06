"""ui.py — terminal UI, ported from ENDEAVOR_LOCAL_AGENT_MAX_VLM/ui_cli.py.

MAX_VLM's version (572 lines) is a full-featured multi-mode CLI: skills,
RAG-index building, plan display, ghost-text /command completion, a
prompt_toolkit Application-based input box. `rich`/`prompt_toolkit` are pure
terminal-rendering libraries (no model weights, negligible RAM), so porting
the polished *look* costs nothing against PLAN.md's RAM budget — what's
dropped here is scope the lite agent simply doesn't have: skills,
create_plan/multi-step planning, RAG-index-build UI, ghost-text completion.

2026-07-23 update (user request: "bring back all of MAX_VLM's ui_cli
graphics — a live status telling the user what the agent is doing while it
waits"): restored MAX_VLM's real two-line Spinner (main phase label + a
sub-status line showing exactly which tool is running right now, not just a
generic "thinking…"), driven by agent.run_turn()'s on_step callback which
now streams LangGraph's per-node events (react.py/agent.py) instead of the
single-shot .invoke() the first port used — that first port could only show
one static spinner for the whole turn because .invoke() doesn't yield
anything until the turn is completely done. Also restored web-reference
extraction/display (MAX_VLM's print_web_refs + extract_refs_from_search_result)
since tools/web_search.py's own "[source: URL]" convention maps onto it
directly. Kept dropped: skills, create_plan/multi-step planning, RAG-index-
build UI, ghost-text completion, and the full prompt_toolkit input-box
Application (its main value — live ghost-text — doesn't apply here; the
input line stays a plain colored `input()` call in ui_cli.py).
"""
from __future__ import annotations
import itertools
import re
import sys
import threading
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

import agent
import config

_theme = Theme({
    "markdown.h1": "bold white",
    "markdown.h2": "bold white",
    "markdown.h3": "bold white",
    "markdown.code": "bold cyan",
    "markdown.code_block": "dim",
})
_console = Console(theme=_theme, highlight=False)

R, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
C_USER = "\033[38;5;75m"
C_AGENT = "\033[38;5;86m"
C_TOOL = "\033[38;5;208m"
C_META = "\033[38;5;244m"
C_ARROW = "\033[38;5;240m"
C_WARN = "\033[38;5;214m"
C_DIM = "\033[38;5;237m"
C_HEADER = "\033[38;5;252m"
C_ORANGE = "\033[38;5;208m"
C_RED = "\033[38;5;196m"
C_REF = "\033[38;5;117m"

WIDTH = 62

_TOOL_LABELS = {
    "bash": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "web_search": "Web Search",
    "local_search": "Desktop/Documents Search",
    "rag_search": "File Search",
    "read_image": "Read Image",
}

_IO_LOCK = threading.RLock()


# ── Header ───────────────────────────────────────────────────────────────
def print_header(model: str, tool_count: int) -> None:
    short = model.split("/")[-1]
    inner = WIDTH - 2
    bar = f" {C_DIM}{'═' * WIDTH}{R}"
    title_txt = "  ENDEAVOR AGENT LITE"
    model_txt = f"  Model  {short}"
    pad_title = max(0, inner - len(title_txt))
    pad_model = max(0, inner - len(model_txt))
    print()
    print(bar)
    print(f" {C_DIM}║{R}  {BOLD}\033[38;5;220mENDEAVOR AGENT LITE{R}{C_DIM}{'':>{pad_title - 20}}║{R}")
    print(f" {C_DIM}║{R}  {C_META}Model  {R}{BOLD}{C_AGENT}{short}{R}{C_DIM}{'':>{pad_model - 8 - len(short)}}║{R}")
    print(bar)
    print(f"   {C_META}{tool_count} tools{R}  {C_META}via mlx_vlm.server{R}")
    print()


def print_divider() -> None:
    print(f"\n{C_DIM}{'╌' * (WIDTH // 2)}{R}\n")


# ── Turn rendering ───────────────────────────────────────────────────────
def print_user_prompt(text: str) -> None:
    print(f" {C_USER}{BOLD}คุณ{R}  {text}")
    print()


def print_tool_step(name: str, ok: bool) -> None:
    label = _TOOL_LABELS.get(name, name)
    status = f"{C_AGENT}✓{R}" if ok else f"{C_RED}✗{R}"
    print(f"   {C_TOOL}⎿ {label}{R} {status}")


def print_agent_response(text: str) -> None:
    print()
    _console.print(Markdown(text), end="\n")


# ── Web references ──────────────────────────────────────────────────────
def extract_refs_from_search_result(text: str) -> list[tuple[str, str]]:
    """Extract ``(summary, url)`` from this project's web-search output."""
    refs: list[tuple[str, str]] = []
    previous = ""
    seen: set[str] = set()
    for line in (text or "").splitlines():
        match = re.match(r"^\s*\[source:\s*(https?://[^\]]+)\]\s*$", line)
        if match:
            url = match.group(1)
            if url not in seen:
                seen.add(url)
                title = previous.lstrip("- ").strip()[:55] or url
                refs.append((title, url))
        elif line.strip():
            previous = line.strip()
    return refs


def print_web_refs(refs: list[tuple[str, str]]) -> None:
    """Render de-duplicated source links below the final answer."""
    if not refs:
        return
    print(f"\n   {C_META}📎 แหล่งอ้างอิง{R}")
    for i, (title, url) in enumerate(refs, 1):
        print(f"   {C_ARROW}{DIM} {i}. {title[:55] if title else url}{R}")
        print(f"      {C_REF}{url}{R}")


def _history_budget_line(messages: list[dict]) -> str:
    system = messages[:1] if messages and messages[0].get("role") == "system" else []
    turns = (len(messages) - len(system)) // 2
    chars = sum(len(m.get("content", "") or "") for m in messages[len(system):])
    pct = chars / agent.MAX_HISTORY_CHARS * 100 if agent.MAX_HISTORY_CHARS else 0
    c_pct = C_RED if pct >= 90 else (C_WARN if pct >= 70 else C_META)
    filled = round(min(pct, 100) / 100 * 18)
    bar = "▓" * filled + "░" * (18 - filled)

    def _fmt(v: int) -> str:
        return f"{v / 1000:.1f}k" if v >= 1000 else str(v)

    return (
        f" {C_DIM}⎿{R}  {C_META}{turns}/{agent.MAX_HISTORY_TURNS} turns{R}  "
        f"{c_pct}{pct:.0f}%{R}  {C_DIM}{bar}{R}  "
        f"{C_META}{_fmt(chars)}/{_fmt(agent.MAX_HISTORY_CHARS)} chars (compact history){R}"
    )


def print_history_bar(messages: list[dict]) -> None:
    print(_history_budget_line(messages))


def index_stale_progress_text(path, stats: dict[str, int]) -> str:
    """Compact status for the pre-scan stale-entry cleanup pass, which has
    no scanned/inserted/updated counts of its own yet -- only how far
    through the existing index it has checked."""
    current = str(path) if path is not None else "…"
    if len(current) > 46:
        current = "…" + current[-45:]
    return (
        f"กำลังตรวจ index เดิม {stats['stale_checked']:,}/{stats['stale_total']:,} · {current}"
    )


def index_progress_text(path, stats: dict[str, int]) -> str:
    """Compact current-indexing status for Spinner's live second line."""
    current = str(path) if path is not None else "กำลังเตรียม index…"
    if len(current) > 46:
        current = "…" + current[-45:]
    return (
        f"{stats['scanned']:,} scanned · {stats['inserted']:,} new · "
        f"{stats['updated']:,} updated · {current}"
    )


def sync_kb_progress_text(event: str, path, stats: dict[str, int | float]) -> str:
    """Render the non-LLM Knowledge Base sync phase and estimated progress."""
    percent = float(stats.get("progress_percent", 0) or 0)
    total = int(stats.get("total_files", 0) or 0)
    completed = int(stats.get("completed_files", 0) or 0)
    eta = float(stats.get("estimated_remaining_seconds", 0) or 0)
    current = str(path) if path is not None else ""
    if len(current) > 42:
        current = "…" + current[-41:]
    if event == "prepare":
        return "กำลังรวบรวมรายการไฟล์ Knowledge Base…"
    if event == "prepared":
        return f"พบ {total:,} ไฟล์ · เริ่ม sync · ประมาณ 0%"
    if event == "file":
        detail = f"กำลังอ่าน/วิเคราะห์ {current or 'ไฟล์ถัดไป'}"
    elif event == "result":
        detail = f"กำลังบันทึกผล {current or 'ไฟล์ล่าสุด'}"
    elif event == "cleanup":
        detail = f"กำลังเก็บกวาดรายการเก่า {current or ''}".rstrip()
    elif event == "semantic":
        detail = "กำลังสร้าง semantic vectors"
    elif event == "done":
        return (
            f"sync เสร็จ · 100% · {completed:,}/{total:,} ไฟล์ · "
            f"{int(stats.get('documents', 0) or 0):,} documents"
        )
    else:
        detail = "กำลังเตรียม SQLite…"
    progress = f"ประมาณ {percent:.0f}%"
    counts = f"{completed:,}/{total:,} ไฟล์"
    remaining = f" · เหลือ ~{eta:.1f}s" if eta > 0 else ""
    return f"{detail} · {progress} · {counts}{remaining}"


def print_index_summary(stats: dict[str, int]) -> None:
    print(
        f"\n {C_AGENT}✓ สร้าง RAG index เสร็จ{R}  "
        f"{stats['scanned']:,} scanned · {stats['inserted']:,} new · "
        f"{stats['updated']:,} updated · {stats['skipped']:,} skipped · "
        f"{stats['error']:,} errors · {stats['seconds']:.2f}s"
    )


def print_sync_kb_summary(stats: dict[str, int]) -> None:
    """Render the knowledge-root -> chunk-level SQLite sync result."""
    print(
        f"\n {C_AGENT}✓ sync_kb เสร็จ{R}  "
        f"source {stats.get('raw_scanned', 0):,} ไฟล์ · "
        f"index {stats.get('indexed', 0):,} · "
        f"ไม่เปลี่ยน {stats.get('unchanged', 0):,} · "
        f"ลบ {stats.get('removed', 0):,} · "
        f"documents {stats.get('documents', 0):,} · chunks {stats.get('chunks', 0):,} · "
        f"SQLite {stats.get('inserted', 0):,} new/{stats.get('updated', 0):,} updated · "
        f"errors {stats.get('sync_errors', 0):,} · "
        f"{stats.get('seconds', 0):.2f}s"
    )


# ── Spinner ──────────────────────────────────────────────────────────────
class Spinner:
    """MAX_VLM-style two-line spinner: phase plus the current tool action."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "กำลังคิด…"):
        self.label = label
        self.sub = "กำลังรอการตอบจากโมเดล…"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True, name="spinner")
        self._lines = 0
        # \r + \033[2K (carriage-return-and-clear) only overwrite in place on
        # a real terminal — piped/non-interactive stdout (e.g. a test
        # harness, `| tee`, or a log file) just prints each frame as a new
        # line instead. Skip the animation there; live_print()'s tool-step
        # lines and the final answer still print normally either way.
        self._enabled = sys.stdout.isatty()

    def _clear(self) -> None:
        if not self._lines:
            return
        parts: list[str] = []
        for i in range(self._lines):
            parts.append("\r\033[2K")
            if i < self._lines - 1:
                parts.append("\033[F")
        sys.stdout.write("".join(parts))
        sys.stdout.flush()
        self._lines = 0

    def _draw(self, frame: str) -> None:
        self._clear()
        lines = [f"   {C_META}{frame} {self.label}{R}"]
        if self.sub:
            lines.append(f"      {C_ARROW}{DIM}⎿  {self.sub[:70]}{R}")
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()
        self._lines = len(lines)

    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            with _IO_LOCK:
                if self._stop.is_set():
                    break
                self._draw(frame)
            time.sleep(0.08)
        with _IO_LOCK:
            self._clear()

    def __enter__(self):
        if self._enabled:
            self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._enabled:
            self._thread.join(timeout=2.0)

    def update(self, label: str) -> None:
        with _IO_LOCK:
            self.label = label

    def update_sub(self, message: str) -> None:
        with _IO_LOCK:
            self.sub = message or ""

    def live_print(self, fn) -> None:
        """Run a main-thread print without racing the spinner's redraw."""
        with _IO_LOCK:
            self._clear()
            fn()


def make_on_step(spinner: Spinner):
    """Build the on_step callback agent.run_turn() calls — updates the
    spinner label and prints a ⎿ tool-step line live, without interleaving
    with the spinner's own redraw (both go through spinner.live_print)."""

    def on_step(event: str, **kw) -> None:
        if event == "thinking":
            spinner.update("กำลังคิด…")
            spinner.update_sub("กำลังวิเคราะห์คำขอ…")
        elif event == "tool_start":
            label = _TOOL_LABELS.get(kw["name"], kw["name"])
            spinner.update("กำลังใช้เครื่องมือ…")
            spinner.update_sub(f"{label}: กำลังทำงาน…")
        elif event == "tool_done":
            spinner.live_print(lambda: print_tool_step(kw["name"], kw["ok"]))
            spinner.update("กำลังคิด…")
            spinner.update_sub("กำลังประมวลผลผลลัพธ์…")

    return on_step
