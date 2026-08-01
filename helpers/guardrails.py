# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""guardrails.py — code-level safety net for a small model's agent loop.

Rationale (see PLAN.md §6): live testing on Qwen3.5-2B-OptiQ-4bit showed it
cannot reliably self-correct after a tool failure — it re-issued an
identical failing `find /` command four times in a row rather than
narrowing scope. A 2B model is not going to get smarter via better prompting
alone, so these checks run in Python, not in the prompt.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import re


_CONSECUTIVE_REPETITION_RE = re.compile(r"(.{12,200}?)(?:\1){3,}", re.DOTALL)


def _call_hash(tool_name: str, args: dict[str, str]) -> str:
    key = tool_name + "|" + "|".join(f"{k}={v}" for k, v in sorted(args.items()))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class RepeatCallGuard:
    """Tracks identical (tool, args) calls within a single agent turn."""

    max_repeats: int = 2
    _counts: dict[str, int] = field(default_factory=dict)

    def check(self, tool_name: str, args: dict[str, str]) -> str | None:
        """Register this call attempt. Returns a Thai feedback string if the
        call should be BLOCKED (not executed) for repeating too often, or
        None if it's fine to run.
        """
        h = _call_hash(tool_name, args)
        self._counts[h] = self._counts.get(h, 0) + 1
        count = self._counts[h]
        if count == 2:
            return (
                f"คำสั่งนี้ ({tool_name}) เพิ่งล้มเหลว/ไม่ได้ผลไปแล้ว ห้ามรันซ้ำแบบเดิม "
                f"ให้เปลี่ยนวิธี เช่น จำกัด scope ให้แคบลง แก้คำค้น หรือถ้าไม่มีทางอื่นให้สรุปบอกผู้ใช้ตรงๆ ว่าหาไม่เจอ"
            )
        if count >= self.max_repeats + 1:
            return "__ABORT__"
        return None

    def reset(self) -> None:
        self._counts.clear()


def is_degenerate_repetition(text: str, chunk_len: int = 12, min_repeats: int = 4) -> bool:
    """Detect a small model spiraling into repeating the same phrase — live-
    observed 2026-07-23: asked to search file content, Qwen3.5-2B emitted a
    growing list of the same few file-extension tokens ("*.gz" "*.bz2" "*.xz"
    ...) dozens of times, burning the whole max_tokens budget before its
    <tool_call> XML could even close (parse_tool_call then sees unclosed tags
    and the garbled text gets returned as if it were a final answer).

    Look for a consecutive repeated span in the tail.  Counting a short span
    anywhere in the answer is too broad: a perfectly valid list of absolute
    paths repeats the project-directory prefix on every line and used to be
    rejected once the list reached roughly 8-10 entries.  A generation loop,
    by contrast, repeats the same phrase *back-to-back*.
    """
    if len(text) < chunk_len * min_repeats:
        return False
    tail = re.sub(r"\s+", " ", text[-2000:]).strip()
    # Keep the public parameters meaningful for the default production
    # policy while avoiding a dynamically-built regex on every model turn.
    if chunk_len == 12 and min_repeats == 4:
        return _CONSECUTIVE_REPETITION_RE.search(tail) is not None
    max_unit_len = min(200, len(tail) // min_repeats)
    for unit_len in range(chunk_len, max_unit_len + 1):
        repeated_len = unit_len * min_repeats
        for start in range(0, len(tail) - repeated_len + 1):
            unit = tail[start : start + unit_len]
            if tail.startswith(unit * min_repeats, start):
                return True
    return False


_LEAKED_TOOL_SYNTAX_MARKERS = ("<tool_call>", "<function=", "<parameter=")


def has_leaked_tool_syntax(text: str) -> bool:
    """Detect the model emitting its own tool-call XML as plain text instead
    of a real structured tool call (create_react_agent/bind_tools never
    needs these literal tags in visible content -- their presence at all is
    itself the signal, no length/repetition threshold needed). Live-observed
    2026-07-25 (/walk T02, 3rd repeat): a `<tool_call><function=bash>
    <parameter=command>...` block never resolved into an actual tool
    invocation (tool_calls=0 logged) and spiraled into is_degenerate_repetition
    territory with fabricated file paths from an unrelated project on the
    user's Desktop -- the two checks are complementary, this one also catches
    a short leaked fragment that hasn't repeated enough to trip that one yet.
    """
    return any(marker in text for marker in _LEAKED_TOOL_SYNTAX_MARKERS)


def truncate_output(text: str, max_chars: int, saved_path: str | None = None) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    note = f"\n[ตัดที่ {max_chars} chars"
    if saved_path:
        note += f" — full output: {saved_path}"
    note += "]"
    return cut + note


_THAI_CHAR_RE = re.compile(r"[฀-๿]")
_REFUSAL_HINTS = (
    "need more steps",
    "sorry, i",
    "i cannot",
    "i am unable",
    "i'm unable",
    "as an ai",
)


def is_boilerplate_refusal(text: str) -> bool:
    """Detect a canned English give-up phrase in place of a real answer.

    Live-observed 2026-07-25 (workspect/.lite_state/logs/session.jsonl):
    after tool calls already ran (some succeeded, some failed cleanly),
    Qwen3.5-2B-OptiQ-4bit sometimes returned "Sorry, need more steps to
    process this request." instead of following the system prompt's own
    instruction to summarize success/failure in Thai. That exact string
    does not appear anywhere in this codebase -- it's the model's own
    generated text, not a code-level message -- and it recurred across
    unrelated turns/tools, so it's a model failure mode, not a one-off.
    Guarded by "no Thai script + short" so a legitimately short English
    answer (e.g. a URL or code snippet) isn't misflagged.
    """
    if not text or len(text) > 200:
        return False
    if _THAI_CHAR_RE.search(text):
        return False
    lowered = text.lower()
    return any(hint in lowered for hint in _REFUSAL_HINTS)


@dataclass
class StepBudget:
    max_steps: int
    used: int = 0

    def take(self) -> bool:
        """Consume one step. Returns False if the budget is exhausted."""
        if self.used >= self.max_steps:
            return False
        self.used += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.used)
