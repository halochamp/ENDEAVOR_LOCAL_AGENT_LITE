"""date_inject.py — deterministic relative-date resolution, injected into the
per-turn user message rather than trusted to the model.

Same pattern as ENDEAVOR_VISSION/graph/chat_graph.py's _parse_datetime_context
(user-flagged reference, 2026-07-23): a small model asked "เมื่อวาน" or
"พรุ่งนี้" cannot reliably compute the actual calendar date itself, so a
Python dict/regex resolves it and appends a bracketed
"[DateTime context: date=YYYY-MM-DD]" note. One design choice carried over
deliberately:
  - Injected onto the per-turn USER message, never the system message — L50
    in the training guide: any per-turn-variable content inside the system
    message breaks prompt-cache prefix matching, and here specifically
    because our sole model call already re-renders the whole history anyway
    so there's no separate "cacheable system block" benefit to preserve.

Change 2026-07-26 (user finding): the note used to be appended only when a
relative-date word ("เมื่อวาน"/"พรุ่งนี้") was present — a "current events"
question (e.g. "กองทุนรวมปัจจุบัน") has no such word, so the model had no
grounding for "today" at all and answered with a vague multi-year range
pulled from training data / search snippets instead of the actual date. The
note is now appended on every turn; only its `resolved_date` field remains
conditional on an actual relative-date word being present.
"""
from __future__ import annotations
from datetime import date, datetime, timedelta

_THAI_WEEKDAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

_DAY_WORDS: dict[str, int] = {
    "เมื่อวานซืน": 2,
    "เมื่อวาน": 1,
    "วันนี้": 0,
    "พรุ่งนี้": -1,
    "มะรืนนี้": -2,
    "yesterday": 1,
    "today": 0,
    "tomorrow": -1,
}


def resolve_relative_date(text: str) -> str | None:
    """Return YYYY-MM-DD for the first recognized relative-date word in
    `text`, or None if none found. Longest words are checked first so
    'เมื่อวานซืน' matches before the 'เมื่อวาน' substring it contains.
    """
    today = date.today()
    for word in sorted(_DAY_WORDS, key=len, reverse=True):
        if word in text.lower():
            return (today - timedelta(days=_DAY_WORDS[word])).strftime("%Y-%m-%d")
    return None


def inject_datetime_context(user_input: str) -> str:
    """Return `user_input` with a trailing '[DateTime context: ...]' note.

    Always appended (see module docstring) so the model has real grounding
    for "today"/"now" on every turn, not just ones with a relative-date
    word. `resolved_date` is only added when such a word is actually found.
    """
    now = datetime.now()
    note = f"[DateTime context: today={now.date().isoformat()} ({_THAI_WEEKDAYS[now.weekday()]}), time={now.strftime('%H:%M')}"
    resolved = resolve_relative_date(user_input)
    if resolved is not None:
        note += f", resolved_date={resolved}"
    note += "]"
    return f"{user_input}\n{note}"
