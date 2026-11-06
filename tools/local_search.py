"""Deterministic, read-only content search for the user's Desktop/Documents.

This is intentionally separate from the Knowledge Base RAG tool.  ``/sync_kb``
indexes only documents placed in or below ``rag_document/``; a generic request
to find content in user files
must have a bounded route to the two ordinary document directories instead of
being told that the Knowledge Base is empty.
"""
from __future__ import annotations

import os
import re
import time
from collections import deque
from pathlib import Path

import config
from helpers import turn_guard
from helpers.document_extract import (
    is_supported_document,
    iter_document_units,
    iter_document_units_with_locations,
)
from helpers.sensitive_paths import contains_secret_content, is_sensitive_path
from helpers.thai_text_normalize import normalize_extracted_text
from helpers.thai_word_break import query_words


SCHEMA = {
    "type": "function",
    "function": {
        "name": "local_search",
        "description": (
            "Search file CONTENT in the user's Desktop and Documents only. "
            "Read-only, deterministic, no Knowledge Base sync and no web. "
            "Use this for a generic request such as finding files that mention a topic; "
            "then use read_file on a returned absolute path for a full answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short content keywords or phrase, not the whole user sentence",
                }
            },
            "required": ["query"],
        },
    },
}

_MAX_QUERY_CHARS = 512
_WORD_RE = re.compile(r"[\wก-๙]+", re.UNICODE)
_TEXT_SUFFIXES = frozenset({
    "", ".cfg", ".conf", ".csv", ".css", ".html", ".ini", ".js", ".json",
    ".jsonl", ".log", ".markdown", ".md", ".py", ".rtf", ".sh", ".sql",
    ".swift", ".text", ".toml", ".ts", ".tsx", ".tsv", ".txt", ".yaml", ".yml",
    ".zsh",
})
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".lite_state",
    "node_modules", "venv", ".venv", "build", "dist", "coverage",
})
_STOP_WORDS = frozenset({
    "หา", "ค้นหา", "ค้น", "ไฟล์", "เอกสาร", "เนื้อหา", "เกี่ยวกับ", "ที่", "มี",
    "คำว่า", "คำ", "ให้", "หน่อย", "อะไร", "บ้าง", "และ", "ของ", "ใน", "จาก",
    "the", "a", "an", "of", "for", "to", "and", "about",
})
_PDF_PAGE_RE = re.compile(r"^\s*---\s*PDF page\s+(\d+)\s*---", re.IGNORECASE)
_SLIDE_RE = re.compile(r"^\s*---\s*Slide\s+(\d+)\s*---", re.IGNORECASE)
_SHEET_RE = re.compile(r"^\s*---\s*Sheet:\s*(.+?)\s*---", re.IGNORECASE)
_TABLE_RE = re.compile(r"^\s*---\s*Table\s+(\d+)\s*---", re.IGNORECASE)


def _normalize(text: str) -> str:
    return normalize_extracted_text(re.sub(r"\s+", " ", text or "").strip().casefold())


def _query_parts(query: str) -> tuple[str, list[str]]:
    normalized = _normalize(query)
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _WORD_RE.findall(normalized):
        for term in query_words(raw):
            if term in _STOP_WORDS or len(term) < 2 or term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= 16:
                return normalized, terms
    return normalized, terms


def _matches(text: str, normalized_query: str, terms: list[str]) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    compact_text = re.sub(r"\s+", "", normalized)
    compact_query = re.sub(r"\s+", "", normalized_query)
    if normalized_query in normalized or (compact_query and compact_query in compact_text):
        return True
    # A document may insert punctuation or line breaks between words. Require
    # every meaningful term in the same bounded unit rather than matching one
    # common word from an unrelated file.
    return len(terms) >= 2 and all(term in normalized or term in compact_text for term in terms)


def _snippet(text: str, query: str) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= config.LOCAL_SEARCH_SNIPPET_CHARS:
        return clean
    needles = [query, *_query_parts(query)[1]]
    folded = clean.casefold()
    position = next(
        (folded.find(needle.casefold()) for needle in needles if needle and folded.find(needle.casefold()) >= 0),
        0,
    )
    half = max(80, config.LOCAL_SEARCH_SNIPPET_CHARS // 2)
    start = max(0, position - half)
    end = min(len(clean), start + config.LOCAL_SEARCH_SNIPPET_CHARS)
    prefix = "…" if start else ""
    suffix = "…" if end < len(clean) else ""
    return prefix + clean[start:end].strip() + suffix


def _roots() -> list[Path]:
    roots: list[Path] = []
    for raw_root in getattr(config, "LOCAL_SEARCH_ROOTS", ()):
        try:
            root = Path(raw_root).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if root.is_dir() and not is_sensitive_path(root) and root not in roots:
            roots.append(root)
    return roots


def _candidate_files(root: Path):
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in _SKIP_DIRS and not name.startswith(".")
        )
        for name in sorted(filenames, key=str.casefold):
            path = current / name
            if path.is_symlink() or is_sensitive_path(path):
                continue
            if path.suffix.casefold() not in _TEXT_SUFFIXES and not is_supported_document(path):
                continue
            yield path


def _scan_plain_file(
    path: Path, normalized_query: str, terms: list[str], deadline: float,
) -> tuple[str, str] | None:
    try:
        if path.stat().st_size > config.LOCAL_SEARCH_MAX_PLAIN_BYTES:
            return None
        with path.open("rb") as raw:
            if b"\0" in raw.read(4096):
                return None
        window_lines: deque[tuple[int, str]] = deque()
        window_chars = 0
        with path.open("r", encoding="utf-8", errors="replace", newline=None) as handle:
            for line_number, line in enumerate(handle, 1):
                if time.monotonic() >= deadline:
                    return None
                window_lines.append((line_number, line.rstrip("\n")))
                window_chars += len(line)
                while window_lines and window_chars > 8_000:
                    _old_number, old_line = window_lines.popleft()
                    window_chars -= len(old_line) + 1
                window = "\n".join(text for _number, text in window_lines)
                if _matches(window, normalized_query, terms):
                    first = window_lines[0][0]
                    last = window_lines[-1][0]
                    location = f"บรรทัด {first}" if first == last else f"บรรทัด {first}-{last}"
                    return location, _snippet(window, normalized_query)
    except (OSError, UnicodeError):
        return None
    return None


def _document_location(
    unit: str, unit_number: int, previous: str | None,
) -> tuple[str, str | None]:
    """Return a human-readable location and the current section for a unit.

    ``iter_document_units`` intentionally stays a string-only compatibility
    API.  Its stable PDF/slide/sheet/table headers give this search path
    enough deterministic metadata without changing every extractor caller.
    """
    first_line = (unit or "").splitlines()[0] if unit else ""
    match = _PDF_PAGE_RE.match(first_line)
    if match:
        page = match.group(1)
        return f"หน้า {page}", f"หน้า {page}"
    match = _SLIDE_RE.match(first_line)
    if match:
        slide = match.group(1)
        return f"สไลด์ {slide}", f"สไลด์ {slide}"
    match = _SHEET_RE.match(first_line)
    if match:
        sheet = match.group(1).strip()
        return f"ชีต {sheet}", f"ชีต {sheet}"
    match = _TABLE_RE.match(first_line)
    if match:
        table = match.group(1)
        return f"ตาราง {table}", previous or f"ตาราง {table}"
    if previous:
        return f"{previous} · หน่วย {unit_number}", previous
    return f"หน่วย {unit_number}", None


def _scan_document(
    path: Path, normalized_query: str, terms: list[str], deadline: float,
) -> tuple[str, str] | None:
    current_location: str | None = None
    try:
        if path.suffix.casefold() in {".docx", ".docm"}:
            for location, unit in iter_document_units_with_locations(path):
                if time.monotonic() >= deadline:
                    return None
                if _matches(unit, normalized_query, terms):
                    return location, _snippet(unit, normalized_query)
            return None
        for unit_number, unit in enumerate(iter_document_units(path), 1):
            if time.monotonic() >= deadline:
                return None
            location, current_location = _document_location(unit, unit_number, current_location)
            if _matches(unit, normalized_query, terms):
                sample = unit
                first_line = (unit or "").splitlines()[0] if unit else ""
                if (
                    _PDF_PAGE_RE.match(first_line)
                    or _SLIDE_RE.match(first_line)
                    or _SHEET_RE.match(first_line)
                    or _TABLE_RE.match(first_line)
                ):
                    sample = "\n".join((unit or "").splitlines()[1:]).strip()
                return location, _snippet(sample, normalized_query)
    except (OSError, RuntimeError, ValueError, UnicodeError):
        return None
    return None


def run(args: dict[str, str]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "[error] query is required"
    if len(query) > _MAX_QUERY_CHARS:
        return f"[error] query ยาวเกิน {_MAX_QUERY_CHARS} ตัวอักษร"

    normalized_query, terms = _query_parts(query)
    if not normalized_query or not terms:
        return "[error] query ไม่มี keyword ที่ค้นได้"

    roots = _roots()
    if not roots:
        return "[error] ไม่พบโฟลเดอร์ Desktop หรือ Documents ของ account ที่กำลังรันอยู่"

    deadline = time.monotonic() + config.LOCAL_SEARCH_TIMEOUT_SECONDS
    matches: list[tuple[Path, str, str, str]] = []
    scanned = 0
    timed_out = False
    for root in roots:
        for path in _candidate_files(root):
            if scanned >= config.LOCAL_SEARCH_MAX_FILES or time.monotonic() >= deadline:
                timed_out = True
                break
            scanned += 1
            scanner = _scan_document if is_supported_document(path) else _scan_plain_file
            found = scanner(path, normalized_query, terms, deadline)
            if found:
                location, snippet = found
                if not contains_secret_content(snippet):
                    matches.append((path, root.name, location, snippet))
                if len(matches) >= config.LOCAL_SEARCH_MAX_RESULTS:
                    break
        if timed_out or len(matches) >= config.LOCAL_SEARCH_MAX_RESULTS:
            break

    if not matches:
        suffix = " (หยุดที่เวลา/ขีดจำกัดการสแกน)" if timed_out else ""
        return (
            f"ไม่พบเนื้อหาที่ตรงกับ '{query}' ใน Desktop หรือ Documents{suffix} "
            f"(ตรวจ {scanned:,} ไฟล์แบบ read-only; ไม่ได้ค้นทั้งเครื่อง)"
        )

    lines = [f"พบ {len(matches)} ไฟล์ใน Desktop/Documents ที่มีเนื้อหาตรงกับ '{query}':"]
    for path, root_name, location, snippet in matches:
        lines.append(
            f"- ไฟล์: {path}\n"
            f"  พื้นที่: {root_name}\n"
            f"  หน้า/ตำแหน่ง: {location}\n"
            f"  ตัวอย่าง: {snippet}"
        )
    if timed_out:
        lines.append(f"\n[หมายเหตุ] หยุดเมื่อถึงขีดจำกัด; ตรวจแล้ว {scanned:,} ไฟล์")
    return "\n".join(lines)


_REMINDER = (
    "\n\n[reminder] local_search ค้นแบบ read-only เฉพาะ Desktop และ Documents "
    "ตาม path ที่ runtime ตรวจจริงเท่านั้น; ถ้าพบ path ให้ใช้ read_file ต่อ "
    "และห้ามเปลี่ยนไปค้นทั้ง Home/ทั้งเครื่องหรือใช้ /sync_kb แทน"
)

from langchain_core.tools import tool as _lc_tool


@_lc_tool
def local_search(query: str) -> str:
    """Search content in the user's Desktop and Documents only.

    Use for generic local-file topic searches. This does not use the Knowledge
    Base, does not write an index, and does not access the web.
    """
    args = {"query": query}
    blocked = turn_guard.check("local_search", args)
    if blocked:
        return blocked
    result = run(args)
    turn_guard.record("local_search", args, result)
    if turn_guard.call_count("local_search") == 1:
        result += _REMINDER
    return result
