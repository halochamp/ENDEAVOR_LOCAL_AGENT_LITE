# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""tools/rag_search.py — SQLite FTS5 file search (replaces MAX_VLM's rag_tool.py).

MAX_VLM's rag_tool.py queries a MiniLM + ChromaDB semantic index (see
ENDEAVOR_RAG_MAX) — that's ~0.5-1GB of RAM for the embedding model alone,
which is most of this agent's entire RAM budget. This is a lexical
(BM25) search over indexer.py's FTS5 table instead: no embedding model, the
index itself lives on SSD and is mmap-read, and a single MATCH query hits
both filenames and file content in one pass. Trade-off accepted in PLAN.md
§8: no semantic matching — a query with no literal term overlap with the
target file won't find it.
"""
from __future__ import annotations
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

import config
from helpers.sensitive_paths import is_sensitive_path
from helpers.thai_text_normalize import repair_thai_combining_mark_spacing

# 2026-07-24 (user request: query robustness -- synonyms, case, spacing,
# stray characters -- pattern ported from ENDMEMEX/endeavor_db.py's
# query_terms()/fts_expression()/_QUERY_ALIASES, the same idea already
# proven for that project's own FTS5 search). Verified empirically before
# building this (not assumed): SQLite's trigram tokenizer is ALREADY
# case-insensitive by default (tested: 'REPORT'/'Report'/'report' all match
# identically), and NFC vs NFD normalization made no difference for the Thai
# test string tried -- neither needed a fix at the index level. What WAS
# genuinely missing: (1) synonym/alias expansion -- a query for "รูป" never
# found a file whose content only says "ภาพ" or "photo"; (2) stray-character
# robustness -- `q.split()` passes any punctuation/symbol run through as a
# literal FTS token, quietly contributing nothing (trigram tokenizes it) or
# even erroring; a regex-based word extractor drops that noise instead of
# feeding it to MATCH.
_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "รูป": ("ภาพ", "image", "photo", "picture"),
    "ภาพ": ("รูป", "image", "photo", "picture"),
    "เอกสาร": ("document", "ไฟล์", "doc"),
    "วิดีโอ": ("คลิป", "video", "clip"),
    "คลิป": ("วิดีโอ", "video", "clip"),
    "ใบเสร็จ": ("บิล", "receipt", "invoice"),
    "บิล": ("ใบเสร็จ", "receipt", "invoice"),
    "รายงาน": ("report",),
    "สัญญา": ("contract", "agreement"),
    "จดหมาย": ("letter", "mail"),
    "การเงิน": ("เงิน", "finance", "financial"),
    "ประวัติ": ("resume", "cv"),
}
_STOP_WORDS = {
    "หา", "ให้", "หน่อย", "ที่", "ของ", "และ", "อะไร", "บ้าง", "เรื่อง", "ไหน", "ไฟล์",
    "the", "a", "an", "of", "for", "to", "and",
}
_WORD_RE = re.compile(r"[\wก-๙]+", re.UNICODE)
_ENGLISH_SPLIT_MIN_CHARS = 5
_MAX_ENGLISH_SPLIT_VARIANTS = 24
# Keep query normalization aligned with indexer._normalize_extracted_text:
# PDF layout extraction can insert a space before a Thai combining mark, and
# users may paste that same visibly-floating form back into search.  Remove
# only this structurally impossible word break; ordinary Thai word spacing is
# preserved.
def _query_terms(query: str) -> list[str]:
    """Normalize (NFC, casefold, whitespace) then extract word-like runs,
    dropping stray punctuation/symbols and generic filler words. Casefold
    (not .lower()) for correct full-Unicode case folding, though the
    trigram tokenizer already matches case-insensitively on its own --
    this keeps alias lookups and term dedup consistent regardless."""
    normalized = unicodedata.normalize("NFC", query)
    normalized = re.sub(r"\s+", " ", normalized.strip().casefold())
    normalized = repair_thai_combining_mark_spacing(normalized)
    terms = _WORD_RE.findall(normalized)
    seen: set[str] = set()
    result = []
    for term in terms:
        if term in _STOP_WORDS or term in seen:
            continue
        seen.add(term)
        result.append(term)
    return result[:12]

SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_search",
        # See tools/bash.py's matching 2026-07-23 note — reverted alongside it.
        "description": (
            "Search locally indexed files by filename or content keywords. Returns "
            "matching file paths with a short snippet. Use this FIRST when the user "
            "asks to find a file, before trying bash/find."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for (filename or content)"}
            },
            "required": ["query"],
        },
    },
}


def _fts_atom(term: str) -> str:
    return f'"{term.replace(chr(34), chr(34) * 2)}"'


def _safe_match_query(q: str) -> str:
    """Build an FTS5 MATCH expression: each extracted query term OR'd with
    its own synonyms (if any), the per-term groups themselves OR'd together
    -- matches ANY of the user's words or their aliases, same "locator, not
    exact search" intent as before, just synonym-aware now."""
    terms = _query_terms(q)
    expressions = []
    for term in terms:
        alternatives = [_fts_atom(term), *(_fts_atom(alias) for alias in _QUERY_ALIASES.get(term, ()))]
        expressions.append(alternatives[0] if len(alternatives) == 1 else f"({' OR '.join(alternatives)})")
    return " OR ".join(expressions)


def _english_split_match_query(q: str) -> str:
    """Return bounded FTS phrases for English words split by PDF/OCR.

    The index retains ordinary English spaces, so rewriting stored text would
    corrupt valid phrases such as ``project plan``.  Instead query variants
    let ``information`` also match a stored ``infor mation``.  Only ASCII
    alphabetic words of a useful length participate and the whole query has a
    strict cap, keeping this recall path bounded.
    """
    variants: list[str] = []
    for term in _query_terms(q):
        if not (term.isascii() and term.isalpha() and len(term) >= _ENGLISH_SPLIT_MIN_CHARS):
            continue
        for position in range(1, len(term)):
            variants.append(_fts_atom(f"{term[:position]} {term[position:]}"))
            if len(variants) >= _MAX_ENGLISH_SPLIT_VARIANTS:
                return " OR ".join(variants)
    return " OR ".join(variants)


def _fallback_filename_search(query: str, *, reason: str = "no_index") -> str:
    """Fast bounded filename lookup, used two ways: (1) no index exists yet
    at all, (2) the index exists but the FTS query above matched nothing --
    a broader, code-driven retry rather than making the model notice a zero-
    result reply and manually re-issue a looser query itself (small models
    don't reliably do this, per this project's own live findings tonight)."""
    needle = query.casefold()
    home = Path.home()
    # User-facing folders are searched first, so a cold search is useful on a
    # real Mac instead of timing out while walking caches/venvs under $HOME.
    roots = [home / name for name in ("Desktop", "Documents", "Downloads")]
    roots.append(config.PROJECT_ROOT)
    deadline = time.monotonic() + 4.0
    matches: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            if time.monotonic() >= deadline or len(matches) >= 50:
                break
            current = Path(dirpath)
            dirnames[:] = [
                name for name in dirnames
                if not is_sensitive_path(current / name) and name not in {".git", "node_modules", "__pycache__"}
            ]
            for name in filenames:
                if time.monotonic() >= deadline:
                    break
                if needle in name.casefold():
                    candidate = current / name
                    if not is_sensitive_path(candidate):
                        matches.append(str(candidate))
                        if len(matches) >= 50:
                            break
    if not matches:
        if reason == "no_index":
            return (
                "[error] ยังไม่มี local index และค้นชื่อไฟล์สำรองไม่พบผล — "
                "ใช้ /index เพื่อสร้าง index หรือระบุชื่อไฟล์ให้เจาะจงขึ้น"
            )
        return f"ไม่พบไฟล์ที่ตรงกับ '{query}' (ลองค้นหาแบบกว้างขึ้นในชื่อไฟล์ทั้งเครื่องแล้วก็ไม่พบ)"
    suffix = "\n[หยุดหลัง 4 วินาที/50 ผล]" if time.monotonic() >= deadline or len(matches) >= 50 else ""
    header = "[local filename fallback — ยังไม่มี RAG index]" if reason == "no_index" \
        else "[ขยายผลการค้นหา — ไม่พบใน index แต่พบไฟล์ชื่อใกล้เคียง]"
    return f"{header}\n" + "\n".join(matches) + suffix


def run(args: dict[str, str]) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "[error] query is required"
    if not config.FILE_INDEX_DB.exists():
        return _fallback_filename_search(query)

    match_query = _safe_match_query(query)
    if not match_query:
        return "[error] query ว่างเปล่าหลัง sanitize"
    split_match_query = _english_split_match_query(query)

    conn = sqlite3.connect(str(config.FILE_INDEX_DB))
    try:
        # Bug fixed 2026-07-23 (found during final architecture-switch smoke
        # testing): a single combined MATCH over (name, content), even with
        # bm25() column weighting (tried bm25(files_fts, 1.0, 10.0, 1.0) and
        # bm25(files_fts, 0.0, 1.0, 0.0) -- content-only-zeroed), silently
        # ranked the exact file "PLAN.md" itself BELOW files that merely cite
        # "[PLAN.md]", and even below unrelated files like "ENCRYPTION_PLAN.md"
        # for a name-only query. BM25 on a trigram-tokenized short "name"
        # field does not behave like word-level BM25 intuition suggests --
        # not investigated further given diminishing returns. Fix: search
        # `name MATCH` for candidates (recall via FTS5), then rank them in
        # plain Python with a simple, predictable rule (exact match > query
        # is a substring of the name > shorter name = more specific) instead
        # of trusting bm25()'s score on this field at all.
        name_candidates = conn.execute(
            "SELECT f.path, f.name, snippet(f.files_fts, 2, '[', ']', '...', 12), m.size_bytes, m.kind "
            "FROM files_fts f LEFT JOIN meta m ON m.path = f.path WHERE f.name MATCH ? LIMIT 50",
            (match_query,),
        ).fetchall()
        split_name_candidates = conn.execute(
            "SELECT f.path, f.name, snippet(f.files_fts, 2, '[', ']', '...', 12), m.size_bytes, m.kind "
            "FROM files_fts f LEFT JOIN meta m ON m.path = f.path WHERE f.name MATCH ? LIMIT 50",
            (split_match_query,),
        ).fetchall() if split_match_query else []

        def _name_rank(row):
            name = row[1]
            exact = name.lower() == query.lower()
            substring = query.lower() in name.lower()
            return (not exact, not substring, len(name))

        name_rows = sorted(name_candidates, key=_name_rank)
        direct_name_paths = {row[0] for row in name_rows}
        name_rows.extend(row for row in sorted(split_name_candidates, key=_name_rank) if row[0] not in direct_name_paths)
        name_rows = name_rows[: config.RAG_MAX_RESULTS]
        seen_paths = {r[0] for r in name_rows}

        remaining = config.RAG_MAX_RESULTS - len(name_rows)
        general_rows = []
        if remaining > 0:
            general_rows = conn.execute(
            "SELECT f.path, f.name, snippet(f.files_fts, 2, '[', ']', '...', 12), m.size_bytes, m.kind "
            "FROM files_fts f LEFT JOIN meta m ON m.path = f.path WHERE f.files_fts MATCH ? ORDER BY bm25(f.files_fts) LIMIT ?",
                (match_query, config.RAG_MAX_RESULTS + len(name_rows)),
            ).fetchall()
            general_rows = [r for r in general_rows if r[0] not in seen_paths][:remaining]
            if split_match_query and len(general_rows) < remaining:
                split_general_rows = conn.execute(
                    "SELECT f.path, f.name, snippet(f.files_fts, 2, '[', ']', '...', 12), m.size_bytes, m.kind "
                    "FROM files_fts f LEFT JOIN meta m ON m.path = f.path WHERE f.files_fts MATCH ? ORDER BY bm25(f.files_fts) LIMIT ?",
                    (split_match_query, config.RAG_MAX_RESULTS + len(name_rows)),
                ).fetchall()
                direct_general_paths = {row[0] for row in general_rows}
                general_rows.extend(
                    row for row in split_general_rows
                    if row[0] not in seen_paths and row[0] not in direct_general_paths
                )
                general_rows = general_rows[:remaining]

        rows = [row for row in name_rows + general_rows if not is_sensitive_path(row[0])]
    except sqlite3.OperationalError as e:
        return f"[error] ค้นหาล้มเหลว: {e}"
    finally:
        conn.close()

    if not rows:
        return _fallback_filename_search(query, reason="fts_zero_hits")

    lines = [f"พบ {len(rows)} ไฟล์ที่ตรงกับ '{query}':"]
    for path, name, snippet, size_bytes, kind in rows:
        snippet = (snippet or "").replace("\n", " ").strip()
        if len(snippet) > config.RAG_SNIPPET_CHARS:
            snippet = snippet[: config.RAG_SNIPPET_CHARS] + "..."
        metadata = ""
        if kind and kind != "text excerpt":
            metadata = f" [{kind}, {(size_bytes or 0):,} bytes; content not indexed]"
        lines.append(f"- {path}\n  ({name}) {snippet}{metadata}")
    return "\n".join(lines)


# LangChain entry point — see tools/bash.py's 2026-07-23 note.
# PLAN.md §9.5 docstring-split pattern (2026-07-25) applied here: this
# reminder only matters AFTER a result comes back (it governs how to react to
# a fallback/empty result, not whether to call rag_search in the first
# place), so it's deferred into this tool's own return value on the first
# call each turn. See tools/bash.py for the original design/verification
# writeup and helpers.turn_guard.call_count().
_FALLBACK_REMINDER = (
    "\n\n[reminder] A local-file request never switches to web_search. "
    "If this result contains the requested exact filename, use that path. "
    "If no relevant content hit exists, continue once with bash: grep on the "
    "project's absolute path, or mdfind for whole-machine content. For list/count, "
    "RAG snippets cannot prove completeness: continue with bash using an absolute path. "
    "Preserve the requested document kind in the find command; never count every file: "
    "PDF => find /absolute/Documents -type f -iname '*.pdf' | wc -l; "
    "Word => '*.docx'; Excel => '*.xlsx'; PowerPoint => '*.pptx'."
)

from langchain_core.tools import tool as _lc_tool


@_lc_tool
def rag_search(query: str) -> str:
    """Search locally indexed files by filename OR content keywords (find a
    file named X; find which file mentions Y). Returns matching file paths
    with a short snippet.

    Use this FIRST for every on-device file discovery/search request. The only
    exception is when the user explicitly says to use/run bash. For an exact
    file result, call read_file on the returned path instead of bash.
    ❌ "Where is rental_contract.docx, and summarize it" -> do not call bash directly
    ✅ rag_search(query="rental_contract.docx") first, then read_file its path

    For counting/listing, call this first to establish scope, then continue to
    bash because snippets cannot prove a complete count/list."""
    from helpers import turn_guard
    args = {"query": query}
    blocked = turn_guard.check("rag_search", args)
    if blocked:
        return blocked
    first_call_this_turn = turn_guard.call_count("rag_search") == 1
    result = run(args)
    turn_guard.record("rag_search", args, result)
    if first_call_this_turn:
        result += _FALLBACK_REMINDER
    return result
