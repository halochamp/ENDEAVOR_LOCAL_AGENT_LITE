# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Lean read_file for text plus PDF and Office documents.

Extraction is read-only and its output is capped at 5,000 characters. Larger
sources are sampled across their contents rather than exposing full bodies to
Lite's small context window.

PLAN.md §9.1: the hot path (no explicit line range) never materializes a
whole large document/text file just to keep a 5,000-char sample. A two-pass
streaming approach is used instead: pass 1 scans the entire source for
secret-shaped content and counts totals; pass 2 re-reads only the handful of
already-selected units. The deterministic sample indices/format are
byte-identical to the original in-memory algorithm (kept intact below as
`_sample_for_coverage`/`_sample_pdf_pages`, still used for the less-common
line-range request and for sources already small enough to materialize
directly).
"""
from __future__ import annotations
from collections.abc import Callable, Iterator
import hashlib
import heapq
import random
import re
from pathlib import Path

import config
from helpers.sensitive_paths import contains_secret_content, is_sensitive_path, StreamingSecretScanner
from helpers.document_extract import (
    extract_text,
    is_supported_document,
    iter_document_units,
    pdf_page_count,
    pdf_page_text,
    unit_separator,
)
from helpers.streaming_sample import build_sample_output, select_sample_indices

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        # See tools/bash.py's 2026-07-23 note — reverted alongside it (B was
        # applied to all 6 tools at once; only bash/rag_search's regression
        # was isolated, but reverting all of them together is the safer call
        # given B's total RAM benefit was modest next to approach A/D).
        "description": (
            "Read a text/markdown/code/PDF/Word (.doc/.docx)/Excel/PowerPoint file from disk, given its full path. "
            "Returns at most 5,000 characters; larger content is sampled across the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~-relative path to the file"},
                "start_line": {"type": "string", "description": "Optional: 1-indexed first line to read"},
                "end_line": {"type": "string", "description": "Optional: 1-indexed last line to read"},
            },
            "required": ["path"],
        },
    },
}

_BINARY_MARKER = b"\x00"
_BLOCKED_SECRET = "[blocked] เนื้อหามีข้อมูลที่มีรูปแบบคล้าย credential, token หรือ private key"


def _looks_binary(sample: bytes) -> bool:
    return _BINARY_MARKER in sample


def _sample_for_coverage(text: str, path: Path) -> str:
    """Keep a deterministic spread of lines within Lite's fixed output budget.

    Unchanged, in-memory algorithm -- used only when `text` is already small
    (bounded OCR output, a single-page PDF, or a line-range slice), never on
    a whole large document body.
    """
    limit = config.READ_FILE_MAX_CHARS
    if len(text) <= limit:
        return text
    if path.suffix.lower() == ".pdf":
        sampled_pages = _sample_pdf_pages(text, path, limit)
        if sampled_pages is not None:
            return sampled_pages
    units = [line.strip()[:700] for line in text.splitlines() if line.strip()]
    if not units:
        return text[:limit]
    header = f"[{path} — {len(text):,} chars total; sampled across file, max {limit:,} chars]\n\n"
    budget = max(200, limit - len(header))
    count = min(18, len(units))
    rng = random.Random(int(hashlib.sha256(f"{path}:{len(text)}".encode()).hexdigest()[:16], 16))
    selected: set[int] = {0, len(units) - 1}
    for bucket in range(1, max(1, count - 1)):
        start = bucket * len(units) // count
        end = max(start, (bucket + 1) * len(units) // count - 1)
        selected.add(rng.randint(start, end))
    body: list[str] = []
    used = 0
    for index in sorted(selected):
        entry = f"[sample {index + 1}/{len(units)}] {units[index]}"
        if used + len(entry) + 1 > budget:
            continue
        body.append(entry)
        used += len(entry) + 1
    return header + "\n".join(body)


_PDF_PAGE_MARKER_RE = re.compile(r"(?=--- PDF page \d+(?: \(OCR\))? ---)")


def _sample_pdf_pages(text: str, path: Path, limit: int) -> str | None:
    """Sample complete PDF page blocks across the document, never raw lines.

    A line-based sampler can retain page 1's marker but discard every other
    marker, making a broad OCR pass look like a one-page read.  Keep each
    selected page's identity and take evenly distributed pages instead.
    """
    page_blocks = [block.strip() for block in _PDF_PAGE_MARKER_RE.split(text) if block.strip().startswith("--- PDF page")]
    if len(page_blocks) < 2:
        return None

    header_lines = [line for line in text.splitlines() if line.startswith("[") and "OCR" in line]
    header = header_lines[0] if header_lines else f"[{path} — PDF sampled by page]"
    budget = max(600, limit - len(header) - 120)
    count = min(8, len(page_blocks))
    indexes = {0, len(page_blocks) - 1}
    for slot in range(1, max(1, count - 1)):
        indexes.add(round(slot * (len(page_blocks) - 1) / max(1, count - 1)))

    selected: list[str] = []
    used = 0
    # Reserve the separators too, so the last (coverage) page cannot fall
    # just outside the budget due to rounding.
    per_page = max(240, (budget - 2 * len(indexes)) // len(indexes))
    for index in sorted(indexes):
        block = page_blocks[index]
        suffix = "\n… [หน้านี้ตัดบางส่วน]"
        clipped = block if len(block) <= per_page else block[:per_page - len(suffix)].rstrip() + suffix
        if used + len(clipped) + 2 > budget:
            continue
        selected.append(clipped)
        used += len(clipped) + 2
    if not selected:
        return None
    return header + f"\n[แสดงตัวอย่าง {len(selected)}/{len(page_blocks)} หน้า OCR แบบกระจายทั่วเอกสาร]\n\n" + "\n\n".join(selected)


# ---------------------------------------------------------------------------
# Streaming (bounded-RAM) plain-text path -- PLAN.md §9.1
# ---------------------------------------------------------------------------

def _scan_plain_text(path: Path) -> tuple[int, int, bool]:
    """Pass 1: total char length (matches path.read_text()'s len(text)),
    count of non-empty stripped lines, and whether any chunk anywhere in
    the file trips the secret scanner -- all without retaining content.

    Known limitation: str.splitlines() also breaks on a handful of rare
    unicode line separators (\\v, \\f, U+2028/2029) that plain text-mode
    line iteration does not split on. Ordinary code/text files never
    contain these; a file that did would only affect the deterministic
    sample's exact slice, not the secret scan or output size cap.
    """
    scanner = StreamingSecretScanner()
    total_chars = 0
    unit_count = 0
    with open(path, "r", encoding="utf-8", errors="replace", newline=None) as fh:
        for line in fh:
            total_chars += len(line)
            scanner.feed(line)
            if line.strip():
                unit_count += 1
    return total_chars, unit_count, scanner.found


def _sample_plain_text(path: Path, total_chars: int, unit_count: int) -> str:
    limit = config.READ_FILE_MAX_CHARS
    if unit_count == 0:
        with open(path, "r", encoding="utf-8", errors="replace", newline=None) as fh:
            return fh.read(limit)
    selected = select_sample_indices(unit_count, f"{path}:{total_chars}")
    picked: dict[int, str] = {}
    max_needed = max(selected)
    index = 0
    with open(path, "r", encoding="utf-8", errors="replace", newline=None) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                if index in selected:
                    picked[index] = stripped[:700]
                index += 1
                if index > max_needed and len(picked) == len(selected):
                    break
    return build_sample_output(path, total_chars, unit_count, limit, picked)


def _read_streamed_line_range(
    path: Path,
    start_line: str,
    end_line: str,
    line_factory: Callable[[], Iterator[str]],
) -> str:
    """Read a line range in two bounded-memory passes.

    The previous plain-text path accumulated every selected line, while the
    Office/PDF path materialized the complete extracted document first.  A
    missing end_line therefore defeated the 5,000-character output cap's RAM
    purpose.  Pass one now counts and scans secrets; pass two keeps either the
    small requested slice or only deterministic samples from a large one.
    """
    try:
        s = int(start_line) if start_line else 1
        e = int(end_line) if end_line else None
    except ValueError:
        return "[error] start_line/end_line ต้องเป็นตัวเลข"
    s = max(1, s)
    scanner = StreamingSecretScanner()
    total_lines = 0
    selected_lines = 0
    selected_chars = 0
    selected_nonempty = 0
    for line_no, line in enumerate(line_factory(), 1):
        total_lines = line_no
        scanner.feed(line + "\n")
        if line_no >= s and (e is None or line_no <= e):
            if selected_lines:
                selected_chars += 1  # the join newline
            selected_lines += 1
            selected_chars += len(line)
            if line.strip():
                selected_nonempty += 1
    if scanner.found:
        return _BLOCKED_SECRET
    e_final = min(total_lines, e) if e is not None else total_lines
    header = f"[บรรทัด {s}-{e_final} จาก {total_lines}]\n"
    if len(header) + selected_chars <= config.READ_FILE_MAX_CHARS:
        picked = [
            line for line_no, line in enumerate(line_factory(), 1)
            if line_no >= s and (e is None or line_no <= e)
        ]
        return header + "\n".join(picked)
    if selected_nonempty == 0:
        return header + "[ช่วงที่ขอมีเฉพาะบรรทัดว่างและยาวเกินขีดจำกัด]"

    selected = select_sample_indices(
        selected_nonempty,
        f"{path}:range:{s}:{e_final}:{selected_chars}",
    )
    picked: dict[int, tuple[int, str]] = {}
    unit_index = 0
    for line_no, line in enumerate(line_factory(), 1):
        if line_no < s or (e is not None and line_no > e):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if unit_index in selected:
            picked[unit_index] = (line_no, stripped[:700])
        unit_index += 1
    note = f"[ช่วงที่ขอมี {selected_chars:,} chars; sampled, max {config.READ_FILE_MAX_CHARS:,} chars]\n"
    budget = config.READ_FILE_MAX_CHARS - len(header) - len(note)
    rendered: list[str] = []
    used = 0
    for index in sorted(picked):
        line_no, text = picked[index]
        entry = f"[sample line {line_no}] {text}"
        if used + len(entry) + 1 > budget:
            continue
        rendered.append(entry)
        used += len(entry) + 1
    return header + note + "\n".join(rendered)


def _plain_text_lines(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace", newline=None) as fh:
        for raw_line in fh:
            yield raw_line[:-1] if raw_line.endswith("\n") else raw_line


def _document_lines(path: Path) -> Iterator[str]:
    separator_blank_lines = max(0, unit_separator(path.suffix).count("\n") - 1)
    first = True
    for unit in iter_document_units(path):
        if not first:
            for _ in range(separator_blank_lines):
                yield ""
        first = False
        yield from unit.splitlines()


# ---------------------------------------------------------------------------
# Streaming (bounded-RAM) document path -- PLAN.md §9.1
# ---------------------------------------------------------------------------

def _scan_office_document(path: Path, sep: str) -> tuple[int, int, bool]:
    """Pass 1 for DOCX/XLSX/PPTX: total joined length (matches
    sep.join(units)'s len()), non-empty unit count, and secret-scan result
    -- without retaining any unit's text."""
    scanner = StreamingSecretScanner()
    total_chars = 0
    unit_count = 0
    first = True
    for unit in iter_document_units(path):
        if not first:
            # Feed the separator too, not just each raw unit: concatenating
            # unit A's tail directly against unit B's head can turn e.g.
            # "...filler y" + "api_key: xxx" into "...yapi_key: xxx", which
            # silently destroys the \b word-boundary the secret patterns
            # rely on (a real false negative, found via test).
            scanner.feed(sep)
            total_chars += len(sep)
        first = False
        total_chars += len(unit)
        scanner.feed(unit)
        if unit.strip():
            unit_count += 1
    return total_chars, unit_count, scanner.found


def _sample_office_document(path: Path, total_chars: int, unit_count: int, sep: str) -> str:
    limit = config.READ_FILE_MAX_CHARS
    selected = select_sample_indices(unit_count, f"{path}:{total_chars}")
    picked: dict[int, str] = {}
    max_needed = max(selected)
    index = 0
    for unit in iter_document_units(path):
        stripped = unit.strip()
        if stripped:
            if index in selected:
                picked[index] = stripped[:700]
            index += 1
            if index > max_needed and len(picked) == len(selected):
                break
    return build_sample_output(path, total_chars, unit_count, limit, picked)


def _sample_native_pdf(path: Path, page_count: int) -> str:
    """Direct-page-access equivalent of _sample_pdf_pages: re-fetches only
    the handful of selected pages via pdf_page_text(), never all of them."""
    limit = config.READ_FILE_MAX_CHARS
    header = f"[{path} — PDF sampled by page]"
    budget = max(600, limit - len(header) - 120)
    count = min(8, page_count)
    indexes = {0, page_count - 1}
    for slot in range(1, max(1, count - 1)):
        indexes.add(round(slot * (page_count - 1) / max(1, count - 1)))
    per_page = max(240, (budget - 2 * len(indexes)) // len(indexes))
    selected: list[str] = []
    used = 0
    for index in sorted(indexes):
        block = f"--- PDF page {index + 1} ---\n{pdf_page_text(path, index)}"
        suffix = "\n… [หน้านี้ตัดบางส่วน]"
        clipped = block if len(block) <= per_page else block[:per_page - len(suffix)].rstrip() + suffix
        if used + len(clipped) + 2 > budget:
            continue
        selected.append(clipped)
        used += len(clipped) + 2
    if not selected:
        text = f"--- PDF page 1 ---\n{pdf_page_text(path, 0)}"
        return _sample_for_coverage(text, path)
    return header + f"\n[แสดงตัวอย่าง {len(selected)}/{page_count} หน้า OCR แบบกระจายทั่วเอกสาร]\n\n" + "\n\n".join(selected)


def _read_pdf(path: Path) -> str:
    limit = config.READ_FILE_MAX_CHARS
    import fitz
    with fitz.open(path) as pdf:
        has_content = any(page.get_text("text").strip() for page in pdf)
    if not has_content:
        # Scanned PDF: OCR fallback is already bounded to <=30 pages, so
        # materializing it fully stays small regardless of total page count.
        text = extract_text(path)
        if text.startswith("[error]"):
            return text
        if not text.strip():
            return "[error] PDF นี้ไม่มีข้อความที่สกัดได้หรือ OCR อ่านไม่ได้"
        if contains_secret_content(text):
            return _BLOCKED_SECRET
        return _sample_for_coverage(text, path)

    page_count = pdf_page_count(path)
    sep = "\n\n"
    scanner = StreamingSecretScanner()
    total_chars = 0
    last_unit = ""
    for index in range(page_count):
        unit = f"--- PDF page {index + 1} ---\n{pdf_page_text(path, index)}"
        if index > 0:
            # See _scan_office_document's comment: feed the separator too so
            # a secret can't slip past the \b word-boundary check by having
            # its start concatenated directly onto the previous page's tail.
            scanner.feed(sep)
            total_chars += len(sep)
        total_chars += len(unit)
        scanner.feed(unit)
        last_unit = unit
    if scanner.found:
        return _BLOCKED_SECRET
    # extract_text() strips the full joined text; every unit already starts
    # with the non-whitespace "---" header, so only trailing whitespace on
    # the very last unit can ever be trimmed by that final .strip().
    total_chars -= len(last_unit) - len(last_unit.rstrip())

    if total_chars <= limit:
        return sep.join(
            f"--- PDF page {i + 1} ---\n{pdf_page_text(path, i)}" for i in range(page_count)
        ).strip()
    if page_count < 2:
        # _sample_pdf_pages requires >=2 page blocks -- fall back exactly
        # like the original generic sampler did for a single-page PDF.
        text = f"--- PDF page 1 ---\n{pdf_page_text(path, 0)}"
        return _sample_for_coverage(text, path)
    return _sample_native_pdf(path, page_count)


def _read_document(path: Path) -> str:
    """Bounded-RAM path for PDF/DOCX/XLSX/PPTX/legacy doc/ppt: only ever
    materializes a small prefix or a handful of scattered samples, never
    the whole body, while still scanning the ENTIRE document for
    secret-shaped content before returning anything (PLAN.md §9.1.4)."""
    limit = config.READ_FILE_MAX_CHARS
    suffix = path.suffix.lower()

    if suffix in {".doc", ".ppt"}:
        # Legacy formats: no streaming extractor exists (textutil already
        # returns a single string; .ppt is unsupported) -- same as before.
        text = extract_text(path)
        if text.startswith("[error]"):
            return text
        if contains_secret_content(text):
            return _BLOCKED_SECRET
        return _sample_for_coverage(text, path)

    if suffix == ".pdf":
        return _read_pdf(path)

    sep = unit_separator(suffix)
    total_chars, unit_count, secret_found = _scan_office_document(path, sep)
    if secret_found:
        return _BLOCKED_SECRET
    if total_chars <= limit:
        return sep.join(iter_document_units(path))
    return _sample_office_document(path, total_chars, unit_count, sep)


def run(args: dict[str, str]) -> str:
    raw_path = (args.get("path") or "").strip()
    if not raw_path:
        return "[error] path is required"

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return f"[error] ต้องเป็น absolute path (หรือขึ้นต้นด้วย ~) ได้ path: '{raw_path}'"
    if is_sensitive_path(path):
        return "[blocked] ไม่อนุญาตให้อ่านไฟล์หรือโฟลเดอร์ที่อาจมีรหัสผ่าน, token, key หรือ session"
    if not path.exists():
        parent = path.parent
        listing = ""
        if parent.exists():
            names = heapq.nsmallest(20, (p.name for p in parent.iterdir()))
            listing = f" ไฟล์ที่มีจริงใน {parent}: {', '.join(names)}" if names else f" {parent} ว่างเปล่า"
        return f"[error] ไม่พบไฟล์: {path}{listing}"
    if path.is_dir():
        names = heapq.nsmallest(30, (p.name for p in path.iterdir()))
        return f"[error] '{path}' เป็นโฟลเดอร์ ไม่ใช่ไฟล์ รายการข้างใน: {', '.join(names)}"

    start_line = args.get("start_line")
    end_line = args.get("end_line")

    try:
        if is_supported_document(path):
            if start_line or end_line:
                return _read_streamed_line_range(
                    path, start_line or "", end_line or "", lambda: _document_lines(path)
                )
            return _read_document(path)

        with open(path, "rb") as f:
            sample = f.read(4096)
        if _looks_binary(sample):
            return f"[error] '{path}' ดูเหมือนเป็นไฟล์ binary ที่ไม่รองรับใน read_file"
        if start_line or end_line:
            return _read_streamed_line_range(
                path, start_line or "", end_line or "", lambda: _plain_text_lines(path)
            )
        total_chars, unit_count, secret_found = _scan_plain_text(path)
        if secret_found:
            return _BLOCKED_SECRET
        if total_chars <= config.READ_FILE_MAX_CHARS:
            with open(path, "r", encoding="utf-8", errors="replace", newline=None) as fh:
                return fh.read()
        return _sample_plain_text(path, total_chars, unit_count)
    except Exception as e:
        return f"[error] อ่านไฟล์ไม่ได้: {e}"


# LangChain entry point — see tools/bash.py's 2026-07-23 note.
from langchain_core.tools import tool as _lc_tool


@_lc_tool
def read_file(path: str, start_line: str = "", end_line: str = "") -> str:
    """Read text, code, PDF, and Office files from disk, given their full path.
    Optionally read only a line range for large files.

    Use this directly when the exact path is already known -- given in the
    question, or already found by a previous rag_search call. Do not call
    rag_search again for a path you already have.
    ✅ "Read /Users/me/project/config.py for me" -> read_file(path="/Users/me/project/config.py")
    ✅ after rag_search already returned PLAN.md's path -> read_file that exact path next"""
    from helpers import turn_guard
    args = {"path": path, "start_line": start_line, "end_line": end_line}
    blocked = turn_guard.check("read_file", args)
    if blocked:
        return blocked
    result = run(args)
    turn_guard.record("read_file", args, result)
    return result
