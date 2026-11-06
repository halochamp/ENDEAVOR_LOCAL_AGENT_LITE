"""Direct PDF -> text conversion for the Agent Lite browser panel.

The pipeline is deliberately deterministic-first and SSD-first:

1. Read one physical PDF page at a time.
2. Use the embedded text layer when present; otherwise rasterize only that page
   and run Apple Vision OCR.
3. Reconstruct a confident OCR grid with the same conservative table logic used
   by MAX_VLM's read_image sensor.
4. Persist each page into SQLite immediately.  The complete document is never
   held as one Python string.
5. Optionally clean Thai OCR prose with the already-running Lite model.  One
   no-think request carries several pages; native PDF text and OCR tables never
   use the LLM.
6. Stream the final rows back out in physical PDF page order with explicit page
   separators.

SQLite is staging/cache state only.  The user-facing .txt output is written to
workspect/; old staging rows are aged out automatically.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
import difflib
import hashlib
import json
import math
import re
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path

import httpx

import config
from helpers.thai_text_normalize import repair_thai_orthographic_order
from helpers.vision_ocr import read_layout


Progress = Callable[[str, int, int, dict[str, int]], None]

_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_PROSE_COL_CHARS = 14


def _connect() -> sqlite3.Connection:
    config.PDF_TO_TEXT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.PDF_TO_TEXT_DB), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pdf_text_jobs ("
        "job_id TEXT PRIMARY KEY, source_name TEXT NOT NULL, source_sha256 TEXT NOT NULL, "
        "total_pages INTEGER NOT NULL, rewrite_thai INTEGER NOT NULL, status TEXT NOT NULL, "
        "output_path TEXT, warning TEXT, created_at REAL NOT NULL, completed_at REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pdf_text_pages ("
        "job_id TEXT NOT NULL, page_number INTEGER NOT NULL, source_kind TEXT NOT NULL, "
        "raw_text TEXT NOT NULL, final_text TEXT NOT NULL, rewrite_status TEXT NOT NULL, "
        "PRIMARY KEY(job_id, page_number))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS pdf_text_pages_job_idx "
        "ON pdf_text_pages(job_id, page_number)"
    )
    conn.commit()
    return conn


def _cleanup_cache(conn: sqlite3.Connection) -> None:
    cutoff = time.time() - config.PDF_TO_TEXT_CACHE_MAX_AGE_SECONDS
    stale = [
        row[0]
        for row in conn.execute(
            "SELECT job_id FROM pdf_text_jobs WHERE created_at < ?", (cutoff,)
        )
    ]
    if not stale:
        return
    conn.executemany("DELETE FROM pdf_text_pages WHERE job_id=?", ((job,) for job in stale))
    conn.executemany("DELETE FROM pdf_text_jobs WHERE job_id=?", ((job,) for job in stale))
    conn.commit()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(text: str) -> str:
    """Conservative layout cleanup that keeps page/paragraph line breaks."""
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        lines.append(repair_thai_orthographic_order(line.rstrip()))
    # Keep intentional blank lines but avoid unlimited vertical whitespace.
    out: list[str] = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank <= 2:
                out.append("")
    return "\n".join(out).strip()


def _column_intervals(boxes: list[dict], min_gap: float = 0.025) -> list[tuple[float, float]]:
    """Current MAX_VLM OCR table column projection."""
    spans = sorted(
        ((box["x"], box["x"] + box["w"]) for box in boxes if box["w"] <= 0.4),
        key=lambda span: span[0],
    )
    if not spans:
        return []
    columns: list[list[float]] = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo - columns[-1][1] <= min_gap:
            columns[-1][1] = max(columns[-1][1], hi)
        else:
            columns.append([lo, hi])
    return [(lo, hi) for lo, hi in columns]


def _assign_column(box: dict, columns: list[tuple[float, float]]) -> int:
    left, right = box["x"], box["x"] + box["w"]
    best_index, best_overlap = 0, -1.0
    for index, (lo, hi) in enumerate(columns):
        overlap = min(right, hi) - max(left, lo)
        if overlap > best_overlap:
            best_overlap, best_index = overlap, index
    if best_overlap > 0:
        return best_index
    center = left + box["w"] / 2
    return min(
        range(len(columns)),
        key=lambda index: abs((columns[index][0] + columns[index][1]) / 2 - center),
    )


def reconstruct_ocr_table(boxes: list[dict]) -> str | None:
    """Port MAX_VLM's conservative OCR grid reconstruction byte-for-byte in policy."""
    if len(boxes) < 4:
        return None

    items = sorted(boxes, key=lambda box: -(box["y"] + box["h"] / 2))
    heights = sorted(box["h"] for box in boxes)
    median_height = heights[len(heights) // 2]
    row_threshold = max(median_height * 0.6, 0.005)

    rows: list[list[dict]] = [[items[0]]]
    for box in items[1:]:
        center = box["y"] + box["h"] / 2
        row_center = sum(item["y"] + item["h"] / 2 for item in rows[-1]) / len(rows[-1])
        if row_center - center <= row_threshold:
            rows[-1].append(box)
        else:
            rows.append([box])

    columns = _column_intervals(boxes)
    if len(columns) < 2 or len(rows) < 2:
        return None

    grid: list[list[str]] = []
    for row in rows:
        cells = [""] * len(columns)
        for box in sorted(row, key=lambda item: item["x"]):
            column = _assign_column(box, columns)
            cells[column] = (
                (cells[column] + " " + box["text"]).strip()
                if cells[column]
                else box["text"]
            )
        grid.append(cells)

    multi_rows = sum(1 for row in grid if sum(1 for cell in row if cell) >= 2)
    if multi_rows < max(2, len(grid) * 0.5):
        return None

    filled = sum(1 for row in grid for cell in row if cell)
    if filled < len(grid) * len(columns) * 0.6:
        return None

    flat = [cell for row in grid for cell in row if cell]
    numeric = sum(1 for cell in flat if any(char.isdigit() for char in cell))
    if flat and numeric / len(flat) < 0.15:
        column_medians = []
        for column in range(len(columns)):
            lengths = sorted(
                len(grid[row][column])
                for row in range(len(grid))
                if grid[row][column]
            )
            if lengths:
                column_medians.append(lengths[len(lengths) // 2])
        if column_medians and min(column_medians) > _PROSE_COL_CHARS:
            return None

    def render(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", r"\|") for cell in row) + " |"

    lines = [
        render(grid[0]),
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(render(row) for row in grid[1:])
    return "\n".join(lines)


def _ocr_page(page, page_number: int) -> tuple[str, str]:
    """Return (source_kind, text) for one image-only physical PDF page."""
    temporary: Path | None = None
    try:
        import fitz

        work_dir = config.DATA_DIR / "pdf_to_text"
        work_dir.mkdir(parents=True, exist_ok=True)
        work_dir.chmod(0o700)
        with tempfile.NamedTemporaryFile(
            prefix=f"page-{page_number:05d}-",
            suffix=".png",
            dir=work_dir,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)

        rect = page.rect
        source_pixels = float(rect.width) * float(rect.height)
        if source_pixels <= 0:
            raise RuntimeError(f"PDF page {page_number} has invalid dimensions")
        requested_scale = config.PDF_TO_TEXT_OCR_DPI / 72.0
        scale = requested_scale
        if source_pixels * scale * scale > config.PDF_TO_TEXT_OCR_MAX_PIXELS:
            scale = math.sqrt(config.PDF_TO_TEXT_OCR_MAX_PIXELS / source_pixels)
        if scale <= 0:
            raise RuntimeError(f"PDF page {page_number} is too large to rasterize safely")
        page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(str(temporary))
        boxes = read_layout(str(temporary))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    cleaned_boxes = []
    for item in boxes or ():
        text = repair_thai_orthographic_order(str(item.get("text") or "").strip())
        if not text:
            continue
        clone = dict(item)
        clone["text"] = text
        cleaned_boxes.append(clone)

    if not cleaned_boxes:
        return "ocr", ""

    table = reconstruct_ocr_table(cleaned_boxes)
    if table:
        return "ocr_table", table

    return "ocr", _clean_text("\n".join(item["text"] for item in cleaned_boxes))


def _rewrite_batches(rows: Iterable[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    for page_number, text in rows:
        size = len(text)
        if current and (
            len(current) >= config.PDF_TO_TEXT_LLM_BATCH_PAGES
            or current_chars + size > config.PDF_TO_TEXT_LLM_BATCH_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        # A single unusually dense OCR page may exceed the batch character
        # preference; keep it alone rather than split a physical page.
        current.append((page_number, text))
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _rewrite_prompt(batch: list[tuple[int, str]]) -> str:
    payload = [{"page": page, "text": text} for page, text in batch]
    return (
        "แก้ข้อความ OCR ภาษาไทยแบบอนุรักษ์นิยมเท่านั้น ห้ามสรุป ห้ามแปล "
        "ห้ามเพิ่มข้อมูล ห้ามเปลี่ยนตัวเลข/หน่วย/ชื่อเฉพาะ แก้เฉพาะอักขระ OCR "
        "หรือการเว้นวรรคที่ผิดอย่างชัดเจน รักษาบรรทัดและลำดับเนื้อหาให้ใกล้ต้นฉบับที่สุด "
        "ตอบเป็น JSON object รูป {\"pages\":[{\"page\":1,\"text\":\"...\"}]} "
        "และต้องคืนทุก page id ที่ได้รับมาครบหนึ่งครั้งเท่านั้น\n\n"
        + json.dumps({"pages": payload}, ensure_ascii=False)
    )


def _validate_rewrite(source: str, rewritten: str) -> bool:
    if not rewritten.strip():
        return False
    ratio = len(rewritten) / max(1, len(source))
    if ratio < 0.55 or ratio > 1.45:
        return False
    # This stage is an OCR corrector, not a paraphraser. A large semantic
    # rewrite can preserve length and numbers yet still invent content, so
    # require substantial character-level similarity before accepting it.
    if difflib.SequenceMatcher(None, source, rewritten, autojunk=False).ratio() < 0.72:
        return False
    # Numeric facts are cheap to protect deterministically.  Any number that
    # disappears makes this page fall back to the OCR original. Python's \d is
    # Unicode-aware, so this protects Thai digits as well as ASCII digits.
    numbers = re.findall(r"\d+(?:[.,:/-]\d+)*", source)
    return all(number in rewritten for number in numbers)


def _llm_rewrite_batch(batch: list[tuple[int, str]]) -> dict[int, str]:
    """One local no-think request for several OCR pages."""
    url = config.MLX_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative OCR text corrector. "
                    "Return JSON only. Never summarize or invent content."
                ),
            },
            {"role": "user", "content": _rewrite_prompt(batch)},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "enable_thinking": False,
        "thinking_budget": 0,
        "repetition_penalty": config.REPETITION_PENALTY,
    }
    headers = {
        "Authorization": f"Bearer {config.MLX_API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=config.PDF_TO_TEXT_LLM_TIMEOUT_SECONDS) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    content = str(body["choices"][0]["message"]["content"])
    content = _THINK_RE.sub("", content).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)
    pages = parsed.get("pages") if isinstance(parsed, dict) else None
    if not isinstance(pages, list):
        raise ValueError("LLM rewrite JSON has no pages list")

    expected = {page for page, _ in batch}
    result: dict[int, str] = {}
    for item in pages:
        if not isinstance(item, dict):
            raise ValueError("LLM rewrite page is not an object")
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError) as exc:
            raise ValueError("LLM rewrite page id is invalid") from exc
        text = str(item.get("text") or "")
        if page in result:
            raise ValueError("LLM rewrite duplicated a page")
        result[page] = text
    if set(result) != expected:
        raise ValueError("LLM rewrite changed the page set")
    return result


def _apply_thai_rewrite(
    conn: sqlite3.Connection,
    job_id: str,
    total_pages: int,
    progress: Progress | None,
) -> tuple[int, int, list[str]]:
    rows = [
        (int(page), str(text))
        for page, text in conn.execute(
            "SELECT page_number, raw_text FROM pdf_text_pages "
            "WHERE job_id=? AND source_kind='ocr' ORDER BY page_number",
            (job_id,),
        )
        if text and _THAI_RE.search(str(text))
    ]
    if not rows:
        return 0, 0, []

    rewritten_count = 0
    fallback_count = 0
    warnings: list[str] = []
    batches = _rewrite_batches(rows)
    for batch_index, batch in enumerate(batches, 1):
        if progress:
            progress(
                "rewrite",
                batch[0][0],
                total_pages,
                {
                    "batch": batch_index,
                    "batches": len(batches),
                    "pages": len(batch),
                    "rewritten": rewritten_count,
                },
            )
        try:
            rewritten = _llm_rewrite_batch(batch)
        except Exception as exc:
            fallback_count += len(batch)
            warnings.append(
                f"LLM cleanup batch {batch_index}/{len(batches)} ใช้ไม่ได้: "
                f"{type(exc).__name__}; ใช้ OCR เดิม"
            )
            conn.executemany(
                "UPDATE pdf_text_pages SET rewrite_status='fallback' "
                "WHERE job_id=? AND page_number=?",
                ((job_id, page) for page, _ in batch),
            )
            conn.commit()
            continue

        for page, source in batch:
            candidate = _clean_text(rewritten.get(page, ""))
            if _validate_rewrite(source, candidate):
                conn.execute(
                    "UPDATE pdf_text_pages SET final_text=?, rewrite_status='rewritten' "
                    "WHERE job_id=? AND page_number=?",
                    (candidate, job_id, page),
                )
                rewritten_count += 1
            else:
                conn.execute(
                    "UPDATE pdf_text_pages SET rewrite_status='fallback' "
                    "WHERE job_id=? AND page_number=?",
                    (job_id, page),
                )
                fallback_count += 1
        conn.commit()
    return rewritten_count, fallback_count, warnings


def _safe_output_path(source_name: str) -> Path:
    stem = Path(source_name).stem.strip() or "document"
    stem = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", stem).strip(" .") or "document"
    base = config.WORKSPECT_DIR / f"{stem}_pdf_to_text.txt"
    if not base.exists():
        return base
    for index in range(2, 10000):
        candidate = config.WORKSPECT_DIR / f"{stem}_pdf_to_text_{index}.txt"
        if not candidate.exists():
            return candidate
    raise RuntimeError("ไม่สามารถสร้างชื่อ output ที่ไม่ชนไฟล์เดิมได้")


def _write_output(conn: sqlite3.Connection, job_id: str, total_pages: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for page_number, text in conn.execute(
            "SELECT page_number, final_text FROM pdf_text_pages "
            "WHERE job_id=? ORDER BY page_number",
            (job_id,),
        ):
            handle.write(f"===== PDF PAGE {page_number}/{total_pages} =====\n")
            if text:
                handle.write(str(text).rstrip())
            handle.write("\n")
            if page_number != total_pages:
                handle.write("\n")


def convert_pdf(
    source: str | Path,
    *,
    source_name: str | None = None,
    rewrite_thai: bool = True,
    progress: Progress | None = None,
) -> dict[str, object]:
    """Convert a PDF into page-delimited text and return a compact job summary."""
    path = Path(source).expanduser().resolve()
    if not path.is_file() or path.suffix.casefold() != ".pdf":
        raise ValueError("ต้องเป็นไฟล์ PDF ที่มีอยู่จริง")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("PDF ว่าง")
    if size > config.PDF_TO_TEXT_MAX_BYTES:
        raise ValueError(
            f"PDF ใหญ่เกิน {config.PDF_TO_TEXT_MAX_BYTES // (1024 * 1024)}MB"
        )

    try:
        import fitz
        pdf = fitz.open(path)
    except Exception as exc:
        raise ValueError(f"เปิด PDF ไม่สำเร็จ: {exc}") from exc

    name = Path(source_name or path.name).name
    job_id = uuid.uuid4().hex
    stats = {
        "native": 0,
        "ocr": 0,
        "tables": 0,
        "rewritten": 0,
        "rewrite_fallback": 0,
    }
    warnings: list[str] = []
    conn = _connect()
    try:
        _cleanup_cache(conn)
        total_pages = len(pdf)
        if total_pages <= 0:
            raise ValueError("PDF ไม่มีหน้า")
        conn.execute(
            "INSERT INTO pdf_text_jobs("
            "job_id, source_name, source_sha256, total_pages, rewrite_thai, status, created_at"
            ") VALUES(?,?,?,?,?,'running',?)",
            (job_id, name, _sha256(path), total_pages, int(bool(rewrite_thai)), time.time()),
        )
        conn.commit()

        for page_number, page in enumerate(pdf, 1):
            if progress:
                progress("extract", page_number, total_pages, dict(stats))
            native = _clean_text(page.get_text("text") or "")
            native_signal_chars = sum(char.isalnum() for char in native)
            if native and native_signal_chars >= config.PDF_TO_TEXT_NATIVE_MIN_CHARS:
                source_kind, text = "native", native
                stats["native"] += 1
            else:
                # A hybrid page can contain a tiny native footer/page number
                # over a scanned body. Treat sparse native text as a hint, not
                # proof that the page is born-digital: OCR once and keep the
                # richer result only when it materially adds content.
                ocr_kind, ocr_text = _ocr_page(page, page_number)
                if native and len(ocr_text.strip()) <= len(native.strip()) + 24:
                    source_kind, text = "native", native
                    stats["native"] += 1
                else:
                    source_kind, text = ocr_kind, ocr_text
                    if source_kind == "ocr_table":
                        stats["tables"] += 1
                    else:
                        stats["ocr"] += 1
            conn.execute(
                "INSERT INTO pdf_text_pages("
                "job_id, page_number, source_kind, raw_text, final_text, rewrite_status"
                ") VALUES(?,?,?,?,?,'not_needed')",
                (job_id, page_number, source_kind, text, text),
            )
            conn.commit()
    finally:
        pdf.close()

    try:
        if rewrite_thai:
            rewritten, fallback, rewrite_warnings = _apply_thai_rewrite(
                conn, job_id, total_pages, progress
            )
            stats["rewritten"] = rewritten
            stats["rewrite_fallback"] = fallback
            warnings.extend(rewrite_warnings)

        if progress:
            progress("write", total_pages, total_pages, dict(stats))
        output = _safe_output_path(name)
        _write_output(conn, job_id, total_pages, output)
        warning_text = "\n".join(warnings)
        conn.execute(
            "UPDATE pdf_text_jobs SET status='done', output_path=?, warning=?, completed_at=? "
            "WHERE job_id=?",
            (str(output), warning_text, time.time(), job_id),
        )
        conn.commit()
        return {
            "job_id": job_id,
            "source_name": name,
            "total_pages": total_pages,
            "output_path": str(output),
            "rewrite_thai": bool(rewrite_thai),
            "stats": stats,
            "warnings": warnings,
        }
    except Exception as exc:
        conn.execute(
            "UPDATE pdf_text_jobs SET status='error', warning=?, completed_at=? WHERE job_id=?",
            (f"{type(exc).__name__}: {exc}", time.time(), job_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()
