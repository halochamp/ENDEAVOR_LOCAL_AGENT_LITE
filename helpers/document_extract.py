# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Lean, read-only text extraction for PDF and Office documents.

extract_text() stays the full-body, byte-identical entry point existing
tests and small-file callers use. Large-file callers (indexer.py,
tools/read_file.py) that only need a bounded prefix or a scattered sample
use the streaming primitives below (iter_document_units, pdf_page_count,
pdf_page_text) instead, so a multi-hundred-page PDF or huge XLSX never has
its whole extracted body held in memory at once (PLAN.md §9.1).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path

from helpers.vision_ocr import read_layout


_DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".docm", ".odt", ".xlsx", ".xls", ".xlsm",
    ".ppt", ".pptx", ".pptm", ".ods", ".odp",
}
_SCANNED_PDF_OCR_MAX_PAGES = 30
_UNIT_SEPARATORS = {
    ".pdf": "\n\n", ".doc": "\n", ".docx": "\n", ".docm": "\n", ".odt": "\n",
    ".xlsx": "\n", ".xls": "\n", ".xlsm": "\n", ".ppt": "\n", ".pptx": "\n",
    ".pptm": "\n", ".ods": "\n", ".odp": "\n",
}
_TEXTUTIL_DOCUMENT_EXTENSIONS = {".doc", ".odt"}
_ODF_CONTENT_EXTENSIONS = {".ods", ".odp"}


def is_supported_document(path: Path) -> bool:
    return path.suffix.lower() in _DOCUMENT_EXTENSIONS


def unit_separator(suffix: str) -> str:
    """Separator extract_text() used to join this format's units."""
    return _UNIT_SEPARATORS.get(suffix.lower(), "\n")


def pdf_page_count(path: Path) -> int:
    import fitz
    with fitz.open(path) as pdf:
        return len(pdf)


def pdf_page_text(path: Path, index: int) -> str:
    """Extract one page's text on demand (0-indexed), without touching any
    other page -- lets a caller re-fetch a handful of scattered pages for
    sampling without ever extracting the whole document."""
    import fitz
    with fitz.open(path) as pdf:
        return pdf[index].get_text("text").strip()


def iter_document_units(path: Path) -> Iterator[str]:
    """Yield one formatted unit at a time -- PDF page, DOCX paragraph/table
    row, XLSX/XLS sheet header/row, or PPTX slide/shape/table row -- in the
    same order and exact text extract_text() joins with unit_separator().
    A caller only needing a bounded prefix can stop iterating early; the
    unyielded remainder of the document is never extracted.

    Scanned (image-only) PDFs are the one exception: their content isn't
    discoverable page-by-page without running OCR, so this yields the whole
    OCR'd result (already bounded to _SCANNED_PDF_OCR_MAX_PAGES) as a single
    unit, exactly matching extract_text()'s fallback.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz
        with fitz.open(path) as pdf:
            has_content = any(page.get_text("text").strip() for page in pdf)
        if not has_content:
            ocr_text = _ocr_scanned_pdf(path)
            if ocr_text.startswith("[error]"):
                raise RuntimeError(ocr_text)
            yield ocr_text
            return
        with fitz.open(path) as pdf:
            for number, page in enumerate(pdf, 1):
                yield f"--- PDF page {number} ---\n{page.get_text('text').strip()}"
    elif suffix in {".docx", ".docm"}:
        yield from _iter_docx_units(path)
    elif suffix in _TEXTUTIL_DOCUMENT_EXTENSIONS:
        text = _extract_textutil_document(path)
        if text.startswith("[error]"):
            raise RuntimeError(text)
        yield text
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        yield from _iter_workbook_units(path, suffix)
    elif suffix in {".pptx", ".pptm"}:
        yield from _iter_pptx_units(path)
    elif suffix in _ODF_CONTENT_EXTENSIONS:
        yield from _iter_odf_content_units(path)
    elif suffix == ".ppt":
        text = _extract_spotlight_document(path)
        if text.startswith("[error]"):
            raise RuntimeError(text)
        yield text
    else:
        raise ValueError(f"no streaming extractor for {suffix}")


def _iter_docx_units(path: Path) -> Iterator[str]:
    from docx import Document
    doc = Document(path)
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            yield text
    for table_number, table in enumerate(doc.tables, 1):
        yield f"--- Table {table_number} ---"
        for row in table.rows:
            cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
            if any(cells):
                yield " | ".join(cells)


def _iter_workbook_units(path: Path, suffix: str) -> Iterator[str]:
    if suffix in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                yield f"--- Sheet: {sheet.title} ---"
                for row in sheet.iter_rows(values_only=True):
                    cells = [str(value).strip() if value is not None else "" for value in row]
                    if any(cells):
                        yield " | ".join(cells)
        finally:
            workbook.close()
    else:
        import xlrd
        workbook = xlrd.open_workbook(path, on_demand=True)
        try:
            for sheet in workbook.sheets():
                yield f"--- Sheet: {sheet.name} ---"
                for row_index in range(sheet.nrows):
                    cells = [str(value).strip() if value is not None else "" for value in sheet.row_values(row_index)]
                    if any(cells):
                        yield " | ".join(cells)
        finally:
            workbook.release_resources()


def _iter_pptx_units(path: Path) -> Iterator[str]:
    """Stream visible text from OOXML slides without python-pptx.

    PowerPoint files are ZIP packages.  Parsing one shape at a time avoids
    both an undeclared runtime dependency and python-pptx's full-deck object
    graph, which is unnecessary for this read-only text path.
    """
    try:
        package = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"เปิด PowerPoint package ไม่สำเร็จ: {exc}") from exc
    with package:
        slides: list[tuple[int, str]] = []
        for name in package.namelist():
            prefix = "ppt/slides/slide"
            suffix = ".xml"
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            number_text = name[len(prefix) : -len(suffix)]
            if number_text.isdigit():
                slides.append((int(number_text), name))
        for number, member in sorted(slides):
            yield f"--- Slide {number} ---"
            try:
                stream = package.open(member)
            except (KeyError, OSError) as exc:
                raise RuntimeError(f"อ่าน slide {number} ไม่สำเร็จ: {exc}") from exc
            with stream:
                try:
                    for _event, element in ET.iterparse(stream, events=("end",)):
                        local_name = element.tag.rsplit("}", 1)[-1]
                        if local_name == "sp":
                            text_body = next(
                                (child for child in element if child.tag.rsplit("}", 1)[-1] == "txBody"),
                                None,
                            )
                            text = _ooxml_text(text_body)
                            if text:
                                yield text
                            element.clear()
                        elif local_name == "graphicFrame":
                            table = next(
                                (child for child in element.iter() if child.tag.rsplit("}", 1)[-1] == "tbl"),
                                None,
                            )
                            if table is not None:
                                for row in (child for child in table if child.tag.rsplit("}", 1)[-1] == "tr"):
                                    cells = [
                                        _ooxml_text(cell).replace("\n", " ").strip()
                                        for cell in row
                                        if cell.tag.rsplit("}", 1)[-1] == "tc"
                                    ]
                                    if any(cells):
                                        yield " | ".join(cells)
                            element.clear()
                except ET.ParseError as exc:
                    raise RuntimeError(f"slide{number}.xml ไม่ถูกต้อง: {exc}") from exc


def _ooxml_text(container: ET.Element | None) -> str:
    if container is None:
        return ""
    paragraphs: list[str] = []
    for paragraph in (item for item in container.iter() if item.tag.rsplit("}", 1)[-1] == "p"):
        pieces: list[str] = []
        for item in paragraph.iter():
            local_name = item.tag.rsplit("}", 1)[-1]
            if local_name == "t":
                pieces.append(item.text or "")
            elif local_name == "br":
                pieces.append("\n")
        text = "".join(pieces).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _iter_odf_content_units(path: Path) -> Iterator[str]:
    """Yield visible paragraph/heading text from an ODS or ODP content.xml.

    OpenDocument files are ZIP packages.  Their content.xml uses text:p/text:h
    for spreadsheet cells and presentation text boxes alike, so parsing those
    elements preserves user-visible text without adding an ODF dependency.
    """
    try:
        with zipfile.ZipFile(path) as package, package.open("content.xml") as stream:
            for _event, element in ET.iterparse(stream, events=("end",)):
                if element.tag.rsplit("}", 1)[-1] in {"p", "h"}:
                    text = "".join(element.itertext()).strip()
                    if text:
                        yield text
                    element.clear()
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"เปิด OpenDocument package ไม่สำเร็จ: {exc}") from exc
    except ET.ParseError as exc:
        raise RuntimeError(f"content.xml ไม่ถูกต้อง: {exc}") from exc


def _extract_spotlight_document(path: Path) -> str:
    """Read text imported by macOS Spotlight for a legacy PowerPoint file."""
    mdls = shutil.which("mdls")
    if not mdls:
        return "[error] อ่านไฟล์ .ppt ไม่ได้: ไม่พบ macOS mdls"
    try:
        result = subprocess.run(
            [mdls, "-raw", "-name", "kMDItemTextContent", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[error] สกัดข้อความจาก .ppt ไม่สำเร็จ: {exc}"
    text = result.stdout.strip()
    if result.returncode != 0 or not text or text == "(null)":
        return "[error] ไฟล์ .ppt ไม่มีข้อความจาก macOS Spotlight importer"
    return text


def extract_text(path: Path) -> str:
    """Return document text or an ``[error]`` string; never writes a file.

    Built on iter_document_units() so the full-body path and the streaming
    (bounded-prefix/sample) path used by indexer.py/read_file.py can never
    silently diverge -- there is exactly one extraction implementation per
    format, just two ways of consuming it.
    """
    suffix = path.suffix.lower()
    try:
        if suffix not in _DOCUMENT_EXTENSIONS:
            return f"[error] ไม่รองรับไฟล์ {suffix}"
        text = unit_separator(suffix).join(iter_document_units(path))
        return text.strip() if suffix == ".pdf" else text
    except Exception as exc:
        return f"[error] สกัดข้อความจาก {suffix} ไม่สำเร็จ: {exc}"


def _ocr_scanned_pdf(path: Path) -> str:
    """Port Agent Max's bounded PDF rasterization + Apple Vision OCR fallback."""
    try:
        import fitz
        with fitz.open(path) as pdf:
            total = len(pdf)
            page_limit = min(total, _SCANNED_PDF_OCR_MAX_PAGES)
            blocks: list[str] = []
            for index in range(page_limit):
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                        temporary = Path(handle.name)
                    pdf[index].get_pixmap(dpi=150).save(str(temporary))
                    lines = [box["text"] for box in read_layout(str(temporary)) if box.get("text")]
                    if lines:
                        blocks.append(f"--- PDF page {index + 1} (OCR) ---\n" + "\n".join(lines))
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
    except Exception as exc:
        return f"[error] OCR PDF สแกนไม่สำเร็จ: {exc}"

    if not blocks:
        return f"[error] PDF สแกนนี้ไม่มีข้อความที่ OCR อ่านได้ ({total} หน้า)"
    note = (
        f"\n\n[หมายเหตุ: OCR พบข้อความ {len(blocks)}/{page_limit} หน้าที่ตรวจจากทั้งหมด {total} หน้า]"
        if total > _SCANNED_PDF_OCR_MAX_PAGES else ""
    )
    header = f"[PDF สแกน — OCR พบข้อความ {len(blocks)}/{page_limit} หน้าที่ตรวจ]"
    return header + "\n\n" + "\n\n".join(blocks) + note


def _extract_textutil_document(path: Path) -> str:
    """Extract a legacy Office text document with macOS's textutil utility."""
    suffix = path.suffix.lower()
    textutil = shutil.which("textutil")
    if not textutil:
        return f"[error] อ่านไฟล์ {suffix} ไม่ได้: ไม่พบ macOS textutil"
    try:
        result = subprocess.run(
            [textutil, "-convert", "txt", "-stdout", "-encoding", "UTF-8", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[error] สกัดข้อความจาก {suffix} ไม่สำเร็จ: {exc}"
    if result.returncode != 0:
        detail = " ".join(result.stderr.split())[:240]
        suffix = f" ({detail})" if detail else ""
        return f"[error] สกัดข้อความจาก {path.suffix.lower()} ไม่สำเร็จ{suffix}"
    text = result.stdout.strip()
    if not text:
        return f"[error] ไฟล์ {path.suffix.lower()} ไม่มีข้อความที่สกัดได้ (อาจเป็นเอกสารสแกนหรือไฟล์เสีย)"
    return text
