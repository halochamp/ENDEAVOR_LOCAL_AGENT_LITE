"""Document extraction and bounded sampling for Lite read_file."""
from __future__ import annotations

import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import read_file
from test_suite.pptx_fixture import write_pptx


def test_read_file_extracts_pdf_docx_xlsx_and_pptx(tmp_path):
    import fitz
    from docx import Document
    from openpyxl import Workbook

    pdf = tmp_path / "report.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF extraction works")
    doc.save(pdf)
    doc.close()

    docx = tmp_path / "report.docx"
    word = Document()
    word.add_paragraph("Word extraction works")
    word.save(docx)

    xlsx = tmp_path / "report.xlsx"
    book = Workbook()
    book.active.title = "Data"
    book.active.append(["Metric", "Value"])
    book.active.append(["Revenue", 42])
    book.save(xlsx)

    pptx = tmp_path / "report.pptx"
    write_pptx(pptx, ["PowerPoint extraction works"])

    docm = tmp_path / "report.docm"
    xlsm = tmp_path / "report.xlsm"
    pptm = tmp_path / "report.pptm"
    shutil.copyfile(docx, docm)
    shutil.copyfile(xlsx, xlsm)
    shutil.copyfile(pptx, pptm)

    assert "PDF extraction works" in read_file.run({"path": str(pdf)})
    assert "Word extraction works" in read_file.run({"path": str(docx)})
    assert "Revenue | 42" in read_file.run({"path": str(xlsx)})
    assert "PowerPoint extraction works" in read_file.run({"path": str(pptx)})
    assert "Word extraction works" in read_file.run({"path": str(docm)})
    assert "Revenue | 42" in read_file.run({"path": str(xlsm)})
    assert "PowerPoint extraction works" in read_file.run({"path": str(pptm)})


def test_read_file_extracts_ods_and_odp_content_xml(tmp_path):
    ods = tmp_path / "ledger.ods"
    odp = tmp_path / "deck.odp"
    spreadsheet_xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<office:document-content xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\" xmlns:text=\"urn:oasis:names:tc:opendocument:xmlns:text:1.0\"><office:body><office:spreadsheet><text:p>ODS revenue 42</text:p></office:spreadsheet></office:body></office:document-content>"""
    presentation_xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<office:document-content xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\" xmlns:text=\"urn:oasis:names:tc:opendocument:xmlns:text:1.0\"><office:body><office:presentation><text:h>ODP roadmap</text:h><text:p>Launch in Q4</text:p></office:presentation></office:body></office:document-content>"""
    for path, content in ((ods, spreadsheet_xml), (odp, presentation_xml)):
        with zipfile.ZipFile(path, "w") as package:
            package.writestr("content.xml", content)

    assert "ODS revenue 42" in read_file.run({"path": str(ods)})
    out = read_file.run({"path": str(odp)})
    assert "ODP roadmap" in out
    assert "Launch in Q4" in out


def test_read_file_extracts_legacy_ppt_via_spotlight(tmp_path, monkeypatch):
    from helpers import document_extract

    legacy_ppt = tmp_path / "legacy.ppt"
    legacy_ppt.write_bytes(b"legacy-powerpoint")
    monkeypatch.setattr(document_extract, "_extract_spotlight_document", lambda _path: "Legacy PPT extraction works")

    assert "Legacy PPT extraction works" in read_file.run({"path": str(legacy_ppt)})


def test_read_file_extracts_legacy_doc_via_textutil(tmp_path, monkeypatch):
    from helpers import document_extract

    legacy_doc = tmp_path / "legacy.doc"
    legacy_doc.write_bytes(b"\xd0\xcf\x11\xe0legacy-word-document")
    commands: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = "Legacy Word extraction works\n"
        stderr = ""

    def fake_run(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 30
        return _Result()

    monkeypatch.setattr(document_extract.shutil, "which", lambda _: "/usr/bin/textutil")
    monkeypatch.setattr(document_extract.subprocess, "run", fake_run)

    out = read_file.run({"path": str(legacy_doc)})
    assert "Legacy Word extraction works" in out
    assert commands == [[
        "/usr/bin/textutil", "-convert", "txt", "-stdout", "-encoding", "UTF-8", str(legacy_doc),
    ]]


def test_read_file_extracts_odt_via_textutil(tmp_path, monkeypatch):
    from helpers import document_extract

    legacy_odt = tmp_path / "legacy.odt"
    legacy_odt.write_bytes(b"odt-document")

    class _Result:
        returncode = 0
        stdout = "ODT extraction works\n"
        stderr = ""

    monkeypatch.setattr(document_extract.shutil, "which", lambda _: "/usr/bin/textutil")
    monkeypatch.setattr(document_extract.subprocess, "run", lambda *_args, **_kwargs: _Result())

    out = read_file.run({"path": str(legacy_odt)})

    assert "ODT extraction works" in out


def test_read_file_uses_ocr_fallback_for_scanned_pdf(tmp_path, monkeypatch):
    import fitz
    from helpers import document_extract

    pdf = tmp_path / "scanned.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf)
    document.close()
    monkeypatch.setattr(document_extract, "_ocr_scanned_pdf", lambda _path: "OCR fallback text")

    assert "OCR fallback text" in read_file.run({"path": str(pdf)})


def test_large_ocr_pdf_sampling_keeps_page_markers_across_document(tmp_path):
    path = tmp_path / "scanned.pdf"
    pages = [f"--- PDF page {number} (OCR) ---\n" + (f"page {number} text " * 180) for number in range(1, 13)]
    text = "[PDF สแกน — OCR พบข้อความ 12/12 หน้าที่ตรวจ]\n\n" + "\n\n".join(pages)

    out = read_file._sample_for_coverage(text, path)
    assert "--- PDF page 1 (OCR) ---" in out
    assert "--- PDF page 12 (OCR) ---" in out
    assert "ตัวอย่าง" in out
    assert len(out) <= 5_000


def test_large_read_is_sampled_to_fixed_5000_chars(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("\n".join(f"line {i}: " + "x" * 120 for i in range(200)), encoding="utf-8")
    out = read_file.run({"path": str(path)})
    assert "sampled across file" in out
    assert len(out) <= 5_000
    assert "line 0:" in out
    assert "line 199:" in out
