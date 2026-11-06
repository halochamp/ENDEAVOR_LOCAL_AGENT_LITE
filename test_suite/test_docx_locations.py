"""Regression tests for page-aware DOCX extraction used by KB search."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import zipfile

from helpers.document_extract import document_page_count, iter_document_units_with_locations


def _write_minimal_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>หน้าแรก</w:t></w:r></w:p>
    <w:p><w:r><w:lastRenderedPageBreak/><w:t>ฝึกบนหน้าสอง</w:t></w:r></w:p>
    <w:p><w:r><w:t>เนื้อหาต่อหน้าเดิม</w:t></w:r></w:p>
    <w:p><w:r><w:br w:type="page"/><w:lastRenderedPageBreak/><w:t>หน้าสาม</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    app_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Pages>3</Pages>
</Properties>"""
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("word/document.xml", document_xml)
        package.writestr("docProps/app.xml", app_xml)


def test_docx_units_report_real_page_locations_and_deduplicate_break_markers(tmp_path):
    path = tmp_path / "manual.docx"
    _write_minimal_docx(path)

    assert document_page_count(path) == 3
    units = list(iter_document_units_with_locations(path))

    assert units == [
        ("page 1/3", "หน้าแรก"),
        ("page 2/3", "ฝึกบนหน้าสอง"),
        ("page 2/3", "เนื้อหาต่อหน้าเดิม"),
        ("page 3/3", "หน้าสาม"),
    ]


def test_kb_sync_persists_docx_page_location_on_each_chunk(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    raw = knowledge  # Compatibility fixture name; sources live at knowledge/ root.
    md = knowledge / "md"
    knowledge.mkdir(parents=True)
    md.mkdir()
    source = raw / "manual.docx"
    _write_minimal_docx(source)

    import config
    import kb_sync

    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", raw)
    monkeypatch.setattr(config, "KNOWLEDGE_MD_DIR", md)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "MD_DIR", md)
    monkeypatch.setattr(config, "FILE_INDEX_DB", tmp_path / "file_index.sqlite3")
    monkeypatch.setattr(config, "INDEX_ROOTS", [md])
    monkeypatch.setattr(config, "KB_CHUNK_TARGET_CHARS", 240)
    monkeypatch.setattr(config, "KB_CHUNK_OVERLAP_CHARS", 0)

    stats = kb_sync.sync_kb()
    assert stats["inserted"] == 1

    with sqlite3.connect(config.FILE_INDEX_DB) as conn:
        rows = conn.execute(
            "SELECT location, content FROM kb_chunks_fts "
            "WHERE path = ? ORDER BY chunk_index",
            (str(source.resolve()),),
        ).fetchall()

    matching = [location for location, content in rows if "ฝึกบนหน้าสอง" in content]
    assert matching == ["page 2/3"]

    from tools import rag_search

    result = rag_search.run({"query": "ฝึกบนหน้าสอง"})
    assert "หน้า/ตำแหน่ง: page 2/3" in result
