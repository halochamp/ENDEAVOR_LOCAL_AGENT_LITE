"""Deterministic user-folder content search tests."""
from __future__ import annotations

from pathlib import Path

import config
from tools import local_search


def test_local_search_finds_content_in_desktop_and_documents_only(tmp_path, monkeypatch):
    desktop = tmp_path / "Desktop"
    documents = tmp_path / "Documents"
    outside = tmp_path / "Other"
    desktop.mkdir()
    documents.mkdir()
    outside.mkdir()
    target = documents / "camera-notes.txt"
    target.write_text("อายุการใช้งานกล้องขึ้นกับสภาพแวดล้อมและการดูแล", encoding="utf-8")
    (outside / "not-in-scope.txt").write_text("อายุการใช้งานกล้อง", encoding="utf-8")

    monkeypatch.setattr(config, "LOCAL_SEARCH_ROOTS", (desktop, documents), raising=False)

    result = local_search.run({"query": "อายุการใช้งานกล้อง"})

    assert str(target) in result
    assert "หน้า/ตำแหน่ง: บรรทัด 1" in result
    assert "ตัวอย่าง: อายุการใช้งานกล้อง" in result
    assert "อายุการใช้งานกล้อง" in result
    assert str(outside / "not-in-scope.txt") not in result


def test_local_search_reports_pdf_page_without_using_an_llm(tmp_path, monkeypatch):
    documents = tmp_path / "Documents"
    documents.mkdir()
    target = documents / "camera-manual.pdf"
    target.write_bytes(b"synthetic pdf fixture")

    def fake_units(_path):
        yield "--- PDF page 1 ---\nทั่วไป"
        yield "--- PDF page 2 ---\nอายุการใช้งานกล้องประมาณห้าปี"

    monkeypatch.setattr(config, "LOCAL_SEARCH_ROOTS", (documents,), raising=False)
    monkeypatch.setattr(local_search, "iter_document_units", fake_units)

    result = local_search.run({"query": "อายุการใช้งานกล้อง"})

    assert str(target) in result
    assert "หน้า/ตำแหน่ง: หน้า 2" in result
    assert "ตัวอย่าง: อายุการใช้งานกล้องประมาณห้าปี" in result


def test_local_search_rejects_empty_query():
    assert local_search.run({"query": ""}).startswith("[error]")
