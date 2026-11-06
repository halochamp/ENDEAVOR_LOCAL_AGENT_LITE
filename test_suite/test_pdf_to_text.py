"""Deterministic tests for the direct PDF -> text pipeline."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import fitz

import config
import pdf_to_text


def _configure_tmp(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "workspect"
    data_dir.mkdir()
    out_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "PDF_TO_TEXT_DB", data_dir / "pdf_to_text.sqlite3")
    monkeypatch.setattr(config, "WORKSPECT_DIR", out_dir)
    monkeypatch.setattr(config, "PDF_TO_TEXT_CACHE_MAX_AGE_SECONDS", 86400)
    monkeypatch.setattr(config, "PDF_TO_TEXT_OCR_DPI", 150)
    monkeypatch.setattr(config, "PDF_TO_TEXT_MAX_BYTES", 50 * 1024 * 1024)


def test_native_pdf_preserves_physical_page_numbers(monkeypatch, tmp_path):
    _configure_tmp(monkeypatch, tmp_path)
    source = tmp_path / "native.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Native page one contains enough born digital text to bypass OCR completely.")
        page = doc.new_page()
        page.insert_text((72, 72), "Native page two also contains enough searchable text to stay on the native path.")
        doc.save(source)

    result = pdf_to_text.convert_pdf(source, rewrite_thai=False)
    output = Path(result["output_path"]).read_text(encoding="utf-8")

    assert "===== PDF PAGE 1/2 =====" in output
    assert "===== PDF PAGE 2/2 =====" in output
    assert output.index("PDF PAGE 1/2") < output.index("PDF PAGE 2/2")
    assert "Native page one contains enough born digital text" in output
    assert "Native page two also contains enough searchable text" in output
    assert result["stats"]["native"] == 2
    assert result["stats"]["ocr"] == 0

    with sqlite3.connect(config.PDF_TO_TEXT_DB) as conn:
        rows = conn.execute(
            "SELECT page_number, source_kind FROM pdf_text_pages ORDER BY page_number"
        ).fetchall()
    assert rows == [(1, "native"), (2, "native")]


def test_scanned_page_uses_max_vlm_table_reconstruction(monkeypatch, tmp_path):
    _configure_tmp(monkeypatch, tmp_path)
    source = tmp_path / "scan.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(source)

    boxes = [
        {"text": "หัวข้อ", "x": 0.10, "y": 0.80, "w": 0.18, "h": 0.05},
        {"text": "ค่า", "x": 0.60, "y": 0.80, "w": 0.10, "h": 0.05},
        {"text": "A", "x": 0.10, "y": 0.60, "w": 0.10, "h": 0.05},
        {"text": "10", "x": 0.60, "y": 0.60, "w": 0.10, "h": 0.05},
    ]
    monkeypatch.setattr(pdf_to_text, "read_layout", lambda _path: boxes)

    result = pdf_to_text.convert_pdf(source, rewrite_thai=False)
    output = Path(result["output_path"]).read_text(encoding="utf-8")

    assert result["stats"]["tables"] == 1
    assert result["stats"]["ocr"] == 0
    assert "| หัวข้อ | ค่า |" in output
    assert "| --- | --- |" in output
    assert "| A | 10 |" in output

    with sqlite3.connect(config.PDF_TO_TEXT_DB) as conn:
        row = conn.execute(
            "SELECT source_kind, raw_text, final_text FROM pdf_text_pages WHERE page_number=1"
        ).fetchone()
    assert row[0] == "ocr_table"
    assert row[1] == row[2]


def test_rewrite_batches_pack_multiple_pages_without_parallel_requests(monkeypatch):
    monkeypatch.setattr(config, "PDF_TO_TEXT_LLM_BATCH_PAGES", 4)
    monkeypatch.setattr(config, "PDF_TO_TEXT_LLM_BATCH_CHARS", 100)
    rows = [(1, "ก" * 20), (2, "ข" * 20), (3, "ค" * 20), (4, "ง" * 20), (5, "จ" * 20)]
    batches = pdf_to_text._rewrite_batches(rows)
    assert [len(batch) for batch in batches] == [4, 1]
    assert [page for page, _ in batches[0]] == [1, 2, 3, 4]


def test_llm_rewrite_is_one_no_think_request_for_many_pages(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "pages": [
                                        {"page": 1, "text": "ข้อความหนึ่ง"},
                                        {"page": 2, "text": "ข้อความสอง"},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["payload"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(pdf_to_text.httpx, "Client", FakeClient)
    result = pdf_to_text._llm_rewrite_batch(
        [(1, "ขอ้ความหนึ่ง"), (2, "ขอ้ความสอง")]
    )

    assert result == {1: "ข้อความหนึ่ง", 2: "ข้อความสอง"}
    payload = captured["payload"]
    assert payload["enable_thinking"] is False
    assert payload["thinking_budget"] == 0
    assert len(json.loads(payload["messages"][1]["content"].split("\n\n", 1)[1])["pages"]) == 2


def test_sparse_native_page_uses_richer_ocr_result(monkeypatch, tmp_path):
    _configure_tmp(monkeypatch, tmp_path)
    source = tmp_path / "hybrid.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "P1")
        doc.save(source)

    monkeypatch.setattr(
        pdf_to_text,
        "_ocr_page",
        lambda _page, _number: (
            "ocr",
            "เนื้อหาจากภาพสแกนที่ยาวกว่าข้อความ native footer อย่างชัดเจน " * 3,
        ),
    )
    result = pdf_to_text.convert_pdf(source, rewrite_thai=False)
    output = Path(result["output_path"]).read_text(encoding="utf-8")

    assert result["stats"]["native"] == 0
    assert result["stats"]["ocr"] == 1
    assert "เนื้อหาจากภาพสแกน" in output
    assert "P1" not in output


def test_ocr_rasterization_obeys_pixel_ceiling(monkeypatch, tmp_path):
    _configure_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "PDF_TO_TEXT_OCR_MAX_PIXELS", 1_000_000)
    monkeypatch.setattr(pdf_to_text, "read_layout", lambda _path: [])
    captured = {}

    class Rect:
        width = 10_000.0
        height = 10_000.0

    class Pixmap:
        def save(self, path):
            Path(path).write_bytes(b"png")

    class Page:
        rect = Rect()

        def get_pixmap(self, *, matrix, alpha):
            captured["scale"] = matrix.a
            captured["alpha"] = alpha
            return Pixmap()

    kind, text = pdf_to_text._ocr_page(Page(), 1)
    assert kind == "ocr"
    assert text == ""
    assert captured["alpha"] is False
    assert captured["scale"] <= 0.100001


def test_rewrite_validation_fails_closed_on_lost_numeric_fact_or_large_rewrite():
    assert pdf_to_text._validate_rewrite("ค่า 123 บาท", "ค่า 123 บาท")
    assert not pdf_to_text._validate_rewrite("ค่า 123 บาท", "ค่า บาท")
    assert not pdf_to_text._validate_rewrite(
        "รายการเดิม 123 ต้องรักษาข้อความต้นฉบับนี้ไว้",
        "ประโยคใหม่ 123 ที่มีความหมายต่างออกไปทั้งหมดครับ",
    )
