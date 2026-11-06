"""Batch/concurrent read_file behavior."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import read_file


def test_read_file_reads_four_paths_in_input_order(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"doc-{index}.md"
        path.write_text(f"content-{index}\n", encoding="utf-8")
        paths.append(str(path))

    result = read_file.run({"paths": paths})
    assert result.startswith("[read_file batch: 4 files; concurrent]")
    positions = [result.index(f"===== {path} =====") for path in paths]
    assert positions == sorted(positions)
    for index in range(4):
        assert f"content-{index}" in result


def test_read_file_rejects_more_than_four_paths(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"doc-{index}.txt"
        path.write_text("content\n", encoding="utf-8")
        paths.append(str(path))
    result = read_file.run({"paths": paths})
    assert "ไม่เกิน 4" in result



def test_read_file_requests_support_independent_contains_and_line_ranges(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    third = tmp_path / "third.txt"
    first.write_text("zero\nAlpha target\nafter alpha\nnoise\n", encoding="utf-8")
    second.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    third.write_text("before\nBETA TARGET\nafter\n", encoding="utf-8")

    result = read_file.run({
        "requests": [
            {"path": str(first), "contains": "alpha", "context_lines": 1},
            {"path": str(second), "start_line": "2", "end_line": "4"},
            {"path": str(third), "contains": "beta target", "context_lines": 0},
        ]
    })

    assert result.startswith("[read_file batch: 3 files; independent requests; concurrent]")
    positions = [result.index(f"===== {path} =====") for path in (first, second, third)]
    assert positions == sorted(positions)
    assert "1: zero" in result
    assert "2: Alpha target" in result
    assert "3: after alpha" in result
    assert "[บรรทัด 2-4 จาก 5]" in result
    assert "two\nthree\nfour" in result
    assert "2: BETA TARGET" in result


def test_contains_searches_entire_large_file_not_normal_sample(tmp_path):
    path = tmp_path / "large.txt"
    lines = [f"ordinary line {index}" for index in range(900)]
    lines.append("TAIL_ONLY_MAGIC marker")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = read_file.run({
        "requests": [{"path": str(path), "contains": "tail_only_magic", "context_lines": 1}]
    })

    assert "TAIL_ONLY_MAGIC marker" in result
    assert "1 matches" in result
    assert "901:" in result


def test_contains_is_case_insensitive_and_output_is_bounded(tmp_path):
    path = tmp_path / "many.txt"
    path.write_text("\n".join(f"Needle match {index} " + "x" * 80 for index in range(200)), encoding="utf-8")

    result = read_file.run({"path": str(path), "contains": "needle", "context_lines": 0})

    assert "200 matches" in result
    assert "Needle match 0" in result
    assert len(result) <= 5_000
    assert "truncated" in result


def test_requests_reject_unknown_fields_and_mixed_top_level_options(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("alpha\n", encoding="utf-8")

    unknown = read_file.run({"requests": [{"path": str(path), "contians": "alpha"}]})
    mixed = read_file.run({"requests": [{"path": str(path), "contains": "alpha"}], "contains": "shared"})
    conflicting = read_file.run({
        "requests": [{"path": str(path), "contains": "alpha", "start_line": "1"}]
    })

    assert "field ที่ไม่รองรับ" in unknown
    assert "requests ใช้ร่วมกับ" in mixed
    assert "ห้ามใช้ contains ร่วมกับ start_line/end_line" in conflicting


def test_requests_reject_more_than_four_and_bad_context(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"req-{index}.txt"
        path.write_text("alpha\n", encoding="utf-8")
        paths.append(path)

    too_many = read_file.run({"requests": [{"path": str(path)} for path in paths]})
    bad_context = read_file.run({
        "requests": [{"path": str(paths[0]), "contains": "alpha", "context_lines": 99}]
    })

    assert "ไม่เกิน 4" in too_many
    assert "0-20" in bad_context


def test_contains_still_scans_entire_file_for_secret_before_returning(tmp_path):
    path = tmp_path / "secret-late.txt"
    path.write_text(
        "TARGET appears early\n"
        + "ordinary\n" * 20
        + "api_key = sk-abcdefghijklmnopqrstuvwxyz123456\n",
        encoding="utf-8",
    )

    result = read_file.run({"path": str(path), "contains": "TARGET", "context_lines": 0})

    assert result.startswith("[blocked]")
    assert "TARGET appears early" not in result


def test_contains_filters_real_docx_document(tmp_path):
    from docx import Document

    path = tmp_path / "manual.docx"
    doc = Document()
    doc.add_paragraph("Introduction")
    doc.add_paragraph("Hydraulic pressure normal")
    doc.add_paragraph("Closing note")
    doc.save(path)

    result = read_file.run({
        "requests": [{"path": str(path), "contains": "HYDRAULIC", "context_lines": 1}]
    })

    assert "Hydraulic pressure normal" in result
    assert "Introduction" in result
    assert "Closing note" in result


def test_read_file_schemas_expose_requests_contains_and_context_lines():
    manual_props = read_file.SCHEMA["function"]["parameters"]["properties"]
    lc_props = read_file.read_file.args_schema.model_json_schema()["properties"]

    for name in ("requests", "contains", "context_lines", "paths", "path", "start_line", "end_line"):
        assert name in manual_props
    for name in ("requests", "contains", "context_lines"):
        assert name in lc_props


def test_langchain_tool_invocation_accepts_independent_requests(tmp_path):
    from helpers import turn_guard

    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("noise\nSYNC_KB here\n", encoding="utf-8")
    second.write_text("a\nb\nc\nd\n", encoding="utf-8")
    turn_guard.reset("read-file-independent-requests")

    result = read_file.read_file.invoke({
        "requests": [
            {"path": str(first), "contains": "sync_kb", "context_lines": 0},
            {"path": str(second), "start_line": "2", "end_line": "3"},
        ]
    })

    assert "SYNC_KB here" in result
    assert "[บรรทัด 2-3 จาก 4]" in result
    assert "b\nc" in result


def test_contains_preserves_keyword_inside_one_very_long_line(tmp_path):
    path = tmp_path / "minified.json"
    path.write_text("x" * 6000 + " UNIQUE_NEEDLE " + "y" * 6000, encoding="utf-8")

    result = read_file.run({"path": str(path), "contains": "unique_needle", "context_lines": 0})

    assert "UNIQUE_NEEDLE" in result
    assert len(result) <= 5_000


def test_contains_rejects_unbounded_filter_string(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("hello\n", encoding="utf-8")

    result = read_file.run({"path": str(path), "contains": "x" * 513})
    request_result = read_file.run({"requests": [{"path": str(path), "contains": "x" * 513}]})

    assert "512" in result
    assert "512" in request_result
