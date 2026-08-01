# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Regression checks for UI-only parsing; no terminal or model is needed."""
from __future__ import annotations

import ui


def test_extract_web_refs_preserves_summary_and_deduplicates_url():
    text = """- First useful result
  [source: https://example.com/a]
- Same source later
  [source: https://example.com/a]
- Second result
  [source: https://example.org/b]
"""
    assert ui.extract_refs_from_search_result(text) == [
        ("First useful result", "https://example.com/a"),
        ("Second result", "https://example.org/b"),
    ]


def test_extract_web_refs_ignores_non_source_lines():
    assert ui.extract_refs_from_search_result("(no results found)") == []


def test_index_progress_text_includes_counts_and_current_path():
    text = ui.index_progress_text("/very/long/path/" + "nested/" * 10 + "report.md", {
        "scanned": 12, "inserted": 3, "updated": 2, "skipped": 4, "error": 0,
    })
    assert "12 scanned" in text
    assert "3 new" in text
    assert "report.md" in text


def test_print_header_handles_a_long_model_name(capsys):
    ui.print_header("mlx-community/Qwen3-VL-2B-Instruct-4bit", 6)
    output = capsys.readouterr().out
    assert "Qwen3-VL-2B-Instruct-4bit" in output
    assert "6 tools" in output
