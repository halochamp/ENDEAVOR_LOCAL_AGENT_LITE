# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""No model, no network — pure logic tests for helpers/summarize_lite.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.summarize_lite import query_terms, extractive_summary


def test_query_terms_english_splits_on_whitespace():
    terms = query_terms("hello world")
    assert "hello" in terms and "world" in terms


def test_query_terms_thai_includes_trigrams():
    terms = query_terms("อากาศ")
    assert any(len(t) == 3 for t in terms)


def test_extractive_summary_picks_matching_paragraph():
    text = "Unrelated intro paragraph here.\n\nThis paragraph mentions Qwen3.5 explicitly.\n\nAnother unrelated bit."
    out = extractive_summary(text, "Qwen3.5", max_chars=500)
    assert "Qwen3.5" in out
    assert "Unrelated intro" not in out


def test_extractive_summary_empty_text():
    assert extractive_summary("", "query", 100) == ""


def test_extractive_summary_respects_max_chars():
    text = "keyword " * 500
    out = extractive_summary(text, "keyword", max_chars=50)
    assert len(out) <= 60  # small slack for the "..." suffix


def test_extractive_summary_falls_back_when_no_term_match():
    text = "First paragraph with nothing relevant.\n\nSecond paragraph also unrelated."
    out = extractive_summary(text, "zzz_nonexistent_term", max_chars=500)
    assert out  # still returns something rather than empty


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
