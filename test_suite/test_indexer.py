"""No model needed — indexer.py + tools/rag_search.py against a throwaway corpus.

Uses env var overrides (LITE_INDEX_ROOTS / LITE_DATA_DIR) so this never
touches Lite's normal index; state remains under workspect as required.
"""
import os
import errno
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

CORPUS = Path("/tmp/_lite_test_corpus")
DATA_DIR = Path(__file__).resolve().parents[1] / "workspect" / "_test_index_data"


@pytest.fixture(scope="module", autouse=True)
def corpus_and_index():
    shutil.rmtree(CORPUS, ignore_errors=True)
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    (CORPUS / "subdir").mkdir(parents=True)
    (CORPUS / "รายงานการเงิน.md").write_text(
        "# รายงานการเงินประจำเดือน\nกำไรเติบโตขึ้น 12% จากเดือนก่อน\nน้ำสะอาด\nคำสั่งระบบ\n", encoding="utf-8"
    )
    (CORPUS / "subdir" / "notes.txt").write_text(
        "Meeting notes about the agent lite project.\nUses SQLite FTS5 for search.\nContains infor mation split by PDF layout.\n",
        encoding="utf-8",
    )
    (CORPUS / "unrelated.txt").write_text("nothing relevant here\n", encoding="utf-8")
    (CORPUS / "large-not-stored.txt").write_text("large-file-keyword\n" + "x" * (2 * 1024 * 1024), encoding="utf-8")
    (CORPUS / "คู่มือ.pdf").write_bytes(b"%PDF-1.7\x00binary document body")

    # A real, valid PDF (not the fake-bytes one above) whose extracted text
    # (PyMuPDF's page.get_text) has the ragged whitespace/newlines real PDF
    # layout extraction commonly produces -- verifies normalization, not
    # just that extraction runs.
    import fitz
    real_pdf = CORPUS / "real_report.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly   revenue")
    page.insert_text((72, 100), "grew by 12 percent")
    doc.save(real_pdf)
    doc.close()
    # Content says "ภาพ" only -- never "รูป" -- for testing _QUERY_ALIASES:
    # a query for "รูป" must still find this via the รูป<->ภาพ synonym pair.
    (CORPUS / "photo_notes.txt").write_text(
        "บันทึกเกี่ยวกับภาพถ่ายทริปเชียงใหม่\n", encoding="utf-8"
    )
    (CORPUS / "passwords.txt").write_text("do not index this secret", encoding="utf-8")
    (CORPUS / ".env").write_text("API_KEY=do-not-index", encoding="utf-8")
    (CORPUS / "agent.py").write_text("print('source code must not be indexed')\n", encoding="utf-8")
    (CORPUS / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (CORPUS / "__pycache__").mkdir()
    (CORPUS / "__pycache__" / "included_when_scanning_all.txt").write_text("ordinary text", encoding="utf-8")

    os.environ["LITE_INDEX_ROOTS"] = str(CORPUS)
    os.environ["LITE_DATA_DIR"] = str(DATA_DIR)

    import config
    import importlib
    importlib.reload(config)

    import indexer
    importlib.reload(indexer)
    stats = indexer.build_index(roots=[CORPUS], verbose=False)
    yield stats
    shutil.rmtree(CORPUS, ignore_errors=True)
    shutil.rmtree(DATA_DIR, ignore_errors=True)


def _search(query: str) -> str:
    import importlib
    import config
    importlib.reload(config)
    # Filename-first lookup also scans config.KNOWLEDGE_DIR. Point it at an empty
    # directory so these fixture-only tests never match files in the developer's
    # real Knowledge Base (e.g. a document named "Information_...").
    empty_kb = DATA_DIR / "empty_kb"
    empty_kb.mkdir(parents=True, exist_ok=True)
    for name in ("KNOWLEDGE_DIR", "KNOWLEDGE_SOURCE_DIR", "KNOWLEDGE_RAW_DIR", "RAW_DIR"):
        setattr(config, name, empty_kb)
    for name in ("KNOWLEDGE_MD_DIR", "MD_DIR"):
        setattr(config, name, empty_kb / "md")
    import tools.rag_search as rag_search
    importlib.reload(rag_search)
    return rag_search.run({"query": query})


def test_index_scanned_expected_file_count(corpus_and_index):
    # Only approved document extensions are eligible; sensitive names are excluded.
    assert corpus_and_index["scanned"] == 8
    assert corpus_and_index["inserted"] == 8
    assert corpus_and_index["error"] == 0


def test_non_document_extensions_are_not_scanned_or_indexed():
    import config
    import sqlite3

    candidate_paths = {path.name for path in __import__("indexer")._iter_candidate_files(CORPUS)}
    assert "agent.py" not in candidate_paths
    assert "screenshot.png" not in candidate_paths

    conn = sqlite3.connect(str(config.FILE_INDEX_DB))
    try:
        names = {row[0] for row in conn.execute("SELECT name FROM meta")}
    finally:
        conn.close()
    assert "agent.py" not in names
    assert "screenshot.png" not in names


def test_reindex_prunes_legacy_non_document_rows(monkeypatch, tmp_path):
    """The document allowlist also cleans a database made before it existed."""
    import config
    import indexer

    db_path = tmp_path / "file_index.sqlite3"
    legacy_code = tmp_path / "legacy.py"
    legacy_code.write_text("print('old index row')\n", encoding="utf-8")
    monkeypatch.setattr(config, "FILE_INDEX_DB", db_path)

    conn = indexer._connect()
    try:
        fts_rowid = conn.execute(
            "INSERT INTO files_fts(path, name, content) VALUES (?, ?, ?)",
            (str(legacy_code), legacy_code.name, "old source excerpt"),
        ).lastrowid
        conn.execute(
            "INSERT INTO meta(path, name, mtime, fts_rowid, size_bytes, kind) VALUES (?, ?, ?, ?, ?, ?)",
            (str(legacy_code), legacy_code.name, legacy_code.stat().st_mtime, fts_rowid, legacy_code.stat().st_size, "text excerpt"),
        )
        conn.commit()
    finally:
        conn.close()

    indexer.build_index(roots=[tmp_path / "empty-root"], verbose=False)
    conn = __import__("sqlite3").connect(str(db_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0] == 0
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 8
    finally:
        conn.close()


def test_index_persists_only_compact_content_excerpt(corpus_and_index):
    import config
    import sqlite3
    conn = sqlite3.connect(str(config.FILE_INDEX_DB))
    try:
        lengths = conn.execute("SELECT length(content) FROM files_fts").fetchall()
    finally:
        conn.close()
    assert lengths
    assert all(length <= config.RAG_INDEX_CONTENT_CHARS for (length,) in lengths)


def test_pdf_is_indexed_as_filename_and_metadata_only():
    out = _search("คู่มือ.pdf")
    assert "คู่มือ.pdf" in out
    assert "PDF" in out
    assert "content not indexed" in out
    import config
    import sqlite3
    conn = sqlite3.connect(str(config.FILE_INDEX_DB))
    try:
        content, kind = conn.execute(
            "SELECT f.content, m.kind FROM files_fts f JOIN meta m ON m.path = f.path WHERE f.name = ?",
            ("คู่มือ.pdf",),
        ).fetchone()
    finally:
        conn.close()
    assert content == ""
    assert kind == "PDF"


def test_real_pdf_content_is_extracted_and_whitespace_normalized():
    """2026-07-24: PDF/Office bodies are no longer metadata-only -- verifies
    a genuinely parseable PDF gets its text extracted, whitespace-normalized
    (indexer._normalize_extracted_text), and is content-searchable."""
    import config
    import sqlite3
    conn = sqlite3.connect(str(config.FILE_INDEX_DB))
    try:
        content, kind = conn.execute(
            "SELECT f.content, m.kind FROM files_fts f JOIN meta m ON m.path = f.path WHERE f.name = ?",
            ("real_report.pdf",),
        ).fetchone()
    finally:
        conn.close()
    assert content.strip() == content  # no leading/trailing whitespace left over
    assert "  " not in content  # no collapsed-run remnants (double spaces)
    assert "\n" not in content and "\t" not in content
    assert "revenue" in content.lower() or "percent" in content.lower()
    assert kind == "PDF"

    out = _search("revenue")
    assert "real_report.pdf" in out


def test_normalize_extracted_text_removes_internal_thai_whitespace():
    """RAG content removes every gap within a Thai run, not glyph errors."""
    import indexer

    assert indexer._normalize_extracted_text("น ้ำ สม ่าเสมอ ก า") == "น้ำสม่าเสมอกา"
    assert indexer._normalize_extracted_text("ก ่ \u0e49") == "ก่้"
    assert indexer._normalize_extracted_text("คำสั่ ง") == "คำสั่ง"
    assert indexer._normalize_extracted_text("นำ้") == "น้ำ"
    assert indexer._normalize_extracted_text("คำ สั่ง ระบบ ภาษาไทย English 2567") == "คำสั่งระบบภาษาไทย English 2567"
    assert indexer._normalize_extracted_text("Latin x ่") == "Latin x ่"


def test_index_policy_migrates_existing_excerpts_without_rebuilding(monkeypatch, tmp_path):
    import config
    import indexer

    db_path = tmp_path / "file_index.sqlite3"
    monkeypatch.setattr(config, "FILE_INDEX_DB", db_path)
    conn = indexer._connect()
    try:
        conn.execute("INSERT INTO files_fts(path, name, content) VALUES (?, ?, ?)", ("/tmp/thai.pdf", "thai.pdf", "น ้ำ"))
        conn.execute("INSERT INTO files_fts(path, name, content) VALUES (?, ?, ?)", ("/tmp/latin.txt", "latin.txt", "Latin x ่"))
        conn.execute("INSERT INTO files_fts(path, name, content) VALUES (?, ?, ?)", ("/tmp/command.pdf", "command.pdf", "คำสั่ ง"))
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    finally:
        conn.close()

    conn = indexer._connect()
    try:
        content = dict(conn.execute("SELECT path, content FROM files_fts"))
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert content == {"/tmp/thai.pdf": "น้ำ", "/tmp/latin.txt": "Latin x ่", "/tmp/command.pdf": "คำสั่ง"}
    assert version == 8


def test_legacy_office_textutil_content_is_indexed(monkeypatch, tmp_path):
    import config
    import indexer

    db_path = tmp_path / "file_index.sqlite3"
    legacy_doc = tmp_path / "legacy.doc"
    legacy_odt = tmp_path / "legacy.odt"
    legacy_doc.write_bytes(b"legacy doc")
    legacy_odt.write_bytes(b"legacy odt")
    monkeypatch.setattr(config, "FILE_INDEX_DB", db_path)
    monkeypatch.setattr(indexer, "iter_document_units", lambda path: iter([f"{path.suffix} content from textutil"]))

    conn = indexer._connect()
    try:
        assert indexer._index_file(conn, legacy_doc) == "inserted"
        assert indexer._index_file(conn, legacy_odt) == "inserted"
        content = dict(conn.execute("SELECT name, content FROM files_fts"))
    finally:
        conn.close()

    assert content == {
        "legacy.doc": ".doc content from textutil",
        "legacy.odt": ".odt content from textutil",
    }


def test_open_document_spreadsheet_and_presentation_content_is_indexed(monkeypatch, tmp_path):
    import config
    import indexer

    db_path = tmp_path / "file_index.sqlite3"
    ods = tmp_path / "ledger.ods"
    odp = tmp_path / "slides.odp"
    contents = {
        ods: "<office:document-content xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\" xmlns:text=\"urn:oasis:names:tc:opendocument:xmlns:text:1.0\"><office:body><text:p>ODS indexed revenue</text:p></office:body></office:document-content>",
        odp: "<office:document-content xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\" xmlns:text=\"urn:oasis:names:tc:opendocument:xmlns:text:1.0\"><office:body><text:p>ODP indexed roadmap</text:p></office:body></office:document-content>",
    }
    for path, xml in contents.items():
        with zipfile.ZipFile(path, "w") as package:
            package.writestr("content.xml", xml)
    monkeypatch.setattr(config, "FILE_INDEX_DB", db_path)

    conn = indexer._connect()
    try:
        assert indexer._index_file(conn, ods) == "inserted"
        assert indexer._index_file(conn, odp) == "inserted"
        content = dict(conn.execute("SELECT name, content FROM files_fts"))
    finally:
        conn.close()

    assert content == {
        "ledger.ods": "ODS indexed revenue",
        "slides.odp": "ODP indexed roadmap",
    }


def test_index_policy_backfills_existing_metadata_only_odp(monkeypatch, tmp_path):
    import config
    import indexer

    db_path = tmp_path / "file_index.sqlite3"
    odp = tmp_path / "legacy.odp"
    with zipfile.ZipFile(odp, "w") as package:
        package.writestr(
            "content.xml",
            "<office:document-content xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\" xmlns:text=\"urn:oasis:names:tc:opendocument:xmlns:text:1.0\"><office:body><text:p>Backfilled ODP text</text:p></office:body></office:document-content>",
        )
    monkeypatch.setattr(config, "FILE_INDEX_DB", db_path)

    conn = indexer._connect()
    try:
        fts_rowid = conn.execute(
            "INSERT INTO files_fts(path, name, content) VALUES (?, ?, ?)",
            (str(odp), odp.name, ""),
        ).lastrowid
        conn.execute(
            "INSERT INTO meta(path, name, mtime, fts_rowid, size_bytes, kind) VALUES (?, ?, ?, ?, ?, ?)",
            (str(odp), odp.name, odp.stat().st_mtime, fts_rowid, odp.stat().st_size, "OpenDocument Presentation"),
        )
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
    finally:
        conn.close()

    conn = indexer._connect()
    try:
        content = conn.execute("SELECT content FROM files_fts WHERE rowid = ?", (fts_rowid,)).fetchone()[0]
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert content == "Backfilled ODP text"
    assert version == 8


def test_index_policy_backfills_existing_legacy_word_rows(monkeypatch, tmp_path):
    import config
    import indexer

    db_path = tmp_path / "file_index.sqlite3"
    legacy_doc = tmp_path / "legacy.doc"
    legacy_doc.write_bytes(b"legacy doc")
    monkeypatch.setattr(config, "FILE_INDEX_DB", db_path)
    conn = indexer._connect()
    try:
        fts_rowid = conn.execute(
            "INSERT INTO files_fts(path, name, content) VALUES (?, ?, ?)",
            (str(legacy_doc), legacy_doc.name, ""),
        ).lastrowid
        conn.execute(
            "INSERT INTO meta(path, name, mtime, fts_rowid, size_bytes, kind) VALUES (?, ?, ?, ?, ?, ?)",
            (str(legacy_doc), legacy_doc.name, legacy_doc.stat().st_mtime, fts_rowid, legacy_doc.stat().st_size, "Word"),
        )
        conn.execute("PRAGMA user_version = 3")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(indexer, "_extract_document_prefix", lambda *_args: "Legacy Word content")

    conn = indexer._connect()
    try:
        content = conn.execute("SELECT content FROM files_fts WHERE rowid = ?", (fts_rowid,)).fetchone()[0]
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert content == "Legacy Word content"
    assert version == 8


def test_large_text_file_is_indexed_but_content_remains_compact():
    import config
    import sqlite3
    conn = sqlite3.connect(str(config.FILE_INDEX_DB))
    try:
        content = conn.execute(
            "SELECT content FROM files_fts WHERE name = ?", ("large-not-stored.txt",)
        ).fetchone()[0]
    finally:
        conn.close()
    assert "large-file-keyword" in content
    assert len(content) <= config.RAG_INDEX_CONTENT_CHARS


def test_search_thai_filename_keyword():
    out = _search("การเงิน")
    assert "รายงานการเงิน.md" in out


def test_search_thai_content_keyword():
    out = _search("เติบโต")
    assert "รายงานการเงิน.md" in out


def test_search_repairs_floating_thai_combining_marks_in_query():
    """Search must match an indexed normalized word even if pasted with a
    PDF-style space before its Thai combining mark."""
    out = _search("น ้ำสะอาด")
    assert "รายงานการเงิน.md" in out


def test_search_repairs_space_after_thai_combining_mark_in_query():
    out = _search("คำสั่ งระบบ")
    assert "รายงานการเงิน.md" in out


def test_search_repairs_thai_sara_am_tone_order_in_query():
    out = _search("นำ้สะอาด")
    assert "รายงานการเงิน.md" in out


def test_search_removes_every_internal_thai_space_in_query():
    out = _search("คำ สั่ง ระบบ")
    assert "รายงานการเงิน.md" in out


def test_search_english_content_keyword():
    out = _search("SQLite FTS5")
    assert "notes.txt" in out


def test_search_repairs_english_word_split_by_pdf_layout():
    out = _search("information")
    assert "notes.txt" in out


def test_search_synonym_finds_content_using_the_other_word():
    """2026-07-24 (user request: synonyms for the small-model search
    helper): photo_notes.txt's content says only "ภาพ", never "รูป" --
    querying "รูป" must still find it via _QUERY_ALIASES."""
    out = _search("รูป")
    assert "photo_notes.txt" in out


def test_search_is_case_insensitive_and_whitespace_tolerant():
    out_lower = _search("sqlite fts5")
    out_upper = _search("SQLITE FTS5")
    out_messy_whitespace = _search("  sqlite   \t fts5  ")
    for out in (out_lower, out_upper, out_messy_whitespace):
        assert "notes.txt" in out


def test_search_ignores_stray_punctuation_in_query():
    """A query with stray/floating punctuation around real words must not
    be passed through to FTS as literal tokens -- _query_terms extracts
    only word-like runs."""
    out = _search("...!! เติบโต ?? ///")
    assert "รายงานการเงิน.md" in out


def test_search_zero_fts_hits_falls_back_to_filename_search():
    """Zero-hit recovery stays inside the configured Knowledge Base root."""
    out = _search("PLAN.md")  # absent from the disposable Knowledge Base corpus
    assert "ขยายผลการค้นหา" not in out
    assert "/sync_kb" in out or "ไม่พบ" in out


def test_zero_fts_hits_fall_back_to_generated_knowledge_content(monkeypatch, tmp_path):
    """Fallback content search reads generated Markdown, never project source."""
    import config
    import tools.rag_search as rag_search
    from helpers import search_scope

    md = tmp_path / "knowledge" / "md"
    md.mkdir(parents=True)
    generated = md / "indexer.md"
    generated.write_text("# trigram tokenizer implementation\n", encoding="utf-8")
    monkeypatch.setattr(config, "KNOWLEDGE_MD_DIR", md)
    monkeypatch.setattr(config, "INDEX_ROOTS", [md])
    token = search_scope.set_request_query("มีไฟล์ไหนในโปรเจกต์นี้พูดถึง trigram tokenizer บ้าง")
    try:
        out = rag_search.run({"query": "trigram tokenizer"})
    finally:
        search_scope.reset_request_query(token)
    assert "indexer.md" in out
    assert str(md) in out
    assert str(CORPUS) not in out


def test_project_artifact_paths_are_not_user_search_results(monkeypatch, tmp_path):
    import config
    import tools.rag_search as rag_search

    md = tmp_path / "knowledge" / "md"
    (md / "logs" / "tool_outputs").mkdir(parents=True)
    (md / "logs" / "tool_outputs" / "old.md").write_text("old", encoding="utf-8")
    (md / "config.md").write_text("KNOWLEDGE", encoding="utf-8")
    monkeypatch.setattr(config, "KNOWLEDGE_MD_DIR", md)
    monkeypatch.setattr(config, "INDEX_ROOTS", [md])
    assert rag_search._is_project_artifact_path(str(md / "logs" / "tool_outputs" / "old.md"))
    assert rag_search._is_project_artifact_path(str(md / "workspect" / "old.md")) is False
    assert rag_search._allowed_result_path(str(md / "config.md"))
    assert not rag_search._allowed_result_path(str(md / "missing.md"))


def test_concise_filename_query_prefers_bounded_filename_fallback(monkeypatch):
    import tools.rag_search as rag_search

    monkeypatch.setattr(
        rag_search,
        "_fallback_filename_search",
        lambda query, reason="no_index": "[filename fallback]\n/Users/test/config.py",
    )
    out = rag_search.run({"query": "config"})
    assert "filename fallback" in out
    assert "config.py" in out


def test_search_no_match():
    out = _search("zzzznonexistentterm")
    assert "ไม่พบ" in out


def test_sensitive_files_are_never_indexed_or_returned():
    out = _search("secret")
    assert "passwords.txt" not in out
    assert "ไม่พบ" in out


def test_search_empty_query_error():
    out = _search("")
    assert out.startswith("[error]")


def test_no_index_uses_local_filename_fallback(monkeypatch, tmp_path):
    import config
    import tools.rag_search as rag_search
    monkeypatch.setattr(config, "FILE_INDEX_DB", tmp_path / "missing.sqlite3")
    monkeypatch.setattr(rag_search, "_fallback_filename_search", lambda query: "[local filename fallback]\n/Users/test/report.txt")
    out = rag_search.run({"query": "report"})
    assert "local filename fallback" in out
    assert "/Users/test/report.txt" in out


def test_filename_fallback_extracts_name_from_thai_sentence():
    import tools.rag_search as rag_search
    assert rag_search._filename_fallback_needle("กล้อง") == "กล้อง"
    assert rag_search._filename_fallback_needle("ค้นหาไฟล์ในเครื่องที่ชื่อมีคำว่ากล้อง") == "ค้นหาไฟล์ในเครื่องที่ชื่อมีคำว่ากล้อง"


def test_query_latency_under_one_second():
    t0 = time.time()
    _search("agent")
    assert (time.time() - t0) < 1.0


def test_incremental_reindex_skips_unchanged():
    import importlib
    import config
    importlib.reload(config)
    import indexer
    importlib.reload(indexer)
    stats = indexer.build_index(roots=[CORPUS], verbose=False)
    assert stats["inserted"] == 0
    assert stats["skipped"] == 8


def test_index_progress_streams_each_candidate_and_completion():
    import importlib
    import config
    importlib.reload(config)
    import indexer
    importlib.reload(indexer)
    events: list[tuple[str, Path | None, dict[str, int]]] = []
    stats = indexer.build_index(
        roots=[CORPUS], verbose=False,
        on_progress=lambda event, path, current: events.append((event, path, current)),
    )
    file_events = [event for event in events if event[0] == "file"]
    assert len(file_events) == stats["scanned"] == 8
    assert all(path is not None for _, path, _ in file_events)
    # Existing rows are checked for staleness before the root walk begins, so
    # an incremental run may legitimately emit one or more ``stale`` events
    # first.  The first non-stale phase must still be the requested root.
    assert next(event for event in events if event[0] != "stale")[0] == "root"
    assert events[-1][0] == "done"


def test_candidate_walk_skips_filename_too_long_errors(monkeypatch):
    """A whole-machine scan must survive deep cloud-sync paths.

    `Path.is_symlink()` was outside indexer's OSError guard, so an ENAMETOOLONG
    there killed the worker and made the browser report a failed `/index`.
    """
    import indexer

    bad_path = CORPUS / "one-drive-deep-entry.pptx"
    bad_path.write_text("not reachable", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def raises_name_too_long(path):
        if path == bad_path:
            raise OSError(errno.ENAMETOOLONG, "File name too long", str(path))
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", raises_name_too_long)

    found = list(indexer._iter_candidate_files(CORPUS))
    assert bad_path not in found
    assert CORPUS / "รายงานการเงิน.md" in found


def test_default_index_root_is_knowledge_markdown(monkeypatch):
    """`/index` must not regress to a whole-machine walk by default."""
    import importlib
    import config

    monkeypatch.delenv("LITE_INDEX_ROOTS", raising=False)
    importlib.reload(config)
    assert config.INDEX_ROOTS == [config.KNOWLEDGE_MD_DIR]

    # The module-scoped fixture intentionally directs all subsequent tests to
    # its disposable corpus; restore that process-global configuration.
    monkeypatch.setenv("LITE_INDEX_ROOTS", str(CORPUS))
    monkeypatch.setenv("LITE_DATA_DIR", str(DATA_DIR))
    importlib.reload(config)


def test_stale_prune_check_that_hangs_does_not_block_the_whole_run(monkeypatch):
    """PLAN.md §9.2 finding: build_index's stale-entry prune loop used to
    call Path.exists()/path_is_or_contains_secret() with no timeout guard --
    unlike every read in _index_file(). Reproduced live on this machine's
    real (unrelated) index hanging indefinitely on a single stale path.
    Verify the fix: a hung stale-check is abandoned within the configured
    timeout, the run still completes, and the entry is left alone (never
    guessed at) rather than deleted."""
    import importlib
    import config
    importlib.reload(config)
    # _run_with_timeout's `timeout` parameter defaults to the module-level
    # _READ_TIMEOUT_SECONDS, bound once at function-definition time -- a
    # post-import monkeypatch of that constant would not reach the already
    # -bound default, so the short timeout must be set via env + reload
    # instead, same as config's own env-driven settings.
    monkeypatch.setenv("LITE_INDEX_READ_TIMEOUT", "0.2")
    import indexer
    importlib.reload(indexer)

    real_is_stale = indexer._is_stale
    hung_path = str(CORPUS / "รายงานการเงิน.md")

    def fake_is_stale(stored_path, roots=None):
        if stored_path == hung_path:
            time.sleep(5)  # far longer than the timeout above
            return True
        return real_is_stale(stored_path, roots)

    monkeypatch.setattr(indexer, "_is_stale", fake_is_stale)

    t0 = time.time()
    stats = indexer.build_index(roots=[CORPUS], verbose=False)
    elapsed = time.time() - t0
    assert elapsed < 3, f"stale-check hang blocked the whole run for {elapsed:.1f}s"
    assert stats["error"] == 0

    conn_check = __import__("sqlite3").connect(str(config.FILE_INDEX_DB))
    try:
        still_present = conn_check.execute(
            "SELECT 1 FROM meta WHERE path = ?", (hung_path,)
        ).fetchone()
    finally:
        conn_check.close()
    assert still_present is not None  # timed-out check must not have deleted it


def test_stalled_reads_have_a_fixed_thread_ceiling():
    script = """
import threading
import indexer

gate = threading.Event()
before = threading.active_count()
for _ in range(32):
    indexer._run_with_timeout(gate.wait, timeout=0.001)
print(threading.active_count() - before)
"""
    result = __import__("subprocess").run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert int(result.stdout.strip()) <= 8


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
