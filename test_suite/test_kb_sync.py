"""End-to-end tests for recursive rag_document/ -> SQLite FTS sync."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    import config

    knowledge = tmp_path / "knowledge"
    raw = knowledge  # Compatibility fixture name; sources now live at knowledge/ root.
    md = knowledge / "md"
    knowledge.mkdir(parents=True)
    md.mkdir()
    db = tmp_path / "file_index.sqlite3"
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", raw)
    monkeypatch.setattr(config, "KNOWLEDGE_MD_DIR", md)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "MD_DIR", md)
    monkeypatch.setattr(config, "FILE_INDEX_DB", db)
    monkeypatch.setattr(config, "INDEX_ROOTS", [md])
    monkeypatch.setattr(config, "KB_CHUNK_TARGET_CHARS", 240)
    monkeypatch.setattr(config, "KB_CHUNK_OVERLAP_CHARS", 30)
    return raw, md, db




def test_production_knowledge_root_scans_nested_documents(tmp_path, monkeypatch):
    import config
    import kb_sync

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "manual.txt").write_text("root knowledge source", encoding="utf-8")
    (knowledge / ".sync_kb_manifest.json").write_text("{}", encoding="utf-8")
    legacy_raw = knowledge / "RAW"
    legacy_raw.mkdir()
    (legacy_raw / "old.txt").write_text("legacy copy", encoding="utf-8")
    legacy_md = knowledge / "md"
    legacy_md.mkdir()
    (legacy_md / "generated.md").write_text("legacy markdown", encoding="utf-8")

    monkeypatch.setattr(config, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_SOURCE_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_RAW_DIR", knowledge)
    monkeypatch.setattr(config, "RAW_DIR", knowledge)
    monkeypatch.setattr(config, "KNOWLEDGE_MD_DIR", legacy_md)

    rows = list(kb_sync._iter_raw_files())
    assert {(key, path.name) for key, path in rows} == {
        ("manual.txt", "manual.txt"),
        ("RAW/old.txt", "old.txt"),
        ("md/generated.md", "generated.md"),
    }
def _counts(db: Path) -> tuple[int, int]:
    with sqlite3.connect(db) as conn:
        return (
            conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM kb_chunks_fts").fetchone()[0],
        )


def _search(query: str) -> str:
    from tools import rag_search
    return rag_search.run({"query": query})


def test_thai_newmm_chunk_boundaries_match_word_boundaries():
    from helpers import thai_word_break

    if not thai_word_break.available():
        pytest.skip("PyThaiNLP/newmm is optional on this runtime")

    text = "ระบบน้ำมันเครื่องยนต์ทำงานปกติแรงดันเชื้อเพลิงตรวจสอบ"
    tokens = thai_word_break.query_words(text)
    boundaries = set()
    cursor = 0
    for token in tokens:
        cursor += len(token)
        boundaries.add(cursor)

    pieces = thai_word_break.bounded_pieces(text, 14)
    assert "".join(pieces) == text
    cursor = 0
    for piece in pieces[:-1]:
        cursor += len(piece)
        assert cursor in boundaries
    assert max(map(len, pieces)) <= 14


def test_thai_query_terms_use_newmm_segmentation():
    from helpers import thai_word_break
    from tools import rag_search

    if not thai_word_break.available():
        pytest.skip("PyThaiNLP/newmm is optional on this runtime")

    assert rag_search._query_terms("ระบบน้ำมันเครื่องยนต์") == [
        "ระบบ", "น้ำมัน", "เครื่องยนต์",
    ]


def test_thai_word_break_falls_back_without_pythainlp(monkeypatch):
    from helpers import thai_word_break

    monkeypatch.setattr(thai_word_break, "_word_tokenizer", lambda: None)
    text = "ระบบน้ำมันเครื่องยนต์"
    assert thai_word_break.query_words(text) == [text]
    assert "".join(thai_word_break.bounded_pieces(text, 5)) == text
    assert all(len(piece) <= 5 for piece in thai_word_break.bounded_pieces(text, 5))


def test_kb_thai_chunks_and_overlap_end_on_newmm_boundaries(isolated_kb, monkeypatch):
    from helpers import thai_word_break

    if not thai_word_break.available():
        pytest.skip("PyThaiNLP/newmm is optional on this runtime")

    raw, _md, db = isolated_kb
    import config
    import kb_sync

    monkeypatch.setattr(config, "KB_CHUNK_TARGET_CHARS", 24)
    monkeypatch.setattr(config, "KB_CHUNK_OVERLAP_CHARS", 8)
    source = raw / "thai-manual.txt"
    text = "ระบบน้ำมันเครื่องยนต์ทำงานปกติแรงดันเชื้อเพลิงตรวจสอบอุณหภูมิ"
    source.write_text(text, encoding="utf-8")

    assert kb_sync.sync_kb()["inserted"] == 1
    normalized = kb_sync.normalize_extracted_text(text)
    tokens = thai_word_break.query_words(normalized)

    starts = {0}
    ends = set()
    cursor = 0
    for token in tokens:
        cursor += len(token)
        ends.add(cursor)
        starts.add(cursor)
    starts.discard(len(normalized))

    with sqlite3.connect(db) as conn:
        chunks = [
            row[0] for row in conn.execute(
                "SELECT content FROM kb_chunks_fts WHERE path = ? ORDER BY chunk_index",
                (str(source.resolve()),),
            )
        ]
    assert len(chunks) > 1
    for chunk in chunks:
        positions = []
        offset = 0
        while True:
            found = normalized.find(chunk, offset)
            if found < 0:
                break
            positions.append(found)
            offset = found + 1
        assert positions, chunk
        assert any(
            start in starts and start + len(chunk) in ends
            for start in positions
        ), chunk


def test_sync_kb_indexes_tail_content_without_generated_markdown(isolated_kb):
    raw, md, db = isolated_kb
    source = raw / "manual.txt"
    source.write_text(
        ("alpha beta gamma delta\n" * 300)
        + "hydraulic accumulator emergency pressure\n",
        encoding="utf-8",
    )

    import kb_sync

    first = kb_sync.sync_kb()
    assert first["raw_scanned"] == 1
    assert first["inserted"] == 1
    assert first["chunks"] > 1
    assert first["md_files"] == 0
    assert list(md.iterdir()) == []
    assert _counts(db)[0] == 1

    result = _search("hydraulic accumulator")
    assert str(source.resolve()) in result
    assert "hydraulic" in result.casefold()

    unchanged = kb_sync.sync_kb()
    assert unchanged["unchanged"] == 1
    assert unchanged["inserted"] == 0
    assert unchanged["updated"] == 0
    assert _counts(db) == (first["documents"], first["chunks"])

    source.write_text("replacement body with avionics bus data\n", encoding="utf-8")
    changed = kb_sync.sync_kb()
    assert changed["updated"] == 1
    assert "avionics" in _search("avionics").casefold()

    source.unlink()
    removed = kb_sync.sync_kb()
    assert removed["removed"] == 1
    assert _counts(db) == (0, 0)


def test_sync_kb_progress_reports_phase_estimate_and_completion(isolated_kb):
    raw, _md, _db = isolated_kb
    (raw / "manual.txt").write_text("camera service life\n", encoding="utf-8")

    import kb_sync

    events = []
    stats = kb_sync.sync_kb(
        on_progress=lambda event, path, snapshot: events.append((event, path, snapshot))
    )

    names = [event for event, _path, _snapshot in events]
    assert names[:2] == ["prepare", "prepared"]
    assert "file" in names and "result" in names and "cleanup" in names
    assert names[-1] == "done"
    prepared = next(snapshot for event, _path, snapshot in events if event == "prepared")
    completed = next(snapshot for event, _path, snapshot in events if event == "result")
    assert prepared["total_files"] == 1
    assert completed["completed_files"] == 1
    assert completed["progress_percent"] == 99.0
    assert stats["progress_percent"] == 100.0


def test_sync_kb_indexes_image_ocr_text(isolated_kb, monkeypatch):
    raw, _md, _db = isolated_kb
    image = raw / "receipt.png"
    image.write_bytes(b"fake-image-bytes")

    import kb_sync

    monkeypatch.setattr(
        kb_sync,
        "read_layout",
        lambda _path: [
            {"text": "Invoice 1234"},
            {"text": "Total amount 500 baht"},
        ],
    )
    stats = kb_sync.sync_kb()
    assert stats["inserted"] == 1
    result = _search("Invoice 1234")
    assert str(image.resolve()) in result
    assert "Image, image" in result


def test_ocr_alternative_is_searchable_without_rewriting_primary_content(isolated_kb, monkeypatch):
    raw, _md, db = isolated_kb
    image = raw / "fuel.png"
    image.write_bytes(b"fake-image-bytes")

    import kb_sync

    monkeypatch.setattr(
        kb_sync,
        "read_layout",
        lambda _path: [{
            "text": "ระดับน้ามันต่ำ",
            "confidence": 0.60,
            "candidates": ["ระดับน้ามันต่ำ", "ระดับน้ำมันต่ำ", "ระดับนำ้มันต่ำ"],
        }],
    )
    stats = kb_sync.sync_kb()
    assert stats["inserted"] == 1

    with sqlite3.connect(db) as conn:
        content, search_text = conn.execute(
            "SELECT content, search_text FROM kb_chunks_fts WHERE path = ?",
            (str(image.resolve()),),
        ).fetchone()
    assert "ระดับน้ามันต่ำ" in content
    assert "ระดับน้ำมันต่ำ" not in content
    assert "ระดับน้ำมันต่ำ" in search_text

    result = _search("น้ำมัน")
    assert str(image.resolve()) in result
    assert "ระดับน้ามันต่ำ" in result
    assert "OCR fuzzy" not in result


def test_high_confidence_second_candidate_is_searchable_without_rewriting_primary(isolated_kb, monkeypatch):
    raw, _md, db = isolated_kb
    image = raw / "short-thai.png"
    image.write_bytes(b"fake-image-bytes")

    import kb_sync

    monkeypatch.setattr(
        kb_sync,
        "read_layout",
        lambda _path: [{
            "text": "ระดับน้าต่ำ",
            "confidence": 1.0,
            "candidates": ["ระดับน้าต่ำ", "ระดับน้ำต่ำ", "ระดับนำ้ต่ำ"],
        }],
    )
    assert kb_sync.sync_kb()["inserted"] == 1

    with sqlite3.connect(db) as conn:
        content, search_text = conn.execute(
            "SELECT content, search_text FROM kb_chunks_fts WHERE path = ?",
            (str(image.resolve()),),
        ).fetchone()
    assert "ระดับน้าต่ำ" in content
    assert "ระดับน้ำต่ำ" not in content
    assert "ระดับน้ำต่ำ" in search_text
    assert "ระดับนำ้ต่ำ" not in search_text

    result = _search("น้ำ")
    assert str(image.resolve()) in result
    assert "ระดับน้าต่ำ" in result
    assert "OCR fuzzy" not in result


def test_ocr_zero_hit_fuzzy_recovers_one_character_error(isolated_kb, monkeypatch):
    raw, _md, db = isolated_kb
    image = raw / "fuel-high-confidence.png"
    image.write_bytes(b"fake-image-bytes")

    import kb_sync

    monkeypatch.setattr(
        kb_sync,
        "read_layout",
        lambda _path: [{
            "text": "ระดับน้ามันต่ำ",
            "confidence": 0.99,
            "candidates": ["ระดับน้ามันต่ำ"],
        }],
    )
    stats = kb_sync.sync_kb()
    assert stats["inserted"] == 1

    with sqlite3.connect(db) as conn:
        content, search_text = conn.execute(
            "SELECT content, search_text FROM kb_chunks_fts WHERE path = ?",
            (str(image.resolve()),),
        ).fetchone()
    assert content == search_text
    assert "น้ำมัน" not in search_text

    result = _search("น้ำมัน")
    assert str(image.resolve()) in result
    assert "ระดับน้ามันต่ำ" in result
    assert "OCR fuzzy" in result


def test_deterministic_search_result_shows_file_location_and_sample():
    from tools import rag_search

    result = rag_search._format_chunk_rows([
        ("/tmp/manual.pdf", "manual.pdf", "page 4", "กล้องมีอายุการใช้งานห้าปี", 1234, "PDF"),
    ], "อายุการใช้งานกล้อง")

    assert "ไฟล์: /tmp/manual.pdf" in result
    assert "หน้า/ตำแหน่ง: page 4" in result
    assert "[source: /tmp/manual.pdf]" in result
    assert "ตัวอย่างเนื้อหา: กล้องมีอายุการใช้งานห้าปี" in result


def test_text_sync_persists_line_location_on_each_chunk(isolated_kb):
    raw, _md, db = isolated_kb
    source = raw / "notes.txt"
    source.write_text("หัวข้อ\nบรรทัดที่มีคำค้น\nรายละเอียดเพิ่มเติม\n", encoding="utf-8")

    import kb_sync

    stats = kb_sync.sync_kb()
    assert stats["inserted"] == 1
    with sqlite3.connect(db) as conn:
        location, content = conn.execute(
            "SELECT location, content FROM kb_chunks_fts WHERE path = ?",
            (str(source.resolve()),),
        ).fetchone()

    assert location == "lines 1-3"
    assert "คำค้น" in content
    result = _search("คำค้น")
    assert "หน้า/ตำแหน่ง: lines 1-3" in result


def test_ocr_fuzzy_result_label_has_readable_spacing():
    from tools import rag_search

    rows = [("/tmp/x.png", "x.png", "image-target", "ระดับน้ามันต่ำ", 10, "Image")]
    result = rag_search._format_chunk_rows(rows, "น้ำมัน", fuzzy=True)
    assert "ช่วงข้อความ OCR ที่ใกล้เคียงกับ 'น้ำมัน'" in result
    assert "OCRกับ" not in result


def test_ocr_fuzzy_prefers_closer_later_window_over_first_acceptable_match():
    from tools import rag_search

    assert rag_search._fuzzy_term_match("abcdefgh", "abcxefgy ... abcdefgx") == (1, 13)


def test_ocr_search_alternative_secret_fails_closed(isolated_kb, monkeypatch):
    raw, _md, db = isolated_kb
    image = raw / "candidate-alt.png"
    image.write_bytes(b"fake-image-bytes")

    import kb_sync

    secret = "sk-" + ("A" * 40)
    monkeypatch.setattr(
        kb_sync,
        "read_layout",
        lambda _path: [{
            "text": "ordinary safe primary text",
            "confidence": 0.50,
            "candidates": ["ordinary safe primary text", secret],
        }],
    )
    stats = kb_sync.sync_kb()
    assert stats["sync_errors"] == 1
    assert stats["documents"] == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM kb_chunks_fts").fetchone()[0] == 0


def test_ocr_fuzzy_rejects_distant_thai_word(isolated_kb, monkeypatch):
    raw, _md, _db = isolated_kb
    image = raw / "sugar.png"
    image.write_bytes(b"fake-image-bytes")

    import kb_sync

    monkeypatch.setattr(
        kb_sync,
        "read_layout",
        lambda _path: [{
            "text": "ระดับน้ำตาลต่ำ",
            "confidence": 0.99,
            "candidates": ["ระดับน้ำตาลต่ำ"],
        }],
    )
    assert kb_sync.sync_kb()["inserted"] == 1
    result = _search("น้ำมัน")
    assert str(image.resolve()) not in result
    assert "OCR fuzzy" not in result


def test_ocr_fuzzy_seed_quotas_find_target_after_many_common_seed_rows(isolated_kb):
    raw, _md, db = isolated_kb
    image = raw / "bulk-ocr.png"
    image.write_bytes(b"x")

    import kb_sync

    conn = kb_sync._connect_kb()
    try:
        stat = image.stat()
        conn.execute(
            "INSERT INTO kb_documents("
            "path, source_key, name, kind, suffix, size_bytes, mtime_ns, sha256, chunk_count, pipeline_version"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(image.resolve()), "raw/bulk-ocr.png", image.name, "Image", ".png",
                stat.st_size, stat.st_mtime_ns, "synthetic", 201, kb_sync._pipeline_signature(),
            ),
        )
        rows = [
            (
                str(image.resolve()), index, image.name, f"image-{index}",
                f"รายการระดับน้ำตาลปกติ หมายเลข {index}",
                f"รายการระดับน้ำตาลปกติ หมายเลข {index}",
            )
            for index in range(200)
        ]
        rows.append(
            (
                str(image.resolve()), 200, image.name, "image-target",
                "ระดับน้ามันต่ำ", "ระดับน้ามันต่ำ",
            )
        )
        conn.executemany(
            "INSERT INTO kb_chunks_fts(path, chunk_index, name, location, content, search_text) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    result = _search("น้ำมัน")
    assert str(image.resolve()) in result
    assert "ระดับน้ามันต่ำ" in result
    assert "OCR fuzzy" in result


def test_connect_kb_preserves_equivalent_lowercase_v3_schema(isolated_kb):
    _raw, _md, db = isolated_kb
    with sqlite3.connect(db) as conn:
        conn.execute(
            "create virtual table kb_chunks_fts using fts5("
            "path unindexed, chunk_index unindexed, name, location unindexed, "
            "content    unindexed, search_text, tokenize='trigram')"
        )
        conn.execute(
            "insert into kb_chunks_fts(path, chunk_index, name, location, content, search_text) "
            "values (?, ?, ?, ?, ?, ?)",
            ("/tmp/lowercase.png", 0, "lowercase.png", "image", "PRIMARY", "SEARCH"),
        )
        conn.commit()

    import kb_sync

    conn = kb_sync._connect_kb()
    try:
        assert conn.execute("SELECT COUNT(*) FROM kb_chunks_fts").fetchone()[0] == 1
        row = conn.execute(
            "SELECT content, search_text FROM kb_chunks_fts WHERE path=?",
            ("/tmp/lowercase.png",),
        ).fetchone()
        assert row == ("PRIMARY", "SEARCH")
    finally:
        conn.close()


def test_schema_classification_database_error_returns_sync_recovery_hint(isolated_kb, monkeypatch):
    _raw, _md, db = isolated_kb
    db.write_bytes(b"placeholder")

    from tools import rag_search

    class _BrokenConn:
        def execute(self, *_args, **_kwargs):
            raise sqlite3.DatabaseError("database disk image is malformed")

        def close(self):
            return None

    monkeypatch.setattr(rag_search.sqlite3, "connect", lambda *_args, **_kwargs: _BrokenConn())

    result = rag_search.run({"query": "สอง คำ"})
    assert result.startswith("[error] Knowledge Base index เสียหายหรืออ่านไม่ได้")
    assert "rag_manage(action='sync')" in result
    assert "database disk image is malformed" in result


def test_old_kb_schema_fails_closed_until_sync(isolated_kb):
    raw, _md, db = isolated_kb
    source = raw / "manual.txt"
    source.write_text("current raw content", encoding="utf-8")

    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE kb_documents ("
            "path TEXT PRIMARY KEY, source_key TEXT NOT NULL UNIQUE, name TEXT NOT NULL, "
            "kind TEXT NOT NULL, suffix TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
            "mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL, chunk_count INTEGER NOT NULL, "
            "pipeline_version INTEGER NOT NULL)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE kb_chunks_fts USING fts5("
            "path UNINDEXED, chunk_index UNINDEXED, name, location UNINDEXED, content, "
            "tokenize='trigram')"
        )
        # Deliberately add a legacy whole-machine table too: rag_search must not
        # leak through to it while a stale KB schema is present.
        conn.execute("CREATE TABLE meta(path TEXT PRIMARY KEY, name TEXT, size_bytes INTEGER, kind TEXT)")
        conn.execute(
            "CREATE VIRTUAL TABLE files_fts USING fts5(path UNINDEXED, name, content, tokenize='trigram')"
        )
        conn.execute(
            "INSERT INTO files_fts(path, name, content) VALUES (?, ?, ?)",
            (str(source.resolve()), source.name, "legacy secret-looking result"),
        )
        conn.commit()

    result = _search("legacy")
    assert result.startswith("[error]")
    assert "rag_manage(action='sync')" in result


def test_sync_kb_migrates_old_chunk_schema_and_reindexes(isolated_kb):
    raw, _md, db = isolated_kb
    source = raw / "migrate.txt"
    source.write_text("hydraulic migration marker", encoding="utf-8")

    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE kb_documents ("
            "path TEXT PRIMARY KEY, source_key TEXT NOT NULL UNIQUE, name TEXT NOT NULL, "
            "kind TEXT NOT NULL, suffix TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
            "mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL, chunk_count INTEGER NOT NULL, "
            "pipeline_version INTEGER NOT NULL)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE kb_chunks_fts USING fts5("
            "path UNINDEXED, chunk_index UNINDEXED, name, location UNINDEXED, content, "
            "tokenize='trigram')"
        )
        stat = source.stat()
        conn.execute(
            "INSERT INTO kb_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(source.resolve()), "raw/migrate.txt", source.name, "Text", ".txt",
                stat.st_size, stat.st_mtime_ns, "old-hash", 1, 1,
            ),
        )
        conn.execute(
            "INSERT INTO kb_chunks_fts VALUES (?, 0, ?, ?, ?)",
            (str(source.resolve()), source.name, "text block 1", "stale old body"),
        )
        conn.commit()

    import kb_sync

    stats = kb_sync.sync_kb()
    assert stats["updated"] == 1
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks_fts)")}
        assert "search_text" in columns
        content, search_text = conn.execute(
            "SELECT content, search_text FROM kb_chunks_fts WHERE path = ?",
            (str(source.resolve()),),
        ).fetchone()
    assert "hydraulic migration marker" in content
    assert content == search_text
    assert str(source.resolve()) in _search("hydraulic migration marker")


def test_sync_kb_searches_the_last_of_one_thousand_document_units(isolated_kb, monkeypatch):
    raw, _md, db = isolated_kb
    source = raw / "thousand.pdf"
    source.write_bytes(b"placeholder-pdf")

    import kb_sync

    monkeypatch.setattr(kb_sync, "path_is_or_contains_secret", lambda _path: False)

    def fake_units(path: Path):
        assert path == source.resolve()
        for index in range(1, 1001):
            tail = " hydraulic tail marker" if index == 1000 else ""
            yield index, f"page {index} ordinary searchable text{tail}"

    monkeypatch.setattr(kb_sync, "_iter_source_units", fake_units)
    stats = kb_sync.sync_kb()
    assert stats["chunks"] > 100
    assert _counts(db)[0] == 1

    result = _search("hydraulic tail marker")
    assert str(source.resolve()) in result
    assert "page" in result.casefold()


def test_late_secret_removes_previous_searchable_snapshot(isolated_kb):
    raw, _md, db = isolated_kb
    source = raw / "policy.txt"
    source.write_text("baseline searchable phrase\n", encoding="utf-8")

    import kb_sync

    first = kb_sync.sync_kb()
    assert first["inserted"] == 1

    # path_is_or_contains_secret intentionally skips content reads above 2 MB;
    # the streaming scanner must still catch this credential near EOF.
    with source.open("w", encoding="utf-8") as stream:
        stream.write("safe filler\n" * 220_000)
        stream.write("sk-" + ("a" * 40) + "\n")

    failed = kb_sync.sync_kb()
    assert failed["sync_errors"] == 1
    assert failed["skipped"] == 1
    assert failed["removed"] == 1

    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT 1 FROM kb_documents WHERE path = ?", (str(source.resolve()),)
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM kb_chunks_fts WHERE path = ?", (str(source.resolve()),)
        ).fetchone() is None
    assert str(source.resolve()) not in _search("baseline searchable phrase")


def test_chunk_policy_change_reindexes_unchanged_source(isolated_kb, monkeypatch):
    raw, _md, _db = isolated_kb
    source = raw / "policy-change.txt"
    source.write_text("alpha beta gamma delta " * 200, encoding="utf-8")

    import config
    import kb_sync

    first = kb_sync.sync_kb()
    first_chunks = first["chunks"]
    assert first["inserted"] == 1

    monkeypatch.setattr(config, "KB_CHUNK_TARGET_CHARS", 120)
    monkeypatch.setattr(config, "KB_CHUNK_OVERLAP_CHARS", 20)
    second = kb_sync.sync_kb()
    assert second["updated"] == 1
    assert second["unchanged"] == 0
    assert second["chunks"] != first_chunks


def test_mid_extraction_failure_rolls_back_to_last_good_snapshot(isolated_kb, monkeypatch):
    raw, _md, db = isolated_kb
    source = raw / "rollback.txt"
    source.write_text("last good searchable snapshot\n", encoding="utf-8")

    import kb_sync

    first = kb_sync.sync_kb()
    assert first["inserted"] == 1
    with sqlite3.connect(db) as conn:
        before = conn.execute(
            "SELECT content FROM kb_chunks_fts WHERE path = ? ORDER BY chunk_index",
            (str(source.resolve()),),
        ).fetchall()

    source.write_text("changed body that will fail extraction\n", encoding="utf-8")

    def broken_units(_path):
        yield 1, "new partial text before failure " * 20
        raise RuntimeError("synthetic extractor failure")

    monkeypatch.setattr(kb_sync, "_iter_source_units", broken_units)
    failed = kb_sync.sync_kb()
    assert failed["sync_errors"] == 1
    assert failed["removed"] == 0

    with sqlite3.connect(db) as conn:
        after = conn.execute(
            "SELECT content FROM kb_chunks_fts WHERE path = ? ORDER BY chunk_index",
            (str(source.resolve()),),
        ).fetchall()
    assert after == before
    assert str(source.resolve()) in _search("last good searchable snapshot")
    assert str(source.resolve()) not in _search("new partial text before failure")


def test_real_pdf_docx_xlsx_tail_content_is_searchable(isolated_kb):
    raw, _md, db = isolated_kb
    nested = raw / "nested"
    nested.mkdir()

    import fitz
    pdf_path = nested / "real_manual.pdf"
    pdf = fitz.open()
    for page_number in range(1, 13):
        page = pdf.new_page()
        text = f"ordinary manual page {page_number}"
        if page_number == 12:
            text += " hydraulic accumulator unique"
        page.insert_text((72, 72), text)
    pdf.save(pdf_path)
    pdf.close()

    from docx import Document
    docx_path = nested / "real_word.docx"
    doc = Document()
    for index in range(1, 101):
        text = f"ordinary paragraph {index}"
        if index == 100:
            text += " fuel control unit unique"
        doc.add_paragraph(text)
    doc.save(docx_path)

    from openpyxl import Workbook
    xlsx_path = nested / "real_sheet.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Parts"
    for index in range(1, 501):
        note = "ordinary row"
        if index == 500:
            note = "emergency oxygen regulator unique"
        sheet.append([index, note])
    workbook.save(xlsx_path)

    import kb_sync
    stats = kb_sync.sync_kb()
    assert stats["documents"] == 3
    assert str(pdf_path.resolve()) in _search("hydraulic accumulator unique")
    assert str(docx_path.resolve()) in _search("fuel control unit unique")
    assert str(xlsx_path.resolve()) in _search("emergency oxygen regulator unique")
    with sqlite3.connect(db) as conn:
        stored_paths = {
            row[0] for row in conn.execute("SELECT path FROM kb_documents")
        }
    assert stored_paths == {
        str(pdf_path.resolve()),
        str(docx_path.resolve()),
        str(xlsx_path.resolve()),
    }


def test_scanned_pdf_pages_use_ocr_during_kb_sync(isolated_kb, monkeypatch):
    raw, _md, _db = isolated_kb

    import fitz
    pdf_path = raw / "scan.pdf"
    pdf = fitz.open()
    for _ in range(3):
        pdf.new_page()
    pdf.save(pdf_path)
    pdf.close()

    import kb_sync
    calls = []

    def fake_page_ocr(_page, page_number):
        calls.append(page_number)
        text = "scanned hydraulic marker" if page_number == 3 else "ordinary scanned page"
        layout = [{"text": text, "confidence": 1.0, "candidates": [text]}]
        return layout, "data:image/png;base64,AA=="

    # Mock the OCR boundary rather than Apple Vision itself. Hands' outer
    # sandbox intentionally blocks deletion of kb_sync's temporary raster,
    # while the production Lite runtime is not nested inside that sandbox.
    monkeypatch.setattr(kb_sync, "_ocr_pdf_page", fake_page_ocr)
    stats = kb_sync.sync_kb()
    assert stats["documents"] == 1
    assert calls == [1, 2, 3]
    result = _search("scanned hydraulic marker")
    assert str(pdf_path.resolve()) in result
    assert "page" in result.casefold()
