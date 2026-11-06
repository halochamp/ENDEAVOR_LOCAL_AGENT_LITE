"""indexer.py — build/update the SQLite FTS5 file index used by tools/rag_search.py.

Run standalone (`python indexer.py`), not during a chat turn — indexing is
I/O-bound and can take a while on a large home directory; the agent should
never block a user turn on it.

Design (PLAN.md §3, §7 Phase 3): everything lives on SSD
(config.FILE_INDEX_DB, deliberately outside the iCloud-synced project tree
— see config.py). Incremental via per-file mtime so a re-run only touches
changed files. The normal root is generated knowledge Markdown; build_index's
stale pass also removes rows that are outside the roots used for the current
run, which prevents an old whole-machine database from leaking into the new
knowledge-only RAG.

2026-07-24 (user request: index the whole machine + extract PDF/Office
content): two changes.
  1. Fixed the previously-documented, previously-unfixed hang (PLAN.md
     §Known limitations): opening a cloud-sync placeholder file (Google
     Drive, iCloud) triggers a real download on read that can block for
     minutes with no timeout, and a whole-machine scan multiplies that risk
     far beyond the earlier ~/Desktop-scoped finding. `_run_with_timeout`
     wraps every actual file read/extraction in its own daemon thread with
     a bounded join(); a slow file is skipped, not waited on. Each call
     gets a fresh thread rather than a shared pool, so one genuinely stuck
     read never blocks subsequent files from being scanned.
  2. PDF/Office files are no longer metadata-only: their first
     RAG_INDEX_CONTENT_CHARS of extracted text is now indexed too, reusing
     tools/read_file.py's own extractor (helpers/document_extract.py) so
     there is exactly one text-extraction implementation in the project.
    Legacy .doc/.odt files use macOS textutil; .ods/.odp are read from their
    OpenDocument content.xml, while legacy .ppt uses macOS Spotlight's text
    importer when it supplies content.
"""
from __future__ import annotations
import os
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path
from collections.abc import Callable

import config
from helpers.document_extract import is_supported_document, iter_document_units, unit_separator
from helpers.sensitive_paths import contains_secret_content, is_sensitive_path, path_is_or_contains_secret
from helpers.thai_text_normalize import normalize_extracted_text as _normalize_thai_extracted_text

# Pseudo-filesystems can block indefinitely, mutate state merely by probing,
# or expose device streams. Everything else is eligible, subject to the shared
# secret policy in sensitive_paths.py.
_PSEUDO_FILESYSTEM_DIRS = {"/dev", "/proc", "/sys"}
_BINARY_MARKER = b"\x00"
_INDEX_POLICY_VERSION = 8  # v8 removes all internal whitespace from Thai runs.
_METADATA_ONLY_DOCUMENTS = {
    ".pdf": "PDF", ".doc": "Word", ".docx": "Word", ".docm": "Word", ".xls": "Excel",
    ".xlsx": "Excel", ".xlsm": "Excel", ".ppt": "PowerPoint", ".pptx": "PowerPoint", ".pptm": "PowerPoint",
    ".odt": "OpenDocument Text", ".ods": "OpenDocument Spreadsheet",
    ".odp": "OpenDocument Presentation",
}
# This is a document locator, not a general-purpose filesystem index.  Keep
# source code, application config, images, media, archives, and arbitrary
# binary files out even when they happen to decode as text.  The supported
# Office/PDF/OpenDocument extensions above and these plain-text document
# formats are the complete default scan policy.
_TEXT_DOCUMENT_SUFFIXES = frozenset({
    ".md", ".markdown", ".mdown", ".mkdn", ".txt", ".text",
    ".rtf", ".csv", ".tsv", ".tex",
})
_INDEXABLE_DOCUMENT_SUFFIXES = frozenset(_METADATA_ONLY_DOCUMENTS) | _TEXT_DOCUMENT_SUFFIXES
_READ_TIMEOUT_SECONDS = float(os.getenv("LITE_INDEX_READ_TIMEOUT", "5"))
_MAX_ABANDONED_READ_THREADS = 8
_READ_THREAD_SLOTS = threading.BoundedSemaphore(_MAX_ABANDONED_READ_THREADS)
_STALE_BATCH_SIZE = 256


def _run_with_timeout(fn, timeout: float = _READ_TIMEOUT_SECONDS):
    """Run fn() in a bounded daemon thread; return its result or None.

    A timed-out filesystem call cannot be killed safely in CPython.  Keep at
    most a small fixed number alive, then skip further reads until one exits.
    This preserves progress without letting a cloud-placeholder-heavy scan
    create one thread (and stack) per file inside the index subprocess.
    """
    if not _READ_THREAD_SLOTS.acquire(blocking=False):
        return None
    result_q: queue.Queue = queue.Queue(maxsize=1)

    def worker():
        try:
            result_q.put(("ok", fn()))
        except Exception as e:
            result_q.put(("error", e))
        finally:
            _READ_THREAD_SLOTS.release()

    t = threading.Thread(target=worker, daemon=True)
    try:
        t.start()
    except RuntimeError:
        _READ_THREAD_SLOTS.release()
        return None
    t.join(timeout)
    if t.is_alive():
        return None
    try:
        status, value = result_q.get_nowait()
    except queue.Empty:
        return None
    return value if status == "ok" else None


def _path_is_under_roots(stored_path: str, roots: list[Path]) -> bool:
    path = Path(stored_path).resolve(strict=False)
    for root in roots:
        try:
            path.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def _is_stale(stored_path: str, roots: list[Path] | None = None) -> bool:
    """True if a previously-indexed path should be pruned: gone from disk,
    outside the current document-extension policy, or now secret-shaped/secret-path.
    Run only
    through _run_with_timeout -- both checks touch the filesystem and can
    stall on the same class of mount that _index_file's reads guard against.
    """
    path = Path(stored_path)
    outside_roots = roots is not None and not _path_is_under_roots(stored_path, roots)
    return (
        outside_roots
        or
        path.suffix.lower() not in _INDEXABLE_DOCUMENT_SUFFIXES
        or not path.exists()
        or path_is_or_contains_secret(stored_path)
    )


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.FILE_INDEX_DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta ("
        " path TEXT PRIMARY KEY, name TEXT, mtime REAL, fts_rowid INTEGER, "
        "size_bytes INTEGER, kind TEXT)"
    )
    # Existing metadata tables predate the non-text document mode.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(meta)")}
    if "size_bytes" not in columns:
        conn.execute("ALTER TABLE meta ADD COLUMN size_bytes INTEGER")
    if "kind" not in columns:
        conn.execute("ALTER TABLE meta ADD COLUMN kind TEXT")
    # trigram tokenizer (not unicode61): Thai has no spaces between words, so a
    # word-based tokenizer stores e.g. "รายงานการเงินประจำเดือน" as one opaque
    # token and a query for the substring "การเงิน" never matches it. Trigram
    # indexes 3-character sequences instead, which supports substring search
    # in any script (Thai, English, mixed) at the cost of a larger index and
    # some noise on very short (<3 char) queries. Verified against a live Thai
    # test corpus 2026-07-23 — unicode61 missed both Thai filename and content
    # matches that trigram found correctly.
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5("
        " path UNINDEXED, name, content, tokenize='trigram')"
    )
    # Existing v1 databases contain complete file bodies. Trim them in place
    # once so the policy removes already-persisted text without a whole-disk
    # rescan or losing the path/mtime metadata.
    policy_version = conn.execute("PRAGMA user_version").fetchone()[0]
    if policy_version < 2:
        conn.execute(
            "UPDATE files_fts SET content = substr(content, 1, ?)",
            (config.RAG_INDEX_CONTENT_CHARS,),
        )
        policy_version = 2
    if policy_version < 3:
        # Do not clear or rescan the whole index just to correct an extraction
        # normalization bug. Update only existing excerpts that actually
        # change; meta mtimes and the rest of the FTS rows stay intact.
        for rowid, content in conn.execute("SELECT rowid, content FROM files_fts"):
            normalized = _normalize_extracted_text(content)
            if normalized != content:
                conn.execute("UPDATE files_fts SET content = ? WHERE rowid = ?", (normalized, rowid))
        policy_version = 3
    if policy_version < 4:
        # Legacy .doc/.odt rows used to be metadata-only. Re-extract only
        # those existing files so this capability does not require a whole
        # disk scan or an mtime-changing user edit.
        rows = conn.execute(
            "SELECT m.path, m.fts_rowid FROM meta AS m WHERE m.kind IN ('Word', 'OpenDocument Text')"
        ).fetchall()
        for stored_path, fts_rowid in rows:
            path = Path(stored_path)
            if path.suffix.lower() not in {".doc", ".odt"}:
                continue
            extracted = _run_with_timeout(
                lambda p=path: _extract_document_prefix(p, config.RAG_INDEX_CONTENT_CHARS)
            )
            if extracted is not None and not contains_secret_content(extracted):
                conn.execute("UPDATE files_fts SET content = ? WHERE rowid = ?", (extracted, fts_rowid))
        policy_version = 4
    if policy_version < 5:
        # v5's allowed-document policy is enforced by build_index's stale
        # pass, where removals are safely transactionally applied. Mark this
        # schema ready here so an existing v4 database can reach that pass on
        # its next index run.
        policy_version = 5
    if policy_version < 6:
        # ODS/ODP/PPT were admitted by earlier policies but retained only
        # filename/metadata. Backfill their new extractable content without
        # requiring an mtime-changing edit or a full index rebuild.
        rows = conn.execute(
            "SELECT m.path, m.fts_rowid FROM meta AS m "
            "WHERE m.kind IN ('OpenDocument Spreadsheet', 'OpenDocument Presentation', 'PowerPoint')"
        ).fetchall()
        for stored_path, fts_rowid in rows:
            path = Path(stored_path)
            if path.suffix.lower() not in {".ods", ".odp", ".ppt"}:
                continue
            extracted = _run_with_timeout(
                lambda p=path: _extract_document_prefix(p, config.RAG_INDEX_CONTENT_CHARS)
            )
            if extracted is not None and not contains_secret_content(extracted):
                conn.execute("UPDATE files_fts SET content = ? WHERE rowid = ?", (extracted, fts_rowid))
        policy_version = 6
    if policy_version < 7:
        # v7 extends the Thai layout repair from spaces before a combining
        # mark (น ้ำ) to spaces after it (คำสั่ ง). Re-normalize existing
        # compact excerpts in place without a whole-disk rescan.
        for rowid, content in conn.execute("SELECT rowid, content FROM files_fts"):
            normalized = _normalize_extracted_text(content)
            if normalized != content:
                conn.execute("UPDATE files_fts SET content = ? WHERE rowid = ?", (normalized, rowid))
        policy_version = 7
    if policy_version < 8:
        # v8 broadens the Thai repair to every whitespace run between Thai
        # characters (not only mark-adjacent gaps), improving FTS recall for
        # PDF/OCR text whose Thai words were arbitrarily split.
        for rowid, content in conn.execute("SELECT rowid, content FROM files_fts"):
            normalized = _normalize_extracted_text(content)
            if normalized != content:
                conn.execute("UPDATE files_fts SET content = ? WHERE rowid = ?", (normalized, rowid))
        policy_version = 8
    if policy_version < _INDEX_POLICY_VERSION:
        raise RuntimeError(f"unsupported index policy version: {policy_version}")
    if conn.execute("PRAGMA user_version").fetchone()[0] != _INDEX_POLICY_VERSION:
        conn.execute(f"PRAGMA user_version = {_INDEX_POLICY_VERSION}")
        conn.commit()
    return conn


def _is_binary(sample: bytes) -> bool:
    return _BINARY_MARKER in sample


def _read_sample_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        # Never load a full text file just to index it: index retention is
        # capped at 250 characters, regardless of file size.
        return f.read(max(4096, config.RAG_INDEX_CONTENT_CHARS * 4))


def _normalize_extracted_text(text: str) -> str:
    return _normalize_thai_extracted_text(text)


def _extract_document_prefix(path: Path, limit: int) -> str:
    """Collect just enough normalized text to fill `limit` chars without
    materializing a large PDF/Office document's full body first (PLAN.md
    §9.1) -- stops consuming iter_document_units() as soon as the
    accumulated, whitespace-normalized prefix is long enough."""
    sep = unit_separator(path.suffix)
    collected: list[str] = []
    for unit in iter_document_units(path):
        collected.append(unit)
        if len(_normalize_extracted_text(sep.join(collected))) >= limit:
            break
    return _normalize_extracted_text(sep.join(collected))[:limit]


def _iter_candidate_files(root: Path):
    try:
        if not root.exists():
            return
    except OSError:
        # A configured root can itself be a disconnected or otherwise
        # unstatable mount.  Treat it like an unreadable branch, rather than
        # failing the complete index build.
        return

    def _onerror(_error):
        # Permission-denied and transient filesystem entries are expected
        # during a whole-machine scan; the rest of the tree remains useful.
        return None

    def _skip_directory(current: Path, name: str) -> bool:
        """Return whether an os.walk child directory must be pruned.

        macOS can yield a directory entry whose complete path is longer than
        the filesystem limit (notably deep cloud-sync trees).  `resolve()`
        and the shared sensitive-path check both perform filesystem probes,
        so neither may be allowed to abort the entire walk.
        """
        child = current / name
        try:
            return (
                str(child.resolve(strict=False)) in _PSEUDO_FILESYSTEM_DIRS
                or is_sensitive_path(child)
            )
        except OSError:
            return True

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False, onerror=_onerror):
        current = Path(dirpath)
        dirnames[:] = [
            name for name in dirnames
            if not _skip_directory(current, name)
        ]
        for name in filenames:
            p = current / name
            # Filter by the explicit document policy before any filesystem
            # probe.  A .py/.json/image/log file is neither opened nor added
            # to the index merely because it is ordinary UTF-8 text.
            if p.suffix.lower() not in _INDEXABLE_DOCUMENT_SUFFIXES:
                continue
            try:
                # All of these probe the filesystem.  Keep them in the same
                # OSError boundary as stat(): a single ENAMETOOLONG from a
                # deeply nested OneDrive/iCloud entry must be skipped, not
                # terminate the indexer subprocess.
                if p.is_symlink() or path_is_or_contains_secret(p):
                    continue
                stat = p.stat()
                if not p.is_file():
                    continue
            except OSError:
                continue
            yield p


def _index_file(conn: sqlite3.Connection, path: Path) -> str:
    """Returns 'skipped' | 'updated' | 'inserted' | 'error'."""
    try:
        st = path.stat()
    except OSError:
        return "error"
    mtime = st.st_mtime
    kind = _METADATA_ONLY_DOCUMENTS.get(path.suffix.lower())

    row = conn.execute("SELECT mtime, fts_rowid FROM meta WHERE path = ?", (str(path),)).fetchone()
    if row and row[0] == mtime:
        return "skipped"

    if kind and is_supported_document(path):
        # Extraction (esp. PDF page layout) can itself run long on a large
        # document, same risk class as a stalled read below -- same guard.
        # _extract_document_prefix stops once it has enough normalized text,
        # so a multi-hundred-page PDF is never fully materialized just to
        # keep its first 250 characters.
        extracted = _run_with_timeout(lambda: _extract_document_prefix(path, config.RAG_INDEX_CONTENT_CHARS))
        if extracted is None:
            content = ""  # falls back to metadata-only for this file, not fatal
        else:
            content = extracted
            if contains_secret_content(content):
                content = ""
    elif kind:
        # Defensive fallback for an extension classified as a document but
        # lacking an extractor. Current allowlisted Office/OpenDocument types
        # all have an extraction path above.
        content = ""
    else:
        sample = _run_with_timeout(lambda: _read_sample_bytes(path))
        if sample is None:
            return "error"  # stalled read (e.g. cloud-sync placeholder) -- skip, don't hang
        if _is_binary(sample):
            return "skipped"
        content = _normalize_extracted_text(
            sample.decode("utf-8", errors="replace")
        )[:config.RAG_INDEX_CONTENT_CHARS]
        if contains_secret_content(content):
            return "skipped"
        # RAG is a locator: retain only a compact leading excerpt, never a
        # full document body, regardless of the source file's size.
        kind = "text excerpt"
    name = path.name
    if row:
        _, fts_rowid = row
        conn.execute(
            "UPDATE files_fts SET name = ?, content = ? WHERE rowid = ?",
            (name, content, fts_rowid),
        )
        conn.execute(
            "UPDATE meta SET name = ?, mtime = ?, size_bytes = ?, kind = ? WHERE path = ?",
            (name, mtime, st.st_size, kind, str(path)),
        )
        return "updated"
    else:
        cur = conn.execute(
            "INSERT INTO files_fts (path, name, content) VALUES (?, ?, ?)",
            (str(path), name, content),
        )
        conn.execute(
            "INSERT INTO meta (path, name, mtime, fts_rowid, size_bytes, kind) VALUES (?, ?, ?, ?, ?, ?)",
            (str(path), name, mtime, cur.lastrowid, st.st_size, kind),
        )
        return "inserted"


IndexProgress = Callable[[str, Path | None, dict[str, int]], None]


def build_index(
    roots: list[Path] | None = None,
    verbose: bool = True,
    on_progress: IndexProgress | None = None,
) -> dict[str, int]:
    """Build/update the index and optionally report each eligible file live.

    ``on_progress`` is deliberately UI-agnostic so the CLI can stream the
    current path without routing the long-running job through the LLM.
    Callback failures are ignored: a rendering problem must not corrupt the
    index build itself.
    """
    roots = roots or config.INDEX_ROOTS
    conn = _connect()
    stats = {"scanned": 0, "inserted": 0, "updated": 0, "skipped": 0, "error": 0}
    t0 = time.time()

    def report(event: str, path: Path | None = None) -> None:
        if on_progress is None:
            return
        try:
            on_progress(event, path, dict(stats))
        except Exception:
            # Progress display is optional and must never interrupt indexing.
            pass

    try:
        # Remove data indexed under an older policy before serving new results.
        # PLAN.md §9.2 finding: Path.exists()/path_is_or_contains_secret() can
        # stall exactly like the per-file reads below do (a cloud-sync
        # placeholder path recorded in a previous run, now on a mount that
        # blocks on access) -- reproduced live on this machine's real index
        # (44k entries from an earlier whole-disk scan) hanging indefinitely
        # here with no timeout, unlike every read in _index_file(). Same
        # guard, applied here too: on timeout, leave the entry as-is rather
        # than guess -- the next run gets another chance to prune it.
        stats["stale_checked"] = 0
        stats["stale_total"] = conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
        last_path = ""
        while True:
            stale = conn.execute(
                "SELECT path, fts_rowid FROM meta WHERE path > ? ORDER BY path LIMIT ?",
                (last_path, _STALE_BATCH_SIZE),
            ).fetchall()
            if not stale:
                break
            last_path = stale[-1][0]
            for stored_path, fts_rowid in stale:
                should_delete = _run_with_timeout(lambda p=stored_path: _is_stale(p, roots))
                if should_delete:
                    conn.execute("DELETE FROM files_fts WHERE rowid = ?", (fts_rowid,))
                    conn.execute("DELETE FROM meta WHERE path = ?", (stored_path,))
                stats["stale_checked"] += 1
                report("stale", Path(stored_path))
        del stats["stale_checked"]
        del stats["stale_total"]
        for root in roots:
            report("root", root)
            for path in _iter_candidate_files(root):
                stats["scanned"] += 1
                report("file", path)
                result = _index_file(conn, path)
                stats[result] += 1
                report("result", path)
                if verbose and stats["scanned"] % 500 == 0:
                    conn.commit()
                    print(f"  ...{stats['scanned']} scanned", file=sys.stderr)
        conn.commit()
    finally:
        conn.close()
    stats["seconds"] = round(time.time() - t0, 2)
    report("done")
    return stats


if __name__ == "__main__":
    if "--migrate-only" in sys.argv[1:]:
        conn = _connect()
        conn.close()
        result = {"index_policy_version": _INDEX_POLICY_VERSION}
    else:
        result = build_index()
    print(result)
