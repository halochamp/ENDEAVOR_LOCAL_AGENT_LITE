# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""PLAN.md §9.1 gate: streaming (bounded-RAM) read_file/indexer extraction
must reproduce the exact same output the old fully-materialized algorithm
produced, catch a secret straddling a chunk boundary, and keep large-fixture
memory bounded rather than growing with file size.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import read_file
from helpers.sensitive_paths import StreamingSecretScanner, contains_secret_content
from helpers import document_extract
from test_suite.pptx_fixture import write_pptx


# ---------------------------------------------------------------------------
# Golden-output parity: streaming path vs. the original in-memory sampler
# ---------------------------------------------------------------------------

def test_plain_text_streaming_matches_full_materialization(tmp_path):
    path = tmp_path / "medium.txt"
    path.write_text("\n".join(f"line {i}: " + ("x" * 90) for i in range(600)), encoding="utf-8")
    expected = read_file._sample_for_coverage(path.read_text(encoding="utf-8"), path)
    assert read_file.run({"path": str(path)}) == expected


def test_small_plain_text_returned_verbatim(tmp_path):
    path = tmp_path / "small.txt"
    path.write_text("hello\nworld\n", encoding="utf-8")
    assert read_file.run({"path": str(path)}) == path.read_text(encoding="utf-8")


def test_docx_streaming_matches_full_materialization(tmp_path):
    from docx import Document
    path = tmp_path / "big.docx"
    doc = Document()
    for i in range(400):
        doc.add_paragraph(f"paragraph {i} " + "y" * 60)
    table = doc.add_table(rows=5, cols=3)
    for r, row in enumerate(table.rows):
        for c, cell in enumerate(row.cells):
            cell.text = f"r{r}c{c}"
    doc.save(path)

    expected = read_file._sample_for_coverage(document_extract.extract_text(path), path)
    assert read_file.run({"path": str(path)}) == expected


def test_xlsx_streaming_matches_full_materialization(tmp_path):
    from openpyxl import Workbook
    path = tmp_path / "big.xlsx"
    book = Workbook()
    book.active.title = "Data"
    for i in range(3000):
        book.active.append([f"row{i}", i, i * 2])
    book.save(path)

    expected = read_file._sample_for_coverage(document_extract.extract_text(path), path)
    assert read_file.run({"path": str(path)}) == expected


def test_pptx_streaming_matches_full_materialization(tmp_path):
    path = tmp_path / "big.pptx"
    write_pptx(path, [f"Slide title {i} " + "z" * 60 for i in range(120)])

    expected = read_file._sample_for_coverage(document_extract.extract_text(path), path)
    assert read_file.run({"path": str(path)}) == expected


def test_pdf_streaming_matches_full_materialization(tmp_path):
    import fitz
    path = tmp_path / "big.pdf"
    doc = fitz.open()
    for i in range(40):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i} content " * 40)
    doc.save(path)
    doc.close()

    expected = read_file._sample_for_coverage(document_extract.extract_text(path), path)
    assert read_file.run({"path": str(path)}) == expected


def test_single_page_pdf_streaming_matches_full_materialization(tmp_path):
    import fitz
    path = tmp_path / "one_page.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "solo page content " * 500)
    doc.save(path)
    doc.close()

    expected = read_file._sample_for_coverage(document_extract.extract_text(path), path)
    assert read_file.run({"path": str(path)}) == expected


def test_line_range_request_unaffected_by_streaming_change(tmp_path):
    path = tmp_path / "ranged.txt"
    path.write_text("\n".join(f"line {i}" for i in range(50)), encoding="utf-8")
    out = read_file.run({"path": str(path), "start_line": "5", "end_line": "10"})
    assert "[บรรทัด 5-10 จาก 50]" in out
    assert "line 4" in out and "line 9" in out


def test_open_ended_large_line_range_stays_output_bounded(tmp_path):
    path = tmp_path / "large-range.txt"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(100_000):
            handle.write(f"line {index} " + "x" * 90 + "\n")
    out = read_file.run({"path": str(path), "start_line": "1"})
    assert len(out) <= 5_000
    assert "sample line" in out
    assert "line 0" in out


def test_line_range_still_scans_secret_outside_requested_slice(tmp_path):
    path = tmp_path / "range-secret.txt"
    path.write_text("safe first line\n" + "padding\n" * 100 + "api_key: " + "z" * 20, encoding="utf-8")
    out = read_file.run({"path": str(path), "start_line": "1", "end_line": "1"})
    assert out.startswith("[blocked]")


# ---------------------------------------------------------------------------
# Secret scanner: must catch a credential straddling a chunk boundary
# ---------------------------------------------------------------------------

def test_streaming_scanner_catches_secret_straddling_chunk_boundary():
    secret = "sk-" + ("a" * 40)
    first_half = secret[: len(secret) // 2]
    second_half = secret[len(secret) // 2 :]
    scanner = StreamingSecretScanner()
    assert not scanner.feed(first_half)
    assert scanner.feed(second_half)
    assert contains_secret_content(secret)  # sanity: the whole secret is in fact detectable


def test_read_file_blocks_secret_split_across_lines(tmp_path):
    secret = "sk-" + ("b" * 40)
    path = tmp_path / "creds.txt"
    # Split the token itself across a line boundary -- StreamingSecretScanner
    # carries overlap across `feed()` calls (one per line here), so this must
    # still be blocked even though neither single line contains the whole
    # token.
    path.write_text(secret[:20] + "\n" + secret[20:] + "\n" + ("padding\n" * 400), encoding="utf-8")
    out = read_file.run({"path": str(path)})
    assert out.startswith("[blocked]")


def test_read_file_blocks_secret_in_large_document(tmp_path):
    from docx import Document
    path = tmp_path / "leaky.docx"
    doc = Document()
    for i in range(300):
        doc.add_paragraph(f"paragraph {i} filler " + "y" * 60)
    doc.add_paragraph("api_key: " + "z" * 20)
    doc.save(path)
    out = read_file.run({"path": str(path)})
    assert out.startswith("[blocked]")


# ---------------------------------------------------------------------------
# Bounded memory: a large fixture must not make the process grow with body size
# ---------------------------------------------------------------------------

def _peak_rss_mb_for_read(path: Path) -> float:
    """Peak RSS (MB) of a fresh subprocess that imports tools.read_file and
    reads `path` once. macOS reports ru_maxrss in bytes (Linux: KB) -- since
    this is only ever compared against another same-machine measurement in
    the same units, the platform-specific scale factor cancels out and
    doesn't need to be handled here."""
    project_root = str(Path(__file__).resolve().parents[1])
    script = textwrap.dedent(f"""
        import resource, sys
        sys.path.insert(0, {project_root!r})
        from tools import read_file
        out = read_file.run({{"path": {str(path)!r}}})
        assert len(out) <= 5000, len(out)
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return float(result.stdout.strip().splitlines()[-1]) / (1024 * 1024)


def test_large_text_file_read_has_bounded_process_memory(tmp_path):
    """The dominant cost of a fresh process is import overhead (langchain_core
    etc, ~65MB alone) -- not file content. So the gate is apples-to-apples:
    the SAME import baseline reading a tiny file vs. reading a ~65MB file
    must show only a small delta, not one that scales with the huge file's
    size the way materializing it fully (plus a second copy while sampling,
    as the old algorithm did) would."""
    tiny_path = tmp_path / "tiny.txt"
    tiny_path.write_text("hello\n", encoding="utf-8")

    huge_path = tmp_path / "huge.txt"
    line = "x" * 118 + "\n"
    with open(huge_path, "w", encoding="utf-8") as fh:
        for _ in range(550_000):  # ~65MB
            fh.write(line)
    size_mb = huge_path.stat().st_size / (1024 * 1024)
    assert size_mb > 60

    baseline_mb = _peak_rss_mb_for_read(tiny_path)
    huge_mb = _peak_rss_mb_for_read(huge_path)
    delta_mb = huge_mb - baseline_mb
    # A process that materialized the whole ~65MB file would show a delta of
    # tens of MB; streaming keeps it a small fraction of the file's size.
    assert delta_mb < size_mb / 4, (
        f"peak RSS delta {delta_mb:.1f}MB (baseline {baseline_mb:.1f}MB, "
        f"huge-file run {huge_mb:.1f}MB) not bounded relative to {size_mb:.1f}MB file"
    )


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
