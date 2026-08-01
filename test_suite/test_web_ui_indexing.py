# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""PLAN.md §9.2 gate: /index runs as its own short-lived subprocess (so a
stalled per-file read never accumulates state in the always-on UI process),
and /index + agent chat turns from different browser tabs serialize through
one shared heavy-work gate instead of silently running concurrently.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
import config
import indexer
import web_ui


def _write_pdf_corpus(directory, count: int, pages: int) -> None:
    """A batch of real, multi-page PDFs -- enough genuine extraction work
    (PyMuPDF parsing, not synthetic strings) that a whole indexing run takes
    long enough to interrupt mid-flight, and enough of it to show a native-
    allocator retention difference across many documents in one process."""
    import fitz
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        doc = fitz.open()
        for _ in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), ("real extraction workload text " * 200))
        doc.save(directory / f"doc_{i}.pdf")
        doc.close()


class _FakeWebSocket:
    """Records every payload sent, tagged with a monotonic timestamp, so
    ordering across two concurrent "connections" can be asserted."""

    def __init__(self, label: str, timeline: list[tuple[str, dict]]):
        self.label = label
        self.timeline = timeline

    async def send_json(self, payload: dict) -> None:
        self.timeline.append((self.label, payload))


def test_index_and_chat_serialize_through_the_shared_heavy_work_lock(monkeypatch, tmp_path):
    """A chat turn started while /index is running must wait for it (in
    order), be told it's queued, and only actually run afterward -- not
    interleave with the index job the way the old separate locks allowed."""
    # A deliberately slow fake worker: a real 1-file corpus indexes fast
    # enough that the race window below could close before the chat task
    # even checks the lock, making this test flaky. A short, controlled
    # sleep makes the "index is still running" window deterministic.
    slow_worker = tmp_path / "slow_worker.py"
    slow_worker.write_text(
        "import time, sys\n"
        "sys.stdout.write('{\"event\": \"root\", \"stats\": {}}\\n'); sys.stdout.flush()\n"
        "time.sleep(0.5)\n"
        "sys.stdout.write('{\"event\": \"done\", \"stats\": {\"scanned\": 0, \"inserted\": 0, "
        "\"updated\": 0, \"skipped\": 0, \"error\": 0, \"seconds\": 0.5}}\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_ui, "_INDEX_WORKER", slow_worker)
    monkeypatch.setattr(web_ui, "_active_index_process", None)
    timeline: list[tuple[str, dict]] = []
    turn_started_at: list[float] = []

    def fake_run_turn(
        user_input, messages, on_step=None, attachment_image=None,
        cancel_event=None, stream_tokens=False,
    ):
        assert stream_tokens is True
        turn_started_at.append(time.time())
        return agent.TurnResult(text="ok")

    monkeypatch.setattr(agent, "run_turn", fake_run_turn)

    async def scenario():
        index_ws = _FakeWebSocket("index", timeline)
        chat_ws = _FakeWebSocket("chat", timeline)

        index_task = asyncio.create_task(web_ui._run_index(index_ws, web_ui.BrowserSession()))
        await asyncio.sleep(0.1)  # let the index job actually acquire the lock first
        index_started_at = time.time()
        chat_task = asyncio.create_task(
            web_ui._run_query(chat_ws, web_ui.BrowserSession(), "hello")
        )
        await asyncio.gather(index_task, chat_task)
        return index_started_at

    index_started_at = asyncio.run(scenario())

    # The chat turn must not have started until after the index job began.
    assert turn_started_at[0] >= index_started_at

    # The chat connection must have been told it was queued while indexing
    # was in progress, before its own response arrived.
    chat_events = [payload for label, payload in timeline if label == "chat"]
    queued_messages = [p for p in chat_events if p.get("type") == "progress" and "index" in p.get("msg", "")]
    assert queued_messages, f"chat tab was never told indexing was in progress: {chat_events}"
    response_index = next(i for i, p in enumerate(chat_events) if p.get("type") == "response")
    assert chat_events.index(queued_messages[0]) < response_index

    # The index job's own "done" must appear before the chat's "response" in
    # the interleaved global timeline -- proof of real serialization, not
    # just independently-correct per-connection ordering.
    index_done_pos = next(i for i, (label, p) in enumerate(timeline) if label == "index" and p.get("type") == "done")
    chat_response_pos = next(i for i, (label, p) in enumerate(timeline) if label == "chat" and p.get("type") == "response")
    assert index_done_pos < chat_response_pos


def test_double_index_click_is_rejected_not_queued(monkeypatch, tmp_path):
    """Two /index requests (e.g. a double click) must reject the second
    immediately -- unlike cross-type (index vs. chat) contention, which
    queues -- since running two index jobs at once would race the same
    SQLite file for no benefit."""
    slow_worker = tmp_path / "slow_worker.py"
    slow_worker.write_text(
        "import time, sys\n"
        "sys.stdout.write('{\"event\": \"root\", \"stats\": {}}\\n'); sys.stdout.flush()\n"
        "time.sleep(1)\n"
        "sys.stdout.write('{\"event\": \"done\", \"stats\": {\"scanned\": 0, \"inserted\": 0, "
        "\"updated\": 0, \"skipped\": 0, \"error\": 0, \"seconds\": 1.0}}\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_ui, "_INDEX_WORKER", slow_worker)
    monkeypatch.setattr(web_ui, "_active_index_process", None)
    timeline: list[tuple[str, dict]] = []

    async def scenario():
        ws1 = _FakeWebSocket("first", timeline)
        ws2 = _FakeWebSocket("second", timeline)
        task1 = asyncio.create_task(web_ui._run_index(ws1, web_ui.BrowserSession()))
        await asyncio.sleep(0.2)  # let the first job actually start (still sleeping)
        await web_ui._run_index(ws2, web_ui.BrowserSession())
        await task1

    asyncio.run(scenario())

    second_events = [p for label, p in timeline if label == "second"]
    assert second_events == [{"type": "error", "msg": "กำลังสร้าง RAG index อยู่แล้วในอีกหน้าต่างหนึ่ง"}]


def test_cross_tab_stop_does_not_cancel_another_tabs_index(monkeypatch, tmp_path):
    """A real two-connection scenario through the actual WebSocket endpoint
    (not fakes): Tab A starts /index; Tab B -- which has nothing of its own
    running -- sends /stop (e.g. to abandon a queued chat that never
    actually got sent). That must NOT kill Tab A's index -- _active_index_owner
    exists precisely to prevent one tab's /stop from cancelling a stranger's
    running job."""
    slow_worker = tmp_path / "slow_worker.py"
    slow_worker.write_text(
        "import time, sys\n"
        "sys.stdout.write('{\"event\": \"root\", \"stats\": {}}\\n'); sys.stdout.flush()\n"
        "time.sleep(1)\n"
        "sys.stdout.write('{\"event\": \"done\", \"stats\": {\"scanned\": 0, \"inserted\": 0, "
        "\"updated\": 0, \"skipped\": 0, \"error\": 0, \"seconds\": 1.0}}\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_ui, "_INDEX_WORKER", slow_worker)
    monkeypatch.setattr(web_ui, "_active_index_process", None)
    monkeypatch.setattr(web_ui, "_active_index_owner", None)

    from fastapi.testclient import TestClient
    origin = {"origin": f"http://127.0.0.1:{config.UI_PORT}"}
    with TestClient(web_ui.app) as client:
        with client.websocket_connect("/ws", headers=origin) as ws_a, \
                client.websocket_connect("/ws", headers=origin) as ws_b:
            ws_a.receive_json(); ws_a.receive_json()  # status, files
            ws_b.receive_json(); ws_b.receive_json()

            ws_a.send_json({"type": "command", "cmd": "/index"})
            # Indexing uses ``busy`` (not the chat-only ``start`` event) so
            # the browser keeps its dedicated indexing label and does not
            # create an assistant response bubble.
            assert ws_a.receive_json()["type"] == "busy"

            # Tab B has no turn/index of its own running; its /stop must be a
            # no-op for Tab A's job (this is also the existing idle-/stop
            # no-op contract, just from a second connection).
            ws_b.send_json({"type": "command", "cmd": "/stop"})

            msg = ws_a.receive_json()
            while msg.get("type") == "progress":
                msg = ws_a.receive_json()
            assert msg["type"] == "response", f"Tab A's index was cancelled by Tab B's /stop: {msg}"
            assert ws_a.receive_json()["type"] == "done"


def test_owning_tabs_stop_still_cancels_its_own_index(monkeypatch, tmp_path):
    """The flip side of the ownership check: the tab that actually started
    the index must still be able to cancel it with its own /stop."""
    hanging_worker = tmp_path / "hang_forever.py"
    hanging_worker.write_text("import time\ntime.sleep(300)\n", encoding="utf-8")
    monkeypatch.setattr(web_ui, "_INDEX_WORKER", hanging_worker)
    monkeypatch.setattr(web_ui, "_active_index_process", None)
    monkeypatch.setattr(web_ui, "_active_index_owner", None)

    from fastapi.testclient import TestClient
    origin = {"origin": f"http://127.0.0.1:{config.UI_PORT}"}
    with TestClient(web_ui.app) as client:
        with client.websocket_connect("/ws", headers=origin) as ws:
            ws.receive_json(); ws.receive_json()
            ws.send_json({"type": "command", "cmd": "/index"})
            assert ws.receive_json()["type"] == "busy"
            ws.send_json({"type": "command", "cmd": "/stop"})
            assert ws.receive_json() == {"type": "cancelled"}


def test_index_subprocess_is_terminated_on_cancel(monkeypatch, tmp_path):
    """Simulates a `/stop` arriving mid-index (the real handler sets
    _index_cancel_requested before terminating -- see web_ui._ws_reader):
    the child process must actually be killed, not left running/orphaned,
    and must be reported as a deliberate cancel, never as a crash."""
    hanging_worker = tmp_path / "hang_forever.py"
    hanging_worker.write_text("import time\ntime.sleep(300)\n", encoding="utf-8")
    monkeypatch.setattr(web_ui, "_INDEX_WORKER", hanging_worker)

    timeline: list[tuple[str, dict]] = []
    ws = _FakeWebSocket("cancel", timeline)

    async def scenario():
        task = asyncio.create_task(web_ui._run_index_subprocess(ws))
        await asyncio.sleep(0.3)  # let the child process actually spawn
        process = web_ui._active_index_process
        assert process is not None
        pid = process.pid
        web_ui._index_cancel_requested = True  # what the real "/stop" handler sets
        process.terminate()
        result = await asyncio.wait_for(task, timeout=10)
        return result, pid

    (stats, crashed, detail), pid = asyncio.run(scenario())
    assert stats is None
    assert crashed is False, f"a deliberate cancel must never be reported as a crash (detail={detail!r})"
    assert web_ui._active_index_process is None
    # The child must actually be gone (not left as a zombie/orphan) -- signaling
    # it again should fail with "no such process".
    with __import__("pytest").raises(ProcessLookupError):
        os.kill(pid, 0)


def test_index_subprocess_crash_is_reported_as_error_not_cancelled(monkeypatch, tmp_path):
    """A child that dies on its own (uncaught exception, no "done" line)
    must never be silently reported as a user-initiated cancel -- that
    would hide a genuine failure as if nothing happened."""
    crashing_worker = tmp_path / "crash.py"
    crashing_worker.write_text(
        "import sys\nprint('partial progress, then boom', file=sys.stdout)\n"
        "raise RuntimeError('synthetic crash for the 9.2 gate test')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(web_ui, "_INDEX_WORKER", crashing_worker)
    monkeypatch.setattr(web_ui, "_active_index_process", None)

    timeline: list[tuple[str, dict]] = []
    ws = _FakeWebSocket("crash", timeline)
    stats, crashed, detail = asyncio.run(web_ui._run_index_subprocess(ws))

    assert stats is None
    assert crashed is True
    assert "boom" in detail or "RuntimeError" in detail


def test_indexer_worker_subprocess_streams_jsonl_and_exits(tmp_path):
    """End-to-end smoke test of the actual child entry point (not a fake):
    a small isolated corpus indexes to completion, emitting well-formed
    JSONL progress lines and a final "done" event, then exits promptly."""
    import json
    import subprocess

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("hello from the subprocess test\n", encoding="utf-8")
    data_dir = config.WORKSPECT_DIR / "_test_9_2_worker_smoke_data"
    shutil.rmtree(data_dir, ignore_errors=True)

    env = dict(os.environ)
    env["LITE_INDEX_ROOTS"] = str(corpus)
    env["LITE_DATA_DIR"] = str(data_dir)
    try:
        result = subprocess.run(
            [sys.executable, str(config.PROJECT_ROOT / "indexer_worker.py")],
            capture_output=True, text=True, timeout=30, env=env,
            cwd=str(config.PROJECT_ROOT),
        )
        assert result.returncode == 0, result.stderr
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert lines, "worker produced no progress output"
        events = [json.loads(line) for line in lines]
        assert events[0]["event"] == "root"
        assert events[-1]["event"] == "done"
        assert events[-1]["stats"]["scanned"] == 1
        assert events[-1]["stats"]["inserted"] == 1
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_cancelled_index_leaves_a_valid_queryable_database(monkeypatch, tmp_path):
    """Killing the child mid-run must not corrupt the SQLite file: SQLite's
    own crash safety (rollback journal) should mean the DB is still a valid,
    queryable file afterward, just missing whatever wasn't committed yet."""
    corpus = tmp_path / "corpus"
    _write_pdf_corpus(corpus, count=400, pages=30)
    data_dir = config.WORKSPECT_DIR / "_test_9_2_cancel_integrity_data"
    shutil.rmtree(data_dir, ignore_errors=True)
    monkeypatch.setenv("LITE_INDEX_ROOTS", str(corpus))
    monkeypatch.setenv("LITE_DATA_DIR", str(data_dir))
    import importlib
    importlib.reload(config)
    monkeypatch.setattr(web_ui, "_active_index_process", None)

    timeline: list[tuple[str, dict]] = []
    ws = _FakeWebSocket("cancel-db", timeline)

    async def scenario():
        task = asyncio.create_task(web_ui._run_index_subprocess(ws))
        # Poll for real progress rather than a fixed sleep: cancel as soon as
        # the child has genuinely started extracting, regardless of how fast
        # this machine happens to process PDFs.
        for _ in range(200):
            if timeline or task.done():
                break
            await asyncio.sleep(0.02)
        process = web_ui._active_index_process
        assert process is not None and not task.done(), (
            "index job finished before we could cancel it -- corpus too small/fast"
        )
        web_ui._index_cancel_requested = True
        process.terminate()
        return await asyncio.wait_for(task, timeout=10)

    try:
        stats, crashed, _detail = asyncio.run(scenario())
        assert stats is None
        assert crashed is False

        db_path = config.FILE_INDEX_DB
        assert db_path.exists(), "expected a partially-written DB file to exist after a mid-run cancel"
        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            conn.execute("SELECT COUNT(*) FROM meta").fetchone()  # must not raise
        finally:
            conn.close()
    finally:
        importlib.reload(config)
        shutil.rmtree(data_dir, ignore_errors=True)


def test_subprocess_delegated_index_matches_in_process_result(tmp_path):
    """The subprocess-delegated run must index the exact same content as
    calling indexer.build_index() directly in-process over the same
    corpus -- moving *how* it's invoked must not change *what* gets
    indexed."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha beta gamma\n", encoding="utf-8")
    (corpus / "b.txt").write_text("delta epsilon zeta\n", encoding="utf-8")
    (corpus / "รายงาน.md").write_text("เนื้อหาภาษาไทย\n", encoding="utf-8")

    inprocess_dir = config.WORKSPECT_DIR / "_test_9_2_parity_inprocess"
    subprocess_dir = config.WORKSPECT_DIR / "_test_9_2_parity_subprocess"
    shutil.rmtree(inprocess_dir, ignore_errors=True)
    shutil.rmtree(subprocess_dir, ignore_errors=True)

    def rows_from(db_path) -> set:
        conn = sqlite3.connect(str(db_path))
        try:
            return set(conn.execute(
                "SELECT m.name, m.size_bytes, m.kind, f.content FROM meta m "
                "JOIN files_fts f ON f.path = m.path ORDER BY m.name"
            ).fetchall())
        finally:
            conn.close()

    try:
        # In-process (direct call, same as before this phase existed).
        os.environ["LITE_DATA_DIR"] = str(inprocess_dir)
        import importlib
        importlib.reload(config)
        importlib.reload(indexer)
        indexer.build_index(roots=[corpus], verbose=False)
        inprocess_rows = rows_from(config.FILE_INDEX_DB)

        # Subprocess-delegated (the new §9.2 path).
        env = dict(os.environ)
        env["LITE_INDEX_ROOTS"] = str(corpus)
        env["LITE_DATA_DIR"] = str(subprocess_dir)
        result = subprocess.run(
            [sys.executable, str(config.PROJECT_ROOT / "indexer_worker.py")],
            capture_output=True, text=True, timeout=30, env=env,
            cwd=str(config.PROJECT_ROOT),
        )
        assert result.returncode == 0, result.stderr
        subprocess_rows = rows_from(subprocess_dir / "file_index.sqlite3")

        assert subprocess_rows == inprocess_rows
        assert len(subprocess_rows) == 3
    finally:
        os.environ.pop("LITE_DATA_DIR", None)
        os.environ.pop("LITE_INDEX_ROOTS", None)
        importlib.reload(config)
        importlib.reload(indexer)
        shutil.rmtree(inprocess_dir, ignore_errors=True)
        shutil.rmtree(subprocess_dir, ignore_errors=True)


def test_many_hung_child_threads_do_not_affect_ui_process_thread_count(monkeypatch, tmp_path):
    """indexer._run_with_timeout abandons a fresh daemon thread per stalled
    file rather than waiting on it -- so a whole-machine scan that hits many
    stalled files could leak many threads. PLAN.md §9.2's fix is moving the
    whole job into its own process specifically so that leak's lifetime is
    bounded to a short-lived child, invisible to the always-on UI process.
    Prove it: a fake child that leaks 20 never-finishing daemon threads must
    still let this (stand-in-for-UI) process's own thread count stay flat,
    and must not block on those threads either (daemon threads never block
    process exit)."""
    leaky_worker = tmp_path / "leaky_worker.py"
    leaky_worker.write_text(textwrap.dedent("""
        import json, sys, threading, time

        def stuck():
            time.sleep(300)

        for _ in range(20):
            threading.Thread(target=stuck, daemon=True).start()

        sys.stdout.write(json.dumps({"event": "root", "stats": {}}) + "\\n")
        sys.stdout.write(json.dumps({"event": "done", "stats": {
            "scanned": 0, "inserted": 0, "updated": 0, "skipped": 0, "error": 0, "seconds": 0.0,
        }}) + "\\n")
        sys.stdout.flush()
    """), encoding="utf-8")
    monkeypatch.setattr(web_ui, "_INDEX_WORKER", leaky_worker)
    monkeypatch.setattr(web_ui, "_active_index_process", None)

    before = threading.active_count()
    timeline: list[tuple[str, dict]] = []
    ws = _FakeWebSocket("leaky", timeline)
    t0 = time.time()
    stats, crashed, _detail = asyncio.run(asyncio.wait_for(web_ui._run_index_subprocess(ws), timeout=10))
    elapsed = time.time() - t0
    after = threading.active_count()

    assert crashed is False
    assert stats is not None
    # The 20 stuck threads live and die entirely inside the child process --
    # this process's own thread count must be unaffected by how many there were.
    assert after == before, f"UI-process thread count changed ({before} -> {after}) from a child's leaked threads"
    # Daemon threads must not have blocked the child's (and thus our) exit.
    assert elapsed < 5, f"took {elapsed:.1f}s -- looks like something waited on the child's stuck threads"


def _peak_rss_mb(script: str, env: dict) -> float:
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stderr
    return float(result.stdout.strip().splitlines()[-1]) / (1024 * 1024)


def test_index_subprocess_isolates_extraction_memory_from_ui_process(tmp_path):
    """This is the actual point of PLAN.md §9.2: verify, don't just assert
    by architecture, that delegating indexing to a child process keeps
    extraction's memory footprint out of the long-running UI process.

    Measured directly (peak RSS, apples-to-apples import baseline) rather
    than inferred: calling indexer.build_index() in-process should spike
    *this* process's peak; delegating the identical corpus to a subprocess
    should not, because the extraction work (and whatever PyMuPDF/openpyxl's
    native allocators retain afterward) happens in a disposable child.
    """
    corpus = tmp_path / "corpus"
    _write_pdf_corpus(corpus, count=60, pages=20)
    project_root = str(config.PROJECT_ROOT)
    worker_path = str(config.PROJECT_ROOT / "indexer_worker.py")

    data_dir_inprocess = config.WORKSPECT_DIR / "_test_9_2_ram_inprocess"
    data_dir_subprocess = config.WORKSPECT_DIR / "_test_9_2_ram_subprocess"
    shutil.rmtree(data_dir_inprocess, ignore_errors=True)
    shutil.rmtree(data_dir_subprocess, ignore_errors=True)

    baseline_script = textwrap.dedent(f"""
        import resource, sys
        sys.path.insert(0, {project_root!r})
        import indexer  # noqa: F401 -- import cost only, no indexing work
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """)
    inprocess_script = textwrap.dedent(f"""
        import resource, sys
        sys.path.insert(0, {project_root!r})
        import os
        from pathlib import Path
        os.environ["LITE_DATA_DIR"] = {str(data_dir_inprocess)!r}
        import indexer
        indexer.build_index(roots=[Path({str(corpus)!r})], verbose=False)
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """)
    delegated_script = textwrap.dedent(f"""
        import os, resource, subprocess, sys
        sys.path.insert(0, {project_root!r})
        env = dict(os.environ)
        env["LITE_INDEX_ROOTS"] = {str(corpus)!r}
        env["LITE_DATA_DIR"] = {str(data_dir_subprocess)!r}
        subprocess.run([sys.executable, {worker_path!r}], capture_output=True, timeout=90, env=env, cwd={project_root!r})
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """)

    try:
        baseline_mb = _peak_rss_mb(baseline_script, dict(os.environ))
        inprocess_mb = _peak_rss_mb(inprocess_script, dict(os.environ))
        delegated_mb = _peak_rss_mb(delegated_script, dict(os.environ))
    finally:
        shutil.rmtree(data_dir_inprocess, ignore_errors=True)
        shutil.rmtree(data_dir_subprocess, ignore_errors=True)

    inprocess_delta = inprocess_mb - baseline_mb
    delegated_delta = delegated_mb - baseline_mb
    print(
        f"\n[PLAN.md 9.2 RAM] baseline={baseline_mb:.1f}MB "
        f"in-process delta={inprocess_delta:.1f}MB delegated delta={delegated_delta:.1f}MB "
        f"(60 PDFs x 20 pages)"
    )

    # The delegated (measuring) process must show a much smaller delta:
    # the actual extraction memory lives, and is released, in the disposable
    # child -- not in the process we're measuring here (standing in for the
    # always-on UI process).
    assert delegated_delta < max(5.0, inprocess_delta * 0.5), (
        f"subprocess delegation did not isolate extraction memory from the "
        f"measured process: in-process delta {inprocess_delta:.1f}MB, "
        f"delegated delta {delegated_delta:.1f}MB"
    )


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
