"""Standalone child-process entry point for indexer.build_index (PLAN.md
§9.2). web_ui.py spawns this as a short-lived subprocess instead of running
build_index() in-thread inside the always-on UI process.

Why a subprocess and not just a thread (as before): indexer._run_with_timeout
abandons a per-file daemon thread whenever a read stalls (a cloud-sync
placeholder download, an unresponsive network mount) rather than waiting on
it. Those threads are harmless one at a time, but a whole-machine scan that
hits many stalled files in one run could otherwise accumulate an unbounded
number of them for the lifetime of the UI process. Running the whole job in
its own process means every one of those leaked threads dies for free when
this process exits, instead of accumulating in a process meant to stay up
indefinitely.

Progress streams out as one bounded JSON line per event on stdout (never the
whole scan buffered at once), so the parent can relay live progress to the
browser without holding more than the current line in memory either.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import rag_mcp_service


def _emit(event: str, path: Path | None, stats: dict) -> None:
    payload: dict = {"event": event, "stats": stats}
    if path is not None:
        payload["path"] = str(path)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    # /sync_kb owns the complete knowledge-root -> streaming chunks -> SQLite flow.
    # Keep the old plain index mode for deterministic developer/test callers,
    # but the browser's knowledge command always takes the explicit flag.
    if "--sync-kb" in sys.argv[1:]:
        # The UI shortcut and MCP sync_kb share this exact service boundary.
        # kb_sync underneath emits the authoritative final "done" event.
        rag_mcp_service.sync_kb(on_progress=_emit)
        return

    # Retain the old environment override for isolated test corpora and
    # one-off developer runs. Normal Lite startup has no such variable and
    # therefore indexes only config.INDEX_ROOTS (rag_document/md).
    configured = os.getenv("LITE_INDEX_ROOTS")
    roots = [Path(item).expanduser() for item in configured.split(",") if item.strip()] if configured else None
    # Legacy /index writes the same SQLite file as /sync_kb.  Route it through
    # the shared exclusive lock so an external MCP/UI sync cannot interleave
    # schema changes or FTS writes with this compatibility job.
    rag_mcp_service.index_legacy(roots=roots, verbose=False, on_progress=_emit)


if __name__ == "__main__":
    main()
