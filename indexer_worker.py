# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
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
import sys
from pathlib import Path

import indexer


def _emit(event: str, path: Path | None, stats: dict) -> None:
    payload: dict = {"event": event, "stats": stats}
    if path is not None:
        payload["path"] = str(path)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    # build_index() itself emits a final on_progress("done", None, stats)
    # call once stats["seconds"] is set -- that line carries the complete
    # summary the parent needs, no separate final emit required here.
    indexer.build_index(verbose=False, on_progress=_emit)


if __name__ == "__main__":
    main()
