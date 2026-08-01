# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""save_output.py — spill full tool output to SSD before truncating for context.

Keeping the model's context small keeps its KV cache small, which is the
main lever on RAM for an in-process model (see PLAN.md §3). Nothing here
is ever loaded back into RAM automatically — the saved path is just a
pointer the user (or a future read_file call) can follow.
"""
from __future__ import annotations
import hashlib
import time
from pathlib import Path

import config

_OUT_DIR = config.LOG_DIR / "tool_outputs"


def save_full_output(text: str, tool_name: str) -> str:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_DIR.chmod(0o700)
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:10]
    path = _OUT_DIR / f"{tool_name}_{int(time.time())}_{h}.txt"
    path.write_text(text, encoding="utf-8", errors="replace")
    path.chmod(0o600)
    return str(path)
