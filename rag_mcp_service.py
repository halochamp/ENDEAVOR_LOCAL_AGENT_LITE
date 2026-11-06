"""Agent Lite adapter for the standalone MCP-RagDoc service.

All retrieval/sync/semantic logic lives under MCP-RagDoc.  This adapter keeps
Lite's legacy in-process UI/CLI entry points and its old /index compatibility
path without adding those host-specific concerns to MCP-RagDoc itself.
"""
from __future__ import annotations

from collections.abc import Callable
import importlib
import sys

import config as _lite_config  # Host config keeps priority over standalone config.
import helpers as _lite_helpers  # Host helpers keep priority for legacy callers.

_RAG_DIR = _lite_config.RAG_CODE_DIR
_rag_dir = str(_RAG_DIR)
if _rag_dir not in sys.path:
    sys.path.append(_rag_dir)
_core = importlib.import_module("rag_service")

search = _core.search
# Lite's in-process UI/CLI keeps the historical blocking callback contract.
sync_kb = _core.sync_kb_blocking
format_sync_result = _core.format_sync_result
health = _core.health
_kb_lock = _core._kb_lock


def index_legacy(
    *,
    roots: list[Path] | None = None,
    verbose: bool = False,
    on_progress: Callable | None = None,
) -> dict[str, int]:
    """Lite-only compatibility for the historical /index command."""
    import indexer

    with _core._kb_lock(exclusive=True):
        return indexer.build_index(roots=roots, verbose=verbose, on_progress=on_progress)
