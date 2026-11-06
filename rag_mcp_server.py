#!/usr/bin/env python3
"""Compatibility shim; MCP server implementation lives in MCP-RagDoc/rag_server.py."""
from __future__ import annotations

import importlib
import runpy
import sys

import config as _lite_config  # Compatibility path intentionally uses Lite state.

_RAG_DIR = _lite_config.RAG_CODE_DIR
_rag_dir = str(_RAG_DIR)
_added = _rag_dir not in sys.path
if _added:
    sys.path.insert(0, _rag_dir)
try:
    if __name__ == "__main__":
        runpy.run_path(str(_RAG_DIR / "rag_server.py"), run_name="__main__")
    else:
        _impl = importlib.import_module("rag_server")
        sys.modules[__name__] = _impl
finally:
    if _added:
        sys.path.remove(_rag_dir)
