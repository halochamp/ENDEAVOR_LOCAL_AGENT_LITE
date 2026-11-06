#!/usr/bin/env python3
"""Compatibility shim; semantic worker lives in MCP-RagDoc/semantic_worker.py."""
from __future__ import annotations

import importlib.util
import runpy
import sys

import config as _lite_config  # Host config keeps priority over standalone config.
import helpers as _lite_helpers

_RAG_DIR = _lite_config.RAG_CODE_DIR
_rag_dir = str(_RAG_DIR)
if _rag_dir not in sys.path:
    sys.path.append(_rag_dir)

if __name__ == "__main__":
    runpy.run_path(str(_RAG_DIR / "semantic_worker.py"), run_name="__main__")
else:
    _spec = importlib.util.spec_from_file_location("_lite_semantic_worker_impl", _RAG_DIR / "semantic_worker.py")
    if _spec is None or _spec.loader is None:
        raise ImportError("unable to load MCP-RagDoc/semantic_worker.py")
    _impl = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _impl
    _spec.loader.exec_module(_impl)
    sys.modules[__name__] = _impl
