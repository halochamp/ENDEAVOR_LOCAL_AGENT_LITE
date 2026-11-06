"""Compatibility shim; semantic client lives in MCP-RagDoc/semantic_client.py."""
from __future__ import annotations

import importlib
import sys

import config as _lite_config  # Host config keeps priority over standalone config.

_RAG_DIR = _lite_config.RAG_CODE_DIR
_rag_dir = str(_RAG_DIR)
if _rag_dir not in sys.path:
    sys.path.append(_rag_dir)
sys.modules[__name__] = importlib.import_module("semantic_client")
