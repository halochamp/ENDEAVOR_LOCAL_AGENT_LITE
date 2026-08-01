# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""config.py — ENDEAVOR_AGENT_LITE configuration

Single source of truth for model id, paths, and limits. Everything is
env-overridable.

Architecture change (2026-07-23, user request): switched from an in-process
mlx_vlm.generate() wrapper to a real mlx_vlm.server + langchain_openai
ChatOpenAI client, matching ENDEAVOR_LOCAL_AGENT_MAX_VLM's llm.py/react.py
pattern exactly, so LangGraph's create_react_agent (which needs a LangChain
chat-model object, not a raw generate() function) can be used directly —
see PLAN.md for the full rationale and the AskUserQuestion decision trail.

This reintroduces a standing server process, which the project's original
design (PLAN.md §2) deliberately avoided for RAM reasons. Consequence: the
Phase 7 adaptive-chunked-prefill work (per-request, threshold-based
prefill_step_size) doesn't apply the way it did in-process — mlx_vlm.server
only accepts --prefill-step-size as a GLOBAL startup flag, not a per-request
override, so it can't distinguish a short turn-1 routing prompt from a long
in-turn tool_response the way llm.py used to.

PLAN.md §9.4 (2026-07-25) measured whether that mattered in THIS
architecture (structured tool-calling via bind_tools(), not the old raw-XML
parser Phase 7 tested against) before assuming per-request patching was
required: routing-probe (8 cases x3 rounds) was byte-identical between
mlx_vlm's own default (2048) and --prefill-step-size 512, and the live
6-scenario suite matched exactly (5/6 x3, same single pre-existing
unrelated failure) at both settings. Turn-1 routing prompts in this
architecture (~3,000 tokens with the current tool schema) are already
chunked at the default 2048 regardless -- there was never an unchunked
short-prompt path to protect here in the first place, unlike Phase 7's era.
An isolated long tool-response request (3,321 tokens) showed a real
peak_memory drop from 3.85GB to 3.03GB (~21%) at step_size=512, fresh
server restart per config -- traded against ~30% higher latency on that
same long request (~13.5s -> ~17.5s). The server is started WITH
--prefill-step-size 512 (see PLAN.md §9.4/"วิธีรัน") -- a global launch
flag, no per-request patch needed.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# ── Model / server ──────────────────────────────────────────────────────
MODEL_ID = os.getenv("LITE_MODEL", "mlx-community/Qwen3.5-2B-OptiQ-4bit")

MLX_SERVER_HOST = os.getenv("LITE_SERVER_HOST", "127.0.0.1")
# Use a dedicated default port so this agent can coexist with other local
# model servers on the same machine.
MLX_SERVER_PORT = int(os.getenv("LITE_SERVER_PORT", "8090"))
MLX_BASE_URL = os.getenv("LITE_MLX_BASE_URL", f"http://{MLX_SERVER_HOST}:{MLX_SERVER_PORT}/v1")
MLX_API_KEY = os.getenv("LITE_MLX_API_KEY", "x")  # server doesn't check it; ChatOpenAI needs a non-empty string
# Global launch flag, not a per-request param — see the module docstring's
# PLAN.md §9.4 measurement (2026-07-25) for why 512 was picked.
MLX_SERVER_PREFILL_STEP_SIZE = int(os.getenv("LITE_PREFILL_STEP_SIZE", "512"))
# Lite serializes heavy work and normally handles one image at a time.  The
# mlx-vlm default retains features for 20 distinct images, which is needless
# unified-memory pressure on an 8GB Mac.  Live A/B established 2 as the
# quality/latency-neutral production value.
MLX_SERVER_VISION_CACHE_SIZE = int(os.getenv("LITE_VISION_CACHE_SIZE", "2"))

# Browser UI is intentionally local-only.  It is a separate process from the
# model server, so it needs its own port.
# Do not make this environment-configurable: broadening to 0.0.0.0 would turn
# a local agent (with filesystem tools) into a LAN service.
UI_HOST = "127.0.0.1"
UI_PORT = int(os.getenv("LITE_UI_PORT", "8766"))

TEMPERATURE = float(os.getenv("LITE_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("LITE_MAX_TOKENS", "768"))
# Keep reasoning bounded inside MAX_TOKENS: 256 leaves roughly the previous
# 512-token visible-answer budget while adding explicit routing/reasoning room.
ENABLE_THINKING = os.getenv("LITE_ENABLE_THINKING", "1").lower() in ("1", "true", "yes", "on")
THINKING_BUDGET = int(os.getenv("LITE_THINKING_BUDGET", "256"))
# Ported from ENDEAVOR_LOCAL_AGENT_MAX_VLM/config.py (2026-07-24, live
# finding): without this, Qwen3.5-2B occasionally degenerates into massive
# token-level repetition on longer tool-augmented turns (observed live:
# a numbered list looping the same 4 bullet points 30+ times instead of
# stopping) -- reproducible on this exact model/server, not just MAX_VLM's
# 35B. MAX_VLM's comment: "1.05->1.1: breaks repetitive thinking loops
# faster." Passed via extra_body since it's an mlx_vlm.server-specific
# sampling param, not a standard OpenAI Chat Completions field.
REPETITION_PENALTY = float(os.getenv("LITE_REPETITION_PENALTY", "1.1"))

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# Lite may create files only in this explicit workspace.  This includes its
# own cache/index/log state; an environment override outside it is ignored.
WORKSPECT_DIR = (PROJECT_ROOT / "workspect").resolve()
WORKSPECT_DIR.mkdir(parents=True, exist_ok=True)

# The web UI starts mlx_vlm.server with the same interpreter that runs the
# agent. Override only when the model runtime deliberately lives elsewhere.
MLX_SERVER_PYTHON = Path(os.getenv("LITE_MLX_SERVER_PYTHON", sys.executable)).resolve()

_default_data_dir = WORKSPECT_DIR / ".lite_state"
_requested_data_dir = Path(os.getenv("LITE_DATA_DIR", str(_default_data_dir))).expanduser()
try:
    _requested_data_dir = _requested_data_dir.resolve()
    _requested_data_dir.relative_to(WORKSPECT_DIR)
    DATA_DIR = _requested_data_dir
except ValueError:
    DATA_DIR = _default_data_dir
FILE_INDEX_DB = DATA_DIR / "file_index.sqlite3"
WEB_CACHE_DB = DATA_DIR / "web_cache.sqlite3"
LOG_DIR = DATA_DIR / "logs"

# Roots the indexer walks by default. Do not scan `/`: whole-machine walks
# enter cloud-sync and system trees that are neither useful nor reliably
# readable. A comma-separated LITE_INDEX_ROOTS override remains available for
# one-off narrow indexing.
_DEFAULT_INDEX_ROOTS = [
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Downloads",
]
_configured_index_roots = os.getenv("LITE_INDEX_ROOTS")
INDEX_ROOTS = (
    [Path(p).expanduser() for p in _configured_index_roots.split(",") if p.strip()]
    if _configured_index_roots is not None
    else _DEFAULT_INDEX_ROOTS
)

# ── Tool / loop guardrails (see PLAN.md §6) ────────────────────────────────
MAX_TOOL_STEPS = int(os.getenv("LITE_MAX_TOOL_STEPS", "6"))
MAX_REPEAT_CALLS = int(os.getenv("LITE_MAX_REPEAT_CALLS", "2"))  # same (tool,args) hash

# Per-tool call quota per turn, independent of exact args. Added after a live
# finding (2026-07-23, training round): asked for "Python 3.13 features",
# the model called web_search 5 times with slightly reworded queries each
# time instead of synthesizing from the first result — RepeatCallGuard only
# catches byte-identical args, so varying the wording each call dodges it.
# Matches training_guide.md L69 (M5 browse-loop, no stopping criterion) and
# L12 ("web search loop -> hard counter at code, not prompt").
MAX_CALLS_PER_TOOL = int(os.getenv("LITE_MAX_CALLS_PER_TOOL", "3"))
TOOL_OUTPUT_TRUNCATE_CHARS = int(os.getenv("LITE_TOOL_TRUNCATE", "2000"))
BASH_TIMEOUT_SECONDS = int(os.getenv("LITE_BASH_TIMEOUT", "10"))

# All agent-created files and bash's implicit working directory stay inside workspect/, the
# only writable workspace directly beneath the LITE root. write_file resolves
# paths before checking this boundary, preventing direct and symlink traversal.
BASH_ALLOWED_ROOT = WORKSPECT_DIR

# read_file / tool-output size caps (keep context small -> keep KV cache small)
READ_FILE_MAX_CHARS = 5_000
# This is a security ceiling, not a performance preference.  Do not make it
# environment-overridable: a launch environment must not be able to turn an
# image tool into an unbounded file reader.
READ_IMAGE_MAX_BYTES = 10 * 1024 * 1024

# web_search
WEB_SEARCH_MAX_RESULTS = int(os.getenv("LITE_WEB_MAX_RESULTS", "10"))
WEB_SEARCH_FETCH_TOP = int(os.getenv("LITE_WEB_FETCH_TOP", "4"))
WEB_SEARCH_FETCH_TIMEOUT = int(os.getenv("LITE_WEB_FETCH_TIMEOUT", "10"))
WEB_SEARCH_SUMMARY_CHARS = int(os.getenv("LITE_WEB_SUMMARY_CHARS", "600"))
WEB_SEARCH_SEMANTIC_FILTER = os.getenv("LITE_WEB_SEMANTIC_FILTER", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
WEB_SEARCH_SEMANTIC_CANDIDATES = max(
    1, int(os.getenv("LITE_WEB_SEMANTIC_CANDIDATES", "8")),
)

# rag_search
RAG_MAX_RESULTS = int(os.getenv("LITE_RAG_MAX_RESULTS", "8"))
RAG_SNIPPET_CHARS = int(os.getenv("LITE_RAG_SNIPPET_CHARS", "300"))
# RAG is only a lightweight file locator. Detailed reading uses read_file.
RAG_INDEX_CONTENT_CHARS = 250

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
# The directory boundary is the security boundary for Lite runtime state.
# Keep it private even if the user's default umask is permissive.
for _private_dir in (DATA_DIR, LOG_DIR):
    try:
        _private_dir.chmod(0o700)
    except OSError:
        pass
