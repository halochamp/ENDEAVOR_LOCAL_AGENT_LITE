"""config.py — ENDEAVOR_LOCAL_AGENT_LITE configuration

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
import hashlib
import os
import platform
import pwd
import re
import socket
import ipaddress
from urllib.parse import urlsplit
from pathlib import Path
import sys


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}

# ── Model / server ──────────────────────────────────────────────────────
MODEL_ID = os.getenv("LITE_MODEL", "mlx-community/Qwen3.5-2B-OptiQ-4bit").strip() or "mlx-community/Qwen3.5-2B-OptiQ-4bit"

def _loopback_host(value: str) -> str:
    """Keep the model API local even when an inherited env is too broad."""
    candidate = (value or "127.0.0.1").strip()
    if candidate == "localhost":
        return candidate
    try:
        if ipaddress.ip_address(candidate).is_loopback:
            return candidate
    except ValueError:
        pass
    return "127.0.0.1"


MLX_SERVER_HOST = _loopback_host(os.getenv("LITE_SERVER_HOST", "127.0.0.1"))
# Deliberately NOT :8080/:8085 (MAX/MAX_VLM's ports on the Main Mac, per root
# CLAUDE.md §6.5) even though this runs on the Backup Mac where those aren't
# active — keeps the convention unambiguous if this ever runs alongside them.
MLX_SERVER_PORT = _env_int("LITE_SERVER_PORT", 8090, 1024, 65535)


def _local_base_url(value: str, host: str, port: int) -> str:
    default = f"http://{'[' + host + ']' if ':' in host else host}:{port}/v1"
    candidate = (value or default).strip()
    try:
        parsed = urlsplit(candidate)
        hostname = (parsed.hostname or "").casefold()
        is_loopback = hostname == "localhost" or ipaddress.ip_address(hostname).is_loopback
        if (
            parsed.scheme != "http"
            or not is_loopback
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            return default
        if parsed.port not in (None, port):
            return default
        # ChatOpenAI appends /chat/completions to this base.  A nested path
        # such as /proxy/v1 can make the UI probe the real local server while
        # the model client silently sends every request to a dead endpoint.
        if parsed.path.rstrip("/") != "/v1":
            return default
    except (ValueError, UnicodeError):
        return default
    return candidate.rstrip("/")


MLX_BASE_URL = _local_base_url(os.getenv("LITE_MLX_BASE_URL", ""), MLX_SERVER_HOST, MLX_SERVER_PORT)
_raw_mlx_api_key = os.getenv("LITE_MLX_API_KEY", "x") or "x"
# The key is local-only, but it still crosses an HTTP header and a subprocess
# argument. Reject control characters so a malformed inherited environment
# cannot break urllib/httpx header construction or inject a line break.
MLX_API_KEY = (
    _raw_mlx_api_key.strip()
    if _raw_mlx_api_key.strip()
    and all(
        ord(char) >= 0x20 and ord(char) != 0x7F and not char.isspace()
        for char in _raw_mlx_api_key
    )
    else "x"
)
# Global launch flag, not a per-request param — see the module docstring's
# PLAN.md §9.4 measurement (2026-07-25) for why 512 was picked.
MLX_SERVER_PREFILL_STEP_SIZE = _env_int("LITE_PREFILL_STEP_SIZE", 512, 1, 4096)
# Lite serializes heavy work and normally handles one image at a time.  The
# mlx-vlm default retains features for 20 distinct images, which is needless
# unified-memory pressure on the target 8GB Mac.  PLAN.md §9.3 records the
# live A/B that established 2 as the quality/latency-neutral production value.
MLX_SERVER_VISION_CACHE_SIZE = _env_int("LITE_VISION_CACHE_SIZE", 2, 0, 20)

# Browser UI is intentionally local-only.  It is a separate process from the
# model server, so it needs its own port.
# Do not make this environment-configurable: broadening to 0.0.0.0 would turn
# a local agent (with filesystem tools) into a LAN service.
UI_HOST = "127.0.0.1"
UI_PORT = _env_int("LITE_UI_PORT", 8766, 1024, 65535)
if UI_PORT == MLX_SERVER_PORT:
    UI_PORT = 8767 if UI_PORT == 8766 else 8766

TEMPERATURE = _env_float("LITE_TEMPERATURE", 0.1, 0.0, 2.0)
MAX_TOKENS = _env_int("LITE_MAX_TOKENS", 768, 64, 8192)
# Keep reasoning bounded inside MAX_TOKENS: 256 leaves roughly the previous
# 512-token visible-answer budget while adding explicit routing/reasoning room.
# The same budget applies to tool-bound turns: choosing a tool, constructing
# safe arguments, and interpreting its result must not fall back to a
# non-thinking graph.
ENABLE_THINKING = os.getenv("LITE_ENABLE_THINKING", "1").lower() in ("1", "true", "yes", "on")
THINKING_BUDGET = _env_int("LITE_THINKING_BUDGET", 256, 0, 4096)
# The UI exposes these bounded profiles instead of accepting an arbitrary
# per-request token count.  Medium preserves the existing production default;
# the other profiles are derived from it so an operator changing
# LITE_THINKING_BUDGET still gets a coherent Low/Medium/High set.
THINKING_EFFORTS = ("low", "medium", "high")
DEFAULT_THINKING_EFFORT = "medium"
THINKING_BUDGET_PRESETS = {
    "low": max(0, THINKING_BUDGET // 2),
    "medium": THINKING_BUDGET,
    "high": min(4096, THINKING_BUDGET * 2),
}


def normalize_thinking_effort(value: str | None) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in THINKING_EFFORTS else DEFAULT_THINKING_EFFORT


def thinking_budget_for_effort(value: str | None) -> int:
    return THINKING_BUDGET_PRESETS[normalize_thinking_effort(value)]


# Ported from ENDEAVOR_LOCAL_AGENT_MAX_VLM/config.py (2026-07-24, live
# finding): without this, Qwen3.5-2B occasionally degenerates into massive
# token-level repetition on longer tool-augmented turns (observed live:
# a numbered list looping the same 4 bullet points 30+ times instead of
# stopping) -- reproducible on this exact model/server, not just MAX_VLM's
# 35B. MAX_VLM's comment: "1.05->1.1: breaks repetitive thinking loops
# faster." Passed via extra_body since it's an mlx_vlm.server-specific
# sampling param, not a standard OpenAI Chat Completions field.
REPETITION_PENALTY = _env_float("LITE_REPETITION_PENALTY", 1.1, 1.0, 2.0)

# ── Paths / runtime identity ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT
RAG_CODE_DIR = (PROJECT_ROOT / "MCP-RagDoc").resolve()

# Machine hints are derived from the running process/account database rather
# than embedding either Mac's username.  This keeps the same code correct on
# the Main and Backup Macs and protects the 2B model from guessing a familiar
# but wrong /Users/<name>/... path.
try:
    _runtime_account = pwd.getpwuid(os.getuid())
    RUNTIME_USER = _runtime_account.pw_name
    RUNTIME_HOME = Path(_runtime_account.pw_dir).expanduser().resolve()
    RUNTIME_IDENTITY_SOURCE = "account database"
except (KeyError, OSError):
    RUNTIME_USER = os.environ.get("USER", "unknown")
    RUNTIME_HOME = Path.home().resolve()
    RUNTIME_IDENTITY_SOURCE = "process fallback"
RUNTIME_OS = platform.system() or os.name
RUNTIME_OS_VERSION = platform.mac_ver()[0] if RUNTIME_OS == "Darwin" else platform.release()
RUNTIME_ARCH = platform.machine() or "unknown"
RUNTIME_USERLAND = "macOS/BSD" if RUNTIME_OS == "Darwin" else RUNTIME_OS
RUNTIME_DESKTOP = (RUNTIME_HOME / "Desktop").resolve()
RUNTIME_DOCUMENTS = (RUNTIME_HOME / "Documents").resolve()

# Generic user-file content search is deliberately narrower than a whole-home
# scan.  These roots are derived from the live account on each Mac, so the same
# code works for the Main and Backup identities without embedding a username.
LOCAL_SEARCH_ROOTS = (RUNTIME_DESKTOP, RUNTIME_DOCUMENTS)
LOCAL_SEARCH_MAX_RESULTS = _env_int("LITE_LOCAL_SEARCH_MAX_RESULTS", 20, 1, 50)
LOCAL_SEARCH_MAX_FILES = _env_int("LITE_LOCAL_SEARCH_MAX_FILES", 3000, 100, 20_000)
LOCAL_SEARCH_TIMEOUT_SECONDS = _env_int("LITE_LOCAL_SEARCH_TIMEOUT", 8, 1, 60)
LOCAL_SEARCH_MAX_PLAIN_BYTES = _env_int("LITE_LOCAL_SEARCH_MAX_PLAIN_BYTES", 10 * 1024 * 1024, 100_000, 100 * 1024 * 1024)
LOCAL_SEARCH_SNIPPET_CHARS = _env_int("LITE_LOCAL_SEARCH_SNIPPET_CHARS", 360, 120, 2_000)


def runtime_environment_hint() -> str:
    """Compact machine/path hint safe to repeat after a failed bash call."""
    return (
        f"runtime ที่ตรวจจริง: {RUNTIME_USERLAND}, arch={RUNTIME_ARCH}, "
        f"account={RUNTIME_USER}, Home={RUNTIME_HOME}, Desktop={RUNTIME_DESKTOP}, "
        f"Documents={RUNTIME_DOCUMENTS}, "
        f"Project={PROJECT_ROOT}. ห้ามเดา username/path อื่น; "
        "ใช้ path เหล่านี้และ syntax/flags ที่รองรับระบบนี้เมื่อ retry"
    )

# Lite may create user-facing files only in this explicit workspace. Runtime
# databases, logs, locks, OCR snapshots, and uploaded attachments live in the
# per-machine Application Support directory below so iCloud never syncs an
# in-use SQLite/process state between Macs. A workspect state override remains
# accepted for isolated legacy/test runs, but is not the production default.
WORKSPECT_DIR = (PROJECT_ROOT / "workspect").resolve()
WORKSPECT_DIR.mkdir(parents=True, exist_ok=True)

# The Web UI launches the model server with the same Python environment that
# runs Agent Lite by default. Override LITE_MLX_PYTHON only when the MLX runtime
# is deliberately installed in another interpreter.
MLX_SERVER_PYTHON = Path(os.getenv("LITE_MLX_PYTHON", sys.executable)).expanduser().absolute()

STATE_PARENT_DIR = RUNTIME_HOME / "Library" / "Application Support"
STATE_DIR_NAME = "ENDEAVOR_LOCAL_AGENT_LITE"
# The project used to be called ENDEAVOR_AGENT_LITE. An install that already holds
# data under that name keeps using it (nothing is moved or deleted); new installs
# use STATE_DIR_NAME.
PRE_RENAME_STATE_DIR_NAME = "ENDEAVOR_AGENT_LITE"


def _choose_default_data_dir(parent: Path) -> Path:
    """Pick the per-machine state directory under ``parent`` (see above)."""
    current = (parent / STATE_DIR_NAME).resolve()
    previous = (parent / PRE_RENAME_STATE_DIR_NAME).resolve()
    return previous if previous.is_dir() and not current.exists() else current


DEFAULT_DATA_DIR = _choose_default_data_dir(STATE_PARENT_DIR)
LEGACY_DATA_DIR = (WORKSPECT_DIR / ".lite_state").resolve()
# Both names live under the same Application Support parent, so either is a valid
# explicit LITE_DATA_DIR / LITE_FILE_INDEX_DB target.
_STATE_ROOTS = (
    DEFAULT_DATA_DIR,
    (STATE_PARENT_DIR / STATE_DIR_NAME).resolve(),
    (STATE_PARENT_DIR / PRE_RENAME_STATE_DIR_NAME).resolve(),
    WORKSPECT_DIR,
)


def _state_path_allowed(path: Path) -> bool:
    """Accept local per-machine state, plus explicit legacy/test state paths."""
    try:
        resolved = path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    for root in _STATE_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


_default_data_dir = DEFAULT_DATA_DIR
_requested_data_dir = Path(os.getenv("LITE_DATA_DIR", str(_default_data_dir))).expanduser()
try:
    _requested_data_dir = _requested_data_dir.resolve()
    DATA_DIR = _requested_data_dir if _state_path_allowed(_requested_data_dir) else _default_data_dir
except (OSError, RuntimeError, ValueError):
    DATA_DIR = _default_data_dir
# MCP calls run the deterministic RAG service in a child process. The optional
# override lets the trusted parent pass its already-selected per-run index to
# that child (notably for isolated live/test corpora) while retaining the
# local-state/workspect boundary above.
_default_file_index_db = DATA_DIR / "file_index.sqlite3"
_requested_file_index_db = Path(
    os.getenv("LITE_FILE_INDEX_DB", str(_default_file_index_db))
).expanduser()
try:
    _requested_file_index_db = _requested_file_index_db.resolve()
    FILE_INDEX_DB = (
        _requested_file_index_db
        if _state_path_allowed(_requested_file_index_db)
        else _default_file_index_db
    )
except (OSError, RuntimeError, ValueError):
    FILE_INDEX_DB = _default_file_index_db
WEB_CACHE_DB = DATA_DIR / "web_cache.sqlite3"
PDF_TO_TEXT_DB = DATA_DIR / "pdf_to_text.sqlite3"
LOG_DIR = DATA_DIR / "logs"

# Knowledge-base paths. /sync_kb indexes supported source documents anywhere
# below the root-level rag_document/ directory. rag_document/md remains the
# default root for the legacy /index compatibility path.
# KNOWLEDGE_RAW_DIR/RAW_DIR are compatibility aliases for older callers/tests;
# the production sync walks the configured source root recursively.
KNOWLEDGE_DIR = (PROJECT_ROOT / "rag_document").resolve()
KNOWLEDGE_SOURCE_DIR = KNOWLEDGE_DIR
KNOWLEDGE_RAW_DIR = KNOWLEDGE_SOURCE_DIR
KNOWLEDGE_MD_DIR = KNOWLEDGE_DIR / "md"
RAW_DIR = KNOWLEDGE_SOURCE_DIR
MD_DIR = KNOWLEDGE_MD_DIR
# Kept as a compatibility constant for older callers.  It is runtime
# bookkeeping, not user knowledge, so it must never be placed in the
# iCloud-synced RAG document tree where the two Macs could overwrite it.
KB_SYNC_MANIFEST = DATA_DIR / "sync_kb_manifest.json"

# MCP client/server boundary. Agent Lite provisions only its own deterministic
# local RAG server; unlike MAX_VLM, Lite intentionally does not let the 2B model
# self-register arbitrary servers. The stdio child runs from a private workspace
# while the developer-provisioned sandbox capability grants only the state/KB
# paths needed by sync_kb.
MCP_WORKSPACE = DATA_DIR / "mcp_workspace"
MCP_PYTHON = Path(os.getenv("LITE_MCP_PYTHON", str(MLX_SERVER_PYTHON))).expanduser()
MCP_SERVERS: dict[str, dict] = {
    "lite-rag": {
        "transport": "stdio",
        "command": str(MCP_PYTHON),
        "args": [str(RAG_CODE_DIR / "rag_server.py")],
        "cwd": str(MCP_WORKSPACE),
        # Search needs only read access to the RAG document root. sync_kb writes its
        # SQLite/temp state under DATA_DIR and must never gain write access to
        # source documents.
        "sandbox_write_paths": [str(DATA_DIR)],
    },
}
MCP_MAX_CHARS = _env_int("LITE_MCP_MAX_CHARS", 8000, 500, 100_000)
MCP_TIMEOUT = _env_int("LITE_MCP_TIMEOUT", 300, 1, 600)

# In-process compatibility contract for the bundled standalone MCP-RagDoc.
# The stdio child loads MCP-RagDoc/config.py itself; legacy Lite imports keep
# host config priority so these names mirror the RagDoc defaults explicitly.
LLM_MODEL = os.getenv("RAGDOC_LLM_MODEL", MODEL_ID).strip() or MODEL_ID
LLM_BASE_URL = MLX_BASE_URL
LLM_CONTROL_URL = f"http://127.0.0.1:{UI_PORT}"
LLM_CONTROL_ORIGIN = LLM_CONTROL_URL
LLM_CONTROL_TIMEOUT_SECONDS = _env_float("RAGDOC_LLM_CONTROL_TIMEOUT", 5.0, 1.0, 30.0)
LLM_SERVER_START_TIMEOUT_SECONDS = _env_int("RAGDOC_LLM_START_TIMEOUT", 180, 10, 600)
LLM_API_KEY = MLX_API_KEY
LLM_REPETITION_PENALTY = _env_float("RAGDOC_LLM_REPETITION_PENALTY", REPETITION_PENALTY, 1.0, 2.0)
OCR_REWRITE_ENABLED = _env_bool("RAGDOC_OCR_REWRITE", True)
OCR_REWRITE_BATCH_PAGES = _env_int("RAGDOC_OCR_REWRITE_BATCH_PAGES", 1, 1, 1)
OCR_REWRITE_BATCH_CHARS = _env_int("RAGDOC_OCR_REWRITE_BATCH_CHARS", 9_000, 1_000, 20_000)
OCR_REWRITE_TIMEOUT_SECONDS = _env_int("RAGDOC_OCR_REWRITE_TIMEOUT", 90, 10, 300)
OCR_REWRITE_MAX_TOKENS = _env_int("RAGDOC_OCR_REWRITE_MAX_TOKENS", 8_192, 256, 8_192)
OCR_REWRITE_IMAGE_DPI = _env_int("RAGDOC_OCR_REWRITE_IMAGE_DPI", 300, 150, 600)
OCR_REWRITE_AUTO_START = _env_bool("RAGDOC_OCR_AUTO_START", True)
OCR_REWRITE_AUTO_STOP = _env_bool("RAGDOC_OCR_AUTO_STOP", True)

# The normal runtime has exactly one local RAG root: documents under
# rag_document.  ``LITE_INDEX_ROOTS`` remains a narrow, explicit compatibility
# override for isolated test/developer runs; /sync_kb never uses it and always
# indexes KNOWLEDGE_MD_DIR.  No user request can broaden the default scope.
_DEFAULT_INDEX_ROOTS = [KNOWLEDGE_MD_DIR]
_configured_index_roots = os.getenv("LITE_INDEX_ROOTS")
INDEX_ROOTS = (
    [Path(p).expanduser() for p in _configured_index_roots.split(",") if p.strip()]
    if _configured_index_roots is not None
    else _DEFAULT_INDEX_ROOTS
)
# In-process compatibility only for Lite's historical /index path.  The
# standalone MCP-RagDoc child loads its own config where this defaults false,
# so normal MCP search remains scoped to direct rag_document/ documents.
ALLOW_INDEX_ROOT_RESULTS = True

# ── Tool / loop guardrails (see PLAN.md §6) ────────────────────────────────
MAX_TOOL_STEPS = _env_int("LITE_MAX_TOOL_STEPS", 10, 1, 50)
MAX_REPEAT_CALLS = _env_int("LITE_MAX_REPEAT_CALLS", 2, 1, 10)  # same (tool,args) hash

# Per-tool call quota per turn, independent of exact args. Changed commands or
# queries may continue up to this bounded ceiling; exact consecutive repeats
# are stopped much earlier by RepeatCallGuard. This gives the small model room
# for genuine recovery without permitting a same-command loop.
MAX_CALLS_PER_TOOL = _env_int("LITE_MAX_CALLS_PER_TOOL", 10, 1, 50)
# Web search is the one tool where a small model can emit many materially
# different guesses (for example, a sequence of future dates) without making
# progress. Keep its retry budget below the general tool ceiling; the general
# ceiling still wins when an operator deliberately lowers it for a test/run.
MAX_WEB_CALLS_PER_TURN = _env_int("LITE_MAX_WEB_CALLS_PER_TURN", 4, 1, 50)
TOOL_OUTPUT_TRUNCATE_CHARS = _env_int("LITE_TOOL_TRUNCATE", 2000, 100, 100_000)
BASH_TIMEOUT_SECONDS = _env_int("LITE_BASH_TIMEOUT", 10, 1, 120)
INDEX_TIMEOUT_SECONDS = _env_int("LITE_INDEX_TIMEOUT", 1800, 30, 86_400)

# All agent-created files and bash's implicit working directory stay inside workspect/, the
# only writable workspace directly beneath the LITE root. write_file resolves
# paths before checking this boundary, preventing direct and symlink traversal.
BASH_ALLOWED_ROOT = WORKSPECT_DIR

# read_file / tool-output size caps (keep context small -> keep KV cache small)
READ_FILE_MAX_CHARS = 5_000
# One tool call may read at most four known knowledge documents concurrently.
READ_FILE_MAX_FILES = 4
# This is a security ceiling, not a performance preference.  Do not make it
# environment-overridable: a launch environment must not be able to turn an
# image tool into an unbounded file reader.
READ_IMAGE_MAX_BYTES = 10 * 1024 * 1024

# Direct PDF -> text panel.  Conversion is page-streamed through SQLite so the
# source PDF is never materialized as one giant Python string.  Thai OCR cleanup
# uses one no-think model request for at most four pages and is always optional;
# native PDF text and reconstructed tables never need the LLM.
PDF_TO_TEXT_MAX_BYTES = 250 * 1024 * 1024
PDF_TO_TEXT_OCR_DPI = _env_int("LITE_PDF_OCR_DPI", 180, 120, 300)
# Hard raster ceiling: a malicious/odd PDF page must not expand into an
# unbounded bitmap on an 8GB machine. Normal A4/Letter pages at 180 dpi are
# only ~4-5M pixels, comfortably below this limit.
PDF_TO_TEXT_OCR_MAX_PIXELS = 20_000_000
PDF_TO_TEXT_NATIVE_MIN_CHARS = _env_int("LITE_PDF_NATIVE_MIN_CHARS", 40, 0, 500)
PDF_TO_TEXT_LLM_BATCH_PAGES = _env_int("LITE_PDF_LLM_BATCH_PAGES", 4, 1, 8)
PDF_TO_TEXT_LLM_BATCH_CHARS = _env_int("LITE_PDF_LLM_BATCH_CHARS", 6000, 1000, 16000)
PDF_TO_TEXT_LLM_TIMEOUT_SECONDS = _env_int("LITE_PDF_LLM_TIMEOUT", 90, 10, 300)
PDF_TO_TEXT_CACHE_MAX_AGE_SECONDS = _env_int("LITE_PDF_CACHE_MAX_AGE", 86400, 3600, 604800)

# web_search
WEB_SEARCH_MAX_RESULTS = _env_int("LITE_WEB_MAX_RESULTS", 10, 1, 50)
WEB_SEARCH_FETCH_TOP = _env_int("LITE_WEB_FETCH_TOP", 3, 1, 3)
WEB_SEARCH_FETCH_TIMEOUT = _env_int("LITE_WEB_FETCH_TIMEOUT", 10, 1, 120)
WEB_SEARCH_SUMMARY_CHARS = _env_int("LITE_WEB_SUMMARY_CHARS", 600, 100, 10_000)

# rag_search
RAG_MAX_RESULTS = _env_int("LITE_RAG_MAX_RESULTS", 8, 1, 50)
RAG_SNIPPET_CHARS = _env_int("LITE_RAG_SNIPPET_CHARS", 300, 50, 10_000)

# Optional semantic retrieval keeps Lite's normal process free of ML runtimes.
# A short-lived worker imports ONNX Runtime + NumPy + SentencePiece only while
# building/querying vectors, then exits and returns that RAM to macOS. The
# official ARM64 INT8 MiniLM artifact is stored in the ordinary Hugging Face
# cache beside the local Qwen model caches; no model download is allowed from
# a chat/search request.
SEMANTIC_ENABLED = os.getenv("LITE_SEMANTIC", "1").lower() in ("1", "true", "yes", "on")
SEMANTIC_MODEL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_MODEL_ONNX = "onnx/model_qint8_arm64.onnx"
SEMANTIC_SENTENCEPIECE = "sentencepiece.bpe.model"
SEMANTIC_POLICY_ID = "onnx-int8-arm64:v3-generation-word-boundary"
SEMANTIC_DIM = 384
# MiniLM has a 128-token input ceiling. Reserve CLS+SEP and use smaller
# overlapping children so a 3,000-char lexical parent is never silently
# truncated into one semantic vector. Semantic windows prefer whole-word
# boundaries (Thai newmm when available) and hard-split only an oversized word.
SEMANTIC_CHILD_TOKENS = _env_int("LITE_SEMANTIC_CHILD_TOKENS", 112, 32, 126)
SEMANTIC_CHILD_OVERLAP_TOKENS = min(
    SEMANTIC_CHILD_TOKENS // 2,
    _env_int("LITE_SEMANTIC_CHILD_OVERLAP_TOKENS", 24, 0, SEMANTIC_CHILD_TOKENS // 2),
)
SEMANTIC_EMBED_BATCH = _env_int("LITE_SEMANTIC_EMBED_BATCH", 16, 1, 64)
SEMANTIC_SCAN_BATCH = _env_int("LITE_SEMANTIC_SCAN_BATCH", 256, 32, 2048)
SEMANTIC_CANDIDATES = _env_int("LITE_SEMANTIC_CANDIDATES", 24, 4, 100)
SEMANTIC_TIMEOUT_SECONDS = _env_int("LITE_SEMANTIC_TIMEOUT", 20, 5, 120)
SEMANTIC_SYNC_TIMEOUT_SECONDS = _env_int("LITE_SEMANTIC_SYNC_TIMEOUT", 600, 30, 3600)
# Host-config mirror of the bundled MCP-RagDoc defaults. Host config has
# priority for in-process imports, so every name MCP-RagDoc reads from `config`
# must exist here too. Up to SEMANTIC_FULL_SCAN_LIMIT vectors use streaming exact
# cosine; larger generations may use the optional, rebuildable HNSW sidecar (only
# if `hnswlib` is installed — otherwise search keeps working without it).
SEMANTIC_THREADS = _env_int("LITE_SEMANTIC_THREADS", 4, 1, 16)
SEMANTIC_FULL_SCAN_LIMIT = _env_int("LITE_SEMANTIC_FULL_SCAN_LIMIT", 20_000, 1_000, 1_000_000)
SEMANTIC_ANN_CANDIDATES = _env_int("LITE_SEMANTIC_ANN_CANDIDATES", 256, 64, 5_000)
SEMANTIC_ANN_SIDECAR_SCHEMA = 2


def semantic_ann_sidecar_dir(db_path: Path) -> Path:
    """Return the machine/database-specific rebuildable ANN state directory."""
    resolved = db_path.resolve()
    machine = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", resolved.stem)[:48] or "ragdoc"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    return resolved.parent / "ann" / machine / f"{stem}-{digest}"


def semantic_ann_sidecar_paths(db_path: Path) -> tuple[Path, Path, Path]:
    root = semantic_ann_sidecar_dir(db_path)
    return root / "vectors.hnsw", root / "metadata.json", root / "state.json"

# /sync_kb stores the full searchable corpus as bounded overlapping chunks.
# Character-based chunking avoids adding a tokenizer/embedding dependency to Lite.
KB_CHUNK_TARGET_CHARS = _env_int("LITE_KB_CHUNK_CHARS", 3000, 1000, 100_000)
KB_CHUNK_OVERLAP_CHARS = min(
    KB_CHUNK_TARGET_CHARS // 2,
    _env_int("LITE_KB_CHUNK_OVERLAP_CHARS", 250, 0, KB_CHUNK_TARGET_CHARS // 2),
)
# Legacy /index compatibility still uses a compact one-row-per-file excerpt.
RAG_INDEX_CONTENT_CHARS = _env_int("LITE_RAG_INDEX_CONTENT_CHARS", 1200, 100, 100_000)

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_MD_DIR.mkdir(parents=True, exist_ok=True)
MCP_WORKSPACE.mkdir(parents=True, exist_ok=True)
# The directory boundary is the security boundary for Lite runtime state.
# Keep it private even if the user's default umask is permissive.
for _private_dir in (DATA_DIR, LOG_DIR, KNOWLEDGE_DIR, KNOWLEDGE_MD_DIR, MCP_WORKSPACE):
    try:
        _private_dir.chmod(0o700)
    except OSError:
        pass
