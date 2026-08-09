# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""tools/bash.py — trimmed from ENDEAVOR_LOCAL_AGENT_MAX_VLM/tools/bash.py.

Kept: deny-list against destructive commands, timeout with partial-output
capture, output truncation with full-body spilled to disk.
Dropped: MAX_VLM's broader shell surface and workspace-write profile.
LITE keeps a read-only command allow-list plus a kernel sandbox. Relative
paths intentionally resolve inside BASH_ALLOWED_ROOT (the private workspect);
the system prompt therefore requires absolute paths for project inspection.
"""
from __future__ import annotations
import os
import re
import shlex
import signal
import selectors
import subprocess
import tempfile
import time
from pathlib import Path

import config
from helpers import search_scope
from helpers.guardrails import truncate_output
from helpers.save_output import save_full_output
from helpers.sensitive_paths import contains_secret_content, is_sensitive_path, path_is_or_contains_secret

SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        # 2026-07-23: shortening this description (RAM reduction attempt B)
        # caused a reproducible routing regression -- 3/3 runs called bash
        # with a relative find path instead of rag_search, something the
        # full description below never did in extensive prior testing. Root
        # cause not fully isolated (a chunked-prefill A/B split was
        # inconclusive, 1 pass/1 fail), so reverted rather than risk it;
        # config.PREFILL_STEP_SIZE (approach A) and agent.py's
        # mx.clear_cache() (approach D) are kept -- both are code-level
        # changes with no model-facing text, so no routing risk from them.
        "description": (
            "Run a read-only-oriented bash command (find, grep, ls, cat, wc, etc.) to "
            "search or inspect files/processes on this Mac. Destructive commands "
            "(rm, sudo, disk-wiping, etc.) are blocked. Project file operands must "
            "use absolute paths; relative paths resolve inside the private workspect."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to run"}},
            "required": ["command"],
        },
    },
}

_DENY_PATTERNS = [
    # Bug fixed 2026-07-23 (bug-review pass): the old pattern
    # r"\brm\s+-[a-z]*r[a-z]*f" required 'r' to appear before 'f' in the SAME
    # flag group, so `rm -fr`, `rm -fR`, `rm -r -f` (separate flags), and
    # `rm --recursive --force` all bypassed the deny-list — verified live
    # (all 4 return False against the old pattern, True against this one).
    # Lookaheads check that a recursive-ish flag AND a force-ish flag both
    # appear somewhere after `rm` on the same shell statement (stops at
    # ;/&/| so it doesn't bleed into an unrelated later command).
    re.compile(
        r"\brm\b(?=[^;&|\n]*(?:-[a-zA-Z]*r[a-zA-Z]*\b|--recursive\b))"
        r"(?=[^;&|\n]*(?:-[a-zA-Z]*f[a-zA-Z]*\b|--force\b))",
        re.I,
    ),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdiskutil\s+(erase|reformat|partitiondisk)", re.I),
    re.compile(r">\s*/dev/(disk|rdisk|sd)"),
    re.compile(r"\bdd\s+.*of=/dev"),
    re.compile(r"\bchmod\s+-R\s+777\s+/"),
    re.compile(r":\(\)\s*\{.*:\|:.*\};:"),  # fork bomb
    re.compile(r"\b(shutdown|reboot|halt)\b"),
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh\b"),  # pipe-to-shell
]

# This tool is intentionally an inspection/search tool, not a general shell.
# An allow-list is the primary policy; deny patterns remain explanatory
# defense-in-depth for a clearer error when the model proposes a known hazard.
#
# 2026-07-23 (user request: port bash from ENDEAVOR_LOCAL_AGENT_MAX, "the
# parts necessary for lite"): MAX's bash.py uses a much looser model --
# allow-default plus a sandbox-exec profile that only denies WRITES outside
# workspace, with no allow-list of specific executables at all (it documents
# osascript notifications, pbpaste/pbcopy, network scan tools like nmap).
# Porting that wholesale would regress lite's stricter posture (read-only
# allow-list + kernel-level deny of credential directories + deny-network
# entirely, all already in place before this port). Brought over only what
# fits that existing scope: `mdfind` (macOS Spotlight -- fast, whole-disk,
# read-only, MAX's own docstring recommends it as the first thing to try for
# file search) and the evidence-gated error-hint pattern below
# (_classify_bash_error). Deliberately NOT ported: app/clipboard control,
# GUI notifications, network diagnostic tools (ping/netstat/nmap) -- outside
# the "search/inspect files" scope this tool's own SCHEMA description
# commits to; ask before adding if broader capability is actually wanted.
_READ_ONLY_COMMANDS = {
    "cat", "date", "df", "du", "echo", "find", "git", "grep", "head", "hostname", "id", "ls",
    "mdfind", "pwd", "rg", "sort", "sw_vers", "tail", "uname", "uptime", "wc", "whoami",
}
_SHELL_META_RE = re.compile(r"[;&><`$()\\\n]")
_FIND_MUTATING_PREFIXES = ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fls", "-fprint", "-fprintf")
_SIMPLE_SEARCH_FLAGS = {
    "-n", "-i", "-r", "-R", "-l", "--line-number", "--ignore-case", "--recursive",
    "--fixed-strings", "--files", "--files-with-matches",
}
_READ_ONLY_GIT_SUBCOMMANDS = {"branch", "diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}

_FIND_PATH_RE = re.compile(r"\bfind\s+(\S+)")
_GREP_R_PATH_RE = re.compile(r"\bgrep\s+-[a-zA-Z]*[rR][a-zA-Z]*\s+\S+\s+(\S+)")


def _is_denied(command: str) -> str | None:
    for pat in _DENY_PATTERNS:
        if pat.search(command):
            return f"คำสั่งนี้ถูกบล็อก (ตรงกับ pattern อันตราย: {pat.pattern})"
    return None


def _read_only_violation(command: str) -> str | None:
    """Return a policy error unless every pipeline stage is read-only.

    The model never needs general shell programming for this tool. Rejecting
    substitutions, redirects, chaining, and unknown executables makes the
    policy auditable; sandbox-exec below is the independent OS-level backstop.
    """
    if _SHELL_META_RE.search(command):
        return "ไม่อนุญาต shell syntax ที่อาจเปลี่ยนไฟล์หรือรันคำสั่งแฝง (redirect, chain, substitution หรือ newline)"
    try:
        stages = [shlex.split(stage) for stage in command.split("|")]
    except ValueError:
        return "รูปแบบ shell ไม่ถูกต้องหรือมี quote ไม่ครบ"
    if not stages or any(not stage for stage in stages):
        return "pipeline ต้องมีคำสั่งอ่านข้อมูลทุกช่วง"
    for tokens in stages:
        if "/" in tokens[0]:
            return "ต้องเรียกเฉพาะ executable ระบบที่อยู่ใน allow-list ห้ามรัน path หรือไฟล์ executable โดยตรง"
        executable = os.path.basename(tokens[0])
        if executable not in _READ_ONLY_COMMANDS:
            return f"ไม่อนุญาตคำสั่ง '{executable}' — bash ของ LITE รองรับเฉพาะการอ่าน/ค้นหา"
        if any(t.startswith("/dev/") for t in tokens[1:]):
            return "ไม่อนุญาตให้อ่าน special device ใต้ /dev เพราะอาจทำให้ output/ทรัพยากรไม่จำกัด"
        if executable == "find" and any(t.startswith(_FIND_MUTATING_PREFIXES) for t in tokens[1:]):
            return "find แบบที่สั่ง execute/delete/write ถูกบล็อก"
        if executable == "rg" and any(t == "--pre" or t.startswith("--pre=") for t in tokens[1:]):
            return "rg --pre ถูกบล็อก เพราะสามารถรัน executable ภายนอกได้"
        if executable == "sort" and any(
            t == "-o" or t.startswith("-o") or t == "--output" or t.startswith("--output=")
            or t == "--compress-program" or t.startswith("--compress-program=")
            for t in tokens[1:]
        ):
            return "sort ใช้ได้เฉพาะส่งผลลัพธ์ออก stdout ห้ามเขียนไฟล์หรือเรียกโปรแกรมบีบอัด"
        if executable == "hostname" and len(tokens) > 1:
            return "hostname ใช้ได้เฉพาะอ่านชื่อเครื่อง ห้ามตั้งค่า"
        if executable == "date" and any(re.fullmatch(r"\d{8,14}", t) for t in tokens[1:]):
            return "date ใช้ได้เฉพาะอ่านเวลา ห้ามตั้งค่า"
        if executable == "git" and (violation := _git_violation(tokens[1:])):
            return violation
        violation = _file_read_violation(executable, tokens[1:])
        if violation:
            return violation
    return None


def _safe_operand(path_text: str) -> str | None:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = config.BASH_ALLOWED_ROOT / path
    path = path.resolve(strict=False)
    if is_sensitive_path(path) or path_is_or_contains_secret(path):
        return f"ไม่อนุญาตให้อ่าน path ที่อาจมี credential, token, key หรือ session: {path}"
    return None


def _file_read_violation(executable: str, args: list[str]) -> str | None:
    """Validate file operands for the intentionally small read-only grammar."""
    if executable in {"cat", "head", "tail", "ls"}:
        for arg in args:
            if arg.startswith("-") or arg.isdigit():
                continue
            if violation := _safe_operand(arg):
                return violation
    elif executable == "find":
        roots = [arg for arg in args if not arg.startswith("-")]
        if roots and (violation := _safe_operand(roots[0])):
            return violation
    elif executable == "mdfind":
        # `-live` never terminates (Spotlight watches for new matches
        # indefinitely) -- would hang until config.BASH_TIMEOUT_SECONDS,
        # same failure shape the timeout/capped-output path already handles,
        # but there's no reason to let a small model discover that the hard
        # way. `-onlyin <path>` is the only operand that names a filesystem
        # location; bare query terms carry no path risk.
        if "-live" in args:
            return "mdfind -live ไม่อนุญาต เพราะรันไม่จบ (เฝ้าดูผลลัพธ์ใหม่ตลอด)"
        if "-onlyin" in args:
            idx = args.index("-onlyin")
            if idx + 1 < len(args) and (violation := _safe_operand(args[idx + 1])):
                return violation
    elif executable in {"grep", "rg"}:
        if any(arg.startswith("-") and arg not in _SIMPLE_SEARCH_FLAGS for arg in args):
            return "grep/rg อนุญาตเฉพาะ flags อ่านข้อมูลพื้นฐาน (-n, -i, -r/-R, fixed-strings)"
        operands = [arg for arg in args if not arg.startswith("-")]
        # `rg --files <root>` has no search pattern: every operand is a path.
        paths = operands if executable == "rg" and "--files" in args else operands[1:]
        # For ordinary grep/rg, the first operand is the search pattern and
        # remaining operands are paths.
        for path_text in paths:
            if violation := _safe_operand(path_text):
                return violation
    return None


def _git_violation(args: list[str]) -> str | None:
    # Permit `git -C <repo> <read-only-subcommand>` so the agent can inspect
    # repositories elsewhere on the machine without making the source tree a
    # write target.  All other global options are rejected deliberately.
    if args[:1] == ["-C"]:
        if len(args) < 3:
            return "git -C ต้องตามด้วย path และ read-only subcommand"
        if violation := _safe_operand(args[1]):
            return violation
        args = args[2:]
    if not args or args[0].startswith("-"):
        return "git ต้องเริ่มด้วย read-only subcommand ที่อนุญาต"
    subcommand = args[0]
    if subcommand not in _READ_ONLY_GIT_SUBCOMMANDS:
        return "git อนุญาตเฉพาะ branch/diff/grep/log/ls-files/rev-parse/show/status"
    options = args[1:]
    if any(
        opt in {"--ext-diff", "--textconv", "--output", "-o", "--no-optional-locks", "-O", "--open-files-in-pager"}
        or opt.startswith("--output=")
        or opt.startswith("-O")
        or opt.startswith("--open-files-in-pager=")
        for opt in options
    ):
        return "git option ที่อาจรันโปรแกรมภายนอกหรือเขียน output ถูกบล็อก"
    if subcommand == "branch":
        blocked_branch_options = {"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy", "--edit-description", "--set-upstream-to", "--unset-upstream"}
        if any(opt in blocked_branch_options or opt.startswith("--set-upstream-to=") for opt in options):
            return "git branch อนุญาตเฉพาะดูรายการ branch; option เปลี่ยน Git metadata ถูกบล็อก"
        if any(not opt.startswith("-") for opt in options):
            return "git branch อนุญาตเฉพาะดูรายการ branch โดยไม่มีชื่อ branch เพิ่ม"
    return None


# find/grep/rg over a real project directory otherwise surface build/cache/
# log noise ahead of the actual source match within the tool's output-size
# cap -- observed live 2026-07-24: `grep -r -l trigram <project>` put
# __pycache__/.pyc, .pytest_cache, and this tool's own logs/ ahead of the one
# real source-code match (indexer.py), and the model never saw it because
# the noise filled the truncation window first. This must be deterministic,
# not left to the model to notice and filter after the cutoff already ate
# the signal (training guide: behavior needing 100% correctness -> code).
_NOISE_PATH_MARKERS = ("__pycache__", ".pytest_cache", "/.lite_state/", "/logs/", "/.git/", ".pyc")


def _is_noise_path(line: str) -> bool:
    return any(marker in line for marker in _NOISE_PATH_MARKERS)


def _sanitize_read_output(executable: str, output: str) -> str:
    """Never disclose secret-shaped text; hide sensitive paths from listings."""
    if contains_secret_content(output):
        return "[blocked] ผลลัพธ์มีข้อมูลที่มีรูปแบบคล้าย credential, token หรือ private key"
    if executable in {"find", "ls", "grep", "rg"}:
        kept = [
            line for line in output.splitlines()
            if not is_sensitive_path(line.strip()) and not _is_noise_path(line)
        ]
        return "\n".join(kept) or "(no non-sensitive output)"
    return output


def _build_readonly_sandbox_profile() -> str:
    """Kernel-enforced fallback for anything an allow-list parser misses."""
    home = os.path.realpath(os.path.expanduser("~"))
    blocked_dirs = sorted(_SENSITIVE_SANDBOX_DIRS(home))
    denied_reads = " ".join(f'(subpath "{path}")' for path in blocked_dirs)
    return f'''(version 1)
(allow default)
(deny file-write*)
(deny file-read*
  {denied_reads}
)
(deny network*)
'''


def _SENSITIVE_SANDBOX_DIRS(home: str) -> set[str]:
    """Concrete OS-level deny paths matching the shared sensitive-path policy."""
    names = {".aws", ".claude", ".codex", ".config", ".docker", ".gnupg", ".kube", ".npm", ".ssh"}
    paths = {os.path.join(home, name) for name in names}
    paths.add(os.path.join(home, "Library", "Keychains"))
    # Browsers store cookies, profiles, and sessions under these roots.
    paths.update({
        os.path.join(home, "Library", "Application Support", "Google"),
        os.path.join(home, "Library", "Application Support", "Firefox"),
        os.path.join(home, "Library", "Application Support", "BraveSoftware"),
        os.path.join(home, "Library", "Safari"),
    })
    return paths


# A real Documents PDF listing on the target machine is ~1.67 MB. The old
# 64-KB process cap killed a correct read-only find, after which the 2B model
# changed the user's "list" request into a count. Capture at most 2 MB, then
# immediately spill/truncate for model context below; this costs only a small
# bounded transient buffer and preserves the complete common-case listing.
_MAX_CAPTURE_BYTES = 2 * 1024 * 1024


def _run_capped(command: str, profile_path: str) -> tuple[str, bool, bool, int | None]:
    """Return (output, timed_out, capped, returncode); kill the whole process
    group on timeout/cap. returncode is None when killed before exit (timeout
    or capped) -- there's no exit code to report in that case.
    """
    proc = subprocess.Popen(
        ["sandbox-exec", "-f", profile_path, "bash", "-c", "set -f; " + command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        cwd=str(config.BASH_ALLOWED_ROOT), start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    assert proc.stdout is not None and proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ)
    selector.register(proc.stderr, selectors.EVENT_READ)
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + config.BASH_TIMEOUT_SECONDS
    timed_out = capped = False
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            for key, _ in selector.select(remaining):
                data = os.read(key.fileobj.fileno(), 8192)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                take = min(len(data), _MAX_CAPTURE_BYTES - total)
                if take:
                    chunks.append(data[:take])
                    total += take
                if take < len(data) or total >= _MAX_CAPTURE_BYTES:
                    capped = True
                    break
            if capped:
                break
    finally:
        selector.close()
        if timed_out or capped:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=5)
    returncode = None if (timed_out or capped) else proc.returncode
    return b"".join(chunks).decode("utf-8", errors="replace"), timed_out, capped, returncode


# Ported from ENDEAVOR_LOCAL_AGENT_MAX/tools/bash.py (2026-07-23, user
# request) -- evidence-gated recovery hint for a failed call. A wrong hint is
# worse than no hint, so each branch checks an actual signal (returncode,
# a specific stderr/output marker), never guesses from the command text.
_SANDBOX_DENY_MARKERS = ("Operation not permitted", "Permission denied", "Read-only file system")


def _classify_bash_error(returncode: int | None, output: str) -> str | None:
    if returncode == 127:
        return "command not found — ตรวจตัวสะกด หรือ path นั้นอาจไม่มีอยู่จริง"
    if returncode == 126:
        return "permission denied ตอน execute — ไฟล์นั้นอาจไม่ใช่ executable"
    if returncode not in (None, 0) and any(m in output for m in _SANDBOX_DENY_MARKERS):
        return "sandbox บล็อกการเข้าถึงนี้ — bash ของ LITE อ่านได้เฉพาะไฟล์ที่ไม่ sensitive เท่านั้น"
    return None


def _compat_rewrite(command: str) -> str:
    """Provide file listing when ripgrep is absent from the sandbox PATH."""
    try:
        first, *rest = command.split("|", 1)
        tokens = shlex.split(first.strip())
    except ValueError:
        return command
    if len(tokens) == 3 and tokens[0] == "rg" and tokens[1] == "--files":
        # This is the common model fallback after RAG is unavailable.  `find`
        # is present on macOS and already passes the same read-only checks.
        rewritten = f"find {shlex.quote(tokens[2])} -type f"
        return rewritten if not rest else f"{rewritten} |{rest[0]}"
    return command


_DOCUMENT_KIND_PATTERNS = (
    (re.compile(r"\bpdf\b", re.IGNORECASE), ".pdf"),
    (re.compile(r"\b(?:word|docx)\b", re.IGNORECASE), ".docx"),
    (re.compile(r"\b(?:excel|xlsx)\b", re.IGNORECASE), ".xlsx"),
    (re.compile(r"\b(?:powerpoint|pptx)\b", re.IGNORECASE), ".pptx"),
)


def _align_find_to_document_kind(command: str, user_query: str) -> str:
    """Keep a simple ``find`` fallback faithful to the requested file kind.

    Live trajectory testing found that the 2B model retained the requested
    count but dropped ``*.pdf`` after the RAG result. This bounded rewrite is
    safer and cheaper than asking the model to rediscover its missing filter.
    """
    requested = [suffix for pattern, suffix in _DOCUMENT_KIND_PATTERNS if pattern.search(user_query or "")]
    if len(requested) != 1:
        return command
    first, *pipeline = command.split("|", 1)
    try:
        tokens = shlex.split(first.strip())
    except ValueError:
        return command
    if not tokens or tokens[0] != "find":
        return command
    suffix = requested[0]
    name_positions = [index for index, token in enumerate(tokens) if token in {"-name", "-iname"}]
    if name_positions:
        index = name_positions[0]
        if index + 1 < len(tokens) and tokens[index + 1].casefold() != f"*{suffix}".casefold():
            tokens[index:index + 2] = ["-iname", f"*{suffix}"]
    else:
        tokens.extend(["-iname", f"*{suffix}"])
    aligned = shlex.join(tokens)
    return aligned if not pipeline else f"{aligned} | {pipeline[0].strip()}"


_WHOLE_MACHINE_RE = re.compile(r"(?:ทั้งเครื่อง|ทั้งคอม|ทั่วเครื่อง|ทุกไฟล์ใน\s*Mac)", re.IGNORECASE)
_LIST_INTENT_RE = re.compile(r"(?:รายการ|ลิสต์|ดูรายการ|\blist\b|\bshow\b)", re.IGNORECASE)
_COUNT_INTENT_RE = re.compile(r"(?:นับ|กี่ไฟล์|จำนวน|\bcount\b|\bhow many\b)", re.IGNORECASE)
def _align_whole_machine_content_scope(command: str, user_query: str) -> str:
    """Repair a project-scoped grep when the user asked for the whole Mac."""
    if not _WHOLE_MACHINE_RE.search(user_query or ""):
        return command
    first = command.split("|", 1)[0].strip()
    try:
        tokens = shlex.split(first)
    except ValueError:
        return command
    if not tokens or tokens[0] not in {"grep", "rg"}:
        return command
    operands = [token for token in tokens[1:] if not token.startswith("-")]
    if not operands:
        return command
    # grep/rg's first non-flag operand is the literal search phrase. A plain
    # quoted mdfind query searches Spotlight metadata/content machine-wide
    # without embedding project scope or constructing shell syntax.
    return f"mdfind {shlex.quote(operands[0])}"


def _align_filename_search(command: str, user_query: str) -> str:
    """Use Spotlight when the agent itself constructed a filename glob.

    The model owns natural-language intent and the search term. This adapter
    only changes the execution backend for an explicit ``find -name``/``-iname``
    operand, preserving that operand verbatim; it never guesses a term from
    the user's sentence.
    """
    try:
        tokens = shlex.split(command.split("|", 1)[0].strip())
    except ValueError:
        return command
    if not tokens or tokens[0] != "find":
        return command
    pattern = None
    for index, token in enumerate(tokens[:-1]):
        if token in {"-name", "-iname"}:
            pattern = tokens[index + 1]
            break
    if not pattern:
        return command
    query = f"kMDItemFSName == '{pattern}'cd"
    return f"mdfind {shlex.quote(query)}"


def _align_project_scope(command: str, user_query: str) -> str:
    """Keep model-generated project searches inside the named project.

    The model remains responsible for the search term and command intent. This
    adapter only replaces filesystem roots when it ignored the explicit
    project scope and pointed grep/find/Spotlight at a sibling project.
    """
    if not search_scope.is_project_scoped_query(user_query):
        return command
    first, *pipeline = command.split("|", 1)
    try:
        tokens = shlex.split(first.strip())
    except ValueError:
        return command
    if not tokens:
        return command
    executable = tokens[0]
    root = str(config.PROJECT_ROOT)
    if executable in {"grep", "rg"}:
        operands = [index for index, token in enumerate(tokens[1:], start=1) if not token.startswith("-")]
        if executable == "rg" and "--files" in tokens:
            path_indexes = operands
        else:
            path_indexes = operands[1:]
        if path_indexes:
            for index in path_indexes:
                tokens[index] = root
        else:
            tokens.append(root)
    elif executable == "find":
        if len(tokens) > 1 and not tokens[1].startswith("-"):
            tokens[1] = root
    elif executable == "mdfind":
        if "-onlyin" in tokens:
            index = tokens.index("-onlyin")
            if index + 1 < len(tokens):
                tokens[index + 1] = root
        else:
            tokens[1:1] = ["-onlyin", root]
    else:
        return command
    aligned = shlex.join(tokens)
    return aligned if not pipeline else f"{aligned} | {pipeline[0].strip()}"


def _align_listing_intent(command: str, user_query: str) -> str:
    """Do not turn a list request into a count followed by a retry."""
    if not _LIST_INTENT_RE.search(user_query or "") or _COUNT_INTENT_RE.search(user_query or ""):
        return command
    stages = [stage.strip() for stage in command.split("|")]
    try:
        parsed = [shlex.split(stage) for stage in stages]
    except ValueError:
        return command
    if not parsed or not parsed[0] or parsed[0][0] not in {"find", "ls"}:
        return command
    kept = [stage for stage, tokens in zip(stages, parsed) if not (tokens and tokens[0] == "wc")]
    return " | ".join(kept)


def normalize_command_for_query(command: str, user_query: str) -> str:
    command = _align_find_to_document_kind(command, user_query)
    command = _align_filename_search(command, user_query)
    command = _align_project_scope(command, user_query)
    command = _align_whole_machine_content_scope(command, user_query)
    return _align_listing_intent(command, user_query)


def run(args: dict[str, str]) -> str:
    command = (args.get("command") or "").strip()
    if not command:
        return "[error] command is required"

    deny_reason = _is_denied(command)
    if deny_reason:
        return f"[blocked] {deny_reason}"

    read_only_reason = _read_only_violation(command)
    if read_only_reason:
        return f"[blocked] {read_only_reason}"

    command = _compat_rewrite(command)
    executable = shlex.split(command.split("|")[0])[0]

    profile_path = None
    try:
        # A test or a user cleanup may remove the private state directory
        # after config import; recreating it remains inside workspect.
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", dir=config.DATA_DIR, delete=False) as profile:
            profile.write(_build_readonly_sandbox_profile())
            profile_path = profile.name
        output, timed_out, capped, returncode = _run_capped(command, profile_path)
        if timed_out:
            return f"[error] command timed out after {config.BASH_TIMEOUT_SECONDS}s; process group was stopped"
        if capped:
            return f"[error] command output exceeded {_MAX_CAPTURE_BYTES} bytes; process group was stopped"
        output = output.strip() or "(no output)"
        output = _sanitize_read_output(os.path.basename(executable), output)
        hint = _classify_bash_error(returncode, output)
        if hint:
            output += f"\n[hint] {hint}"
    except Exception as e:
        return f"[error] bash failed: {e}"
    finally:
        if profile_path:
            try:
                os.unlink(profile_path)
            except OSError:
                pass

    if len(output) > config.TOOL_OUTPUT_TRUNCATE_CHARS:
        saved_path = save_full_output(output, "bash")
        line_count = len(output.splitlines())
        preview = truncate_output(output, config.TOOL_OUTPUT_TRUNCATE_CHARS)
        output = (
            f"[full output: {line_count} lines; saved: {saved_path}; preview below is incomplete]\n"
            f"{preview}"
        )
    return output


# PLAN.md §9.6 (2026-07-25, RAM/token experiment, tested before adopting):
# this docstring is split into a short, always-visible schema half (routing-
# decision content only -- the bash-vs-rag_search distinction that fixes a
# real observed misroute, must be visible BEFORE the model chooses a tool)
# and a syntax/retry manual deferred here, injected into the FIRST bash call's
# own ToolMessage result each turn (helpers.turn_guard.call_count) -- by then
# the model has already decided to call bash, so this only needs to be
# visible once execution/retry syntax actually matters. Injecting into the
# tool's own return value (not a separate message type) needs no ToolNode
# override and is prefix-cache-safe (pure conversation append, schema itself
# never changes turn-to-turn). See PLAN.md §9.6 for the routing-probe
# before/after comparison that justified this split.
_SYNTAX_MANUAL = (
    "\n\n[bash usage notes]\n"
    "grep -r -l (list matching FILE NAMES, not every matching line) on an "
    "ABSOLUTE path -- a relative path like \".\" searches this tool's own "
    "private sandbox directory, not the user's project, so it will silently "
    "find nothing even when the command itself is correct. Use the project's "
    "real full path (given in the system prompt) here, never a placeholder.\n"
    "✅ rag_search found nothing for a contract phrase -> bash(command=\"grep -r -l 'contract phrase' /real/Documents\")\n"
    "Plain grep -r without -l prints every matching line and commonly exceeds "
    "this tool's output cap on a real project (logs, caches) -- always add -l "
    "for a \"which file mentions X\" search; write -r -l as separate flags, not "
    "the combined -rl (combined short flags are not recognized).\n"
    "Do not add redirects (>, 2>/dev/null), parentheses/grouping, or a "
    "trailing pipe to head -- those are always rejected here and output is "
    "already capped automatically, so they add nothing. A rejected command "
    "means change approach (simplify the command, use an absolute path), not "
    "retry the same syntax.\n"
    "If a command fails or returns empty, do not rerun the exact same command "
    "-- narrow the scope or rephrase. If it still doesn't work, tell the user "
    "plainly it wasn't found; do not loop."
)

# LangChain entry point (2026-07-23, architecture switch to create_react_agent
# — see config.py's module docstring). Wraps the existing, already-tested
# run() logic (deny-list, path scoping, etc.) rather than rewriting it —
# @tool's docstring becomes the schema description LangGraph sends the model.
from langchain_core.tools import tool as _lc_tool


@_lc_tool
def bash(command: str) -> str:
    """Run a read-only-oriented bash command (find, grep, ls, cat, wc, etc.) to
    search or inspect files/processes on this Mac. Destructive commands
    (rm, sudo, disk-wiping, etc.) are blocked.

    Call this first only when the user explicitly asks to use/run bash.
    Otherwise rag_search must have run earlier this turn and returned no
    relevant hit, or only snippets that cannot complete a list/count task.
    SELF-CHECK: explicit bash request OR prior insufficient RAG result? If NO,
    call rag_search first. If YES, continue here.

    Use this for counting/listing files matching a condition, whole-machine
    Spotlight content search, or fallback full-content grep -- not for an
    exact filename already returned by rag_search.
    For a document-type list/count, use find with an absolute root and -name
    extension filter; grep searches file CONTENT and is wrong for this task.
    Never add 2>/dev/null or pipe to head: redirects are blocked and output is
    already capped. Copy the real absolute root from the system prompt.
    Every filesystem operand must be an absolute path from the system prompt;
    relative paths run inside the private workspect directory, not the project.

    If rag_search finds nothing for a content search, fall back here with
    grep -r -l on an ABSOLUTE path (never a relative path or placeholder) --
    detailed syntax/retry notes appear in this tool's own result the first
    time you call it this turn."""
    from helpers import turn_guard
    command = normalize_command_for_query(command, turn_guard.user_query())
    args = {"command": command}
    blocked = turn_guard.check("bash", args)
    if blocked:
        return blocked
    first_call_this_turn = turn_guard.call_count("bash") == 1
    result = run(args)
    turn_guard.record("bash", args, result)
    if first_call_this_turn:
        result += _SYNTAX_MANUAL
    return result


# Qwen3.5-2B can select bash correctly yet hallucinate a familiar folder
# beneath Desktop or the project root. Keep the machine's exact, configured
# document roots at the call site where tool arguments are generated. This is
# static for the process lifetime, so it does not break prompt-prefix caching.
_ROOT_GUIDE = ", ".join(f"{path.name}={path}" for path in config.INDEX_ROOTS)
bash.description += (
    f"\nExact indexed document roots: {_ROOT_GUIDE}. "
    "Copy these paths exactly; never move Documents/Downloads under Desktop "
    "or under the project root."
)
