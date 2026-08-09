# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Turn-scoped search intent shared by the local search tools.

The model still owns the actual filename/content keywords.  This module only
records the user's requested *scope* so a tool result cannot silently escape
the project the user named, especially when the global index contains rows
from several sibling projects.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
import re


_REQUEST_QUERY: ContextVar[str] = ContextVar("lite_search_request_query", default="")

# These are scope markers, not filename/content intent rules.  The whole
# machine markers win, so "ทั้งเครื่อง" never becomes project-scoped merely
# because the sentence also mentions a project.
_WHOLE_MACHINE_RE = re.compile(
    r"(?:ทั้งเครื่อง|ทั้งคอม|ทั่วเครื่อง|ทุกไฟล์ใน\s*Mac|whole\s+machine|entire\s+computer)",
    re.IGNORECASE,
)
_PROJECT_SCOPE_RE = re.compile(
    r"(?:โปรเจกต์|โครงการ|\bproject\b|\brepository\b|\brepo\b)",
    re.IGNORECASE,
)


def set_request_query(query: str) -> Token:
    """Set the current user request and return its reset token."""
    return _REQUEST_QUERY.set(query or "")


def reset_request_query(token: Token) -> None:
    _REQUEST_QUERY.reset(token)


def request_query() -> str:
    return _REQUEST_QUERY.get()


def is_whole_machine_query(query: str | None = None) -> bool:
    return bool(_WHOLE_MACHINE_RE.search(request_query() if query is None else (query or "")))


def is_project_scoped_query(query: str | None = None) -> bool:
    """Whether the request explicitly names a project/repository scope."""
    text = request_query() if query is None else (query or "")
    return bool(_PROJECT_SCOPE_RE.search(text)) and not is_whole_machine_query(text)
