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
_LOCAL_DOCUMENT_RE = re.compile(
    r"(?:ไฟล์|เอกสาร|ในเครื่อง|บนเครื่อง|โปรเจกต์|โฟลเดอร์|\bpath\b|"
    r"/Users/|/Volumes/|\bfile\b|\bdocument\b)",
    re.IGNORECASE,
)
_EXPLICIT_WEB_RE = re.compile(
    r"(?:ค้น(?:หา)?(?:ข้อมูล)?บนเว็บ|ค้นเว็บ|อินเทอร์เน็ต|บนอินเทอร์เน็ต|"
    r"\binternet\b|\bonline\b|\bweb\s+search\b)",
    re.IGNORECASE,
)
_KNOWLEDGE_BASE_TERM = (
    r"(?:\b(?:rag|kb)\b|knowledge\s*base|knowledge/(?:raw|md)|"
    r"rag_document/(?:raw|md)|\brag_document\b|"
    r"ฐานความรู้|คลังความรู้)"
)
_KNOWLEDGE_BASE_SCOPE_RE = re.compile(
    rf"(?:"
    rf"(?:ใน|จาก|ภายใน|ผ่าน|อ้างอิงจาก|in|from|within)\s*{_KNOWLEDGE_BASE_TERM}"
    rf"|(?:ค้น(?:หา)?|หา|ดึง|เปิด|อ่าน|สรุป|แสดง|อ้างอิง|ตอบ).{{0,50}}"
    rf"{_KNOWLEDGE_BASE_TERM}"
    rf"|{_KNOWLEDGE_BASE_TERM}"
    rf")",
    re.IGNORECASE,
)
_KNOWLEDGE_BASE_EXPLANATION_RE = re.compile(
    rf"(?:"
    rf"^\s*{_KNOWLEDGE_BASE_TERM}\s*"
    rf"(?:คือ|หมายถึง|ทำงาน|ใช้งาน|ใช้|ทำอะไรได้บ้าง|มีไว้ทำอะไร|"
    rf"อธิบาย|ดึงข้อมูล|ค้นข้อมูล)"
    rf"|(?:อะไรคือ|ความหมายของ|อธิบาย(?:คำว่า)?|วิธีการทำงานของ)"
    rf"\s*{_KNOWLEDGE_BASE_TERM}"
    rf"|^\s*(?:what\s+is|how\s+does|explain)\s+{_KNOWLEDGE_BASE_TERM}"
    rf")",
    re.IGNORECASE,
)
_OUTSIDE_KNOWLEDGE_BASE_RE = re.compile(
    rf"(?:"
    rf"(?:นอก|นอกจาก|ไม่ใช้|ไม่ต้องใช้|ไม่ค้นใน|ไม่ค้นจาก|นอกขอบเขต)\s*"
    rf"{_KNOWLEDGE_BASE_TERM}"
    rf"|(?:outside(?:\s+of)?|without|not\s+in)\s+"
    rf"{_KNOWLEDGE_BASE_TERM}"
    rf")",
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


def is_local_document_query(query: str | None = None) -> bool:
    """Whether a request is explicitly about a local file/document.

    This is a routing safety boundary, not a filename parser.  A local
    document request must never fall through to an internet search just
    because the small model selected the wrong tool.  Explicit web wording
    wins when the user really asks to search online for a document topic.
    """
    text = request_query() if query is None else (query or "")
    return bool(_LOCAL_DOCUMENT_RE.search(text)) and not bool(_EXPLICIT_WEB_RE.search(text))


def is_knowledge_base_query(query: str | None = None) -> bool:
    """Whether the user explicitly requested Lite's Knowledge Base/RAG scope.

    A bare mention such as ``RAG คืออะไร`` is a general question, not a
    request to search Lite's local index.  Scope must be expressed through a
    source relation (``ใน RAG``, ``จาก KB``) or an unmistakable local-source
    phrase (``ค้นหาฐานความรู้``).  Explicit online wording still wins.
    """
    text = request_query() if query is None else (query or "")
    # An explicit online/web qualifier wins when the user is discussing the
    # word "RAG" as a subject rather than selecting Lite's local RAG scope.
    return (
        bool(_KNOWLEDGE_BASE_SCOPE_RE.search(text))
        and not bool(_KNOWLEDGE_BASE_EXPLANATION_RE.search(text))
        and not bool(_OUTSIDE_KNOWLEDGE_BASE_RE.search(text))
        and not bool(_EXPLICIT_WEB_RE.search(text))
    )


def is_knowledge_base_explanation_query(query: str | None = None) -> bool:
    """Whether the user is explaining/asking about RAG itself, not its data."""
    text = request_query() if query is None else (query or "")
    return bool(_KNOWLEDGE_BASE_EXPLANATION_RE.search(text)) and not bool(
        _EXPLICIT_WEB_RE.search(text) or _OUTSIDE_KNOWLEDGE_BASE_RE.search(text)
    )


def is_outside_knowledge_base_query(query: str | None = None) -> bool:
    """Whether the user explicitly wants a search outside Lite's local KB."""
    text = request_query() if query is None else (query or "")
    return bool(_OUTSIDE_KNOWLEDGE_BASE_RE.search(text)) and not bool(
        _EXPLICIT_WEB_RE.search(text) or _KNOWLEDGE_BASE_EXPLANATION_RE.search(text)
    )
