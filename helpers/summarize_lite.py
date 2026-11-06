"""summarize_lite.py — extractive (non-LLM) summarization for web_search.

MAX_VLM's tools/_summarize.py calls an LLM (llama3.1:8b) to write a
query-aware summary of each fetched page. That's not an option here — this
agent's only model is the 2B doing the actual conversation, and routing a
second inference call through it for every search result would both slow
things down and compete for the same limited RAM. Instead: score paragraphs
by literal query-term overlap and concatenate the best few — extractive, not
abstractive, but free and instant.

Thai has no spaces between words, so plain `query.split()` produces useless
single "terms" for Thai text (the whole phrase). For any query containing
Thai characters, this also generates character 3-grams as terms so partial
overlap still scores.
"""
from __future__ import annotations
import re

_THAI_RE = re.compile(r"[฀-๿]")


def query_terms(query: str) -> list[str]:
    terms = [t.lower() for t in re.split(r"\s+", query.strip()) if t]
    if _THAI_RE.search(query):
        compact = re.sub(r"\s+", "", query.lower())
        terms += [compact[i : i + 3] for i in range(max(0, len(compact) - 2))]
    return terms


def _score(paragraph: str, terms: list[str]) -> int:
    p = paragraph.lower()
    return sum(p.count(t) for t in terms if t)


def extractive_summary(text: str, query: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    terms = query_terms(query)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    scored = sorted(
        ((( _score(p, terms)), i, p) for i, p in enumerate(paragraphs)),
        key=lambda x: (-x[0], x[1]),
    )

    picked: list[str] = []
    used_chars = 0
    for score, _i, p in scored:
        if score <= 0 and picked:
            break  # once we run out of matching paragraphs, stop rather than padding with noise
        if used_chars >= max_chars:
            break
        picked.append(p)
        used_chars += len(p)

    if not picked:
        picked = [paragraphs[0]]

    out = "\n".join(picked)
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "..."
    return out



def query_filtered_summary(text: str, query: str, max_chars: int) -> str:
    """Return only query-matching source passages, bounded and extractive.

    This is intentionally stricter than ``extractive_summary``: a fetched page
    that has no positive query overlap returns an empty string instead of its
    first paragraph.  That prevents navigation/boilerplate text from replacing
    a useful search-engine snippet merely because the HTTP fetch succeeded.
    """
    text = (text or "").strip()
    if not text or max_chars <= 0:
        return ""
    terms = query_terms(query)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    scored = sorted(
        (((_score(paragraph, terms)), index, paragraph) for index, paragraph in enumerate(paragraphs)),
        key=lambda item: (-item[0], item[1]),
    )
    picked: list[str] = []
    used = 0
    for score, _index, paragraph in scored:
        if score <= 0:
            break
        separator = 1 if picked else 0
        room = max_chars - used - separator
        if room <= 0:
            break
        piece = paragraph if len(paragraph) <= room else paragraph[:room].rstrip() + "..."
        picked.append(piece)
        used += len(piece) + separator
        if used >= max_chars:
            break
    return "\n".join(picked)[:max_chars]
