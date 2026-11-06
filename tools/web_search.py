"""tools/web_search.py — bounded deterministic multi-query web evidence.

One call accepts one search query or up to four focused queries. Search runs in
parallel for a batch, then code-only relevance ranking normally selects up to
three domain-diverse sources, expanding only to one source per query for a
four-query batch. HTTP connections are pinned to validated public IPs and
every redirect is revalidated to prevent DNS-rebinding/SSRF. Fetched page text
is reduced to short query-matching passages; search snippets remain the fallback
when a page cannot be fetched or yields no matching text.

Every evidence line keeps its source URL. This tool never invokes an LLM.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import re
import ipaddress
import socket
import ssl
from urllib.parse import urljoin, urlparse

import config
from helpers.summarize_lite import extractive_summary, query_filtered_summary

SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet and return short query-filtered evidence with source URLs. "
            "Pass one query string, or 2-4 focused queries in one list for distinct facts/angles; "
            "list queries run concurrently and deterministic relevance ranking normally selects up to three sources, "
            "or one source per query for a four-query batch. "
            "If a retry is needed, change the weak query materially; never repeat the same query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
                    ],
                    "description": "One search query or up to four focused queries to run concurrently",
                }
            },
            "required": ["query"],
        },
    },
}

_THAI_RE = re.compile(r"[฀-๿]")
_ADULT_DOMAINS = {"xxx", "porn", "sex", "adult", "hentai", "nude", "xhamster", "xvideos", "pornhub"}
_ADULT_KW = {"คลิปหลุด", "หนังโป๊", "nude", "porn", "xxx", "explicit", "18+"}
_NAVIGATION_HOSTS = {
    "google.com", "bing.com", "search.brave.com", "duckduckgo.com",
    "myaccount.google.com", "accounts.google.com",
}
_NAVIGATION_PATH_PREFIXES = ("/channel/", "/user/", "/@")
_RELEVANCE_DROP_PHRASES = (
    "ล่าสุด", "วันนี้", "ปัจจุบัน", "เท่าไร", "เท่าไหร่", "ช่วย", "ค้นหา",
    "พร้อมแหล่งที่มา", "แหล่งที่มา", "ภาพรวม", "สำหรับมือใหม่",
    "latest", "today", "current", "please", "overview", "official", "source", "sources",
)
_MIN_RELEVANCE = 0.55
_MIN_TITLE_RELEVANCE = 0.55
_MIN_OVERVIEW_SOURCES = 2
_NEWS_EN_RE = re.compile(r"\bnews\b", re.IGNORECASE)
_CURRENT_EN_RE = re.compile(r"\b(?:now|latest|today|current)\b", re.IGNORECASE)
_FRESH_EN_RE = re.compile(r"\b(?:today|latest|now|current|breaking|this week)\b", re.IGNORECASE)
_FRESH_TH = ("วันนี้", "ล่าสุด", "ตอนนี้", "ปัจจุบัน", "เมื่อวาน", "สัปดาห์นี้", "เดือนนี้")
_YEAR_RE = re.compile(r"\b(?:20|25)\d{2}\b")
_VERSION_RE = re.compile(r"(?<![\d.])v?(\d+)\.(\d+)(?:\.\d+)?(?![\d.])", re.IGNORECASE)
_DEADLINE_QUERY_RE = re.compile(
    r"(?:กำหนด(?:วัน)?(?:ยื่น)?|ยื่น.{0,20}(?:เมื่อไร|เมื่อไหร่|วันไหน)|deadline|due\s+date)",
    re.IGNORECASE,
)
_DEADLINE_EVIDENCE_RE = re.compile(
    r"(?:กำหนด|ภายใน|วันสุดท้าย|ยื่น(?:แบบ|ภาษี|ออนไลน์)|deadline|due\s+date)",
    re.IGNORECASE,
)
_PRICE_QUERY_RE = re.compile(r"(?:ราคา|price)", re.IGNORECASE)
_PRICE_TITLE_RE = re.compile(r"(?:ราคา|price|market\s*cap|มูลค่าตลาด)", re.IGNORECASE)
_FEATURE_QUERY_RE = re.compile(r"(?:ฟีเจอร์|มีอะไรใหม่|features?|what'?s\s+new|new\s+in)", re.IGNORECASE)
_FEATURE_TITLE_RE = re.compile(r"(?:ฟีเจอร์|มีอะไรใหม่|features?|what'?s\s+new|new\s+in)", re.IGNORECASE)
_OFFICIAL_QUERY_RE = re.compile(r"(?:official|ทางการ)", re.IGNORECASE)
_FORECAST_QUERY_RE = re.compile(r"(?:แนวโน้ม|คาดการณ์|forecast|prediction|outlook)", re.IGNORECASE)
_FORECAST_TITLE_RE = re.compile(r"(?:แนวโน้ม|คาดการณ์|คาดว่า|forecast|prediction|outlook)", re.IGNORECASE)
_MAX_BATCH_QUERIES = 4
_FETCH_MAX_BYTES = 2_000_000
_FETCH_MAX_REDIRECTS = 3
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "th,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
}
_TRAFILATURA_EXTRACT_LOCK = threading.Lock()


def _relevance_grams(text: str) -> set[str]:
    """Language-agnostic 3-grams after removing search-instruction filler."""
    normalized = (text or "").casefold()
    for phrase in _RELEVANCE_DROP_PHRASES:
        normalized = normalized.replace(phrase, "")
    # Search-engine operators constrain retrieval but are not evidence words;
    # do not lower the score of the page they intentionally target.
    normalized = re.sub(r"\bsite:[^\s]+", " ", normalized)
    compact = re.sub(r"[^0-9a-zก-๙]+", "", normalized)
    return {compact[i : i + 3] for i in range(max(0, len(compact) - 2))}


def _canonical_years(text: str) -> set[int]:
    """Normalize equivalent Buddhist/Gregorian years to Gregorian years."""
    years = set()
    for raw in _YEAR_RE.findall(text or ""):
        year = int(raw)
        years.add(year - 543 if year >= 2400 else year)
    return years


def _simplified_query_variant(query: str) -> str:
    """Remove request/currentness filler without guessing new search facts."""
    value = query
    for phrase in sorted(_RELEVANCE_DROP_PHRASES, key=len, reverse=True):
        value = re.sub(re.escape(phrase), " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" \"'()[]{}.,:;!?")
    return value or query


def _feature_query_variant(query: str) -> str:
    """Turn a natural-language feature question into a search-shaped query.

    This is web-query normalization only. It does not decide whether a
    request is local or online and it does not invent a subject; it removes
    request/link filler while retaining the literal version/entity words.
    """
    value = query or ""
    value = re.sub(r"(?:ช่วย\s*)?ค้นหา(?:ว่า)?|please\s+search(?:\s+for)?", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"พร้อม\s*(?:แนบ\s*)?(?:ลิงก์|แหล่งที่มา)(?:\s*ที่มา)?", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"(?:มี\s*)?(?:ฟีเจอร์|features?)\s*ใหม่(?:\s*อะไรบ้าง)?", " what's new ", value, flags=re.IGNORECASE)
    value = re.sub(r"อะไรบ้าง", " ", value, flags=re.IGNORECASE)
    value = _simplified_query_variant(value)
    python_version = re.search(r"\bPython\s+(3\.\d+)\b", value, re.IGNORECASE)
    if python_version:
        version = python_version.group(1)
        return f"site:docs.python.org/{version}/whatsnew/{version}.html Python {version} what's new"
    if not re.search(r"what'?s\s+new", value, re.IGNORECASE):
        value = f"{value} what's new".strip()
    return value or query


def _topic_anchors(query: str) -> list[str]:
    """Extract literal subject anchors that a result title must contain."""
    value = _simplified_query_variant(query)
    value = re.sub(r"\bsite:[^\s]+", " ", value, flags=re.IGNORECASE)
    value = value.strip(" \"'()[]{}.,:;!?")
    for prefix in (
        "กำหนดวัน", "กำหนด", "วิธี", "ราคา", "ข่าว", "ข้อมูล",
        "how to", "price of", "price", "news about", "news",
    ):
        if value.casefold().startswith(prefix.casefold()):
            value = value[len(prefix):].strip(" \"'()[]{}.,:;!?")
            break
    tokens = [
        token.strip(" \"'()[]{}.,:;!?")
        for token in value.split()
        if len(token.strip(" \"'()[]{}.,:;!?")) >= 4
        and not _YEAR_RE.fullmatch(token.strip(" \"'()[]{}.,:;!?"))
    ]
    anchors = tokens or ([value] if len(value) >= 4 else [])
    for anchor in list(anchors):
        for suffix in ("ประเทศไทย", "ไทย"):
            if anchor.endswith(suffix) and len(anchor) - len(suffix) >= 4:
                anchors.append(anchor[:-len(suffix)])
                break
    return list(dict.fromkeys(anchors))


def _relevance_score(result: dict, query: str) -> float:
    """Return lexical coverage of the query by a result title/snippet.

    DDGS rank is not a relevance guarantee.  Comparing character trigrams
    works for Thai (no word-boundary dependency), English, numbers, and mixed
    text without loading an embedding model into an 8GB machine.
    """
    query_grams = _relevance_grams(query)
    if not query_grams:
        return 1.0

    def coverage(text: str) -> float:
        return len(query_grams & _relevance_grams(text)) / len(query_grams)

    title = result.get("title") or ""
    snippet = result.get("snippet") or ""
    combined = f"{title} {snippet}"
    title_score = coverage(title)
    score = max(title_score, coverage(snippet), coverage(combined))

    # A search for tax year 2569 must not cite a result explicitly labelled
    # 2567 merely because the surrounding topic words overlap. A result with
    # no visible year remains eligible (official landing pages often omit it
    # from the snippet); only an explicit contradictory year is rejected.
    query_years = _canonical_years(query)
    result_years = _canonical_years(f"{combined} {result.get('url') or ''}")
    if query_years and result_years and query_years.isdisjoint(result_years):
        return 0.0
    if query_years and not result_years and not _is_authoritative_url(result.get("url") or ""):
        return 0.0
    query_versions = {match[:2] for match in _VERSION_RE.findall(query)}
    result_versions = {match[:2] for match in _VERSION_RE.findall(combined)}
    if query_versions and query_versions.isdisjoint(result_versions):
        return 0.0
    if _DEADLINE_QUERY_RE.search(query) and not _DEADLINE_EVIDENCE_RE.search(combined):
        return 0.0
    if _PRICE_QUERY_RE.search(query) and not _PRICE_TITLE_RE.search(title):
        return 0.0
    if _FEATURE_QUERY_RE.search(query) and not _FEATURE_TITLE_RE.search(title):
        return 0.0
    if _OFFICIAL_QUERY_RE.search(query) and not _is_official_for_query(result, query):
        return 0.0
    if (
        (_is_time_sensitive(query) or "เท่าไร" in query or "เท่าไหร่" in query)
        and not _FORECAST_QUERY_RE.search(query)
        and _FORECAST_TITLE_RE.search(title)
    ):
        return 0.0

    # A long snippet can mention the query in passing while the page itself is
    # about insurance, loans, or another subject. Require a literal topic
    # anchor in the title; this generalizes across query length better than a
    # fixed title-score threshold. Official pages may use generic titles such
    # as "ปี 2569", so those retain the combined-content fallback.
    compact_title = re.sub(r"[^0-9a-zก-๙]+", "", title.casefold())
    title_has_anchor = any(
        re.sub(r"[^0-9a-zก-๙]+", "", anchor.casefold()) in compact_title
        for anchor in _topic_anchors(query)
    )
    if not (title_has_anchor or title_score >= _MIN_TITLE_RELEVANCE) and not (
        _is_authoritative_url(result.get("url") or "") and score >= _MIN_RELEVANCE
    ):
        return 0.0
    return score


def _filter_relevant(results: list[dict], query: str) -> list[dict]:
    return [result for result in results if _relevance_score(result, query) >= _MIN_RELEVANCE]


def _retry_message(query: str) -> str:
    return (
        "[needs_retry] ผลค้นหารอบนี้ยังไม่เกี่ยวข้อง/ครอบคลุมพอสำหรับคำถาม "
        f"{query!r}; ปรับ query ให้เจาะจงหัวข้อ+พื้นที่+ช่วงเวลา แล้วเรียก web_search ใหม่อีกครั้ง "
        "(ไม่เกินโควตาต่อ turn) ก่อนเขียนภาพรวม"
    )


def _news_core_variant(query: str) -> str:
    """Remove request/currentness filler before querying a news backend."""
    value = query
    for phrase in ("ล่าสุด", "วันนี้", "ตอนนี้", "ปัจจุบัน", "ภาพรวม"):
        value = value.replace(phrase, " ")
    value = _CURRENT_EN_RE.sub(" ", value)
    value = re.sub(r"\boverview\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:20|25)\d{2}\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:;")
    if value.casefold().startswith("ข่าว"):
        value = value[4:].strip()
    value = re.sub(r"^news\b", "", value, count=1, flags=re.IGNORECASE).strip(" -,:;")
    return value or query


def _official_site_variant(query: str) -> str | None:
    """Return one high-precision primary-source retry for known Thai bodies.

    Prompting the 2B model to add ``site:`` proved nondeterministic in live
    trials. Keep this deliberately small: it is a search-quality fallback,
    not a hand-built answer database.
    """
    if re.search(r"ธนาคารแห่งประเทศไทย|แบงก์ชาติ|อัตราดอกเบี้ยนโยบาย", query):
        return 'site:bot.or.th "อัตราดอกเบี้ยนโยบาย"'
    if re.search(r"กรมสรรพากร|ภาษีเงินได้บุคคลธรรมดา", query):
        years = " ".join(dict.fromkeys(_YEAR_RE.findall(query)))
        return f'site:rd.go.th "ภาษีเงินได้บุคคลธรรมดา" {years}'.strip()
    return None


def _is_news_query(query: str) -> bool:
    value = query or ""
    return "ข่าว" in value or bool(_NEWS_EN_RE.search(value))


def _is_time_sensitive(query: str) -> bool:
    """Recognize an explicit freshness request without guessing from topic alone."""
    value = query or ""
    return bool(_FRESH_EN_RE.search(value)) or any(word in value for word in _FRESH_TH)


def _query_timelimit(query: str) -> str | None:
    folded = query.casefold()
    if "วันนี้" in folded or re.search(r"\btoday\b", folded):
        return "d"
    if _is_time_sensitive(query) or _is_news_query(query):
        return "w"
    return None


def _is_authoritative_url(url: str) -> bool:
    host = _domain(url)
    return (
        host.endswith(".go.th")
        or host == "bot.or.th"
        or host.endswith(".bot.or.th")
        or host.endswith(".gov")
        or ".gov." in host
    )


def _is_official_for_query(result: dict, query: str) -> bool:
    """Recognize a requested official source without a brand allow-list."""
    url = result.get("url") or ""
    if _is_authoritative_url(url):
        return True
    host = _domain(url)
    parts = [part for part in host.split(".") if part]
    if len(parts) < 2:
        return False
    # For common country-code structures (example.co.th), the organization
    # label is one position further left; otherwise it is the penultimate.
    common_second_level = {"co", "com", "org", "or", "go", "ac"}
    label_index = -3 if len(parts) >= 3 and parts[-2] in common_second_level and len(parts[-1]) == 2 else -2
    label = re.sub(r"[^0-9a-zก-๙]+", "", parts[label_index].casefold())
    query_compact = re.sub(r"[^0-9a-zก-๙]+", "", query.casefold())
    return len(label) >= 4 and label in query_compact


def _is_fresh_news_result(result: dict, query: str) -> bool:
    """Verify backend dates because some providers ignore ``timelimit``."""
    timelimit = _query_timelimit(query)
    if not timelimit:
        return True
    raw_date = result.get("date")
    if not raw_date:
        return False
    try:
        published = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    max_age = timedelta(days=2 if timelimit == "d" else 8)
    now = datetime.now(timezone.utc)
    return now - max_age <= published <= now + timedelta(days=1)


def _domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return url


def _filter_adult(results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        url = (r.get("url") or "").lower()
        text = ((r.get("title") or "") + " " + (r.get("snippet") or "")).lower()
        if any(d in url for d in _ADULT_DOMAINS) or any(kw in text for kw in _ADULT_KW):
            continue
        out.append(r)
    return out


def _is_navigation_result(result: dict) -> bool:
    """Reject search/account landing pages rather than citing them as sources.

    DDGS can occasionally return an engine's locale landing page, an account
    page, or a generic YouTube channel for a Thai query.  Those pages are not
    evidence for the query even if they arrived in a search response.
    """
    url = result.get("url") or ""
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.lower()
    except Exception:
        return True
    if parsed.scheme not in {"http", "https"} or not host:
        return True
    if host in _NAVIGATION_HOSTS or host.endswith(".google.com"):
        return True
    if host in {"youtube.com", "m.youtube.com"} and path.startswith(_NAVIGATION_PATH_PREFIXES):
        return True
    return not (result.get("title") or "").strip() or not (result.get("snippet") or "").strip()


def _filter_noncontent(results: list[dict]) -> list[dict]:
    return [result for result in results if not _is_navigation_result(result)]


def _dedupe_by_domain(results: list[dict], n: int) -> list[dict]:
    top, seen = [], set()
    for r in results:
        dom = _domain(r["url"])
        if dom not in seen:
            seen.add(dom)
            top.append(r)
        if len(top) >= n:
            break
    return top


def _normalize_result(result: dict) -> dict:
    return {
        "title": result.get("title", ""),
        "snippet": result.get("body") or result.get("snippet") or "",
        "url": result.get("href") or result.get("url") or "",
        "date": result.get("date"),
        "_freshness_relaxed": bool(result.get("_freshness_relaxed", False)),
    }


def _usable_results(results: list[dict], query: str, *, require_fresh: bool = False) -> list[dict]:
    normalized = [_normalize_result(result) for result in results]
    normalized = _filter_noncontent(_filter_adult(normalized))
    if require_fresh:
        normalized = [result for result in normalized if _is_fresh_news_result(result, query)]
    return _filter_relevant(normalized, query)


def _coerce_query_batch(raw: object) -> tuple[list[str], str | None]:
    """Accept one query, a native list, or a JSON-list string; preserve order."""
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return [], "[error] query is required"
        if value.startswith("["):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError):
                decoded = None
            raw = decoded if isinstance(decoded, list) else [value]
        else:
            raw = [value]
    elif isinstance(raw, (list, tuple)):
        raw = list(raw)
    else:
        return [], "[error] query ต้องเป็นข้อความหรือ array ของข้อความ"

    if not raw:
        return [], "[error] query is required"
    if len(raw) > _MAX_BATCH_QUERIES:
        return [], f"[error] web_search รับได้สูงสุด {_MAX_BATCH_QUERIES} คำค้นต่อครั้ง"

    queries: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, str) or not item.strip():
            return [], f"[error] query[{index}] ต้องเป็นข้อความที่ไม่ว่าง"
        query = item.strip()
        if len(query) > 512:
            return [], f"[error] query[{index}] ยาวเกิน 512 ตัวอักษร"
        if query not in seen:
            seen.add(query)
            queries.append(query)
    if not queries:
        return [], "[error] query is required"
    return queries, None


def _search_engines(query: str, region: str) -> list[dict] | str:
    try:
        from ddgs import DDGS
    except Exception as e:
        return f"[error] web search unavailable: {e}"
    timelimit = _query_timelimit(query)
    def search_with_limit(limit: str | None) -> list[dict]:
        combined: list[dict] = []
        seen_urls: set[str] = set()
        for backend in ("duckduckgo", "bing"):
            try:
                kwargs = {
                    "max_results": config.WEB_SEARCH_MAX_RESULTS,
                    "region": region,
                    "backend": backend,
                }
                if limit:
                    kwargs["timelimit"] = limit
                raw = list(DDGS(timeout=config.WEB_SEARCH_FETCH_TIMEOUT).text(query, **kwargs))
                for result in raw:
                    url = result.get("href") or result.get("url") or ""
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        combined.append(result)
                if combined:
                    usable = _usable_results(combined, query)
                    if len(_dedupe_by_domain(usable, _MIN_OVERVIEW_SOURCES)) >= _MIN_OVERVIEW_SOURCES:
                        return combined
            except Exception:
                continue
        return combined

    combined = search_with_limit(timelimit)
    if combined:
        return combined
    # Match MAX VLM's bounded recovery: when providers return no rows for a
    # freshness-filtered search, try once without a time filter.  Mark every
    # fallback row so downstream output never presents it as verified-current.
    if timelimit:
        relaxed = search_with_limit(None)
        if relaxed:
            return [{**result, "_freshness_relaxed": True} for result in relaxed]
    return "(no results found)"


def _search_news(query: str, region: str) -> list[dict] | str:
    """Use dated news results for news intent; text search dates are noisy."""
    try:
        from ddgs import DDGS
    except Exception as e:
        return f"[error] web search unavailable: {e}"
    combined: list[dict] = []
    seen_urls: set[str] = set()
    timelimit = _query_timelimit(query) or "w"
    for backend in ("duckduckgo", "bing"):
        try:
            raw = list(DDGS(timeout=config.WEB_SEARCH_FETCH_TIMEOUT).news(
                query,
                max_results=config.WEB_SEARCH_MAX_RESULTS,
                region=region,
                backend=backend,
                timelimit=timelimit,
            ))
            for result in raw:
                url = result.get("url") or result.get("href") or ""
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    combined.append(result)
            usable = _usable_results(combined, query, require_fresh=True)
            if len(_dedupe_by_domain(usable, _MIN_OVERVIEW_SOURCES)) >= _MIN_OVERVIEW_SOURCES:
                return combined
        except Exception:
            continue
    return combined or "(no recent news results found)"


def _search_candidates_for_query(query: str) -> dict:
    """Run deterministic variants for one query, but do not call the semantic LLM."""
    region = "th-th" if _THAI_RE.search(query) else "us-en"
    news_query = _is_news_query(query)
    search = _search_news if news_query else _search_engines
    if news_query:
        variants = (query, _news_core_variant(query))
        quality_query = query
    else:
        feature_variant = _feature_query_variant(query) if _FEATURE_QUERY_RE.search(query) else None
        variants = (query, feature_variant, _simplified_query_variant(query), _official_site_variant(query))
        quality_query = (
            feature_variant
            if feature_variant and not _OFFICIAL_QUERY_RE.search(query)
            else query
        )

    combined: list[dict] = []
    seen_urls: set[str] = set()
    errors: list[str] = []
    official_variant = _official_site_variant(query)
    for search_query in dict.fromkeys(variant for variant in variants if variant):
        raw = search(search_query, region)
        if isinstance(raw, str):
            errors.append(raw)
            continue
        for result in raw:
            url = result.get("href") or result.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(result)
        usable = _usable_results(combined, quality_query, require_fresh=news_query)
        enough_sources = len(_dedupe_by_domain(usable, _MIN_OVERVIEW_SOURCES)) >= _MIN_OVERVIEW_SOURCES
        # For known public authorities, two noisy third-party domains are not
        # enough to stop before the deterministic official-site variant runs.
        # Otherwise a search result such as a social post can win simply
        # because it arrived first.
        has_authoritative_source = any(
            _is_authoritative_url(result.get("url") or "") for result in usable
        )
        if enough_sources and (not official_variant or has_authoritative_source):
            break

    if not combined:
        message = errors[-1] if errors and errors[-1].startswith("[error]") else _retry_message(query)
        return {"query": query, "quality_query": quality_query, "candidates": [], "message": message}

    results = [_normalize_result(result) for result in combined]
    results = _filter_noncontent(_filter_adult(results))
    if news_query:
        results = [
            result for result in results
            if result.get("_freshness_relaxed") or _is_fresh_news_result(result, query)
        ]
    relevant = _filter_relevant(results, quality_query)
    if not relevant:
        return {"query": query, "quality_query": quality_query, "candidates": [], "message": _retry_message(query)}

    relevant.sort(
        key=lambda result: (
            _relevance_score(result, quality_query),
            _is_authoritative_url(result.get("url") or ""),
        ),
        reverse=True,
    )
    candidates = _dedupe_by_domain(relevant, config.WEB_SEARCH_MAX_RESULTS)
    tagged = [dict(result, _query=query, _quality_query=quality_query) for result in candidates]
    return {"query": query, "quality_query": quality_query, "candidates": tagged, "message": ""}


def _search_query_batch(queries: list[str]) -> list[dict]:
    """Search independent queries concurrently while preserving query order."""
    if len(queries) == 1:
        state = _search_candidates_for_query(queries[0])
        for result in state["candidates"]:
            result["_query_index"] = 0
        return [state]

    states: list[dict | None] = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=len(queries), thread_name_prefix="lite-web-query") as pool:
        futures = {pool.submit(_search_candidates_for_query, query): index for index, query in enumerate(queries)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                state = future.result()
            except Exception as exc:
                state = {
                    "query": queries[index], "quality_query": queries[index], "candidates": [],
                    "message": f"[error] web search unavailable: {exc}",
                }
            for result in state["candidates"]:
                result["_query_index"] = index
            states[index] = state
    return [state for state in states if state is not None]


def _select_fetch_sources(candidates: list[dict], query_count: int, limit: int) -> list[dict]:
    """Choose top ranked sources fairly across queries, preferring domains."""
    pools: list[list[dict]] = [[] for _ in range(query_count)]
    for result in candidates:
        index = int(result.get("_query_index", 0))
        if 0 <= index < query_count:
            pools[index].append(result)
    chosen: list[dict] = []
    seen_urls: set[str] = set()
    seen_domains: set[str] = set()
    while len(chosen) < limit:
        progressed = False
        for pool in pools:
            candidate_index = None
            # Prefer a new domain, but allow a repeated domain if that is the
            # only approved evidence for an otherwise uncovered query.
            for idx, candidate in enumerate(pool):
                url = candidate.get("url") or ""
                if url and url not in seen_urls and _domain(url) not in seen_domains:
                    candidate_index = idx
                    break
            if candidate_index is None:
                for idx, candidate in enumerate(pool):
                    url = candidate.get("url") or ""
                    if url and url not in seen_urls:
                        candidate_index = idx
                        break
            if candidate_index is None:
                continue
            candidate = pool.pop(candidate_index)
            url = candidate["url"]
            seen_urls.add(url)
            seen_domains.add(_domain(url))
            chosen.append(candidate)
            progressed = True
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    return chosen


def _resolve_public_endpoint(url: str):
    """Resolve once and return the exact public IP that the HTTP pool must use."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        if not addresses:
            return None
        ips: list[str] = []
        for item in addresses:
            value = str(ipaddress.ip_address(item[4][0]))
            if not ipaddress.ip_address(value).is_global:
                return None
            if value not in ips:
                ips.append(value)
        if not ips:
            return None
        return parsed, ips[0], port
    except (OSError, ValueError):
        return None


def _is_safe_fetch_url(url: str) -> bool:
    return _resolve_public_endpoint(url) is not None


def _request_pinned_once(endpoint) -> tuple[int, dict[str, str], bytes]:
    """One direct-IP request with original Host/SNI; never resolve inside the client."""
    import urllib3

    parsed, ip, port = endpoint
    timeout = urllib3.Timeout(connect=min(5, config.WEB_SEARCH_FETCH_TIMEOUT), read=config.WEB_SEARCH_FETCH_TIMEOUT)
    host = parsed.hostname
    headers = dict(_FETCH_HEADERS)
    default_port = 443 if parsed.scheme == "https" else 80
    headers["Host"] = host if port == default_port else f"{host}:{port}"
    target = parsed.path or "/"
    if parsed.params:
        target += ";" + parsed.params
    if parsed.query:
        target += "?" + parsed.query

    if parsed.scheme == "https":
        pool = urllib3.HTTPSConnectionPool(
            ip,
            port=port,
            timeout=timeout,
            maxsize=1,
            block=True,
            retries=False,
            ssl_context=ssl.create_default_context(),
            assert_hostname=host,
            server_hostname=host,
        )
    else:
        pool = urllib3.HTTPConnectionPool(
            ip, port=port, timeout=timeout, maxsize=1, block=True, retries=False,
        )
    response = None
    try:
        response = pool.request(
            "GET", target, headers=headers, redirect=False, preload_content=False, retries=False,
        )
        data = response.read(_FETCH_MAX_BYTES + 1, decode_content=True)
        if len(data) > _FETCH_MAX_BYTES:
            data = data[:_FETCH_MAX_BYTES]
        return int(response.status), {str(k).lower(): str(v) for k, v in response.headers.items()}, data
    finally:
        if response is not None:
            response.release_conn()
        pool.close()


def _decode_response_body(data: bytes, content_type: str) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([A-Za-z0-9._-]+)", content_type or "", re.IGNORECASE)
    if match:
        charset = match.group(1)
    try:
        return data.decode(charset, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


def _fetch_one(url: str) -> str:
    """Fetch one selected result using pinned public IPs and revalidated redirects."""
    current = url
    for redirect_count in range(_FETCH_MAX_REDIRECTS + 1):
        endpoint = _resolve_public_endpoint(current)
        if endpoint is None:
            return ""
        try:
            status, headers, data = _request_pinned_once(endpoint)
        except Exception:
            return ""
        if status in {301, 302, 303, 307, 308}:
            if redirect_count >= _FETCH_MAX_REDIRECTS:
                return ""
            location = headers.get("location", "").strip()
            if not location:
                return ""
            current = urljoin(current, location)
            continue
        if not 200 <= status < 300:
            return ""
        content_type = headers.get("content-type", "").casefold()
        if content_type and not any(kind in content_type for kind in ("text/html", "application/xhtml+xml", "text/plain")):
            return ""
        html = _decode_response_body(data, content_type)
        try:
            import trafilatura
            # MAX_VLM production found concurrent trafilatura/lxml teardown can
            # abort macOS Python. HTTP remains parallel; extraction is serialized.
            with _TRAFILATURA_EXTRACT_LOCK:
                return trafilatura.extract(
                    html, include_comments=False, include_tables=True, output_format="markdown",
                ) or ""
        except Exception:
            return ""
    return ""


def _fetch_batch(urls: list[str]) -> dict[str, str]:
    """Fetch approved URLs concurrently; failed pages remain empty for snippet fallback."""
    unique = list(dict.fromkeys(url for url in urls if url))
    if not unique:
        return {}
    bodies = {url: "" for url in unique}
    with ThreadPoolExecutor(
        max_workers=min(config.WEB_SEARCH_FETCH_TOP, len(unique)),
        thread_name_prefix="lite-web-fetch",
    ) as pool:
        futures = {pool.submit(_fetch_one, url): url for url in unique}
        for future in as_completed(futures):
            url = futures[future]
            try:
                bodies[url] = future.result()
            except Exception:
                bodies[url] = ""
    return bodies


def _result_summary(result: dict, body: str) -> str:
    query = result.get("_query") or ""
    if body:
        filtered = query_filtered_summary(body, query, config.WEB_SEARCH_SUMMARY_CHARS)
        if filtered:
            return filtered
    snippet = result.get("snippet") or ""
    return extractive_summary(snippet, query, config.WEB_SEARCH_SUMMARY_CHARS) if snippet else ""


def run(args: dict) -> str:
    queries, error = _coerce_query_batch(args.get("query"))
    if error:
        return error

    states = _search_query_batch(queries)
    candidates = [candidate for state in states for candidate in state.get("candidates", [])]
    if not candidates:
        if len(queries) == 1:
            return states[0].get("message") or _retry_message(queries[0])
        return "\n\n".join(
            f"[query:{state['query']}]\n{state.get('message') or _retry_message(state['query'])}"
            for state in states
        )

    # Keep the normal single-query source cap, but a 4-query batch must
    # reserve at least one selected source per query when candidates exist.
    # Fetch concurrency remains bounded separately in _fetch_batch().
    selection_limit = max(config.WEB_SEARCH_FETCH_TOP, len(queries))
    selected = _select_fetch_sources(candidates, len(queries), selection_limit)
    fetch_urls = [result["url"] for result in selected]
    bodies = _fetch_batch(fetch_urls) if fetch_urls else {}

    lines_by_query: list[list[str]] = [[] for _ in queries]
    relaxed_by_query = [False for _ in queries]
    for result in selected:
        summary = _result_summary(result, bodies.get(result["url"], ""))
        if not summary:
            continue
        title = (result.get("title") or "").strip()
        line = f"- {title}: {summary}\n  [source: {result['url']}]"
        index = int(result.get("_query_index", 0))
        if 0 <= index < len(lines_by_query):
            lines_by_query[index].append(line)
            relaxed_by_query[index] = relaxed_by_query[index] or bool(result.get("_freshness_relaxed"))

    def freshness_notice(index: int) -> str:
        if not relaxed_by_query[index]:
            return ""
        return (
            "[freshness_relaxed] ไม่พบผลลัพธ์จากตัวกรองช่วงเวลาที่ร้องขอ; "
            "แหล่งข้อมูลด้านล่างมาจากการค้นหาแบบไม่จำกัดเวลา จึงต้องตรวจวันที่/เวลาในต้นฉบับ "
            "ก่อนสรุปว่าเป็นข้อมูลปัจจุบัน\n"
        )

    if len(queries) == 1:
        if not lines_by_query[0]:
            return states[0].get("message") or _retry_message(queries[0])
        return freshness_notice(0) + "\n".join(lines_by_query[0])

    blocks: list[str] = []
    for index, query in enumerate(queries):
        if lines_by_query[index]:
            body = freshness_notice(index) + "\n".join(lines_by_query[index])
        else:
            state = states[index]
            body = state.get("message") or _retry_message(query)
        blocks.append(f"[query:{query}]\n{body}")
    return "\n\n".join(blocks)


# LangChain entry point — see tools/bash.py's 2026-07-23 note.
# PLAN.md §9.5 docstring-split pattern (2026-07-25) applied here: the
# citation requirement only matters AFTER results come back (it governs how
# the model writes its final answer, not whether to call this tool), so it's
# deferred into this tool's own return value on the first call each turn
# rather than sitting in the always-bound schema. See tools/bash.py for the
# original design/verification writeup and helpers.turn_guard.call_count().
_CITATION_REMINDER = (
    "\n\n[reminder] Write a concise overview that combines the relevant findings, "
    "not a raw result list. If [needs_retry] appears, refine the query and search "
    "again before answering. Do not copy or type URLs in the answer; code appends "
    "the filtered source links automatically."
)

from langchain_core.tools import tool as _lc_tool


@_lc_tool
def web_search(query: str | list[str]) -> str:
    """Search the internet for current/external information and return compact source evidence.

    Pass query="..." for one search. For 2-4 distinct facts or angles, prefer
    query=["q1", "q2", ...] in ONE tool call; those searches run concurrently,
    then code-only relevance ranking normally selects at most three source pages;
    a four-query batch may select four so every query can retain one source.
    Selected pages are reduced to short query-specific passages with URLs.

    Anything that changes over time (today's weather, today's news, current
    prices, recent events) always needs this tool -- even if it feels
    answerable from memory, memory is not current data.
    ❌ "What's the weather like today" -> do not answer from general climate knowledge
    ✅ always call web_search(query="...") first for this kind of question

    If a result says [needs_retry], refine the query with the exact subject,
    location, and time period and call this tool again (at most the per-turn
    quota). Final response: synthesize an overview; do not copy URLs because
    code appends only the filtered relevant links.
    Retry SELF-CHECK: never add an answer-like number/fact that was absent from
    the user query and prior results. Quote the core topic or name its official
    source instead.
    ❌ policy-rate search fails -> retry with a guessed "8.0%"
    ✅ retry with 'site:bot.or.th "อัตราดอกเบี้ยนโยบาย"'

    Do NOT use this for a request to find a local file -- that is rag_search's
    job, not this tool's, even if no local index exists yet."""
    from helpers import turn_guard
    from helpers import search_scope
    if search_scope.is_outside_knowledge_base_query(turn_guard.user_query()):
        return (
            "[blocked] คำขอนี้ระบุให้ค้นนอก Knowledge Base/RAG "
            "จึงต้องใช้ bash กับ Desktop/Documents แทน web_search"
        )
    if search_scope.is_knowledge_base_explanation_query(turn_guard.user_query()):
        return (
            "[blocked] คำถามนี้เป็นการอธิบายความหมายหรือการทำงานของ RAG "
            "ไม่ใช่คำขอค้นเว็บ; ให้ตอบจากความรู้ทั่วไปโดยไม่ใช้ web_search"
        )
    if search_scope.is_knowledge_base_query(turn_guard.user_query()):
        return (
            "[blocked] คำขอนี้ระบุ Knowledge Base/RAG ให้ใช้ "
            "mcp_call_tool(server='lite-rag', tool_name='rag_search') ก่อน "
            "แทน web_search"
        )
    if search_scope.is_local_document_query(turn_guard.user_query()):
        return (
            "[blocked] คำขอนี้เป็นการค้นหาไฟล์/เอกสารในเครื่อง ให้ใช้ "
            "mcp_call_tool(server='lite-rag', tool_name='rag_search') แทน web_search"
        )
    args = {"query": query}
    blocked = turn_guard.check("web_search", args)
    if blocked:
        return blocked
    first_call_this_turn = turn_guard.call_count("web_search") == 1
    result = run(args)
    turn_guard.record("web_search", args, result)
    if first_call_this_turn:
        result += _CITATION_REMINDER
    return result
