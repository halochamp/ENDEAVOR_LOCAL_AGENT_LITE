# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""Quality gates for DDGS results before Lite cites them to users."""
import sys
from types import SimpleNamespace

import pytest

from tools import web_search
import config


@pytest.fixture(autouse=True)
def _disable_live_semantic_filter(monkeypatch):
    """Unit tests never make a resident-model call unless explicitly testing it."""
    monkeypatch.setattr(config, "WEB_SEARCH_SEMANTIC_FILTER", False)


def test_default_search_breadth_is_ten_results_and_four_sources():
    assert config.WEB_SEARCH_MAX_RESULTS == 10
    assert config.WEB_SEARCH_FETCH_TOP == 4


def test_semantic_decision_parser_accepts_only_numbered_yes_no_lines():
    assert web_search._parse_semantic_decisions(
        "1 YES\n2 NO\n3 YES", 3,
    ) == {0, 2}
    assert web_search._parse_semantic_decisions("explanation only", 2) is None


def test_semantic_filter_batches_results_and_rejects_intent_mismatch(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_SEMANTIC_FILTER", True)
    captured = {}

    class _FakeModel:
        def invoke(self, messages):
            captured["prompt"] = messages[-1].content
            return SimpleNamespace(content="1 NO\n2 YES")

    monkeypatch.setattr(web_search, "_semantic_filter_model", lambda: _FakeModel())
    results = [
        {
            "title": "Bitcoin Halving คืออะไร",
            "snippet": "บทความเกี่ยวกับวงจร Halving",
            "url": "https://crypto.test/halving",
        },
        {
            "title": "Bitcoin Price Today",
            "snippet": "Live Bitcoin price and market data",
            "url": "https://market.test/bitcoin",
        },
    ]

    selected = web_search._semantic_reject_irrelevant(results, "ราคา Bitcoin วันนี้")

    assert selected == [results[1]]
    assert "crypto.test" not in captured["prompt"]
    assert "Bitcoin Halving" in captured["prompt"]


def test_run_calls_semantic_llm_once_for_the_whole_candidate_batch(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_SEMANTIC_FILTER", True)
    calls = []

    class _FakeModel:
        def invoke(self, messages):
            calls.append(messages[-1].content)
            return SimpleNamespace(content="1 YES\n2 NO\n3 YES")

    monkeypatch.setattr(web_search, "_semantic_filter_model", lambda: _FakeModel())
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "Bitcoin Price Today — Market A",
            "body": "Live Bitcoin price and market data",
            "href": "https://market-a.test/bitcoin",
        },
        {
            "title": "Bitcoin Price Today — Market B",
            "body": "A superficially matching but unusable result",
            "href": "https://market-b.test/bitcoin",
        },
        {
            "title": "Live Bitcoin Price — Market C",
            "body": "Current Bitcoin price and market data",
            "href": "https://market-c.test/bitcoin",
        },
    ])

    output = web_search.run({"query": "ราคา Bitcoin วันนี้"})

    assert len(calls) == 1
    assert all(name in calls[0] for name in ("Market A", "Market B", "Market C"))
    assert "market-a.test" in output
    assert "market-b.test" not in output
    assert "market-c.test" in output


def test_retry_variant_simplifies_subject_without_inventing_answer_terms():
    assert web_search._simplified_query_variant(
        "อัตราดอกเบี้ยนโยบายไทยล่าสุด 2569"
    ) == "อัตราดอกเบี้ยนโยบายไทย 2569"
    assert web_search._simplified_query_variant(
        "กำหนดยื่นภาษีเงินได้บุคคลธรรมดา 2569"
    ) == "กำหนดยื่นภาษีเงินได้บุคคลธรรมดา 2569"
    assert web_search._topic_anchors(
        "วิธีดูแลกล้วยไม้สำหรับมือใหม่"
    ) == ["ดูแลกล้วยไม้"]
    assert web_search._official_site_variant(
        "อัตราดอกเบี้ยนโยบายไทยล่าสุด 2026"
    ) == 'site:bot.or.th "อัตราดอกเบี้ยนโยบาย"'


def test_run_tries_bounded_simplified_subject_variant(monkeypatch):
    calls = []

    def fake_search(query, _region):
        calls.append(query)
        if query == "อัตราดอกเบี้ยนโยบายไทย 2569":
            return [
                {
                    "title": "อัตราดอกเบี้ยนโยบายไทย 2569",
                    "body": "ประกาศอัตราดอกเบี้ยนโยบายไทย 2569",
                    "href": "https://central-bank.test/policy-rate",
                },
                {
                    "title": "สรุปอัตราดอกเบี้ยนโยบายไทย 2569",
                    "body": "ภาพรวมอัตราดอกเบี้ยนโยบายไทย 2569",
                    "href": "https://economy.test/policy-rate",
                },
            ]
        return [
            {
                "title": "อัตราดอกเบี้ยบ้าน 2569",
                "body": "เปรียบเทียบสินเชื่อบ้าน",
                "href": "https://home-loan.test/rates",
            }
        ]

    monkeypatch.setattr(web_search, "_search_engines", fake_search)

    output = web_search.run({"query": "อัตราดอกเบี้ยนโยบายไทยล่าสุด 2569"})

    assert calls == [
        "อัตราดอกเบี้ยนโยบายไทยล่าสุด 2569",
        "อัตราดอกเบี้ยนโยบายไทย 2569",
    ]
    assert "central-bank.test/policy-rate" in output
    assert "economy.test/policy-rate" in output
    assert "home-loan.test" not in output


def test_filters_search_and_account_landing_pages():
    results = [
        {"title": "Google", "snippet": "Search", "url": "https://www.google.com/?hl=th"},
        {"title": "Personal info", "snippet": "Manage account", "url": "https://myaccount.google.com/intro/personal-info?hl=th"},
        {"title": "Channel", "snippet": "Videos", "url": "https://www.youtube.com/channel/UCrFDdD-EE05N7gjwZho2wqw"},
        {"title": "Useful article", "snippet": "Relevant independent reporting", "url": "https://example.org/article"},
    ]

    assert web_search._filter_noncontent(results) == [results[-1]]


def test_run_never_cites_filtered_navigation_results(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {"title": "Google", "body": "Search", "href": "https://www.google.com/?hl=th"},
        {"title": "Current topic report", "body": "Independent reporting about the current topic.", "href": "https://example.org/report"},
    ])

    output = web_search.run({"query": "current topic"})

    assert "https://example.org/report" in output
    assert "google.com" not in output


def test_run_cites_only_lexically_relevant_results(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "อัตราดอกเบี้ยนโยบาย ธนาคารแห่งประเทศไทย",
            "body": "คณะกรรมการนโยบายการเงินประกาศอัตราดอกเบี้ยนโยบายไทย",
            "href": "https://www.bot.or.th/policy-rate",
        },
        {
            "title": "อัตราดอกเบี้ยบ้านปี 2569",
            "body": "เปรียบเทียบสินเชื่อบ้านและวงเงินกู้ของแต่ละธนาคาร",
            "href": "https://example.com/home-loan",
        },
        {
            "title": "ผลประชุม กนง. และดอกเบี้ยนโยบายไทย 2569",
            "body": "สรุปมติอัตราดอกเบี้ยนโยบายล่าสุด",
            "href": "https://example.org/mpc-summary",
        },
    ])

    output = web_search.run({"query": "อัตราดอกเบี้ยนโยบายไทยล่าสุด 2569"})

    assert "bot.or.th/policy-rate" in output
    assert "example.org/mpc-summary" in output
    assert "home-loan" not in output
    assert "[needs_retry]" not in output


def test_run_requests_refined_search_when_results_are_irrelevant(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "สูตรทำขนมปัง",
            "body": "วัตถุดิบและวิธีอบขนม",
            "href": "https://example.com/bread",
        },
    ])

    output = web_search.run({"query": "กำหนดยื่นภาษีเงินได้บุคคลธรรมดา 2569"})

    assert output.startswith("[needs_retry]")
    assert "https://" not in output


def test_empty_news_backend_also_requests_refined_search(monkeypatch):
    monkeypatch.setattr(web_search, "_search_news", lambda *_args: "(no recent news results found)")

    output = web_search.run({"query": "ข่าวเทคโนโลยีวันนี้"})

    assert output.startswith("[needs_retry]")


def test_single_relevant_source_is_kept_without_forcing_another_search(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "กำหนดยื่นภาษีเงินได้บุคคลธรรมดา 2569",
            "body": "ยื่นภาษีออนไลน์ภายในเดือนเมษายน 2569",
            "href": "https://tax-news.test/tax-2569",
        },
    ])

    output = web_search.run({"query": "กำหนดยื่นภาษีเงินได้บุคคลธรรมดา 2569"})

    assert "https://tax-news.test/tax-2569" in output
    assert "[needs_retry]" not in output


def test_single_authoritative_source_is_sufficient(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "กำหนดยื่นภาษีเงินได้บุคคลธรรมดา 2569",
            "body": "ยื่นภาษีออนไลน์ภายในเดือนเมษายน 2569",
            "href": "https://www.rd.go.th/tax-2569",
        },
    ])

    output = web_search.run({"query": "กำหนดยื่นภาษีเงินได้บุคคลธรรมดา 2569"})

    assert "https://www.rd.go.th/tax-2569" in output
    assert "[needs_retry]" not in output


def test_explicit_conflicting_year_is_not_cited(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "ภาษีเงินได้บุคคลธรรมดา 2567",
            "body": "กำหนดวันยื่นภาษีปี 2567",
            "href": "https://example.com/tax-2567",
        },
        {
            "title": "ภาษีเงินได้บุคคลธรรมดา 2569",
            "body": "กำหนดวันยื่นภาษีปี 2569",
            "href": "https://example.org/tax-2569",
        },
    ])

    output = web_search.run({"query": "กำหนดยื่นภาษีเงินได้บุคคลธรรมดา 2569"})

    assert "tax-2567" not in output
    assert "tax-2569" in output


def test_buddhist_and_gregorian_equivalent_years_are_not_a_conflict(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "อัตราดอกเบี้ยนโยบายไทย 2569",
            "body": "สรุปมติอัตราดอกเบี้ยนโยบายล่าสุดปี 2569",
            "href": "https://economy.test/policy-2569",
        },
    ])

    output = web_search.run({"query": "อัตราดอกเบี้ยนโยบายไทย 2026 ล่าสุด"})

    assert "economy.test/policy-2569" in output


def test_search_engine_fallback_runs_when_first_backend_is_irrelevant(monkeypatch):
    calls = []

    class _FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(self, _query, *, max_results, region, backend, timelimit=None):
            calls.append(backend)
            if backend == "duckduckgo":
                return [{"title": "Bread recipe", "body": "Bake bread", "href": "https://food.test/bread"}]
            return [
                {
                    "title": "Thailand policy interest rate current",
                    "body": "Current Thailand policy interest rate decision",
                    "href": "https://central-bank.test/rate",
                },
                {
                    "title": "Current Thailand policy interest rate analysis",
                    "body": "Thailand policy interest rate overview",
                    "href": "https://economy.test/rate",
                },
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=_FakeDDGS))

    results = web_search._search_engines("current Thailand policy interest rate", "us-en")

    assert calls == ["duckduckgo", "bing"]
    assert isinstance(results, list)
    assert len(results) == 3


def test_news_results_require_recent_backend_date(monkeypatch):
    now = web_search.datetime.now(web_search.timezone.utc)
    monkeypatch.setattr(web_search, "_search_news", lambda *_args: [
        {
            "date": (now - web_search.timedelta(days=30)).isoformat(),
            "title": "ข่าวเศรษฐกิจไทยเก่า",
            "body": "ข่าวเศรษฐกิจไทยจากเดือนก่อน",
            "url": "https://old-news.test/economy",
        },
        {
            "date": now.isoformat(),
            "title": "เศรษฐกิจไทยวันนี้ด้านการส่งออก",
            "body": "รายงานข่าวเศรษฐกิจไทยวันนี้เกี่ยวกับการส่งออก",
            "url": "https://fresh-news.test/exports",
        },
        {
            "date": now.isoformat(),
            "title": "เศรษฐกิจไทยวันนี้ด้านการลงทุน",
            "body": "รายงานข่าวเศรษฐกิจไทยวันนี้เกี่ยวกับการลงทุน",
            "url": "https://second-news.test/investment",
        },
    ])

    output = web_search.run({"query": "ข่าวเศรษฐกิจไทยวันนี้ ภาพรวม"})

    assert "old-news.test" not in output
    assert "fresh-news.test/exports" in output
    assert "second-news.test/investment" in output


def test_title_gate_rejects_page_that_only_mentions_topic_in_snippet(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "ประกันบำนาญที่ไหนดี สำหรับลดหย่อนภาษี 2569",
            "body": "แนวโน้มอัตราดอกเบี้ยนโยบายไทย 2569 อาจเป็นขาลง",
            "href": "https://insurance.test/annuity",
        },
    ])

    output = web_search.run({"query": "อัตราดอกเบี้ยนโยบายไทยล่าสุด 2569"})

    assert output.startswith("[needs_retry]")
    assert "insurance.test" not in output


def test_deadline_intent_rejects_tax_calculator_page(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "คำนวณภาษีเงินได้บุคคลธรรมดา 2569",
            "body": "รายได้จากการขายออนไลน์ถือเป็นเงินได้จากธุรกิจการค้า",
            "href": "https://calculator.test/personal-tax",
        },
    ])

    output = web_search.run({"query": "กำหนดยื่นภาษีเงินได้บุคคลธรรมดา 2569"})

    assert output.startswith("[needs_retry]")
    assert "calculator.test" not in output


def test_price_intent_rejects_related_market_article(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "Bitcoin Halving คืออะไร",
            "body": "นักลงทุนรอการเติบโตของมูลค่า Bitcoin ในอนาคต",
            "href": "https://crypto.test/halving",
        },
    ])

    output = web_search.run({"query": "ราคา Bitcoin ล่าสุดวันนี้"})

    assert output.startswith("[needs_retry]")
    assert "crypto.test" not in output


def test_version_and_feature_intent_must_both_match(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "Python Release Python 3.8.0",
            "body": "Major new features of Python 3.8",
            "href": "https://python.test/3.8",
        },
        {
            "title": "python-pydantic fails to build with Python 3.14",
            "body": "Build options include feature flags",
            "href": "https://bugs.test/python-3.14",
        },
        {
            "title": "What's new in Python 3.14",
            "body": "New Python 3.14 features and changes",
            "href": "https://docs.test/python-3.14",
        },
    ])

    output = web_search.run({"query": "Python 3.14 latest features"})

    assert "python.test/3.8" not in output
    assert "bugs.test/python-3.14" not in output
    assert "docs.test/python-3.14" in output


def test_official_source_constraint_uses_entity_domain_not_brand_allowlist(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "What's New in Node.js 24?",
            "body": "An independent overview of Node.js 24 features",
            "href": "https://third-party.test/node-24",
        },
        {
            "title": "Node.js 24 release notes and features",
            "body": "Official Node.js 24 release documentation and new features",
            "href": "https://nodejs.org/en/blog/release/v24",
        },
    ])

    output = web_search.run({"query": "Node.js 24 new features official sources"})

    assert "third-party.test" not in output
    assert "nodejs.org/en/blog/release/v24" in output
    assert "[needs_retry]" not in output


def test_current_fact_query_rejects_forecast_article(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "แนวโน้มอัตราดอกเบี้ยนโยบายไทยปี 2569",
            "body": "คาดว่าอัตราดอกเบี้ยอาจลดลงในอนาคต",
            "href": "https://forecast.test/policy-rate",
        },
    ])

    output = web_search.run({"query": "อัตราดอกเบี้ยนโยบายไทยล่าสุด 2569"})

    assert output.startswith("[needs_retry]")
    assert "forecast.test" not in output
