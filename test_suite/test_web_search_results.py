"""Quality gates for DDGS results before Lite cites them to users."""
import sys
from types import SimpleNamespace

from tools import web_search
import config


def test_default_search_breadth_is_ten_results_and_three_sources():
    assert config.WEB_SEARCH_MAX_RESULTS == 10
    assert config.WEB_SEARCH_FETCH_TOP == 3


def test_freshness_classification_uses_real_markers_not_substrings():
    assert web_search._is_time_sensitive("ราคาทองปัจจุบัน") is True
    assert web_search._query_timelimit("ราคาทองปัจจุบัน") == "w"
    assert web_search._is_news_query("newspaper history") is False
    assert web_search._is_news_query("newsletter software") is False
    assert web_search._query_timelimit("newspaper history") is None
    assert web_search._query_timelimit("todayish product name") is None
    assert web_search._query_timelimit("latest AI news") == "w"
    assert web_search._news_core_variant("latest Snowflake news") == "Snowflake news"
    assert web_search._news_core_variant("news about Snowflake") == "about Snowflake"


def test_run_selects_top_three_ranked_sources_without_model_call(monkeypatch):
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

    assert "_semantic_filter_model" not in vars(web_search)
    assert "market-a.test" in output
    assert "market-b.test" in output
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
        'site:bot.or.th "อัตราดอกเบี้ยนโยบาย"',
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


def test_time_filtered_empty_search_retries_once_without_filter_and_marks_result(monkeypatch):
    calls = []

    class _FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(self, _query, *, max_results, region, backend, timelimit=None):
            calls.append((backend, timelimit))
            if timelimit:
                return []
            return [
                {
                    "title": "Thailand policy interest rate current",
                    "body": "Current Thailand policy interest rate decision",
                    "href": "https://central-bank.test/rate",
                },
                {
                    "title": "Thailand policy interest rate current analysis",
                    "body": "Thailand policy interest rate overview",
                    "href": "https://economy.test/rate",
                },
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=_FakeDDGS))

    results = web_search._search_engines("current Thailand policy interest rate", "us-en")

    assert calls == [("duckduckgo", "w"), ("bing", "w"), ("duckduckgo", None)]
    assert isinstance(results, list)
    assert all(row["_freshness_relaxed"] for row in results)


def test_relaxed_freshness_is_disclosed_with_source_urls(monkeypatch):
    monkeypatch.setattr(web_search, "_search_engines", lambda *_args: [
        {
            "title": "Thailand policy interest rate current",
            "body": "Current Thailand policy interest rate decision",
            "href": "https://central-bank.test/rate",
            "_freshness_relaxed": True,
        },
        {
            "title": "Thailand policy interest rate current analysis",
            "body": "Thailand policy interest rate overview",
            "href": "https://economy.test/rate",
            "_freshness_relaxed": True,
        },
    ])

    output = web_search.run({"query": "current Thailand policy interest rate"})

    assert output.startswith("[freshness_relaxed]")
    assert "[source: https://central-bank.test/rate]" in output


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


def test_multi_query_runs_four_searches_in_parallel_and_keeps_one_source_each(monkeypatch):
    import threading

    barrier = threading.Barrier(4)
    search_done = []
    queries = ["alpha revenue", "beta margin", "gamma guidance", "delta capex"]

    def fake_search_one(query):
        barrier.wait(timeout=2)
        search_done.append(query)
        idx = queries.index(query)
        return {
            "query": query,
            "quality_query": query,
            "message": "",
            "candidates": [{
                "title": f"Result {idx + 1} for {query}",
                "snippet": f"snippet about {query}",
                "url": f"https://source-{idx + 1}.test/article",
                "date": None,
                "_query": query,
                "_quality_query": query,
            }],
        }

    fetched = []

    def fake_fetch(urls):
        assert len(search_done) == 4
        fetched.extend(urls)
        return {
            "https://source-1.test/article": "Unrelated navigation\n\nAlpha revenue reached 100 million.",
            "https://source-2.test/article": "Beta margin is 20 percent.\n\nOther material.",
            "https://source-3.test/article": "Gamma guidance calls for 12 percent growth.\n\nOther material.",
            "https://source-4.test/article": "Delta capex is 8 billion.\n\nOther material.",
        }

    monkeypatch.setattr(web_search, "_search_candidates_for_query", fake_search_one)
    monkeypatch.setattr(web_search, "_fetch_batch", fake_fetch)

    output = web_search.run({"query": queries})

    assert fetched == [
        "https://source-1.test/article",
        "https://source-2.test/article",
        "https://source-3.test/article",
        "https://source-4.test/article",
    ]
    assert "source-2.test" in output
    assert "Alpha revenue reached 100 million" in output
    assert "Beta margin is 20 percent" in output
    assert "Gamma guidance calls for 12 percent growth" in output
    assert "Delta capex is 8 billion" in output
    assert "Unrelated navigation" not in output
    assert "source-4.test" in output


def test_multi_query_rejects_more_than_four_and_schemas_expose_array_form():
    result = web_search.run({"query": ["a", "b", "c", "d", "e"]})
    assert "สูงสุด 4" in result

    manual_query = web_search.SCHEMA["function"]["parameters"]["properties"]["query"]
    assert manual_query["oneOf"][1]["maxItems"] == 4
    lc_query = web_search.web_search.args_schema.model_json_schema()["properties"]["query"]
    assert "array" in str(lc_query).lower()


def test_json_list_query_string_is_accepted_and_deduplicated():
    queries, error = web_search._coerce_query_batch('["alpha", "beta", "alpha"]')
    assert error is None
    assert queries == ["alpha", "beta"]


def test_fetched_body_with_no_query_match_falls_back_to_search_snippet():
    result = {
        "title": "Policy rate",
        "snippet": "Current policy rate evidence",
        "url": "https://example.test/rate",
        "_query": "policy rate",
    }
    summary = web_search._result_summary(result, "Navigation menu only\n\nContact us")
    assert "Current policy rate evidence" in summary
    assert "Navigation menu" not in summary


def test_langchain_web_search_accepts_native_query_list(monkeypatch):
    from helpers import turn_guard

    def fake_search_one(query):
        return {
            "query": query,
            "quality_query": query,
            "message": "",
            "candidates": [{
                "title": f"{query} source",
                "snippet": f"evidence for {query}",
                "url": f"https://{query.split()[0]}.test/source",
                "date": None,
                "_query": query,
                "_quality_query": query,
            }],
        }

    monkeypatch.setattr(web_search, "_search_candidates_for_query", fake_search_one)
    monkeypatch.setattr(web_search, "_fetch_batch", lambda _urls: {})
    turn_guard.reset("web-search-list-invoke")

    output = web_search.web_search.invoke({"query": ["alpha revenue", "beta margin"]})

    assert "[query:alpha revenue]" in output
    assert "[query:beta margin]" in output
    assert "alpha.test/source" in output
    assert "beta.test/source" in output


def test_four_query_batch_reserves_at_least_one_source_per_query(monkeypatch):
    queries = ["alpha topic", "beta topic", "gamma topic", "delta topic"]

    def fake_search_one(query):
        return {
            "query": query,
            "quality_query": query,
            "message": "",
            "candidates": [{
                "title": f"{query} source",
                "snippet": f"evidence for {query}",
                "url": f"https://{query.split()[0]}.test/source",
                "date": None,
                "_query": query,
                "_quality_query": query,
            }],
        }

    fetched = []
    monkeypatch.setattr(web_search, "_search_candidates_for_query", fake_search_one)
    monkeypatch.setattr(web_search, "_fetch_batch", lambda urls: fetched.extend(urls) or {})

    output = web_search.run({"query": queries})

    assert fetched == [
        "https://alpha.test/source",
        "https://beta.test/source",
        "https://gamma.test/source",
        "https://delta.test/source",
    ]
    for query in queries:
        assert f"[query:{query}]" in output
        assert f"https://{query.split()[0]}.test/source" in output


def test_ranked_result_fetches_page_without_semantic_model(monkeypatch):
    monkeypatch.setattr(web_search, "_search_candidates_for_query", lambda query: {
        "query": query,
        "quality_query": query,
        "message": "",
        "candidates": [{
            "title": "Alpha revenue source",
            "snippet": "Alpha revenue snippet evidence",
            "url": "https://alpha.test/source",
            "date": None,
            "_query": query,
            "_quality_query": query,
        }],
    })

    fetched = []
    monkeypatch.setattr(web_search, "_fetch_batch", lambda urls: fetched.extend(urls) or {})

    output = web_search.run({"query": "alpha revenue"})

    assert fetched == ["https://alpha.test/source"]
    assert "Alpha revenue snippet evidence" in output
    assert "alpha.test/source" in output
