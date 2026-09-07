from research_assistant.retrieval.expand import parse_query_json, template_queries
from research_assistant.retrieval.hybrid import _normalize_arxiv_id, reciprocal_rank_fusion
from research_assistant.retrieval.filters import passes_filters, topk_overlap
from research_assistant.config import RetrievalConfig


def test_rrf_prefers_docs_high_in_multiple_lists():
    semantic = ["a", "b", "c"]
    keyword = ["c", "a", "d"]
    scores = reciprocal_rank_fusion([semantic, keyword], k=60)
    # a is 2nd+2nd, c is 3rd+1st — a should beat b; c should be competitive with a
    assert scores["a"] > scores["b"]
    assert scores["c"] > scores["b"]
    assert "d" in scores


def test_rrf_ignores_duplicates_in_one_ranking():
    scores = reciprocal_rank_fusion([["a", "a", "b"]], k=60)
    assert scores["a"] == 1.0 / 61
    assert scores["b"] == 1.0 / 63


def test_parse_query_json_from_markdown_fence():
    raw = """```json
    {"queries": ["core-set selection", "dataset condensation", "influence functions"]}
    ```"""
    queries = parse_query_json(raw, n=2)
    assert queries == ["core-set selection", "dataset condensation"]


def test_template_queries_include_original_and_are_unique():
    queries = template_queries("dataset pruning", n=4)
    assert queries[0] == "dataset pruning"
    assert len(queries) == len(set(q.lower() for q in queries))
    assert len(queries) >= 4


def test_normalize_arxiv_id_strips_version_and_url():
    assert _normalize_arxiv_id("https://arxiv.org/abs/2205.09329v2") == "2205.09329"
    assert _normalize_arxiv_id("2205.09329v1") == "2205.09329"
    assert _normalize_arxiv_id("hep-th/9901001v3") == "hep-th/9901001"


def test_year_and_category_filters():
    cfg = RetrievalConfig(year_from=2023, year_to=2025, categories=("cs.LG",))
    assert passes_filters({"year": 2024, "matched_categories": "cs.LG cs.AI"}, cfg)
    assert not passes_filters({"year": 2021, "matched_categories": "cs.LG"}, cfg)
    assert not passes_filters({"year": 2024, "matched_categories": "cs.CL"}, cfg)


def test_topk_overlap():
    assert topk_overlap(["a", "b", "c"], ["a", "x"]) == 0.5
    assert topk_overlap(["a"], []) == 0.0


def test_fetch_citation_counts_parses_batch_and_retries_429(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code, payload=None, headers=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}
            self.text = text

        def json(self):
            return self._payload

    calls = {"n": 0}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, params=None, json=None, headers=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResponse(429, headers={"Retry-After": "0"})
            ids = json["ids"]
            assert all(x.startswith("ARXIV:") for x in ids)
            assert headers.get("x-api-key") == "test-key"
            return FakeResponse(
                200,
                payload=[
                    {"citationCount": 12},
                    None,
                    {"citationCount": 3},
                ],
            )

    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
    monkeypatch.setattr("research_assistant.retrieval.hybrid.httpx.Client", FakeClient)
    monkeypatch.setattr("research_assistant.retrieval.hybrid.time.sleep", lambda s: None)

    from research_assistant.retrieval.hybrid import fetch_citation_counts

def test_gemini_response_text_skips_thought_parts():
    from research_assistant.llm.client import _gemini_response_text

    text = _gemini_response_text(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "scratchpad"},
                            {"text": '{"queries":["a"]}'},
                        ]
                    }
                }
            ]
        }
    )
    assert text == '{"queries":["a"]}'
