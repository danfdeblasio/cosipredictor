"""Offline tests for experiments/wikipedia (no network: the API is replaced by fakes)."""
import importlib.util
import io
import os
import urllib.error

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_DIR = os.path.join(ROOT, "experiments", "wikipedia")


def _load(name):
    path = os.path.join(WIKI_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"wikipedia_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wd():
    return _load("wikidata")


# --- collection listing -------------------------------------------------------

def test_talk_page_maps_to_article(wd):
    assert wd.talk_to_article("Talk:BLAST (biotechnology)") == "BLAST (biotechnology)"
    assert wd.talk_to_article("BLAST (biotechnology)") == "BLAST (biotechnology)"


def test_collection_category_names(wd):
    assert wd.collection_category("GA") == "Category:GA-Class Computational Biology articles"
    assert set(wd.COLLECTION_CLASSES) == {"GA", "B", "C", "Start", "Stub", "List", "Redirect"}


def test_members_follows_continuation(wd):
    calls = []

    def api(params):
        calls.append(dict(params))
        if "cmcontinue" not in params:
            return {"query": {"categorymembers": [{"title": "Talk:A"}]}, "continue": {"cmcontinue": "x"}}
        return {"query": {"categorymembers": [{"title": "Talk:B"}]}}

    titles = [m["title"] for m in wd.members("Category:X", api=api)]
    assert titles == ["Talk:A", "Talk:B"]
    assert calls[1]["cmcontinue"] == "x" and calls[0]["cmtitle"] == "Category:X"


def test_collection_maps_talk_pages_to_classes(wd):
    def api(params):
        cls = params["cmtitle"].split(":", 1)[1].split("-Class")[0]
        return {"query": {"categorymembers": [{"title": f"Talk:{cls} article"}]}}

    coll = wd.collection(api=api)
    assert coll["GA article"] == "GA" and coll["Stub article"] == "Stub"
    assert len(coll) == len(wd.COLLECTION_CLASSES)


# --- biography flag -----------------------------------------------------------

@pytest.mark.parametrize("cats,expected", [
    (["1973 births", "Dutch bioinformaticians"], True),
    (["Living people"], True),
    (["1990s deaths"], True),
    (["University of Cambridge alumni"], True),
    (["Bioinformatics software", "Free software"], False),
    ([], False),
])
def test_is_bio(wd, cats, expected):
    assert wd.is_bio(cats) is expected


# --- API retry ----------------------------------------------------------------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _http_429(retry_after):
    hdrs = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError("u", 429, "Too Many Requests", hdrs, None)


def test_api_retries_429_honouring_retry_after(wd):
    attempts, sleeps = [], []

    def opener(req, timeout=None):
        attempts.append(req.full_url)
        if len(attempts) == 1:
            raise _http_429("7")
        return _Resp(b'{"query": {"ok": 1}}')

    out = wd.api({"action": "query"}, opener=opener, sleep=sleeps.append)
    assert out == {"query": {"ok": 1}}
    assert sleeps == [7]
    assert "maxlag=5" in attempts[0] and "format=json" in attempts[0]


def test_api_backs_off_without_retry_after(wd):
    sleeps = []
    n = {"i": 0}

    def opener(req, timeout=None):
        n["i"] += 1
        if n["i"] < 3:
            raise _http_429(None)
        return _Resp(b"{}")

    wd.api({}, opener=opener, sleep=sleeps.append)
    assert sleeps == [10, 20]


def test_api_raises_other_http_errors(wd):
    def opener(req, timeout=None):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        wd.api({}, opener=opener, sleep=lambda s: None)


def test_api_gives_up(wd):
    def opener(req, timeout=None):
        raise _http_429("1")

    with pytest.raises(wd.WikipediaAPIError):
        wd.api({}, opener=opener, sleep=lambda s: None, retries=3)


def test_api_sends_project_user_agent(wd):
    import wiki
    seen = {}

    def opener(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        return _Resp(b"{}")

    wd.api({}, opener=opener, sleep=lambda s: None)
    assert seen["ua"] == wiki.USER_AGENT


# --- batched fetch ------------------------------------------------------------

BATCH_RESPONSES = [
    {"query": {
        "normalized": [{"from": "blast", "to": "Blast"}],
        "redirects": [{"from": "Blast", "to": "BLAST (biotechnology)"}],
        "pages": [
            {"title": "BLAST (biotechnology)", "fullurl": "https://en.wikipedia.org/wiki/BLAST_(biotechnology)",
             "description": "Bioinformatics algorithm", "length": 5000, "extract": " BLAST is ... ",
             "categories": [{"title": "Category:Bioinformatics software"}]},
            {"title": "Mercury", "pageprops": {"disambiguation": ""}, "length": 10, "extract": "Mercury may refer to"},
            {"title": "No such page", "missing": True},
        ]},
     "continue": {"clcontinue": "1|x"}},
    {"query": {"pages": [
        {"title": "BLAST (biotechnology)", "categories": [{"title": "Category:Sequence alignment algorithms"}]},
    ]}},
]


def test_fetch_batch_parses_pages(wd):
    sent = []

    def api(params):
        sent.append(dict(params))
        return BATCH_RESPONSES[len(sent) - 1]

    recs = wd.fetch_batch(["blast", "Mercury", "No such page"], api=api)
    by = {r["query"]: r for r in recs}
    assert [r["query"] for r in recs] == ["blast", "Mercury", "No such page"]
    b = by["blast"]
    assert b["title"] == "BLAST (biotechnology)" and b["status"] == "ok"
    assert b["text"] == "BLAST is ..." and b["length"] == 5000
    assert b["categories"] == ["Bioinformatics software", "Sequence alignment algorithms"]
    assert by["Mercury"]["status"] == "disambiguation"
    assert by["No such page"]["status"] == "missing"
    assert sent[0]["titles"] == "blast|Mercury|No such page"
    assert sent[0]["exintro"] == "1" and sent[0]["exlimit"] == "20" and sent[0]["redirects"] == "1"
    assert sent[1]["clcontinue"] == "1|x"


def test_fetch_batch_rejects_more_than_20(wd):
    with pytest.raises(ValueError):
        wd.fetch_batch([f"T{i}" for i in range(21)], api=lambda p: {})


def test_record_fields_match_wiki_to_fields(wd):
    import wiki
    rec = {"query": "x", "title": "T", "url": "", "description": "D", "categories": ["C1", "C2"],
           "text": "Lead.", "length": 1, "status": "ok"}
    expected = wiki.to_fields({"title": "T", "description": "D", "categories": ["C1", "C2"], "text": "Lead."})
    assert wd.record_fields(rec) == expected
    assert wd.record_fields(rec, drop=("title", "keywords")) == {**expected, "title": "", "keywords": ""}


def test_model_frame_uses_prepare_text(wd):
    rec = {"title": "T", "description": "D", "categories": ["C1"], "text": "L"}
    df = wd.model_frame([rec])
    assert list(df.columns)[:3] == ["title", "keywords", "abstract"]
    assert df.loc[0, "keywords"] == "D; C1"


def test_classifier_text(wd):
    assert wd.classifier_text({"title": "T", "description": "D", "text": "L"}) == "T. D. L"


# --- ranking metrics ----------------------------------------------------------

def test_rank_metrics_perfect_and_counts(wd):
    hidden = np.array([0.9, 0.8])
    pool = np.linspace(0.0, 0.5, 198)
    m = wd.rank_metrics(hidden, pool)
    assert m["auc"] == 1.0 and m["n_hidden"] == 2 and m["n_pool"] == 198
    assert m["hidden_in_top100"] == 2
    assert m["recall_top1pct"] == 1.0  # top 2 of 200


def test_rank_metrics_worst_case(wd):
    m = wd.rank_metrics(np.array([0.0]), np.array([0.5, 0.6, 0.7]))
    assert m["auc"] == 0.0 and m["hidden_in_top100"] == 1  # top 100 covers all 4


def test_percentiles(wd):
    p = wd.percentiles(np.array([0.5, 1.0]), np.array([0.1, 0.2, 0.6, 0.7]))
    assert list(p) == [0.5, 1.0]


def test_auc_by_mask(wd):
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.2, 0.5, 0.1])
    assert wd.auc(y, s) == 0.75
    assert wd.auc(y, s, mask=np.array([True, False, True, True])) == 1.0


# --- cross-fitting ------------------------------------------------------------

def test_crossfit_halves_are_disjoint_and_seeded(wd):
    hf, nf = wd.crossfit_halves(n_hidden=11, near_idx=list(range(100)), n_neg=30, seed=2026)
    assert hf[0].isdisjoint(hf[1]) and hf[0] | hf[1] == set(range(11))
    assert nf[0].isdisjoint(nf[1]) and len(nf[0]) == len(nf[1]) == 30
    hf2, nf2 = wd.crossfit_halves(n_hidden=11, near_idx=list(range(100)), n_neg=30, seed=2026)
    assert hf == hf2 and nf == nf2


def test_crossfit_scores_never_use_own_training(wd):
    """Every hidden article is scored by the model trained on the other half."""
    hf, nf = wd.crossfit_halves(n_hidden=10, near_idx=list(range(40)), n_neg=10, seed=1)
    for k in (0, 1):
        train_h, test_h = hf[k], hf[1 - k]
        assert train_h.isdisjoint(test_h)
        test_pool = set(range(40)) - nf[k]
        assert test_pool.isdisjoint(nf[k])


def test_step1_classifier_learns(wd):
    X = ["protein sequence alignment software"] * 6 + ["football club city stadium"] * 6
    y = [1] * 6 + [0] * 6
    m = wd.step1_classifier().fit(X, y)
    p = m.predict_proba(["sequence alignment software", "city football stadium"])[:, 1]
    assert p[0] > 0.5 > p[1]
