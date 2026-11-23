"""Shared helpers for the Wikipedia CompBio experiments (#25).

Batched Wikipedia API access, the WikiProject collection listing, the biography flag, the model input
fields and the ranking metrics used by collect.py, fetch.py, evaluate.py, hideseek.py and report.py.

The model fields come from wiki.to_fields and common.prepare_text, as in predict_wikipedia.py, so the
COSI scores match the website. wiki.py itself is not modified.
"""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import wiki  # noqa: E402
from common import prepare_text  # noqa: E402

SEED = 2026
DEFAULT_CACHE = os.path.join("cache", "wikipedia")
BATCH = 20  # prop=extracts returns at most 20 intro extracts per request

COLLECTION_CLASSES = ["GA", "B", "C", "Start", "Stub", "List", "Redirect"]
QUALITY_ORDER = {"Stub": 1, "Start": 2, "C": 3, "B": 4, "GA": 5}
NEAR_FIELDS = ["Molecular biology", "Cell biology", "Genetics", "Biochemistry", "Ecology", "Microbiology",
               "Neuroscience", "Pharmacology", "Immunology", "Oncology", "Statistics", "Machine learning",
               "Computer science", "Chemistry", "Physics", "Mathematics"]
SEED_CATEGORIES = ["Bioinformatics", "Computational biology"]
LONNIE_EXAMPLES = ["Mexico City", "Nice", "Edinburgh", "Athens, Ohio", "Elon Musk", "Airport", "Airplane",
                   "Mother", "Father", "Food"]
BIOINFO_SOURCE = "Bioinformatics/Comp. biology categories"

BIO = re.compile(r"(\d{3,4}s? births|\d{3,4}s? deaths|Living people|People from|alumni|faculty)", re.I)


class WikipediaAPIError(Exception):
    pass


# --- API ----------------------------------------------------------------------

def api(params, opener=urllib.request.urlopen, sleep=time.sleep, retries=10, timeout=60):
    """One API call with maxlag; retries HTTP 429/503 (honouring Retry-After) and network errors."""
    p = dict(params, format="json", formatversion="2", maxlag="5")
    req = urllib.request.Request(f"{wiki.API}?{urllib.parse.urlencode(p)}", headers={"User-Agent": wiki.USER_AGENT})
    for i in range(retries):
        try:
            with opener(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                raise
            wait = int((e.headers or {}).get("Retry-After") or 0) or 10 * (i + 1)
        except (urllib.error.URLError, TimeoutError):
            wait = 5 * (i + 1)
        sleep(wait)
    raise WikipediaAPIError(f"Wikipedia API still failing after {retries} attempts")


def members(category, api=api, **kw):
    """All members of a category (list=categorymembers), following continuation."""
    out, cont = [], {}
    while True:
        r = api({"action": "query", "list": "categorymembers", "cmtitle": category, "cmlimit": "500", **kw, **cont})
        out += r.get("query", {}).get("categorymembers", [])
        if "continue" not in r:
            return out
        cont = {"cmcontinue": r["continue"]["cmcontinue"]}


def talk_to_article(title):
    return title[len("Talk:"):] if title.startswith("Talk:") else title


def collection_category(cls):
    return f"Category:{cls}-Class Computational Biology articles"


def collection(api=api):
    """{article title: quality class} for WikiProject Computational Biology (talk-page categories)."""
    out = {}
    for cls in COLLECTION_CLASSES:
        for m in members(collection_category(cls), api=api):
            out.setdefault(talk_to_article(m["title"]), cls)
    return out


def fetch_batch(titles, api=api):
    """Fetch up to 20 articles: same fields as wiki.fetch(scope='lead'), plus page length and a status."""
    if len(titles) > BATCH:
        raise ValueError(f"at most {BATCH} titles per batch")
    pages, cont, norm, redir = {}, {}, {}, {}
    while True:
        r = api({"action": "query", "prop": "extracts|categories|info|pageprops|description",
                 "exintro": "1", "explaintext": "1", "exlimit": str(BATCH), "clshow": "!hidden", "cllimit": "max",
                 "inprop": "url", "ppprop": "disambiguation", "redirects": "1", "titles": "|".join(titles), **cont})
        q = r.get("query", {})
        norm.update({x["from"]: x["to"] for x in q.get("normalized", [])})
        redir.update({x["from"]: x["to"] for x in q.get("redirects", [])})
        for p in q.get("pages", []):
            d = pages.setdefault(p["title"], {"cats": []})
            for k in ("extract", "fullurl", "description", "length", "missing", "invalid", "pageprops"):
                if k in p:
                    d[k] = p[k]
            d["cats"] += [c["title"].split(":", 1)[1] for c in p.get("categories", [])]
        if "continue" not in r:
            break
        cont = dict(r["continue"])
    recs = []
    for t in titles:
        tt = norm.get(t, t)
        tt = redir.get(tt, tt)
        p = pages.get(tt, {})
        if not p or "missing" in p or "invalid" in p:
            status = "missing"
        elif "disambiguation" in p.get("pageprops", {}):
            status = "disambiguation"
        else:
            status = "ok"
        recs.append({"query": t, "title": tt, "url": p.get("fullurl", ""), "description": p.get("description", ""),
                     "categories": p.get("cats", []), "text": (p.get("extract") or "").strip(),
                     "length": p.get("length", 0), "status": status})
    return recs


# --- records ------------------------------------------------------------------

def read_jsonl(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["query"]] = r
    return out


def usable(r):
    return r is not None and r["status"] == "ok" and len(r["text"]) > 0


def is_bio(categories):
    return any(BIO.search(c) for c in categories)


def record_fields(rec, drop=()):
    f = wiki.to_fields({"title": rec["title"], "description": rec["description"],
                        "categories": rec["categories"], "text": rec["text"]})
    for k in drop:
        f[k] = ""
    return f


def model_frame(recs, drop=()):
    return prepare_text(pd.DataFrame([record_fields(r, drop) for r in recs]))


def classifier_text(rec):
    return f"{rec['title']}. {rec['description']}. {rec['text']}"


def step1_classifier():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), stop_words="english"),
        LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"))


def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)


# --- metrics ------------------------------------------------------------------

def auc(y, s, mask=None):
    y, s = np.asarray(y), np.asarray(s)
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask)
    if len(set(y[m])) < 2:
        return None  # undefined when the subset has only one class
    return round(float(roc_auc_score(y[m], s[m])), 3)


def rank_metrics(score_hidden, score_pool):
    """AUC, hidden articles in the top K, and recall at the top 1/5/10% of the combined ranking."""
    s = np.concatenate([score_hidden, score_pool])
    y = np.r_[np.ones(len(score_hidden)), np.zeros(len(score_pool))]
    yr = y[np.argsort(-s, kind="stable")]
    out = {"auc": round(float(roc_auc_score(y, s)), 3), "n_hidden": len(score_hidden), "n_pool": len(score_pool)}
    for k in (100, 500, 1000):
        out[f"hidden_in_top{k}"] = int(yr[:k].sum())
    for pct in (1, 5, 10):
        k = int(len(s) * pct / 100)
        out[f"recall_top{pct}pct"] = round(float(yr[:k].sum() / len(score_hidden)), 3)
    return out


def percentiles(score_hidden, score_pool):
    """Share of pool articles scored below each hidden article."""
    score_pool = np.asarray(score_pool)
    return np.array([(score_pool < v).mean() for v in score_hidden])


def crossfit_halves(n_hidden, near_idx, n_neg=3000, seed=SEED):
    """Two halves of the hidden articles and two disjoint sets of n_neg nearby-field negatives."""
    rng = random.Random(seed)
    hi = list(range(n_hidden))
    rng.shuffle(hi)
    near = list(near_idx)
    rng.shuffle(near)
    return [set(hi[0::2]), set(hi[1::2])], [set(near[:n_neg]), set(near[n_neg:2 * n_neg])]
