"""Hide-and-seek test: hide unused collection articles in the candidate pool and see how high each method ranks them (#25).

Reads sets.json, cats.json, hidden.json, articles.jsonl and hidden.jsonl from --cache. The hidden articles are the
WikiProject Computational Biology articles (class B, C, Start, Stub) not used for training; the pool is every
candidate that is not in the collection. A method is good when the hidden articles come out near the top.

Methods: the step-1 classifier trained on the labelled sample, Dan's COSI scores (top score, text-only top score,
variance, entropy), and the step-1 classifier with more data. The more-data classifier uses 2-fold cross-fitting:
each half of the hidden set plus 3,000 seeded nearby-field candidates (as negatives) trains a model that scores
the other half and the rest of the pool, so no article is scored by a model that trained on it.

Writes hideseek.json (metrics) and pool_scores.csv (one row per pool article; gitignored like the rest of the
cache).

    python experiments/wikipedia/hideseek.py --model model.joblib
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikidata as wd  # noqa: E402

PROTO = "Step-1 classifier, labelled sample only"
MORE = "Step-1 classifier, more data (2-fold cross-fitting)"
METHODS = [PROTO, "Dan's tool: top COSI score", "Top COSI score, text only", "Variance of COSI scores",
           "Entropy of COSI scores (lower = more peaked)"]
HIDDEN_CLASSES = ["Stub", "Start", "C", "B"]
POOL_COLUMNS = ["title", "url", "source", "description", "chars", "bio", "score", "proto", "cosi", "cosi_score",
                "used_as_negative"]


def labelled(sets, A):
    pos = [(A[t], 1) for v in sets["pos"].values() for t in v if wd.usable(A.get(t))]
    neg = [t for v in sets["near"].values() for t in v] + sets["random"] + sets["easy"]
    return pos + [(A[t], 0) for t in neg if wd.usable(A.get(t))]


def hidden_and_pool(sets, A, H, hidden_titles, coll):
    hid = [H[t] for t in hidden_titles if wd.usable(H.get(t))]
    hidden_set = {r["title"] for r in hid}
    seen, pool = set(), []
    for t in sets["discovery"]:
        r = A.get(t)
        if wd.usable(r) and r["title"] not in seen and r["title"] not in coll and r["title"] not in hidden_set:
            seen.add(r["title"])
            pool.append(r)
    return hid, pool


def hideseek(sets, A, H, hidden_titles, coll, model, n_neg=3000, seed=wd.SEED):
    lab = labelled(sets, A)
    hid, pool = hidden_and_pool(sets, A, H, hidden_titles, coll)
    src = [sets["source"].get(r["query"], "") for r in pool]
    X_lab, y_lab = [wd.classifier_text(r) for r, _ in lab], [y for _, y in lab]
    X_hid, X_pool = [wd.classifier_text(r) for r in hid], [wd.classifier_text(r) for r in pool]

    proto = wd.step1_classifier().fit(X_lab, y_lab)
    sh, sp = proto.predict_proba(X_hid)[:, 1], proto.predict_proba(X_pool)[:, 1]
    Ph, Pp = model.predict_proba(wd.model_frame(hid)), model.predict_proba(wd.model_frame(pool))
    drop = ("title", "keywords")
    Pha, Ppa = model.predict_proba(wd.model_frame(hid, drop)), model.predict_proba(wd.model_frame(pool, drop))
    pairs = {PROTO: (sh, sp), METHODS[1]: (Ph.max(1), Pp.max(1)), METHODS[2]: (Pha.max(1), Ppa.max(1)),
             METHODS[3]: (Ph.var(1), Pp.var(1)), METHODS[4]: (-wd.entropy(Ph), -wd.entropy(Pp))}
    res = {"methods": {k: wd.rank_metrics(a, b) for k, (a, b) in pairs.items()}}

    near_idx = [i for i, s in enumerate(src) if s != wd.BIOINFO_SOURCE]
    hf, nf = wd.crossfit_halves(len(hid), near_idx, n_neg=n_neg, seed=seed)
    more_h = np.zeros(len(hid))
    pool_sum, pool_n = np.zeros(len(pool)), np.zeros(len(pool))
    folds = []
    for k in (0, 1):
        train_h, test_h = sorted(hf[k]), sorted(hf[1 - k])
        X = X_lab + [X_hid[i] for i in train_h] + [X_pool[i] for i in sorted(nf[k])]
        y = y_lab + [1] * len(train_h) + [0] * len(nf[k])
        m = wd.step1_classifier().fit(X, y)
        test_p = [i for i in range(len(pool)) if i not in nf[k]]
        a = m.predict_proba([X_hid[i] for i in test_h])[:, 1]
        b = m.predict_proba([X_pool[i] for i in test_p])[:, 1]
        more_h[test_h] = a
        pool_sum[test_p] += b
        pool_n[test_p] += 1
        folds.append(wd.rank_metrics(a, b))
    res["methods"][MORE] = {k: round(float(np.mean([f[k] for f in folds])), 3) for k in folds[0]}
    res["more_data_folds"] = folds
    more_p = pool_sum / np.maximum(pool_n, 1)

    cls = [coll.get(r["title"], coll.get(r["query"], "")) for r in hid]
    bio = np.array([wd.is_bio(r["categories"]) for r in hid])
    by = pd.DataFrame({"cls": cls, "bio": bio, "proto": wd.percentiles(sh, sp),
                       "more": wd.percentiles(more_h, more_p), "dan": wd.percentiles(Ph.max(1), Pp.max(1))})
    present = [c for c in HIDDEN_CLASSES if c in set(cls)]
    g = by.groupby("cls")
    res["by_class"] = [{"cls": c, "n": int(g.size()[c]), **{k: round(float(g[k].median()[c]), 3)
                                                             for k in ("proto", "more", "dan")}} for c in present]

    def med(s):
        return round(float(s.median()), 3) if len(s) else None
    res["bio"] = {"n": int(bio.sum()), **{k: med(by[by.bio][k]) for k in ("proto", "more", "dan")},
                  **{k + "_nonbio": med(by[~by.bio][k]) for k in ("proto", "more", "dan")}}
    res["n_labelled"], res["n_hidden"], res["n_pool"] = len(lab), len(hid), len(pool)

    cosi_classes = [str(c) for c in model.classes_]
    scores = pd.DataFrame({
        "title": [r["title"] for r in pool], "url": [r["url"] for r in pool], "source": src,
        "description": [r["description"] for r in pool], "chars": [len(r["text"]) for r in pool],
        "bio": [wd.is_bio(r["categories"]) for r in pool], "score": more_p, "proto": sp,
        "cosi": [cosi_classes[i] for i in Pp.argmax(1)], "cosi_score": Pp.max(1),
        "used_as_negative": [i in nf[0] or i in nf[1] for i in range(len(pool))]})
    return res, scores[POOL_COLUMNS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=wd.DEFAULT_CACHE)
    ap.add_argument("--model", default="model.joblib")
    args = ap.parse_args(argv)

    def path(name):
        return os.path.join(args.cache, name)
    sets = json.load(open(path("sets.json")))
    coll = json.load(open(path("cats.json")))["collection"]
    hidden = json.load(open(path("hidden.json")))
    A, H = wd.read_jsonl(path("articles.jsonl")), wd.read_jsonl(path("hidden.jsonl"))
    res, scores = hideseek(sets, A, H, hidden, coll, joblib.load(args.model))
    res["model"] = os.path.basename(args.model)
    with open(path("hideseek.json"), "w") as f:
        json.dump(res, f, indent=1)
    scores.to_csv(path("pool_scores.csv"), index=False)
    print(f"labelled {res['n_labelled']}, hidden {res['n_hidden']}, pool {res['n_pool']}")
    for name, m in res["methods"].items():
        print(f"AUC {m['auc']:.3f}  top100 {m['hidden_in_top100']:>5}  recall@10% {m['recall_top10pct']:.3f}  {name}")
    print(f"Wrote {path('hideseek.json')} and {path('pool_scores.csv')}")


if __name__ == "__main__":
    main()
