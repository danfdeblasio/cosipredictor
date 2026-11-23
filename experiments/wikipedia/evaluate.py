"""Step 1 (CompBio or not) and step 2 (quality) on the labelled Wikipedia sample (#25).

Reads sets.json and articles.jsonl from --cache and scores every labelled article with the COSI model (--model)
and with the step-1 text classifier (5-fold cross-validation, seed 2026). Writes results.json:

    stage1          AUC of each method against each control type, and without biographies
    length_filter   AUC when short leads are dropped
    bio             median scores of CompBio biographies vs other CompBio articles
    stage2          Spearman correlation of each measure with the WikiProject quality class
    stage2_by_class medians per quality class

    python experiments/wikipedia/evaluate.py --model model.joblib
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikidata as wd  # noqa: E402

STEP1 = "Text classifier trained on Wikipedia (5-fold CV)"
METHODS = {
    "Dan's tool: top score (title+keywords+text)": "top_all",
    "Top score, title removed": "top_notitle",
    "Top score, text only": "top_abs",
    "Variance (all fields)": "var_all",
    "Entropy (all fields, lower = more peaked)": "neg_ent_all",
    "Learned on COSI scores + length + biography flag (5-fold CV)": "lr_cosi",
    STEP1: "lr_text",
}
CONTROLS = ["Lonnie examples", "Random", "Near field"]
LENGTH_THRESHOLDS = [0, 100, 200, 300, 500]
QUALITY_MEASURES = [("Top score (Dan's tool)", "top_all"), ("Top score, text only", "top_abs"),
                    ("Variance", "var_all"), ("Entropy (lower = more peaked)", "ent_all"),
                    ("Lead length (characters)", "chars"), ("Whole page size (bytes)", "page_bytes")]


def labelled_frame(sets, A):
    """One row per usable labelled article: set, quality class (CompBio only), label, length, biography flag."""
    rows = [dict(query=t, set="CompBio", cls=c, y=1) for c, ts in sets["pos"].items() for t in ts]
    rows += [dict(query=t, set="Near field", cls=f, y=0) for f, ts in sets["near"].items() for t in ts]
    rows += [dict(query=t, set="Random", cls="", y=0) for t in sets["random"]]
    rows += [dict(query=t, set="Lonnie examples", cls="", y=0) for t in sets["easy"]]
    L = pd.DataFrame(rows)
    L = L[L["query"].map(lambda q: wd.usable(A.get(q)))].reset_index(drop=True)
    R = [A[q] for q in L["query"]]
    L["title"] = [r["title"] for r in R]
    L["url"] = [r["url"] for r in R]
    L["chars"] = [len(r["text"]) for r in R]
    L["page_bytes"] = [r["length"] for r in R]
    L["bio"] = [wd.is_bio(r["categories"]) for r in R]
    return L, R


def evaluate(sets, A, model, folds=5, seed=wd.SEED):
    L, R = labelled_frame(sets, A)
    P_all = model.predict_proba(wd.model_frame(R))
    P_abs = model.predict_proba(wd.model_frame(R, drop=("title", "keywords")))
    P_not = model.predict_proba(wd.model_frame(R, drop=("title",)))
    L["top_all"], L["top_notitle"], L["top_abs"] = P_all.max(1), P_not.max(1), P_abs.max(1)
    L["var_all"], L["ent_all"] = P_all.var(1), wd.entropy(P_all)
    L["neg_ent_all"] = -L["ent_all"]
    y = L.y.values
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    X1 = np.hstack([P_all, P_abs, np.log1p(L[["chars"]].values), L[["bio"]].values.astype(float)])
    L["lr_cosi"] = cross_val_predict(LogisticRegression(max_iter=2000), X1, y, cv=cv, method="predict_proba")[:, 1]
    txt = [wd.classifier_text(r) for r in R]
    L["lr_text"] = cross_val_predict(wd.step1_classifier(), txt, y, cv=cv, method="predict_proba")[:, 1]

    res = {"model_classes": len(model.classes_),
           "n": {k: int(v) for k, v in L.set.value_counts().items()},
           "bio_counts": {k: int(v) for k, v in L[L.bio].set.value_counts().items()}}
    stage1 = []
    for name, col in METHODS.items():
        row = {"method": name, "all": wd.auc(y, L[col])}
        for ns in CONTROLS:
            row[ns] = wd.auc(y, L[col], ((L.set == "CompBio") | (L.set == ns)).values)
        row["no biographies"] = wd.auc(y, L[col], (~L.bio).values)
        stage1.append(row)
    res["stage1"] = stage1
    res["length_filter"] = []
    for th in LENGTH_THRESHOLDS:
        m = (L.chars >= th).values
        res["length_filter"].append({"min_chars": th, "kept_compbio": int((m & (y == 1)).sum()),
                                     "kept_other": int((m & (y == 0)).sum()), "auc_dan": wd.auc(y, L.top_all, m),
                                     "auc_text_only": wd.auc(y, L.top_abs, m), "auc_text_clf": wd.auc(y, L.lr_text, m)})
    bp = L[L.set == "CompBio"]

    def med(s):
        return round(float(s.median()), 3) if len(s) else None
    res["bio"] = {"compbio_bio_n": int(bp.bio.sum()),
                  "median_top_bio": med(bp[bp.bio].top_all), "median_top_nonbio": med(bp[~bp.bio].top_all),
                  "median_text_bio": med(bp[bp.bio].lr_text), "median_text_nonbio": med(bp[~bp.bio].lr_text)}
    q = bp.cls.map(wd.QUALITY_ORDER)
    res["stage2"] = []
    for name, col in QUALITY_MEASURES:
        if bp[col].nunique() < 2:  # correlation undefined for a constant measure
            res["stage2"].append({"measure": name, "spearman": None, "p": None})
            continue
        rho, p = spearmanr(bp[col], q)
        res["stage2"].append({"measure": name, "spearman": round(float(rho), 3), "p": float(p)})
    by = bp.groupby("cls").agg(n=("y", "size"), top=("top_all", "median"), top_abs=("top_abs", "median"),
                               chars=("chars", "median"), bytes=("page_bytes", "median"))
    res["stage2_by_class"] = by.reindex(list(wd.QUALITY_ORDER)).round(3).reset_index().to_dict("records")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=wd.DEFAULT_CACHE)
    ap.add_argument("--model", default="model.joblib")
    args = ap.parse_args(argv)
    sets = json.load(open(os.path.join(args.cache, "sets.json")))
    A = wd.read_jsonl(os.path.join(args.cache, "articles.jsonl"))
    res = evaluate(sets, A, joblib.load(args.model))
    res["model"] = os.path.basename(args.model)
    out = os.path.join(args.cache, "results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    for row in res["stage1"]:
        print(f"{row['all']:.3f}  near {row['Near field']:.3f}  {row['method']}")
    print("quality (Spearman):", {r["measure"]: r["spearman"] for r in res["stage2"]})
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
