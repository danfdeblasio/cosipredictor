"""Compare TF-IDF vs SPECTER2-embedding classifiers on a held-out conference.

Example:
    .venv/bin/python compare.py "~/Downloads/ISCB accepted submissions" ~/Downloads/ISMB2020_accepted.csv \
        ~/Downloads/ISMBECCB2019_accepted.csv --holdout ISMB2026
"""
import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, top_k_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import read_cosi_list
from train import build_pipeline, load_training_data, restrict


def embeddings(df, adapter, cache_dir):
    path = os.path.join(cache_dir, f"specter2_{adapter}.npz")
    if os.path.exists(path):
        cached = np.load(path, allow_pickle=True)
        if list(cached["titles"]) == df["title"].tolist():
            return cached["X"]
    from embed import Specter2
    print(f"Embedding {len(df)} submissions with SPECTER2 ({adapter}) ...", flush=True)
    X = Specter2(adapter).encode(df, progress=True)
    np.savez(path, X=X, titles=np.array(df["title"].tolist(), dtype=object))
    return X


def score(name, proba, classes, y):
    pred = classes[proba.argmax(axis=1)]
    print(f"{name:<34} top1={np.mean(pred == y):.3f}  "
          f"top3={top_k_accuracy_score(y, proba, k=3, labels=classes):.3f}  "
          f"top5={top_k_accuracy_score(y, proba, k=5, labels=classes):.3f}  "
          f"macroF1={f1_score(y, pred, average='macro'):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--cosis", default="cosis.txt")
    ap.add_argument("--holdout", default="ISMB2026")
    ap.add_argument("--adapters", default="proximity,classification")
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    df = load_training_data(args.data, read_cosi_list(args.cosis)).reset_index(drop=True)
    test = (df["conference"].astype(str) == args.holdout).values
    y_tr, y_te = df.loc[~test, "label"].values, df.loc[test, "label"].values
    allowed = sorted(set(y_te))
    print(f"train={(~test).sum()}  test={test.sum()} ({args.holdout}, {len(allowed)} COSIs)\n")

    # Embed first (GPU memory is tight), then free the model before fitting anything else.
    adapters = args.adapters.split(",")
    embs = {a: embeddings(df, a, args.cache_dir) for a in adapters}

    results = {}
    tfidf = build_pipeline().fit(df[~test], y_tr)
    results["TF-IDF + LR (current)"] = restrict(tfidf.predict_proba(df[test]), tfidf.classes_, allowed)

    for adapter in adapters:
        X = embs[adapter]
        for C in (0.1, 1.0):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
            clf.fit(X[~test], y_tr)
            p = restrict(clf.predict_proba(X[test]), clf.classes_, allowed)
            results[f"SPECTER2-{adapter} + LR (C={C})"] = p
            # Late fusion: average the two models' probabilities.
            fused = (p[0] + results["TF-IDF + LR (current)"][0]) / 2
            results[f"  avg(TF-IDF, {adapter} C={C})"] = (fused, p[1])

    print()
    for name, (proba, classes) in results.items():
        score(name, proba, classes, y_te)


if __name__ == "__main__":
    main()
