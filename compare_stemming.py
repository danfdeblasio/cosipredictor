"""Compare the classifier with and without Porter stemming on a held-out conference.

Everything except stemming is identical (fields, n-grams, min_df/max_df, sublinear TF-IDF, C=10).
Differences are reported with a paired bootstrap over the same test submissions.

Example:
    python3 compare_stemming.py "~/Downloads/ISCB accepted submissions" ~/Downloads/ISMB2020_accepted.csv \
        ~/Downloads/ISMBECCB2019_accepted.csv --holdout ISMB2026
"""
import argparse

import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score

from common import read_cosi_list
from train import build_pipeline, load_training_data, restrict

METRICS = ["top1", "top3", "top5", "macroF1"]


def metrics(y, proba, classes, idx):
    y, proba = y[idx], proba[idx]
    pred = classes[proba.argmax(axis=1)]
    return np.array([
        np.mean(pred == y),
        top_k_accuracy_score(y, proba, k=3, labels=classes),
        top_k_accuracy_score(y, proba, k=5, labels=classes),
        f1_score(y, pred, average="macro", labels=classes, zero_division=0),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--cosis", default="cosis.txt")
    ap.add_argument("--holdout", default="ISMB2026")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    df = load_training_data(args.data, read_cosi_list(args.cosis)).reset_index(drop=True)
    test = (df["conference"].astype(str) == args.holdout).values
    y = df.loc[test, "label"].values
    allowed = sorted(set(y))
    print(f"train={(~test).sum()}  test={test.sum()} ({args.holdout})\n")

    results = {}
    for name, stem in [("no stemming", False), ("Porter stemming", True)]:
        model = build_pipeline(stem=stem).fit(df[~test], df.loc[~test, "label"])
        n_feat = sum(len(v.vocabulary_) for _, v, _ in model.named_steps["features"].transformers_
                     if hasattr(v, "vocabulary_"))
        proba, classes = restrict(model.predict_proba(df[test]), model.classes_, allowed)
        results[name] = (proba, classes, n_feat)

    rng = np.random.default_rng(0)
    idx_all = np.arange(len(y))
    boots = [rng.integers(0, len(y), len(y)) for _ in range(args.bootstrap)]
    point, samples = {}, {}
    for name, (proba, classes, n_feat) in results.items():
        point[name] = metrics(y, proba, classes, idx_all)
        samples[name] = np.array([metrics(y, proba, classes, b) for b in boots])
        lo, hi = np.percentile(samples[name], [2.5, 97.5], axis=0)
        print(f"{name:16} features={n_feat:,}")
        for i, m in enumerate(METRICS):
            print(f"   {m:8} {point[name][i]:.3f}  [{lo[i]:.3f}, {hi[i]:.3f}]")

    diff = samples["Porter stemming"] - samples["no stemming"]
    d = point["Porter stemming"] - point["no stemming"]
    lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
    print("\nstemmed minus unstemmed (paired bootstrap 95% CI):")
    for i, m in enumerate(METRICS):
        print(f"   {m:8} {d[i]:+.3f}  [{lo[i]:+.3f}, {hi[i]:+.3f}]")

    p_ns, c_ns, _ = results["no stemming"]
    p_st, c_st, _ = results["Porter stemming"]
    right_ns = c_ns[p_ns.argmax(1)] == y
    right_st = c_st[p_st.argmax(1)] == y
    print(f"\ntop-1 disagreements: stemming fixes {np.sum(~right_ns & right_st)}, "
          f"breaks {np.sum(right_ns & ~right_st)} of {len(y)} submissions")


if __name__ == "__main__":
    main()
