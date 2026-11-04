"""Compare single-label vs multi-label COSI classification on the independent evaluation split.

Systems (same TF-IDF features as train.build_pipeline):
    SL top-1        current model (softmax logistic regression), best COSI only
    SL top-3        current model, three best COSIs
    SL threshold    current model, every COSI with probability >= t (at least one)
    OvR top-1       one-vs-rest logistic regression trained on label sets, best COSI only
    OvR threshold   one-vs-rest, every COSI with score >= t (at least one)

Thresholds are tuned on val (train 2019-2024 -> predict 2025) to maximize sample-averaged F1, then
models are refit on 2019-2025 and evaluated once on the 2026 test set (near-duplicates removed).

Example:
    python experiments/compare_multilabel.py "../data/ISCB accepted submissions"

Writes multilabel_report.md, set_metrics.csv and threshold_curve.csv to --out (default experiments/results).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from eval_set import build  # noqa: E402
from train import build_pipeline  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def fit_single(docs):
    rows = docs.explode("labels")
    return build_pipeline().fit(rows, rows["labels"])


def fit_ovr(docs, mlb):
    pipe = build_pipeline()
    pipe.set_params(clf=OneVsRestClassifier(
        LogisticRegression(C=10, class_weight="balanced", max_iter=2000), n_jobs=-1))
    return pipe.fit(docs, mlb.transform(docs["labels"]))


def scores(model, docs, classes):
    """Probability/score matrix with columns ordered as `classes`."""
    p = model.predict_proba(docs)
    model_classes = list(model.classes_) if not isinstance(model.named_steps["clf"], OneVsRestClassifier) \
        else list(classes)
    return p[:, [model_classes.index(c) for c in classes]]


def topk_sets(P, k):
    S = np.zeros_like(P, dtype=bool)
    idx = np.argsort(-P, axis=1)[:, :k]
    np.put_along_axis(S, idx, True, axis=1)
    return S


def threshold_sets(P, t):
    S = P >= t
    S[np.arange(len(P)), P.argmax(axis=1)] = True  # always suggest at least one COSI
    return S


def set_metrics(S, Y):
    inter = (S & Y).sum(axis=1)
    size = S.sum(axis=1)
    prec, rec = inter / size, inter / Y.sum(axis=1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return {
        "hit_rate": float(np.mean(inter > 0)),             # a correct COSI is among the suggestions
        "avg_set_size": float(size.mean()),
        "sample_precision": float(prec.mean()),
        "sample_recall": float(rec.mean()),
        "sample_f1": float(f1.mean()),
        "exact_match": float(np.mean((S == Y).all(axis=1))),
        "macro_f1": float(f1_score(Y, S, average="macro", zero_division=0)),
    }


def tune_threshold(P, Y):
    best = max(THRESHOLDS, key=lambda t: set_metrics(threshold_sets(P, t), Y)["sample_f1"])
    return float(best)


def topk_acc(P, Y, k):
    return float(np.mean((topk_sets(P, k) & Y).any(axis=1)))


def paired_bootstrap(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.mean(a) - np.mean(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--cosis", default="cosis.txt")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    t0 = time.time()

    docs = build(args.data, args.cosis)
    train, val, test = (docs[docs["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    classes = sorted(set(docs["labels"].explode()))
    mlb = MultiLabelBinarizer(classes=classes).fit([classes])
    print(f"train={len(train)} val={len(val)} test={len(test)} COSIs={len(classes)}", flush=True)

    # 1) Tune thresholds on val.
    Yv = mlb.transform(val["labels"]).astype(bool)
    Pv_sl = scores(fit_single(train), val, classes)
    print(f"[{time.time() - t0:.0f}s] single-label (train->val) done", flush=True)
    Pv_ovr = scores(fit_ovr(train, mlb), val, classes)
    print(f"[{time.time() - t0:.0f}s] one-vs-rest (train->val) done", flush=True)
    t_sl, t_ovr = tune_threshold(Pv_sl, Yv), tune_threshold(Pv_ovr, Yv)

    # 2) Refit on train+val, evaluate once on test.
    trval = pd.concat([train, val], ignore_index=True)
    Yt = mlb.transform(test["labels"]).astype(bool)
    Pt_sl = scores(fit_single(trval), test, classes)
    print(f"[{time.time() - t0:.0f}s] single-label (train+val->test) done", flush=True)
    Pt_ovr = scores(fit_ovr(trval, mlb), test, classes)
    print(f"[{time.time() - t0:.0f}s] one-vs-rest (train+val->test) done", flush=True)

    systems = {
        "SL top-1 (current)": topk_sets(Pt_sl, 1),
        "SL top-3": topk_sets(Pt_sl, 3),
        f"SL threshold (p >= {t_sl:.2f})": threshold_sets(Pt_sl, t_sl),
        "OvR top-1": topk_sets(Pt_ovr, 1),
        f"OvR threshold (s >= {t_ovr:.2f})": threshold_sets(Pt_ovr, t_ovr),
    }
    res = pd.DataFrame({name: set_metrics(S, Yt) for name, S in systems.items()}).T
    res.to_csv(os.path.join(args.out, "set_metrics.csv"))

    rank = pd.DataFrame({
        "SL (softmax)": {f"top-{k}": topk_acc(Pt_sl, Yt, k) for k in (1, 3, 5)},
        "OvR": {f"top-{k}": topk_acc(Pt_ovr, Yt, k) for k in (1, 3, 5)},
    }).T
    hit_sl = (topk_sets(Pt_sl, 1) & Yt).any(axis=1).astype(float)
    hit_ovr = (topk_sets(Pt_ovr, 1) & Yt).any(axis=1).astype(float)
    diff = paired_bootstrap(hit_ovr, hit_sl)

    # Trade-off: hit rate vs average number of suggested COSIs, over all thresholds.
    curve = []
    for name, P in (("SL", Pt_sl), ("OvR", Pt_ovr)):
        for t in THRESHOLDS:
            m = set_metrics(threshold_sets(P, t), Yt)
            curve.append({"system": name, "threshold": t, **m})
    curve = pd.DataFrame(curve)
    curve.to_csv(os.path.join(args.out, "threshold_curve.csv"), index=False)

    def hit_at_size(name, target):
        c = curve[curve["system"] == name]
        row = c.iloc[(c["avg_set_size"] - target).abs().argsort().iloc[0]]
        return f"{row['hit_rate']:.3f} (t={row['threshold']:.2f}, size={row['avg_set_size']:.2f})"

    fmt = lambda df: df.to_markdown(floatfmt=".3f")  # noqa: E731
    report = [
        "# Single-label vs multi-label COSI classification", "",
        f"Train 2019–2024 → tune thresholds on 2025 (n={len(val)}) → refit on 2019–2025 → "
        f"test on 2026 with near-duplicates removed (n={len(test)}).",
        f"Multi-COSI documents: train {int((train['n_labels'] > 1).sum())}, val {int((val['n_labels'] > 1).sum())}, "
        f"test {int((test['n_labels'] > 1).sum())}.", "",
        "## Ranking quality", "", fmt(rank), "",
        f"OvR top-1 minus SL top-1: {diff[0]:+.3f} [95% paired bootstrap CI {diff[1]:+.3f}, {diff[2]:+.3f}]", "",
        "## Suggested COSI sets", "",
        "`hit_rate` = a correct COSI is among the suggestions; `avg_set_size` = COSIs suggested per submission.", "",
        fmt(res), "",
        "## Hit rate at a matched number of suggestions (test)", "",
        "| Avg. COSIs suggested | SL threshold | OvR threshold |", "|---|---|---|",
    ]
    for target in (1.5, 2.0, 3.0):
        report.append(f"| {target} | {hit_at_size('SL', target)} | {hit_at_size('OvR', target)} |")
    report += ["", f"Runtime: {(time.time() - t0) / 60:.1f} min."]
    with open(os.path.join(args.out, "multilabel_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
