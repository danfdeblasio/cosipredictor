"""Train a COSI track classifier from title, keywords and abstract.

Example:
    python train.py "~/Downloads/ISCB accepted submissions" ~/Downloads/ISMB2020_accepted.csv \
        --cosis cosis.txt --holdout ISMB2026 --out model.joblib
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, top_k_accuracy_score
from sklearn.pipeline import Pipeline

from common import load_csvs, make_normalizer, prepare_text, read_cosi_list


def build_pipeline():
    def tfidf(**kw):
        return TfidfVectorizer(sublinear_tf=True, strip_accents="unicode",
                               stop_words="english", **kw)

    features = ColumnTransformer([
        ("title", tfidf(ngram_range=(1, 2), min_df=2), "title"),
        ("keywords", tfidf(ngram_range=(1, 2), min_df=2), "keywords"),
        ("abstract", tfidf(ngram_range=(1, 2), min_df=2, max_df=0.5), "abstract"),
    ])
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=2000)
    return Pipeline([("features", features), ("clf", clf)])


def load_training_data(paths, cosis):
    df = load_csvs([os.path.expanduser(p) for p in paths])
    df = df[~df["decision"].fillna("").str.contains("reject", case=False)]
    df["label"] = df["trackname"].map(make_normalizer(cosis))
    df = prepare_text(df[df["label"].notna()])
    return df.drop_duplicates(subset=["title", "label"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="CSV files or directories of CSVs")
    ap.add_argument("--cosis", default="cosis.txt", help="file or comma-separated list of COSI names")
    ap.add_argument("--holdout", help="conference value (e.g. ISMB2026) to hold out for evaluation first")
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    cosis = read_cosi_list(args.cosis)
    df = load_training_data(args.data, cosis)
    print(f"{len(df)} labeled submissions across {df['label'].nunique()} COSIs")
    print(df["label"].value_counts().to_string(), "\n")
    missing = sorted(set(cosis) - set(df["label"]))
    if missing:
        print(f"WARNING: no training examples for {missing}; they cannot be predicted.\n")

    if args.holdout:
        test = df["conference"].astype(str) == args.holdout
        if not test.any():
            raise SystemExit(f"No rows with conference == {args.holdout!r}")
        model = build_pipeline().fit(df[~test], df.loc[~test, "label"])
        X_test, y_test = df[test], df.loc[test, "label"]
        # Mimic real use: only COSIs that ran at the held-out conference are allowed.
        allowed = sorted(set(y_test))
        proba, classes = restrict(model.predict_proba(X_test), model.classes_, allowed)
        pred = classes[proba.argmax(axis=1)]
        print(f"=== Held-out evaluation on {args.holdout} ({len(y_test)} submissions, "
              f"restricted to {len(allowed)} COSIs) ===")
        print(f"top-1 accuracy: {(pred == y_test.values).mean():.3f}")
        for k in (3, 5):
            print(f"top-{k} accuracy: {top_k_accuracy_score(y_test, proba, k=k, labels=classes):.3f}")
        print(classification_report(y_test, pred, zero_division=0))

    model = build_pipeline().fit(df, df["label"])
    joblib.dump(model, args.out)
    print(f"Trained on all {len(df)} submissions; saved model to {args.out}")


def restrict(proba, classes, allowed):
    """Keep only allowed classes' probability columns and renormalize."""
    idx = [i for i, c in enumerate(classes) if c in set(allowed)]
    p = proba[:, idx]
    return p / p.sum(axis=1, keepdims=True), np.asarray(classes)[idx]


if __name__ == "__main__":
    main()
