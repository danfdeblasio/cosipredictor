"""Build a frozen, independent evaluation split for the COSI classifier.

Split by conference year so evaluation always looks forward in time:
    train = ISMB 2019-2024, val = ISMB 2025 (only for tuning thresholds), test = ISMB 2026.

Each document is one submission with a *set* of COSIs: rows from the same conference with the same
normalized title are merged and their labels unioned (a few talks were accepted under 2+ COSIs).

The test set is made independent of earlier years by removing near-duplicates: test documents whose
title + abstract has TF-IDF cosine similarity >= --dup-threshold to any train or val document
(typically the same work resubmitted). METHODS.md notes this was not checked before.

Example:
    python experiments/eval_set.py "../data/ISCB accepted submissions"

Writes eval_set_report.md and eval_set_manifest.csv to --out (default experiments/results). The manifest
contains submission titles and is git-ignored.
"""
import argparse
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load_csvs, make_normalizer, prepare_text, read_cosi_list  # noqa: E402

YEAR_SPLITS = {"train": range(2019, 2025), "val": [2025], "test": [2026]}


def conference_year(conf):
    m = re.search(r"(20\d\d)", str(conf))
    return int(m.group(1)) if m else None


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def load_multilabel(data_paths, cosis_path="cosis.txt"):
    """One row per (year, normalized title) with `labels` = sorted list of COSIs."""
    df = load_csvs([os.path.expanduser(p) for p in data_paths])
    df = df[~df["decision"].fillna("").str.contains("reject", case=False)]
    df["label"] = df["trackname"].map(make_normalizer(read_cosi_list(cosis_path)))
    df = prepare_text(df[df["label"].notna()])
    df["year"] = df["conference"].map(conference_year)
    df["tkey"] = df["title"].map(norm_title)
    df = df[df["tkey"] != ""]
    grouped = df.groupby(["year", "tkey"], sort=False)
    docs = grouped.agg(title=("title", "first"), keywords=("keywords", "first"),
                       abstract=("abstract", "first"), conference=("conference", "first"),
                       labels=("label", lambda s: sorted(set(s)))).reset_index()
    docs["n_labels"] = docs["labels"].map(len)
    docs["doc_id"] = [hashlib.sha1(f"{y}|{k}".encode()).hexdigest()[:12]
                      for y, k in zip(docs["year"], docs["tkey"])]
    return docs


def assign_split(year):
    for name, years in YEAR_SPLITS.items():
        if year in years:
            return name
    return None


def near_duplicates(test_text, ref_text, threshold):
    """Max cosine similarity of each test doc to any reference doc, plus a boolean mask."""
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=1)
    vec.fit(pd.concat([ref_text, test_text]))
    R, T = vec.transform(ref_text), vec.transform(test_text)
    best = np.zeros(T.shape[0])
    for start in range(0, T.shape[0], 256):
        sims = (T[start:start + 256] @ R.T).toarray()
        best[start:start + 256] = sims.max(axis=1)
    return best, best >= threshold


def build(data_paths, cosis_path="cosis.txt", dup_threshold=0.8):
    docs = load_multilabel(data_paths, cosis_path)
    docs["split"] = docs["year"].map(assign_split)
    docs = docs[docs["split"].notna()].reset_index(drop=True)
    text = docs["title"] + " " + docs["abstract"]
    is_test = docs["split"] == "test"
    sim, dup = near_duplicates(text[is_test], text[~is_test], dup_threshold)
    docs["max_sim_to_earlier"] = np.nan
    docs.loc[is_test, "max_sim_to_earlier"] = sim
    docs.loc[docs.index[is_test][dup], "split"] = "test_excluded_dup"
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--cosis", default="cosis.txt")
    ap.add_argument("--dup-threshold", type=float, default=0.8)
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    docs = build(args.data, args.cosis, args.dup_threshold)
    os.makedirs(args.out, exist_ok=True)
    cols = ["doc_id", "split", "year", "conference", "title", "labels", "n_labels", "max_sim_to_earlier"]
    out = docs[cols].copy()
    out["labels"] = out["labels"].map(";".join)
    out.to_csv(os.path.join(args.out, "eval_set_manifest.csv"), index=False)

    lines = ["# Independent evaluation set", "",
             f"Near-duplicate threshold: cosine >= {args.dup_threshold} (title + abstract TF-IDF) "
             "against all train/val documents.", "",
             "| Split | Documents | Multi-COSI documents |", "|---|---|---|"]
    for name, g in docs.groupby("split"):
        lines.append(f"| {name} | {len(g)} | {(g['n_labels'] > 1).sum()} |")
    t = docs.loc[docs["split"].str.startswith("test"), "max_sim_to_earlier"]
    lines += ["", "Test documents by max similarity to an earlier-year document:", "",
              "| Similarity | Documents |", "|---|---|"]
    for lo, hi in [(0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.5, 0.7), (0, 0.5)]:
        lines.append(f"| {lo:.1f}–{min(hi, 1):.1f} | {((t >= lo) & (t < hi)).sum()} |")
    test = docs[docs["split"] == "test"]
    lines += ["", "Test COSI distribution (a multi-COSI document counts once per COSI):", "",
              "| COSI | Documents |", "|---|---|"]
    for c, n in test["labels"].explode().value_counts().items():
        lines.append(f"| {c} | {n} |")
    with open(os.path.join(args.out, "eval_set_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
