"""Predict COSI probabilities for a list of English Wikipedia articles.

Articles can be titles or URLs, given as arguments and/or in a file (one per line, '#' = comment).
Output CSV has one row per article: the resolved title and URL, a status, the top suggestions,
and a probability column for every COSI.

Examples:
    python3 predict_wikipedia.py "BLAST (biotechnology)" "Mass spectrometry"
    python3 predict_wikipedia.py --file articles.txt --scope full --cosis "MLCSB,3DSIG,BOSC" --out wiki.csv
"""
import argparse
import sys
import time

import joblib
import pandas as pd

import wiki
from common import make_normalizer, prepare_text, read_cosi_list
from train import restrict


def read_articles(args):
    articles = list(args.articles)
    if args.file:
        with open(args.file) if args.file != "-" else sys.stdin as f:
            articles += [l.split("#", 1)[0].strip() for l in f]
    return [a for a in articles if a]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("articles", nargs="*", help="article titles or URLs")
    ap.add_argument("--file", help="file with one article per line ('-' for stdin)")
    ap.add_argument("--scope", choices=["lead", "full"], default="lead",
                    help="text used as the abstract: lead section (default) or whole article")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--cosis", help="file or comma-separated list of COSIs allowed (default: all)")
    ap.add_argument("--top", type=int, default=3, help="number of ranked suggestions to output")
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between Wikipedia requests")
    ap.add_argument("--out", default="wikipedia_predictions.csv")
    args = ap.parse_args()

    articles = read_articles(args)
    if not articles:
        ap.error("give article titles/URLs as arguments or with --file")

    model = joblib.load(args.model)
    classes = list(model.classes_)
    allowed = classes
    if args.cosis:
        normalize = make_normalizer(classes)
        requested = read_cosi_list(args.cosis)
        allowed = sorted({normalize(c) for c in requested} - {None})
        unknown = [c for c in requested if normalize(c) is None]
        if unknown:
            print(f"WARNING: model has no training data for {unknown}; they are skipped.", file=sys.stderr)

    rows, fields = [], []
    for i, query in enumerate(articles):
        if i:
            time.sleep(args.delay)
        row = {"query": query, "title": "", "url": "", "status": "ok"}
        try:
            article = wiki.fetch(query, args.scope)
            row.update(title=article["title"], url=article["url"])
            fields.append(wiki.to_fields(article))
        except wiki.WikipediaError as e:
            row["status"] = f"error: {e}"
            fields.append(None)
        rows.append(row)
        print(f"[{i + 1}/{len(articles)}] {query}: {row['status']}", file=sys.stderr)

    ok = [i for i, f in enumerate(fields) if f is not None]
    if ok:
        X = prepare_text(pd.DataFrame([fields[i] for i in ok]))
        proba, labels = restrict(model.predict_proba(X), classes, allowed)
        for j, i in enumerate(ok):
            order = (-proba[j]).argsort()[:args.top]
            rows[i]["predicted_track"] = labels[order[0]]
            for k, c in enumerate(order):
                rows[i][f"track_{k + 1}"] = labels[c]
                rows[i][f"prob_{k + 1}"] = round(float(proba[j, c]), 4)
            for c, label in enumerate(labels):
                rows[i][f"p_{label}"] = round(float(proba[j, c]), 4)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} rows ({len(ok)} predicted) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
