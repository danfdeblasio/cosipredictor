"""Top stems per COSI by mean TF-IDF, for the web app's "COSI vocabulary" page.

Text (title, keywords and abstract combined) goes through the same preprocessing as the model
(lowercasing, accent stripping, the same token pattern and English stop words). Each token is then
reduced to its Porter stem. Stems are weighted with TF-IDF (sublinear tf, smoothed idf, L2 norm),
and each stem's score for a COSI is its mean weight over that COSI's submissions. Each stem is
displayed as its most frequent word form.

Example:
    python3 top_terms.py "~/Downloads/ISCB accepted submissions" ~/Downloads/ISMB2020_accepted.csv \
        ~/Downloads/ISMBECCB2019_accepted.csv --out web/data/top_terms.json
"""
import argparse
import collections
import datetime
import json
import os

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from common import read_cosi_list
from train import load_training_data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="CSV files or directories of CSVs")
    ap.add_argument("--cosis", default="cosis.txt")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-df", type=int, default=5, help="minimum number of submissions containing a stem")
    ap.add_argument("--out", default="web/data/top_terms.json")
    args = ap.parse_args()

    df = load_training_data([os.path.expanduser(p) for p in args.data], read_cosi_list(args.cosis))
    text = (df["title"] + ". " + df["keywords"] + ". " + df["abstract"]).tolist()

    base = TfidfVectorizer(strip_accents="unicode", stop_words="english")
    tokenize = base.build_analyzer()  # lowercase, strip accents, token pattern, stop words
    stemmer = PorterStemmer()
    cache, forms = {}, collections.defaultdict(collections.Counter)

    def analyze(doc):
        stems = []
        for tok in tokenize(doc):
            s = cache.get(tok) or cache.setdefault(tok, stemmer.stem(tok))
            forms[s][tok] += 1
            stems.append(s)
        return stems

    vec = TfidfVectorizer(analyzer=analyze, sublinear_tf=True, min_df=args.min_df, max_df=0.5)
    X = vec.fit_transform(text)
    stems = vec.get_feature_names_out()
    labels = df["label"].values

    out = {}
    for cosi in sorted(set(labels), key=str.lower):
        rows = labels == cosi
        mean = np.asarray(X[rows].mean(axis=0)).ravel()
        top = mean.argsort()[::-1][:args.top]
        out[cosi] = {
            "submissions": int(rows.sum()),
            "terms": [{
                "stem": stems[i],
                "word": forms[stems[i]].most_common(1)[0][0],
                "forms": [w for w, _ in forms[stems[i]].most_common(5)],
                "score": round(float(mean[i]), 4),
            } for i in top],
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"generated": datetime.date.today().isoformat(), "submissions": len(df),
                   "method": "mean TF-IDF of Porter stems over each COSI's submissions",
                   "cosis": out}, f, indent=1, ensure_ascii=False)
    print(f"Wrote top {args.top} stems for {len(out)} COSIs to {args.out}")


if __name__ == "__main__":
    main()
