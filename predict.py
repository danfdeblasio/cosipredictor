"""Predict the COSI track for new submissions.

Input CSV needs title, keywords and abstract columns (others are passed through).

Example (restrict to COSIs running this year):
    python predict.py new_submissions.csv --cosis "MLCSB,3DSIG,HiTSeq,RegSys" --top 3 --out predictions.csv
"""
import argparse

import joblib
import pandas as pd

from common import make_normalizer, prepare_text, read_cosi_list
from train import restrict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="CSV of submissions to classify")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--cosis", help="file or comma-separated list of COSIs allowed this time "
                                    "(default: every COSI the model was trained on)")
    ap.add_argument("--top", type=int, default=3, help="number of ranked suggestions to output")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    model = joblib.load(args.model)
    classes = list(model.classes_)
    allowed = classes
    if args.cosis:
        normalize = make_normalizer(classes)
        requested = read_cosi_list(args.cosis)
        allowed = sorted({normalize(c) for c in requested} - {None})
        unknown = [c for c in requested if normalize(c) is None]
        if unknown:
            print(f"WARNING: model has no training data for {unknown}; they are skipped.")
        if not allowed:
            raise SystemExit("None of the requested COSIs are known to the model.")

    df = pd.read_csv(args.input)
    proba, labels = restrict(model.predict_proba(prepare_text(df)), classes, allowed)

    order = (-proba).argsort(axis=1)[:, :args.top]
    out = df.copy()
    out["predicted_track"] = labels[order[:, 0]]
    for k in range(order.shape[1]):
        out[f"track_{k + 1}"] = labels[order[:, k]]
        out[f"prob_{k + 1}"] = proba[range(len(df)), order[:, k]].round(3)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} predictions to {args.out}")


if __name__ == "__main__":
    main()
