"""Count the records left after each training-data filter, to check them against METHODS.md.

Uses common.filter_training_data, the same filter as train.load_training_data:
    1. all CSV rows (files without a `trackname` column are skipped, as in common.load_csvs)
    2. remove rows whose decision contains "reject"
    3. remove rows whose track maps to no COSI
    4. remove duplicates with the same title and COSI

Example:
    python experiments/repro_counts.py "../data/ISCB accepted submissions"

Writes repro_counts.md to --out (default experiments/results). It contains counts only.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import filter_training_data, load_csvs, read_cosi_list  # noqa: E402

METHODS_MD = [14678, 14405, 9336, 9103]  # remaining after each step, as reported in METHODS.md


# Row labels for the report, one per step of common.filter_training_data.
STEP_LABELS = ["All CSV rows", "Rejected submissions", "Tracks that are not COSIs",
               "Duplicates (same title and COSI)"]


def filter_counts(paths, cosis):
    _, stages = filter_training_data(load_csvs([os.path.expanduser(p) for p in paths]), cosis)
    return [(label, n) for label, (_, n) in zip(STEP_LABELS, stages, strict=True)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="CSV files or directories of CSVs")
    ap.add_argument("--cosis", default=os.path.join(HERE, "..", "cosis.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    steps = filter_counts(args.data, read_cosi_list(args.cosis))
    lines = ["# Training-data filter counts", "",
             "Same steps and order as `train.load_training_data`.", "",
             "| Step | Removed | Remaining | METHODS.md | Match |", "|---|---|---|---|---|"]
    prev = None
    for (name, n), expected in zip(steps, METHODS_MD, strict=True):
        removed = "" if prev is None else f"−{prev - n:,}"
        lines.append(f"| {name} | {removed} | {n:,} | {expected:,} | {'yes' if n == expected else 'NO'} |")
        prev = n
    report = "\n".join(lines) + "\n"
    print(report)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "repro_counts.md"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
