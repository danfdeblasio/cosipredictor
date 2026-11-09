"""Write the data-exploration (EDA) report for the submission CSVs.

The report holds aggregate counts only: files, records per conference, the training-data filtering steps,
and missing or empty fields per conference. Per-record rows, titles included, go to the details CSV, which
lives under the gitignored cache/ by default.

Example:
    python explore.py "~/Downloads/ISCB accepted submissions" --cosis cosis.txt
"""
import argparse
import os

import pandas as pd

from common import TEXT_COLS, csv_files, filter_training_data, load_csvs, make_normalizer, read_cosi_list

UNKNOWN = "(unknown)"


def file_table(paths):
    """One row per CSV found: file name, row count, and whether load_csvs reads or skips it."""
    rows = []
    for f in csv_files(paths):
        df = pd.read_csv(f)
        status = "read" if "trackname" in df.columns else "skipped: no trackname"
        rows.append({"file": os.path.basename(f), "rows": len(df), "status": status})
    return pd.DataFrame(rows, columns=["file", "rows", "status"])


def load_raw(paths):
    """The loaded rows exactly as read (before any filter or blank filling), with `conference` filled in.
    Exits with a clear message when no CSV has a `trackname` column."""
    if not any("trackname" in pd.read_csv(f, nrows=0).columns for f in csv_files(paths)):
        raise SystemExit("explore.py: no CSV with a 'trackname' column in: " + ", ".join(paths))
    raw = load_csvs(paths)
    raw["conference"] = raw["conference"].fillna(UNKNOWN) if "conference" in raw else UNKNOWN
    raw["conference"] = raw["conference"].astype(str)
    return raw


def filter_raw(raw, cosis):
    """Run the shared training filter. The kept frame keeps raw's index, so raw.loc[kept.index] gives the
    training rows as they were before prepare_text."""
    return filter_training_data(raw, cosis)


def _conference_order(names):
    return sorted((n for n in names if n != UNKNOWN)) + ([UNKNOWN] if UNKNOWN in names else [])


def conference_counts(raw):
    counts = raw["conference"].value_counts()
    return counts.reindex(_conference_order(list(counts.index)))


def funnel_table(stages):
    remaining = [n for _, n in stages]
    removed = [0] + [a - b for a, b in zip(remaining, remaining[1:])]
    return pd.DataFrame({"step": [s for s, _ in stages], "remaining": remaining, "removed": removed})


def missing_flags(raw):
    """True where a text field is NaN, empty or whitespace only; a missing column is missing everywhere."""
    flags = pd.DataFrame(index=raw.index)
    for c in TEXT_COLS:
        if c in raw.columns:
            flags[c] = raw[c].isna() | raw[c].astype(str).str.strip().eq("")
        else:
            flags[c] = True
    return flags


def missing_rates(raw):
    """Per conference (plus an "All" row): row count and the share of rows missing each text field."""
    flags = missing_flags(raw)
    flags["conference"] = raw["conference"] if "conference" in raw else UNKNOWN
    grouped = flags.groupby("conference")
    rates = grouped[TEXT_COLS].mean()
    rates.insert(0, "rows", grouped.size())
    rates = rates.reindex(_conference_order(list(rates.index)))
    total = pd.DataFrame([[len(flags), *flags[TEXT_COLS].mean()]], columns=["rows", *TEXT_COLS], index=["All"])
    rates = pd.concat([rates, total])
    rates["rows"] = rates["rows"].astype(int)
    return rates


def md_table(df, index_name=None):
    """Render a DataFrame as a Markdown table (no tabulate dependency)."""
    if index_name is not None:
        df = df.reset_index().rename(columns={"index": index_name})
    cells = [[str(v) for v in row] for row in df.itertuples(index=False)]
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([header, sep] + ["| " + " | ".join(r) + " |" for r in cells])


def _pct(rates):
    out = rates.copy()
    for c in TEXT_COLS:
        out[c] = out[c].map(lambda x: "-" if pd.isna(x) else f"{x * 100:.1f}%")  # "-" = no rows
    return out


def build_report(paths, cosis):
    files = file_table(paths)
    raw = load_raw(paths)
    kept, stages = filter_raw(raw, cosis)
    counts = conference_counts(raw).rename_axis("conference").reset_index(name="rows")
    lines = [
        "# Exploratory data analysis",
        "",
        "Aggregate counts only. Per-record rows are in the details CSV (gitignored).",
        "",
        "## Files",
        "",
        md_table(files),
        "",
        "## Records per conference",
        "",
        f"{len(raw):,} rows loaded from {int((files['status'] == 'read').sum())} files.",
        "",
        md_table(counts),
        "",
        "## Filtering steps",
        "",
        "Same steps and order as `train.load_training_data` (`common.filter_training_data`).",
        "",
        md_table(funnel_table(stages)),
        "",
        "## Missing or empty fields",
        "",
        "Share of rows whose field is empty, blank or absent, measured before blanks are filled.",
        "",
        "### All loaded rows",
        "",
        md_table(_pct(missing_rates(raw)), index_name="conference"),
        "",
        "### Rows that reach training",
        "",
        md_table(_pct(missing_rates(raw.loc[kept.index])), index_name="conference"),
        "",
    ]
    return "\n".join(lines), details_table(raw, kept, cosis)


def details_table(raw, kept, cosis):
    """One row per loaded record. Contains titles, so it must stay out of git (cache/ is ignored)."""
    flags = missing_flags(raw)
    label = raw["trackname"].map(make_normalizer(cosis))  # mapped COSI, also for rows later dropped
    title = raw["title"] if "title" in raw else pd.Series("", index=raw.index)
    return pd.DataFrame({
        "source_file": raw["source_file"],
        "row": raw.groupby("source_file").cumcount() + 1,
        "conference": raw["conference"],
        "trackname": raw["trackname"],
        "label": label.fillna(""),
        "in_training": raw.index.isin(kept.index),
        "title_missing": flags["title"],
        "keywords_missing": flags["keywords"],
        "abstract_missing": flags["abstract"],
        "title": title.fillna(""),
    })


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="CSV files or directories of CSVs")
    ap.add_argument("--cosis", default="cosis.txt", help="file or comma-separated list of COSI names")
    ap.add_argument("--out", default=os.path.join("reports", "eda.md"), help="Markdown report (aggregates only)")
    ap.add_argument("--details", default=os.path.join("cache", "eda_details.csv"),
                    help="per-record CSV; contains titles, keep it out of git")
    args = ap.parse_args(argv)

    paths = [os.path.expanduser(p) for p in args.data]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("explore.py: path not found: " + ", ".join(missing))
    report, details = build_report(paths, read_cosi_list(args.cosis))
    for path in (args.out, args.details):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    details.to_csv(args.details, index=False)
    print(f"Wrote {args.out} and {args.details}")


if __name__ == "__main__":
    main()
