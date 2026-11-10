"""Write the data-exploration (EDA) report for the submission CSVs.

The report holds aggregate counts and conference, track and COSI names only: files, records per
conference, the training-data filtering steps, training rows per COSI and conference with the COSIs that
have no (recent) rows, the unmapped track names per conference, and missing or empty fields per
conference. Per-record rows, titles included, go to the details CSV, which lives under the
gitignored cache/ by default.

Examples:
    python explore.py "~/Downloads/ISCB accepted submissions" --cosis cosis.txt
    python explore.py "~/Downloads/ISCB accepted submissions" --latest ISMB2025
"""
import argparse
import os
import re

import pandas as pd

from common import (TEXT_COLS, csv_files, filter_training_data, load_csvs, make_normalizer, read_cosi_list,
                    rejected_mask)

UNKNOWN = "(unknown)"


def file_table(paths):
    """One row per CSV found: file name, row count, and whether load_csvs reads or skips it."""
    rows = []
    for f in csv_files(paths):
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            raise SystemExit(f"explore.py: empty file (no header): {f}")
        status = "read" if "trackname" in df.columns else "skipped: no trackname"
        rows.append({"file": os.path.basename(f), "rows": len(df), "status": status})
    return pd.DataFrame(rows, columns=["file", "rows", "status"])


def load_raw(paths, files=None):
    """The loaded rows as read, before any filter or blank filling, apart from `conference`, which is
    filled with (unknown) and made a string. `files` is file_table(paths), the one place that decides
    which files are read; it is computed when not given. Exits with a clear message when no CSV has a
    `trackname` column."""
    if files is None:
        files = file_table(paths)
    if not (files["status"] == "read").any():
        raise SystemExit("explore.py: no CSV with a 'trackname' column in: " + ", ".join(paths))
    raw = load_csvs(paths)
    raw["conference"] = raw["conference"].map(_conference_label) if "conference" in raw else UNKNOWN
    return raw


def _conference_label(value):
    """Conference as text: blank -> (unknown); a whole number read as float (2023.0) -> "2023"."""
    if pd.isna(value) or not str(value).strip():
        return UNKNOWN
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def filter_raw(raw, cosis):
    """Run the shared training filter. The kept frame keeps raw's index, so raw.loc[kept.index] gives the
    training rows as they were before prepare_text."""
    return filter_training_data(raw, cosis)


def _conference_order(names):
    return sorted((n for n in names if n != UNKNOWN)) + ([UNKNOWN] if UNKNOWN in names else [])


def conference_counts(raw):
    counts = raw["conference"].value_counts()
    return counts.reindex(_conference_order(list(counts.index)))


def _label_year(label):
    """The trailing four-digit year of a conference label (e.g. "ISMB/ECCB 2019" -> 2019), or None."""
    m = re.search(r"(?<!\d)(\d{4})\s*$", str(label))  # a standalone 4-digit number only
    return int(m.group(1)) if m else None


def _year_order(labels):
    """Conference labels ordered by trailing year, then label; labels without a year and (unknown) last."""
    labels = list(dict.fromkeys(labels))
    dated = sorted((l for l in labels if l != UNKNOWN and _label_year(l) is not None),
                   key=lambda l: (_label_year(l), l))
    undated = sorted(l for l in labels if l != UNKNOWN and _label_year(l) is None)
    return dated + undated + ([UNKNOWN] if UNKNOWN in labels else [])


def latest_ismb(labels):
    """Labels of the most recent ISMB conference: start with "ISMB" (any case) and have the highest year."""
    ismb = [l for l in set(labels) if str(l).upper().startswith("ISMB") and _label_year(l) is not None]
    if not ismb:
        return []
    top = max(_label_year(l) for l in ismb)
    return sorted(l for l in ismb if _label_year(l) == top)


def cosi_conference_table(train_rows, cosis):
    """Training rows per COSI (rows, in cosis.txt order, zeros included) x conference (columns, by year),
    with a Total row and column. `train_rows` is the filtered frame, which has `label`."""
    clash = [n for n in (*cosis, *train_rows["conference"].unique()) if n == "Total"]
    if clash:
        raise SystemExit('explore.py: "Total" is reserved for the totals row and column; '
                         "rename the COSI or conference label that uses it")
    counts = pd.crosstab(train_rows["label"], train_rows["conference"])
    table = counts.reindex(index=list(dict.fromkeys(cosis)), columns=_year_order(counts.columns), fill_value=0)
    table.index.name = table.columns.name = None
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    return table.astype(int)


def stale_cosis(table, latest):
    """(COSIs with no training rows at all, COSIs with none in the `latest` labels but some earlier)."""
    rows = table.drop(index="Total")
    never = [c for c in rows.index if rows.loc[c, "Total"] == 0]
    in_latest = rows[[l for l in latest if l in rows.columns]].sum(axis=1)
    not_latest = [c for c in rows.index if rows.loc[c, "Total"] > 0 and in_latest[c] == 0]
    return never, not_latest


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


def _cell(value):
    """Table cell text: a '|' or line break in the data must not break the Markdown table."""
    return " ".join(str(value).splitlines()).replace("|", "\\|")


def md_table(df, index_name=None):
    """Render a DataFrame as a Markdown table (no tabulate dependency)."""
    if index_name is not None:
        df = df.reset_index().rename(columns={"index": index_name})
    cells = [[_cell(v) for v in row] for row in df.itertuples(index=False)]
    header = "| " + " | ".join(map(_cell, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([header, sep] + ["| " + " | ".join(r) + " |" for r in cells])


def _pct(rates):
    out = rates.copy()
    for c in TEXT_COLS:
        out[c] = out[c].map(lambda x: "-" if pd.isna(x) else f"{x * 100:.1f}%")  # "-" = no rows
    return out


def cosi_section(train_rows, cosis, latest=None):
    """Markdown lines for the "COSIs by conference" section. `latest` is a label given with --latest."""
    table = cosi_conference_table(train_rows, cosis)
    labels = [c for c in table.columns if c not in ("Total", UNKNOWN)]  # real conference labels only
    if latest is not None:
        if latest not in labels:
            raise SystemExit(f"explore.py: --latest {latest} is not a conference with training rows; "
                             "labels: " + (", ".join(labels) or "(none)"))
        latest_labels, source = [latest], "--latest"
    else:
        latest_labels, source = latest_ismb(labels), "default"
    never, not_latest = stale_cosis(table, latest_labels)
    lines = [
        "## COSIs by conference",
        "",
        "Training rows per COSI in `cosis.txt` and conference label (labels as they appear in the data).",
        "",
        md_table(table, index_name="COSI"),
        "",
    ]
    if latest_labels:
        name = ", ".join(latest_labels)
        lines += [f"Most recent ISMB conference: {name} ({source})", ""]
    else:
        lines += ["No ISMB conference found; pass --latest to choose one.", ""]
    lines += ["### No training rows at all", "", ", ".join(never) or "None", ""]
    if latest_labels:
        lines += [f"### No training rows in {name}, but some earlier", "", ", ".join(not_latest) or "None", ""]
    return lines


BLANK = "(blank)"

# Placeholder titles (D11, decided by @eri2905 on #14): exact title keys, "paper<N>", an empty key, or
# TBA/TBD/TBC as a separate word in the original title. Removing them from training is #8.
PLACEHOLDER_KEYS = {"tba", "tbd", "tbc", "qa", "closingremarks", "openingremarks", "welcome"}
_PAPER_N = re.compile(r"paper\d+")
_TBX_WORD = re.compile(r"\b(tba|tbd|tbc)\b", re.IGNORECASE)


def title_key(title):
    """A title's key: lowercase with every non-alphanumeric character removed (same as norm_title in
    experiments/eval_set.py). Missing titles give ""."""
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def placeholder_rule(title):
    """The placeholder rule a title matches ("list", "paperN", "empty" or "word"), or "" for a real title."""
    key = title_key(title)
    if key == "":
        return "empty"
    if key in PLACEHOLDER_KEYS:
        return "list"
    if _PAPER_N.fullmatch(key):
        return "paperN"
    if _TBX_WORD.search(str(title)):
        return "word"
    return ""


def unmapped_tracks(raw, cosis):
    """Rows per conference and raw track name for tracks that map to no COSI, after reject removal.
    Same reject rule and COSI mapping as filter_training_data, so the total equals its unmapped step."""
    kept = raw[~rejected_mask(raw)]
    unmapped = kept[kept["trackname"].map(make_normalizer(cosis)).isna()]
    track = unmapped["trackname"].map(lambda t: BLANK if pd.isna(t) or not str(t).strip() else str(t))
    counts = (pd.DataFrame({"conference": unmapped["conference"], "track": track})
              .value_counts().rename("rows").reset_index())
    order = {c: i for i, c in enumerate(_year_order(counts["conference"]))}
    counts["_conf"] = counts["conference"].map(order)
    counts = counts.sort_values(["_conf", "rows", "track"], ascending=[True, False, True], kind="stable")
    return counts.drop(columns="_conf").reset_index(drop=True)[["conference", "track", "rows"]]


def unmapped_section(raw, cosis, stages):
    """Markdown lines for the "Unmapped tracks" section. `stages` are filter_training_data's step counts,
    so the total line shows the filter's own number next to this section's."""
    table = unmapped_tracks(raw, cosis)
    lines = ["## Unmapped tracks", "",
             "Tracks that map to no COSI, after reject removal. Track names are shown as they appear in the "
             "data, so names that differ only in spacing or case are listed separately.", ""]
    if table.empty:
        return lines + ["None", ""]
    total = int(table["rows"].sum())
    step = int(funnel_table(stages).set_index("step").loc["after unmapped removal", "removed"])
    return lines + [md_table(table), "",
                    f'Total: {total:,} rows (the "after unmapped removal" step removes {step:,}).', ""]


def _dedupe_rows(raw, cosis):
    """The rows that reach the dedupe step (not rejected, mapped to a COSI), with `label`, `key` and `year`."""
    rows = raw[~rejected_mask(raw)].copy()
    rows["label"] = rows["trackname"].map(make_normalizer(cosis))
    rows = rows[rows["label"].notna()].copy()
    title = rows["title"] if "title" in rows else pd.Series(None, index=rows.index)
    rows["key"] = title.map(title_key)
    rows["year"] = rows["conference"].map(_label_year)
    # Multi-COSI groups are formed within a conference year; a label with no year is its own group, so two
    # different undated conferences are never pooled together.
    rows["year_group"] = [str(y) if y is not None else f"label:{c}" for y, c in zip(rows["year"], rows["conference"])]
    return rows


DUP_CLASSES = ["exact title, same COSI", "same key, same COSI", "same key, different COSIs (same year)"]


def _duplicate_masks(rows):
    """Row masks for the same-key same-COSI groups (all years) and the multi-COSI groups (same year).
    Rows with an empty key are left out of both (D2)."""
    keyed = rows[rows["key"] != ""]
    same = keyed.groupby(["key", "label"])["key"].transform("size") > 1
    multi = keyed.groupby(["year_group", "key"])["label"].transform("nunique") > 1
    return (same.reindex(rows.index, fill_value=False), multi.reindex(rows.index, fill_value=False))


def duplicate_counts(raw, cosis):
    """(table of groups / rows / extra rows per duplicate class, number of multi-COSI groups whose title is a
    placeholder). The exact class uses the raw title, as train.py's dedupe does (blank titles count as equal)."""
    rows = _dedupe_rows(raw, cosis)
    title = rows["title"] if "title" in rows else pd.Series(None, index=rows.index)
    exact = rows.assign(_t=title).groupby(["_t", "label"], dropna=False).size()
    exact = exact[exact > 1]
    same_mask, multi_mask = _duplicate_masks(rows)
    same = rows[same_mask].groupby(["key", "label"]).size()
    multi_rows = rows[multi_mask].assign(_ph=title[multi_mask].map(placeholder_rule) != "")
    multi = multi_rows.groupby(["year_group", "key"]).size()
    table = pd.DataFrame(
        [[len(exact), int(exact.sum()), int((exact - 1).sum())],
         [len(same), int(same.sum()), int((same - 1).sum())],
         [len(multi), int(multi.sum()), "-"]],
        index=DUP_CLASSES, columns=["groups", "rows", "extra rows"])
    # A group counts as a placeholder group when any of its rows' original titles is a placeholder (the
    # "word" rule needs the original title, not the key).
    multi_placeholders = int(multi_rows.groupby(["year_group", "key"])["_ph"].any().sum())
    return table, multi_placeholders


def duplicate_section(raw, cosis):
    table, multi_placeholders = duplicate_counts(raw, cosis)
    return ["## Duplicate titles", "",
            "Rows that reach the dedupe step. The key is the title in lowercase without punctuation or spaces. "
            "Same-COSI classes span all years, as train.py's dedupe does; the multi-COSI class is counted "
            "within a conference year. Titles with an empty key are counted as placeholders instead.", "",
            md_table(table, index_name="class"), "",
            f"Multi-COSI groups with a placeholder title: {multi_placeholders}.", ""]


def placeholder_counts(raw, kept, cosis):
    """Placeholder rows after reject removal: by rule, by conference and by COSI (mapped rows only), each with
    how many reach training (the rows in `kept`)."""
    rows = raw[~rejected_mask(raw)].copy()
    title = rows["title"] if "title" in rows else pd.Series(None, index=rows.index)
    rows["rule"] = title.map(placeholder_rule)
    rows = rows[rows["rule"] != ""].copy()
    rows["in_training"] = rows.index.isin(kept.index)
    rows["label"] = rows["trackname"].map(make_normalizer(cosis))

    def by(col, order):
        g = rows.groupby(col).agg(rows=("rule", "size"), in_training=("in_training", "sum"))
        g = g.reindex([o for o in order if o in g.index]).astype(int)
        return g.rename_axis(col).reset_index()

    rule_order = ["list", "paperN", "word", "empty"]
    by_rule = rows["rule"].value_counts().reindex([r for r in rule_order if r in set(rows["rule"])])
    return {
        "by_rule": by_rule.rename_axis("rule").reset_index(name="rows"),
        "by_conference": by("conference", _year_order(rows["conference"])),
        "by_cosi": by("label", list(dict.fromkeys(cosis))),
        "rows": len(rows),
        "in_training": int(rows["in_training"].sum()),
    }


def placeholder_section(raw, kept, cosis):
    ph = placeholder_counts(raw, kept, cosis)
    lines = ["## Placeholder titles", "",
             "Titles that are agenda placeholders (D11 on #14): an exact key (tba, tbd, tbc, qa, closingremarks, "
             "openingremarks, welcome), paper + number (e.g. paper12), an empty key, or TBA/TBD/TBC as a separate word. "
             "Rows after "
             "reject removal; titles are not shown. Detection only: removing them from training is #8.", ""]
    if ph["rows"] == 0:
        return lines + ["None", ""]
    by_cosi = ph["by_cosi"].rename(columns={"label": "COSI"})
    return lines + [
        f"Placeholder rows: {ph['rows']:,} after reject removal; {ph['in_training']:,} reach training.", "",
        md_table(ph["by_rule"]), "", md_table(ph["by_conference"]), "", md_table(by_cosi), ""]


def build_report(paths, cosis, latest=None):
    files = file_table(paths)
    raw = load_raw(paths, files)
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
        *cosi_section(kept, cosis, latest),
        *unmapped_section(raw, cosis, stages),
        *duplicate_section(raw, cosis),
        *placeholder_section(raw, kept, cosis),
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
    """One row per loaded record. Contains titles, so it must stay out of git (cache/ is ignored).
    `row` counts within `source_file`, which is the file's basename, so input files are assumed to have
    distinct names."""
    flags = missing_flags(raw)
    label = raw["trackname"].map(make_normalizer(cosis))  # mapped COSI, also for rows later dropped
    title = raw["title"] if "title" in raw else pd.Series("", index=raw.index)
    dedupe_rows = _dedupe_rows(raw, cosis)
    same, multi = _duplicate_masks(dedupe_rows)
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
        "title_key": title.map(title_key),
        "placeholder": title.map(placeholder_rule),
        "dup_same_cosi": same.reindex(raw.index, fill_value=False),
        "dup_multi_cosi": multi.reindex(raw.index, fill_value=False),
        "title": title.fillna(""),
    })


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+", help="CSV files or directories of CSVs")
    ap.add_argument("--cosis", default="cosis.txt", help="file or comma-separated list of COSI names")
    ap.add_argument("--out", default=os.path.join("reports", "eda.md"), help="Markdown report (aggregates only)")
    ap.add_argument("--details", default=os.path.join("cache", "eda_details.csv"),
                    help="per-record CSV; contains titles, keep it out of git")
    ap.add_argument("--latest", help="conference label to treat as the most recent ISMB (default: detected)")
    args = ap.parse_args(argv)

    paths = [os.path.expanduser(p) for p in args.data]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("explore.py: path not found: " + ", ".join(missing))
    report, details = build_report(paths, read_cosi_list(args.cosis), args.latest)
    for path in (args.out, args.details):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    details.to_csv(args.details, index=False)
    print(f"Wrote {args.out} and {args.details}")


if __name__ == "__main__":
    main()
