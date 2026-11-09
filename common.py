"""Shared helpers: loading submission CSVs and normalizing track names."""
import glob
import os
import re

import pandas as pd

TEXT_COLS = ["title", "keywords", "abstract"]

# Historical names -> current canonical name. Keys are compared after _key()
# (lowercased, spaces/punctuation stripped, trailing "COSI" removed).
ALIASES = {
    "rna": "iRNA",                  # "RNA COSI" (2019) became iRNA
    "functioncafa4": "Function",    # "Function/CAFA 4" (2020)
    "bioontologies": "BOKR",        # Bio-Ontologies merged into BOKR (2025+)
}


def _key(name):
    s = str(name).strip()
    s = re.sub(r"\s*COSI\s*$", "", s, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_cosi_list(path_or_names):
    """Read COSI names from a file (one per line, # comments) or a comma-separated string."""
    if os.path.isfile(path_or_names):
        with open(path_or_names) as f:
            lines = [l.split("#", 1)[0].strip() for l in f]
    else:
        lines = [x.strip() for x in path_or_names.split(",")]
    return [l for l in lines if l]


def make_normalizer(cosis):
    """Return a function mapping a raw trackname to a canonical COSI name, or None."""
    canon = {_key(c): c for c in cosis}
    for alias, target in ALIASES.items():
        if _key(target) in canon:
            canon.setdefault(alias, canon[_key(target)])

    def normalize(track):
        return canon.get(_key(track))

    return normalize


def csv_files(paths):
    """List the CSV files that `paths` names: a directory expands to its sorted *.csv files."""
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p) else [p]
    return files


def load_csvs(paths):
    """Load CSVs from files and/or directories into one DataFrame. Files without a `trackname`
    column are skipped."""
    frames = []
    for f in csv_files(paths):
        df = pd.read_csv(f)
        if "trackname" not in df.columns:
            continue
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def filter_training_data(df, cosis):
    """Apply the training filters and return (filtered frame, [(stage, row count), ...]).

    Order matters and matches what train.py has always done: drop rejected submissions, map track
    names to COSIs and drop unmapped rows, fill text fields, then drop rows with the same exact title
    and COSI. A missing `decision` column counts as "not rejected".
    """
    stages = [("loaded", len(df))]
    decision = df["decision"] if "decision" in df.columns else pd.Series("", index=df.index)
    df = df[~decision.fillna("").astype(str).str.contains("reject", case=False)].copy()
    stages.append(("after reject removal", len(df)))
    df["label"] = df["trackname"].map(make_normalizer(cosis))
    df = prepare_text(df[df["label"].notna()])
    stages.append(("after unmapped removal", len(df)))
    df = df.drop_duplicates(subset=["title", "label"])
    stages.append(("after duplicate removal", len(df)))
    return df, stages


def prepare_text(df):
    """Fill missing text fields; turn newline-separated keywords into '; '-joined phrases."""
    out = df.copy()
    for c in TEXT_COLS:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    out["keywords"] = out["keywords"].str.replace(r"\s*[\r\n]+\s*", "; ", regex=True)
    return out
