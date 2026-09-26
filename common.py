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


def load_csvs(paths):
    """Load CSVs from files and/or directories into one DataFrame."""
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p) else [p]
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "trackname" not in df.columns:
            continue
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def prepare_text(df):
    """Fill missing text fields; turn newline-separated keywords into '; '-joined phrases."""
    out = df.copy()
    for c in TEXT_COLS:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    out["keywords"] = out["keywords"].str.replace(r"\s*[\r\n]+\s*", "; ", regex=True)
    return out
