"""Pin the training-data filters (reject removal, track mapping, deduplication).

`legacy_load_training_data` is a verbatim copy of `train.load_training_data` as of c1b1c34. The tests
check that the current code still produces exactly the same frame, so refactoring the filters cannot
change what the model is trained on (issue #2, FR-011).
"""
import os

import numpy as np
import pandas as pd

from common import filter_training_data, load_csvs, make_normalizer, prepare_text
from train import load_training_data

# Expected (title, label) pairs on the fixtures, worked out by hand:
# 17 rows loaded (no_trackname.csv is skipped) -> 2 rejects removed -> 3 unmapped tracks removed
# (General Computational Biology, 2x GLBIO General Track) -> 2 duplicates removed -> 10.
# The duplicates are one exact title pair and two BioVis rows with no title: blank titles count as
# the same title "", so those two collapse into one row (as train.py has always done).
EXPECTED = {
    ("Protein folding with graphs", "3DSIG"),
    ("Deep learning for gene regulation", "MLCSB"),
    ("Visualizing single-cell atlases", "BioVis"),
    ("Mining biomedical literature", "TextMining"),
    ("Protein language models", "3DSIG"),
    ("Protein language models", "MLCSB"),
    ("Small RNA regulation", "iRNA"),
    ("Deep Learning for Gene Regulation!", "MLCSB"),
    ("Microbiome network inference", "MLCSB"),
    ("", "BioVis"),
}


def legacy_load_training_data(paths, cosis):
    df = load_csvs([os.path.expanduser(p) for p in paths])
    df = df[~df["decision"].fillna("").str.contains("reject", case=False)]
    df["label"] = df["trackname"].map(make_normalizer(cosis))
    df = prepare_text(df[df["label"].notna()])
    return df.drop_duplicates(subset=["title", "label"])


def test_expected_rows(fixture_dir, cosis):
    df = load_training_data([fixture_dir], cosis)
    assert set(zip(df["title"], df["label"])) == EXPECTED
    assert len(df) == len(EXPECTED)


def test_matches_legacy_frame(fixture_dir, cosis):
    new = load_training_data([fixture_dir], cosis)
    old = legacy_load_training_data([fixture_dir], cosis)
    pd.testing.assert_frame_equal(new, old)


def test_stage_counts(fixture_dir, cosis):
    _, stages = filter_training_data(load_csvs([fixture_dir]), cosis)
    assert stages == [
        ("loaded", 17),
        ("after reject removal", 15),
        ("after unmapped removal", 12),
        ("after duplicate removal", 10),
    ]


def test_missing_decision_column_is_not_rejected(fixture_dir, cosis):
    # The old code raised KeyError when no file had a `decision` column (FR-012).
    df = load_csvs([os.path.join(fixture_dir, "no_decision.csv")])
    out, stages = filter_training_data(df, cosis)
    assert list(out["title"]) == ["Microbiome network inference"]
    assert stages[1] == ("after reject removal", 1)


def _repro_counts():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "repro_counts.py")
    spec = importlib.util.spec_from_file_location("repro_counts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repro_counts_matches_shared_filter(fixture_dir, cosis):
    # repro_counts.py reports the same numbers as the shared filter, under its own row labels.
    steps = _repro_counts().filter_counts([fixture_dir], cosis)
    _, stages = filter_training_data(load_csvs([fixture_dir]), cosis)
    assert [n for _, n in steps] == [n for _, n in stages]
    assert [name for name, _ in steps] == [
        "All CSV rows",
        "Rejected submissions",
        "Tracks that are not COSIs",
        "Duplicates (same title and COSI)",
    ]


def test_repro_counts_missing_decision_column(fixture_dir, cosis):
    steps = _repro_counts().filter_counts([os.path.join(fixture_dir, "no_decision.csv")], cosis)
    assert [n for _, n in steps] == [1, 1, 1, 1]


def test_blank_titles_with_same_cosi_collapse(fixture_dir, cosis):
    df = load_training_data([fixture_dir], cosis)
    blank = df[df["title"] == ""]
    assert list(blank["label"]) == ["BioVis"]


def test_non_string_decision_values_are_not_rejected(cosis):
    # The shared filter reads `decision` as text, so numbers or blanks never raise and never count as rejected.
    df = pd.DataFrame({
        "title": ["Numeric decision row", "Blank decision row"],
        "trackname": ["MLCSB", "3DSIG"],
        "decision": [1.0, np.nan],
    })
    out, stages = filter_training_data(df, cosis)
    assert len(out) == 2
    assert stages[1] == ("after reject removal", 2)
