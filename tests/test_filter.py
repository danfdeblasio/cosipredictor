"""Pin the training-data filters (reject removal, track mapping, deduplication).

`legacy_load_training_data` is a verbatim copy of `train.load_training_data` as of c1b1c34. The tests
check that the current code still produces exactly the same frame, so refactoring the filters cannot
change what the model is trained on (issue #2, FR-011).
"""
import os

import pandas as pd

from common import filter_training_data, load_csvs, make_normalizer, prepare_text
from train import load_training_data

# Expected (title, label) pairs on the fixtures, worked out by hand:
# 15 rows loaded (no_trackname.csv is skipped) -> 2 rejects removed -> 3 unmapped tracks removed
# (General Computational Biology, 2x GLBIO General Track) -> 1 exact duplicate removed -> 9.
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
        ("loaded", 15),
        ("after reject removal", 13),
        ("after unmapped removal", 10),
        ("after duplicate removal", 9),
    ]


def test_missing_decision_column_is_not_rejected(fixture_dir, cosis):
    # The old code raised KeyError when no file had a `decision` column (FR-012).
    df = load_csvs([os.path.join(fixture_dir, "no_decision.csv")])
    out, stages = filter_training_data(df, cosis)
    assert list(out["title"]) == ["Microbiome network inference"]
    assert stages[1] == ("after reject removal", 1)
