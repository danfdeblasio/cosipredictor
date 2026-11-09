"""Tests for explore.py on the synthetic fixtures. Expected values are worked out by hand from tests/fixtures/.

Fixture rows per conference (17 loaded, no_trackname.csv skipped):
    GLBIO2025 2 · ISMB2025 6 · ISMB2026 8 · (unknown) 1 (no_decision.csv has no `conference` column)
Missing or empty fields among the loaded rows:
    ISMB2025: keywords 2 of 6 (the two "Mining biomedical literature" rows)
    ISMB2026: title 2 of 8 (two blank BioVis rows), keywords 1 of 8, abstract 2 of 8
    (unknown): keywords 1 of 1 (no `keywords` column)
Among the 10 training rows: ISMB2025 4 (keywords 1), ISMB2026 5 (title 1, abstract 1), (unknown) 1 (keywords 1).
"""
import os

import pandas as pd
import pytest

import explore

UNKNOWN = "(unknown)"


def run(fixture_dir, tmp_path, *paths):
    out, details = tmp_path / "eda.md", tmp_path / "cache" / "eda_details.csv"
    cosis = os.path.join(fixture_dir, "cosis_fixture.txt")
    explore.main([*(paths or [fixture_dir]), "--cosis", cosis, "--out", str(out), "--details", str(details)])
    return out.read_text(), pd.read_csv(details, keep_default_na=False)


def test_file_table_marks_skipped_files(fixture_dir):
    table = explore.file_table([fixture_dir])
    assert list(table["file"]) == ["glbio_fixture.csv", "ismb_fixture.csv", "no_decision.csv", "no_trackname.csv"]
    assert list(table["rows"]) == [2, 14, 1, 1]
    assert list(table["status"]) == ["read", "read", "read", "skipped: no trackname"]


def test_conference_counts(fixture_dir):
    raw = explore.load_raw([fixture_dir])
    counts = explore.conference_counts(raw)
    assert counts.to_dict() == {"GLBIO2025": 2, "ISMB2025": 6, "ISMB2026": 8, UNKNOWN: 1}
    assert list(counts.index)[-1] == UNKNOWN


def test_funnel_matches_shared_filter(fixture_dir, cosis):
    from train import load_training_data
    raw = explore.load_raw([fixture_dir])
    _, stages = explore.filter_raw(raw, cosis)
    funnel = explore.funnel_table(stages)
    assert list(funnel["remaining"]) == [17, 15, 12, 10]
    assert list(funnel["removed"]) == [0, 2, 3, 2]
    assert funnel["remaining"].iloc[-1] == len(load_training_data([fixture_dir], cosis))


def test_missing_rates_all_loaded_rows(fixture_dir):
    rates = explore.missing_rates(explore.load_raw([fixture_dir]))
    assert rates.loc["ISMB2025", "rows"] == 6
    assert rates.loc["ISMB2025", "keywords"] == pytest.approx(2 / 6)
    assert rates.loc["ISMB2026", "title"] == pytest.approx(2 / 8)
    assert rates.loc["ISMB2026", "keywords"] == pytest.approx(1 / 8)
    assert rates.loc["ISMB2026", "abstract"] == pytest.approx(2 / 8)
    assert rates.loc[UNKNOWN, "keywords"] == 1.0  # the column is missing
    assert rates.loc["GLBIO2025", ["title", "keywords", "abstract"]].tolist() == [0, 0, 0]
    assert rates.loc["All", "rows"] == 17
    assert rates.loc["All", "keywords"] == pytest.approx(4 / 17)


def test_missing_rates_training_rows(fixture_dir, cosis):
    raw = explore.load_raw([fixture_dir])
    kept, _ = explore.filter_raw(raw, cosis)
    rates = explore.missing_rates(raw.loc[kept.index])
    assert "GLBIO2025" not in rates.index  # no GLBIO row reaches training
    assert rates.loc["ISMB2025", "rows"] == 4
    assert rates.loc["ISMB2025", "keywords"] == pytest.approx(1 / 4)
    assert rates.loc["ISMB2026", "rows"] == 5
    assert rates.loc["ISMB2026", "title"] == pytest.approx(1 / 5)
    assert rates.loc["ISMB2026", "abstract"] == pytest.approx(1 / 5)
    assert rates.loc[UNKNOWN, "keywords"] == 1.0
    assert rates.loc["All", "rows"] == 10


def test_whitespace_counts_as_missing():
    raw = pd.DataFrame({"conference": ["X", "X"], "title": ["  ", "Real"], "abstract": ["text", None]})
    rates = explore.missing_rates(raw)
    assert rates.loc["X", "title"] == 0.5
    assert rates.loc["X", "abstract"] == 0.5
    assert rates.loc["X", "keywords"] == 1.0


def test_report_sections_and_values(fixture_dir, tmp_path):
    report, _ = run(fixture_dir, tmp_path)
    for heading in ("## Files", "## Records per conference", "## Filtering steps", "## Missing or empty fields"):
        assert heading in report
    assert "| no_trackname.csv | 1 | skipped: no trackname |" in report
    assert "| (unknown) | 1 |" in report
    assert "| after duplicate removal | 10 | 2 |" in report
    assert "33.3%" in report  # ISMB2025 keywords, all loaded rows


def test_details_csv_has_one_row_per_loaded_record(fixture_dir, tmp_path):
    _, details = run(fixture_dir, tmp_path)
    assert len(details) == 17
    assert details["in_training"].astype(str).str.lower().eq("true").sum() == 10
    assert set(details.columns) >= {"source_file", "row", "conference", "trackname", "label", "in_training",
                                    "title_missing", "keywords_missing", "abstract_missing", "title"}
    ismb = details[details["source_file"] == "ismb_fixture.csv"]
    assert list(ismb["row"]) == list(range(1, 15))


def test_report_contains_no_submission_text(fixture_dir, tmp_path):
    # Privacy (FR-006): no title, keyword or abstract from the fixtures may reach the report. Values shorter
    # than 8 characters (e.g. "rna", "misc") are skipped, since they can occur inside names by chance.
    report, _ = run(fixture_dir, tmp_path)
    for f in explore.csv_files([fixture_dir]):
        df = pd.read_csv(f)
        for col in ("title", "keywords", "abstract", "authors"):
            if col not in df.columns:
                continue
            for value in df[col].dropna().astype(str):
                if len(value.strip()) >= 8:
                    assert value not in report, f"{col} from {os.path.basename(f)} leaked into the report"


def test_no_csv_with_trackname_exits_with_message(fixture_dir, tmp_path):
    only_skipped = os.path.join(fixture_dir, "no_trackname.csv")
    with pytest.raises(SystemExit) as exc:
        run(fixture_dir, tmp_path, only_skipped)
    assert "trackname" in str(exc.value.code)
    assert "no_trackname.csv" in str(exc.value.code)


def test_no_training_rows_shows_dash_not_nan(fixture_dir, tmp_path):
    # Every row maps to no COSI, so nothing reaches training: rates read "-" instead of "nan%".
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.csv").write_text("title,trackname,conference\nSome talk,Nowhere Track,X2025\n")
    report, _ = run(fixture_dir, tmp_path, str(data))
    assert "nan" not in report
    assert "| All | 0 | - | - | - |" in report


def test_missing_path_exits_with_message(fixture_dir, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run(fixture_dir, tmp_path, os.path.join(fixture_dir, "does_not_exist.csv"))
    assert "path not found" in str(exc.value.code)
    assert "does_not_exist.csv" in str(exc.value.code)
