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


def test_numeric_conference_with_blank_reads_as_integer_text(fixture_dir, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.csv").write_text("title,trackname,conference\nOne,MLCSB,2023\nTwo,MLCSB,\n")
    raw = explore.load_raw([str(data)])
    assert sorted(raw["conference"]) == ["(unknown)", "2023"]


def test_pipes_and_line_breaks_do_not_break_tables():
    table = explore.md_table(pd.DataFrame({"conference": ["A | B", "C\nD"], "rows": [1, 2]}))
    lines = table.splitlines()
    assert len(lines) == 4
    assert lines[2] == "| A \\| B | 1 |"
    assert lines[3] == "| C D | 2 |"


def test_zero_byte_csv_exits_with_message(fixture_dir, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "empty.csv").write_text("")
    with pytest.raises(SystemExit) as exc:
        explore.file_table([str(data)])
    assert "empty file" in str(exc.value.code)
    assert "empty.csv" in str(exc.value.code)


# --- #12: COSIs by conference -----------------------------------------------------------------------
# Training rows per COSI x conference on the fixtures, worked out by hand (fixture COSI list order):
#   3DSIG 1/1/0, MLCSB 1/2/1, BioVis 1/1/0, TextMining 1/0/0, iRNA 0/1/0, CAMDA 0/0/0
#   (columns ISMB2025 / ISMB2026 / (unknown)); totals 4/5/1 = 10.

def _training_rows(fixture_dir, cosis):
    raw = explore.load_raw([fixture_dir])
    kept, _ = explore.filter_raw(raw, cosis)
    return kept  # the filtered training rows, with `label` and `conference`


def run_args(fixture_dir, tmp_path, *args):
    out, details = tmp_path / "eda.md", tmp_path / "cache" / "eda_details.csv"
    cosis = os.path.join(fixture_dir, "cosis_fixture.txt")
    explore.main([*args, "--cosis", cosis, "--out", str(out), "--details", str(details)])
    return out.read_text()


def test_cosi_conference_table_matches_hand_counts(fixture_dir, cosis):
    table = explore.cosi_conference_table(_training_rows(fixture_dir, cosis), cosis)
    assert list(table.index) == ["3DSIG", "MLCSB", "BioVis", "TextMining", "iRNA", "CAMDA", "Total"]
    assert list(table.columns) == ["ISMB2025", "ISMB2026", "(unknown)", "Total"]
    assert table.loc["MLCSB"].tolist() == [1, 2, 1, 4]
    assert table.loc["TextMining"].tolist() == [1, 0, 0, 1]
    assert table.loc["CAMDA"].tolist() == [0, 0, 0, 0]
    assert table.loc["Total"].tolist() == [4, 5, 1, 10]


def test_year_order_puts_years_first_then_unknown():
    labels = ["ismb2020", "ISMBECCB2021", "(unknown)", "ISMB2026", "ISMB/ECCB 2019", "ISMB2020", "Workshop"]
    assert explore._year_order(labels) == ["ISMB/ECCB 2019", "ISMB2020", "ismb2020", "ISMBECCB2021", "ISMB2026",
                                           "Workshop", "(unknown)"]


def test_latest_ismb_handles_label_variants():
    assert explore.latest_ismb(["ISMB/ECCB 2019", "ISMBECCB2025", "ismb2020", "GLBIO2026"]) == ["ISMBECCB2025"]
    assert explore.latest_ismb(["ISMB2026", "GLBIO2027", "(unknown)"]) == ["ISMB2026"]
    assert explore.latest_ismb(["GLBIO2025", "(unknown)"]) == []


def test_stale_lists_are_disjoint(fixture_dir, cosis):
    table = explore.cosi_conference_table(_training_rows(fixture_dir, cosis), cosis)
    never, not_latest = explore.stale_cosis(table, ["ISMB2026"])
    assert never == ["CAMDA"]
    assert not_latest == ["TextMining"]
    assert not set(never) & set(not_latest)
    never, not_latest = explore.stale_cosis(table, ["ISMB2025"])
    assert never == ["CAMDA"]
    assert not_latest == ["iRNA"]
    assert not set(never) & set(not_latest)


def test_report_section_default_latest(fixture_dir, tmp_path):
    report = run_args(fixture_dir, tmp_path, fixture_dir)
    assert "## COSIs by conference" in report
    assert "| MLCSB | 1 | 2 | 1 | 4 |" in report
    assert "Most recent ISMB conference: ISMB2026 (default)" in report
    section = report.split("## COSIs by conference", 1)[1]
    never = section.split("### No training rows at all", 1)[1].split("###", 1)[0]
    latest = section.split("### No training rows in ISMB2026", 1)[1].split("##", 1)[0]
    assert "CAMDA" in never and "TextMining" not in never
    assert "TextMining" in latest and "CAMDA" not in latest


def test_report_section_with_latest_override(fixture_dir, tmp_path):
    report = run_args(fixture_dir, tmp_path, fixture_dir, "--latest", "ISMB2025")
    assert "Most recent ISMB conference: ISMB2025 (--latest)" in report
    latest = report.split("### No training rows in ISMB2025", 1)[1].split("##", 1)[0]
    assert "iRNA" in latest and "TextMining" not in latest


def test_unknown_latest_exits_with_message(fixture_dir, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_args(fixture_dir, tmp_path, fixture_dir, "--latest", "ISMB2099")
    assert "ISMB2099" in str(exc.value.code)


def test_no_ismb_label_is_noted(fixture_dir, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.csv").write_text("title,trackname,conference\nSome talk,MLCSB,GLBIO2025\n")
    report = run_args(fixture_dir, tmp_path, str(data))
    assert "No ISMB conference found" in report
    assert "### No training rows at all" in report
    assert "### No training rows in" not in report  # the latest-conference list is omitted


def test_latest_rejects_total_and_unknown(fixture_dir, tmp_path):
    for bad in ("Total", "(unknown)"):
        with pytest.raises(SystemExit) as exc:
            run_args(fixture_dir, tmp_path, fixture_dir, "--latest", bad)
        assert "not a conference" in str(exc.value.code)


def test_year_needs_a_standalone_four_digit_number():
    assert explore._label_year("ISMB 99999") is None
    assert explore._label_year("ISMB2026") == 2026
    assert explore.latest_ismb(["ISMB2026", "ISMB 99999"]) == ["ISMB2026"]


def test_duplicate_cosi_names_count_once(fixture_dir, cosis):
    table = explore.cosi_conference_table(_training_rows(fixture_dir, cosis), cosis + ["MLCSB"])
    assert list(table.index).count("MLCSB") == 1
    assert table.loc["Total", "Total"] == 10


def test_latest_ismb_returns_every_label_of_the_top_year(fixture_dir, cosis):
    assert explore.latest_ismb(["ISMB2025", "ISMB2026", "ismb2026"]) == ["ISMB2026", "ismb2026"]
    table = explore.cosi_conference_table(_training_rows(fixture_dir, cosis), cosis)
    # TextMining is stale whichever of the two top-year labels is present.
    assert explore.stale_cosis(table, ["ISMB2026", "ismb2026"])[1] == ["TextMining"]


def test_empty_training_set(fixture_dir, cosis):
    empty = _training_rows(fixture_dir, cosis).iloc[0:0]
    table = explore.cosi_conference_table(empty, cosis)
    assert table.loc["Total", "Total"] == 0
    never, not_latest = explore.stale_cosis(table, [])
    assert never == list(cosis) and not_latest == []
    lines = explore.cosi_section(empty, cosis)
    assert "No ISMB conference found; pass --latest to choose one." in lines


def test_total_is_a_reserved_name(fixture_dir, cosis):
    rows = _training_rows(fixture_dir, cosis)
    with pytest.raises(SystemExit) as exc:
        explore.cosi_conference_table(rows, cosis + ["Total"])
    assert "reserved" in str(exc.value.code)
    renamed = rows.assign(conference=rows["conference"].replace({"ISMB2025": "Total"}))
    with pytest.raises(SystemExit):
        explore.cosi_conference_table(renamed, cosis)


def test_latest_message_when_no_row_reaches_training(fixture_dir, cosis):
    empty = _training_rows(fixture_dir, cosis).iloc[0:0]
    with pytest.raises(SystemExit) as exc:
        explore.cosi_section(empty, cosis, latest="ISMB2025")
    assert str(exc.value.code).endswith("labels: (none)")
