"""Tests for the score_video1 helpers without TrackEval, a GPU, or a network."""

from pathlib import Path

import pytest

from score_video1 import format_table, parse_summary, run_name, summary_path

SUMMARY = "HOTA DetA IDSW CLR_FP\n29.46 18.095 25 337\n"


def test_summary_path_layout(tmp_path: Path) -> None:
    path = summary_path(tmp_path, "botsort_conf15_video1")
    assert path == (
        tmp_path / "data" / "trackers" / "mot_challenge" / "lab-train"
        / "botsort_conf15_video1" / "pedestrian_summary.txt"
    )


def test_run_name_scales_conf() -> None:
    assert run_name("botsort", 0.15) == "botsort_conf15_video1"
    assert run_name("bytetrack", 0.5) == "bytetrack_conf50_video1"


def test_parse_summary_reads_values() -> None:
    assert parse_summary(SUMMARY) == {"HOTA": 29.46, "DetA": 18.095, "IDSW": 25.0, "CLR_FP": 337.0}


def test_parse_summary_rejects_mismatched_columns() -> None:
    with pytest.raises(ValueError):
        parse_summary("HOTA DetA\n29.46\n")


def test_format_table_has_one_line_per_run_and_dashes_for_missing() -> None:
    table = format_table([("botsort conf=0.3", parse_summary(SUMMARY))])
    header, row = table.splitlines()
    assert "HOTA" in header and "IDF1" in header
    assert row.startswith("botsort conf=0.3")
    assert "29.46" in row and "337.00" in row and "-" in row
