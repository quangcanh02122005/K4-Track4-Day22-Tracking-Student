"""Tests for summary_path without TrackEval, a GPU, or a network."""

from pathlib import Path

from score_video1 import summary_path


def test_summary_path_layout(tmp_path: Path) -> None:
    path = summary_path(tmp_path, "botsort_video1")
    assert path == tmp_path / "data" / "trackers" / "mot_challenge" / "lab-train" / "botsort_video1" / "pedestrian_summary.txt"
