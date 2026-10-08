"""Tests for find_lab_data without Kaggle, a GPU, or a network."""

from pathlib import Path

from run_kaggle import find_lab_data


def test_finds_nested_lab_data(tmp_path: Path) -> None:
    (tmp_path / "data-day6" / "data_lab21" / "video_1" / "img1").mkdir(parents=True)
    assert find_lab_data([tmp_path]) == tmp_path / "data-day6" / "data_lab21"


def test_ignores_video_without_images(tmp_path: Path) -> None:
    (tmp_path / "video_1").mkdir()
    assert find_lab_data([tmp_path]) is None


def test_missing_root_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "video_1" / "img1").mkdir(parents=True)
    assert find_lab_data([tmp_path / "khong_co", tmp_path]) == tmp_path
