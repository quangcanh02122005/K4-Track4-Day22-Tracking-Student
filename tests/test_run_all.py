"""Tests for run_all helpers without a GPU, lab images, or a network."""

import json
from pathlib import Path

import pytest

from run_all import load_config, missing_submission_files

TRACKERS = ["bytetrack", "ocsort", "botsort", "strongsort", "deepocsort"]


def _write_config(path: Path, **overrides: dict) -> Path:
    config = {f"video_{i}": {"tracker": "bytetrack", "conf": 0.3, "iou": 0.5} for i in range(1, 6)}
    config["_ghi_chu"] = "bỏ qua"
    config.update(overrides)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_all_files_present(tmp_path: Path) -> None:
    for i in range(1, 6):
        (tmp_path / f"video_{i}.txt").write_text("1,1,0,0,10,10,1,-1,-1,-1\n")
    assert missing_submission_files(tmp_path) == []


def test_missing_and_empty_files_reported(tmp_path: Path) -> None:
    (tmp_path / "video_1.txt").write_text("1,1,0,0,10,10,1,-1,-1,-1\n")
    (tmp_path / "video_2.txt").write_text("")
    assert missing_submission_files(tmp_path) == ["video_2.txt", "video_3.txt", "video_4.txt", "video_5.txt"]


def test_load_config_valid(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path / "c.json"), valid_trackers=TRACKERS)
    assert sorted(config) == [f"video_{i}" for i in range(1, 6)]
    assert config["video_3"] == {"tracker": "bytetrack", "conf": 0.3, "iou": 0.5}


def test_load_config_missing_video(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "c.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["video_5"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="video_5"):
        load_config(path)


def test_load_config_unknown_tracker(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "c.json", video_2={"tracker": "deepsort", "conf": 0.3, "iou": 0.5})
    with pytest.raises(ValueError, match="deepsort"):
        load_config(path, valid_trackers=TRACKERS)


def test_load_config_threshold_out_of_range(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "c.json", video_4={"tracker": "ocsort", "conf": 1.5, "iou": 0.5})
    with pytest.raises(ValueError, match="conf"):
        load_config(path)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "khong_co.json")
