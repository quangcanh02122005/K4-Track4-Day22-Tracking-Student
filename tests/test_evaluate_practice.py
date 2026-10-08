"""Tests for the TrackEval command builder without TrackEval, a GPU, or a network."""

import subprocess
import sys
from pathlib import Path

from evaluate_practice import NUMPY_SHIM, trackeval_command


def test_command_has_expected_arguments(tmp_path: Path) -> None:
    cmd = trackeval_command(tmp_path, "nhom01_video1", "lab", "train")
    assert cmd[:3] == [sys.executable, "-c", NUMPY_SHIM]
    assert cmd[3] == str(tmp_path / "scripts" / "run_mot_challenge.py")
    assert cmd[cmd.index("--SEQ_INFO") + 1] == "video_1"
    assert cmd[cmd.index("--BENCHMARK") + 1] == "lab"
    assert cmd[cmd.index("--TRACKERS_TO_EVAL") + 1] == "nhom01_video1"


def test_shim_restores_removed_numpy_aliases_in_child_process(tmp_path: Path) -> None:
    script = tmp_path / "fake_trackeval.py"
    script.write_text("import numpy as np\nprint(np.float(1.5) + np.int(2), np.bool(1))\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", NUMPY_SHIM, str(script)], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["3.5", "True"]
