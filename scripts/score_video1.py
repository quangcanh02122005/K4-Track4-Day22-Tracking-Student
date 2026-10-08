#!/usr/bin/env python
"""Chạy nhiều tracker trên video_1 rồi chấm HOTA / MOTA / IDF1 để so sánh.

Với mỗi tracker: ``run_tracking.py`` sinh ``video_1.txt`` (đủ frame), sau đó
``evaluate_practice.py`` chấm bằng TrackEval. Mọi tracker dùng cùng ``--conf`` / ``--iou``
để so sánh công bằng. Chỉ video_1 có nhãn nên chỉ chấm số cho video này.

Ví dụ:
    python scripts/score_video1.py --lab-data-root "$LAB_DATA" \\
        --trackers bytetrack botsort --out runs/score --device cuda:0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TRACKEVAL_URL = "https://github.com/JonathonLuiten/TrackEval.git"
# Tên benchmark trung lập: TrackEval chỉ dùng nó để đặt tên thư mục.
BENCHMARK = "lab"
SPLIT = "train"
PRACTICE_VIDEO = "video_1"


def summary_path(trackeval_root: Path, run_name: str) -> Path:
    """Đường dẫn file tóm tắt người đi bộ mà TrackEval ghi cho một lần chấm.

    Args:
        trackeval_root: Thư mục gốc bản clone TrackEval.
        run_name: Tên lần chấm, trùng với ``--run-name`` của ``evaluate_practice.py``.

    Returns:
        Đường dẫn tới ``pedestrian_summary.txt`` của lần chấm đó.
    """
    return (
        trackeval_root / "data" / "trackers" / "mot_challenge" / f"{BENCHMARK}-{SPLIT}"
        / run_name / "pedestrian_summary.txt"
    )


def ensure_trackeval(trackeval_root: Path) -> None:
    """Clone và cài TrackEval nếu chưa có.

    Args:
        trackeval_root: Nơi đặt bản clone TrackEval.

    Raises:
        subprocess.CalledProcessError: Khi ``git clone`` hoặc ``pip install`` thất bại.
    """
    if not trackeval_root.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", TRACKEVAL_URL, str(trackeval_root)], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(trackeval_root)],
        check=True,
    )


def main() -> None:
    """Chạy và chấm video_1 cho từng tracker, rồi in bảng tóm tắt."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab-data-root", required=True, type=Path)
    parser.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort"])
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("runs/score"))
    parser.add_argument("--trackeval-root", type=Path, default=Path("TrackEval"))
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    ensure_trackeval(args.trackeval_root)
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for tracker in args.trackers:
        run_dir = args.out / tracker
        print(f"\n=== {tracker} conf={args.conf} iou={args.iou} ===", flush=True)
        subprocess.run(
            [
                sys.executable, str(scripts / "run_tracking.py"),
                "--source", str(args.lab_data_root / PRACTICE_VIDEO / "img1"),
                "--seq-name", PRACTICE_VIDEO,
                "--tracker", tracker,
                "--conf", str(args.conf), "--iou", str(args.iou),
                "--device", args.device,
                "--out", str(run_dir), "--save-video",
            ],
            check=True,
        )
        run_name = f"{tracker}_video1"
        subprocess.run(
            [
                sys.executable, str(scripts / "evaluate_practice.py"),
                "--trackeval-root", str(args.trackeval_root),
                "--lab-data-root", str(args.lab_data_root),
                "--submission", str(run_dir / f"{PRACTICE_VIDEO}.txt"),
                "--run-name", run_name,
                "--benchmark", BENCHMARK, "--split", SPLIT,
            ],
            check=True,
        )
        summaries[tracker] = summary_path(args.trackeval_root, run_name).read_text()
        (args.out / f"{tracker}_summary.txt").write_text(summaries[tracker])

    print("\n===== TÓM TẮT video_1 =====")
    for tracker, text in summaries.items():
        print(f"\n[{tracker}]\n{text}")


if __name__ == "__main__":
    main()
