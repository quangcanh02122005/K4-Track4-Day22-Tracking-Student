#!/usr/bin/env python
"""Chạy tracker trên video_1 với nhiều ngưỡng ``conf`` rồi chấm HOTA / MOTA / IDF1.

Với mỗi cặp (tracker, conf): ``run_tracking.py`` sinh ``video_1.txt`` (đủ frame), sau đó
``evaluate_practice.py`` chấm bằng TrackEval. ``--iou`` giữ nguyên giữa các lần để chỉ đổi
một tham số mỗi lần. Chỉ video_1 có nhãn nên chỉ chấm số cho video này.

Ví dụ:
    python scripts/score_video1.py --lab-data-root "$LAB_DATA" \\
        --trackers botsort --confs 0.15 0.3 0.5 --out runs/score --device cuda:0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

TRACKEVAL_URL = "https://github.com/JonathonLuiten/TrackEval.git"
# Tên benchmark trung lập: TrackEval chỉ dùng nó để đặt tên thư mục.
BENCHMARK = "lab"
SPLIT = "train"
PRACTICE_VIDEO = "video_1"
TABLE_METRICS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_Re", "IDs")


def run_name(tracker: str, conf: float) -> str:
    """Đặt tên lần chấm theo tracker và ngưỡng conf.

    Args:
        tracker: Tên tracker, ví dụ ``botsort``.
        conf: Ngưỡng confidence của detector.

    Returns:
        Tên dạng ``botsort_conf15_video1`` (conf nhân 100, làm tròn).
    """
    return f"{tracker}_conf{round(conf * 100):02d}_{PRACTICE_VIDEO.replace('_', '')}"


def summary_path(trackeval_root: Path, name: str) -> Path:
    """Đường dẫn file tóm tắt người đi bộ mà TrackEval ghi cho một lần chấm.

    Args:
        trackeval_root: Thư mục gốc bản clone TrackEval.
        name: Tên lần chấm, trùng với ``--run-name`` của ``evaluate_practice.py``.

    Returns:
        Đường dẫn tới ``pedestrian_summary.txt`` của lần chấm đó.
    """
    return (
        trackeval_root / "data" / "trackers" / "mot_challenge" / f"{BENCHMARK}-{SPLIT}"
        / name / "pedestrian_summary.txt"
    )


def parse_summary(text: str) -> Dict[str, float]:
    """Đọc file ``pedestrian_summary.txt`` (dòng tên cột, dòng giá trị) thành dict.

    Args:
        text: Nội dung file tóm tắt của TrackEval.

    Returns:
        Dict ánh xạ tên chỉ số sang giá trị số.

    Raises:
        ValueError: Khi file không đúng hai dòng có cùng số cột.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"Cần đúng 2 dòng (tên cột, giá trị), nhận {len(lines)} dòng")
    names, values = lines[0].split(), lines[1].split()
    if len(names) != len(values):
        raise ValueError(f"Số cột ({len(names)}) khác số giá trị ({len(values)})")
    return {name: float(value) for name, value in zip(names, values)}


def format_table(rows: Sequence[Tuple[str, Dict[str, float]]]) -> str:
    """Dựng bảng so sánh gọn các chỉ số chính cho nhiều lần chấm.

    Args:
        rows: Dãy cặp ``(nhãn, chỉ_số)``, chỉ_số là kết quả của ``parse_summary``.

    Returns:
        Bảng văn bản căn cột, mỗi lần chấm một dòng.
    """
    label_width = max([len("lần chấm")] + [len(label) for label, _ in rows])
    header = "lần chấm".ljust(label_width) + "".join(f"{m:>9}" for m in TABLE_METRICS)
    body: List[str] = []
    for label, metrics in rows:
        cells = "".join(f"{metrics[m]:>9.2f}" if m in metrics else f"{'-':>9}" for m in TABLE_METRICS)
        body.append(label.ljust(label_width) + cells)
    return "\n".join([header, *body])


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
    """Chạy và chấm video_1 cho từng cặp (tracker, conf), rồi in bảng so sánh."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab-data-root", required=True, type=Path)
    parser.add_argument("--trackers", nargs="+", default=["botsort"])
    parser.add_argument("--confs", nargs="+", type=float, default=[0.15, 0.3, 0.5])
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("runs/score"))
    parser.add_argument("--trackeval-root", type=Path, default=Path("TrackEval"))
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    ensure_trackeval(args.trackeval_root)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, Dict[str, float]]] = []
    for tracker in args.trackers:
        for conf in args.confs:
            name = run_name(tracker, conf)
            run_dir = args.out / name
            print(f"\n=== {tracker} conf={conf} iou={args.iou} ===", flush=True)
            subprocess.run(
                [
                    sys.executable, str(scripts / "run_tracking.py"),
                    "--source", str(args.lab_data_root / PRACTICE_VIDEO / "img1"),
                    "--seq-name", PRACTICE_VIDEO,
                    "--tracker", tracker,
                    "--conf", str(conf), "--iou", str(args.iou),
                    "--device", args.device,
                    "--out", str(run_dir),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable, str(scripts / "evaluate_practice.py"),
                    "--trackeval-root", str(args.trackeval_root),
                    "--lab-data-root", str(args.lab_data_root),
                    "--submission", str(run_dir / f"{PRACTICE_VIDEO}.txt"),
                    "--run-name", name,
                    "--benchmark", BENCHMARK, "--split", SPLIT,
                ],
                check=True,
            )
            text = summary_path(args.trackeval_root, name).read_text()
            (args.out / f"{name}_summary.txt").write_text(text)
            rows.append((f"{tracker} conf={conf:g}", parse_summary(text)))

    table = format_table(rows)
    (args.out / "sweep_summary.txt").write_text(table + "\n", encoding="utf-8")
    print("\n===== TÓM TẮT video_1 =====")
    print(table)


if __name__ == "__main__":
    main()
