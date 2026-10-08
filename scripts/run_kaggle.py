#!/usr/bin/env python
"""Chạy toàn bộ bản nộp trên Kaggle: cài gói, tìm dữ liệu, kiểm tra, rồi chạy 5 video.

Notebook Kaggle chỉ clone repo và gọi script này, nên sửa lỗi chỉ cần ``git push``.

Ví dụ (trong notebook Kaggle):
    !python scripts/run_kaggle.py --out /kaggle/working/runs/nop_bai
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

SEARCH_ROOTS = (Path("/kaggle/input"), Path("/kaggle/working/data_lab21"))

# boxmot 10.0.42 ghim numpy==1.23.1 (không có wheel cho Python 3.13 của Kaggle),
# nên cài boxmot với --no-deps rồi cài riêng phần còn lại, dùng numpy có sẵn.
PIP_STEPS = (
    ["lapx", "filterpy", "ftfy", "gdown", "GitPython", "loguru", "regex", "yacs", "ultralytics>=8.4"],
    ["--no-deps", "boxmot==10.0.42"],
)


def find_lab_data(roots: Sequence[Path] = SEARCH_ROOTS) -> Optional[Path]:
    """Tìm thư mục chứa trực tiếp ``video_1`` … ``video_5``.

    Args:
        roots: Các thư mục gốc để tìm, theo thứ tự ưu tiên.

    Returns:
        Thư mục cha của ``video_1`` (có ``img1/``), hoặc ``None`` nếu không thấy.
    """
    for root in roots:
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("video_1")):
            if candidate.is_dir() and (candidate / "img1").is_dir():
                return candidate.parent
    return None


def install_packages() -> None:
    """Cài các gói cần thiết rồi import thử để lỗi hiện ra sớm.

    Raises:
        subprocess.CalledProcessError: Khi một bước ``pip install`` thất bại.
        ImportError: Khi cài xong mà ``boxmot`` hoặc ``ultralytics`` không import được.
    """
    for step in PIP_STEPS:
        cmd = [sys.executable, "-m", "pip", "install", "-q", *step]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    subprocess.run(
        [sys.executable, "-c", "import boxmot, ultralytics; from boxmot.tracker_zoo import create_tracker"],
        check=True,
    )


def main() -> None:
    """Cài gói, tìm dữ liệu, chạy ``check_data.py`` rồi ``run_all.py``.

    Raises:
        SystemExit: Khi không tìm thấy dữ liệu hoặc thiếu file nộp.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/runs/nop_bai"))
    parser.add_argument("--config", type=Path, default=Path("configs/nop_bai.json"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    install_packages()
    lab_data = find_lab_data()
    if lab_data is None:
        raise SystemExit("Không thấy video_1/img1. Hãy Add Input một Kaggle dataset chứa video_1 ... video_5.")
    print("LAB_DATA =", lab_data, flush=True)

    scripts = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(scripts / "check_data.py"), "--lab-data-root", str(lab_data)], check=True)
    subprocess.run(
        [
            sys.executable, str(scripts / "run_all.py"),
            "--lab-data-root", str(lab_data),
            "--config", str(args.config),
            "--out", str(args.out),
            "--device", args.device,
            "--save-video",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
