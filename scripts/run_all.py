#!/usr/bin/env python
"""Chạy tracker cho cả 5 video theo file cấu hình và kiểm tra đủ file nộp.

Mỗi video có một dòng cấu hình (tracker, conf, iou) trong file JSON. Script gọi
lại ``run_tracking.run`` cho từng video, chạy ĐỦ frame (không ``--max-frames``),
rồi báo video nào còn thiếu ``video_N.txt``.

Ví dụ:
    python scripts/run_all.py \\
        --lab-data-root "$LAB_DATA" \\
        --config configs/nop_bai.json \\
        --out runs/nop_bai --device cuda:0 --save-video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from check_data import VIDEOS

REQUIRED_KEYS = ("tracker", "conf", "iou")


def load_config(config_path: Path, valid_trackers: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    """Đọc và kiểm tra file cấu hình tracker của từng video.

    Args:
        config_path: File JSON dạng ``{"video_1": {"tracker": ..., "conf": ..., "iou": ...}, ...}``.
            Khóa không phải tên video (ví dụ ``_ghi_chu``) được bỏ qua.
        valid_trackers: Danh sách tracker hợp lệ. ``None`` thì không kiểm tra tên tracker.

    Returns:
        Dict ánh xạ tên video sang dict ``tracker`` / ``conf`` / ``iou``, đủ cả năm video.

    Raises:
        FileNotFoundError: Khi không thấy file cấu hình.
        ValueError: Khi thiếu video, thiếu khóa, tracker lạ, hoặc conf / iou ngoài khoảng (0, 1].
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Không thấy file cấu hình {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config: Dict[str, dict] = {}
    for name in VIDEOS:
        if name not in raw:
            raise ValueError(f"Cấu hình thiếu {name}")
        entry = raw[name]
        missing = [key for key in REQUIRED_KEYS if key not in entry]
        if missing:
            raise ValueError(f"{name} thiếu khóa: {', '.join(missing)}")
        if valid_trackers is not None and entry["tracker"] not in valid_trackers:
            raise ValueError(f"{name}: tracker '{entry['tracker']}' không hợp lệ, chọn một trong {list(valid_trackers)}")
        for key in ("conf", "iou"):
            if not 0 < float(entry[key]) <= 1:
                raise ValueError(f"{name}: {key}={entry[key]} phải nằm trong (0, 1]")
        config[name] = {"tracker": entry["tracker"], "conf": float(entry["conf"]), "iou": float(entry["iou"])}
    return config


def missing_submission_files(out_dir: Path, videos: Sequence[str] = VIDEOS) -> List[str]:
    """Liệt kê file nộp còn thiếu hoặc rỗng.

    Args:
        out_dir: Thư mục kết quả, thường là ``runs/nop_bai``.
        videos: Các video cần có file. Mặc định là cả năm video.

    Returns:
        Danh sách tên file (``video_N.txt``) không tồn tại hoặc rỗng. Rỗng nghĩa là đủ.
    """
    missing = []
    for name in videos:
        path = out_dir / f"{name}.txt"
        if not path.exists() or path.stat().st_size == 0:
            missing.append(path.name)
    return missing


def main() -> None:
    """Chạy tracker cho từng video rồi kiểm tra đủ năm file nộp.

    Raises:
        SystemExit: Khi sau khi chạy vẫn còn thiếu file nộp.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab-data-root", required=True, type=Path, help="Thư mục chứa video_1 ... video_5")
    parser.add_argument("--config", type=Path, default=Path("configs/nop_bai.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/nop_bai"))
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda:0', ...")
    parser.add_argument("--save-video", action="store_true", help="Xuất video xem thử cho từng video")
    parser.add_argument("--fps", type=int, default=20, help="FPS của video xem thử")
    parser.add_argument("--only", nargs="+", choices=VIDEOS, help="Chỉ chạy các video này")
    args = parser.parse_args()

    # Nhập muộn: run_tracking cần ultralytics / boxmot, test của file này thì không.
    import run_tracking

    config = load_config(args.config, valid_trackers=run_tracking.TRACKER_CHOICES)
    for name in args.only or VIDEOS:
        entry = config[name]
        print(f"\n=== {name}: {entry['tracker']} conf={entry['conf']} iou={entry['iou']} ===")
        run_tracking.run(argparse.Namespace(
            source=str(args.lab_data_root / name / "img1"),
            seq_name=name,
            tracker=entry["tracker"],
            conf=entry["conf"],
            iou=entry["iou"],
            device=args.device,
            out=str(args.out),
            save_video=args.save_video,
            fps=args.fps,
            max_frames=0,
        ))

    selected = args.only or VIDEOS
    missing = missing_submission_files(args.out, selected)
    if missing:
        raise SystemExit(f"Còn thiếu file nộp trong {args.out}: {', '.join(missing)}")
    print(f"\nĐủ {len(selected)} file kết quả trong {args.out}.")


if __name__ == "__main__":
    main()
