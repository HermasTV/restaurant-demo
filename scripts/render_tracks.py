"""Re-render an annotated MP4 from a JSONL track log + source video.

Useful when re-styling overlays without re-running the GPU pipeline, or
when running on a machine without torch/ultralytics installed.

Usage:
    python -m scripts.render_tracks --cam CAM-01
    python -m scripts.render_tracks --cam CAM-04 --no-conf --out /tmp/cam04.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils.cameras import DATA_DIR, camera_by_id  # noqa: E402
from app.pipeline.draw import draw_tracks  # noqa: E402

TRACKS_DIR = DATA_DIR / "tracks"
ANNOTATED_DIR = DATA_DIR / "annotated"


def _load_tracks_by_frame(path: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_frame[int(rec["frame_idx"])] = rec.get("tracks", [])
    return by_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render JSONL tracks onto source video.")
    p.add_argument("--cam", required=True, help="Camera ID, e.g. CAM-01.")
    p.add_argument("--jsonl", default=None, help="Override JSONL path.")
    p.add_argument("--out", default=None, help="Override output MP4 path.")
    p.add_argument("--no-ids", action="store_true", help="Hide track ID labels.")
    p.add_argument("--no-conf", action="store_true", help="Hide confidence labels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cam = camera_by_id(args.cam)
    jsonl_path = Path(args.jsonl) if args.jsonl else TRACKS_DIR / f"{cam.cam_id}.jsonl"
    out_path = Path(args.out) if args.out else ANNOTATED_DIR / f"{cam.cam_id}.rendered.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        raise SystemExit(f"missing tracks: {jsonl_path}")
    if not cam.path.exists():
        raise SystemExit(f"missing video: {cam.path}")

    print(f"loading {jsonl_path}")
    by_frame = _load_tracks_by_frame(jsonl_path)
    print(f"  {len(by_frame)} frames of tracks")

    cap = cv2.VideoCapture(str(cam.path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {cam.path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer: {out_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            tracks = by_frame.get(frame_idx, [])
            draw_tracks(
                frame,
                tracks,
                show_ids=not args.no_ids,
                show_conf=not args.no_conf,
            )
            writer.write(frame)
            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"  rendered {frame_idx} frames")
    finally:
        cap.release()
        writer.release()
    print(f"wrote {out_path}  ({frame_idx} frames)")


if __name__ == "__main__":
    main()
