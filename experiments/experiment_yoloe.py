"""Zero-shot YOLOE experiment for hairnet + gloves on a single camera.

Detection only (no tracking). Writes an annotated MP4 named after the model
so multiple runs are kept side-by-side for visual comparison.

Usage:
    python -m scripts.experiment_yoloe                       # CAM-04, default model
    python -m scripts.experiment_yoloe --cam CAM-04 \
        --weights yoloe-11s-seg.pt --prompts glove hairnet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils.cameras import DATA_DIR, camera_by_id  # noqa: E402
from app.pipeline.draw import _color_for_id  # reuse the hue hash  # noqa: E402

ANNOTATED_DIR = DATA_DIR / "annotated"
DEFAULT_WEIGHTS = "data/weights/yoloe-11s-seg.pt"
DEFAULT_PROMPTS = ("glove", "hairnet")
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.15  # zero-shot recall is fragile — start permissive


def _draw_detections(
    frame: np.ndarray,
    xyxy: np.ndarray,
    cls: np.ndarray,
    conf: np.ndarray,
    names: list[str],
) -> None:
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i].tolist())
        cid = int(cls[i])
        color = _color_for_id(cid + 1)  # +1 so id=0 isn't pure red
        name = names[cid] if 0 <= cid < len(names) else str(cid)
        label = f"{name} {float(conf[i]):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly2 = max(y1, th + 4)
        cv2.rectangle(frame, (x1, ly2 - th - 4), (x1 + tw + 4, ly2), color, -1)
        cv2.putText(frame, label, (x1 + 2, ly2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_footer(frame: np.ndarray, text: str) -> None:
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    pad = 6
    cv2.rectangle(frame, (0, h - th - 2 * pad), (tw + 2 * pad, h),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (pad, h - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOE zero-shot PPE experiment.")
    p.add_argument("--cam", default="CAM-04")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS),
                   help="Class prompts for YOLOE (e.g. glove hairnet).")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None,
                   help="Override output MP4 path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cam = camera_by_id(args.cam)
    if not cam.path.exists():
        raise SystemExit(f"missing video: {cam.path}")

    from ultralytics import YOLOE
    print(f"loading YOLOE  weights={args.weights}  device={args.device}")
    model = YOLOE(args.weights)
    model.set_classes(args.prompts, model.get_text_pe(args.prompts))
    print(f"prompts set: {args.prompts}")

    cap = cv2.VideoCapture(str(cam.path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open: {cam.path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    model_tag = Path(args.weights).stem
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        ANNOTATED_DIR / f"{cam.cam_id}.{model_tag}.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer: {out_path}")

    footer = f"model: {model_tag} | classes: {', '.join(args.prompts)} | conf>={args.conf}"
    print(f"writing {out_path}")
    print(f"footer: {footer}")

    frame_idx = 0
    total_dets = 0
    per_class = {n: 0 for n in args.prompts}
    t0 = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            r = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                _draw_detections(frame, xyxy, cls, conf, list(args.prompts))
                total_dets += len(cls)
                for c in cls:
                    name = args.prompts[int(c)]
                    per_class[name] += 1

            _draw_footer(frame, footer)
            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.monotonic() - t0
                print(f"  frame {frame_idx}/{total}  "
                      f"({frame_idx / max(elapsed, 1e-6):.1f} fps)  "
                      f"dets so far: {total_dets}  {per_class}")
    finally:
        cap.release()
        writer.release()

    elapsed = time.monotonic() - t0
    print()
    print(f"done  {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps)")
    print(f"total detections: {total_dets}")
    print(f"per class:        {per_class}")
    print(f"output:           {out_path}")


if __name__ == "__main__":
    main()
