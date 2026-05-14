"""YOLOE text-prompt experiment for kitchen PPE detection.

Loads YOLOE-26m, sets text prompts (default: "person", "head-hat", "glove"),
runs through the kitchen video and writes a per-class colored annotated MP4.

Edit the constants below before running. Quick self-contained experiment —
no project-config wiring.

Usage:
    python -m experiments.experiment_yoloe_text
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================================
# USER CONFIG
# ============================================================================
VIDEO_PATH    = Path("videos/billing_area.mp4")
OUTPUT_MP4    = Path("data/annotated/CAM-04.yoloe_text.mp4")

TEXT_PROMPTS  = ["head-hat", "white-glove"]

YOLOE_WEIGHTS = "data/weights/yoloe-11m-seg.pt"
DEVICE        = "cuda"
IMGSZ         = 640
CONF          = 0.30
HALF          = False
# ============================================================================


# Per-class BGR colors (cycled if more classes than colors).
CLASS_COLORS = [
    (255, 200, 80),    # blue   — person
    (0, 255, 255),     # yellow — head-hat
    (180, 220, 100),   # green  — glove
    (200, 100, 200),
    (0, 165, 255),
]


def _color(cls_idx: int) -> tuple[int, int, int]:
    return CLASS_COLORS[cls_idx % len(CLASS_COLORS)]


def _draw_predictions(frame: np.ndarray, result, class_names: list[str],
                      conf_thresh: float) -> dict[str, int]:
    counts = {n: 0 for n in class_names}
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return counts
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for i in range(len(xyxy)):
        if float(confs[i]) < conf_thresh:
            continue
        cls_idx = int(cls[i])
        if cls_idx >= len(class_names):
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i].tolist())
        color = _color(cls_idx)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls_idx]} {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly2 = max(y1, th + 4)
        cv2.rectangle(frame, (x1, ly2 - th - 4),
                      (x1 + tw + 4, ly2), color, -1)
        cv2.putText(frame, label, (x1 + 2, ly2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        counts[class_names[cls_idx]] += 1
    return counts


def _draw_footer(frame: np.ndarray, text: str) -> None:
    h = frame.shape[0]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad = 6
    cv2.rectangle(frame, (0, h - th - 2 * pad),
                  (tw + 2 * pad, h), (0, 0, 0), -1)
    cv2.putText(frame, text, (pad, h - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 245), 1, cv2.LINE_AA)


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"video not found: {VIDEO_PATH}")
        sys.exit(1)
    print(f"video   : {VIDEO_PATH}")
    print(f"prompts : {TEXT_PROMPTS}")
    print(f"weights : {YOLOE_WEIGHTS}  imgsz={IMGSZ}  conf={CONF}")

    print(f"\nloading YOLOE  device={DEVICE}")
    from ultralytics import YOLOE
    model = YOLOE(YOLOE_WEIGHTS)
    # Text prompts → CLIP text embeddings → set as class embeddings.
    model.set_classes(TEXT_PROMPTS, model.get_text_pe(TEXT_PROMPTS))
    model.model.names = {i: n for i, n in enumerate(TEXT_PROMPTS)}
    print(f"classes : {TEXT_PROMPTS}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"cannot open video: {VIDEO_PATH}")
        sys.exit(1)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    OUTPUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, fps, (W, H))
    if not writer.isOpened():
        print(f"cannot open writer: {OUTPUT_MP4}")
        sys.exit(1)

    footer = (
        f"YOLOE-TEXT  |  {YOLOE_WEIGHTS}  |  imgsz={IMGSZ}  |  "
        f"conf>={CONF}  |  classes: {', '.join(TEXT_PROMPTS)}"
    )
    print(f"\nwriting {OUTPUT_MP4}")
    print(f"footer: {footer}")

    totals = {n: 0 for n in TEXT_PROMPTS}
    frame_idx = 0
    t0 = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            result = model.predict(
                source=frame, imgsz=IMGSZ, conf=CONF,
                device=DEVICE, half=HALF, verbose=False,
            )[0]
            counts = _draw_predictions(frame, result, TEXT_PROMPTS, CONF)
            for n, c in counts.items():
                totals[n] += c
            _draw_footer(frame, footer)
            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.monotonic() - t0
                print(f"  frame {frame_idx}/{total}  "
                      f"({frame_idx / max(elapsed, 1e-6):.1f} fps wall)  "
                      f"totals: {totals}")
    finally:
        cap.release()
        writer.release()

    elapsed = time.monotonic() - t0
    print()
    print(f"done. {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.1f} fps).")
    print(f"per-class detection totals: {totals}")
    print(f"output: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
