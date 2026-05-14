"""YOLOE *visual-prompt* experiment for gloves + hairnets.

Workflow:
  1. Open SAMPLE_FRAME_PATH in an OpenCV window.
  2. For each class (e.g. "glove", "hairnet"), drag rectangles around example
     instances. Press SPACE / ENTER to add the box, ESC to finish that class.
  3. The selected bboxes become YOLOE's *visual prompt* — the model derives
     a per-class embedding from them and detects similar-looking objects in
     every frame of VIDEO_PATH.
  4. Annotated MP4 is written to OUTPUT_MP4 with per-class boxes + counts.

Edit the constants at the top before running. The script is intentionally
self-contained (no project config) so it stays a quick experiment.

Usage:
    python -m experiments.experiment_yoloe_visual
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make `app.*` importable in case the user wants to swap in reference frames
# from data/reference_frames/ — kept optional, the script doesn't import app.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================================
# USER CONFIG — edit these before running
# ============================================================================
SAMPLE_FRAME_PATH = Path("videos/imgs/img.png")  # frame with clear examples
VIDEO_PATH        = Path("videos/kitchen.mp4")
OUTPUT_MP4        = Path("data/annotated/CAM-04.yoloe_visual.mp4")

# Cache for the computed Visual-Prompt Embeddings (VPE). If this file exists
# the interactive bbox picker is skipped and the cached embeddings are loaded
# directly. Delete the file (or change the path) to force a fresh selection.
VPE_CACHE_PATH = Path("data/weights/yoloe_vpe_kitchen.pt")

# Classes to prompt for, in display order. The user will be asked to draw
# example bboxes for each class in this order.
CLASS_NAMES = ["glove", "hairnet"]

# YOLOE model + inference knobs
YOLOE_WEIGHTS = "data/weights/yoloe-11s-seg.pt"
DEVICE        = "cuda"
IMGSZ         = 1280   # higher = better recall on small/distant objects
CONF          = 0.10   # YOLOE visual prompts often score lower than text
HALF          = False  # FP16 sometimes destabilises VPE — leave off for the demo
# ============================================================================


# Per-class colors (BGR) used both for the picker preview and final overlay.
CLASS_COLORS = [
    (0, 165, 255),     # orange — class 0
    (255, 200, 80),    # blue   — class 1
    (180, 220, 100),   # green  — class 2
    (200, 100, 200),   # magenta — class 3
    (0, 230, 230),     # yellow — class 4
]


def _color(cls_idx: int) -> tuple[int, int, int]:
    return CLASS_COLORS[cls_idx % len(CLASS_COLORS)]


# ---------------------------------------------------------------------------
# Interactive bbox picker
# ---------------------------------------------------------------------------

def _paint_previous(frame: np.ndarray,
                    prior_bboxes: list[tuple[int, int, int, int, int]]) -> None:
    """Paint already-selected boxes from earlier classes onto a working copy."""
    for (x1, y1, x2, y2, cls_idx) in prior_bboxes:
        color = _color(cls_idx)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls{cls_idx}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def select_bboxes_for(class_name: str, base_frame: np.ndarray,
                       prior_bboxes: list[tuple[int, int, int, int, int]]
                       ) -> list[tuple[int, int, int, int]]:
    """Show base_frame (with prior-class boxes painted) and let the user
    drag rectangles for `class_name`. Returns [(x, y, w, h), ...] in image
    coords. SPACE/ENTER adds another box; ESC ends selection for this class.
    """
    canvas = base_frame.copy()
    _paint_previous(canvas, prior_bboxes)
    title = (
        f"Select examples of '{class_name}'  |  "
        "drag a box, SPACE/ENTER = add another, ESC = done"
    )
    rois = cv2.selectROIs(title, canvas, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    return [tuple(int(v) for v in r) for r in rois]


# ---------------------------------------------------------------------------
# Drawing on prediction output
# ---------------------------------------------------------------------------

def _draw_predictions(frame: np.ndarray, result, class_names: list[str],
                      conf_thresh: float) -> dict[str, int]:
    """Render result.boxes on `frame`. Returns per-class hit count."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _compute_and_save_vpe(model, sample: np.ndarray) -> None:
    """Run the interactive bbox picker, compute visual-prompt embeddings,
    write them (along with class names) to VPE_CACHE_PATH."""
    print()
    print("== Interactive bbox selection ==")
    print("For each class you'll see the sample frame. Drag a rectangle around")
    print("each example, SPACE/ENTER to add another, ESC when done with that class.")
    visual_bboxes_xyxy: list[list[float]] = []
    visual_cls: list[int] = []
    painted: list[tuple[int, int, int, int, int]] = []
    for cls_idx, name in enumerate(CLASS_NAMES):
        print(f"\n-- class {cls_idx}: '{name}' --")
        rois = select_bboxes_for(name, sample, painted)
        if not rois:
            print(f"  (no examples for '{name}')")
            continue
        for (x, y, w, h) in rois:
            x1, y1, x2, y2 = x, y, x + w, y + h
            visual_bboxes_xyxy.append([float(x1), float(y1),
                                       float(x2), float(y2)])
            visual_cls.append(cls_idx)
            painted.append((x1, y1, x2, y2, cls_idx))
        print(f"  {len(rois)} example(s) recorded for '{name}'")
    if not visual_bboxes_xyxy:
        print("\nno visual prompts were selected — aborting.")
        sys.exit(1)
    print(f"\ntotal visual prompts: {len(visual_bboxes_xyxy)}")

    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    prompts = {
        "bboxes": np.array(visual_bboxes_xyxy, dtype=np.float32),
        "cls":    np.array(visual_cls, dtype=np.int32),
    }
    print("computing visual-prompt embeddings from sample frame...")
    _ = model.predict(
        source=str(SAMPLE_FRAME_PATH),
        visual_prompts=prompts,
        refer_image=str(SAMPLE_FRAME_PATH),
        predictor=YOLOEVPSegPredictor,
        imgsz=IMGSZ,
        conf=CONF,
        device=DEVICE,
        half=HALF,
        verbose=False,
    )
    import torch
    vpe = model.model.pe.detach().cpu()
    VPE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "names": CLASS_NAMES,
        "vpe": vpe,
        "source_frame": str(SAMPLE_FRAME_PATH),
        "bboxes_xyxy": visual_bboxes_xyxy,
        "cls": visual_cls,
    }, VPE_CACHE_PATH)
    print(f"saved VPE cache  -> {VPE_CACHE_PATH}  shape={tuple(vpe.shape)}")


def _load_vpe_into(model) -> None:
    """Load embeddings from VPE_CACHE_PATH and install them on the model.
    Skips the interactive picker entirely."""
    import torch
    ckpt = torch.load(str(VPE_CACHE_PATH), weights_only=False)
    cached_names = list(ckpt["names"])
    if cached_names != CLASS_NAMES:
        print(f"  WARNING: cached classes {cached_names} != configured {CLASS_NAMES}")
        print("  using cached names — edit CLASS_NAMES or delete the cache to redo selection.")
        names = cached_names
    else:
        names = CLASS_NAMES
    vpe = ckpt["vpe"]
    model.model.set_classes(names, vpe)
    print(f"loaded VPE cache <- {VPE_CACHE_PATH}  shape={tuple(vpe.shape)}  "
          f"classes={names}  ({len(ckpt.get('bboxes_xyxy', []))} prompt bboxes)")


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"video not found: {VIDEO_PATH}")
        sys.exit(1)
    print(f"video  : {VIDEO_PATH}")
    print(f"classes: {CLASS_NAMES}")
    print(f"weights: {YOLOE_WEIGHTS}  imgsz={IMGSZ}  conf={CONF}")
    print(f"vpe cache : {VPE_CACHE_PATH}  ({'hit' if VPE_CACHE_PATH.exists() else 'miss — will pick now'})")

    from ultralytics import YOLOE
    print(f"\nloading YOLOE  weights={YOLOE_WEIGHTS}  device={DEVICE}")
    model = YOLOE(YOLOE_WEIGHTS)

    if VPE_CACHE_PATH.exists():
        _load_vpe_into(model)
    else:
        if not SAMPLE_FRAME_PATH.exists():
            print(f"sample frame not found: {SAMPLE_FRAME_PATH}")
            print("hint: run scripts/extract_reference_frames.py first")
            sys.exit(1)
        sample = cv2.imread(str(SAMPLE_FRAME_PATH))
        if sample is None:
            print("failed to read sample frame")
            sys.exit(1)
        print(f"sample : {SAMPLE_FRAME_PATH}  ({sample.shape[1]}x{sample.shape[0]})")
        _compute_and_save_vpe(model, sample)

    # User-facing class names override (the model otherwise labels classes
    # as objectN after a visual-prompt warmup).
    model.model.names = {i: name for i, name in enumerate(CLASS_NAMES)}

    # ---- 3) Run on video, write annotated MP4 ----
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
        f"YOLOE-VP  |  {YOLOE_WEIGHTS}  |  imgsz={IMGSZ}  |  "
        f"conf>={CONF}  |  classes: {', '.join(CLASS_NAMES)}"
    )
    print(f"\nwriting {OUTPUT_MP4}")
    print(f"footer: {footer}")

    totals = {n: 0 for n in CLASS_NAMES}
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
            counts = _draw_predictions(frame, result, CLASS_NAMES, CONF)
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
