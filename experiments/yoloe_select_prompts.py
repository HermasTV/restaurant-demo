"""YOLOE visual-prompt **selector only**.

Opens SAMPLE_FRAME_PATH in an OpenCV window, lets the user drag bboxes for
each class (default: glove, hairnet), runs the YOLOE warmup that computes
the Visual-Prompt Embeddings from those bboxes, and saves the embeddings
plus their metadata to VPE_CACHE_PATH.

Nothing else — no video processing, no annotated MP4. The inference run
is a separate script that just loads the saved .pt.

Usage:
    python -m experiments.yoloe_select_prompts
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================================
# USER CONFIG — edit before running
# ============================================================================
SAMPLE_FRAME_PATH = Path("videos/imgs/img.png")
VPE_CACHE_PATH    = Path("data/weights/yoloe_vpe_kitchen.pt")

CLASS_NAMES = ["glove", "hairnet"]

YOLOE_WEIGHTS = "data/weights/yoloe-11s-seg.pt"
DEVICE        = "cuda"
IMGSZ         = 1280
CONF          = 0.10
HALF          = False
# ============================================================================


CLASS_COLORS = [
    (0, 165, 255),     # orange — class 0
    (255, 200, 80),    # blue   — class 1
    (180, 220, 100),
    (200, 100, 200),
    (0, 230, 230),
]


def _color(cls_idx: int) -> tuple[int, int, int]:
    return CLASS_COLORS[cls_idx % len(CLASS_COLORS)]


def _paint_previous(frame: np.ndarray,
                    prior: list[tuple[int, int, int, int, int]]) -> None:
    for (x1, y1, x2, y2, cls_idx) in prior:
        color = _color(cls_idx)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls{cls_idx}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def select_bboxes_for(class_name: str, base_frame: np.ndarray,
                      prior: list[tuple[int, int, int, int, int]]
                      ) -> list[tuple[int, int, int, int]]:
    """Drag rectangles for `class_name` until ESC. SPACE/ENTER adds another."""
    canvas = base_frame.copy()
    _paint_previous(canvas, prior)
    title = (
        f"Select examples of '{class_name}'  |  "
        "drag a box, SPACE/ENTER = add another, ESC = done"
    )
    rois = cv2.selectROIs(title, canvas, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    return [tuple(int(v) for v in r) for r in rois]


def main() -> None:
    if not SAMPLE_FRAME_PATH.exists():
        print(f"sample frame not found: {SAMPLE_FRAME_PATH}")
        sys.exit(1)
    sample = cv2.imread(str(SAMPLE_FRAME_PATH))
    if sample is None:
        print("failed to read sample frame")
        sys.exit(1)
    print(f"sample  : {SAMPLE_FRAME_PATH}  ({sample.shape[1]}x{sample.shape[0]})")
    print(f"classes : {CLASS_NAMES}")
    print(f"output  : {VPE_CACHE_PATH}")

    print()
    print("== bbox selection ==")
    print("For each class: drag a rectangle around an example, SPACE/ENTER to")
    print("add another, ESC to move on to the next class.")
    bboxes_xyxy: list[list[float]] = []
    cls: list[int] = []
    painted: list[tuple[int, int, int, int, int]] = []
    for cls_idx, name in enumerate(CLASS_NAMES):
        print(f"\n-- class {cls_idx}: '{name}' --")
        rois = select_bboxes_for(name, sample, painted)
        if not rois:
            print(f"  (no examples for '{name}')")
            continue
        for (x, y, w, h) in rois:
            x1, y1, x2, y2 = x, y, x + w, y + h
            bboxes_xyxy.append([float(x1), float(y1), float(x2), float(y2)])
            cls.append(cls_idx)
            painted.append((x1, y1, x2, y2, cls_idx))
        print(f"  {len(rois)} example(s) recorded for '{name}'")

    if not bboxes_xyxy:
        print("\nno visual prompts selected — aborting.")
        sys.exit(1)
    print(f"\ntotal visual prompts: {len(bboxes_xyxy)}")

    print(f"\nloading YOLOE  weights={YOLOE_WEIGHTS}  device={DEVICE}")
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    model = YOLOE(YOLOE_WEIGHTS)

    prompts = {
        "bboxes": np.array(bboxes_xyxy, dtype=np.float32),
        "cls":    np.array(cls, dtype=np.int32),
    }
    print("computing visual-prompt embeddings...")
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
        "names":        CLASS_NAMES,
        "vpe":          vpe,
        "weights":      YOLOE_WEIGHTS,
        "imgsz":        IMGSZ,
        "source_frame": str(SAMPLE_FRAME_PATH),
        "bboxes_xyxy":  bboxes_xyxy,
        "cls":          cls,
    }, VPE_CACHE_PATH)
    print(f"saved -> {VPE_CACHE_PATH}  shape={tuple(vpe.shape)}")


if __name__ == "__main__":
    main()
