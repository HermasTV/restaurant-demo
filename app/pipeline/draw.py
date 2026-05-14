"""Pure-OpenCV overlay drawing shared by the live worker and the render script.

Intentionally has NO torch / ultralytics / supervision import so the render
script remains usable on a machine without the ML stack.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR color per track ID."""
    # Golden-ratio hue spacing keeps adjacent IDs visually distinct.
    hue = int((track_id * 47) % 180)  # OpenCV hue range is 0-179
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(
    frame_bgr: np.ndarray,
    tracks: list[dict[str, Any]],
    show_ids: bool = True,
    show_conf: bool = True,
) -> np.ndarray:
    """Draw boxes + labels onto `frame_bgr` in place; also return it."""
    for t in tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in t["bbox"])
        tid = int(t["id"])
        color = _color_for_id(tid)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        if show_ids or show_conf:
            parts = []
            if show_ids:
                parts.append(f"ID:{tid}")
            if show_conf:
                parts.append(f"{t.get('conf', 0.0):.2f}")
            label = " ".join(parts)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            ly2 = max(y1, th + 4)
            cv2.rectangle(
                frame_bgr,
                (x1, ly2 - th - 4),
                (x1 + tw + 4, ly2),
                color,
                -1,
            )
            cv2.putText(
                frame_bgr,
                label,
                (x1 + 2, ly2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return frame_bgr


# Per-class colors (BGR) for stateless detections like PPE. Anything not in
# the map falls back to bright yellow.
DEFAULT_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "glove":   (180, 220, 100),   # green
    "hairnet": (0, 255, 255),     # yellow
    "mask":    (255, 100, 200),   # pink
    "apron":   (200, 150, 50),    # tan
}
DEFAULT_FALLBACK_COLOR = (0, 255, 255)


def draw_labels(
    frame_bgr: np.ndarray,
    items: list[dict[str, Any]],
    class_colors: dict[str, tuple[int, int, int]] | None = None,
    show_conf: bool = True,
) -> np.ndarray:
    """Draw boxes labeled by their `class` field (no track IDs).

    Each item is `{bbox, class, conf}`. Color is picked from `class_colors`
    (or `DEFAULT_CLASS_COLORS` if omitted), falling back to bright yellow.
    Used for stateless detections (PPE classes like glove / hairnet).
    """
    colors = class_colors or DEFAULT_CLASS_COLORS
    for it in items:
        x1, y1, x2, y2 = (int(round(v)) for v in it["bbox"])
        name = str(it.get("class", "?"))
        color = colors.get(name, DEFAULT_FALLBACK_COLOR)
        label = f"{name} {it.get('conf', 0.0):.2f}" if show_conf else name
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly2 = max(y1, th + 4)
        cv2.rectangle(frame_bgr, (x1, ly2 - th - 4),
                      (x1 + tw + 4, ly2), color, -1)
        cv2.putText(frame_bgr, label, (x1 + 2, ly2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame_bgr


def tracks_from_detections(detections) -> list[dict[str, Any]]:
    """Convert an sv.Detections (with tracker_id) into the JSONL track schema.

    Kept here so the worker can hand the same dicts to draw_tracks and the
    JSONL writer without duplicating the conversion.
    """
    out: list[dict[str, Any]] = []
    if detections.tracker_id is None:
        return out
    for i, tid in enumerate(detections.tracker_id):
        if tid is None:
            continue
        x1, y1, x2, y2 = detections.xyxy[i].tolist()
        conf = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else 0.0
        )
        out.append({
            "id": int(tid),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "conf": conf,
            "class": "person",
        })
    return out
