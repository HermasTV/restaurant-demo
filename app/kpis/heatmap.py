"""Occupancy heatmap (KPI 7).

Time-weighted per-pixel accumulator over a session: each frame, for each
tracked customer, add `1/fps` seconds to the grid cell at the foot point
(bbox bottom-center). Smoothed with a Gaussian and rendered with a JET
colormap overlaid on the reference frame.

Inputs:
    data/tracks/<CAM>.jsonl  (per-frame tracks)
    data/kpis/<CAM>.roles.json  (optional — to filter to customer role only)

Outputs (via scripts.render_heatmap):
    data/kpis/<CAM>.heatmap.npy       raw accumulator (height × width float32)
    data/kpis/<CAM>.heatmap.png       colormap render (no overlay)
    data/annotated/<CAM>.heatmap.png  colormap overlaid on the reference frame
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Match the rest of the KPI stack
GAUSSIAN_KSIZE = 51       # blur kernel — wider = smoother, less detail
OVERLAY_ALPHA = 0.55      # weight of heatmap colormap vs reference frame
MIN_DOT_RADIUS = 4        # rasterize each foot-point as a small filled circle
                          # so 10 fps samples don't leave sparse pixel-aliased dots


@dataclass
class HeatmapAccumulator:
    """Per-pixel residence-time accumulator for a single camera.

    `add_foot_point(x, y, dt_s)` deposits `dt_s` seconds into a small disc
    around the foot point. After processing all frames, `render()` produces
    a colourmap (and optionally an overlay on a reference frame).
    """

    width: int
    height: int
    dot_radius: int = MIN_DOT_RADIUS
    grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

    def add_foot_point(self, x: float, y: float, dt_s: float) -> None:
        cx = int(round(x))
        cy = int(round(y))
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return
        # Use cv2.circle on a float32 mask added in-place — fast and correct
        # at sub-pixel reproducibility because radius is in integer pixels.
        cv2.circle(
            self.grid, (cx, cy), self.dot_radius,
            color=float(dt_s), thickness=-1,
        )

    def render(
        self,
        gaussian_ksize: int = GAUSSIAN_KSIZE,
        colormap: int = cv2.COLORMAP_JET,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (smoothed grid in seconds, BGR colormap render)."""
        k = max(3, gaussian_ksize | 1)  # must be odd
        smoothed = cv2.GaussianBlur(self.grid, (k, k), sigmaX=0)
        if smoothed.max() > 0:
            normed = (smoothed / smoothed.max() * 255.0).astype(np.uint8)
        else:
            normed = np.zeros_like(smoothed, dtype=np.uint8)
        coloured = cv2.applyColorMap(normed, colormap)
        return smoothed, coloured

    def overlay_on(
        self,
        reference_bgr: np.ndarray,
        gaussian_ksize: int = GAUSSIAN_KSIZE,
        alpha: float = OVERLAY_ALPHA,
        colormap: int = cv2.COLORMAP_JET,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (smoothed grid in seconds, overlaid BGR image)."""
        smoothed, coloured = self.render(gaussian_ksize, colormap)
        # Use the smoothed grid as a per-pixel alpha mask so background stays
        # untouched where no one has stood.
        if smoothed.max() > 0:
            mask = smoothed / smoothed.max()  # [0..1]
        else:
            mask = np.zeros_like(smoothed)
        mask = (mask * alpha)[:, :, None]  # broadcast over BGR channels
        out = reference_bgr.astype(np.float32) * (1 - mask) + \
              coloured.astype(np.float32) * mask
        return smoothed, out.astype(np.uint8)


def foot_point(bbox: list[float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def build_from_jsonl(
    jsonl_path: Path,
    customer_ids: set[int] | None = None,
    fps: float = 10.0,
    frame_size: tuple[int, int] | None = None,
    dot_radius: int = MIN_DOT_RADIUS,
) -> HeatmapAccumulator:
    """Walk a tracks JSONL and accumulate foot-points.

    If `customer_ids` is provided, only those track IDs are accumulated
    (drop worker/unknown).
    """
    if frame_size is None:
        # Peek first record for frame_size.
        with jsonl_path.open("r", encoding="utf-8") as fh:
            first = json.loads(next(fh))
            frame_size = (int(first["frame_size"][0]), int(first["frame_size"][1]))

    accum = HeatmapAccumulator(
        width=frame_size[0], height=frame_size[1], dot_radius=dot_radius
    )
    dt = 1.0 / fps

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for t in rec.get("tracks", []):
                tid = int(t["id"])
                if customer_ids is not None and tid not in customer_ids:
                    continue
                fx, fy = foot_point(t["bbox"])
                accum.add_foot_point(fx, fy, dt)
    return accum
