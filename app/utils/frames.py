"""Extract a representative still frame from a video.

Uses OpenCV to grab the middle frame — usually more informative than frame 0
(which can be black or pre-roll).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.utils.cameras import DATA_DIR, Camera

REFERENCE_DIR = DATA_DIR / "reference_frames"


def grab_frame(video_path: Path, position: float = 0.5) -> Image.Image:
    """Return a PIL RGB frame at the given normalized position [0, 1]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        target = max(0, min(total - 1, int(total * position)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read frame from {video_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def reference_frame_path(cam: Camera) -> Path:
    return REFERENCE_DIR / f"{cam.cam_id}.png"


def get_reference_frame(cam: Camera) -> Image.Image:
    """Load the saved reference frame; extract & cache to disk if missing."""
    path = reference_frame_path(cam)
    if not path.exists():
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        img = grab_frame(cam.path)
        img.save(path, format="PNG")
        return img
    return Image.open(path).convert("RGB")


def fit_to_width(img: Image.Image, max_width: int) -> tuple[Image.Image, float]:
    """Resize image so its width <= max_width. Returns (resized, scale)
    where original_xy = canvas_xy / scale."""
    if img.width <= max_width:
        return img, 1.0
    scale = max_width / img.width
    new_size = (max_width, int(round(img.height * scale)))
    return img.resize(new_size, Image.LANCZOS), scale
