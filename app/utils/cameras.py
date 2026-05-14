"""Camera registry shared across dashboard pages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "videos"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Camera:
    cam_id: str
    name: str
    tag: str
    filename: str
    kpis: tuple[str, ...]

    @property
    def path(self) -> Path:
        return VIDEO_DIR / self.filename


CAMERAS: tuple[Camera, ...] = (
    Camera("CAM-01", "Billing Area", "FOH · Cashier",
           "billing_area.mp4", ("Queue", "Drawer state", "Receipt/POS")),
    Camera("CAM-02", "Counter", "FOH · Order point",
           "counter.mp4", ("Queue", "Unattended", "Dwell")),
    Camera("CAM-03", "Dining", "FOH · Seating",
           "dining.mp4", ("Occupancy", "Heatmap", "Dwell", "Demographics")),
    Camera("CAM-04", "Kitchen", "BOH · Prep",
           "kitchen.mp4", ("Gloves", "Hairnet", "Handwash", "Hygiene")),
)


def camera_by_id(cam_id: str) -> Camera:
    for c in CAMERAS:
        if c.cam_id == cam_id:
            return c
    raise KeyError(cam_id)
