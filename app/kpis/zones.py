"""Zone loading + worker/customer categorization by polygon name.

A zone is classified as a *worker* zone if its name starts with any of
`config.kpis.worker_zone_prefixes`; everything else is treated as a
*customer* zone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.config import CONFIG
from app.utils.zones_io import load_zones


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: np.ndarray  # (N, 2) int32, in original frame coords
    is_worker: bool

    @property
    def role_label(self) -> str:
        return "worker" if self.is_worker else "customer"


def _is_worker_zone(name: str) -> bool:
    return any(name.startswith(p) for p in CONFIG.kpis.worker_zone_prefixes)


def load_camera_zones(cam_id: str) -> list[Zone]:
    out: list[Zone] = []
    for z in load_zones(cam_id):
        poly = np.array(z["points"], dtype=np.int32)
        out.append(Zone(name=z["name"], polygon=poly,
                        is_worker=_is_worker_zone(z["name"])))
    return out


def split_zones(zones: Iterable[Zone]) -> tuple[list[Zone], list[Zone]]:
    """Return (worker_zones, customer_zones)."""
    workers, customers = [], []
    for z in zones:
        (workers if z.is_worker else customers).append(z)
    return workers, customers


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Ray-casting; polygon is (N, 2) int32 in pixel coords."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def foot_point(bbox: list[float]) -> tuple[float, float]:
    """Bottom-center of an [x1, y1, x2, y2] bbox — matches sv.Position.BOTTOM_CENTER."""
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def center_point(bbox: list[float]) -> tuple[float, float]:
    """Bbox centroid. Use this instead of foot_point when legs are occluded
    (e.g. a worker standing behind a counter — bbox bottom = top of counter,
    not the floor, so BOTTOM_CENTER misclassifies them as outside the zone)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def zone_kind(zone_name: str) -> str:
    """'counter' | 'queue' | 'general'

    Used by the live KPI overlay to split customer counts into
    serving (at counter) vs waiting (in queue).
    """
    n = zone_name.lower()
    if "counter" in n:
        return "counter"
    if "queue" in n:
        return "queue"
    return "general"
