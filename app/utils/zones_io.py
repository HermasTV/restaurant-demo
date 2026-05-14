"""Polygon-zone persistence.

Schema:
    {
      "CAM-01": [
        {"name": "queue_polygon",
         "points": [[x1, y1], [x2, y2], ...],
         "frame_size": [width, height]}
      ],
      ...
    }
Points are stored in the *original frame* coordinate space (not the canvas).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.utils.cameras import DATA_DIR

ZONES_FILE = DATA_DIR / "zones.json"


def _load_raw() -> dict[str, list[dict[str, Any]]]:
    if not ZONES_FILE.exists():
        return {}
    try:
        return json.loads(ZONES_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def load_zones(cam_id: str) -> list[dict[str, Any]]:
    return _load_raw().get(cam_id, [])


def save_zone(cam_id: str, name: str, points: list[list[float]],
              frame_size: tuple[int, int]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_raw()
    cam_zones = data.setdefault(cam_id, [])
    # replace existing zone with same name
    cam_zones = [z for z in cam_zones if z["name"] != name]
    cam_zones.append({
        "name": name,
        "points": [[float(x), float(y)] for x, y in points],
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
    })
    data[cam_id] = cam_zones
    ZONES_FILE.write_text(json.dumps(data, indent=2))


def delete_zone(cam_id: str, name: str) -> None:
    data = _load_raw()
    if cam_id not in data:
        return
    data[cam_id] = [z for z in data[cam_id] if z["name"] != name]
    ZONES_FILE.write_text(json.dumps(data, indent=2))
