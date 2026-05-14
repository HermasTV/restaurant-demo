"""Per-track role classifier (worker / customer) using zone-based prior.

For each track, count frames where the bbox center sits inside any worker
zone vs any customer zone. A track is a worker only if its worker-zone
count strictly exceeds its customer-zone count; everything else
— including tracks that never intersected any polygon (background foot
traffic) and tracks too short to classify — is treated as a customer.
There is no "unknown" output: the predefined regions cover the meaningful
frame area so the partition is total.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.config import CONFIG
from app.kpis.zones import Zone, center_point, point_in_polygon

# A track must have at least this many frames before we trust its role label.
MIN_FRAMES_FOR_CLASSIFICATION = 5
# How decisively the majority must win. 0.5 = simple majority.
WORKER_FRAME_RATIO = 0.5

# Per-camera default-role override. When set, every tracked person on that
# camera is forced to this role regardless of zones — config-driven.
CAMERA_DEFAULT_ROLE: dict[str, str] = CONFIG.kpis.camera_default_role


def classify_tracks(
    records: list[dict[str, Any]],
    worker_zones: list[Zone],
    customer_zones: list[Zone],
    default_role: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Return {track_id: {"role": "worker"|"customer", ...stats}}.

    If `default_role` is set, every track is assigned that role and zone
    geometry is ignored — for cameras that are entirely BOH or FOH.
    """
    in_worker: dict[int, int] = defaultdict(int)
    in_customer: dict[int, int] = defaultdict(int)
    total: dict[int, int] = defaultdict(int)

    if default_role is not None:
        for rec in records:
            for t in rec.get("tracks", []):
                total[int(t["id"])] += 1
        return {
            tid: {
                "role": default_role,
                "frames_total": frames,
                "frames_in_worker": 0,
                "frames_in_customer": 0,
                "source": "camera_default",
            }
            for tid, frames in total.items()
        }

    # Role classification uses bbox CENTER (not BOTTOM_CENTER) so workers
    # whose legs are occluded by a counter still register inside the worker
    # zone via their visible torso.
    for rec in records:
        for t in rec.get("tracks", []):
            tid = int(t["id"])
            total[tid] += 1
            cx, cy = center_point(t["bbox"])
            if any(point_in_polygon(cx, cy, z.polygon) for z in worker_zones):
                in_worker[tid] += 1
            elif any(point_in_polygon(cx, cy, z.polygon) for z in customer_zones):
                in_customer[tid] += 1

    out: dict[int, dict[str, Any]] = {}
    for tid, frames in total.items():
        # Too-short tracks haven't accumulated enough samples to be confidently
        # called workers — default them to customer.
        if frames < MIN_FRAMES_FOR_CLASSIFICATION:
            role = "customer"
        else:
            worker_ratio = in_worker[tid] / frames
            role = "worker" if worker_ratio >= WORKER_FRAME_RATIO else "customer"
        out[tid] = {
            "role": role,
            "frames_total": frames,
            "frames_in_worker": in_worker[tid],
            "frames_in_customer": in_customer[tid],
        }
    return out


def write_roles_json(path: Path, cam_id: str, roles: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cam_id": cam_id,
        "tracks": {str(k): v for k, v in sorted(roles.items())},
    }
    path.write_text(json.dumps(payload, indent=2))
