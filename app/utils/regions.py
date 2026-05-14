"""Per-camera canonical region registry.

The annotation tab offers exactly this list per camera — no free-text names.
Region names here are the only ones recognised as worker/customer zones by
the downstream KPI pipeline. Cameras with an empty list are not annotated
at all (CAM-03 dining → implicit-customer-only; CAM-04 kitchen →
implicit-worker-only via `kpis.camera_default_role` in config.toml).

Names that start with one of `[kpis] worker_zone_prefixes` are worker zones;
everything else is a customer zone (see app/kpis/zones.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str       # canonical key stored in data/zones.json
    label: str      # human-readable label shown in the dropdown
    is_worker: bool # for the UI color hint only


CAMERA_REGIONS: dict[str, list[Region]] = {
    "CAM-01": [
        Region("workers_area",     "Workers Area",     True),
        Region("customer_counter", "Customer Counter", False),
        Region("customer_queue",   "Customer Queue",   False),
    ],
    "CAM-02": [
        Region("workers_area",     "Workers Area",     True),
        Region("customer_counter", "Customer Counter", False),
        Region("customer_queue",   "Customer Queue",   False),
    ],
    # No annotation needed — full-camera implicit role from camera_default_role.
    "CAM-03": [],
    "CAM-04": [],
}


def regions_for(cam_id: str) -> list[Region]:
    return CAMERA_REGIONS.get(cam_id, [])
