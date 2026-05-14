"""Compute role classification + zone dwell records.

Reads tracks + zones, applies the camera-default-role override from
config.toml where set. No CLI flags — fully config-driven.

Usage:
    python -m scripts.run_kpis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils.cameras import CAMERAS, DATA_DIR, camera_by_id  # noqa: E402
from app.config import CONFIG  # noqa: E402
from app.kpis.dwell import compute_dwell, write_dwell_json  # noqa: E402
from app.kpis.roles import (  # noqa: E402
    CAMERA_DEFAULT_ROLE,
    classify_tracks,
    write_roles_json,
)
from app.kpis.zones import load_camera_zones, split_zones  # noqa: E402

TRACKS_DIR = DATA_DIR / "tracks"
KPIS_DIR = DATA_DIR / "kpis"


def _load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _process(cam_id: str) -> None:
    cam = camera_by_id(cam_id)
    tracks_path = TRACKS_DIR / f"{cam.cam_id}.jsonl"
    if not tracks_path.exists():
        print(f"[{cam.cam_id}] skip: missing {tracks_path}")
        return

    default_role = CAMERA_DEFAULT_ROLE.get(cam.cam_id)
    zones = load_camera_zones(cam.cam_id)
    if not zones and default_role is None:
        print(f"[{cam.cam_id}] skip: no zones and no default role")
        return
    worker_z, customer_z = split_zones(zones)
    if default_role:
        print(f"[{cam.cam_id}] default_role={default_role!r} (zones ignored)")
    else:
        print(f"[{cam.cam_id}] zones: worker={[z.name for z in worker_z]}  "
              f"customer={[z.name for z in customer_z]}")

    records = _load_records(tracks_path)
    if not records:
        print(f"[{cam.cam_id}] skip: empty tracks file")
        return
    fps = 10.0
    for a, b in zip(records, records[1:]):
        dt = b["ts_s"] - a["ts_s"]
        if dt > 0:
            fps = 1.0 / dt
            break
    print(f"[{cam.cam_id}] {len(records)} frames @ {fps:.2f} fps")

    roles = classify_tracks(
        records, worker_z, customer_z, default_role=default_role
    )
    write_roles_json(KPIS_DIR / f"{cam.cam_id}.roles.json", cam.cam_id, roles)
    counts = {"worker": 0, "customer": 0, "unknown": 0}
    for r in roles.values():
        counts[r["role"]] += 1
    print(f"[{cam.cam_id}] roles: {counts}")

    if not customer_z:
        write_dwell_json(KPIS_DIR / f"{cam.cam_id}.dwell.json", cam.cam_id, [])
        print(f"[{cam.cam_id}] dwell skipped (no customer zones)")
        return

    dwell = compute_dwell(records, customer_z, fps=fps, roles=roles)
    write_dwell_json(KPIS_DIR / f"{cam.cam_id}.dwell.json", cam.cam_id, dwell)
    print(f"[{cam.cam_id}] dwell visits emitted: {len(dwell)}")
    if dwell:
        per_zone: dict[str, list[float]] = {}
        for d in dwell:
            per_zone.setdefault(d["zone"], []).append(d["duration_s"])
        for zname, durs in per_zone.items():
            print(f"  {zname:>20}: n={len(durs):>3}  "
                  f"mean={sum(durs) / len(durs):.1f}s  max={max(durs):.1f}s")


def main() -> None:
    KPIS_DIR.mkdir(parents=True, exist_ok=True)
    cam_ids = CONFIG.pipeline.cameras or [c.cam_id for c in CAMERAS]
    print(f"running on  {cam_ids}")
    for cid in cam_ids:
        _process(cid)


if __name__ == "__main__":
    main()
