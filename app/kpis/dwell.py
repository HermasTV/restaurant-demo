"""Per-track per-customer-zone dwell timer.

For each (track_id, customer_zone), maintain enter/last-in state across frames.
Close a visit when the track has been *out* of the zone for > tolerance frames,
or when the stream ends. Emit only visits whose total in-zone time >= MIN_DWELL_S.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import CONFIG
from app.kpis.zones import Zone, foot_point, point_in_polygon

# Sourced from [kpis] in config.toml.
MIN_DWELL_S = CONFIG.kpis.min_dwell_s
REENTER_TOLERANCE_S = CONFIG.kpis.reenter_tolerance_s


@dataclass
class _OpenVisit:
    track_id: int
    zone: str
    enter_frame: int
    enter_ts_s: float
    last_in_frame: int
    last_in_ts_s: float
    in_frames: int = 0  # count of frames actually inside zone (not just span)


def compute_dwell(
    records: list[dict[str, Any]],
    customer_zones: list[Zone],
    fps: float,
    roles: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not customer_zones:
        return []
    tolerance_frames = max(1, int(round(REENTER_TOLERANCE_S * fps)))

    # (track_id, zone_name) -> _OpenVisit
    open_visits: dict[tuple[int, str], _OpenVisit] = {}
    closed: list[dict[str, Any]] = []

    def _close(visit: _OpenVisit) -> None:
        in_seconds = visit.in_frames / fps
        if in_seconds < MIN_DWELL_S:
            return
        role = (roles or {}).get(visit.track_id, {}).get("role", "unknown")
        closed.append({
            "track_id": visit.track_id,
            "role": role,
            "zone": visit.zone,
            "enter_frame": visit.enter_frame,
            "exit_frame": visit.last_in_frame,
            "enter_ts_s": round(visit.enter_ts_s, 3),
            "exit_ts_s": round(visit.last_in_ts_s, 3),
            "duration_s": round(in_seconds, 2),
        })

    for rec in records:
        frame_idx = int(rec["frame_idx"])
        ts_s = float(rec["ts_s"])

        # First flush any open visits that have gone cold (track-and-zone pair
        # not refreshed within tolerance).
        for key in list(open_visits.keys()):
            v = open_visits[key]
            if frame_idx - v.last_in_frame > tolerance_frames:
                _close(v)
                del open_visits[key]

        # Build per-track set of zones it's currently in.
        track_zone_hits: dict[int, set[str]] = {}
        for t in rec.get("tracks", []):
            tid = int(t["id"])
            fx, fy = foot_point(t["bbox"])
            zones_now = {z.name for z in customer_zones
                         if point_in_polygon(fx, fy, z.polygon)}
            if zones_now:
                track_zone_hits[tid] = zones_now

        # Update / open visits.
        for tid, zones_now in track_zone_hits.items():
            for zname in zones_now:
                key = (tid, zname)
                v = open_visits.get(key)
                if v is None:
                    open_visits[key] = _OpenVisit(
                        track_id=tid, zone=zname,
                        enter_frame=frame_idx, enter_ts_s=ts_s,
                        last_in_frame=frame_idx, last_in_ts_s=ts_s,
                        in_frames=1,
                    )
                else:
                    v.last_in_frame = frame_idx
                    v.last_in_ts_s = ts_s
                    v.in_frames += 1

    # End of stream: flush remaining.
    for v in open_visits.values():
        _close(v)

    closed.sort(key=lambda r: (r["enter_frame"], r["track_id"]))
    return closed


def write_dwell_json(path: Path, cam_id: str, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        durations = sorted(r["duration_s"] for r in records)
        summary = {
            "count": len(records),
            "mean_s": round(sum(durations) / len(durations), 2),
            "median_s": durations[len(durations) // 2],
            "p95_s": durations[max(0, int(round(0.95 * len(durations))) - 1)],
            "max_s": durations[-1],
        }
    else:
        summary = {"count": 0}
    payload = {"cam_id": cam_id, "summary": summary, "records": records}
    path.write_text(json.dumps(payload, indent=2))
