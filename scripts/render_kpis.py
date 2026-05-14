"""Render an annotated MP4 showing zones, role-colored tracks, and live dwell timers.

Inputs:
    data/tracks/<CAM>.jsonl         (per-frame tracks)
    data/zones.json                 (saved polygons)
    data/kpis/<CAM>.roles.json      (worker/customer per track-id)

Output:
    data/annotated/<CAM>.kpis.mp4

Usage:
    python -m scripts.render_kpis --cam CAM-01
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils.cameras import DATA_DIR, camera_by_id  # noqa: E402
from app.kpis.zones import (  # noqa: E402
    foot_point,
    load_camera_zones,
    point_in_polygon,
    split_zones,
)

TRACKS_DIR = DATA_DIR / "tracks"
ANNOTATED_DIR = DATA_DIR / "annotated"
KPIS_DIR = DATA_DIR / "kpis"

ROLE_COLORS_BGR = {
    "worker": (0, 165, 255),    # orange
    "customer": (255, 200, 80),  # light blue
    "unknown": (180, 180, 180),  # grey
}
WORKER_ZONE_COLOR = (0, 140, 220)
CUSTOMER_ZONE_COLOR = (220, 160, 70)
ZONE_FILL_ALPHA = 0.15
HUD_BG = (24, 28, 35)
HUD_FG = (235, 240, 245)


def _draw_zones(frame: np.ndarray, zones) -> None:
    """Filled translucent polygons + outline + name label."""
    overlay = frame.copy()
    for z in zones:
        color = WORKER_ZONE_COLOR if z.is_worker else CUSTOMER_ZONE_COLOR
        cv2.fillPoly(overlay, [z.polygon], color)
    cv2.addWeighted(overlay, ZONE_FILL_ALPHA, frame, 1 - ZONE_FILL_ALPHA, 0, frame)
    for z in zones:
        color = WORKER_ZONE_COLOR if z.is_worker else CUSTOMER_ZONE_COLOR
        cv2.polylines(frame, [z.polygon], isClosed=True, color=color, thickness=2)
        # Label at centroid.
        cx = int(np.mean(z.polygon[:, 0]))
        cy = int(np.mean(z.polygon[:, 1]))
        label = z.name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 6),
                      (cx + tw // 2 + 4, cy + 2), HUD_BG, -1)
        cv2.putText(frame, label, (cx - tw // 2, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_FG, 1, cv2.LINE_AA)


def _draw_track(
    frame: np.ndarray,
    bbox: list[float],
    tid: int,
    role: str,
    dwell_s: float | None,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    color = ROLE_COLORS_BGR.get(role, ROLE_COLORS_BGR["unknown"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    parts = [f"{role}:{tid}"]
    if dwell_s is not None and dwell_s >= 1.0:
        parts.append(f"{dwell_s:.0f}s")
    label = "  ".join(parts)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly2 = max(y1, th + 4)
    cv2.rectangle(frame, (x1, ly2 - th - 4),
                  (x1 + tw + 4, ly2), color, -1)
    cv2.putText(frame, label, (x1 + 2, ly2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_hud(
    frame: np.ndarray,
    cam_id: str,
    ts_s: float,
    zone_counts: dict[str, dict[str, int]],
) -> None:
    """Top-left panel: time + per-customer-zone live counts."""
    h, w = frame.shape[:2]
    lines = [f"{cam_id}   t={ts_s:6.1f}s"]
    for zname, counts in zone_counts.items():
        c = counts["customer"]
        wk = counts["worker"]
        lines.append(f"  {zname}: customers={c}  workers={wk}")

    line_h = 22
    pad = 8
    box_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                for l in lines) + pad * 2
    box_h = line_h * len(lines) + pad * 2
    cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), HUD_BG, -1)
    cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), (60, 70, 80), 1)
    for i, line in enumerate(lines):
        y = 8 + pad + (i + 1) * line_h - 6
        cv2.putText(frame, line, (8 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_FG, 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render zones + roles + dwell.")
    p.add_argument("--cam", required=True, help="Camera ID, e.g. CAM-01.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cam = camera_by_id(args.cam)

    zones = load_camera_zones(cam.cam_id)
    if not zones:
        raise SystemExit(f"no zones for {cam.cam_id}")
    _worker_z, customer_z = split_zones(zones)

    roles_path = KPIS_DIR / f"{cam.cam_id}.roles.json"
    if not roles_path.exists():
        raise SystemExit(f"missing {roles_path} — run scripts.run_kpis first")
    roles = {int(k): v for k, v in json.loads(roles_path.read_text())["tracks"].items()}

    tracks_path = TRACKS_DIR / f"{cam.cam_id}.jsonl"
    if not tracks_path.exists():
        raise SystemExit(f"missing {tracks_path}")
    by_frame: dict[int, list[dict]] = {}
    with tracks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_frame[int(rec["frame_idx"])] = rec.get("tracks", [])

    cap = cv2.VideoCapture(str(cam.path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {cam.path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    out_path = Path(args.out) if args.out else (
        ANNOTATED_DIR / f"{cam.cam_id}.kpis.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer: {out_path}")

    # Live dwell state per (track_id, customer_zone_name) → enter_frame.
    open_enter: dict[tuple[int, str], int] = {}

    print(f"rendering {out_path}")
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            _draw_zones(frame, zones)

            ts_s = frame_idx / fps
            tracks = by_frame.get(frame_idx, [])

            # Recompute per-track in-zone state + update live dwell counters.
            zone_counts = {z.name: {"customer": 0, "worker": 0}
                           for z in customer_z}

            # Mark which (tid, zone) pairs are present this frame.
            present_pairs: set[tuple[int, str]] = set()
            track_dwell: dict[int, float] = {}

            for t in tracks:
                tid = int(t["id"])
                role = roles.get(tid, {}).get("role", "unknown")
                fx, fy = foot_point(t["bbox"])
                for z in customer_z:
                    if point_in_polygon(fx, fy, z.polygon):
                        zone_counts[z.name][role if role in zone_counts[z.name] else "customer"] += 1
                        key = (tid, z.name)
                        present_pairs.add(key)
                        if key not in open_enter:
                            open_enter[key] = frame_idx
                        elapsed = (frame_idx - open_enter[key]) / fps
                        # Track's displayed dwell: max across zones currently in.
                        track_dwell[tid] = max(track_dwell.get(tid, 0.0), elapsed)

            # Close pairs absent this frame (1 s tolerance handled in batch job;
            # here we use a quicker reset for live display).
            for key in list(open_enter.keys()):
                if key not in present_pairs:
                    # if missing for >1s, drop the open visit
                    if (frame_idx - open_enter[key]) / fps > 0:
                        # we only update display dwell while present; close cleanly
                        # by removing the stale key after a small grace.
                        pass
                    # simple reset when track left
                    del open_enter[key]

            for t in tracks:
                tid = int(t["id"])
                role = roles.get(tid, {}).get("role", "unknown")
                _draw_track(frame, t["bbox"], tid, role, track_dwell.get(tid))

            _draw_hud(frame, cam.cam_id, ts_s, zone_counts)
            writer.write(frame)
            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"  frame {frame_idx}")
    finally:
        cap.release()
        writer.release()
    print(f"done  {frame_idx} frames -> {out_path}")


if __name__ == "__main__":
    main()
