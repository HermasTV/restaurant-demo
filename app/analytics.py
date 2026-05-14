"""Analytics tab — simulated live processing of 4 streams.

Each tile replays the source video and overlays detector/tracker output
(read from the pre-computed JSONL) frame-by-frame, while a live stats panel
tracks current zone occupancy, completed dwell visits, etc.

This is *simulated* live: detection ran offline once into data/tracks/*.jsonl;
the playback loop here re-reads those records in sync with the video stream,
which is fast enough for an interactive demo without re-running the GPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from app.utils.cameras import CAMERAS, Camera, DATA_DIR
from app.kpis.zones import (
    Zone,
    foot_point,
    load_camera_zones,
    point_in_polygon,
    split_zones,
)

TRACKS_DIR = DATA_DIR / "tracks"
KPIS_DIR = DATA_DIR / "kpis"

# Display
TILE_WIDTH = 640                # downscale frame width before sending to browser
DISPLAY_FPS_BASE = 5            # target ticks/sec at 1x speed
JPEG_QUALITY = 75               # quality for the in-browser preview

# Overlay colors (BGR — OpenCV native)
ROLE_COLORS_BGR = {
    "worker":   (0, 165, 255),
    "customer": (255, 200, 80),
    "unknown":  (180, 180, 180),
}
WORKER_ZONE_COLOR = (0, 140, 220)
CUSTOMER_ZONE_COLOR = (220, 160, 70)
ZONE_FILL_ALPHA = 0.15
HUD_BG = (24, 28, 35)
HUD_FG = (235, 240, 245)

MIN_DWELL_S = 3.0  # match app/kpis/dwell.py
DWELL_TOLERANCE_S = 1.0


# ---------------------------------------------------------------------------
# Cached IO
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_tracks_by_frame(cam_id: str) -> dict[int, dict]:
    path = TRACKS_DIR / f"{cam_id}.jsonl"
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["frame_idx"])] = rec
    return out


@st.cache_data(show_spinner=False)
def _load_roles(cam_id: str) -> dict[int, dict]:
    path = KPIS_DIR / f"{cam_id}.roles.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw["tracks"].items()}


@st.cache_resource(show_spinner=False)
def _load_zones_cached(cam_id: str) -> list[Zone]:
    return load_camera_zones(cam_id)


# ---------------------------------------------------------------------------
# Drawing helpers (BGR)
# ---------------------------------------------------------------------------

def _draw_zones(frame: np.ndarray, zones: list[Zone]) -> None:
    if not zones:
        return
    overlay = frame.copy()
    for z in zones:
        color = WORKER_ZONE_COLOR if z.is_worker else CUSTOMER_ZONE_COLOR
        cv2.fillPoly(overlay, [z.polygon], color)
    cv2.addWeighted(overlay, ZONE_FILL_ALPHA, frame, 1 - ZONE_FILL_ALPHA, 0, frame)
    for z in zones:
        color = WORKER_ZONE_COLOR if z.is_worker else CUSTOMER_ZONE_COLOR
        cv2.polylines(frame, [z.polygon], True, color, 2)


def _draw_track(
    frame: np.ndarray, t: dict, role: str, dwell_s: float | None
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in t["bbox"])
    color = ROLE_COLORS_BGR.get(role, ROLE_COLORS_BGR["unknown"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    parts = [f"{role}:{int(t['id'])}"]
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
    workers_now: int,
    customers_now: int,
) -> None:
    line1 = f"{cam_id}  t={ts_s:6.1f}s"
    line2 = f"workers:{workers_now}  customers:{customers_now}"
    box_w = 220
    box_h = 50
    cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), HUD_BG, -1)
    cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), (60, 70, 80), 1)
    cv2.putText(frame, line1, (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_FG, 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (16, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_FG, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Per-camera simulator
# ---------------------------------------------------------------------------

class CameraSim:
    """Plays back a single camera frame-by-frame, drawing offline detections
    on each frame and maintaining a small live state machine for dwell timers.
    """

    def __init__(self, cam: Camera) -> None:
        self.cam = cam
        self.tracks_by_frame = _load_tracks_by_frame(cam.cam_id)
        self.roles = _load_roles(cam.cam_id)
        self.zones = _load_zones_cached(cam.cam_id)
        self.worker_zones, self.customer_zones = split_zones(self.zones)
        self.cap = cv2.VideoCapture(str(cam.path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 10.0
        self.frame_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # Live state
        self.dwell_open: dict[tuple[int, str], int] = {}  # (tid, zone) -> enter_frame_idx
        self.dwell_last_seen: dict[tuple[int, str], int] = {}
        self.completed_dwells: list[dict] = []  # {track_id, zone, duration_s, role}
        self.frame_idx = 0

    def reset(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.dwell_open.clear()
        self.dwell_last_seen.clear()
        self.completed_dwells.clear()
        self.frame_idx = 0

    def _close_visit(self, key: tuple[int, str]) -> None:
        enter = self.dwell_open[key]
        duration = (self.dwell_last_seen[key] - enter) / self.fps
        if duration >= MIN_DWELL_S:
            tid, zname = key
            self.completed_dwells.append({
                "track_id": tid,
                "zone": zname,
                "duration_s": round(duration, 1),
                "role": self.roles.get(tid, {}).get("role", "unknown"),
            })
        del self.dwell_open[key]
        del self.dwell_last_seen[key]

    def next(self) -> tuple[np.ndarray | None, dict]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.reset()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None, {}

        idx = self.frame_idx
        ts_s = idx / self.fps
        rec = self.tracks_by_frame.get(idx, {})
        tracks = rec.get("tracks", [])

        _draw_zones(frame, self.zones)

        # Per-frame dwell update
        zone_counts = {z.name: {"worker": 0, "customer": 0}
                       for z in self.customer_zones}
        track_dwell_now: dict[int, float] = {}
        present_pairs: set[tuple[int, str]] = set()
        tolerance_frames = max(1, int(round(DWELL_TOLERANCE_S * self.fps)))

        # Close any open visit that exceeded tolerance.
        for key in list(self.dwell_open.keys()):
            if idx - self.dwell_last_seen[key] > tolerance_frames:
                self._close_visit(key)

        workers_now = customers_now = 0
        for t in tracks:
            tid = int(t["id"])
            role = self.roles.get(tid, {}).get("role", "unknown")
            if role == "worker":
                workers_now += 1
            elif role == "customer":
                customers_now += 1
            fx, fy = foot_point(t["bbox"])
            for z in self.customer_zones:
                if point_in_polygon(fx, fy, z.polygon):
                    key = (tid, z.name)
                    present_pairs.add(key)
                    if key not in self.dwell_open:
                        self.dwell_open[key] = idx
                    self.dwell_last_seen[key] = idx
                    elapsed = (idx - self.dwell_open[key]) / self.fps
                    track_dwell_now[tid] = max(
                        track_dwell_now.get(tid, 0.0), elapsed
                    )
                    if role in ("worker", "customer"):
                        zone_counts[z.name][role] += 1

        # Draw boxes + labels
        for t in tracks:
            tid = int(t["id"])
            role = self.roles.get(tid, {}).get("role", "unknown")
            _draw_track(frame, t, role, track_dwell_now.get(tid))

        _draw_hud(frame, self.cam.cam_id, ts_s, workers_now, customers_now)

        longest_dwell = max(
            (d["duration_s"] for d in self.completed_dwells), default=0.0
        )
        stats = {
            "frame_idx": idx,
            "ts_s": ts_s,
            "tracks_now": len(tracks),
            "workers_now": workers_now,
            "customers_now": customers_now,
            "zone_counts": zone_counts,
            "open_dwells": len(self.dwell_open),
            "completed_dwells": len(self.completed_dwells),
            "longest_dwell_s": longest_dwell,
            "has_roles": bool(self.roles),
            "has_zones": bool(self.zones),
        }
        self.frame_idx += 1
        return frame, stats


# ---------------------------------------------------------------------------
# Stats panel HTML
# ---------------------------------------------------------------------------

def _stats_html(cam: Camera, stats: dict) -> str:
    if not stats:
        return f'<div class="ana-stats"><i>no data for {cam.cam_id}</i></div>'

    rows: list[str] = []
    rows.append(
        f'<div class="ana-row ana-head">'
        f'<span>{cam.cam_id} · {cam.name}</span>'
        f'<span class="ana-time">t={stats["ts_s"]:6.1f}s</span></div>'
    )
    rows.append(_row("People in view", stats["tracks_now"]))
    if stats.get("has_roles"):
        rows.append(_row("&nbsp;&nbsp;Workers", stats["workers_now"], "worker"))
        rows.append(_row("&nbsp;&nbsp;Customers", stats["customers_now"], "customer"))
    for zname, counts in (stats.get("zone_counts") or {}).items():
        rows.append(_row(
            f"In <code>{zname}</code>",
            f"{counts['customer']} cust · {counts['worker']} wkr",
        ))
    if stats.get("has_zones"):
        rows.append(_row("Active dwell timers", stats["open_dwells"]))
        rows.append(_row("Completed dwell visits", stats["completed_dwells"]))
        rows.append(_row(
            "Longest dwell (s)",
            f"{stats['longest_dwell_s']:.1f}",
        ))
    return f'<div class="ana-stats">{"".join(rows)}</div>'


def _row(label: str, value, klass: str = "") -> str:
    cls = f' class="ana-{klass}"' if klass else ""
    return (
        f'<div class="ana-row"><span>{label}</span>'
        f'<span{cls}>{value}</span></div>'
    )


# ---------------------------------------------------------------------------
# Tab entry
# ---------------------------------------------------------------------------

_STYLES = """
<style>
.ana-stats {
  background: #161b22;
  border: 1px solid #2a313c;
  border-radius: 6px;
  padding: 8px 10px;
  margin-top: 6px;
  font-size: 12px;
  color: #c9d1d9;
}
.ana-row {
  display: flex; justify-content: space-between;
  padding: 2px 0; border-bottom: 1px dashed #21262d;
}
.ana-row:last-child { border-bottom: none; }
.ana-row.ana-head { font-weight: 600; color: #e6edf3; font-size: 13px; border-bottom: 1px solid #30363d; margin-bottom: 4px; padding-bottom: 4px; }
.ana-time { color: #8b949e; font-variant-numeric: tabular-nums; }
.ana-worker { color: #ffb86b; }
.ana-customer { color: #58a6ff; }
.ana-stats code { background:#0d1117; padding:1px 4px; border-radius:3px; color:#8b949e; }
</style>
"""


def _get_sim(cam: Camera) -> CameraSim:
    key = f"ana_sim_{cam.cam_id}"
    if key not in st.session_state:
        st.session_state[key] = CameraSim(cam)
    return st.session_state[key]


def _reset_sims() -> None:
    for cam in CAMERAS:
        key = f"ana_sim_{cam.cam_id}"
        if key in st.session_state:
            st.session_state[key].reset()


def render() -> None:
    st.markdown(_STYLES, unsafe_allow_html=True)
    st.markdown("#### Analytics — simulated live processing")
    st.caption(
        "Each tile replays its source video while drawing person tracks, "
        "role tags (worker / customer), zone overlays and dwell timers in "
        "real time. Detections are loaded from the offline pipeline output."
    )

    if "ana_playing" not in st.session_state:
        st.session_state.ana_playing = False
    if "ana_speed" not in st.session_state:
        st.session_state.ana_speed = "1×"

    c1, c2, c3, _ = st.columns([1, 1, 1, 5])
    play_label = "⏸ Pause" if st.session_state.ana_playing else "▶ Play"
    if c1.button(play_label, use_container_width=True, key="ana_play_btn"):
        st.session_state.ana_playing = not st.session_state.ana_playing
        st.rerun()
    if c2.button("↺ Restart", use_container_width=True, key="ana_restart_btn"):
        _reset_sims()
        st.session_state.ana_playing = False
        st.rerun()
    st.session_state.ana_speed = c3.selectbox(
        "Speed", ["0.5×", "1×", "2×", "4×"],
        index=["0.5×", "1×", "2×", "4×"].index(st.session_state.ana_speed),
        label_visibility="collapsed",
        key="ana_speed_select",
    )
    speed_factor = float(st.session_state.ana_speed.rstrip("×"))

    # Pre-allocate the 2x2 grid of placeholders.
    rows = [st.columns(2, gap="small"), st.columns(2, gap="small")]
    slots = [rows[0][0], rows[0][1], rows[1][0], rows[1][1]]
    frame_phs: list = []
    stats_phs: list = []
    for slot, cam in zip(slots, CAMERAS):
        with slot:
            st.markdown(
                f'<div class="tile-header">'
                f'<span class="cam-name">{cam.name}</span>'
                f'<span class="cam-tag">{cam.cam_id} · {cam.tag}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            frame_phs.append(st.empty())
            stats_phs.append(st.empty())

    sims = [_get_sim(cam) for cam in CAMERAS]

    def _tick() -> None:
        for sim, fph, sph in zip(sims, frame_phs, stats_phs):
            frame, stats = sim.next()
            if frame is None:
                continue
            if frame.shape[1] != TILE_WIDTH:
                scale = TILE_WIDTH / frame.shape[1]
                new_h = int(round(frame.shape[0] * scale))
                frame = cv2.resize(frame, (TILE_WIDTH, new_h),
                                   interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fph.image(rgb, use_container_width=True)
            sph.markdown(_stats_html(sim.cam, stats), unsafe_allow_html=True)

    if not st.session_state.ana_playing:
        _tick()
        st.caption("Paused. Click ▶ Play to start the live simulation.")
        return

    # Live loop. Streamlit interrupts this when a widget is clicked
    # (Pause / Restart / Speed) — the next placeholder.image() call raises
    # RerunException and we fall through to a rerun.
    target_dt = 1.0 / (DISPLAY_FPS_BASE * speed_factor)
    try:
        while st.session_state.ana_playing:
            t0 = time.monotonic()
            _tick()
            slept = time.monotonic() - t0
            remaining = target_dt - slept
            if remaining > 0:
                time.sleep(remaining)
    except Exception:
        # Any Streamlit-internal rerun signal: let the next run pick up.
        raise
