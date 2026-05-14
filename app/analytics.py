"""Analytics tab — simulated live processing of 4 streams.

Each tile replays the source video while re-running the **same** overlay
code that the live worker uses (`LiveKpiOverlay` + `draw_labels`) against
the offline JSONL. Roles, dwell timers, the unattended-customer state
machine, the HUD, and PPE labels are all rendered by the shared
primitives, so a tile must look identical to the saved annotated MP4
for the same camera. Demographics labels come from the saved sidecar
(`data/kpis/<CAM>.demographics.json`) via a tiny static shim that
exposes the same `label_for(tid)` API as the live aggregator.
"""
from __future__ import annotations

import json
import time
from typing import Any

import cv2
import numpy as np
import streamlit as st

from app.config import CONFIG
from app.utils.cameras import CAMERAS, Camera, DATA_DIR
from app.kpis.zones import load_camera_zones
from app.pipeline.draw import draw_labels, draw_tracks
from app.pipeline.live_kpis import LiveKpiOverlay

TRACKS_DIR = DATA_DIR / "tracks"
KPIS_DIR = DATA_DIR / "kpis"

TILE_WIDTH = 640        # downscale frame width before sending to browser
DISPLAY_FPS_BASE = 5    # target ticks/sec at 1× speed
JPEG_QUALITY = 75


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


class _StaticDemographics:
    """Read-only demographics shim built from the saved sidecar JSON.

    Exposes `label_for(tid)` so it slots into `LiveKpiOverlay` exactly
    where the live `DemographicsAggregator` would — no overlay changes
    needed."""

    def __init__(self, cam_id: str) -> None:
        path = KPIS_DIR / f"{cam_id}.demographics.json"
        self._labels: dict[int, str] = {}
        if not path.exists():
            return
        data = json.loads(path.read_text())
        for tid, rec in (data.get("tracks") or {}).items():
            age = rec.get("age_band")
            gender = rec.get("gender")
            if age and gender:
                self._labels[int(tid)] = f"{age}, {gender}"

    def label_for(self, tid: int) -> str | None:
        return self._labels.get(tid)


# ---------------------------------------------------------------------------
# Per-camera simulator
# ---------------------------------------------------------------------------

class CameraSim:
    """Plays back a single camera frame-by-frame using the same overlay
    pipeline as `app/pipeline/worker.py`."""

    def __init__(self, cam: Camera) -> None:
        self.cam = cam
        self.tracks_by_frame = _load_tracks_by_frame(cam.cam_id)
        self.cap = cv2.VideoCapture(str(cam.path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 10.0
        self.frame_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame_idx = 0
        self._overlay = self._build_overlay()
        # Tracks have_ppe so the stats panel can label the camera even on
        # frames where the PPE pass didn't fire (PPE runs every_n frames).
        self.has_ppe = any(
            (rec.get("ppe") or []) for rec in self.tracks_by_frame.values()
        )

    def _build_overlay(self) -> LiveKpiOverlay | None:
        zones = load_camera_zones(self.cam.cam_id)
        default_role = CONFIG.kpis.camera_default_role.get(self.cam.cam_id)
        if not zones and not default_role:
            return None
        return LiveKpiOverlay(
            cam_id=self.cam.cam_id, zones=zones, fps=self.fps,
            default_role=default_role,
            demographics_aggregator=_StaticDemographics(self.cam.cam_id),
            show_track_ids=CONFIG.pipeline.show_track_ids,
        )

    def reset(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0
        # Rebuild so role-counts / dwell / not-served state restart cleanly.
        self._overlay = self._build_overlay()

    def next(self) -> tuple[np.ndarray | None, dict[str, Any]]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.reset()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None, {}

        idx = self.frame_idx
        rec = self.tracks_by_frame.get(idx, {})
        tracks = rec.get("tracks", [])
        ppe = rec.get("ppe") or []

        if self._overlay is not None:
            stats = self._overlay.step(frame, idx, tracks)
        else:
            draw_tracks(
                frame, tracks,
                show_ids=CONFIG.pipeline.show_track_ids, show_conf=False,
            )
            stats = {
                "ts_s": idx / self.fps,
                "workers_now": 0,
                "customers_now": len(tracks),
            }

        if ppe:
            draw_labels(frame, ppe, show_conf=False)

        stats["tracks_now"] = len(tracks)
        stats["ppe_now"] = len(ppe)
        self.frame_idx += 1
        return frame, stats


# ---------------------------------------------------------------------------
# Stats panel HTML
# ---------------------------------------------------------------------------

def _stats_html(cam: Camera, sim: "CameraSim", stats: dict) -> str:
    if not stats:
        return f'<div class="ana-stats"><i>no data for {cam.cam_id}</i></div>'

    rows: list[str] = [
        f'<div class="ana-row ana-head">'
        f'<span>{cam.cam_id} · {cam.name}</span>'
        f'<span class="ana-time">t={stats["ts_s"]:6.1f}s</span></div>',
        _row("People in view", stats.get("tracks_now", 0)),
        _row("&nbsp;&nbsp;Workers", stats.get("workers_now", 0), "worker"),
        _row("&nbsp;&nbsp;Customers", stats.get("customers_now", 0), "customer"),
    ]
    if "serving_now" in stats:
        rows.append(_row(
            "At counter / queueing",
            f"{stats['serving_now']} / {stats.get('waiting_now', 0)}",
        ))
    if "completed_visits" in stats:
        rows.append(_row("Completed dwell visits", stats["completed_visits"]))
        rows.append(_row(
            "Longest dwell (s)", f"{stats.get('longest_dwell_s', 0.0):.1f}",
        ))
    if stats.get("not_served_active") or stats.get("not_served_pending_s", 0) > 0:
        active = stats.get("not_served_active", False)
        rows.append(_row(
            "Customer not served",
            f"{stats.get('not_served_pending_s', 0):.0f}s"
            + (" 🚨" if active else ""),
            "alert" if active else "",
        ))
    if sim.has_ppe:
        rows.append(_row("PPE labels (this frame)", stats.get("ppe_now", 0)))
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
.ana-alert { color: #ff6b6b; font-weight: 600; }
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
        "zones, dwell timers, PPE labels and demographics from the offline "
        "pipeline output. The overlay uses the same code as the live "
        "worker, so a tile matches its saved annotated MP4."
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
            sph.markdown(_stats_html(sim.cam, sim, stats), unsafe_allow_html=True)

    if not st.session_state.ana_playing:
        _tick()
        st.caption("Paused. Click ▶ Play to start the live simulation.")
        return

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
        raise
