"""Live Processing tab — kicks off the GPU pipeline on all 4 streams and
visualises the live annotated frames + stats in a 2×2 grid.

Mechanically similar to the Analytics tab (which replays pre-computed JSONL)
but here the detector + tracker + overlay run in background threads on the
*actual* video, producing fresh annotated frames each tick.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from app.utils.cameras import CAMERAS, Camera
from app.live_pipeline import LivePipelineService


TILE_WIDTH = 640
DISPLAY_FPS = 5  # browser-side refresh rate (not detector rate)


_STYLES = """
<style>
.live-stats {
  background: #161b22;
  border: 1px solid #2a313c;
  border-radius: 6px;
  padding: 8px 10px;
  margin-top: 6px;
  font-size: 12px;
  color: #c9d1d9;
}
.live-row {
  display: flex; justify-content: space-between;
  padding: 2px 0; border-bottom: 1px dashed #21262d;
}
.live-row:last-child { border-bottom: none; }
.live-row.head { font-weight: 600; color: #e6edf3; font-size: 13px;
                 border-bottom: 1px solid #30363d;
                 margin-bottom: 4px; padding-bottom: 4px; }
.live-row .alert { color: #ff5c5c; font-weight: 600; }
.live-row .worker   { color: #ffb86b; }
.live-row .customer { color: #58a6ff; }
.live-time { color: #8b949e; font-variant-numeric: tabular-nums; }
</style>
"""


def _row(label: str, value, klass: str = "") -> str:
    cls = f' class="{klass}"' if klass else ""
    return (
        f'<div class="live-row"><span>{label}</span>'
        f'<span{cls}>{value}</span></div>'
    )


def _stats_html(cam: Camera, stats: dict, frame_idx: int | None,
                last_update_ts: float | None) -> str:
    if not stats:
        return f'<div class="live-stats"><i>no frames yet for {cam.cam_id}</i></div>'
    age = ""
    if last_update_ts is not None:
        d = time.monotonic() - last_update_ts
        if d > 0.5:
            age = f"  (last update {d:.1f}s ago)"
    head = (
        f'<div class="live-row head"><span>{cam.cam_id} · {cam.name}</span>'
        f'<span class="live-time">f{frame_idx if frame_idx is not None else "—"}'
        f' · t={stats.get("ts_s", 0.0):.1f}s{age}</span></div>'
    )
    rows = [head]
    rows.append(_row("Workers", stats.get("workers_now", 0), "worker"))
    rows.append(_row("Customers", stats.get("customers_now", 0), "customer"))
    if "serving_now" in stats and (stats.get("serving_now", 0) or stats.get("avg_serve_s", 0)):
        rows.append(_row(
            "&nbsp;&nbsp;Serving (counter)",
            f'{stats["serving_now"]}  (avg {stats.get("avg_serve_s", 0):.1f}s)',
        ))
    if "waiting_now" in stats and (stats.get("waiting_now", 0) or stats.get("avg_wait_s", 0)):
        rows.append(_row(
            "&nbsp;&nbsp;Waiting (queue)",
            f'{stats["waiting_now"]}  (avg {stats.get("avg_wait_s", 0):.1f}s)',
        ))
    if "completed_visits" in stats:
        rows.append(_row("Completed visits",
                         f'{stats["completed_visits"]}  '
                         f'(max {stats.get("longest_dwell_s", 0):.1f}s)'))
    if stats.get("not_served_active"):
        rows.append(_row(
            "⚠ Customer not served",
            f'{stats.get("not_served_pending_s", 0):.1f}s',
            "alert",
        ))
    elif "not_served_events" in stats and stats["not_served_events"] > 0:
        rows.append(_row("Not-served events", stats["not_served_events"]))
    if "tracks_now" in stats and "customers_now" not in stats:
        # Camera with no zones/role (shouldn't happen given current config)
        rows.append(_row("People in view", stats["tracks_now"]))
    return f'<div class="live-stats">{"".join(rows)}</div>'


def _resize_for_display(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == TILE_WIDTH:
        return frame
    scale = TILE_WIDTH / w
    new_h = int(round(h * scale))
    return cv2.resize(frame, (TILE_WIDTH, new_h), interpolation=cv2.INTER_AREA)


def render() -> None:
    st.markdown(_STYLES, unsafe_allow_html=True)
    st.markdown("#### Live Processing")
    st.caption(
        "Runs the real GPU pipeline (YOLO26 + BoT-SORT + KPI overlay) on all "
        "four streams as if they were live CCTV. Background threads hold the "
        "detector resident on the GPU between starts/stops."
    )

    service = LivePipelineService.get()

    c1, c2, c3 = st.columns([1, 1, 5])
    if c1.button(
        "▶ Start" if not service.is_running() else "Running...",
        disabled=service.is_running(),
        use_container_width=True,
        key="live_start_btn",
    ):
        with st.spinner("Loading detector and starting workers..."):
            service.start()
        st.rerun()
    if c2.button(
        "⏹ Stop",
        disabled=not service.is_running(),
        use_container_width=True,
        key="live_stop_btn",
    ):
        service.stop()
        st.rerun()

    if not service.is_running():
        st.info(
            "Press **▶ Start** to begin real-time processing on the 4 streams. "
            "Each stream is throttled to its native FPS (10), and the page "
            "refreshes the latest annotated frame ~5 times per second."
        )
        return

    # Pre-allocate the 2x2 grid of placeholders.
    rows = [st.columns(2, gap="small"), st.columns(2, gap="small")]
    slots = [rows[0][0], rows[0][1], rows[1][0], rows[1][1]]
    frame_phs = []
    stats_phs = []
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

    target_dt = 1.0 / DISPLAY_FPS
    try:
        while service.is_running():
            t0 = time.monotonic()
            for fph, sph, cam in zip(frame_phs, stats_phs, CAMERAS):
                frame, stats, idx, last_ts = service.get_latest(cam.cam_id)
                if frame is not None:
                    rgb = cv2.cvtColor(_resize_for_display(frame),
                                       cv2.COLOR_BGR2RGB)
                    fph.image(rgb, use_container_width=True)
                sph.markdown(_stats_html(cam, stats, idx, last_ts),
                             unsafe_allow_html=True)
            slept = time.monotonic() - t0
            if slept < target_dt:
                time.sleep(target_dt - slept)
    except Exception:
        # Streamlit fires RerunException on widget interaction — let it propagate.
        raise
