"""Live monitor tab — 2x2 video wall."""
from __future__ import annotations

import streamlit as st

from app.utils.cameras import CAMERAS, Camera, PROJECT_ROOT


def _render_tile(cam: Camera) -> None:
    st.markdown(
        f"""
        <div class="tile-header">
          <span class="cam-name">{cam.name}</span>
          <span class="cam-tag">{cam.cam_id} · {cam.tag}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if cam.path.exists():
        st.video(str(cam.path), loop=True, autoplay=True, muted=True)
    else:
        st.error(f"missing: {cam.path.relative_to(PROJECT_ROOT)}")

    chips = "".join(f'<span class="kpi-chip">{k}</span>' for k in cam.kpis)
    st.markdown(f'<div class="tile-footer">{chips}</div>', unsafe_allow_html=True)


def render() -> None:
    top_left, top_right = st.columns(2, gap="small")
    bottom_left, bottom_right = st.columns(2, gap="small")
    for slot, cam in zip((top_left, top_right, bottom_left, bottom_right), CAMERAS):
        with slot:
            _render_tile(cam)
