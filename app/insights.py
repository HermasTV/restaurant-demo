"""Insights tab — single fixed layout.

* Occupancy heatmap is only produced for **CAM-03 (dining)** — that's the
  only camera with persistent customers worth a heatmap.
* Demographics is only produced for **CAM-01 (billing)** in the current POC.

The tab reads exclusively from files already on disk; no GPU work runs here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from app.utils.cameras import DATA_DIR

HEATMAP_CAM = "CAM-03"
DEMOGRAPHICS_CAM = "CAM-01"

ANNOTATED_DIR = DATA_DIR / "annotated"
KPIS_DIR = DATA_DIR / "kpis"


_STYLES = """
<style>
.ins-empty {
  color: #8b949e; font-style: italic; font-size: 13px;
  background: #0d1117; border: 1px dashed #2a313c;
  border-radius: 8px; padding: 14px;
}
.ins-section h5 { margin: 0 0 6px 0; color: #e6edf3; font-weight: 600; font-size: 14px; }
.ins-section .cam-tag {
  display: inline-block; background: #1f6feb22; color: #58a6ff;
  border: 1px solid #1f6feb55; border-radius: 10px;
  padding: 1px 8px; font-size: 11px; margin-left: 6px;
}
</style>
"""


def _load_demographics() -> dict | None:
    path = KPIS_DIR / f"{DEMOGRAPHICS_CAM}.demographics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _heatmap_paths() -> tuple[Path | None, Path | None]:
    overlay = ANNOTATED_DIR / f"{HEATMAP_CAM}.heatmap.png"
    raw = KPIS_DIR / f"{HEATMAP_CAM}.heatmap.npy"
    return (overlay if overlay.exists() else None,
            raw if raw.exists() else None)


def _render_heatmap_section() -> None:
    st.markdown(
        f'<div class="ins-section"><h5>Occupancy heatmap '
        f'<span class="cam-tag">{HEATMAP_CAM} · Dining</span></h5></div>',
        unsafe_allow_html=True,
    )
    overlay, raw = _heatmap_paths()
    if overlay is None:
        st.markdown(
            f'<div class="ins-empty">No heatmap yet. Run '
            f'<code>python -m scripts.render_heatmap --cam {HEATMAP_CAM}</code>.'
            f'</div>',
            unsafe_allow_html=True,
        )
        return
    st.image(
        str(overlay), use_container_width=True,
        caption=f"{HEATMAP_CAM} time-weighted customer footprint "
                "(customer role only, blurred + jet colormap).",
    )
    if raw is not None:
        try:
            grid = np.load(raw)
            total_s = float(grid.sum())
            peak_s = float(grid.max())
            non_zero = int((grid > 0).sum())
            coverage_pct = 100 * non_zero / grid.size
            c1, c2, c3 = st.columns(3)
            c1.metric("Total customer-seconds", f"{total_s:,.0f}")
            c2.metric("Hottest pixel (s)", f"{peak_s:.2f}")
            c3.metric("Frame coverage", f"{coverage_pct:.1f}%")
        except Exception:
            pass


def _render_demographics_section() -> None:
    st.markdown(
        f'<div class="ins-section"><h5>Demographics '
        f'<span class="cam-tag">{DEMOGRAPHICS_CAM} · Billing</span></h5></div>',
        unsafe_allow_html=True,
    )
    demo = _load_demographics()
    if demo is None:
        st.markdown(
            '<div class="ins-empty">No demographics file yet. Enable '
            '<code>[demographics]</code> in <code>config.toml</code> and '
            're-run the pipeline.</div>',
            unsafe_allow_html=True,
        )
        return

    tracks: dict[str, dict[str, Any]] = demo.get("tracks", {})
    summary = demo.get("summary", {}) or {}
    age_bands = dict(summary.get("age_bands", {}))
    genders = dict(summary.get("genders", {}))
    finalized = summary.get("finalized", 0)
    pending = summary.get("pending", 0)

    if not tracks:
        st.markdown(
            '<div class="ins-empty">Sidecar present but no finalized '
            'tracks yet.</div>',
            unsafe_allow_html=True,
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Finalized", finalized)
    c2.metric("Pending", pending)
    c3.metric("Male", genders.get("M", 0))
    c4.metric("Female", genders.get("F", 0))

    chart_l, chart_r = st.columns(2)
    canonical = ["<18", "18-30", "30-50", "50+"]
    if age_bands:
        with chart_l:
            st.caption("**Age band**")
            ordered = {b: age_bands.get(b, 0) for b in canonical if b in age_bands}
            for b, c in age_bands.items():
                if b not in ordered:
                    ordered[b] = c
            st.bar_chart(ordered, height=200, use_container_width=True)
    if genders:
        with chart_r:
            st.caption("**Gender**")
            st.bar_chart({"Male": genders.get("M", 0), "Female": genders.get("F", 0)},
                         height=200, use_container_width=True)

    with st.expander(f"Per-track records ({len(tracks)} tracks)"):
        rows = [
            {
                "Track ID": int(tid),
                "Age band": v.get("age_band"),
                "Gender": v.get("gender"),
                "Confidence": v.get("confidence"),
                "Observations": v.get("n_observations"),
                "First seen (s)": v.get("first_seen_s"),
                "Last seen (s)": v.get("last_seen_s"),
            }
            for tid, v in sorted(tracks.items(), key=lambda kv: int(kv[0]))
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render() -> None:
    st.markdown(_STYLES, unsafe_allow_html=True)
    st.markdown("#### Insights")
    st.caption(
        "Static, end-of-run analytics. Heatmap is fixed to the dining "
        f"camera ({HEATMAP_CAM}); demographics is fixed to the billing "
        f"camera ({DEMOGRAPHICS_CAM})."
    )
    left, right = st.columns([3, 4], gap="medium")
    with left:
        _render_heatmap_section()
    with right:
        _render_demographics_section()
