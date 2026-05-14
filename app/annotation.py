"""Zone annotation tab.

For each camera with a defined region registry (`app/regions.py`) the user
picks a canonical region name from a dropdown and draws its polygon. No
free-text names. If that region already has a saved polygon it's painted
onto the reference frame as a translucent overlay so the user can see what
they're about to replace.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from app.utils import _canvas_shim  # noqa: F401  (monkey-patches before next import)
from streamlit_drawable_canvas import st_canvas

from app.utils.auth import is_authed, login_form, logout
from app.utils.cameras import CAMERAS, camera_by_id
from app.utils.frames import (
    fit_to_width,
    get_reference_frame,
    grab_frame,
    reference_frame_path,
)
from app.utils.regions import Region, regions_for
from app.utils.zones_io import delete_zone, load_zones, save_zone

MAX_CANVAS_WIDTH = 900

WORKER_FILL = (255, 140, 0, 70)     # orange, alpha
WORKER_STROKE = (255, 140, 0, 255)
CUSTOMER_FILL = (88, 166, 255, 70)  # light blue, alpha
CUSTOMER_STROKE = (88, 166, 255, 255)


# ---------------------------------------------------------------------------
# Reference-frame caching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_reference_frame(cam_id: str, mtime: float) -> Image.Image:
    del mtime  # part of cache key
    return get_reference_frame(camera_by_id(cam_id))


# ---------------------------------------------------------------------------
# Polygon overlay on the reference frame
# ---------------------------------------------------------------------------

def _overlay_existing_polygons(
    base: Image.Image,
    zones: list[dict[str, Any]],
    region_map: dict[str, Region],
) -> Image.Image:
    """Paint every saved polygon onto a copy of the reference frame, then
    return that. Worker zones get orange, customer zones blue.

    Painting is done on an RGBA layer composited on top so the user always
    sees the un-tinted frame underneath."""
    if not zones:
        return base
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for z in zones:
        region = region_map.get(z["name"])
        if region is not None and region.is_worker:
            fill, stroke = WORKER_FILL, WORKER_STROKE
        else:
            fill, stroke = CUSTOMER_FILL, CUSTOMER_STROKE
        pts = [(float(p[0]), float(p[1])) for p in z["points"]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=fill, outline=stroke)
            # text label at centroid
            cx = sum(x for x, _ in pts) / len(pts)
            cy = sum(y for _, y in pts) / len(pts)
            draw.text((cx - 40, cy - 6), z["name"],
                      fill=(255, 255, 255, 230))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Canvas helpers (unchanged)
# ---------------------------------------------------------------------------

def _extract_polygon_points(obj: dict[str, Any]) -> list[tuple[float, float]]:
    obj_type = obj.get("type")
    if obj_type == "path":
        pts: list[tuple[float, float]] = []
        for cmd in obj.get("path", []):
            if cmd and cmd[0] in ("M", "L") and len(cmd) >= 3:
                pts.append((float(cmd[1]), float(cmd[2])))
        return pts
    if obj_type == "polygon":
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        return [(left + float(p["x"]), top + float(p["y"]))
                for p in obj.get("points", [])]
    return []


def _polygons_from_canvas(result: Any) -> list[list[tuple[float, float]]]:
    if result is None or result.json_data is None:
        return []
    polys = []
    for obj in result.json_data.get("objects", []):
        pts = _extract_polygon_points(obj)
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def _canvas_nonce_key(cam_id: str, region_name: str) -> str:
    return f"canvas_nonce_{cam_id}_{region_name}"


def _bump_nonce(cam_id: str, region_name: str) -> None:
    key = _canvas_nonce_key(cam_id, region_name)
    st.session_state[key] = st.session_state.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Top-level render
# ---------------------------------------------------------------------------

def render() -> None:
    if not is_authed():
        login_form()
        return

    header_l, header_r = st.columns([4, 1])
    with header_l:
        st.markdown("#### Zone Annotation")
        st.caption(
            "Pick a region from the dropdown, draw a polygon "
            "(click to add vertices, double-click to close), then save."
        )
    with header_r:
        if st.button("Sign out", use_container_width=True):
            logout()
            st.rerun()

    cam_col, refresh_col = st.columns([4, 1])
    with cam_col:
        cam_id = st.selectbox(
            "Camera",
            [c.cam_id for c in CAMERAS],
            format_func=lambda cid: f"{cid} — {camera_by_id(cid).name}",
            key="annot_cam_select",
        )
    cam = camera_by_id(cam_id)
    with refresh_col:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button(
            "Regenerate frame",
            help="Re-extract the still frame from the video",
            use_container_width=True,
        ):
            ref_path = reference_frame_path(cam)
            new_img = grab_frame(cam.path)
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            new_img.save(ref_path, format="PNG")
            _cached_reference_frame.clear()
            st.rerun()

    # Per-camera region registry — if empty, this camera isn't annotated.
    regions = regions_for(cam.cam_id)
    if not regions:
        st.info(
            f"**{cam.cam_id} — {cam.name}** has no annotated regions in this "
            "POC. Its role is set globally via `kpis.camera_default_role` in "
            "`config.toml` (e.g. CAM-03 dining → everyone is `customer`, "
            "CAM-04 kitchen → everyone is `worker`)."
        )
        return

    ref_path = reference_frame_path(cam)
    if not ref_path.exists() and not cam.path.exists():
        st.error(f"No reference frame and source video missing: {cam.path}")
        return

    mtime = ref_path.stat().st_mtime if ref_path.exists() else 0.0
    frame_full = _cached_reference_frame(cam.cam_id, mtime)

    region_map = {r.name: r for r in regions}
    saved_zones = load_zones(cam.cam_id)
    saved_by_name = {z["name"]: z for z in saved_zones}

    # Paint EVERY saved polygon onto the background so the user sees all
    # context (e.g. the workers area when editing the queue).
    background_full = _overlay_existing_polygons(
        frame_full, saved_zones, region_map
    )
    background_display, scale = fit_to_width(background_full, MAX_CANVAS_WIDTH)

    # -----------------------------------------------------------------------
    # Layout: canvas + region picker on the left, saved-zones list on the right
    # -----------------------------------------------------------------------
    canvas_col, side_col = st.columns([3, 1], gap="medium")

    with canvas_col:
        region_options = [r.name for r in regions]

        def _label(name: str) -> str:
            r = region_map[name]
            tag = "★ existing" if name in saved_by_name else "+ new"
            kind = "worker" if r.is_worker else "customer"
            return f"{r.label}   ({kind} · {tag})"

        chosen_name = st.selectbox(
            "Region",
            region_options,
            format_func=_label,
            key=f"region_select_{cam_id}",
        )
        chosen_region = region_map[chosen_name]
        already = saved_by_name.get(chosen_name)

        nonce = st.session_state.get(_canvas_nonce_key(cam_id, chosen_name), 0)
        stroke = "#ff8c00" if chosen_region.is_worker else "#58a6ff"
        fill = "rgba(255,140,0,0.25)" if chosen_region.is_worker else "rgba(88,166,255,0.25)"

        canvas_result = st_canvas(
            fill_color=fill,
            stroke_width=2,
            stroke_color=stroke,
            background_image=background_display,
            drawing_mode="polygon",
            update_streamlit=True,
            width=background_display.width,
            height=background_display.height,
            key=f"canvas_{cam_id}_{chosen_name}_{nonce}",
        )

        action_l, action_r = st.columns(2)
        with action_l:
            save_clicked = st.button(
                f"Save {chosen_region.label}",
                type="primary",
                use_container_width=True,
                key=f"save_{cam_id}_{chosen_name}",
            )
        with action_r:
            if already is not None:
                if st.button(
                    f"Delete {chosen_region.label}",
                    use_container_width=True,
                    key=f"del_{cam_id}_{chosen_name}",
                ):
                    delete_zone(cam.cam_id, chosen_name)
                    _bump_nonce(cam_id, chosen_name)
                    st.rerun()
            else:
                st.caption("No saved polygon yet for this region.")

        if save_clicked:
            polys = _polygons_from_canvas(canvas_result)
            if not polys:
                st.warning(
                    "Draw a polygon first (≥ 3 vertices, double-click to close)."
                )
            else:
                last = polys[-1]
                points_orig = [(x / scale, y / scale) for x, y in last]
                save_zone(
                    cam.cam_id,
                    chosen_name,
                    [list(p) for p in points_orig],
                    (frame_full.width, frame_full.height),
                )
                _bump_nonce(cam_id, chosen_name)
                st.success(
                    f"Saved {chosen_region.label!r} on {cam.cam_id} "
                    f"({len(points_orig)} vertices)."
                )
                st.rerun()

    # -----------------------------------------------------------------------
    # Sidebar: saved-zones list for this camera
    # -----------------------------------------------------------------------
    with side_col:
        st.markdown("**Saved zones**")
        st.caption(f"{cam.cam_id} · {cam.name}")
        if not saved_zones:
            st.markdown(
                '<div style="color:#8b949e;font-size:12px;font-style:italic;">'
                "None yet — pick a region and draw the polygon on the canvas."
                "</div>",
                unsafe_allow_html=True,
            )
        for r in regions:
            z = saved_by_name.get(r.name)
            color = "#ff8c00" if r.is_worker else "#58a6ff"
            kind = "worker" if r.is_worker else "customer"
            if z is None:
                body = '<span style="color:#8b949e;">not drawn yet</span>'
            else:
                body = (
                    f'<span style="color:#8b949e;">{len(z["points"])} pts · {kind}</span>'
                )
            st.markdown(
                f'<div style="padding:6px 8px;background:#1c232c;'
                f'border-left:3px solid {color}; border:1px solid #2a313c;'
                f'border-radius:6px;font-size:12px; margin-bottom:6px;">'
                f'<b>{r.label}</b><br/>{body}'
                f'</div>',
                unsafe_allow_html=True,
            )
