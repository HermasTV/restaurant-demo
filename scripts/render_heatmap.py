"""Build and render an occupancy heatmap from JSONL tracks.

Usage:
    python -m scripts.render_heatmap --cam CAM-03
    python -m scripts.render_heatmap --cam CAM-03 --all-tracks   # include workers
    python -m scripts.render_heatmap --cam CAM-03 --blur 71 --alpha 0.7

Inputs:
    data/tracks/<CAM>.jsonl        (per-frame tracks)
    data/kpis/<CAM>.roles.json     (filters to role == "customer" by default)
    data/reference_frames/<CAM>.png  (background for the overlay)

Outputs:
    data/kpis/<CAM>.heatmap.npy        raw (height × width float32) seconds
    data/kpis/<CAM>.heatmap.png        colormap only
    data/annotated/<CAM>.heatmap.png   colormap overlaid on the reference frame
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils.cameras import DATA_DIR, camera_by_id  # noqa: E402
from app.utils.frames import get_reference_frame  # noqa: E402
from app.kpis.heatmap import (  # noqa: E402
    GAUSSIAN_KSIZE,
    OVERLAY_ALPHA,
    build_from_jsonl,
)

TRACKS_DIR = DATA_DIR / "tracks"
KPIS_DIR = DATA_DIR / "kpis"
ANNOTATED_DIR = DATA_DIR / "annotated"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render occupancy heatmap.")
    p.add_argument("--cam", required=True)
    p.add_argument("--all-tracks", action="store_true",
                   help="Include all roles (default: customers only).")
    p.add_argument("--blur", type=int, default=GAUSSIAN_KSIZE,
                   help=f"Gaussian kernel size (odd). Default {GAUSSIAN_KSIZE}.")
    p.add_argument("--alpha", type=float, default=OVERLAY_ALPHA,
                   help=f"Overlay weight 0..1. Default {OVERLAY_ALPHA}.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cam = camera_by_id(args.cam)
    tracks_path = TRACKS_DIR / f"{cam.cam_id}.jsonl"
    if not tracks_path.exists():
        raise SystemExit(f"missing tracks: {tracks_path}")

    # Decide whose foot-points to accumulate.
    customer_ids: set[int] | None = None
    if not args.all_tracks:
        roles_path = KPIS_DIR / f"{cam.cam_id}.roles.json"
        if not roles_path.exists():
            raise SystemExit(
                f"missing {roles_path} — run scripts.run_kpis first, or pass --all-tracks"
            )
        roles = json.loads(roles_path.read_text())["tracks"]
        customer_ids = {int(k) for k, v in roles.items() if v["role"] == "customer"}
        print(f"[{cam.cam_id}] filtering to {len(customer_ids)} customer tracks")
    else:
        print(f"[{cam.cam_id}] including all tracks")

    # Infer FPS from JSONL ts_s deltas.
    fps = 10.0
    with tracks_path.open("r", encoding="utf-8") as fh:
        prev_ts = None
        for line in fh:
            rec = json.loads(line)
            if prev_ts is not None and rec["ts_s"] > prev_ts:
                fps = 1.0 / (rec["ts_s"] - prev_ts)
                break
            prev_ts = rec["ts_s"]

    print(f"[{cam.cam_id}] building accumulator at fps={fps:.2f}")
    accum = build_from_jsonl(tracks_path, customer_ids=customer_ids, fps=fps)

    total_seconds = float(accum.grid.sum())
    peak_seconds = float(accum.grid.max())
    print(f"[{cam.cam_id}] total deposited time: {total_seconds:.1f} s   "
          f"peak per-pixel deposit: {peak_seconds:.2f} s")

    # Save outputs.
    KPIS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = KPIS_DIR / f"{cam.cam_id}.heatmap.npy"
    flat_png_path = KPIS_DIR / f"{cam.cam_id}.heatmap.png"
    overlay_path = ANNOTATED_DIR / f"{cam.cam_id}.heatmap.png"

    np.save(raw_path, accum.grid)
    smoothed, coloured = accum.render(gaussian_ksize=args.blur)
    cv2.imwrite(str(flat_png_path), coloured)

    ref_pil = get_reference_frame(cam)
    ref_bgr = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)
    _, overlay = accum.overlay_on(
        ref_bgr, gaussian_ksize=args.blur, alpha=args.alpha
    )
    cv2.imwrite(str(overlay_path), overlay)

    # Report top-K hotspots (peaks in the smoothed grid).
    k = 5
    flat_idx = np.argpartition(-smoothed.flatten(), k)[:k]
    print(f"\ntop-{k} hotspots (smoothed):")
    for idx in flat_idx[np.argsort(-smoothed.flatten()[flat_idx])]:
        y, x = divmod(int(idx), accum.width)
        print(f"  ({x:>4}, {y:>4})  ≈ {smoothed[y, x]:.2f} s/pixel after blur")

    print()
    print(f"wrote raw     : {raw_path}")
    print(f"wrote colormap: {flat_png_path}")
    print(f"wrote overlay : {overlay_path}")


if __name__ == "__main__":
    main()
