"""Extract one representative still frame per camera and save to disk.

Used by the annotation tool as the canvas background and as a stable reference
image for downstream visualization.

Run from project root:
    python scripts/extract_reference_frames.py
    python scripts/extract_reference_frames.py --position 0.25  # quarter-way through
    python scripts/extract_reference_frames.py --force          # overwrite existing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils.cameras import CAMERAS, DATA_DIR
from app.utils.frames import grab_frame

REFERENCE_DIR = DATA_DIR / "reference_frames"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", type=float, default=0.5,
                        help="Normalized frame position in [0, 1] (default: 0.5)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing reference frames")
    args = parser.parse_args()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    for cam in CAMERAS:
        out = REFERENCE_DIR / f"{cam.cam_id}.png"
        if out.exists() and not args.force:
            print(f"skip   {cam.cam_id}  ({out.relative_to(_PROJECT_ROOT)} exists)")
            continue
        if not cam.path.exists():
            print(f"miss   {cam.cam_id}  source video not found: {cam.path}")
            continue
        img = grab_frame(cam.path, position=args.position)
        img.save(out, format="PNG")
        print(f"saved  {cam.cam_id}  {img.size[0]}x{img.size[1]}  -> "
              f"{out.relative_to(_PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
