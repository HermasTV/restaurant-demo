"""Orchestrator: spin up one worker thread per camera. Config-driven.

Usage:
    python -m app.pipeline.run

All parameters (detector, tracker, ppe, demographics, which cameras to
process, etc.) live in config.toml — no CLI flags.
"""
from __future__ import annotations

import signal
import sys
import threading
from pathlib import Path

# Allow `python -m app.pipeline.run` to find `app.*` when run from project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.utils.cameras import CAMERAS, DATA_DIR, camera_by_id  # noqa: E402
from app.config import CONFIG  # noqa: E402
from app.pipeline.demographics import DemographicsClassifier  # noqa: E402
from app.pipeline.model import BatchedPersonDetector  # noqa: E402
from app.pipeline.ppe import PPEDetector  # noqa: E402
from app.pipeline.worker import run_worker  # noqa: E402

TRACKS_DIR = DATA_DIR / "tracks"
ANNOTATED_DIR = DATA_DIR / "annotated"


def main() -> None:
    cam_ids = CONFIG.pipeline.cameras or [c.cam_id for c in CAMERAS]
    cams = [camera_by_id(cid) for cid in cam_ids]
    ppe_cams = set(CONFIG.ppe.cameras)
    demographics_cams = set(CONFIG.demographics.cameras)

    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG.pipeline.write_annotated:
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"config  detector={CONFIG.detector.weights} imgsz={CONFIG.detector.imgsz} "
        f"device={CONFIG.detector.device}"
    )
    print(
        f"config  tracker=botsort  reid={CONFIG.tracker.botsort.reid_weights}  "
        f"buffer={CONFIG.tracker.botsort.track_buffer}"
    )
    print(
        f"config  ppe={'on' if CONFIG.ppe.enabled else 'off'} "
        f"cameras={sorted(ppe_cams) or '[]'}"
    )
    print(
        f"config  demographics={'on' if CONFIG.demographics.enabled else 'off'} "
        f"cameras={sorted(demographics_cams) or '[]'}"
    )
    print(f"running on  {[c.cam_id for c in cams]}")

    # Single detector shared across all workers, but calls are batched:
    # the dispatcher coalesces concurrent infer() requests from each
    # stream into one model.predict([f1, f2, f3, f4]) — far cheaper on
    # the GPU than four serialised single-frame calls (a batch of 4 at
    # 640x384 ≈ 1.5–2× a single forward pass, not 4×).
    detector = BatchedPersonDetector(max_batch_size=max(1, len(cams)))
    print(f"person detector ready  (batched, max_batch={max(1, len(cams))})")

    # PPE and demographics remain shared: each runs on at most 1-2 cameras,
    # so contention is negligible and the loaders are heavy (YOLOE backbone,
    # MiVOLO + internal YOLOv8x).
    ppe_detector: PPEDetector | None = None
    needs_ppe = (
        CONFIG.ppe.enabled
        and any(c.cam_id in ppe_cams for c in cams)
    )
    if needs_ppe:
        ppe_detector = PPEDetector()
        print("PPE detector ready")

    demographics: DemographicsClassifier | None = None
    needs_demo = (
        CONFIG.demographics.enabled
        and any(c.cam_id in demographics_cams for c in cams)
    )
    if needs_demo:
        demographics = DemographicsClassifier()
        print("demographics classifier (MiVOLO) ready")

    stop_event = threading.Event()

    def _sigint(_signum, _frame) -> None:
        print("\ninterrupt received, stopping workers...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint)

    threads: list[threading.Thread] = []
    for cam in cams:
        t = threading.Thread(
            target=run_worker,
            name=f"worker-{cam.cam_id}",
            args=(
                cam,
                detector,
                TRACKS_DIR,
                ANNOTATED_DIR,
                stop_event,
                ppe_detector if cam.cam_id in ppe_cams else None,
                demographics if cam.cam_id in demographics_cams else None,
            ),
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    detector.shutdown()
    print("all workers finished.")


if __name__ == "__main__":
    main()
