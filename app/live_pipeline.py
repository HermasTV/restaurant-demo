"""Live-processing service used by the dashboard's Live Processing tab.

Runs the same YOLO26 → BoT-SORT → LiveKpiOverlay stack as the offline
worker, but as **background threads** owned by the Streamlit session (held
in `st.cache_resource`). Each thread reads its video as if it were a live
CCTV stream (throttled to source FPS, loops at EOF), writes the latest
annotated frame and stats to a shared, lock-protected dict; the dashboard
tab reads from that dict each tick.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from app.utils.cameras import CAMERAS, camera_by_id
from app.config import CONFIG
from app.kpis.zones import load_camera_zones
from app.pipeline.draw import draw_tracks
from app.pipeline.live_kpis import LiveKpiOverlay
from app.pipeline.model import BatchedPersonDetector
from app.pipeline.tracker import make_tracker


class LivePipelineService:
    """Singleton-style service. Use `LivePipelineService.get()` to obtain
    the shared instance (cached at module level so a Streamlit rerun does
    not rebuild the GPU-resident detector).
    """

    _SINGLETON: "LivePipelineService | None" = None

    @classmethod
    def get(cls) -> "LivePipelineService":
        if cls._SINGLETON is None:
            cls._SINGLETON = cls()
        return cls._SINGLETON

    def __init__(self) -> None:
        # One shared, batched detector — concurrent infer() calls from each
        # camera worker are coalesced into a single GPU batch. See
        # BatchedPersonDetector for the rationale.
        self._detector: BatchedPersonDetector | None = None
        self._workers: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        # Shared, lock-protected state. The dashboard tab polls these.
        self._lock = threading.Lock()
        self._latest_frame: dict[str, np.ndarray] = {}
        self._latest_stats: dict[str, dict[str, Any]] = {}
        self._frame_idx: dict[str, int] = {}
        self._last_update_ts: dict[str, float] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._running

    def active_cameras(self) -> list[str]:
        return [cid for cid, t in self._workers.items() if t.is_alive()]

    def start(self, cam_ids: list[str] | None = None) -> None:
        if self._running:
            return
        if cam_ids is None:
            cam_ids = [c.cam_id for c in CAMERAS]
        if self._detector is None:
            self._detector = BatchedPersonDetector(
                max_batch_size=max(1, len(cam_ids))
            )
        self._running = True
        for cid in cam_ids:
            cam = camera_by_id(cid)
            stop = threading.Event()
            self._stop_events[cid] = stop
            t = threading.Thread(
                target=self._worker_loop,
                args=(cam, self._detector, stop),
                daemon=True,
                name=f"live-worker-{cid}",
            )
            t.start()
            self._workers[cid] = t

    def stop(self) -> None:
        for ev in self._stop_events.values():
            ev.set()
        for t in self._workers.values():
            t.join(timeout=2.0)
        self._workers.clear()
        self._stop_events.clear()
        with self._lock:
            self._latest_frame.clear()
            self._latest_stats.clear()
            self._frame_idx.clear()
            self._last_update_ts.clear()
        self._running = False

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker_loop(self, cam, detector: BatchedPersonDetector,
                     stop_event: threading.Event) -> None:
        cap = cv2.VideoCapture(str(cam.path))
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

        tracker = make_tracker(
            CONFIG.tracker, fps=int(round(fps)),
            device=CONFIG.detector.device,
        )
        zones = load_camera_zones(cam.cam_id)
        default_role = CONFIG.kpis.camera_default_role.get(cam.cam_id)
        overlay: LiveKpiOverlay | None = None
        if zones or default_role:
            overlay = LiveKpiOverlay(
                cam_id=cam.cam_id, zones=zones, fps=fps,
                default_role=default_role,
            )

        frame_idx = 0
        target_dt = 1.0 / fps
        t0 = time.monotonic()
        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    # Loop the video.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    t0 = time.monotonic()
                    continue

                detections = detector.infer(frame)
                tracks = tracker.update(detections, frame)

                stats: dict[str, Any] = {"frame_idx": frame_idx,
                                         "ts_s": frame_idx / fps}
                if overlay is not None:
                    stats = overlay.step(frame, frame_idx, tracks)
                else:
                    # No zones/role → just draw raw boxes.
                    draw_tracks(frame, tracks, show_ids=True, show_conf=True)
                    stats["tracks_now"] = len(tracks)

                # Publish (copy so the reader can't see a frame mid-draw).
                with self._lock:
                    self._latest_frame[cam.cam_id] = frame.copy()
                    self._latest_stats[cam.cam_id] = stats
                    self._frame_idx[cam.cam_id] = frame_idx
                    self._last_update_ts[cam.cam_id] = time.monotonic()

                frame_idx += 1

                # Throttle to source FPS to feel like a real stream.
                expected = t0 + frame_idx * target_dt
                delay = expected - time.monotonic()
                if delay > 0:
                    stop_event.wait(delay)
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Reader API for the dashboard
    # ------------------------------------------------------------------

    def get_latest(
        self, cam_id: str
    ) -> tuple[np.ndarray | None, dict[str, Any], int | None, float | None]:
        with self._lock:
            f = self._latest_frame.get(cam_id)
            s = dict(self._latest_stats.get(cam_id, {}))
            idx = self._frame_idx.get(cam_id)
            ts = self._last_update_ts.get(cam_id)
            return (f.copy() if f is not None else None, s, idx, ts)
