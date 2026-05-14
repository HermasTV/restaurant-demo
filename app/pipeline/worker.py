"""Per-stream worker: read frames -> detect -> track -> write JSONL + MP4."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

from app.utils.cameras import Camera, DATA_DIR
from app.config import CONFIG
from app.kpis.zones import center_point, load_camera_zones, point_in_polygon
from app.pipeline.demographics import (
    DemographicsAggregator,
    DemographicsClassifier,
)
from app.pipeline.draw import draw_labels, draw_tracks
from app.pipeline.live_kpis import LiveKpiOverlay
from app.pipeline.model import PersonDetector
from app.pipeline.ppe import PPEDetector
from app.pipeline.tracker import JsonlTrackWriter, make_tracker


def run_worker(
    cam: Camera,
    detector: PersonDetector,
    tracks_dir: Path,
    annotated_dir: Path,
    stop_event: threading.Event,
    ppe_detector: PPEDetector | None = None,
    demographics: DemographicsClassifier | None = None,
) -> None:
    cap = cv2.VideoCapture(str(cam.path))
    if not cap.isOpened():
        print(f"[{cam.cam_id}] cannot open: {cam.path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    tracks_path = tracks_dir / f"{cam.cam_id}.jsonl"
    tracker = make_tracker(
        CONFIG.tracker, fps=int(round(fps)), device=CONFIG.detector.device
    )

    write_annotated = CONFIG.pipeline.write_annotated
    realtime = CONFIG.pipeline.realtime
    ppe_every_n = CONFIG.ppe.every_n
    demographics_every_n = CONFIG.demographics.every_n
    demographics_agg: DemographicsAggregator | None = None
    if demographics is not None:
        demographics_agg = DemographicsAggregator(
            cam_id=cam.cam_id, cfg=CONFIG.demographics,
        )

    writer = None
    if write_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        out_path = annotated_dir / f"{cam.cam_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[{cam.cam_id}] cannot open annotated writer for {out_path}")
            writer = None

    frame_idx = 0
    target_dt = 1.0 / fps
    t0 = time.monotonic()
    # Live KPI overlay: zones + per-track role + per-tracklet dwell timer.
    # Active when the camera has saved polygons OR a default-role override.
    zones = load_camera_zones(cam.cam_id)
    default_role = CONFIG.kpis.camera_default_role.get(cam.cam_id)
    kpi_overlay: LiveKpiOverlay | None = None
    if zones or default_role:
        kpi_overlay = LiveKpiOverlay(
            cam_id=cam.cam_id, zones=zones, fps=fps,
            default_role=default_role,
            demographics_aggregator=demographics_agg,
            show_track_ids=CONFIG.pipeline.show_track_ids,
        )

    # PPE detections are kept only inside worker zones — gloves / hairnets
    # outside the worker area are noise. When the camera has *no* worker
    # zones (e.g. CAM-04 with default_role=worker) the whole frame counts.
    ppe_worker_polys = [z.polygon for z in zones if z.is_worker]

    print(
        f"[{cam.cam_id}] start  size={width}x{height} fps={fps:.2f} "
        f"frames={frame_total} realtime={realtime} annotated={writer is not None} "
        f"tracker=botsort "
        f"ppe={ppe_detector is not None} "
        f"kpi_overlay={kpi_overlay is not None} "
        f"demographics={demographics is not None}"
    )

    cached_ppe: list[dict] = []
    try:
        with JsonlTrackWriter(tracks_path, cam.cam_id, (width, height)) as jsonl:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                detections = detector.infer(frame)
                tracks = tracker.update(detections, frame)

                ppe_this_frame: list[dict] | None = None
                if ppe_detector is not None and frame_idx % ppe_every_n == 0:
                    ppe_this_frame = ppe_detector.detect(frame)
                    if ppe_worker_polys:
                        # Filter to worker-area detections only.
                        ppe_this_frame = [
                            p for p in ppe_this_frame
                            if any(
                                point_in_polygon(*center_point(p["bbox"]), poly)
                                for poly in ppe_worker_polys
                            )
                        ]
                    cached_ppe = ppe_this_frame

                ts_s = frame_idx / fps

                # Run the KPI overlay first so we know each track's role for
                # the rest of this frame's work (demographics, etc.). Drawing
                # happens here too; the frame is consumed by writer.write
                # below.
                roles: dict[int, str] = {}
                customers_with_evidence: set[int] = set()
                if kpi_overlay is not None:
                    stats = kpi_overlay.step(frame, frame_idx, tracks)
                    roles = stats.get("roles", {})
                    customers_with_evidence = stats.get(
                        "customers_with_evidence", set()
                    )

                # Demographics: restrict to customer-role tracks with
                # *positive* customer-zone evidence — `role == "customer"`
                # alone is the default fallback (no "unknown" bucket) and
                # would include workers caught in their first frames before
                # reaching `worker_area`. Sampling those locks them in as
                # male customers at ~10 s.
                customer_tracks = [
                    t for t in tracks
                    if roles.get(int(t["id"])) == "customer"
                    and int(t["id"]) in customers_with_evidence
                ]
                if (
                    demographics is not None and demographics_agg is not None
                    and customer_tracks
                    and demographics_agg.should_sample(customer_tracks, frame_idx)
                ):
                    preds = demographics.predict(frame)
                    demographics_agg.update(customer_tracks, preds, ts_s, frame_idx)

                jsonl.write(frame_idx, ts_s, tracks, ppe=ppe_this_frame)

                if writer is not None:
                    if kpi_overlay is None:
                        draw_tracks(
                            frame, tracks,
                            show_ids=CONFIG.pipeline.show_track_ids,
                            show_conf=True,
                        )
                    if cached_ppe:
                        draw_labels(frame, cached_ppe)
                    writer.write(frame)

                frame_idx += 1

                if frame_idx % 200 == 0:
                    elapsed = time.monotonic() - t0
                    print(
                        f"[{cam.cam_id}] frame={frame_idx}/{frame_total} "
                        f"({frame_idx / max(elapsed, 1e-6):.1f} fps wall)"
                    )

                if realtime:
                    expected = t0 + frame_idx * target_dt
                    delay = expected - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if demographics_agg is not None:
            out = DATA_DIR / "kpis" / f"{cam.cam_id}.demographics.json"
            n = demographics_agg.write_sidecar(out)
            print(f"[{cam.cam_id}] demographics: wrote {n} finalized tracks -> {out}")
        elapsed = time.monotonic() - t0
        print(
            f"[{cam.cam_id}] done   frames={frame_idx} "
            f"wall={elapsed:.1f}s  effective_fps={frame_idx / max(elapsed, 1e-6):.1f}"
        )
