"""Per-stream tracker — BoT-SORT + OSNet ReID via boxmot, plus the JSONL sink.

Stripped to the single tracker we actually use; alternative backends
(ByteTrack, DeepOCSORT, passthrough) were removed during the cleanup.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from app.config import BoTSORTSection, TrackerSection


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

def _det_dict(x1: float, y1: float, x2: float, y2: float,
              tid: int, conf: float) -> dict[str, Any]:
    return {
        "id": int(tid),
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "conf": float(conf),
        "class": "person",
    }


class BoTSORTAdapter:
    """BoT-SORT + person-ReID via the `boxmot` library."""

    def __init__(self, cfg: BoTSORTSection, fps: int, device: str = "cuda") -> None:
        import os
        # boxmot accidentally writes CUDA_VISIBLE_DEVICES with the raw device
        # string; pre-clear it so a previous bad value can't break CUDA.
        if os.environ.get("CUDA_VISIBLE_DEVICES") in ("cuda", "cuda:0"):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from boxmot.reid.core.reid import ReID
        from boxmot.trackers.botsort.botsort import BotSort

        from app.pipeline.botsort_patched import BotSortStrictAppearance

        # boxmot's select_device() expects a CUDA index like "0", not "cuda".
        reid_device = "0" if device.startswith("cuda") else device

        # Build the ReID backend and pass the inner *backend* (which exposes
        # `.get_features(boxes, img)`) — not the high-level wrapper.
        reid_backend = None
        if cfg.with_reid:
            reid = ReID(weights=cfg.reid_weights, device=reid_device, half=cfg.half_reid)
            reid_backend = reid.model

        TrackerCls = BotSortStrictAppearance if cfg.strict_appearance_gate else BotSort
        self._tracker = TrackerCls(
            reid_model=reid_backend,
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            proximity_thresh=cfg.proximity_thresh,
            appearance_thresh=cfg.appearance_thresh,
            cmc_method=cfg.cmc_method,
            frame_rate=fps,
            fuse_first_associate=cfg.fuse_score,
            with_reid=cfg.with_reid,
        )

    def update(
        self, detections: sv.Detections, frame_bgr: np.ndarray,
    ) -> list[dict[str, Any]]:
        # boxmot expects (N, 6) [x1, y1, x2, y2, conf, cls] float32
        if len(detections) == 0:
            dets_np = np.empty((0, 6), dtype=np.float32)
        else:
            xyxy = detections.xyxy.astype(np.float32)
            conf = (detections.confidence
                    if detections.confidence is not None
                    else np.zeros(len(detections), dtype=np.float32)).astype(np.float32).reshape(-1, 1)
            cls = (detections.class_id
                   if detections.class_id is not None
                   else np.zeros(len(detections), dtype=np.float32)).astype(np.float32).reshape(-1, 1)
            dets_np = np.concatenate([xyxy, conf, cls], axis=1)

        # Returns (M, 8): [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        tracked = self._tracker.update(dets_np, frame_bgr)
        out: list[dict[str, Any]] = []
        if tracked is None or len(tracked) == 0:
            return out
        for row in tracked:
            x1, y1, x2, y2, tid, conf, _cls, _det_idx = row[:8]
            out.append(_det_dict(x1, y1, x2, y2, int(tid), float(conf)))
        return out


def make_tracker(
    cfg: TrackerSection, fps: int, device: str = "cuda",
) -> BoTSORTAdapter:
    return BoTSORTAdapter(cfg.botsort, fps=fps, device=device)


# ---------------------------------------------------------------------------
# JSONL sink (unchanged schema)
# ---------------------------------------------------------------------------

class JsonlTrackWriter:
    """One JSON object per frame, line-buffered for crash resilience."""

    def __init__(self, path: Path, cam_id: str, frame_size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", buffering=1, encoding="utf-8")
        self._cam_id = cam_id
        self._frame_size = (int(frame_size[0]), int(frame_size[1]))

    def write(
        self,
        frame_idx: int,
        ts_s: float,
        tracks: list[dict[str, Any]],
        ppe: list[dict[str, Any]] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "cam_id": self._cam_id,
            "frame_idx": int(frame_idx),
            "ts_s": float(ts_s),
            "frame_size": list(self._frame_size),
            "tracks": tracks,
        }
        if ppe is not None:
            record["ppe"] = ppe
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlTrackWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
