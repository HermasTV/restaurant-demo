"""Thread-safe YOLO person detector shared across stream workers.

Configuration comes from [detector] in config.toml. A single instance lives on
the GPU and is serialized by an internal lock; per-call latency is small
enough that 4 streams at 10 fps fit comfortably.
"""
from __future__ import annotations

import threading

import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.config import CONFIG, DetectorSection


class PersonDetector:
    """Wraps an Ultralytics YOLO model, person-class only, thread-safe."""

    def __init__(self, cfg: DetectorSection | None = None) -> None:
        self.cfg: DetectorSection = cfg or CONFIG.detector
        self._lock = threading.Lock()

        model = YOLO(self.cfg.weights)
        # Let predict() do device + fp16 conversion. Manual model.model.half()
        # before fuse() trips a dtype mismatch in some YOLO releases.
        dummy = np.zeros(
            (self.cfg.imgsz[1], self.cfg.imgsz[0], 3), dtype=np.uint8
        )
        model.predict(
            dummy,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            classes=self.cfg.classes,
            half=self.cfg.half,
            device=self.cfg.device,
            verbose=False,
        )
        self._model = model

    def infer(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Return person-only detections in original-frame pixel space.

        Applies an explicit NMS pass on top of YOLO's own output. YOLO26 is
        NMS-free at training, but at low conf it still leaks overlapping
        torso-only / full-body bboxes of the same person, which then spawn
        duplicate tracks downstream.
        """
        with self._lock:
            result = self._model.predict(
                frame_bgr,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                classes=self.cfg.classes,
                half=self.cfg.half,
                device=self.cfg.device,
                verbose=False,
            )[0]
        detections = sv.Detections.from_ultralytics(result)
        # Class-agnostic NMS at the configured IoU — keep highest-conf box per
        # cluster, suppress duplicates.
        detections = detections.with_nms(threshold=self.cfg.iou, class_agnostic=True)
        return detections
