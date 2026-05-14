"""YOLO person detectors used by the streaming pipeline.

Two flavours are provided:

* `PersonDetector` — single-frame, lock-guarded. Used by experiments / one-off
  scripts and as the building block for batched inference.
* `BatchedPersonDetector` — same `infer(frame) -> sv.Detections` interface,
  but a small background dispatch thread coalesces concurrent calls into a
  single `model.predict([...])` of up to `max_batch_size` frames. Cuts the
  per-frame cost across N streams: a batch of 4 at 640×384 on a 4070 is
  roughly 1.5–2× a single forward pass, not 4×, because the GPU is the same
  hardware doing one bigger matmul.

Configuration comes from [detector] in config.toml.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.config import CONFIG, DetectorSection


def _new_model(cfg: DetectorSection) -> YOLO:
    model = YOLO(cfg.weights)
    # Warmup so the first real call doesn't pay JIT / cudnn-autotune cost
    # inside the dispatcher loop.
    dummy = np.zeros((cfg.imgsz[1], cfg.imgsz[0], 3), dtype=np.uint8)
    # Letting predict() do device + fp16 conversion. Manual model.model.half()
    # before fuse() trips a dtype mismatch in some YOLO releases.
    model.predict(
        dummy,
        imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, classes=cfg.classes,
        half=cfg.half, device=cfg.device, verbose=False,
    )
    return model


def _result_to_detections(result, iou: float) -> sv.Detections:
    detections = sv.Detections.from_ultralytics(result)
    # Class-agnostic NMS on top of YOLO26's NMS-free output: at low conf it
    # still leaks overlapping torso-only + full-body boxes of the same
    # person, which would spawn duplicate tracks downstream.
    return detections.with_nms(threshold=iou, class_agnostic=True)


class PersonDetector:
    """Single-frame YOLO wrapper, person-class only, lock-guarded."""

    def __init__(self, cfg: DetectorSection | None = None) -> None:
        self.cfg: DetectorSection = cfg or CONFIG.detector
        self._lock = threading.Lock()
        self._model = _new_model(self.cfg)

    def infer(self, frame_bgr: np.ndarray) -> sv.Detections:
        with self._lock:
            result = self._model.predict(
                frame_bgr,
                imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou,
                classes=self.cfg.classes, half=self.cfg.half,
                device=self.cfg.device, verbose=False,
            )[0]
        return _result_to_detections(result, self.cfg.iou)


# ---------------------------------------------------------------------------
# Batched variant
# ---------------------------------------------------------------------------


@dataclass
class _BatchJob:
    frame: np.ndarray
    done: threading.Event = field(default_factory=threading.Event)
    result: sv.Detections | None = None


class BatchedPersonDetector:
    """Coalesces concurrent `infer()` calls into a single batched predict.

    Each caller (per-stream worker) calls `infer(frame)` and blocks on the
    job's event. A dispatcher thread pulls the first job, waits up to
    `max_wait_ms` for more, then runs `model.predict([f1, ..., fN])` once
    and wakes each caller with its own result.

    `infer()` keeps the same single-frame `sv.Detections` API as
    `PersonDetector`, so callers don't need to know they're being batched.
    """

    def __init__(
        self,
        cfg: DetectorSection | None = None,
        max_batch_size: int = 4,
        max_wait_ms: float = 5.0,
    ) -> None:
        self.cfg: DetectorSection = cfg or CONFIG.detector
        self._max_batch = max(1, int(max_batch_size))
        self._max_wait_s = max(0.0, max_wait_ms / 1000.0)
        self._model = _new_model(self.cfg)
        self._queue: queue.Queue[_BatchJob | None] = queue.Queue()
        self._stop = threading.Event()
        self._dispatcher = threading.Thread(
            target=self._loop, name="batched-detector", daemon=True,
        )
        self._dispatcher.start()

    def infer(self, frame_bgr: np.ndarray) -> sv.Detections:
        job = _BatchJob(frame=frame_bgr)
        self._queue.put(job)
        job.done.wait()
        return job.result  # type: ignore[return-value]

    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._queue.put(None)  # unblock dispatcher
        self._dispatcher.join(timeout=2.0)

    # -- internal --------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                return
            batch: list[_BatchJob] = [first]
            # Try to coalesce up to max_batch_size, waiting at most
            # max_wait_s in total for stragglers.
            deadline = time.monotonic() + self._max_wait_s
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self._stop.set()
                    break
                batch.append(nxt)
            self._run_batch(batch)

    def _run_batch(self, batch: list[_BatchJob]) -> None:
        frames = [job.frame for job in batch]
        try:
            results = self._model.predict(
                frames,
                imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou,
                classes=self.cfg.classes, half=self.cfg.half,
                device=self.cfg.device, verbose=False,
            )
            for job, result in zip(batch, results):
                job.result = _result_to_detections(result, self.cfg.iou)
                job.done.set()
        except Exception:
            # Surface failure to every blocked caller as an empty detection
            # set so workers can't hang on a dead dispatcher.
            for job in batch:
                job.result = sv.Detections.empty()
                job.done.set()
            raise
