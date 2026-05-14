"""PPE detector — YOLOE with cached visual-prompt embeddings.

Replaces the previous hairnet-only / text-prompt design. Now handles N
classes simultaneously (typically ``["glove", "hairnet"]``) by loading a
.pt file produced by ``experiments/yoloe_select_prompts.py``. That cache
contains the visual-prompt embeddings the user picked on a sample frame
plus the names they map to.

Inference flow:

  PPEDetector(...)            # loads YOLOE + installs cached VPE via set_classes
  detections = det.detect(frame)
  # → [{"bbox": (x1,y1,x2,y2), "conf": float, "class": "glove"|"hairnet"}, ...]
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.config import CONFIG, PPESection


class PPEDetector:
    """Thread-safe YOLOE wrapper for cached visual-prompt PPE detection."""

    def __init__(self, cfg: PPESection | None = None) -> None:
        self.cfg = cfg or CONFIG.ppe
        self._lock = threading.Lock()

        import torch
        from ultralytics import YOLOE
        from app.config import PROJECT_ROOT

        cache_path = (PROJECT_ROOT / self.cfg.vpe_cache).resolve()
        if not cache_path.exists():
            raise FileNotFoundError(
                f"PPE visual-prompt cache not found: {cache_path}\n"
                "Run `python -m experiments.yoloe_select_prompts` to create it."
            )
        ckpt = torch.load(str(cache_path), weights_only=False)
        self.class_names: list[str] = list(ckpt["names"])
        vpe = ckpt["vpe"]
        if vpe.ndim != 3:
            raise ValueError(
                f"unexpected VPE tensor shape {tuple(vpe.shape)}; "
                "expected (1, N_classes, D)"
            )

        model = YOLOE(self.cfg.weights)
        model.model.set_classes(self.class_names, vpe)
        # Make sure result.boxes.cls gives index → user-facing name lookups.
        model.model.names = {i: n for i, n in enumerate(self.class_names)}

        # Warm up so the first real frame isn't a 1-2 s stall.
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        model.predict(
            dummy,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )
        self._model = model
        print(
            f"PPE detector ready  classes={self.class_names}  "
            f"cache={cache_path.name}  vpe={tuple(vpe.shape)}"
        )

    def detect(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        with self._lock:
            r = self._model.predict(
                frame_bgr,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf,
                device=self.cfg.device,
                half=self.cfg.half,
                verbose=False,
            )[0]
        out: list[dict[str, Any]] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            cls_idx = int(cls[i])
            if cls_idx >= len(self.class_names):
                continue
            x1, y1, x2, y2 = xyxy[i].tolist()
            out.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf[i]),
                "class": self.class_names[cls_idx],
            })
        return out
