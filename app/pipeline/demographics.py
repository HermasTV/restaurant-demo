"""MiVOLO age + gender — minimal demo version.

Per track: run MiVOLO on each frame the track is visible until we have
``min_confident_frames`` matched observations, then finalize by averaging
(mean age → age band; sum of gender confidence scores → majority gender).

Intentionally simple — there is no per-track rate limiter, no attempt cap,
no smoothing of stale tracks. Anything more elaborate belongs in the
technical report, not the demo code.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import CONFIG, DemographicsSection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _age_band(age: float, cfg: DemographicsSection) -> str:
    bins, labels = cfg.age_band_bins, cfg.age_band_labels
    for i in range(len(labels)):
        if bins[i] <= age < bins[i + 1]:
            return labels[i]
    return labels[-1]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass
class _MiVoloConfig:
    detector_weights: str
    checkpoint: str
    device: str
    draw: bool = False
    with_persons: bool = True
    disable_faces: bool = False


class DemographicsClassifier:
    """Thread-safe wrapper around MiVOLO's Predictor."""

    def __init__(self, cfg: DemographicsSection | None = None) -> None:
        self.cfg = cfg or CONFIG.demographics
        self._lock = threading.Lock()
        from mivolo.predictor import Predictor
        from app.config import PROJECT_ROOT
        det_w = str((PROJECT_ROOT / self.cfg.detector_weights).resolve())
        mvw = str((PROJECT_ROOT / self.cfg.weights).resolve())
        self._predictor = Predictor(
            _MiVoloConfig(detector_weights=det_w, checkpoint=mvw,
                          device=self.cfg.device),
            verbose=False,
        )

    def predict(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        with self._lock:
            result, _ = self._predictor.recognize(frame_bgr)
        out: list[dict[str, Any]] = []
        for ind in result.get_bboxes_inds("person"):
            age = result.ages[ind]
            gender = result.genders[ind]
            gscore = result.gender_scores[ind]
            if age is None or gender is None:
                continue
            box = result.yolo_results.boxes[ind].xyxy[0].cpu().numpy()
            out.append({
                "bbox": (float(box[0]), float(box[1]),
                         float(box[2]), float(box[3])),
                "age": float(age),
                "gender": "M" if str(gender).lower().startswith("m") else "F",
                "gender_score": float(gscore) if gscore is not None else 0.0,
            })
        return out


# ---------------------------------------------------------------------------
# Per-track aggregator (minimal)
# ---------------------------------------------------------------------------

@dataclass
class _Votes:
    ages: list[float] = field(default_factory=list)
    genders: list[str] = field(default_factory=list)
    gscores: list[float] = field(default_factory=list)
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    last_sampled_frame: int = -10_000
    finalized: bool = False
    final_age_band: str | None = None
    final_gender: str | None = None
    final_confidence: float | None = None


@dataclass
class DemographicsAggregator:
    cam_id: str
    cfg: DemographicsSection

    _votes: dict[int, _Votes] = field(default_factory=dict, init=False)

    # ---- gating ----

    def should_sample(self, tracks: list[dict[str, Any]],
                      frame_idx: int) -> bool:
        """True if any current track needs another reading right now.
        A track gets at most one reading per ``sample_gap_frames`` frames.
        """
        if not tracks:
            return False
        gap = self.cfg.sample_gap_frames
        needed = self.cfg.min_confident_frames
        for t in tracks:
            v = self._votes.get(int(t["id"]))
            if v is None:
                return True  # brand new track
            if v.finalized or len(v.ages) >= needed:
                continue
            if frame_idx - v.last_sampled_frame >= gap:
                return True
        return False

    # ---- update ----

    def update(self, tracks: list[dict[str, Any]],
               predictions: list[dict[str, Any]],
               ts_s: float, frame_idx: int) -> None:
        if not tracks:
            return

        # Ensure every visible track has a slot, and stamp the sampling time
        # for every unfinalized one (so the per-track gap takes effect).
        for t in tracks:
            tid = int(t["id"])
            v = self._votes.get(tid)
            if v is None:
                v = _Votes(first_seen_s=ts_s)
                self._votes[tid] = v
            if not v.finalized:
                v.last_sampled_frame = frame_idx

        # Match each prediction to its best-IoU current track.
        track_boxes = [(int(t["id"]), tuple(t["bbox"])) for t in tracks]
        for p in predictions:
            pbox = p["bbox"]
            ph = max(0.0, pbox[3] - pbox[1])
            if ph < self.cfg.min_bbox_height_px:
                continue
            best_tid, best_iou = None, self.cfg.iou_match_thresh
            for tid, tbox in track_boxes:
                u = _iou(pbox, tbox)
                if u > best_iou:
                    best_iou, best_tid = u, tid
            if best_tid is None:
                continue
            v = self._votes[best_tid]
            if v.finalized:
                continue
            v.ages.append(p["age"])
            v.genders.append(p["gender"])
            v.gscores.append(p["gender_score"])
            v.last_seen_s = ts_s
            if len(v.ages) >= self.cfg.min_confident_frames:
                self._finalize(best_tid, v)

    def _finalize(self, tid: int, v: _Votes) -> None:
        # Average age → bucket. Sum-of-confidence vote for gender. The chosen
        # gender's average score becomes the reported confidence.
        avg_age = sum(v.ages) / len(v.ages)
        male_score = sum(s for g, s in zip(v.genders, v.gscores) if g == "M")
        female_score = sum(s for g, s in zip(v.genders, v.gscores) if g == "F")
        if male_score >= female_score:
            gender, winning_score = "M", male_score
        else:
            gender, winning_score = "F", female_score
        n_winner = sum(1 for g in v.genders if g == gender) or 1
        avg_conf = winning_score / n_winner
        v.final_age_band = _age_band(avg_age, self.cfg)
        v.final_gender = gender
        v.final_confidence = round(avg_conf, 3)
        v.finalized = True

    # ---- read-out ----

    def label_for(self, tid: int) -> str | None:
        v = self._votes.get(tid)
        if v is None or not v.finalized:
            return None
        return f"{v.final_age_band}, {v.final_gender}"

    def counts(self) -> dict[str, Any]:
        from collections import Counter
        bands: Counter[str] = Counter()
        genders: Counter[str] = Counter()
        finalized = pending = 0
        for v in self._votes.values():
            if v.finalized:
                finalized += 1
                bands[v.final_age_band] += 1
                genders[v.final_gender] += 1
            else:
                pending += 1
        return {"age_bands": dict(bands), "genders": dict(genders),
                "finalized": finalized, "pending": pending}

    def write_sidecar(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        records: dict[str, dict[str, Any]] = {}
        for tid, v in self._votes.items():
            if not v.finalized:
                continue
            records[str(tid)] = {
                "age_band": v.final_age_band,
                "gender": v.final_gender,
                "confidence": v.final_confidence,
                "n_observations": len(v.ages),
                "first_seen_s": round(v.first_seen_s, 2),
                "last_seen_s": round(v.last_seen_s, 2),
            }
        path.write_text(json.dumps(
            {"cam_id": self.cam_id, "tracks": records,
             "summary": self.counts()},
            indent=2,
        ))
        return len(records)
