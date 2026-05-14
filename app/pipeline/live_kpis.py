"""Real-time KPI overlay used inside the live worker.

For each frame, given the current tracks (already produced by the tracker),
this module:

  * decides each track's role (worker / customer) from polygon
    membership accumulated over the track's lifetime — predefined regions
    cover the frame, so there is no "unknown" fallback,
  * keeps a per-tracklet, per-zone dwell timer,
  * runs a "customer-not-served" state machine that raises a flag when one or
    more customers are waiting and no worker is in any worker zone for more
    than `kpis.customer_not_served_threshold_s` seconds,
  * draws zone overlays, role-colored boxes + labels, and a top-left HUD,
  * returns a stats dict (workers_now, customers_now, completed_visits,
    longest_dwell_s, not_served_active, not_served_events, …).

Decision rules:
  * If `default_role` is set (e.g. CAM-04 = worker, CAM-03 = customer),
    every track gets that role unconditionally.
  * Else: per-track running counts of frames-in-worker-zone vs
    frames-in-customer-zone. A track is a worker only if its worker-zone
    count strictly exceeds its customer-zone count; everyone else is a
    customer. Tracks that never intersect any polygon (e.g. background
    foot traffic) fall through to customer by design — the predefined
    regions cover the meaningful frame area.

Coordinates use:
  * Bbox CENTER for role / zone-membership counting — handles workers whose
    legs are occluded by a counter.
  * Bbox FOOT (BOTTOM_CENTER) for dwell — customers stand fully visible.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.config import CONFIG
from app.kpis.zones import (
    Zone,
    center_point,
    foot_point,
    point_in_polygon,
    split_zones,
    zone_kind,
)


ROLE_COLORS_BGR = {
    "worker":   (0, 165, 255),   # orange
    "customer": (255, 200, 80),  # light blue
    "unknown":  (180, 180, 180), # grey
}
WORKER_ZONE_COLOR = (0, 140, 220)
CUSTOMER_ZONE_COLOR = (220, 160, 70)
ZONE_FILL_ALPHA = 0.15
HUD_BG = (24, 28, 35)
HUD_FG = (235, 240, 245)
ALERT_RED = (0, 60, 230)


def _short_zone_name(name: str) -> str:
    """`customer-counter` → `counter`, `employee-billing` → `billing`."""
    return name.split("-", 1)[-1] if "-" in name else name


@dataclass
class _DwellState:
    enter_frame: int
    last_in_frame: int
    in_frames: int = 1


@dataclass
class LiveKpiOverlay:
    """Per-frame overlay + stats. One instance per camera."""

    cam_id: str
    zones: list[Zone]
    fps: float
    default_role: str | None = None
    # Optional: a DemographicsAggregator. If provided, finalized per-track
    # (age band, gender) tags are appended to each box label.
    demographics_aggregator: Any | None = None
    # Render the BoT-SORT id next to customer boxes (workers never get it).
    # Default off — flip from config for dev sessions.
    show_track_ids: bool = False

    # internal
    _worker_zones: list[Zone] = field(default_factory=list, init=False)
    _customer_zones: list[Zone] = field(default_factory=list, init=False)
    # zone_name -> "counter" | "queue" | "general"
    _zone_kind: dict[str, str] = field(default_factory=dict, init=False)
    _has_counter_zones: bool = field(default=False, init=False)
    _has_queue_zones: bool = field(default=False, init=False)
    _role_counts: dict[int, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"worker": 0, "customer": 0}),
        init=False,
    )
    _dwell_open: dict[tuple[int, str], _DwellState] = field(default_factory=dict, init=False)
    completed_visits: list[dict[str, Any]] = field(default_factory=list, init=False)

    # Customer-not-served state machine
    _ns_start_frame: int | None = field(default=None, init=False)
    _ns_active: bool = field(default=False, init=False)
    _ns_max_customers: int = field(default=0, init=False)
    not_served_events: list[dict[str, Any]] = field(default_factory=list, init=False)

    _min_dwell_s: float = field(init=False)
    _tolerance_frames: int = field(init=False)
    _not_served_threshold_s: float = field(init=False)

    def __post_init__(self) -> None:
        self._worker_zones, self._customer_zones = split_zones(self.zones)
        for z in self._customer_zones:
            self._zone_kind[z.name] = zone_kind(z.name)
        self._has_counter_zones = any(k == "counter" for k in self._zone_kind.values())
        self._has_queue_zones = any(k == "queue" for k in self._zone_kind.values())
        self._min_dwell_s = CONFIG.kpis.min_dwell_s
        self._tolerance_frames = max(1, int(round(
            CONFIG.kpis.reenter_tolerance_s * self.fps
        )))
        self._not_served_threshold_s = CONFIG.kpis.customer_not_served_threshold_s

    # ---------------- decision helpers ----------------

    def _role_for(self, tid: int) -> str:
        # Worker only if the track has spent more frames inside a worker
        # zone than inside customer zones. Predefined regions
        # (worker_area + customer_counter/queue) cover the meaningful
        # frame area, so a track that never intersected any polygon —
        # e.g. someone passing in the background — defaults to customer
        # rather than "unknown".
        if self.default_role is not None:
            return self.default_role
        counts = self._role_counts.get(tid, {"worker": 0, "customer": 0})
        if self._worker_zones and counts["worker"] > counts["customer"]:
            return "worker"
        return "customer"

    # ---------------- main step ----------------

    def step(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        tracks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ts_s = frame_idx / self.fps

        # ---- 1) Update role-classification counts (bbox center) and capture
        #      live zone membership per track (used for label + not-served).
        track_in_worker_zone: dict[int, bool] = {}
        track_in_customer_zones: dict[int, list[str]] = {}
        for t in tracks:
            tid = int(t["id"])
            cx, cy = center_point(t["bbox"])
            in_worker = any(point_in_polygon(cx, cy, z.polygon)
                            for z in self._worker_zones)
            customer_hits = [z.name for z in self._customer_zones
                             if point_in_polygon(cx, cy, z.polygon)]
            track_in_worker_zone[tid] = in_worker
            track_in_customer_zones[tid] = customer_hits
            if in_worker:
                self._role_counts[tid]["worker"] += 1
            if customer_hits:
                self._role_counts[tid]["customer"] += 1

        # ---- 2) Dwell timers (foot anchor on customer zones).
        # Per-track per-zone live elapsed seconds.
        track_zone_dwell: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for t in tracks:
            tid = int(t["id"])
            fx, fy = foot_point(t["bbox"])
            for z in self._customer_zones:
                if not point_in_polygon(fx, fy, z.polygon):
                    continue
                key = (tid, z.name)
                state = self._dwell_open.get(key)
                if state is None:
                    self._dwell_open[key] = _DwellState(
                        enter_frame=frame_idx, last_in_frame=frame_idx,
                    )
                else:
                    state.last_in_frame = frame_idx
                    state.in_frames += 1
                elapsed = (frame_idx - self._dwell_open[key].enter_frame) / self.fps
                track_zone_dwell[tid].append((z.name, elapsed))

        # Close stale visits.
        for key in list(self._dwell_open.keys()):
            state = self._dwell_open[key]
            if frame_idx - state.last_in_frame > self._tolerance_frames:
                duration = state.in_frames / self.fps
                if duration >= self._min_dwell_s:
                    tid, zname = key
                    self.completed_visits.append({
                        "track_id": tid,
                        "zone": zname,
                        "duration_s": round(duration, 2),
                        "enter_ts_s": round(state.enter_frame / self.fps, 2),
                        "exit_ts_s": round(state.last_in_frame / self.fps, 2),
                        "role": self._role_for(tid),
                    })
                del self._dwell_open[key]

        # ---- 3) Roles for everyone visible this frame.
        roles: dict[int, str] = {int(t["id"]): self._role_for(int(t["id"]))
                                 for t in tracks}

        # ---- 4a) Per-frame serving / waiting split.
        # A customer is "serving" if currently inside a counter-kind zone,
        # "waiting" if inside a queue-kind zone, "in_zone" if inside any
        # customer zone (used for the not-served check). For cameras with no
        # customer zones (CAM-02 pre-annotation), fall back to "anyone outside
        # the worker zone counts as waiting".
        serving_now = 0
        waiting_now = 0
        customers_in_zone_now = 0  # in any customer zone
        for t in tracks:
            tid = int(t["id"])
            if roles[tid] != "customer":
                continue
            zones_in = track_in_customer_zones.get(tid, [])
            kinds = {self._zone_kind.get(zn, "general") for zn in zones_in}
            if "counter" in kinds:
                serving_now += 1
            if "queue" in kinds:
                waiting_now += 1
            if zones_in:
                customers_in_zone_now += 1
            elif not self._customer_zones and not track_in_worker_zone.get(tid, False):
                # CAM-02-style fallback: no customer zones defined yet.
                waiting_now += 1
                customers_in_zone_now += 1

        # ---- 4b) Customer-not-served state machine.
        # "Worker present" = any track currently inside any worker zone
        # (regardless of classified role — being there is enough).
        worker_present = any(track_in_worker_zone.get(int(t["id"]), False)
                             for t in tracks)
        customers_waiting_now = customers_in_zone_now

        ns_pending_seconds = 0.0
        # Customer-not-served only makes sense when the camera defines worker
        # zones (CAM-01, CAM-02). Skip on CAM-03 (dining) / CAM-04 (kitchen),
        # which use a default-role override and have no service concept.
        ns_applicable = bool(self._worker_zones)
        if ns_applicable and customers_waiting_now > 0 and not worker_present:
            if self._ns_start_frame is None:
                self._ns_start_frame = frame_idx
                self._ns_max_customers = customers_waiting_now
            else:
                self._ns_max_customers = max(
                    self._ns_max_customers, customers_waiting_now
                )
            ns_pending_seconds = (frame_idx - self._ns_start_frame) / self.fps
            if not self._ns_active and ns_pending_seconds >= self._not_served_threshold_s:
                self._ns_active = True
        else:
            if ns_applicable and self._ns_active:
                # Condition cleared — emit the completed event.
                duration = (frame_idx - self._ns_start_frame) / self.fps
                self.not_served_events.append({
                    "start_ts_s": round(self._ns_start_frame / self.fps, 2),
                    "end_ts_s": round(ts_s, 2),
                    "duration_s": round(duration, 2),
                    "max_customers": self._ns_max_customers,
                })
            self._ns_start_frame = None
            self._ns_active = False
            self._ns_max_customers = 0

        # ---- 5) Live averages over completed visits, split by zone kind.
        serve_durations = [v["duration_s"] for v in self.completed_visits
                           if self._zone_kind.get(v["zone"]) == "counter"]
        wait_durations = [v["duration_s"] for v in self.completed_visits
                          if self._zone_kind.get(v["zone"]) == "queue"]
        avg_serve_s = sum(serve_durations) / len(serve_durations) if serve_durations else 0.0
        avg_wait_s = sum(wait_durations) / len(wait_durations) if wait_durations else 0.0

        # ---- 6) Draw.
        self._draw_zones(frame_bgr)
        for t in tracks:
            tid = int(t["id"])
            self._draw_box(frame_bgr, t["bbox"], tid, roles[tid],
                           track_zone_dwell.get(tid))
        workers_now = sum(1 for r in roles.values() if r == "worker")
        customers_now = sum(1 for r in roles.values() if r == "customer")
        longest_dwell = max(
            (v["duration_s"] for v in self.completed_visits), default=0.0
        )
        self._draw_hud(
            frame_bgr, ts_s,
            workers_now=workers_now,
            customers_now=customers_now,
            serving_now=serving_now,
            waiting_now=waiting_now,
            avg_serve_s=avg_serve_s,
            avg_wait_s=avg_wait_s,
            completed_visits=len(self.completed_visits),
            longest_dwell_s=longest_dwell,
            active_dwells=len(self._dwell_open),
            ns_pending_s=ns_pending_seconds,
            ns_active=self._ns_active,
            ns_event_count=len(self.not_served_events) + (1 if self._ns_active else 0),
        )
        if self._ns_active:
            self._draw_alert_banner(frame_bgr, ns_pending_seconds)

        return {
            "ts_s": ts_s,
            "workers_now": workers_now,
            "customers_now": customers_now,
            "serving_now": serving_now,
            "waiting_now": waiting_now,
            "customers_waiting": customers_waiting_now,
            "avg_serve_s": avg_serve_s,
            "avg_wait_s": avg_wait_s,
            "track_zone_dwell": dict(track_zone_dwell),
            "active_dwells": len(self._dwell_open),
            "completed_visits": len(self.completed_visits),
            "longest_dwell_s": longest_dwell,
            "not_served_active": self._ns_active,
            "not_served_pending_s": ns_pending_seconds,
            "not_served_events": len(self.not_served_events),
            "roles": roles,
        }

    # ---------------- drawing helpers ----------------

    def _draw_zones(self, frame: np.ndarray) -> None:
        if not self.zones:
            return
        overlay = frame.copy()
        for z in self.zones:
            color = WORKER_ZONE_COLOR if z.is_worker else CUSTOMER_ZONE_COLOR
            cv2.fillPoly(overlay, [z.polygon], color)
        cv2.addWeighted(overlay, ZONE_FILL_ALPHA, frame,
                        1 - ZONE_FILL_ALPHA, 0, frame)
        for z in self.zones:
            color = WORKER_ZONE_COLOR if z.is_worker else CUSTOMER_ZONE_COLOR
            cv2.polylines(frame, [z.polygon], True, color, 2)
            cx = int(np.mean(z.polygon[:, 0]))
            cy = int(np.mean(z.polygon[:, 1]))
            label = z.name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 6),
                          (cx + tw // 2 + 4, cy + 2), HUD_BG, -1)
            cv2.putText(frame, label, (cx - tw // 2, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_FG, 1, cv2.LINE_AA)

    def _draw_box(
        self, frame: np.ndarray, bbox: list[float], tid: int, role: str,
        zone_dwells: list[tuple[str, float]] | None,
    ) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        color = ROLE_COLORS_BGR.get(role, ROLE_COLORS_BGR["unknown"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label rules:
        #   * workers never get an id (or any per-zone dwell — they don't dwell)
        #   * customers get an id only when show_track_ids is on
        #   * demographics chip only appears for customers (workers aren't sampled)
        if role == "worker":
            label = "worker"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly2 = max(y1, th + 4)
            cv2.rectangle(frame, (x1, ly2 - th - 4),
                          (x1 + tw + 4, ly2), color, -1)
            cv2.putText(frame, label, (x1 + 2, ly2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            return
        parts = [f"customer:{tid}" if self.show_track_ids else "customer"] \
                if role == "customer" else [role]
        if self.demographics_aggregator is not None:
            demo = self.demographics_aggregator.label_for(tid)
            if demo:
                parts.append(f"[{demo}]")
        for zname, ds in (zone_dwells or []):
            if ds < 1.0:
                continue
            parts.append(f"{_short_zone_name(zname)}:{ds:.0f}s")
        label = "  ".join(parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly2 = max(y1, th + 4)
        cv2.rectangle(frame, (x1, ly2 - th - 4),
                      (x1 + tw + 4, ly2), color, -1)
        cv2.putText(frame, label, (x1 + 2, ly2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_hud(
        self, frame: np.ndarray, ts_s: float,
        workers_now: int, customers_now: int,
        serving_now: int, waiting_now: int,
        avg_serve_s: float, avg_wait_s: float,
        completed_visits: int, longest_dwell_s: float, active_dwells: int,
        ns_pending_s: float, ns_active: bool, ns_event_count: int,
    ) -> None:
        lines: list[tuple[str, bool]] = [
            (f"{self.cam_id}   t={ts_s:6.1f}s", False),
            (f"  workers       {workers_now:>3}", False),
            (f"  customers     {customers_now:>3}", False),
        ]
        # Show the serving/waiting split + avgs only when the camera actually
        # has counter and/or queue zones defined.
        if self._has_counter_zones:
            lines.append((
                f"    serving     {serving_now:>3}   (avg {avg_serve_s:>4.1f}s)", False,
            ))
        if self._has_queue_zones:
            lines.append((
                f"    waiting     {waiting_now:>3}   (avg {avg_wait_s:>4.1f}s)", False,
            ))
        lines.append(
            (f"  done visits   {completed_visits:>3}   max {longest_dwell_s:>5.1f}s", False),
        )
        # Customer-not-served line only relevant when worker zones exist.
        if self._worker_zones:
            lines.append((
                f"  no-service    {ns_pending_s:>5.1f}s   evts {ns_event_count:>2}",
                ns_active,
            ))
        line_h = 20
        pad = 8
        box_w = max(
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            for text, _ in lines
        ) + pad * 2
        box_h = line_h * len(lines) + pad * 2
        cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), HUD_BG, -1)
        cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), (60, 70, 80), 1)
        for i, (text, alert) in enumerate(lines):
            y = 8 + pad + (i + 1) * line_h - 6
            color = ALERT_RED if alert else HUD_FG
            cv2.putText(frame, text, (8 + pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _draw_alert_banner(self, frame: np.ndarray, ns_pending_s: float) -> None:
        h, w = frame.shape[:2]
        text = f"  CUSTOMER NOT SERVED  -  {ns_pending_s:.0f}s  "
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        pad = 12
        x = w - tw - 2 * pad - 12
        y = 12
        cv2.rectangle(frame, (x, y), (x + tw + 2 * pad, y + th + 2 * pad),
                      ALERT_RED, -1)
        cv2.putText(frame, text, (x + pad, y + th + pad - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
