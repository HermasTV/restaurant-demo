"""Typed loader for config.toml — the single source of truth for the POC.

Read once at import; downstream modules access fields off the `CONFIG` object.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class PipelineSection:
    cameras: list[str]
    realtime: bool
    write_annotated: bool
    show_track_ids: bool


@dataclass(frozen=True)
class DetectorSection:
    weights: str
    device: str
    imgsz: tuple[int, int]
    conf: float
    iou: float
    half: bool
    classes: list[int]


@dataclass(frozen=True)
class BoTSORTSection:
    track_buffer: int
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    match_thresh: float
    fuse_score: bool
    cmc_method: str
    proximity_thresh: float
    appearance_thresh: float
    with_reid: bool
    reid_weights: str
    half_reid: bool
    strict_appearance_gate: bool


@dataclass(frozen=True)
class TrackerSection:
    botsort: BoTSORTSection


@dataclass(frozen=True)
class PPESection:
    enabled: bool
    weights: str
    vpe_cache: str
    device: str
    imgsz: int
    conf: float
    half: bool
    every_n: int
    cameras: list[str]


@dataclass(frozen=True)
class DemographicsSection:
    enabled: bool
    weights: str
    detector_weights: str
    device: str
    half: bool
    cameras: list[str]
    every_n: int
    sample_gap_frames: int
    min_confident_frames: int
    min_bbox_height_px: int
    iou_match_thresh: float
    age_band_bins: list[int]
    age_band_labels: list[str]


@dataclass(frozen=True)
class KpisSection:
    min_dwell_s: float
    reenter_tolerance_s: float
    customer_not_served_threshold_s: float
    worker_zone_prefixes: tuple[str, ...]
    camera_default_role: dict[str, str]


@dataclass(frozen=True)
class Config:
    pipeline: PipelineSection
    detector: DetectorSection
    tracker: TrackerSection
    ppe: PPESection
    demographics: DemographicsSection
    kpis: KpisSection


def _load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    kp = data.get("kpis", {})
    return Config(
        pipeline=PipelineSection(
            cameras=list(data["pipeline"]["cameras"]),
            realtime=bool(data["pipeline"]["realtime"]),
            write_annotated=bool(data["pipeline"]["write_annotated"]),
            show_track_ids=bool(data["pipeline"].get("show_track_ids", False)),
        ),
        detector=DetectorSection(
            weights=data["detector"]["weights"],
            device=data["detector"]["device"],
            imgsz=tuple(data["detector"]["imgsz"]),  # type: ignore[arg-type]
            conf=float(data["detector"]["conf"]),
            iou=float(data["detector"]["iou"]),
            half=bool(data["detector"]["half"]),
            classes=list(data["detector"]["classes"]),
        ),
        tracker=TrackerSection(
            botsort=BoTSORTSection(**data["tracker"]["botsort"]),
        ),
        ppe=PPESection(
            enabled=bool(data["ppe"]["enabled"]),
            weights=data["ppe"]["weights"],
            vpe_cache=data["ppe"]["vpe_cache"],
            device=data["ppe"]["device"],
            imgsz=int(data["ppe"]["imgsz"]),
            conf=float(data["ppe"]["conf"]),
            half=bool(data["ppe"]["half"]),
            every_n=int(data["ppe"]["every_n"]),
            cameras=list(data["ppe"]["cameras"]),
        ),
        demographics=DemographicsSection(
            enabled=bool(data["demographics"]["enabled"]),
            weights=data["demographics"]["weights"],
            detector_weights=data["demographics"]["detector_weights"],
            device=data["demographics"]["device"],
            half=bool(data["demographics"]["half"]),
            cameras=list(data["demographics"]["cameras"]),
            every_n=int(data["demographics"]["every_n"]),
            sample_gap_frames=int(data["demographics"].get("sample_gap_frames", 10)),
            min_confident_frames=int(data["demographics"]["min_confident_frames"]),
            min_bbox_height_px=int(data["demographics"]["min_bbox_height_px"]),
            iou_match_thresh=float(data["demographics"].get("iou_match_thresh", 0.4)),
            age_band_bins=list(data["demographics"]["age_band_bins"]),
            age_band_labels=list(data["demographics"]["age_band_labels"]),
        ),
        kpis=KpisSection(
            min_dwell_s=float(kp.get("min_dwell_s", 3.0)),
            reenter_tolerance_s=float(kp.get("reenter_tolerance_s", 1.0)),
            customer_not_served_threshold_s=float(
                kp.get("customer_not_served_threshold_s", 20.0)
            ),
            worker_zone_prefixes=tuple(kp.get(
                "worker_zone_prefixes",
                ["employee-", "employee_", "worker_", "staff_"],
            )),
            camera_default_role=dict(kp.get("camera_default_role", {})),
        ),
    )


CONFIG: Config = _load_config()
