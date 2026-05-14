# Restaurant CCTV Monitoring — POC

Python-only computer-vision prototype that flags operational and compliance
KPIs on restaurant CCTV footage. Streamlit for the UI; YOLO26 + BoT-SORT
(with OSNet ReID) for detection + tracking; MiVOLO for demographics; YOLOE
with cached visual prompts for PPE (gloves / hairnet). All KPIs run off
per-frame JSONL track logs so the dashboard stays decoupled from the GPU
pipeline.

---

## KPIs in scope

The brief listed 13 KPIs. Three couldn't be implemented from the supplied
footage (noted below), the rest are covered by this POC.

| # | KPI | Status | Notes |
|---|---|---|---|
| 1 | Queue build-up / crowd congestion | ✅ | Per-zone counts via polygon zones + role classifier. |
| 2 | Kitchen hygiene compliance | ✅ | PPE pass runs on CAM-04 (kitchen) only — gloves / hairnets elsewhere are noise. |
| 3 | Cash counter open – no customer | ❌ | Drawer never opened across the supplied clips. |
| 4 | Cash counter open – extended duration | ❌ | Same reason as #3. |
| 5 | Handwashing duration (20 s) | ❌ | No view of a sink / handwash station provided. |
| 6 | Customer served – receipt not printed | ⚠ | Needs POS integration (out of scope). |
| 7 | Receipt printed – no customer | ⚠ | Same. |
| 8 | Gloves compliance | ✅ | YOLOE visual-prompt detector, kitchen camera only. |
| 9 | Hairnet compliance | ✅ | Same YOLOE pass; second cached class. |
| 10 | Unattended customer | ✅ | "Customer-not-served" timer on CAM-01 / CAM-02. Fires when ≥1 customer is in `customer_counter` AND no worker is in `worker_area` for > threshold seconds. Queueing customers don't trigger. Custom verification script added because no clip naturally showed an unattended customer. |
| 11 | Customer demographics | ✅ | MiVOLO-D1 (face + body), per-track aggregation, age band + gender, privacy-preserving (no identity stored). |
| 12 | Occupancy heatmap | ✅ | Foot-point accumulator, Gaussian-smoothed, overlaid on reference frame. |
| 13 | Customer dwell time | ✅ | Per-zone, wall-clock based so FPS / inference latency doesn't skew the result. |

Other gaps from the supplied footage:

- Only 4 camera views were provided out of a possible 9, so some
  zone-specific KPIs (sanitation room, handwash) simply have no input.
- The "queue" and "counter" share a single camera in this dataset, so they
  are split inside the frame by polygon zones (`customer_queue` vs
  `customer_counter`) rather than by camera.

---

## Project layout

```
restaurant/
├── config.toml                        # single source of truth (paths, thresholds, toggles)
├── README.md
├── app/                               # Streamlit dashboard + pipeline code
│   ├── dashboard.py                   #   entry: `streamlit run app/dashboard.py`
│   ├── live_monitor.py                #   tab — 2x2 raw video wall
│   ├── live_processing.py             #   tab — live GPU pipeline, overlays on screen
│   ├── live_pipeline.py               #   in-process driver behind live_processing
│   ├── analytics.py                   #   tab — simulated replay with offline JSONL overlays
│   ├── insights.py                    #   tab — heatmap + demographics summary
│   ├── annotation.py                  #   tab — polygon zone editor (auth-gated)
│   ├── config.py                      #   typed loader for config.toml
│   ├── utils/
│   │   ├── cameras.py                 #     camera registry (single source of truth for paths)
│   │   ├── frames.py                  #     still-frame extraction helper
│   │   ├── zones_io.py                #     polygon persistence (data/zones.json)
│   │   ├── regions.py                 #     canonical region names + per-camera registry
│   │   ├── auth.py                    #     demo auth (123 / 123)
│   │   └── _canvas_shim.py            #     streamlit-drawable-canvas patch
│   ├── pipeline/                      #   GPU pipeline
│   │   ├── model.py                   #     YOLO26 person detector (with NMS post-pass)
│   │   ├── tracker.py                 #     BoT-SORT adapter + JSONL writer
│   │   ├── botsort_patched.py         #     strict-appearance-veto subclass
│   │   ├── ppe.py                     #     YOLOE with cached visual prompts
│   │   ├── demographics.py            #     MiVOLO classifier + per-track aggregator
│   │   ├── live_kpis.py               #     in-loop KPI overlay (roles, dwell, unattended)
│   │   ├── draw.py                    #     overlay primitives
│   │   ├── worker.py                  #     per-stream loop
│   │   └── run.py                     #     orchestrator (one worker thread per camera)
│   └── kpis/                          #   offline KPI computations off the JSONL tracks
│       ├── zones.py                   #     worker / customer zone categorization
│       ├── roles.py                   #     per-track role classifier
│       ├── dwell.py                   #     per-zone dwell visits
│       └── heatmap.py                 #     occupancy heatmap accumulator
├── scripts/                           # CLI entry points
│   ├── extract_reference_frames.py
│   ├── render_tracks.py               #   JSONL → annotated MP4 (no GPU needed)
│   ├── render_kpis.py                 #   tracks + zones + roles → KPI overlay MP4
│   ├── render_heatmap.py              #   tracks → occupancy heatmap PNG / overlay
│   └── run_kpis.py                    #   compute roles + dwell from tracks.jsonl
├── experiments/                       # one-off probes — not on the main pipeline
│   ├── experiment_yoloe.py            #   zero-shot YOLOE (no prompts)
│   ├── experiment_yoloe_text.py       #   YOLOE with text prompts
│   ├── experiment_yoloe_visual.py     #   YOLOE with visual prompts
│   └── yoloe_select_prompts.py        #   interactive crop selection → VPE cache (.pt)
├── data/
│   ├── weights/                       #   all model weights (gitignored)
│   ├── zones.json                     #   saved polygons (annotation tab output)
│   ├── reference_frames/<CAM>.png     #   first-frame still per camera
│   ├── tracks/<CAM>.jsonl             #   per-frame tracks (regenerable)
│   ├── annotated/<CAM>.mp4            #   annotated streams
│   └── kpis/<CAM>.*.json              #   roles + dwell + demographics outputs
├── videos/                            # source CCTV clips
└── task/                              # Python 3.12 venv (gitignored)
```

## Pipeline

End-to-end flow, from a fresh clone to the dashboards / overlays:

1. **Annotate regions** (`Zone Annotation` tab). Draw worker / customer
   polygons on each camera's reference frame. Names come from a fixed
   dropdown (`worker_area`, `customer_counter`, `customer_queue`) so the
   downstream classifier doesn't have to do free-text matching. Saved to
   `data/zones.json` in original frame coords. CAM-03 (dining) and CAM-04
   (kitchen) skip annotation entirely — they use a default-role override
   in `[kpis.camera_default_role]`.

2. **Process the streams in parallel** — `python -m app.pipeline.run`.
   The orchestrator spawns one worker thread per camera. A single
   `BatchedPersonDetector` is shared across all workers: each worker
   submits its current frame and blocks on its result, while a
   background dispatch thread coalesces concurrent submissions into one
   `model.predict([f1, f2, f3, f4])` batch. The GPU runs one bigger
   forward pass instead of four serial ones.

   Inside each worker, every frame:

   ```
   VideoCapture.read
        │
        ▼
   batched YOLO26l detector ──► sv.Detections (person, NMS-cleaned)
        │
        ▼
   BoT-SORT + OSNet ReID (strict-appearance veto) ──► persistent track_id
        │
        ├──► PPE pass (YOLOE + cached VPE)   ──► gloves / hairnet
        │       (CAM-01 inside worker_area,  CAM-04 whole frame)
        │
        ├──► MiVOLO age + gender              ──► per-track aggregator
        │       (customer-role tracks on CAM-01 only)
        │
        ▼
   LiveKpiOverlay.step:  roles · dwell timers · unattended-customer
        │
        ├──► JSONL line                        data/tracks/<CAM>.jsonl
        └──► annotated MP4 frame               data/annotated/<CAM>.mp4
   ```

3. **Offline KPI pass** — `python -m scripts.run_kpis` re-derives roles +
   per-zone dwell from the JSONL alone, so the live decisions can be
   audited against a deterministic offline classifier and the result is
   reproducible without the GPU.

4. **Visualisation** — three render paths consume the JSONL outputs and
   need no GPU:
   * `scripts.render_tracks` → boxes + IDs only.
   * `scripts.render_kpis` → full KPI overlay (zones, role-colored boxes,
     dwell, HUD).
   * `scripts.render_heatmap` → bottom-center foot points of customer
     tracks accumulated into a float32 grid, Gaussian-smoothed, overlaid
     on the camera reference frame.

5. **Inspect** — `streamlit run app/dashboard.py` exposes the same
   outputs through five tabs (Live Monitor, Live Processing, Analytics,
   Insights, Zone Annotation).

---

## Install

```bash
uv venv task --python 3.12
source task/bin/activate              
uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch
uv pip install -r requirements.txt
```

CUDA 13 quirk on Linux: PyTorch's `libnvrtc-builtins.so.13.0` may not be on
the loader path. Prepend it before running the pipeline:

```bash
export LD_LIBRARY_PATH=$(pwd)/task/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

Weights live under `data/weights/` and are referenced by `config.toml`:

- `yolo26l.pt` — person detector
- `osnet_x1_0_msmt17.pt` — ReID (auto-downloaded by boxmot)
- `mivolo_imdb.pth.tar` + `yolov8x_person_face.pt` — MiVOLO demographics
- `yoloe-11s-seg.pt` + `yoloe_vpe_kitchen.pt` — PPE (YOLOE backbone +
  cached visual prompt embeddings)

---

## Run

Everything is driven by **`config.toml`**. No CLI flags on the main entry
points — every threshold, model name, sampling rate, and per-camera toggle
lives in the config file.

```bash
# 1. Process the streams listed in [pipeline].cameras (empty = all four):
python -m app.pipeline.run

# 2. Compute roles + dwell from the JSONL tracks + saved zones:
python -m scripts.run_kpis

# 3. Render the KPI overlay MP4 (zones + role-colored boxes + dwell timers):
python -m scripts.render_kpis --cam CAM-01

# 4. Heatmap for the dining camera:
python -m scripts.render_heatmap --cam CAM-03

# 5. Launch the dashboard:
streamlit run app/dashboard.py        # http://localhost:8501
```

---

## Technical implementation

### Detection + tracking

Most KPIs are person-dependent, so the first decision was the detection +
tracking stack.

- **Detector**: surveyed the current YOLO benchmarks and went with the
  YOLO26 family for the speed-vs-accuracy tradeoff. **YOLO26l** is the
  current default (24.8 M params, mAP 55.0).
- **Tracker**: BoT-SORT is still the strong real-time baseline. Used it on
  several prior projects, and it pairs motion (Kalman) with an optional
  ReID branch — important here because the scenes have repeated
  occlusions (cashier behind a customer, two staff crossing in the
  kitchen).
- **ReID**: **OSNet x1.0 (MSMT17)** via boxmot. ~17 MB, FP16 inference,
  strong enough to recover stationary tracks across short occlusions.
- **Strict-appearance veto**: stock BoT-SORT lets IoU-only matches succeed
  even when ReID embeddings disagree, which causes the classic "stationary
  person hijacked by someone walking in front" failure mode. A subclass in
  `app/pipeline/botsort_patched.py` rejects an IoU-only match when the
  ReID cosine distance exceeds `appearance_thresh`.

This gives a stable, persistent `tracker_id` per person, which is the
foundation every downstream KPI sits on.

### KPIs

- **Zone-based roles** (`worker` / `customer`): added a Streamlit tab to
  draw the worker / customer polygons directly on each camera's
  reference frame; saved to `data/zones.json`. Those polygons feed both
  the live overlay and the offline classifier. A track is a worker only
  if its bbox center spent more frames inside a worker zone than inside
  customer zones; everything else — including tracks that never crossed
  any polygon — is a customer (so the partition is total, no "unknown"
  bucket). For cameras that are purely BOH or FOH the geometry is
  skipped entirely via `[kpis.camera_default_role]` (CAM-03 → all
  customers, CAM-04 → all workers).
- **Dwell time**: wall-clock based on system time deltas, not frame
  indices. Calculating from frames couples the KPI to FPS and processing
  jitter — wall-clock keeps it stable regardless of pipeline speed.
  Computed per region so different zones can carry independent dwell
  semantics.
- **Unattended customer**: state machine in `live_kpis.py` that counts
  seconds while there's a customer **at the counter** (counter-kind
  zone, e.g. `customer_counter`) AND no worker is in any worker zone.
  Customers in queue don't trigger — queueing without a worker present
  is normal foot traffic. Trips at `customer_not_served_threshold_s`
  (default 20 s). Only meaningful on cameras with both worker and
  counter zones (CAM-01, CAM-02).
- **Demographics**: MiVOLO-D1 on customer tracks only. Per-track
  sampling at `sample_gap_frames` (default 10 = 1 reading / sec at
  10 fps), finalized after `min_confident_frames` matched observations
  (default 10 → ≈ 10 s minimum visibility). Mean age → age band;
  sum-of-confidence vote for gender.
- **PPE (gloves + hairnet)**: YOLOE with a tiny visual prompt cache
  built from manually-selected crops (see
  `experiments/yoloe_select_prompts.py`). The cache is the single .pt
  file `data/weights/yoloe_vpe_kitchen.pt`. Runs on CAM-04 (kitchen)
  only — gloves / hairnets on the FOH cameras are noise (customers in
  street clothes, staff outside food-prep) and were dropped from the
  pipeline.
- **Occupancy heatmap**: bottom-center foot-points of customer tracks
  accumulated into a float32 grid, Gaussian-smoothed, colormap-overlaid
  on the camera's reference frame.

### Dashboard

A Streamlit app gives a single place to inspect everything:

- **Live Monitor** — 2×2 wall of the raw MP4s, browser-decoded.
- **Live Processing** — runs the GPU pipeline in-process and streams
  annotated frames back to the browser. Marked machine-dependent
  because the UI itself adds processing latency; the headless
  `app.pipeline.run` is the path for accurate timing.
- **Analytics** — replays the videos with overlays drawn from offline
  JSONL: zones, role-colored boxes, dwell timers, live counters,
  play / pause / restart / 0.5×–4× speed.
- **Insights** — heatmap + demographics summary per camera.
- **Zone Annotation** — auth-gated polygon editor. The dropdown offers
  only the canonical region names (`worker_area`, `customer_counter`,
  `customer_queue`) — see `app/utils/regions.py`. Anything starting with
  `worker_` is a *worker zone*; everything else is a *customer zone*.

---

## JSONL track schema

One record per frame, coordinates in original 1280×720 space:

```json
{"cam_id":"CAM-01","frame_idx":42,"ts_s":4.2,"frame_size":[1280,720],
 "tracks":[{"id":7,"bbox":[x1,y1,x2,y2],"conf":0.87,"class":"person"}],
 "ppe":[{"bbox":[...],"conf":0.41,"class":"glove"}]}
```

The `ppe` field is only populated on the cadence set by
`[ppe].every_n` and only for cameras in `[ppe].cameras`.

---

## Camera → KPI mapping (POC scope)

| Slot   | Camera       | Candidate KPIs                                       |
|--------|--------------|------------------------------------------------------|
| CAM-01 | Billing      | Queue · Unattended customer (at counter) · Demographics |
| CAM-02 | Counter      | Queue · Unattended customer (at counter) · Dwell        |
| CAM-03 | Dining       | Occupancy · Heatmap · Dwell                             |
| CAM-04 | Kitchen      | Gloves · Hairnet (PPE, whole frame, default=worker)     |

---

## Benchmark

Manual ground-truth labelling wasn't in scope for this POC. The
end-to-end pipeline produced **zero ID switches** across the supplied
clips after the strict-appearance ReID veto was enabled — that was the
single biggest stability win and the result that mattered most for the
downstream dwell / unattended KPIs.

Demographics: aggregate counts per camera + age-band / gender histogram
written to `data/kpis/<CAM>.demographics.json`; visually spot-checked
against the source clips.

---

<!--  -->
## Observations and assumptions 

- Only 4 of 9 camera views were provided — some KPIs (sanitation room,
  handwash) have no input video, so they're listed but not implemented.
- The cash drawer was never open across the supplied clips, so
  drawer-state KPIs (#3, #4) couldn't be tested even though the
  detection scaffolding could plug in here later.
- No clip naturally showed an unattended customer either — a custom
  verification script was added to synthesise this scenario so the
  `customer_not_served` timer could be validated.
- The Cameras were not synced, So Dining and Counter cameras have 0 shared ID which will effect customers counting and ID tracking across cameras.


---

## Future work / optimisation

**Inference path.**  Currently torch on a single GPU. The shared
detector batches across the 4 streams (`BatchedPersonDetector`) so the
GPU runs one forward pass per source tick instead of four, but PPE and
MiVOLO still run per-camera and the per-stream MiVOLO call on CAM-01 is
the new wall-clock bottleneck. Next steps:

- Export the three detectors (YOLO26, YOLOE, MiVOLO's internal YOLOv8x)
  to TensorRT and host them behind a Triton Inference Server. Triton
  gives you cross-process batching, parallel model instances on the
  same GPU, and proper queueing — all of which a single-process
  threaded approach cannot match once you scale past a handful of
  streams.
- Replace `cv2.VideoCapture` with a GStreamer pipeline. OpenCV's
  decode path is the single biggest source of latency on RTSP / 4-stream
  setups; GStreamer's hardware decoders (`nvv4l2decoder` on Jetson /
  `nvdec` on desktop) drop that to almost zero.
- Throttle MiVOLO globally (one prediction per N frames per camera, not
  per-track per N frames) so the customer-heavy camera doesn't bottleneck
  everyone — quick win on the way to TensorRT.
- Collapse per-person attribute heads (gloves, hairnet, age, gender)
  into a single multi-class object detector instead of running three
  separate models. That alone should cut total inference time by at
  least a third.

**Architecture.**  Streamlit is a pragmatic choice for a single-machine
demo but doesn't scale: the UI thread, the GPU pipeline, and the WebSocket
delivery all share one process. The natural next move is to split UI from
inference — a Next.js (or any thin) web frontend talking to a separate
inference service over WebSockets / gRPC, so the inference host can be
swapped, scaled, or moved off-prem without touching the UI.

## The Usage of GenAI
Since its 2026 and We now should make use of LLM in building projects, I've used claude code to enable me to build more and deliver better analytics, utils and visuals. 
The main use of claude was for UI better steamlit visualizations, and for helping writing quick analytics, helper functions. 

But the main logic [custom botsort, heatmap, algorithms, architecture, models selection and implementation ]  was my work completely [new or from old personal work]. 

