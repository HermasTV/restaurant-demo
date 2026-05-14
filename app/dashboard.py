"""Restaurant CCTV monitoring dashboard (Streamlit entry point).

Tabs:
  - Live Monitor:    2x2 video wall, public.
  - Zone Annotation: draw / save polygons per camera (demo auth: 123 / 123).

Run:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Allow `python -m streamlit run app/dashboard.py` style invocation by ensuring
# the project root is on sys.path so `app.*` imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app import analytics, annotation, insights, live_monitor, live_processing  # noqa: E402


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100% !important; }
        header[data-testid="stHeader"] { background: transparent; }
        .stApp { background: #0d1117; color: #e6edf3; }

        .topbar {
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 16px; margin-bottom: 12px;
            background: #161b22; border: 1px solid #2a313c; border-radius: 8px;
        }
        .brand { display: flex; align-items: center; gap: 10px; }
        .brand h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: 0.2px; }
        .logo-dot { width: 10px; height: 10px; border-radius: 50%;
                    background: #58a6ff; box-shadow: 0 0 8px #58a6ff; }
        .meta { display: flex; align-items: center; gap: 16px;
                font-size: 12px; color: #8b949e; }
        .live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                    background: #ff4d4f; margin-right: 6px;
                    animation: pulse 1.4s infinite ease-in-out; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

        .tile-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 12px; background: #1c232c;
            border: 1px solid #2a313c; border-bottom: none;
            border-radius: 8px 8px 0 0;
        }
        .cam-name { font-weight: 600; font-size: 13px; color: #e6edf3; }
        .cam-tag  { font-size: 11px; color: #8b949e; }

        .tile-footer {
            display: flex; flex-wrap: wrap; gap: 4px;
            padding: 8px 10px; background: #1c232c;
            border: 1px solid #2a313c; border-top: none;
            border-radius: 0 0 8px 8px;
            margin-bottom: 10px;
        }
        .kpi-chip {
            font-size: 10px; color: #8b949e;
            background: #21262d; padding: 2px 8px;
            border-radius: 10px; border: 1px solid #2a313c;
        }

        .stVideo { margin: 0 !important; }
        .stVideo > video { border-radius: 0 !important;
                           border-left: 1px solid #2a313c;
                           border-right: 1px solid #2a313c;
                           background: #000; }

        button[kind="primary"] { background: #1f6feb; border-color: #1f6feb; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_topbar() -> None:
    now = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <span class="logo-dot"></span>
            <h1>Restaurant CCTV Monitor</h1>
          </div>
          <div class="meta">
            <span>{now}</span>
            <span><span class="live-dot"></span>LIVE</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Restaurant CCTV Monitor",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()
    _render_topbar()

    monitor_tab, live_tab, analytics_tab, insights_tab, annot_tab = st.tabs(
        ["Live Monitor", "Live Processing", "Analytics", "Insights", "Zone Annotation"]
    )
    with monitor_tab:
        live_monitor.render()
    with live_tab:
        live_processing.render()
    with analytics_tab:
        analytics.render()
    with insights_tab:
        insights.render()
    with annot_tab:
        annotation.render()


if __name__ == "__main__":
    main()
