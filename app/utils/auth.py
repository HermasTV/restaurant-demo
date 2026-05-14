"""Hardcoded demo auth.

Credentials are intentionally trivial (123 / 123) — this gates the annotation
tab so the live monitor stays publicly viewable. Replace with a real auth
provider before any non-demo deployment.
"""
from __future__ import annotations

import streamlit as st

DEMO_USERNAME = "123"
DEMO_PASSWORD = "123"
SESSION_KEY = "authed"


def is_authed() -> bool:
    return bool(st.session_state.get(SESSION_KEY))


def logout() -> None:
    st.session_state[SESSION_KEY] = False


def login_form() -> None:
    st.markdown("#### Sign in to annotate zones")
    st.caption("Demo credentials: `123` / `123`")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", value="")
        password = st.text_input("Password", value="", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            st.session_state[SESSION_KEY] = True
            st.rerun()
        else:
            st.error("Invalid credentials.")
