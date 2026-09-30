"""Logbook station."""

from __future__ import annotations

import streamlit as st

from ui import components


def render() -> None:
    """Draw the Logbook station."""
    components.station_header("logbook")
    st.caption("This station is still being built: saved channel sets and run history arrives in build phase 4.")
