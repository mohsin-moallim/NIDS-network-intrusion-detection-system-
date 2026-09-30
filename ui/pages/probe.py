"""04 Probe station."""

from __future__ import annotations

import streamlit as st

from ui import components


def render() -> None:
    """Draw the 04 Probe station."""
    components.station_header("probe")
    st.caption("This station is still being built: single-flow analysis arrives in build phase 6.")
