"""07 Record station."""

from __future__ import annotations

import streamlit as st

from ui import components


def render() -> None:
    """Draw the 07 Record station."""
    components.station_header("record")
    st.caption("This station is still being built: PDF and CSV exports arrives in build phase 8.")
