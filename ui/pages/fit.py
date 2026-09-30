"""02 Fit station."""

from __future__ import annotations

import streamlit as st

from ui import components


def render() -> None:
    """Draw the 02 Fit station."""
    components.station_header("fit")
    st.caption("This station is still being built: model fitting with live progress arrives in build phase 3.")
