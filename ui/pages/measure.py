"""03 Measure station."""

from __future__ import annotations

import streamlit as st

from ui import components


def render() -> None:
    """Draw the 03 Measure station."""
    components.station_header("measure")
    st.caption("This station is still being built: side-by-side evaluation arrives in build phase 5.")
