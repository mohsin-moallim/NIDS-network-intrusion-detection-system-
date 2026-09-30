"""01 Sample station."""

from __future__ import annotations

import streamlit as st

from ui import components


def render() -> None:
    """Draw the 01 Sample station."""
    components.station_header("sample")
    st.caption("This station is still being built: the data loader, cleaner and sampler arrives in build phase 1.")
