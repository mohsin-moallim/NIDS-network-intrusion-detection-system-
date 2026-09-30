"""06 Sweep station."""

from __future__ import annotations

import streamlit as st

from ui import components


def render() -> None:
    """Draw the 06 Sweep station."""
    components.station_header("sweep")
    st.caption("This station is still being built: the live traffic simulation arrives in build phase 7.")
