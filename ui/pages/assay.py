"""05 Assay station."""

from __future__ import annotations

import streamlit as st

from ui import components


def render() -> None:
    """Draw the 05 Assay station."""
    components.station_header("assay")
    st.caption("This station is still being built: batch scoring arrives in build phase 8.")
