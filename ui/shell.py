"""The app shell: page registration, stepper and dispatch to the selected station.

``app.py`` calls :func:`main`. Tests call ``main(force_key=...)`` through Streamlit's ``AppTest`` to render one station
headlessly with the full navigation registered (page links need that).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import streamlit as st

from graticule import APP_NAME
from ui import components
from ui.pages import assay, bench, fit, logbook, measure, probe, record, sample, sweep
from ui.stations import ALL_STATIONS, PAGE_OBJECTS

ICON = Path(__file__).resolve().parent.parent / "static" / "graticule-icon.svg"

RENDERERS: dict[str, Callable[[], None]] = {
    "sample": sample.render,
    "fit": fit.render,
    "measure": measure.render,
    "probe": probe.render,
    "assay": assay.render,
    "sweep": sweep.render,
    "record": record.render,
    "logbook": logbook.render,
    "bench": bench.render,
}


def register_pages() -> list[st.Page]:
    """Create one ``st.Page`` per station and publish them in :data:`ui.stations.PAGE_OBJECTS`."""
    pages = []
    for station in ALL_STATIONS:
        page = st.Page(
            RENDERERS[station.key],
            title=station.label,
            url_path=station.url_path,
            default=station.key == "sample",
        )
        PAGE_OBJECTS[station.key] = page
        pages.append(page)
    return pages


def main(force_key: str | None = None) -> None:
    """Configure the page, draw the stepper and run the selected station (or ``force_key`` in tests)."""
    st.set_page_config(page_title=APP_NAME, page_icon=str(ICON), layout="wide", initial_sidebar_state="collapsed")
    st.logo(str(ICON), size="large")
    pages = register_pages()
    current = st.navigation(pages, position="hidden")
    if force_key is not None:
        current_key = force_key
    else:
        current_key = next(
            (s.key for s in ALL_STATIONS if PAGE_OBJECTS[s.key].url_path == current.url_path), "sample"
        )
    components.inject_css()
    components.stepper(current_key)
    if force_key is not None:
        RENDERERS[force_key]()
    else:
        current.run()
