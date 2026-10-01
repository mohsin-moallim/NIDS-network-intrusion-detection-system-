"""The app shell: page registration, stepper and dispatch to the selected station.

``app.py`` calls :func:`main`. Tests call ``main(force_key=...)`` through Streamlit's ``AppTest`` to render one station
headlessly with the full navigation registered (page links need that).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import streamlit as st

from graticule import APP_NAME, TAGLINE, __version__
from ui import components, state
from ui.pages import assay, bench, fit, logbook, measure, probe, record, sample, sweep
from ui.stations import ALL_STATIONS, PAGE_OBJECTS

ICON = Path(__file__).resolve().parent.parent / "static" / "graticule-icon.svg"
#: The "About" text of the ⋮ menu. An entry of the app's own keeps that menu on screen under the minimal toolbar
#: (``.streamlit/config.toml``), and with it the Theme choice: light (paper), dark (graphite) or the system setting.
ABOUT = "\n\n".join((
    f"**{APP_NAME}** {__version__}: {TAGLINE}",
    "Light (paper) and dark (graphite) themes are chosen under Theme in the ⋮ menu; the PDF record always uses "
    "the light one.",
    "MIT licence, © 2026 mohsin-moallim.",
))

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
    """Configure the page, draw the stepper and run the selected station (or ``force_key`` in tests).

    The stepper is drawn first, into a placeholder, so it shows while the station draws. A station can earn its tick
    while it draws (a verdict read at 04 Probe, a finished Assay or record build adopted, a Sweep tick), after the
    strip was drawn; the strip is then drawn again into the same placeholder, so the tick shows in that very run
    rather than after the viewer's next click.
    """
    st.set_page_config(page_title=APP_NAME, page_icon=str(ICON), layout="wide", initial_sidebar_state="collapsed",
                       menu_items={"About": ABOUT})
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
    strip = st.empty()
    with strip.container():
        components.stepper(current_key)
    ticked = state.done_stations()
    if force_key is not None:
        RENDERERS[force_key]()
    else:
        current.run()
    if state.done_stations() != ticked:
        with strip.container():
            components.stepper(current_key, suffix="_now")
