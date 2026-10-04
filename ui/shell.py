"""The app shell: page registration, stepper and dispatch to the selected station.

``app.py`` calls :func:`main`. Tests call ``main(force_key=...)`` through Streamlit's ``AppTest`` to render one station
headlessly with the full navigation registered (page links need that).

Addresses. Every station has its own address (``/sample``, ``/fit`` ... ``/bench``, from
:attr:`ui.stations.Station.url_path`). Streamlit always serves its default page at the bare root and ignores that
page's own ``url_path``, so the default page is a hidden one of the shell's (:data:`HOME_TITLE`) rather than 01 Sample:
otherwise ``/sample`` would be an unknown address and Streamlit would show its "page not found" notice before falling
back. On the first run of a session the root hands over to ``/sample`` with ``st.switch_page`` before anything is
drawn, so the address bar names the station and nothing flashes on screen. The browser's Back button can still return
to the root later in the session; the root then draws 01 Sample in place instead of handing over again, because a
second hand-over would push ``/sample`` onto the history once more and Back could never leave the app.
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
#: Title of the hidden page at the bare root address (it never appears in the stepper).
HOME_TITLE = APP_NAME
#: The station the bare root address opens.
HOME_STATION = "sample"
#: Session key set by a session's first run; only that run hands the root address over to 01 Sample's own address.
SHELL_SEEN = "g_shell_seen"

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
    """Create the hidden root page plus one ``st.Page`` per station; publish the stations in
    :data:`ui.stations.PAGE_OBJECTS`.

    The root page comes first and is the default (Streamlit serves it at ``/``); every station keeps its own
    ``url_path``, so each one answers at its own address.
    """
    home = st.Page(RENDERERS[HOME_STATION], title=HOME_TITLE, url_path="", default=True, visibility="hidden")
    pages = [home]
    for station in ALL_STATIONS:
        page = st.Page(RENDERERS[station.key], title=station.label, url_path=station.url_path)
        PAGE_OBJECTS[station.key] = page
        pages.append(page)
    return pages


def station_key_of(page: st.Page) -> str:
    """The key of the station ``page`` shows; the hidden root page shows :data:`HOME_STATION`."""
    return next((s.key for s in ALL_STATIONS if PAGE_OBJECTS[s.key].url_path == page.url_path), HOME_STATION)


def _carried_query_params() -> dict[str, list[str]]:
    """The address's query parameters, for the hand-over to keep (``st.switch_page`` clears them otherwise)."""
    return {key: st.query_params.get_all(key) for key in st.query_params}


def main(force_key: str | None = None) -> None:
    """Configure the page, draw the stepper and run the selected station (or ``force_key`` in tests).

    On a session's first run, the bare root address hands over to 01 Sample's own address before anything is drawn
    (see the module notes); later visits to the root draw 01 Sample in place.

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
    first_run = not st.session_state.get(SHELL_SEEN, False)
    st.session_state[SHELL_SEEN] = True
    if force_key is None and first_run and current.url_path == "":
        st.switch_page(PAGE_OBJECTS[HOME_STATION], query_params=_carried_query_params())
    current_key = force_key if force_key is not None else station_key_of(current)
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
