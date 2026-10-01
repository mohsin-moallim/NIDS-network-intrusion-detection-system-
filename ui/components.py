"""Shared building blocks for every page: theme mode, extra CSS, the station stepper and small notes."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from numbers import Integral

import pandas as pd
import streamlit as st

from graticule import theme
from graticule.theme import Mode
from ui import state
from ui.stations import BY_KEY, PAGE_OBJECTS, STATIONS, UTILITIES, Station


def shown_scores(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """A copy of ``frame`` whose score ``columns`` (those present) hold what four decimals should print.

    Table formats round, so a reading of 0.99998 would print as a perfect 1.0000; such values are held at 0.9999
    (:func:`graticule.theme.shown_score`). Use the copy for display only: sort and compare on the true values.
    """
    out = frame.copy()
    for name in columns:
        if name in out.columns and pd.api.types.is_numeric_dtype(out[name]):
            out[name] = theme.shown_scores(out[name].to_numpy(dtype="float64", na_value=float("nan")))
    return out


def current_mode() -> Mode:
    """The viewer's theme mode. Streamlit can report it late on first load, so light is the safe default."""
    try:
        kind = st.context.theme.type
    except AttributeError:
        kind = None
    return "dark" if kind == "dark" else "light"


def inject_css() -> None:
    """Add the small amount of CSS that config.toml cannot express (station strip, tabular numbers, labels,
    reading cards)."""
    p = theme.palette(current_mode())
    st.html(
        f"""<style>
        html, body, [class*="st-"] {{ font-variant-numeric: tabular-nums; }}
        h1, h2, h3 {{ letter-spacing: 0.01em; }}
        [class*="st-key-stn_"] a {{ border-radius: 4px; padding-top: 2px; padding-bottom: 2px; }}
        [class*="st-key-stn_"] a p {{
            font-family: '{theme.FONT_HEADING}', sans-serif; text-transform: uppercase;
            letter-spacing: 0.06em; font-size: 0.8rem; line-height: 1.25; white-space: normal;
        }}
        [class*="st-key-stn_cur_"] {{ border-bottom: 3px solid {p.secondary}; }}
        [class*="st-key-g_cm_"] {{ overflow-x: auto; }}
        [class*="st-key-g_stepper"] [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important;
            overflow-x: auto; scrollbar-width: thin; }}
        [class*="st-key-g_stepper"] [data-testid="stColumn"] {{ min-width: 5.5rem !important;
            flex: 1 0 auto !important; width: auto !important; }}
        [class*="st-key-stn_cur_"] a p {{ font-weight: 700; }}
        .g-purpose {{ color: {p.muted}; margin-top: -0.6rem; margin-bottom: 0.8rem; }}
        .g-note {{ border-left: 3px solid {p.secondary}; padding: 0.4rem 0.8rem; background: {p.surface};
                   border-radius: 0; margin: 0.4rem 0 0.8rem 0; }}
        .g-chip {{ display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 0.85rem;
                   border: 1px solid currentColor; font-weight: 600; margin-right: 4px; }}
        .g-benign {{ color: {p.benign}; }}
        .g-attack {{ color: {p.attack}; }}
        .g-alert {{ color: {p.text}; border-color: {p.warning} !important; background: {p.warning_tint}; }}
        .g-mono {{ font-family: '{theme.FONT_MONO}', monospace; }}
        .g-chips {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 0.1rem 0 0.6rem 0; }}
        .g-tag {{ color: {p.muted}; border-color: {p.border} !important; font-weight: 500; }}
        .g-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px;
                    margin: 0.2rem 0 1rem 0; }}
        .g-card {{ border: 1px solid {p.border}; background: {p.surface}; border-radius: 4px; padding: 8px 10px; }}
        .g-card-label {{ color: {p.muted}; font-size: 0.8rem; }}
        .g-card-value {{ font-family: '{theme.FONT_MONO}', monospace; font-size: 1.35rem; line-height: 1.3; }}
        .g-card-note {{ color: {p.muted}; font-size: 0.75rem; }}
        </style>"""
    )


def _station_link(station: Station, current_key: str, suffix: str = "") -> None:
    """Draw one station link inside a keyed container so CSS can mark the current one."""
    prefix = "stn_cur_" if station.key == current_key else "stn_"
    page = PAGE_OBJECTS.get(station.key)
    label = station.label
    if state.is_done(station.key):
        label = f"{label} {theme.GLYPH_DONE}"
    with st.container(key=f"{prefix}{station.key}{suffix}"):
        if page is not None:
            st.page_link(page, label=label, width="stretch")
        else:
            st.caption(label)


def stepper(current_key: str, *, suffix: str = "") -> None:
    """The measuring-scale stepper: seven numbered stations, then the Logbook and Bench utilities.

    On narrow screens the strip stays one row and scrolls sideways instead of stacking into a tall list. A background
    fit that has finished since the last rerun is adopted first, so the 02 Fit tick is right on every page.
    ``suffix`` is appended to the strip's element keys, so the shell can draw it a second time in one run (into the
    same placeholder) when a station earned its tick while it was drawn.
    """
    state.collect_finished_job()
    with st.container(key=f"g_stepper{suffix}"):
        cols = st.columns([1] * len(STATIONS) + [0.35] + [1] * len(UTILITIES), gap="small",
                          vertical_alignment="bottom")
        for col, station in zip(cols[: len(STATIONS)], STATIONS):
            with col:
                _station_link(station, current_key, suffix)
        for col, station in zip(cols[len(STATIONS) + 1 :], UTILITIES):
            with col:
                _station_link(station, current_key, suffix)
    st.divider()


def station_header(key: str) -> None:
    """Heading plus a one-line note on what the station does."""
    station = BY_KEY[key]
    st.header(station.label if station.number else station.title, anchor=False)
    st.markdown(f'<p class="g-purpose">{html.escape(station.purpose)}</p>', unsafe_allow_html=True)


def needs(message: str, station_key: str) -> None:
    """Explain a missing prerequisite and link to the station that provides it (never a dead end)."""
    st.markdown(f'<div class="g-note">{html.escape(message)}</div>', unsafe_allow_html=True)
    page = PAGE_OBJECTS.get(station_key)
    if page is not None:
        st.page_link(page, label=f"Go to {BY_KEY[station_key].label}", icon=":material/arrow_forward:")


def chips(labels: Sequence[str]) -> None:
    """A row of small neutral badges for facts about a result (mode, feature set, a cap that applied...)."""
    if labels:
        spans = "".join(f'<span class="g-chip g-tag">{html.escape(str(label))}</span>' for label in labels)
        st.markdown(f'<div class="g-chips">{spans}</div>', unsafe_allow_html=True)


def verdict_chip(label: str, alert: bool = False) -> str:
    """HTML for a verdict pill that never relies on colour alone ("○ Normal", "◆ DoS Hulk", "▲ Alert")."""
    text = html.escape(theme.verdict_text(label))
    if alert:
        return f'<span class="g-chip g-alert">{theme.GLYPH_ALERT} Alert</span><span class="g-chip g-attack">{text}</span>'
    css = "g-benign" if text.startswith(theme.GLYPH_NORMAL) else "g-attack"
    return f'<span class="g-chip {css}">{text}</span>'


def reading_cards(rows: Sequence[Mapping[str, object]]) -> None:
    """Headline readings as a grid of small cards: a label, the value in the mono face, and a short note.

    Each row needs "Reading" and "Value" and may carry "Note" (the shape of
    :meth:`graticule.data.prepare.PreparedDataset.summary_rows`). Integers get thousands separators; floats are shown
    as given, so format them first when a fixed number of decimals matters. Styling comes from :func:`inject_css`.
    """
    cells = []
    for row in rows:
        value = row["Value"]
        shown = f"{int(value):,}" if isinstance(value, Integral) and not isinstance(value, bool) else str(value)
        cells.append(
            '<div class="g-card">'
            f'<div class="g-card-label">{html.escape(str(row["Reading"]))}</div>'
            f'<div class="g-card-value">{html.escape(shown)}</div>'
            f'<div class="g-card-note">{html.escape(str(row.get("Note", "") or ""))}</div></div>'
        )
    st.markdown(f'<div class="g-cards">{"".join(cells)}</div>', unsafe_allow_html=True)
