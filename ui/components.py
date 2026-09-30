"""Shared building blocks for every page: theme mode, extra CSS, the station stepper and small notes."""

from __future__ import annotations

import html

import streamlit as st

from graticule import theme
from graticule.theme import Mode
from ui import state
from ui.stations import BY_KEY, PAGE_OBJECTS, STATIONS, UTILITIES, Station


def current_mode() -> Mode:
    """The viewer's theme mode. Streamlit can report it late on first load, so light is the safe default."""
    try:
        kind = st.context.theme.type
    except AttributeError:
        kind = None
    return "dark" if kind == "dark" else "light"


def inject_css() -> None:
    """Add the small amount of CSS that config.toml cannot express (station strip, tabular numbers, labels)."""
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
        </style>"""
    )


def _station_link(station: Station, current_key: str) -> None:
    """Draw one station link inside a keyed container so CSS can mark the current one."""
    prefix = "stn_cur_" if station.key == current_key else "stn_"
    page = PAGE_OBJECTS.get(station.key)
    label = station.label
    if state.is_done(station.key):
        label = f"{label} {theme.GLYPH_DONE}"
    with st.container(key=f"{prefix}{station.key}"):
        if page is not None:
            st.page_link(page, label=label, width="stretch")
        else:
            st.caption(label)


def stepper(current_key: str) -> None:
    """The measuring-scale stepper: seven numbered stations, then the Logbook and Bench utilities."""
    cols = st.columns([1] * len(STATIONS) + [0.35] + [1] * len(UTILITIES), gap="small", vertical_alignment="bottom")
    for col, station in zip(cols[: len(STATIONS)], STATIONS):
        with col:
            _station_link(station, current_key)
    for col, station in zip(cols[len(STATIONS) + 1 :], UTILITIES):
        with col:
            _station_link(station, current_key)
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


def verdict_chip(label: str, alert: bool = False) -> str:
    """HTML for a verdict pill that never relies on colour alone ("○ Normal", "◆ DoS Hulk", "▲ Alert")."""
    text = html.escape(theme.verdict_text(label))
    if alert:
        return f'<span class="g-chip g-alert">{theme.GLYPH_ALERT} Alert</span><span class="g-chip g-attack">{text}</span>'
    css = "g-benign" if text.startswith(theme.GLYPH_NORMAL) else "g-attack"
    return f'<span class="g-chip {css}">{text}</span>'
