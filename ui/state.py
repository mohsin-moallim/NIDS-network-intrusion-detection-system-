"""Session and process state for the UI.

Pages never compute expensive results themselves: they read what the explicit actions (Prepare, Fit, Score…) stored
here. Finished fit runs are also kept in a process-wide registry so a browser refresh does not lose them.
"""

from __future__ import annotations

import threading
from typing import Any

import streamlit as st

from graticule.settings import AppSettings, load_settings, save_settings

SETTINGS = "g_settings"
DONE = "g_done"
LAST_RUN_ID = "g_last_run_id"


def settings() -> AppSettings:
    """Return this session's settings, loading them from disk the first time."""
    if SETTINGS not in st.session_state:
        st.session_state[SETTINGS] = load_settings()
    return st.session_state[SETTINGS]


def update_settings(new: AppSettings) -> AppSettings:
    """Validate, store and persist new settings; returns the validated copy."""
    clean = new.validated()
    save_settings(clean)
    st.session_state[SETTINGS] = clean
    return clean


def mark_done(station_key: str) -> None:
    """Record that a station produced a result in this session (drives the ticks on the stepper)."""
    done: set[str] = st.session_state.setdefault(DONE, set())
    done.add(station_key)


def is_done(station_key: str) -> bool:
    """True when ``station_key`` has produced a result in this session."""
    return station_key in st.session_state.get(DONE, set())


class RunRegistry:
    """Process-wide store of finished fit runs, keyed by run id, shared by every browser tab of this app."""

    def __init__(self, capacity: int = 3) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, Any] = {}
        self._capacity = capacity

    def put(self, run_id: str, run: Any) -> None:
        """Keep ``run``; the oldest runs are dropped beyond the capacity to bound memory."""
        with self._lock:
            self._runs[run_id] = run
            while len(self._runs) > self._capacity:
                self._runs.pop(next(iter(self._runs)))

    def get(self, run_id: str | None) -> Any | None:
        """Return the run stored under ``run_id``, or ``None``."""
        if run_id is None:
            return None
        with self._lock:
            return self._runs.get(run_id)

    def latest_id(self) -> str | None:
        """Id of the most recently stored run, or ``None`` when empty."""
        with self._lock:
            return next(reversed(self._runs), None) if self._runs else None


@st.cache_resource(show_spinner=False)
def run_registry() -> RunRegistry:
    """The single process-wide :class:`RunRegistry`."""
    return RunRegistry()
