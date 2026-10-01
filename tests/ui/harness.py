"""Shared AppTest harness: one headless session that can move between stations, plus small driving helpers.

``station_app`` is the script body handed to ``AppTest.from_function``. It renders the station named by the session
key ``"_station"`` (falling back to its argument), so a test switches stations by setting
``at.session_state["_station"]`` and rerunning, while the session (sample, run id, widget values) carries over,
exactly as when a viewer clicks through the stepper.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

STATION = "_station"


def station_app(key: str) -> None:
    """AppTest script body: render the station in ``st.session_state["_station"]`` (default ``key``)."""
    import streamlit as st

    from ui.shell import main

    main(force_key=st.session_state.get("_station", key))


def new_app(key: str = "sample", timeout: float = 120) -> AppTest:
    """A fresh (not yet run) AppTest session that starts at station ``key``."""
    return AppTest.from_function(station_app, args=(key,), default_timeout=timeout)


def goto(at: AppTest, key: str) -> AppTest:
    """Move the session to station ``key`` and rerun."""
    at.session_state[STATION] = key
    return at.run()


def errors(at: AppTest) -> list[str]:
    """Exception texts raised by the last run (empty when it ran cleanly)."""
    return [e.value for e in at.exception]


def draw_synthetic_sample(at: AppTest, flows: int = 2_000, budget: int = 1_600) -> AppTest:
    """Draw a small synthetic sample through the 01 Sample form (no data folder is configured in tests)."""
    goto(at, "sample")
    at.number_input(key="smp_syn_flows").set_value(flows)
    at.number_input(key="smp_syn_budget").set_value(budget)
    return at.button(key="smp_syn_draw").click().run()


@pytest.fixture
def fresh_caches(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Empty every ``st.cache_resource`` cache (run registry, file caches) around a test; shrink the models.

    ``GRATICULE_TEST_PROFILE=1`` is the hidden hook that makes 02 Fit use ``profile="test"`` models.
    """
    monkeypatch.setenv("GRATICULE_TEST_PROFILE", "1")
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
