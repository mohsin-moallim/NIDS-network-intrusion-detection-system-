"""Shared AppTest harness: one headless session that can move between stations, plus small driving helpers.

``station_app`` is the script body handed to ``AppTest.from_function``. It renders the station named by the session
key ``"_station"`` (falling back to its argument), so a test switches stations by setting
``at.session_state["_station"]`` and rerunning, while the session (sample, run id, widget values) carries over,
exactly as when a viewer clicks through the stepper.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

STATION = "_station"
# Element-tree attributes of the widget kinds :func:`touch_every_widget` knows how to change.
WIDGET_KINDS: tuple[str, ...] = ("checkbox", "toggle", "radio", "selectbox", "multiselect", "number_input", "slider",
                                 "select_slider", "text_input", "text_area")


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
    """Draw a small synthetic sample through the 01 Sample form (no data folder is configured in tests).

    The values must lie within the inputs' limits: Streamlit drops a value out of range and keeps the default (40,000
    generated flows), which would quietly make every such test far slower.
    """
    goto(at, "sample")
    for key, value in (("smp_syn_flows", flows), ("smp_syn_budget", budget)):
        widget = at.number_input(key=key)
        if not widget.min <= value <= widget.max:
            raise ValueError(f"{key}={value:,} is outside {widget.min:,}..{widget.max:,}")
        widget.set_value(value)
    return at.button(key="smp_syn_draw").click().run()


def different_value(widget: object, kind: str) -> object | None:
    """A valid value for ``widget`` that differs from its current one, or None when there is none."""
    value = widget.value  # type: ignore[attr-defined]
    if kind in ("checkbox", "toggle"):
        return not value
    if kind in ("radio", "selectbox", "select_slider"):
        options = list(widget.options)  # type: ignore[attr-defined]
        if len(options) < 2 or isinstance(value, (list, tuple)):
            return None
        current = widget.format_func(value) if value is not None else None  # type: ignore[attr-defined]
        return next((o for o in options if o != current), None)
    if kind == "multiselect":
        if value:
            return list(value)[:-1]
        options = list(widget.options)  # type: ignore[attr-defined]
        return [options[0]] if options else None
    if kind in ("number_input", "slider"):
        if isinstance(value, (list, tuple)) or value is None:
            return None
        step = widget.step or 1  # type: ignore[attr-defined]
        high, low = widget.max, widget.min  # type: ignore[attr-defined]
        candidate = value + step
        if high is not None and candidate > high:
            candidate = value - step
        if low is not None and candidate < low:
            return None
        return round(candidate, 6) if isinstance(candidate, float) else candidate
    if kind in ("text_input", "text_area"):
        return f"{value or ''}x"
    return None


def touch_every_widget(at: AppTest, after_each: Callable[[str], None]) -> list[str]:
    """Change each widget on the current page once, rerunning after each change; returns what was touched
    (``"kind:key"`` labels). Buttons are never pressed.

    Widgets are addressed by kind and position, re-read after every rerun, so the page may redraw freely. Disabled
    widgets are skipped (a viewer cannot change them either). The run must stay free of exceptions.
    """
    touched: list[str] = []
    counts = {kind: len(getattr(at, kind)) for kind in WIDGET_KINDS}
    for kind, count in counts.items():
        for index in range(count):
            widgets = getattr(at, kind)
            if index >= len(widgets):
                break
            widget = widgets[index]
            if getattr(widget.proto, "disabled", False):
                continue
            new = different_value(widget, kind)
            if new is None:
                continue
            widget.set_value(new)
            at.run()
            label = f"{kind}:{widget.key or getattr(widget, 'label', index)}"
            assert not errors(at), (label, errors(at))
            touched.append(label)
            after_each(label)
    return touched


@pytest.fixture
def fresh_caches(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Empty every ``st.cache_resource`` cache (run registry, file caches) around a test; shrink the models.

    ``GRATICULE_TEST_PROFILE=1`` is the hidden hook that makes 02 Fit use ``profile="test"`` models.
    """
    monkeypatch.setenv("GRATICULE_TEST_PROFILE", "1")
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
