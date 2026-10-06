"""Shared AppTest harness: one headless session that can move between stations, plus small driving helpers.

``station_app`` is the script body handed to ``AppTest.from_function``. It renders the station named by the session
key ``"_station"`` (falling back to its argument), so a test switches stations by setting
``at.session_state["_station"]`` and rerunning, while the session (sample, run id, widget values) carries over,
exactly as when a viewer clicks through the stepper.

Samples and fits that tests only start from are made once per test session (:func:`drawn_sample`,
:func:`fit_synthetic`): the 01 Sample form and the 02 Fit form have their own tests, and the end-to-end tests
(``test_no_retrain.py``, the first fit of ``test_fit_page.py``) still go through both.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.helpers import shared_fit, shared_sample

STATION = "_station"
# Element-tree attributes of the widget kinds :func:`touch_every_widget` knows how to change.
WIDGET_KINDS: tuple[str, ...] = ("checkbox", "toggle", "radio", "selectbox", "multiselect", "pills", "number_input",
                                 "slider", "select_slider", "text_input", "text_area")


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


def drawn_sample(flows: int = 2_000, budget: int = 1_600, seed: int = 42) -> Any:
    """The sample 01 Sample draws from ``flows`` generated flows with a row budget of ``budget`` and every other
    option at its default (no data folder is configured in tests), prepared once per test session.

    Tests whose subject comes after 01 Sample start from it (:func:`app_with_sample`) instead of pressing Draw
    sample each time; ``test_fit_page.py`` checks that a draw through the form gives this very sample. The object is
    shared between tests, so it must never be changed.
    """
    from nids.data.prepare import DataRequest

    return shared_sample(DataRequest(source="synthetic", synthetic_flows=flows, row_budget=budget, seed=seed))


def app_with_sample(prepared: object, key: str = "sample", timeout: float = 120) -> AppTest:
    """A fresh (not yet run) session at station ``key`` that already holds ``prepared`` as its 01 Sample (the
    station ticked), exactly as right after Draw sample."""
    from ui import state

    at = new_app(key, timeout)
    at.session_state[state.PREPARED] = prepared
    at.session_state[state.DONE] = {"sample"}
    return at


def app_with_run(run: object, prepared: object | None = None, key: str = "sample",
                 timeout: float = 120) -> AppTest:
    """A fresh (not yet run) session at station ``key`` that already holds ``run`` as its current run, and
    ``prepared`` as its 01 Sample, exactly as after drawing the sample and pressing Fit (both stations ticked).

    Tests of the later stations use it to skip drawing and fitting through the pages (02 Fit has its own tests),
    which keeps them quick; the run is fitted once per module with ``profile="test"`` models.
    """
    from ui import state

    at = new_app(key, timeout)
    state.run_registry().put(run.run_id, run)  # type: ignore[attr-defined]
    at.session_state[state.LAST_RUN_ID] = run.run_id  # type: ignore[attr-defined]
    at.session_state[state.RUN] = run
    if prepared is not None:
        at.session_state[state.PREPARED] = prepared
    at.session_state[state.DONE] = {"sample", "fit"} if prepared is not None else {"fit"}
    return at


def fit_synthetic(flows: int = 2_000, budget: int = 1_200, seed: int = 42, **request: object) -> tuple[Any, Any]:
    """A synthetic sample (``flows`` generated, ``budget`` kept) and a ``profile="test"`` fit of it, as 01 Sample
    and 02 Fit make them; ``request`` sets other :class:`~nids.models.train.TrainRequest` fields (mode,
    channels...). Returns (prepared sample, run).

    Each distinct fit is made once per test session (:func:`tests.helpers.shared_fit`); every call returns a fresh
    copy of the run, so the readings, results and bundle path a test leaves on it stay with that test.
    """
    from nids.models.train import TrainRequest

    prepared = drawn_sample(flows, budget, seed)
    fit = TrainRequest(**{"profile": "test", "seed": seed, **request})  # type: ignore[arg-type]
    return prepared, shared_fit(prepared, fit)


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
    if kind == "multiselect" or (kind == "pills" and isinstance(value, list)):
        if value:
            return list(value)[:-1]
        options = list(widget.options)  # type: ignore[attr-defined]
        return [options[0]] if options and kind == "multiselect" else None
    if kind == "pills":  # single choice: options are the shown texts, so only clearing it is certain to be valid
        return None
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

    ``NIDS_TEST_PROFILE=1`` is the hidden hook that makes 02 Fit use ``profile="test"`` models.
    """
    monkeypatch.setenv("NIDS_TEST_PROFILE", "1")
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
