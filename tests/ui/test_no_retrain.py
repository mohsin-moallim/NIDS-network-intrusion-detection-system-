"""Definition of done for 02 Fit: after one fit, no interaction anywhere refits a channel or prepares data again.

The test draws a small synthetic sample through 01 Sample, fits all five channels once through the 02 Fit form
(``profile="test"`` models) and checks the readings, including the CH3 row-cap badge. It then changes every widget
on the Fit page without submitting, visits every other station and changes each of its widgets in turn (rerunning
after every change), and finally comes back to 02 Fit. Throughout, ``graticule.models.train.FIT_CALLS`` must not
move, neither the training matrices nor a training job may be built again, ``prepare_dataset`` must not run again,
and the stored run and every fitted estimator must stay the very same objects. Buttons are never pressed (pressing
one is an explicit action), and widgets are found generically, so stations built in later phases are covered as
soon as they exist.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import pytest
from streamlit.testing.v1 import AppTest

from graticule.data import prepare
from graticule.models import train
from graticule.models.zoo import MODEL_KEYS
from tests.ui.harness import draw_synthetic_sample, errors, fresh_caches, goto, new_app  # noqa: F401
from ui import state, training_ui
from ui.stations import ALL_STATIONS

pytestmark = pytest.mark.ui
# Element-tree attributes of the widget kinds this test knows how to change.
WIDGET_KINDS: tuple[str, ...] = ("checkbox", "toggle", "radio", "selectbox", "multiselect", "number_input", "slider",
                                 "select_slider", "text_input", "text_area")
# What one press of Fit builds: the matrices once, the fit loop once, one job.
ONE_FIT = Counter(build_training_data=1, train_all=1, TrainingJob=1)


@pytest.fixture
def prepare_calls(monkeypatch: pytest.MonkeyPatch) -> list[prepare.DataRequest]:
    """Record every call of prepare_dataset (the real function still runs)."""
    calls: list[prepare.DataRequest] = []
    real = prepare.prepare_dataset

    def spy(request: prepare.DataRequest, **kwargs: object) -> prepare.PreparedDataset:
        calls.append(request)
        return real(request, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(prepare, "prepare_dataset", spy)
    return calls


@pytest.fixture
def training_calls(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count the expensive training steps besides the channel fits: the matrices, the fit loop and job creation.

    The job looks ``build_training_data`` and ``train_all`` up on the module when it runs, so wrapping the module
    attributes sees every call.
    """
    calls: Counter[str] = Counter()

    def counted(name: str, real: Callable[..., object]) -> Callable[..., object]:
        def wrapper(*args: object, **kwargs: object) -> object:
            calls[name] += 1
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(train, "build_training_data", counted("build_training_data", train.build_training_data))
    monkeypatch.setattr(train, "train_all", counted("train_all", train.train_all))
    monkeypatch.setattr(training_ui, "TrainingJob", counted("TrainingJob", training_ui.TrainingJob))
    return calls


def _different_value(widget: object, kind: str) -> object | None:
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
    """Change each widget on the current page once, rerunning after each change; returns what was touched.

    Widgets are addressed by kind and position, re-read after every rerun, so the page may redraw freely.
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
                continue  # a viewer cannot change it either
            new = _different_value(widget, kind)
            if new is None:
                continue
            widget.set_value(new)
            at.run()
            label = f"{kind}:{widget.key or getattr(widget, 'label', index)}"
            assert not errors(at), (label, errors(at))
            touched.append(label)
            after_each(label)
    return touched


def _markdown(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def _readings(at: AppTest):  # noqa: ANN202 - a pandas frame from the element tree
    return next(d.value for d in at.dataframe if "Balanced accuracy" in d.value.columns)


def test_no_widget_anywhere_refits_or_reprepares(fresh_caches: None, prepare_calls: list[prepare.DataRequest],
                                                 training_calls: Counter[str]) -> None:
    at = new_app().run()
    # 4,000 generated flows sampled down to 2,000 rows leave about 1,500 training rows: more than the test-profile
    # SVM cap of 1,000, so CH3 is visibly capped.
    draw_synthetic_sample(at, flows=4_000, budget=2_000)
    assert not errors(at), errors(at)
    assert len(prepare_calls) == 1
    goto(at, "fit")
    fits_before = sum(train.FIT_CALLS.values())
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)

    # One fit of each of the five channels, one set of matrices, one job, and every reading on the page.
    assert sum(train.FIT_CALLS.values()) - fits_before == len(MODEL_KEYS)
    assert training_calls == ONE_FIT
    run_id = at.session_state[state.LAST_RUN_ID]
    run = at.session_state[state.RUN]
    run_identity = id(run)
    estimators = {key: id(result.estimator) for key, result in run.channels.items()}
    fits = sum(train.FIT_CALLS.values())
    assert run.request.profile == "test" and run.ok_channels() == list(MODEL_KEYS)
    readings = _readings(at)
    assert len(readings) == len(MODEL_KEYS) and (readings["Status"] == "fitted").all()
    svm = run.channels["svm"]
    assert svm.rows_used == svm.extra["svm_cap"] == 1_000 < svm.rows_available
    badge = f"CH3 trained on 1,000 of {svm.rows_available:,} rows (SVM cap 1,000)"
    assert badge in _markdown(at)

    def unchanged(label: str) -> None:
        assert sum(train.FIT_CALLS.values()) == fits, f"{label} caused a refit"
        assert training_calls == ONE_FIT, f"{label} rebuilt the training data or started a job"
        assert len(prepare_calls) == 1, f"{label} prepared the data again"
        assert at.session_state[state.LAST_RUN_ID] == run_id, f"{label} replaced the run"
        current = at.session_state[state.RUN]
        assert id(current) == run_identity, f"{label} replaced the run object"
        assert {k: id(r.estimator) for k, r in current.channels.items()} == estimators, f"{label} swapped a model"
        assert state.JOB_ID not in at.session_state, f"{label} started a fit job"

    # 1. Every widget of the Fit form, changed without pressing Fit.
    fit_touched = touch_every_widget(at, unchanged)
    fit_keys = {label.split(":", 1)[1] for label in fit_touched}
    assert {"fit_mode", "fit_features", "fit_k", "fit_port", "fit_channels", "fit_balanced", "fit_min_rows",
            "fit_svm_cap", "fit_test_share"} <= fit_keys

    # 2. Every other station, every widget on it.
    touched_elsewhere: dict[str, list[str]] = {}
    for station in ALL_STATIONS:
        if station.key == "fit":
            continue
        goto(at, station.key)
        assert not errors(at), (station.key, errors(at))
        unchanged(f"visiting {station.key}")
        touched_elsewhere[station.key] = touch_every_widget(at, unchanged)
    assert touched_elsewhere["sample"] and touched_elsewhere["bench"]

    # 3. Back at 02 Fit: the same run, read from memory, with the same readings and the CH3 badge.
    goto(at, "fit")
    assert not errors(at), errors(at)
    unchanged("returning to fit")
    assert _readings(at).equals(readings)
    assert badge in _markdown(at)
    assert "nothing refits until you press Fit again" in _markdown(at)
    assert id(state.run_registry().get(run_id)) == run_identity
