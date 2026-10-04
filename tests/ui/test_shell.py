"""Headless render of every station through the real shell (navigation registered, stepper drawn), and the
station addresses: each station answers at its own path and the bare root hands over to 01 Sample."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.commands import navigation
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1 import local_script_runner

from graticule import theme
from ui.stations import ALL_STATIONS, BY_KEY

pytestmark = pytest.mark.ui
APP = Path(__file__).resolve().parents[2] / "app.py"


def _station_script(key: str) -> None:
    """AppTest script body: render one station through the shell."""
    from ui.shell import main

    main(force_key=key)


class Address:
    """Opens the real app (``app.py``) at an address, as a browser does, and records what Streamlit did.

    A browser that opens ``/fit`` asks the server for the page named ``fit``; ``AppTest`` only asks by page hash, so
    its script runner is handed that page name here. Streamlit's "page not found" notice and every
    ``st.switch_page`` are counted.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.not_found: list[str] = []
        self.switches = 0
        real_notice = navigation.send_page_not_found
        real_switch = st.switch_page

        def notice(ctx: Any) -> None:
            self.not_found.append("page not found")
            real_notice(ctx)

        def switch(*args: Any, **kwargs: Any) -> Any:
            self.switches += 1
            return real_switch(*args, **kwargs)

        monkeypatch.setattr(navigation, "send_page_not_found", notice)
        monkeypatch.setattr(st, "switch_page", switch)

    def visit(self, at: AppTest, path: str) -> AppTest:
        """Run ``at`` as a request for ``/<path>`` (``""`` is the bare root)."""
        self.monkeypatch.setattr(local_script_runner, "RerunData", functools.partial(RerunData, page_name=path))
        try:
            return at.run()
        finally:
            self.monkeypatch.setattr(local_script_runner, "RerunData", RerunData)

    def open(self, path: str, **session: object) -> AppTest:
        """A new session (``session`` preset in its state) opened at ``/<path>``."""
        at = AppTest.from_file(str(APP), default_timeout=60)
        for key, value in session.items():
            at.session_state[key] = value
        return self.visit(at, path)


@pytest.fixture
def address(monkeypatch: pytest.MonkeyPatch) -> Address:
    """Open the app at a given address and watch for "page not found" notices and page switches."""
    return Address(monkeypatch)


def _current_station(at: AppTest) -> set[str]:
    """Keys of the stations the stepper marks as current (it should be exactly one)."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        ident = str(getattr(getattr(node, "proto", None), "id", "") or "")
        if "-stn_cur_" in ident:
            found.add(ident.split("-stn_cur_", 1)[1].removesuffix("_now"))
        children = getattr(node, "children", None)
        if isinstance(children, dict):
            for child in children.values():
                walk(child)

    walk(at.main)
    return found


def _heading(key: str) -> str:
    station = BY_KEY[key]
    return station.label if station.number else station.title


@pytest.mark.parametrize("key", [s.key for s in ALL_STATIONS])
def test_every_station_answers_at_its_own_address(address: Address, key: str) -> None:
    """``/sample`` ... ``/bench`` each open their station directly: no notice, no hand-over, the stepper marks it."""
    at = address.open(BY_KEY[key].url_path)
    assert not at.exception, [e.value for e in at.exception]
    assert address.not_found == [] and address.switches == 0
    assert [h.value for h in at.header] == [_heading(key)]
    assert _current_station(at) == {key}


def test_the_root_hands_over_to_01_sample_keeping_the_session(address: Address) -> None:
    """``/`` opens 01 Sample at its own address on a session's first run; session state and query parameters
    survive the hand-over, and nothing is drawn before it."""
    from ui import state

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state[state.DONE] = {"sample"}
    at.session_state["kept"] = 7
    at.query_params["view"] = "sheet"
    address.visit(at, "")
    assert not at.exception, [e.value for e in at.exception]
    assert address.not_found == [] and address.switches == 1
    assert [h.value for h in at.header] == ["01 Sample"]
    assert _current_station(at) == {"sample"}
    assert at.session_state["kept"] == 7 and at.session_state[state.DONE] == {"sample"}
    assert f"01 Sample {theme.GLYPH_DONE}" in [link.proto.label for link in at.get("page_link")]
    assert at.query_params == {"view": ["sheet"]}


def test_the_root_later_in_a_session_draws_01_sample_in_place(address: Address) -> None:
    """Back to ``/`` within the session (the browser's Back button) draws 01 Sample there: a second hand-over would
    push ``/sample`` onto the history again and trap the Back button."""
    at = address.open("")
    assert address.switches == 1 and [h.value for h in at.header] == ["01 Sample"]
    address.visit(at, "")
    assert not at.exception, [e.value for e in at.exception]
    assert address.switches == 1 and address.not_found == []
    assert [h.value for h in at.header] == ["01 Sample"]
    assert _current_station(at) == {"sample"}


def test_an_unknown_address_is_reported_and_lands_on_01_sample(address: Address) -> None:
    """A mistyped address still gets Streamlit's notice (it is unknown), then the visitor lands on 01 Sample."""
    at = address.open("smaple")
    assert not at.exception, [e.value for e in at.exception]
    assert address.not_found == ["page not found"]
    assert [h.value for h in at.header] == ["01 Sample"]
    assert _current_station(at) == {"sample"}


@pytest.mark.parametrize("key", [s.key for s in ALL_STATIONS])
def test_station_renders_without_error(key: str) -> None:
    at = AppTest.from_function(_station_script, args=(key,), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    headers = [h.value for h in at.header]
    assert headers, f"{key}: no page header rendered"


@pytest.mark.parametrize("key", ["sample", "record", "bench"])
def test_the_stepper_brings_the_current_station_into_view(key: str) -> None:
    """On a narrow screen the strip scrolls sideways; a small script (static, with the station key filled in) scrolls
    it to the current station, so a phone at 07 Record does not show 01 to 03 only."""
    at = AppTest.from_function(_station_script, args=(key,), default_timeout=60).run()
    scripts = [e.proto for e in at.get("html") if "<script>" in getattr(e.proto, "body", "")]
    assert len(scripts) == 1 and scripts[0].unsafe_allow_javascript
    assert f'st-key-stn_cur_{key}"' in scripts[0].body and "scrollLeft" in scripts[0].body
    assert _current_station(at) == {key}


def test_bench_shows_synthetic_mode_without_folder() -> None:
    at = AppTest.from_function(_station_script, args=("bench",), default_timeout=60).run()
    assert any("synthetic" in i.value for i in at.info)


def test_app_entrypoint_runs() -> None:
    at = AppTest.from_file("../../app.py", default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [h.value for h in at.header] == ["01 Sample"]


def test_settings_another_program_holds_still_apply_to_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving settings while local_settings.json is locked elsewhere is reported, never raised, and the new values
    still apply to this session."""
    from ui import state

    def held(settings: object, path: object = None) -> None:
        raise PermissionError(13, "Access is denied", "local_settings.json")

    monkeypatch.setattr(state, "save_settings", held)
    at = AppTest.from_function(_station_script, args=("bench",), default_timeout=60).run()
    budget = next(w for w in at.number_input if w.label.startswith("Row budget"))
    budget.set_value(150_000)
    next(b for b in at.button if b.label == "Save settings").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("could not be saved to local_settings.json" in e.value for e in at.error)
    assert at.session_state[state.SETTINGS].row_budget == 150_000
    at.run()  # the message is shown once
    assert not any("could not be saved" in e.value for e in at.error)
