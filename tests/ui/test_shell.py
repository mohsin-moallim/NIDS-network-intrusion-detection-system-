"""Headless render of every station through the real shell (navigation registered, stepper drawn)."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from ui.stations import ALL_STATIONS

pytestmark = pytest.mark.ui


def _station_script(key: str) -> None:
    """AppTest script body: render one station through the shell."""
    from ui.shell import main

    main(force_key=key)


@pytest.mark.parametrize("key", [s.key for s in ALL_STATIONS])
def test_station_renders_without_error(key: str) -> None:
    at = AppTest.from_function(_station_script, args=(key,), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    headers = [h.value for h in at.header]
    assert headers, f"{key}: no page header rendered"


def test_bench_shows_synthetic_mode_without_folder() -> None:
    at = AppTest.from_function(_station_script, args=("bench",), default_timeout=60).run()
    assert any("synthetic" in i.value for i in at.info)


def test_app_entrypoint_runs() -> None:
    at = AppTest.from_file("../../app.py", default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


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
