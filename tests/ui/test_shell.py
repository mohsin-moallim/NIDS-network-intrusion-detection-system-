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
