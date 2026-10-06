"""Setup shared by the headless Streamlit checks (``AppTest``).

Every new ``AppTest`` session scans all installed Python packages for Streamlit's v2 custom-component manifests
before its first run (about 0.15-0.2 s on this machine). NIDS uses no custom components and the installed
packages do not change during a test session, so the scan is made once and its result handed to every session:
the same components (none, here) are registered as before, about 25 s sooner over the whole suite.

The first element any session draws also makes Streamlit decide, once per process, whether to print its "run this
with streamlit run" hint for scripts started without ``streamlit run``. Deciding calls ``inspect.stack()`` with
source lookups on every frame of pytest's deep stack (1 to 2 s on this machine). Under ``AppTest`` the hint never
applies, so the tests mark it as already decided.

While a script runs (in a thread of its own), ``AppTest`` waits for it by waking up every millisecond to look for
the end-of-run event. Each wake-up takes the interpreter lock from the script thread, which slows heavy pages
(03 Measure, 04 Probe, 06 Sweep) by about a tenth. The tests wait for the script thread to end instead (with the
same time limit) and then check for that same event, falling back to Streamlit's own wait (and its timeout error)
when it is missing.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True, scope="session")
def scan_streamlit_components_once() -> Iterator[None]:
    """Remember the first component-manifest scan of the session and reuse it for every later ``AppTest``."""
    try:
        from streamlit.components.v2 import manifest_scanner
    except ImportError:  # an older Streamlit without v2 components: nothing to speed up
        yield
        return
    real = manifest_scanner.scan_component_manifests
    found: list[Any] = []
    scanned = False

    def once(*args: Any, **kwargs: Any) -> Any:
        nonlocal scanned
        if not scanned:
            found.extend(real(*args, **kwargs))
            scanned = True
        return list(found)

    manifest_scanner.scan_component_manifests = once
    try:
        yield
    finally:
        manifest_scanner.scan_component_manifests = real


@pytest.fixture(autouse=True, scope="session")
def skip_the_streamlit_run_hint() -> Iterator[None]:
    """Mark Streamlit's once-per-process "use streamlit run" hint as decided (see the module notes)."""
    from streamlit import delta_generator

    flag = "_use_warning_has_been_displayed"
    if not hasattr(delta_generator, flag):  # another Streamlit version: nothing to skip
        yield
        return
    before = getattr(delta_generator, flag)
    setattr(delta_generator, flag, True)
    try:
        yield
    finally:
        setattr(delta_generator, flag, before)


@pytest.fixture(autouse=True, scope="session")
def wait_for_scripts_without_polling() -> Iterator[None]:
    """Let ``AppTest`` wait for a script run by joining its thread rather than polling (see the module notes)."""
    from streamlit.testing.v1 import local_script_runner

    real = getattr(local_script_runner, "require_widgets_deltas", None)
    if real is None:  # another Streamlit version: keep its own wait
        yield
        return

    def wait(runner: Any, timeout: float = 3) -> None:
        """Wait up to ``timeout`` seconds for the run to end; Streamlit's own wait reports a run that did not."""
        started = time.monotonic()
        thread = getattr(runner, "_script_thread", None)
        if thread is not None:
            thread.join(timeout)
        if runner.script_stopped():
            return
        real(runner, max(timeout - (time.monotonic() - started), 0.0))

    local_script_runner.require_widgets_deltas = wait
    try:
        yield
    finally:
        local_script_runner.require_widgets_deltas = real
