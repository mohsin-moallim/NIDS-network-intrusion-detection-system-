"""Setup shared by the headless Streamlit checks (``AppTest``).

Every new ``AppTest`` session scans all installed Python packages for Streamlit's v2 custom-component manifests
before its first run (about 0.15-0.2 s on this machine). Graticule uses no custom components and the installed
packages do not change during a test session, so the scan is made once and its result handed to every session:
the same components (none, here) are registered as before, about 25 s sooner over the whole suite.
"""

from __future__ import annotations

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
