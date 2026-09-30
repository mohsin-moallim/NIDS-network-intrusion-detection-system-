"""The station map: every page of the app, its number, title, address and one-line purpose.

``app.py`` turns this list into ``st.Page`` objects and stores them in :data:`PAGE_OBJECTS`, so any page can link to any
other page by key without importing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Station:
    """One page of the app. ``number`` is empty for the two bench utilities."""

    key: str
    number: str
    title: str
    url_path: str
    purpose: str

    @property
    def label(self) -> str:
        """The text shown in the stepper, e.g. ``"01 Sample"``."""
        return f"{self.number} {self.title}".strip()


STATIONS: tuple[Station, ...] = (
    Station("sample", "01", "Sample", "sample",
            "Load CIC-IDS2017 files or draw synthetic flows, clean them and take a stratified sample."),
    Station("fit", "02", "Fit", "fit",
            "Choose the detection mode, features and channels, then fit them on the sample."),
    Station("measure", "03", "Measure", "measure",
            "Compare the fitted channels side by side and inspect each one in detail."),
    Station("probe", "04", "Probe", "probe",
            "Examine a single flow: its verdict, its probability and the features behind it."),
    Station("assay", "05", "Assay", "assay",
            "Score a whole CSV of flows with a channel and download the readings."),
    Station("sweep", "06", "Sweep", "sweep",
            "Stream flows through a channel at your chosen pace and attack mix, and watch the alerts."),
    Station("record", "07", "Record", "record",
            "Export the measurement record: a PDF report and CSV files of every result."),
)
UTILITIES: tuple[Station, ...] = (
    Station("logbook", "", "Logbook", "logbook",
            "Saved channel sets and the history of every fit run."),
    Station("bench", "", "Bench", "bench",
            "Settings: data folder, cleaning strategy, sample size, SVM cap, seed and alert threshold."),
)
ALL_STATIONS: tuple[Station, ...] = STATIONS + UTILITIES
BY_KEY: dict[str, Station] = {s.key: s for s in ALL_STATIONS}

# Filled in by app.py on every run: station key -> st.Page object (typed loosely to keep this module framework-free).
PAGE_OBJECTS: dict[str, Any] = {}
