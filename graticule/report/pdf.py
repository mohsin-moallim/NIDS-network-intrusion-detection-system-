"""The PDF measurement record of a run (07 Record), built offline with fpdf2.

:func:`build_report` lays out, on A4 pages in the light palette whatever the app's theme:

* a cover: name, tagline, the run's identity (id, times, data, mode, channels, best channel), the mark key and a
  contents list with page numbers;
* the sample sheet: how the rows were read, cleaned, de-duplicated and sampled, per file and per class (from the
  01 Sample account of the run's own sample when it is still in memory, else from the run's reports);
* the fit settings: mode, feature set, weighting, SVM cap and rows used, test share, seed, bad-value strategy, the
  columns and every channel's notes;
* the readings: the leaderboard (balanced accuracy first, plus the consensus of all channels), its dot plot and
  the per-class table of the best channel;
* charts: every channel's confusion matrix, ROC and precision-recall curves (overlaid for binary runs, per class
  for the best channel otherwise), feature importance and timing;
* optional sections when their results exist: cross-validation, the last Assay batch and the last Sweep;
* notes and limitations, and the dataset citation.

Charts are drawn by the same builders as the app (:mod:`graticule.viz`, ``mode="light"``) and rendered to PNG by
vl-convert (:func:`graticule.viz.to_png`), then reduced to an indexed palette so a five-channel record stays well
under 3 MB. Text uses the bundled TTF fonts (Instrument Sans headings, Atkinson Hyperlegible Next body,
Atkinson Hyperlegible Mono for tables and numbers). Those fonts have no glyphs for the normal/attack/alert marks,
so the record draws them as small vector shapes (hollow circle, diamond, triangle) beside the words; every string
is checked against the font's character map before it is written. Nothing here touches the network.
"""

from __future__ import annotations

import functools
import io
import math
import os
import re
import struct
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.fonts import FontFace

from graticule import APP_NAME, TAGLINE, __version__, evaluate, viz
from graticule.models.jobs import CancelToken, TrainingCancelled
from graticule.models.verdict import ALERT_RULE
from graticule.report.exports import belongs_to_run, consensus_metrics
from graticule.schema import CURATED
from graticule.theme import CHANNEL_BY_KEY, FONT_BODY, FONT_FILES, FONT_HEADING, FONT_MONO, LIGHT

if TYPE_CHECKING:
    from graticule.data.prepare import PreparedDataset
    from graticule.evaluate import ChannelEvaluation
    from graticule.models.train import TrainingRun

ProgressFn = Callable[[str, float], None]

#: Folder of the bundled TTF fonts.
FONT_DIR = Path(__file__).resolve().parents[2] / "static" / "fonts"
#: Font aliases used inside the PDF.
HEAD, BODY, MONO = "head", "body", "mono"
#: The faces the record uses: (alias, style, family, variation axes).
FONT_FACES: tuple[tuple[str, str, str, dict[str, float]], ...] = (
    (HEAD, "", FONT_HEADING, {"wght": 600, "wdth": 100}),
    (BODY, "", FONT_BODY, {"wght": 400}),
    (BODY, "B", FONT_BODY, {"wght": 700}),
    (MONO, "", FONT_MONO, {"wght": 400}),
    (MONO, "B", FONT_MONO, {"wght": 700}),
)
#: Page geometry (mm): A4 portrait with these margins.
PAGE_MARGIN = 16.0
TOP_MARGIN = 19.0
BOTTOM_MARGIN = 16.0
#: Millimetres per CSS pixel of a chart (a 560 px chart spans 179 mm, the text width).
MM_PER_PX = 0.32
#: Scale at which charts are rendered to PNG (device pixels per CSS pixel).
PNG_SCALE = 2
#: Colours kept when a chart PNG is reduced to an indexed palette.
PNG_COLOURS = 256
#: Paper colour behind the charts.
PAPER = "#FFFFFF"
#: Widest a curve chart is drawn (mm), so a ROC and a precision-recall chart share a page.
CURVE_WIDTH = 150.0
#: Smallest share of its size a chart may be shrunk to so that it fits the rest of a page.
SHRINK_TO_FIT = 0.72
#: Features shown in importance charts.
IMPORTANCE_TOP = 15
#: Zoomed range of the second ROC chart of a binary run.
ROC_ZOOM = 0.05
#: Replacements for characters the fonts cannot draw (anything else missing becomes "?").
FALLBACKS: dict[str, str] = {chr(code): text for code, text in (
    (0x25CB, ""), (0x25C6, ""), (0x25B2, ""),  # the normal, attack and alert marks (drawn as shapes instead)
    (0x00B5, "u"), (0x2713, "yes"), (0x2717, "no"), (0x2265, ">="), (0x2264, "<="), (0x00B1, "+/-"), (0x00D7, "x"),
    (0x2192, "->"), (0x2013, "-"), (0x2014, "-"), (0x00B7, "-"), (0x2026, "..."), (0x2022, "-"), (0x2019, "'"),
    (0x2018, "'"), (0x201C, '"'), (0x201D, '"'), (0x00A0, " "),
)}
#: The dataset citation printed at the end of every record.
CITATION = ("Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, \"Toward Generating a New Intrusion "
            "Detection Dataset and Intrusion Traffic Characterization\", 4th International Conference on Information "
            "Systems Security and Privacy (ICISSP), 2018.")
#: What the bad-value strategies do, in words.
STRATEGY_TEXT: dict[str, str] = {
    "drop": "drop: rows holding an infinite or missing value were removed before sampling",
    "impute": "impute: such rows were kept; each channel fills the gaps with its training medians",
    "recompute": "rebuild: the two rate columns were recomputed from totals and duration; remaining gaps are filled "
                 "with each channel's training medians",
}
#: Leaderboard columns the record prints, per mode (titles of :func:`graticule.evaluate.score_columns`).
LEADERBOARD_BINARY: tuple[str, ...] = ("Balanced accuracy", "Accuracy", "Precision (attack)", "Recall (attack)",
                                       "F1 (attack)", "ROC-AUC", "Average precision")
LEADERBOARD_MULTICLASS: tuple[str, ...] = ("Balanced accuracy", "Accuracy", "F1 macro", "F1 weighted",
                                           "Precision macro", "Recall macro", "ROC-AUC", "Average precision")
#: Short table headings of those columns (any column without one keeps its own title).
SHORT_HEADS: dict[str, str] = {
    "Balanced accuracy": "Bal. acc.", "Accuracy": "Accuracy", "Precision (attack)": "Precision",
    "Recall (attack)": "Recall", "F1 (attack)": "F1", "ROC-AUC": "ROC-AUC", "Average precision": "Avg prec.",
    "F1 macro": "F1 macro", "F1 weighted": "F1 wtd", "Precision macro": "Prec. macro", "Recall macro": "Rec. macro",
}
#: Decimals of an alert threshold in the record (the app shows three; 0.999 must not print as 1.00).
THRESHOLD_FORMAT = "{:.3f}"
FEATURE_MODE_TEXT: dict[str, str] = {"curated": "Curated set (seven groups chosen without looking at the data)",
                                     "all": "Every numeric column that varies in the sample",
                                     "topk": "Top-K by XGBoost total gain, ranked on training rows only"}

_MARK_SVG: dict[str, str] = {
    "normal": ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
               f'<circle cx="5" cy="5" r="3.3" fill="#FFFFFF" stroke="{LIGHT.benign}" stroke-width="1.5"/></svg>'),
    "attack": ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
               f'<path d="M5 0.9 L9.1 5 L5 9.1 L0.9 5 Z" fill="{LIGHT.attack}"/></svg>'),
    "alert": ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
              f'<path d="M5 1.0 L9.4 8.8 L0.6 8.8 Z" fill="{LIGHT.warning}" stroke="{LIGHT.text}" '
              'stroke-width="0.5"/></svg>'),
}


# --------------------------------------------------------------------------------------------------------------
# Optional inputs
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AssaySummary:
    """What the last 05 Assay batch found, as plain values for the record."""

    channel: str
    rows: int
    rows_with_bad_values: int
    attacks_found: int | None
    alerts: int | None
    seconds: float | None
    labelled: bool
    accuracy: float | None
    balanced_accuracy: float | None
    confusion: pd.DataFrame | None
    unseen_labels: dict[str, int]
    alert_threshold: float | None = None
    source_name: str = ""


@dataclass(frozen=True)
class SweepSummary:
    """Where the last 06 Sweep simulation stood, as plain values for the record."""

    channel: str
    source: str
    flows: int
    correct: int | None
    live_accuracy: float | None
    live_balanced_accuracy: float | None
    alerts: int | None
    ticks: int | None
    alert_threshold: float | None
    confusion: np.ndarray | None


@dataclass
class ReportExtras:
    """Results beyond the readings that the record includes when they exist (each one may be missing).

    ``cross_validation`` is a table of :func:`graticule.evaluate.cross_validate_run`; ``permutations`` maps a
    channel key to its permutation-importance table; ``assay`` and ``sweep`` summarise the other stations' work.
    """

    cross_validation: pd.DataFrame | None = None
    permutations: dict[str, pd.DataFrame] = field(default_factory=dict)
    assay: AssaySummary | None = None
    sweep: SweepSummary | None = None

    @classmethod
    def from_run(cls, run: "TrainingRun", *, assay: Any = None, sweep: Any = None) -> "ReportExtras":
        """The extras kept with ``run`` (cross-validation, permutation importance), plus the Assay batch and the
        simulation session when they were made with this run (see :func:`graticule.report.exports.belongs_to_run`)."""
        cv = evaluate.stored_cross_validation(run)
        return cls(
            cross_validation=cv if cv is not None and not cv.empty else None,
            permutations=evaluate.stored_permutations(run),
            assay=summarise_assay(assay) if belongs_to_run(assay, run) else None,
            sweep=summarise_sweep(sweep) if belongs_to_run(sweep, run) else None,
        )


def _number(value: Any) -> float | None:
    """A finite float, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _whole(value: Any) -> int | None:
    """An int, or None for anything that is not a whole number."""
    number = _number(value)
    return None if number is None else int(number)


def _alert_rule_text(threshold: float | None) -> str:
    """The alert rule in brackets, with the threshold in three decimals, e.g. ``" (attack verdicts with attack
    probability at least 0.900)"``."""
    if threshold is None:
        return " (attack verdicts with attack probability at or above the alert threshold)"
    return f" (attack verdicts with attack probability at least {THRESHOLD_FORMAT.format(threshold)})"


def summarise_assay(batch: Any) -> AssaySummary | None:
    """Plain readings of an Assay batch (``frame``, ``channel``, ``rows``... read by name), or None without one."""
    if batch is None:
        return None
    frame = getattr(batch, "frame", None)
    has_frame = isinstance(frame, pd.DataFrame)
    rows = _whole(getattr(batch, "rows", None))
    if rows is None:
        rows = len(frame) if has_frame else 0
    channel = str(getattr(batch, "channel", "") or "")
    attacks = alerts = None
    if has_frame and "predicted_label" in frame.columns:
        counts = frame["predicted_label"].astype("str").value_counts()
        attacks = int(sum(int(n) for label, n in counts.items() if viz.kind_of(str(label)) == viz.KIND_ATTACK))
    if has_frame and "alert" in frame.columns:
        alerts = int(pd.Series(frame["alert"]).fillna(False).astype(bool).sum())
    confusion = getattr(batch, "confusion", None)
    unseen = getattr(batch, "unseen_labels", None) or {}
    return AssaySummary(
        channel="Consensus of all channels" if channel == "consensus" else evaluate.channel_label(channel),
        rows=int(rows), rows_with_bad_values=int(_whole(getattr(batch, "rows_with_bad_values", 0)) or 0),
        attacks_found=attacks, alerts=alerts, seconds=_number(getattr(batch, "seconds", None)),
        labelled=bool(getattr(batch, "labelled", False)), accuracy=_number(getattr(batch, "accuracy", None)),
        balanced_accuracy=_number(getattr(batch, "balanced_accuracy", None)),
        confusion=confusion if isinstance(confusion, pd.DataFrame) and not confusion.empty else None,
        unseen_labels={str(k): int(v) for k, v in dict(unseen).items()},
        alert_threshold=_number(getattr(batch, "alert_threshold", None)),
        source_name=str(getattr(batch, "source_name", "") or ""),
    )


def summarise_sweep(session: Any) -> SweepSummary | None:
    """Plain readings of a simulation session (``stats``, ``channel``, ``source``... read by name), or None when it
    has emitted no flow."""
    if session is None:
        return None
    stats = getattr(session, "stats", None)
    if callable(stats):
        stats = stats()
    flows = _whole(getattr(stats, "emitted", None))
    if not flows:
        return None
    source = getattr(session, "source", None)
    kind = type(source).__name__ if source is not None else ""
    describe = getattr(source, "describe", None)
    if callable(describe):
        source_text = str(describe())
    elif "Replay" in kind:
        source_text = "Held-out flows replayed with their true labels"
    elif "Synthetic" in kind:
        source_text = "Synthetic stream from the flow generator"
    else:
        source_text = "Stream"
    confusion = getattr(stats, "confusion", None)
    matrix = np.asarray(confusion) if confusion is not None else None
    if matrix is not None and (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0):
        matrix = None
    return SweepSummary(
        channel=evaluate.channel_label(str(getattr(session, "channel", "") or "")), source=source_text,
        flows=int(flows), correct=_whole(getattr(stats, "correct", None)),
        live_accuracy=_number(getattr(stats, "live_accuracy", None)),
        live_balanced_accuracy=_number(getattr(stats, "live_balanced_accuracy", None)),
        alerts=_whole(getattr(stats, "alerts_total", None)), ticks=_whole(getattr(stats, "ticks", None)),
        alert_threshold=_number(getattr(session, "alert_threshold", None)), confusion=matrix,
    )


def summarise_prepared(prepared: "PreparedDataset") -> dict[str, Any]:
    """The 01 Sample account of a prepared sample as plain values (what the record's sample sheet prints).

    Keys: ``fingerprint``, ``description``, ``source``, ``strategy``, ``readings`` (rows of Reading/Value/Note),
    ``files`` (one dict per source file), ``classes`` (one dict per class: Class, Kind, Available, In sample,
    Share of sample), ``before`` and ``after`` (rows per class before and after sampling) and ``seconds``.
    """
    files = prepared.file_table()
    keep = [c for c in ("File", "Session", "Rows read", "Bad-value rows", "Duplicates in file", "Rows after file stage",
                        "Rows in sample", "Encoding") if c in files.columns]
    return {
        "fingerprint": prepared.fingerprint,
        "description": prepared.request.describe(),
        "source": prepared.request.source,
        "strategy": prepared.nonfinite.strategy,
        "readings": [dict(r) for r in prepared.summary_rows()],
        "files": files[keep].to_dict("records"),
        "classes": prepared.class_table().to_dict("records"),
        "before": {str(k): int(v) for k, v in prepared.sampling.before.items()},
        "after": {str(k): int(v) for k, v in prepared.sampling.after.items()},
        "seconds": float(prepared.seconds),
    }


# --------------------------------------------------------------------------------------------------------------
# Text and number helpers
# --------------------------------------------------------------------------------------------------------------
_FONT_LOCK = threading.Lock()
#: Folder (under the system's temporary folder) caching static instances of the variable fonts.
FONT_CACHE_NAME = "graticule-pdf-fonts"


def static_font(file_name: str, axes: Mapping[str, float]) -> Path:
    """A static instance of the bundled variable font ``file_name`` at ``axes`` (e.g. ``{"wght": 700}``).

    Turning a variable font into a static one takes about half a second per weight, so each instance is made once
    and kept in a small cache folder in the system's temporary folder (written atomically; fonts only). Raises
    ``OSError`` when the cache cannot be written; callers then let fpdf2 make the instance in memory.
    """
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    from graticule.settings import replace_with_retry

    source = FONT_DIR / file_name
    tag = "-".join(f"{name}{float(value):g}" for name, value in sorted(axes.items()))
    folder = Path(tempfile.gettempdir()) / FONT_CACHE_NAME
    target = folder / f"{source.stem}-{tag}-{source.stat().st_size}.ttf"
    if target.is_file():
        return target
    with _FONT_LOCK:
        if target.is_file():
            return target
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f".{target.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
        font = TTFont(str(source))
        try:
            instancer.instantiateVariableFont(font, {k: float(v) for k, v in axes.items()}, inplace=True,
                                              static=True)
            font.save(str(temporary))
        finally:
            font.close()
        try:
            replace_with_retry(temporary, target)
        except OSError:
            temporary.unlink(missing_ok=True)
            if not target.is_file():
                raise
    return target


@functools.lru_cache(maxsize=None)
def font_characters(file_name: str) -> frozenset[int]:
    """Code points the bundled font ``file_name`` can draw (read once per process)."""
    from fontTools.ttLib import TTFont

    font = TTFont(str(FONT_DIR / file_name), lazy=True)
    try:
        return frozenset(font.getBestCmap())
    finally:
        font.close()


_ALIAS_FILES = {HEAD: FONT_FILES[FONT_HEADING], BODY: FONT_FILES[FONT_BODY], MONO: FONT_FILES[FONT_MONO]}


def printable(text: Any, family: str = BODY) -> str:
    """``text`` with every character the font ``family`` (an alias) cannot draw replaced (see :data:`FALLBACKS`)."""
    covered = font_characters(_ALIAS_FILES.get(family, _ALIAS_FILES[BODY]))
    out = []
    for char in str(text):
        if char in "\n\t" or ord(char) in covered:
            out.append(char)
        else:
            out.append(FALLBACKS.get(char, "?"))
    return "".join(out)


def _rgb(colour: str) -> tuple[int, int, int]:
    """A ``#RRGGBB`` colour as an (r, g, b) tuple."""
    value = colour.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _score(value: Any) -> str:
    """A score with four decimals, or "n/a"."""
    number = _number(value)
    return "n/a" if number is None else f"{number:.4f}"


def _count(value: Any) -> str:
    """A whole number with thousands separators, or "n/a"."""
    number = _number(value)
    return "n/a" if number is None else f"{int(round(number)):,}"


def _seconds(value: Any) -> str:
    """Seconds with two decimals (thousands separators), or "n/a"."""
    number = _number(value)
    return "n/a" if number is None else f"{number:,.2f}"


def _utc_text(value: Any) -> str:
    """An ISO time as ``2026-10-01 15:30 UTC`` (the text itself when it does not parse)."""
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def count_pages(data: bytes) -> int:
    """Number of pages in a PDF's bytes (counts its page objects)."""
    return len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", data))


def count_images(data: bytes) -> int:
    """Number of raster images embedded in a PDF's bytes."""
    return len(re.findall(rb"/Subtype\s*/Image", data))


# --------------------------------------------------------------------------------------------------------------
# Charts as images
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Png:
    """A rendered chart: PNG bytes and their pixel size."""

    data: bytes
    width: int
    height: int


def _png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG header."""
    return struct.unpack(">II", data[16:24])


def chart_png(chart: Any) -> bytes:
    """Render an Altair chart for the record: PNG at twice its size on white paper, reduced to an indexed palette.

    The palette reduction keeps the look (charts use few colours plus anti-aliasing) at a fraction of the size.
    """
    from PIL import Image

    raw = viz.to_png(chart, scale=PNG_SCALE, background=PAPER)
    with Image.open(io.BytesIO(raw)) as image:
        reduced = image.convert("RGB").quantize(colors=PNG_COLOURS, method=Image.Quantize.MEDIANCUT)
    out = io.BytesIO()
    reduced.save(out, format="PNG", optimize=True)
    return out.getvalue()


# --------------------------------------------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------------------------------------------
class _RecordPDF(FPDF):
    """A4 pages with Graticule's running header (from page 2) and a footer with page numbers on every page."""

    def __init__(self, run_id: str, built_text: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.run_id = run_id
        self.built_text = built_text

    def header(self) -> None:
        """Record name and run id above a hairline (not on the cover)."""
        if self.page_no() == 1:
            return
        with self.local_context():
            self.set_xy(self.l_margin, 9.5)
            self.set_text_color(*_rgb(LIGHT.muted))
            self.set_font(BODY, "", 7.5)
            self.cell(w=self.epw / 2, h=4, text=f"{APP_NAME} measurement record")
            self.set_font(MONO, "", 7.5)
            self.cell(w=self.epw / 2, h=4, text=printable(f"run {self.run_id}", MONO), align="R")
            self.set_draw_color(*_rgb(LIGHT.border))
            self.set_line_width(0.2)
            self.line(self.l_margin, 14.5, self.w - self.r_margin, 14.5)
        self.set_y(TOP_MARGIN)

    def footer(self) -> None:
        """Build time and version on the left, page number of the total on the right."""
        with self.local_context():
            self.set_y(-11)
            self.set_text_color(*_rgb(LIGHT.muted))
            self.set_font(BODY, "", 7.5)
            self.cell(w=self.epw * 0.7, h=4, text=printable(self.built_text))
            self.set_font(MONO, "", 7.5)
            self.cell(w=self.epw * 0.3, h=4, text=f"page {self.page_no()} of {{nb}}", align="R")


@dataclass(frozen=True)
class RenderedReport:
    """A built record: the PDF bytes and facts about it."""

    data: bytes
    pages: int
    images: int
    sections: tuple[str, ...]
    seconds: float
    problems: tuple[str, ...]

    @property
    def size_bytes(self) -> int:
        """Size of the PDF in bytes."""
        return len(self.data)


class _Builder:
    """Lays out one record (one instance per document)."""

    def __init__(self, run: "TrainingRun", evaluations: Mapping[str, "ChannelEvaluation"], *,
                 prepared_summary: Mapping[str, Any] | None, settings: Mapping[str, Any],
                 extras: ReportExtras, progress: ProgressFn | None, cancel: CancelToken | None,
                 built_utc: datetime) -> None:
        self.run = run
        self.evals = {k: evaluations[k] for k in evaluate.CHANNEL_ORDER if k in evaluations}
        self.extras = extras
        self.settings = dict(settings or {})
        self.progress = progress
        self.cancel = cancel
        self.built_utc = built_utc
        self.summary = None
        self.summary_note = ""
        if prepared_summary is not None:
            if str(prepared_summary.get("fingerprint", "")) == str(run.dataset_fingerprint):
                self.summary = prepared_summary
            else:
                self.summary_note = ("The sample loaded at 01 Sample now is not the one this run was fitted on, so "
                                     "its account is not repeated here.")
        else:
            self.summary_note = ("The 01 Sample account of this run's sample is not in memory (the run was loaded "
                                 "from disk, or another sample was drawn since); the figures below come from the "
                                 "run itself.")
        self.classes = [str(c) for c in run.data.classes]
        self.binary = len(self.classes) == 2
        built = built_utc.strftime("%Y-%m-%d %H:%M UTC")
        self.pdf = _RecordPDF(run.run_id, f"Built {built} by {APP_NAME} {__version__}")
        self.sections: list[str] = []
        self.problems: list[str] = []
        self.number = 0
        self.done = 0
        self.total = max(self._planned_charts() + 3, 1)
        self.board = evaluate.leaderboard(self.evals, run) if self.evals else pd.DataFrame()

    # ---------------------------------------------------------------- progress
    def _planned_charts(self) -> int:
        """How many charts this record will draw (for the progress fraction)."""
        n = len(self.evals)
        native = sum(1 for k in self.evals if k in ("forest", "xgboost"))
        count = 2 + n + 2 + native + len(self.extras.permutations) + 1
        count += int(self.extras.cross_validation is not None)
        count += int(self.extras.assay is not None and self.extras.assay.confusion is not None)
        count += int(self.extras.sweep is not None and self.extras.sweep.confusion is not None)
        return count

    def step(self, message: str) -> None:
        """Count one unit of work, report progress, and stop here when the build was cancelled."""
        if self.cancel is not None and self.cancel.cancelled:
            raise TrainingCancelled("The PDF record build was cancelled.")
        self.done += 1
        if self.progress is not None:
            self.progress(message, min(self.done / self.total, 0.99))

    # ---------------------------------------------------------------- set-up
    def setup(self, compress: bool) -> None:
        """Fonts, metadata, margins and page breaks."""
        pdf = self.pdf
        pdf.set_compression(bool(compress))
        for alias, style, family, axes in FONT_FACES:
            file_name = FONT_FILES[family]
            try:
                pdf.add_font(alias, style, str(static_font(file_name, axes)))
            except OSError:
                pdf.add_font(alias, style, str(FONT_DIR / file_name), variations=dict(axes))
        pdf.set_margins(PAGE_MARGIN, TOP_MARGIN, PAGE_MARGIN)
        pdf.set_auto_page_break(True, margin=BOTTOM_MARGIN)
        pdf.set_title(f"{APP_NAME} measurement record, run {self.run.run_id}")
        pdf.set_author(APP_NAME)
        pdf.set_subject("Readings of network-intrusion classifiers on held-out flows")
        pdf.set_creator(f"{APP_NAME} {__version__}")
        pdf.set_creation_date(self.built_utc)
        pdf.set_lang("en")
        pdf.set_text_color(*_rgb(LIGHT.text))
        pdf.set_draw_color(*_rgb(LIGHT.border))

    # ---------------------------------------------------------------- primitives
    def font(self, family: str, size: float, style: str = "", colour: str = LIGHT.text) -> None:
        """Set the font and text colour."""
        self.pdf.set_font(family, style, size)
        self.pdf.set_text_color(*_rgb(colour))

    def room(self, height: float) -> None:
        """Start a new page unless ``height`` mm still fit on this one."""
        if self.pdf.get_y() + height > self.pdf.page_break_trigger:
            self.pdf.add_page()

    def section(self, title: str, *, new_page: bool = False, need: float = 70.0) -> None:
        """Begin a numbered section (an outline entry and a contents line)."""
        pdf = self.pdf
        if pdf.page_no() == 0:
            pdf.add_page()
        elif new_page:
            if pdf.get_y() > pdf.t_margin + 0.5:  # (a page just begun, e.g. after the contents, is used as it is)
                pdf.add_page()
        else:
            pdf.ln(5)
            self.room(need)
        self.number += 1
        self.sections.append(title)
        pdf.start_section(printable(title, HEAD), level=0)
        y = pdf.get_y()
        self.font(HEAD, 10, colour=LIGHT.secondary)
        pdf.set_xy(pdf.l_margin, y + 1.6)
        pdf.cell(w=10, h=6, text=f"{self.number:02d}")
        self.font(HEAD, 16)
        pdf.set_xy(pdf.l_margin + 10, y)
        pdf.cell(w=pdf.epw - 10, h=8, text=printable(title, HEAD), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*_rgb(LIGHT.border))
        pdf.set_line_width(0.3)
        pdf.line(pdf.l_margin, pdf.get_y() + 0.8, pdf.w - pdf.r_margin, pdf.get_y() + 0.8)
        pdf.ln(3.5)

    def subheading(self, title: str) -> None:
        """A smaller heading inside a section (kept with at least a few lines of what follows)."""
        self.pdf.ln(2.5)
        self.room(22)
        self.font(HEAD, 11.5)
        self.pdf.cell(w=0, h=6, text=printable(title, HEAD), new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(1)

    def para(self, text: str, *, size: float = 9.5, colour: str = LIGHT.text, family: str = BODY,
             gap: float = 1.8, height: float | None = None) -> None:
        """A paragraph across the text width."""
        self.font(family, size, colour=colour)
        self.pdf.multi_cell(w=0, h=height or size * 0.5, text=printable(text, family), align="L",
                            new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(gap)

    def note(self, text: str) -> None:
        """A muted note under a table or chart."""
        self.para(text, size=8.2, colour=LIGHT.muted, gap=2.2)

    def mark(self, kind: str, x: float, y: float, size: float = 3.0) -> None:
        """Draw the normal (hollow circle), attack (diamond) or alert (triangle) mark with its top-left at (x, y)."""
        self.pdf.image(io.BytesIO(_MARK_SVG[kind].encode("ascii")), x=x, y=y, w=size, h=size)

    def marked_line(self, items: Sequence[tuple[str | None, str]], *, size: float = 9.0, gap: float = 1.5) -> None:
        """One line of words, each optionally preceded by its mark (``(kind or None, text)`` pairs)."""
        pdf = self.pdf
        self.room(6)
        self.font(BODY, size)
        height = size * 0.5
        x = pdf.l_margin
        y = pdf.get_y()
        for kind, text in items:
            if kind is not None:
                self.mark(kind, x, y + (height - 3.0) / 2, 3.0)
                x += 4.2
            shown = printable(text, BODY)
            width = pdf.get_string_width(shown) + 1.2
            pdf.set_xy(x, y)
            pdf.cell(w=width, h=height, text=shown)
            x += width + 3.5
        pdf.set_xy(pdf.l_margin, y + height)
        pdf.ln(gap)

    def facts(self, rows: Sequence[tuple[str, str]], *, key_width: float = 46.0, size: float = 9.0) -> None:
        """Label/value lines separated by hairlines (values wrap; long lists continue on the next page)."""
        pdf = self.pdf
        height = size * 0.5
        for label, value in rows:
            shown = printable(value, BODY)
            self.font(BODY, size)
            lines = pdf.multi_cell(w=pdf.epw - key_width, h=height, text=shown, dry_run=True, output="LINES")
            needed = height * max(len(lines), 1) + 2.2
            self.room(needed)
            y = pdf.get_y() + 1.0
            self.font(BODY, size - 0.6, colour=LIGHT.muted)
            pdf.set_xy(pdf.l_margin, y)
            pdf.cell(w=key_width, h=height, text=printable(label, BODY))
            self.font(BODY, size)
            pdf.set_xy(pdf.l_margin + key_width, y)
            pdf.multi_cell(w=pdf.epw - key_width, h=height, text=shown, align="L", new_x="LMARGIN", new_y="NEXT")
            bottom = pdf.get_y() + 1.0
            pdf.set_draw_color(*_rgb(LIGHT.border))
            pdf.set_line_width(0.15)
            pdf.line(pdf.l_margin, bottom, pdf.w - pdf.r_margin, bottom)
            pdf.set_y(bottom)
        pdf.ln(2.5)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]], *, widths: Sequence[float],
              align: Sequence[str], marks: Sequence[str | None] | None = None,
              bold_rows: Sequence[int] = (), size: float = 8.0) -> None:
        """A table in the mono face (headings in bold body type, repeated on every page it spans).

        ``widths`` are relative column widths and ``align`` the text alignment per column (``"LEFT"``,
        ``"RIGHT"``, ``"CENTER"``). ``marks`` (one entry per row: ``"normal"``, ``"attack"``, ``"alert"`` or None)
        adds a narrow first column with the row's vector mark. Rows in ``bold_rows`` are set in bold.
        """
        pdf = self.pdf
        self.room(16)
        heads = [printable(h, BODY) for h in headers]
        widths = list(widths)
        align = list(align)
        if marks is not None:
            heads = [""] + heads
            widths = [3.2] + widths
            align = ["CENTER"] + align
        total = sum(widths)
        widths = [w / total * pdf.epw for w in widths]
        self.font(MONO, size)
        pdf.set_draw_color(*_rgb(LIGHT.border))
        pdf.set_line_width(0.15)
        heading_style = FontFace(family=BODY, emphasis="BOLD", size_pt=size - 0.4, color=_rgb(LIGHT.text),
                                 fill_color=_rgb(LIGHT.background))
        bold = FontFace(emphasis="BOLD")
        with pdf.table(col_widths=widths, text_align=tuple(align), borders_layout="HORIZONTAL_LINES",
                       headings_style=heading_style, line_height=size * 0.48, padding=(0.9, 1.2),
                       v_align="MIDDLE", first_row_as_headings=True) as table:
            head = table.row()
            for text in heads:
                head.cell(text)
            for index, values in enumerate(rows):
                row = table.row()
                style = bold if index in bold_rows else None
                if marks is not None:
                    kind = marks[index]
                    if kind is None:
                        row.cell("")
                    else:
                        row.cell(img=io.BytesIO(_MARK_SVG[kind].encode("ascii")), padding=(1.3, 0.2))
                for value in values:
                    row.cell(printable(value, MONO), style=style)
        pdf.ln(2.5)

    # ---------------------------------------------------------------- images
    def render(self, name: str, build: Callable[[], Any]) -> _Png | None:
        """Render one chart (``build`` returns an Altair chart); a failure is noted and returns None."""
        self.step(f"Drawing {name}")
        try:
            data = chart_png(viz.build_chart(build))  # the whole spec is checked once, when it is rendered
        except TrainingCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - one chart that cannot be drawn must not lose the record
            self.problems.append(f"The chart '{name}' could not be drawn ({type(exc).__name__}: {exc}).")
            return None
        width, height = _png_size(data)
        return _Png(data, width, height)

    def place(self, images: Sequence[_Png | None], *, per_row: int = 1, max_width: float | None = None,
              gap: float = 5.0) -> None:
        """Lay images out left to right, ``per_row`` to a row, at the record's chart scale (never larger)."""
        pdf = self.pdf
        shown = [image for image in images if image is not None]
        slot = (pdf.epw - gap * (per_row - 1)) / per_row
        limit = min(slot, max_width or pdf.epw)
        tallest = pdf.eph - 12
        for start in range(0, len(shown), per_row):
            group = shown[start:start + per_row]
            sizes = []
            for image in group:
                width = min(image.width / PNG_SCALE * MM_PER_PX, limit)
                height = width * image.height / image.width
                if height > tallest:
                    width, height = width * tallest / height, tallest
                sizes.append((width, height))
            row_height = max(h for _, h in sizes)
            left = pdf.page_break_trigger - pdf.get_y() - 1
            if row_height > left >= row_height * SHRINK_TO_FIT:
                # Slightly smaller charts rather than a page left mostly empty.
                factor = left / row_height
                sizes = [(w * factor, h * factor) for w, h in sizes]
                row_height = left
            self.room(row_height + 1)
            y = pdf.get_y()
            x = pdf.l_margin
            for image, (width, height) in zip(group, sizes):
                pdf.image(io.BytesIO(image.data), x=x, y=y, w=width, h=height)
                x += width + gap
            pdf.set_y(y + row_height + gap * 0.6)

    def chart(self, name: str, build: Callable[[], Any], *, max_width: float | None = None) -> None:
        """Render and place one chart across the text width (or ``max_width`` mm)."""
        self.place([self.render(name, build)], max_width=max_width)

    # ---------------------------------------------------------------- sections
    def cover(self) -> None:
        """Page 1: identity of the run, mark key and contents."""
        pdf = self.pdf
        run = self.run
        pdf.add_page()
        pdf.set_y(24)
        self.font(BODY, 8.5, "B", colour=LIGHT.secondary)
        pdf.set_char_spacing(1.4)
        pdf.cell(w=0, h=5, text="MEASUREMENT RECORD", new_x="LMARGIN", new_y="NEXT")
        pdf.set_char_spacing(0)
        pdf.ln(1)
        self.font(HEAD, 34)
        pdf.cell(w=0, h=15, text=APP_NAME, new_x="LMARGIN", new_y="NEXT")
        self.para(TAGLINE, size=11.5, colour=LIGHT.muted, gap=3)
        pdf.set_draw_color(*_rgb(LIGHT.secondary))
        pdf.set_line_width(0.6)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 30, pdf.get_y())
        pdf.ln(5)
        best = self._best()
        best_text = "n/a"
        if best is not None:
            best_text = f"{evaluate.channel_label(best[0])}, balanced accuracy {best[1]:.4f}"
        fitted = [evaluate.channel_label(k) for k in run.ok_channels()]
        others = [f"{evaluate.channel_label(k)} ({r.status.replace('_', ' ')})" for k, r in run.channels.items()
                  if k not in run.ok_channels()]
        origin = "Fitted in this app"
        if getattr(run, "origin", "fitted") == "loaded":
            folder = Path(str(getattr(run, "bundle_path", "") or "")).name
            origin = f"Loaded from the saved channel set {folder}" if folder else "Loaded from a saved channel set"
        held_out = len(run.data.y_test)
        self.facts([
            ("Run", run.run_id),
            ("Fitted", _utc_text(run.created_utc)),
            ("Record built", self.built_utc.strftime("%Y-%m-%d %H:%M UTC")),
            ("Data", run.data_request.describe()),
            ("Mode", self._mode_text()),
            ("Features", self._feature_text()),
            ("Channels fitted", ", ".join(fitted) + (f". Not fitted: {', '.join(others)}" if others else "")),
            ("Best channel", best_text),
            ("Held-out rows", f"{held_out:,}" if held_out else "none in memory (loaded without its data)"),
            ("Origin", origin),
        ], key_width=36, size=9.5)
        pdf.ln(1)
        self.font(BODY, 8.2, colour=LIGHT.muted)
        pdf.cell(w=0, h=4.5, text="Marks used in tables and notes:", new_x="LMARGIN", new_y="NEXT")
        self.marked_line([("normal", "Normal traffic"), ("attack", "Attack"),
                          ("alert", "Alert (high-confidence attack)")], size=9.0, gap=4)
        self.font(HEAD, 12.5)
        pdf.cell(w=0, h=7, text="Contents", new_x="LMARGIN", new_y="NEXT")
        pdf.insert_toc_placeholder(self._contents, pages=1)

    def _contents(self, pdf: FPDF, outline: Sequence[Any]) -> None:
        """Draw the contents list (called by fpdf2 once every section's page is known)."""
        entries = [s for s in outline if getattr(s, "level", 0) == 0]
        space = pdf.page_break_trigger - pdf.get_y() - 2
        height = min(6.0, max(space / max(len(entries), 1), 3.6))
        for index, entry in enumerate(entries, start=1):
            name = str(entry.name)
            page = int(entry.page_number)
            link = pdf.add_link(page=page)
            y = pdf.get_y()
            pdf.set_xy(pdf.l_margin, y)
            pdf.set_font(HEAD, "", 9.5)
            pdf.set_text_color(*_rgb(LIGHT.secondary))
            pdf.cell(w=9, h=height, text=f"{index:02d}", link=link)
            pdf.set_font(BODY, "", 10)
            pdf.set_text_color(*_rgb(LIGHT.text))
            label = printable(name, BODY)
            pdf.cell(w=pdf.get_string_width(label) + 2, h=height, text=label, link=link)
            end = pdf.l_margin + pdf.epw - 14
            start = pdf.get_x() + 1
            if end > start:
                pdf.set_draw_color(*_rgb(LIGHT.border))
                pdf.set_line_width(0.2)
                pdf.set_dash_pattern(dash=0.4, gap=1.2)
                pdf.line(start, y + height * 0.72, end, y + height * 0.72)
                pdf.set_dash_pattern()
            pdf.set_xy(pdf.l_margin + pdf.epw - 14, y)
            pdf.set_font(MONO, "", 9.5)
            pdf.cell(w=14, h=height, text=str(page), align="R", link=link, new_x="LMARGIN", new_y="NEXT")

    def sample_sheet(self) -> None:
        """How the sample was drawn and how it became the training and held-out rows."""
        self.section("Sample sheet", new_page=True)
        run = self.run
        summary = self.summary
        self.para(f"Source: {run.data_request.describe()}.")
        if summary is not None:
            self.subheading("Cleaning and sampling")
            readings = list(summary.get("readings") or [])
            self.table(["Reading", "Value", "Note"],
                       [[str(r.get("Reading", "")), _count(r.get("Value")), str(r.get("Note", "") or "")]
                        for r in readings],
                       widths=[34, 18, 60], align=["LEFT", "RIGHT", "LEFT"])
            files = list(summary.get("files") or [])
            if files:
                self.subheading("Files")
                self.table(["File", "Rows read", "Bad values", "Repeats in file", "Rows in sample", "Encoding"],
                           [[str(f.get("File", "")), _count(f.get("Rows read")), _count(f.get("Bad-value rows")),
                             _count(f.get("Duplicates in file")), _count(f.get("Rows in sample")),
                             str(f.get("Encoding", ""))] for f in files],
                           widths=[52, 18, 16, 18, 18, 14], align=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "LEFT"])
            classes = list(summary.get("classes") or [])
            if classes:
                self.subheading("Classes before and after sampling")
                self.table(["Class", "Available", "In sample", "Share of sample"],
                           [[str(c.get("Class", "")), _count(c.get("Available")), _count(c.get("In sample")),
                             viz.share_text(float(c.get("Share of sample") or 0.0))] for c in classes],
                           widths=[50, 20, 20, 20], align=["LEFT", "RIGHT", "RIGHT", "RIGHT"],
                           marks=["normal" if viz.kind_of(str(c.get("Class", ""))) == viz.KIND_NORMAL else "attack"
                                  for c in classes])
                self.note("Available: rows left after cleaning and removing repeats, before sampling. The sampler "
                          "gives every class a floor first, so rare classes keep more of their rows.")
                before = summary.get("before") or {}
                after = summary.get("after") or {}
                if before or after:
                    self.chart("rows per class", lambda: viz.class_distribution_chart(before, after, "light"))
        else:
            self.note(self.summary_note)
            self.step("Sample sheet")
        self._sample_to_fit()

    def _sample_to_fit(self) -> None:
        """The rows from the prepared sample to the training and held-out split (from the run's reports)."""
        reports = self.run.data.reports or {}
        rows = reports.get("rows") or {}
        self.subheading("From the sample to the fit")
        lines: list[tuple[str, str]] = []
        if rows:
            lines += [
                ("Rows in the sample", _count(rows.get("prepared"))),
                ("In the target", _count(rows.get("in_target")) + " (after the class filters of the mode)"),
                ("After removing repeats", _count(rows.get("after_dedupe")) + " (identical over the chosen columns "
                                                                                "and target)"),
                ("Used", _count(rows.get("used"))),
                ("Training rows", _count(rows.get("train"))),
                ("Held-out rows", _count(rows.get("test"))),
            ]
        else:
            lines += [("Training rows", f"{len(self.run.data.y_train):,}"),
                      ("Held-out rows", f"{len(self.run.data.y_test):,}")]
        target = reports.get("target") or {}
        dropped = {**(target.get("dropped") or {}), **(target.get("dropped_after_dedupe") or {})}
        reasons = {**(target.get("dropped_reasons") or {}), **(target.get("dropped_after_dedupe_reasons") or {})}
        if dropped:
            lines.append(("Classes left out", "; ".join(
                f"{name} ({int(count):,} rows{', ' + str(reasons[name]) if name in reasons else ''})"
                for name, count in dropped.items())))
        conflicts = reports.get("conflicts") or {}
        if conflicts.get("groups"):
            lines.append(("Conflicting labels", f"{int(conflicts.get('rows', 0)):,} rows in "
                                                f"{int(conflicts['groups']):,} groups of identical flows with "
                                                f"different labels; policy {conflicts.get('policy', 'keep')}, "
                                                f"{int(conflicts.get('rows_removed', 0)):,} removed"))
        self.facts(lines)

    def fit_settings(self) -> None:
        """Mode, columns, weighting, caps, split and cleaning choices, then every channel's notes."""
        self.section("Fit settings", need=90)
        run = self.run
        request = run.request
        data_request = run.data_request
        svm = run.channels.get("svm")
        svm_text = f"cap {int(request.svm_cap):,} rows"
        if svm is not None and svm.status == "ok":
            extra = svm.extra or {}
            svm_text += (f"; trained on {int(svm.rows_used):,} of {int(svm.rows_available):,} training rows, "
                         f"probabilities calibrated on {int(extra.get('calibration_rows', 0) or 0):,} other rows")
        elif "svm" not in run.channels:
            svm_text += " (CH3 not fitted)"
        weighting = ("Balanced sample weights for every channel (rows of rare classes count more), capped at 100 "
                     "per row (50 for the neural net) and rescaled to the row count" if request.balanced
                     else "None: every training row counts once")
        lines = [
            ("Mode", self._mode_text()),
            ("Feature set", self._feature_text()),
            ("Weighting", weighting),
            ("SVM", svm_text),
            ("Test share", f"{float(request.test_share):.0%} of the rows held out ({len(run.data.y_test):,} rows)"),
            ("Seed", str(int(request.seed))),
            ("Bad values", STRATEGY_TEXT.get(data_request.nonfinite_strategy, data_request.nonfinite_strategy)),
            ("Conflicting labels", f"{request.conflict_policy} (identical flows with different labels)"),
            ("Web Attack types", "merged into one class" if data_request.merge_web_attacks else "kept apart"),
            ("Row budget", f"{int(data_request.row_budget):,} rows"),
        ]
        if request.mode == "multiclass":
            lines.append(("Smallest class", f"{int(request.min_class_count):,} rows (smaller classes left out)"))
        if request.profile != "full":
            lines.append(("Model profile", f"{request.profile} (much smaller models, for quick checks)"))
        threshold = _number(self.settings.get("alert_threshold"))
        if threshold is not None:
            lines.append(("Alert threshold", f"{THRESHOLD_FORMAT.format(threshold)} (the Bench default at 04 Probe, "
                                             "05 Assay and 06 Sweep: an alert is " + ALERT_RULE + ")"))
        lines.append(("Fit time", f"{float(getattr(run, 'total_seconds', run.seconds)):,.1f} s on this machine"))
        self.facts(lines)
        self.subheading(f"Columns ({len(run.data.feature_names)})")
        self.para(", ".join(str(n) for n in run.data.feature_names), size=7.8, family=MONO, height=3.8)
        degenerate = (run.data.reports or {}).get("degenerate") or {}
        pool = set(CURATED) if str(getattr(run.data.feature_choice, "mode", "")) == "curated" else None
        constant = [c for c in degenerate.get("constant") or [] if pool is None or c in pool]
        twins = {k: v for k, v in dict(degenerate.get("duplicate_of") or {}).items() if pool is None or k in pool}
        if constant or twins:
            parts = []
            if constant:
                parts.append(f"constant in the sample: {', '.join(constant)}")
            if twins:
                parts.append("exact copies of another column: " + ", ".join(f"{k} (= {v})" for k, v in twins.items()))
            self.note("Left out of the candidates because they cannot separate anything: " + "; ".join(parts) + ".")
        self.subheading("Channel notes")
        lines = []
        for key in evaluate.CHANNEL_ORDER:
            result = run.channels.get(key)
            if result is None:
                continue
            words = []
            if result.status != "ok":
                words.append(f"status {result.status.replace('_', ' ')}")
            if result.error:
                words.append(str(result.error).strip().splitlines()[0])
            # Notes are full sentences; strip their own full stop so joining them never doubles it.
            words += [str(n).strip().rstrip(".") for n in result.notes if str(n).strip()]
            if result.status == "ok":
                words.insert(0, f"fitted in {float(result.fit_seconds):,.2f} s on {int(result.rows_used):,} rows")
            lines.append((evaluate.channel_label(key), (". ".join(words) + ".") if words else "no notes"))
        self.facts(lines, key_width=40, size=8.6)

    def readings(self) -> None:
        """The leaderboard, its dot plot and the per-class readings of the best channel."""
        self.section("Readings", need=150)
        run = self.run
        if not self.evals:
            self._recorded_readings()
            return
        counts = evaluate.held_out_counts(run)
        total = sum(counts.values())
        if len(counts) <= 4:
            items: list[tuple[str | None, str]] = [(None, f"Measured on {total:,} held-out rows:")]
            items += [("normal" if viz.kind_of(name) == viz.KIND_NORMAL else "attack", f"{name} {count:,}")
                      for name, count in counts.items()]
            self.marked_line(items, size=9.2, gap=2)
        else:
            self.para(f"Measured on {total:,} held-out rows in {len(counts)} classes (rows per class: see the "
                      "per-class table).")
        self.note("Balanced accuracy is the mean recall over the classes, so a rare class counts as much as a common "
                  "one; it comes first. Gap: best balanced accuracy minus the channel's.")
        board = self.board
        # The layout follows the run's MODE (a multi-class run may hold just two classes, and its leaderboard then
        # has the multi-class columns), and every heading is derived from the column it heads, so the headings and
        # the cells can never drift apart.
        scores = [title for _, title in evaluate.score_columns(run.request.mode)]
        wanted = LEADERBOARD_BINARY if run.request.mode == "binary" else LEADERBOARD_MULTICLASS
        columns = [c for c in wanted if c in scores and c in board.columns]
        heads = [SHORT_HEADS.get(c, c) for c in columns] + ["Gap"]
        rows = [[str(r["Channel"]), *[_score(r[c]) for c in columns], _score(r["Gap to best"])]
                for _, r in board.iterrows()]
        bold: list[int] = []
        consensus = consensus_metrics(run, list(self.evals))
        if consensus is not None:
            metric_of = {title: metric for metric, title in evaluate.score_columns(run.request.mode)}
            best = float(board["Balanced accuracy"].max())
            rows.append([f"Consensus ({consensus['voters']})",
                         *[_score(consensus.get(metric_of.get(c, ""))) for c in columns],
                         _score(best - float(consensus["balanced_accuracy"]))])
        if any(len(row) != len(heads) + 1 for row in rows):  # a guard: never print numbers under the wrong names
            raise RuntimeError("The leaderboard table of the record has rows that do not match its headings.")
            bold.append(len(rows) - 1)
        self.table(["Channel", *heads], rows, widths=[34] + [12] * (len(heads)),
                   align=["LEFT"] + ["RIGHT"] * len(heads), bold_rows=bold, size=7.6)
        if consensus is not None:
            self.note(f"Consensus: the equal-weight mean of the {consensus['voters']} channels' probabilities, read "
                      f"at its most probable class; every channel agreed on {consensus['unanimous']:.1%} of the "
                      "held-out rows.")
        self.chart("readings by channel", lambda: viz.leaderboard_chart(
            board, "light", subtitle=f"Each mark is one channel's reading on {total:,} held-out rows."))
        best = self._best()
        if best is not None and best[0] in self.evals:
            key = best[0]
            ev = self.evals[key]
            self.subheading(f"Per class: {evaluate.channel_label(key)} (best balanced accuracy)")
            table = ev.per_class
            self.table(["Class", "Held-out rows", "Precision", "Recall", "F1", "ROC-AUC", "Avg prec."],
                       [[str(r["class"]), _count(r["support"]), _score(r["precision"]), _score(r["recall"]),
                         _score(r["f1"]), _score(r["roc_auc"]), _score(r["average_precision"])]
                        for _, r in table.iterrows()],
                       widths=[40, 16, 13, 13, 13, 13, 13], align=["LEFT"] + ["RIGHT"] * 6,
                       marks=["normal" if viz.kind_of(str(c)) == viz.KIND_NORMAL else "attack" for c in table["class"]])
            self.note("ROC-AUC and average precision score each class against all the others (n/a when a class "
                      "has no held-out rows).")

    def _recorded_readings(self) -> None:
        """Readings saved with a run that was loaded without its held-out rows."""
        self.note("This run was loaded from disk without its held-out rows, so nothing could be measured again. The "
                  "readings below are those recorded when the channels were fitted; the charts that need the "
                  "held-out rows are left out. Load the run again at the Logbook with its data folder available "
                  "to include them.")
        rows = []
        for key in self.run.ok_channels():
            metrics = (self.run.channels[key].extra or {}).get("metrics") or {}
            rows.append([evaluate.channel_label(key), _score(metrics.get("balanced_accuracy")),
                         _score(metrics.get("accuracy")), _score(metrics.get("f1_macro")),
                         _score(metrics.get("f1_weighted"))])
        if rows:
            self.table(["Channel", "Bal. acc.", "Accuracy", "F1 macro", "F1 weighted"], rows,
                       widths=[40, 15, 15, 15, 15], align=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])
        self.step("Recorded readings")

    def confusions(self) -> None:
        """Every channel's confusion matrix (rows: true class, columns: verdict; shaded by row share)."""
        if not self.evals:
            return
        self.section("Confusion matrices", need=110)
        self.note("Rows: true class. Columns: the channel's verdict. Each cell shows the share of its row and the "
                  "row count; the diagonal holds the flows read correctly.")
        images = [self.render(f"confusion matrix of {evaluate.channel_label(key)}",
                              lambda key=key: viz.confusion_chart(self.evals[key].confusion, self.classes, "light",
                                                                  show="share", title=evaluate.channel_label(key)))
                  for key in self.evals]
        k = len(self.classes)
        self.place(images, per_row=2 if k <= 4 else 1, max_width=None if k <= 4 else 120 if k <= 8 else 165)

    def curves(self) -> None:
        """ROC and precision-recall curves (all channels for a binary run, per class for the best channel)."""
        if not self.evals:
            return
        self.section("Curves", need=130)
        run = self.run
        if self.binary:
            name = self.classes[1]
            roc = {k: ev.roc[name] for k, ev in self.evals.items() if name in ev.roc}
            pr = {k: ev.pr[name] for k, ev in self.evals.items() if name in ev.pr}
            aucs = {k: ev.metrics.get("roc_auc", math.nan) for k, ev in self.evals.items()}
            aps = {k: ev.metrics.get("average_precision", math.nan) for k, ev in self.evals.items()}
            chance = float(np.mean(np.asarray(run.data.y_test) == 1)) if len(run.data.y_test) else None
            if not roc:
                self.note("No curve could be drawn: the held-out rows hold only one class.")
                return
            self.note(f"Curves sweep every threshold on the probability of {name}; the readings use each channel's "
                      "own verdict (its most probable class).")
            self.chart("ROC curves", lambda: viz.roc_chart(roc, "light", scores=aucs), max_width=CURVE_WIDTH)
            self.chart("precision-recall curves", lambda: viz.pr_chart(pr, "light", scores=aps, chance=chance),
                       max_width=CURVE_WIDTH)
            self.note(f"03 Measure can zoom the ROC plot into the low false-positive corner (up to {ROC_ZOOM:.0%}), "
                      "where near-perfect channels differ.")
            return
        best = self._best()
        key = best[0] if best is not None and best[0] in self.evals else next(iter(self.evals))
        ev = self.evals[key]
        table = ev.per_class.set_index("class")
        support = {str(k): int(v) for k, v in table["support"].items()}
        label = evaluate.channel_label(key)
        self.note(f"One-vs-rest curves of {label}, the channel with the best balanced accuracy: each class against "
                  "all the others. The other channels' per-class readings are in the CSV exports.")
        self.chart("one-vs-rest ROC curves", lambda: viz.class_curves_chart(
            ev.roc, "light", kind="roc", support=support, title=f"{label}: one-vs-rest ROC",
            scores={str(k): float(v) for k, v in table["roc_auc"].items()}), max_width=CURVE_WIDTH)
        self.chart("one-vs-rest precision-recall curves", lambda: viz.class_curves_chart(
            ev.pr, "light", kind="pr", support=support, title=f"{label}: one-vs-rest precision-recall",
            scores={str(k): float(v) for k, v in table["average_precision"].items()}), max_width=CURVE_WIDTH)

    def importance(self) -> None:
        """Built-in importance of the tree channels and any permutation importance measured at 03 Measure."""
        native = {}
        for key in ("forest", "xgboost"):
            result = self.run.channels.get(key)
            if result is None or result.status != "ok" or result.estimator is None:
                continue
            ev = self.evals.get(key)
            frame = ev.native_importance if ev is not None else evaluate.native_importance(
                result.estimator, key, self.run.data.feature_names)
            if frame is not None and not frame.empty:
                native[key] = frame
        permutations = {k: v for k, v in self.extras.permutations.items() if isinstance(v, pd.DataFrame)
                        and not v.empty}
        if not native and not permutations:
            return
        self.section("Feature importance", need=130)
        self.note(f"The {IMPORTANCE_TOP} features with the highest importance per chart. Built-in importance comes "
                  "from the fitted trees (random forest: mean decrease in impurity; XGBoost: total gain). "
                  "Permutation importance, when measured at 03 Measure, is the drop in balanced accuracy when a "
                  "feature is shuffled among held-out rows.")
        for key, frame in native.items():
            what = "mean decrease in impurity" if key == "forest" else "total gain over all splits"
            colour = CHANNEL_BY_KEY[key].colour("light")
            label = evaluate.channel_label(key)
            self.chart(f"built-in importance of {label}", lambda frame=frame, what=what, colour=colour, label=label:
                       viz.importance_chart(frame, "light", top=IMPORTANCE_TOP, colour=colour,
                                            title=f"{label}: built-in importance",
                                            subtitle=f"Share of the model's {what}."))
        for key in [k for k in evaluate.CHANNEL_ORDER if k in permutations]:
            frame = permutations[key]
            label = evaluate.channel_label(key)
            rows = int(frame.attrs.get("rows", 0) or 0)
            repeats = int(frame.attrs.get("repeats", 0) or 0)
            self.chart(f"permutation importance of {label}", lambda frame=frame, label=label, rows=rows,
                       repeats=repeats: viz.importance_chart(
                           frame, "light", error="std", top=IMPORTANCE_TOP, number_format=".4f",
                           x_title="Drop in balanced accuracy when shuffled",
                           title=f"{label}: permutation importance",
                           subtitle=f"{rows:,} held-out rows, {repeats} repeats. Bar: mean drop; line: one "
                                    "standard deviation."))

    def timing(self) -> None:
        """Fit time, scoring speed and single-flow latency per channel."""
        run = self.run
        self.section("Timing", need=90)
        rows = []
        if self.evals:
            for _, r in self.board.iterrows():
                rows.append([str(r["Channel"]), _seconds(r["Fit s"]), _count(r["Rows used"]), _count(r["Flows/s"]),
                             _seconds(r["Single-flow ms"])])
        else:
            for key in run.ok_channels():
                result = run.channels[key]
                rows.append([evaluate.channel_label(key), _seconds(result.fit_seconds), _count(result.rows_used),
                             _count((result.extra or {}).get("flows_per_second")), "n/a"])
        self.table(["Channel", "Fit s", "Rows used", "Flows/s", "Single-flow ms"], rows,
                   widths=[40, 15, 17, 17, 17], align=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])
        self.note("Wall-clock times on the machine that fitted the run, while the app was running, so they move a "
                  "little from fit to fit. Flows/s: held-out flows scored in blocks of 5,000. Single-flow: median "
                  "time to score one flow on its own, which costs far more per flow than a block.")
        if not self.evals:
            self.step("Timing")
            return
        board = self.board
        self.chart("timing panels", lambda: viz.timing_panels(board, "light", title="Fit time and scoring speed"))

    def cross_validation(self) -> None:
        """The cross-validation measured at 03 Measure (training rows only)."""
        cv = self.extras.cross_validation
        if cv is None or cv.empty:
            return
        self.section("Cross-validation", need=90)
        attrs = cv.attrs or {}
        text = (f"{int(attrs.get('k', 0) or 0)} stratified folds of {int(attrs.get('rows', 0) or 0):,} training rows "
                f"(of {int(attrs.get('rows_available', 0) or 0):,}); the held-out rows played no part. Spread: "
                "sample standard deviation over the folds.")
        if attrs.get("note"):
            text += f" {attrs['note']}"
        if attrs.get("cancelled"):
            text += " The measurement was cancelled before every channel finished; only complete channels are listed."
        self.para(text, size=9)

        def spread(row: Any, stem: str) -> str:
            mean, std = _number(row.get(f"{stem} mean")), _number(row.get(f"{stem} std"))
            if mean is None:
                return "n/a"
            return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"

        rows = [[str(r["Channel"]), _count(r["Folds"]), spread(r, "Balanced accuracy"), spread(r, "Accuracy"),
                 spread(r, "F1 macro"), _seconds(r.get("Fit s mean"))] for _, r in cv.iterrows()]
        self.table(["Channel", "Folds", "Balanced accuracy", "Accuracy", "F1 macro", "Fit s"], rows,
                   widths=[37, 8, 24, 24, 24, 9], align=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])
        errors = [f"{r['Channel']}: {r['Error']}" for _, r in cv.iterrows() if isinstance(r.get("Error"), str)
                  and r.get("Error")]
        if errors:
            self.note("Failed: " + "; ".join(errors))
        folds = evaluate.cv_fold_frame(cv)
        self.chart("cross-validation spread", lambda: viz.cv_spread_chart(cv, "light", metric="Balanced accuracy",
                                                                          folds=folds))

    def assay(self) -> None:
        """The last batch scored at 05 Assay."""
        summary = self.extras.assay
        if summary is None:
            return
        self.section("Assay", need=80)
        lines = [("File", summary.source_name)] if summary.source_name else []
        lines += [("Channel", summary.channel), ("Rows scored", f"{summary.rows:,}"),
                 ("Rows with bad values", f"{summary.rows_with_bad_values:,} (scored anyway; the channels fill the "
                                          "gaps)")]
        if summary.seconds is not None:
            lines.append(("Time", f"{summary.seconds:,.2f} s"))
        if summary.labelled:
            lines.append(("Accuracy", _score(summary.accuracy)))
            lines.append(("Balanced accuracy", _score(summary.balanced_accuracy)))
        else:
            lines.append(("Labels", "the file had no Label column, so no accuracy could be measured"))
        if summary.unseen_labels:
            lines.append(("Labels the run never saw", "; ".join(f"{k} ({v:,} rows)"
                                                                 for k, v in summary.unseen_labels.items())
                          + " (left out of the accuracy)"))
        self.facts(lines)
        if summary.attacks_found is not None:
            self.marked_line([("attack", f"{summary.attacks_found:,} flows read as attacks")], size=9.2, gap=1.0)
        if summary.alerts is not None:  # on a line of its own: the rule in words is long
            self.marked_line([("alert", f"{summary.alerts:,} alerts{_alert_rule_text(summary.alert_threshold)}")],
                             size=9.2, gap=2.5)
        matrix = self._square(summary.confusion)
        if matrix is not None:
            counts, names = matrix
            self.chart("Assay confusion matrix", lambda: viz.confusion_chart(
                counts, names, "light", title="Assay: confusion matrix",
                subtitle="Rows: label in the file. Columns: verdict. Shade: row %."),
                       max_width=110 if len(names) <= 4 else 150)

    def sweep(self) -> None:
        """Where the last 06 Sweep simulation stood."""
        summary = self.extras.sweep
        if summary is None:
            return
        self.section("Sweep", need=80)
        lines = [("Channel", summary.channel), ("Stream", summary.source), ("Flows streamed", f"{summary.flows:,}")]
        if summary.ticks is not None:
            lines.append(("Ticks", f"{summary.ticks:,}"))
        lines.append(("Live accuracy", _score(summary.live_accuracy)))
        lines.append(("Live balanced accuracy", _score(summary.live_balanced_accuracy)))
        self.facts(lines)
        if summary.alerts is not None:
            self.marked_line([("alert", f"{summary.alerts:,} alerts raised"
                                        f"{_alert_rule_text(summary.alert_threshold)}")], size=9.2, gap=2.5)
        if summary.confusion is not None and summary.confusion.shape[0] == len(self.classes):
            counts = np.asarray(summary.confusion, dtype=np.int64)
            self.chart("Sweep confusion matrix", lambda: viz.confusion_chart(
                counts, self.classes, "light", title="Sweep: confusion matrix",
                subtitle="Rows: true class of the streamed flows. Columns: verdict. Shade: row %."),
                       max_width=110 if len(self.classes) <= 4 else 150)

    def notes(self) -> None:
        """Limitations of the readings, problems met while building, and the dataset citation."""
        self.section("Notes and limitations", need=100)
        run = self.run
        request = run.request
        items = [
            f"Every reading comes from one held-out split of one sample (seed {int(request.seed)}). Another seed or "
            "another sample moves the numbers; the cross-validation section, when present, shows by how much.",
        ]
        if run.data_request.source == "synthetic":
            items.append("The flows come from Graticule's own traffic generator. These readings show how well the "
                         "channels separate the simulated patterns and say nothing about real traffic.")
        else:
            items.append("CIC-IDS2017 was captured on one laboratory network over five working days in 2017. Traffic "
                         "on other networks, or later, differs in mix and behaviour, so these readings describe this "
                         "dataset rather than a deployment.")
        items += [
            "Labels are taken from the data as they are: any labelling mistakes in the source pass straight into "
            "the readings.",
            "Exact repeats were removed before the split (within and across files, then again over the chosen "
            "columns), but near-identical flows can still sit on both sides of it and flatter the readings.",
            "Accuracy follows the largest class; balanced accuracy weighs every class equally, which is why it is "
            "listed first.",
            "Probabilities are those of each model as fitted. Only CH3 is calibrated (on training rows it did not "
            "fit on), so an alert threshold is not a guaranteed error rate for any channel.",
            "Times are wall-clock seconds on the machine that fitted the run and vary between runs.",
        ]
        overlap = (run.data.reports or {}).get("topk_overlap")
        if isinstance(overlap, Mapping) and overlap.get("test_rows"):
            items.append(f"Top-K columns: {int(overlap.get('test_rows_seen_in_train', 0)):,} of "
                         f"{int(overlap['test_rows']):,} held-out rows "
                         f"({float(overlap.get('share', 0.0)):.1%}) have the same K values as some training row.")
        if request.profile != "full":
            items.append("This run used the small test profile: its models are far smaller than the standard ones, "
                         "so its readings understate what the channels can do.")
        failed = [f"{evaluate.channel_label(k)} ({r.status.replace('_', ' ')}"
                  + (f": {str(r.error).strip().splitlines()[0]}" if r.error else "") + ")"
                  for k, r in run.channels.items() if r.status != "ok"]
        if failed:
            items.append("Channels without readings: " + "; ".join(failed) + ".")
        items += self.problems
        items.append("This record holds no dataset rows: only counts, readings and charts drawn from them.")
        for item in items:
            self._bullet(item)
        self.subheading("Dataset")
        self.para("CIC-IDS2017, Canadian Institute for Cybersecurity, University of New Brunswick. If you use it, "
                  "cite the dataset authors:", size=9)
        self.para(CITATION, size=9, gap=2)

    def _bullet(self, text: str) -> None:
        """One item of a list: a small brass square, then the text."""
        pdf = self.pdf
        self.font(BODY, 9.2)
        shown = printable(text, BODY)
        lines = pdf.multi_cell(w=pdf.epw - 5, h=4.6, text=shown, dry_run=True, output="LINES")
        self.room(4.6 * max(len(lines), 1) + 1.5)
        y = pdf.get_y()
        pdf.set_fill_color(*_rgb(LIGHT.secondary))
        pdf.rect(pdf.l_margin + 0.6, y + 1.7, 1.3, 1.3, style="F")
        pdf.set_xy(pdf.l_margin + 5, y)
        pdf.multi_cell(w=pdf.epw - 5, h=4.6, text=shown, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.3)

    # ---------------------------------------------------------------- helpers
    def _best(self) -> tuple[str, float] | None:
        """The channel with the best balanced accuracy and its value (measured, else recorded at the fit)."""
        if not self.board.empty:
            row = self.board.iloc[0]
            value = _number(row["Balanced accuracy"])
            return (str(row["key"]), value) if value is not None else None
        best: tuple[str, float] | None = None
        for key in self.run.ok_channels():
            value = _number(((self.run.channels[key].extra or {}).get("metrics") or {}).get("balanced_accuracy"))
            if value is not None and (best is None or value > best[1]):
                best = (key, value)
        return best

    def _mode_text(self) -> str:
        """The detection mode in words."""
        if self.binary and self.run.request.mode == "binary":
            return "Binary: normal traffic against attacks"
        return f"Multi-class: {len(self.classes)} classes ({', '.join(self.classes)})"

    def _feature_text(self) -> str:
        """The feature set in words, with the column count and the port flag."""
        run = self.run
        choice = run.data.feature_choice
        mode = getattr(choice, "mode", run.request.feature_mode)
        text = FEATURE_MODE_TEXT.get(str(mode), str(mode))
        if mode == "topk":
            overlap = (run.data.reports or {}).get("topk_overlap") or {}
            text += f" (K = {int(run.request.top_k)}"
            if overlap.get("ranked_on_rows"):
                text += f", ranked on {int(overlap['ranked_on_rows']):,} rows"
            text += ")"
        port = "with" if getattr(choice, "include_port", run.request.include_port) else "without"
        return f"{text}; {len(run.data.feature_names)} columns, {port} Destination Port"

    def _square(self, confusion: pd.DataFrame | None) -> tuple[np.ndarray, list[str]] | None:
        """A confusion table (rows true, columns predicted) made square over the union of its labels."""
        if confusion is None or confusion.empty:
            return None
        names = [str(c) for c in confusion.index] + [str(c) for c in confusion.columns]
        order = [c for c in self.classes if c in names] + [c for c in dict.fromkeys(names) if c not in self.classes]
        frame = confusion.copy()
        frame.index = [str(c) for c in frame.index]
        frame.columns = [str(c) for c in frame.columns]
        frame = frame.reindex(index=order, columns=order, fill_value=0).fillna(0)
        return frame.to_numpy(dtype=np.int64), order

    # ---------------------------------------------------------------- the whole document
    def build(self, compress: bool) -> RenderedReport:
        """Lay out every section and return the finished PDF."""
        started = time.perf_counter()
        self.step("Loading the fonts")
        self.setup(compress)
        self.cover()
        self.sample_sheet()
        self.fit_settings()
        self.readings()
        self.confusions()
        self.curves()
        self.importance()
        self.timing()
        self.cross_validation()
        self.assay()
        self.sweep()
        self.notes()
        self.step("Writing the file")
        data = bytes(self.pdf.output())
        if self.progress is not None:
            self.progress("Done", 1.0)
        return RenderedReport(data=data, pages=count_pages(data), images=count_images(data),
                              sections=tuple(self.sections), seconds=time.perf_counter() - started,
                              problems=tuple(self.problems))


def render_report(
    run: "TrainingRun",
    evaluations: Mapping[str, "ChannelEvaluation"],
    *,
    prepared_summary: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
    extras: ReportExtras | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelToken | None = None,
    compress: bool = True,
    built_utc: datetime | None = None,
) -> RenderedReport:
    """Build the record of ``run`` and return it with its page count, image count, sections and build time.

    ``evaluations`` are the run's readings (:func:`graticule.evaluate.evaluate_run`; empty for a run loaded without
    its held-out rows, which gets a shorter record of its recorded readings). ``prepared_summary``
    (:func:`summarise_prepared`) is used only when its fingerprint is the run's. ``settings`` are the Bench settings
    as plain values. ``progress`` receives (message, fraction) as the work advances; ``cancel`` is checked between
    charts (raising :class:`~graticule.models.jobs.TrainingCancelled`). ``compress=False`` leaves the page streams
    uncompressed (for inspection in tests). Nothing is fitted and nothing is fetched.
    """
    builder = _Builder(run, evaluations, prepared_summary=prepared_summary, settings=settings,
                       extras=extras or ReportExtras(), progress=progress, cancel=cancel,
                       built_utc=built_utc or datetime.now(timezone.utc))
    return builder.build(compress)


def build_report(
    run: "TrainingRun",
    evaluations: Mapping[str, "ChannelEvaluation"],
    *,
    prepared_summary: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
    extras: ReportExtras | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelToken | None = None,
    compress: bool = True,
) -> bytes:
    """The PDF record of ``run`` as bytes (see :func:`render_report` for the arguments)."""
    return render_report(run, evaluations, prepared_summary=prepared_summary, settings=settings, extras=extras,
                         progress=progress, cancel=cancel, compress=compress).data


def report_file_name(run_id: str) -> str:
    """File name of a run's record, e.g. ``graticule-record-20261001-153012-ab12.pdf``."""
    return f"graticule-record-{run_id}.pdf"


__all__ = [
    "AssaySummary", "CITATION", "RenderedReport", "ReportExtras", "SweepSummary", "build_report", "chart_png",
    "count_images", "count_pages", "font_characters", "printable", "render_report", "report_file_name",
    "summarise_assay", "summarise_prepared", "summarise_sweep",
]
