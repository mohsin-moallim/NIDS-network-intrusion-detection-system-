"""05 Assay: score a whole CSV of flows with one fitted channel, or with the consensus of every fitted channel.

The file is read exactly as training files are (:func:`nids.data.reader.read_flow_csv`: header spaces, a
byte-order mark, the repeated ``Fwd Header Length`` column, "Infinity" cells, the three text encodings and the
Web Attack label stand-ins), except that the ``Label`` column is optional. Nothing is fitted here; the channels
only predict.

Rules the scorer keeps:

* Every uploaded row is scored and comes back, in file order. A row holding infinite or missing values in the
  columns the channels read is scored all the same and counted in ``rows_with_bad_values``: the run's own
  bad-value strategy is applied to it (``recompute`` rebuilds the two rate columns from the byte and packet totals
  and the duration, as at 01 Sample) and every channel then treats what is left exactly as it did during the fit
  (infinities become gaps; the scaled channels fill gaps with their training medians, the tree channels route them).
* A file lacking any column the channels read is refused as a whole (:class:`MissingColumnsError`); nothing is
  scored partially.
* Labels, when the file has them, are normalised as training labels are and mapped onto the run's classes: in
  binary mode BENIGN (or Normal) is Normal and any other name is Attack, and a note lists the labels read as
  Attack; in multi-class mode a label must name one of the run's classes (after the Web Attack merge, when the run
  used it). A label made only of digits (such as 0 or 1) is not read as Attack: it names no kind of traffic.
  Labels the run never trained on are counted and left out of the accuracy, as are empty labels.
* Alerts follow the rule every station shares (:func:`nids.models.verdict.alert_flags`): an attack verdict
  whose attack probability is at least the threshold.
* Rows the run has seen are named. Each scored row is matched (by a 64-bit hash over the channels' columns) with
  the run's training and held-out rows, when the run holds them; the batch counts the rows that repeat a training
  row and, for a labelled file, also gives accuracy and balanced accuracy over the rows the run never trained on.
  A file the run's own sample was drawn from is flagged by name as well. Accuracy over rows a channel was trained
  on says little about how it reads new traffic.

Memory and the scored file. The upload is read in blocks of whole lines (about ``chunk_rows`` rows each, never more
than :data:`READ_BLOCK_MAX` bytes), each block through the same reader with the file's header in front, and its
rows are scored before the next block is parsed. So besides the uploaded bytes themselves the scorer keeps only
the readings (a few numbers per row) and one parsed block at a time, however long the file. The scored CSV repeats
every uploaded line exactly as written (its own text, re-encoded as UTF-8; no value is re-formatted or rounded)
and appends the result columns. The uploaded bytes stay with the result for that download.

A file whose lines cannot be matched one to one with its rows (a quoted line break, rows of uneven length, a line
of spaces: the fast parser refuses such files and the fallback parser reads them), or whose lines end in a bare
carriage return (the old Mac format), is read whole instead. Its download then writes the columns as read (feature
columns as 32-bit numbers, which round integers beyond 16,777,216; labels tidied), and a note says so; such a file
also needs several times its own size in memory.

The result is a :class:`ScoredBatch`, which also knows how to write itself as a CSV (UTF-8 with a byte-order mark,
so spreadsheet programs read it correctly) for the download at 05 Assay; 07 Record exports its result columns.
"""

from __future__ import annotations

import codecs
import csv
import io
import re
import time
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from nids.data.clean import (
    BWD_BYTES,
    BWD_PACKETS,
    DURATION,
    FWD_BYTES,
    FWD_PACKETS,
    apply_nonfinite_strategy,
    row_hashes,
)
from nids.data.reader import (
    DataFileError,
    FileReadReport,
    clean_column_name,
    read_flow_csv,
    strip_byte_order_mark,
)
from nids.data.sampling import apply_class_options
from nids.evaluate import quick_metrics
from nids.models.jobs import CancelToken, TrainingCancelled
from nids.models.train import TrainingRun, _tidy_proba
from nids.models.verdict import ALERT_RULE, alert_flags, combine
from nids.models.verdict import attack_probability as attack_probability_of
from nids.schema import ATTACK, FEATURE_SET, LABEL, NORMAL, is_normal_traffic
from nids.theme import CHANNEL_BY_KEY, GLYPH_ALERT, score_text, verdict_text

#: Channel value that scores with the consensus of every fitted channel.
CONSENSUS = "consensus"
#: How the consensus choice is named in the app.
CONSENSUS_NAME = "Consensus of all channels"
#: Rows predicted per chunk (and, roughly, rows parsed per block of the file).
DEFAULT_CHUNK_ROWS = 50_000
#: Most rows shown by :meth:`ScoredBatch.preview`.
PREVIEW_ROWS = 200
#: Rows read to check the header before the whole file is read (fail fast on a file that cannot be scored).
HEADER_CHECK_ROWS = 5
#: Size limits (bytes) of one block of whole lines parsed at a time.
READ_BLOCK_MIN = 64 * 1024
READ_BLOCK_MAX = 32 * 1024 * 1024
#: Lines sampled after the header to estimate the length of a line.
LINE_SAMPLE = 256
#: Added to an uploaded column name that a result column (or the preview's verdict column) already uses.
UPLOADED_SUFFIX = " (uploaded)"

#: Result columns added after the uploaded columns.
PREDICTED = "predicted_label"
PROB_PREFIX = "prob_"
ATTACK_PROBABILITY = "attack_probability"
ALERT = "alert"
AGREEMENT = "channels_agreeing"
TRUE_LABEL = "true_label"
VERDICT = "Verdict"
#: Columns the ``recompute`` strategy needs besides the two rates it rebuilds.
_RATE_INPUTS: tuple[str, ...] = (DURATION, FWD_BYTES, BWD_BYTES, FWD_PACKETS, BWD_PACKETS)
#: How the C parser names a column whose header cell is empty.
_UNNAMED = re.compile(r"^Unnamed: \d+$")
#: A label that is only a number (0/1 style labels name no class: neither BENIGN nor an attack type).
_NUMERIC_LABEL = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
#: Labels named one by one in the note on labels a binary run read as Attack.
ATTACK_LABELS_SHOWN = 8

ProgressFn = Callable[[str, float], None]
UploadSource = bytes | BinaryIO | Path | str

#: How rows with infinite or missing values were treated, per bad-value strategy of the run.
_STRATEGY_NOTES: dict[str, str] = {
    "drop": ("The run's sample dropped such rows, so these flows lie outside what the channels were fitted on; the "
             "scaled channels fill the gaps with their training medians and the tree channels route them."),
    "impute": ("As during the fit, infinities became gaps, which the scaled channels fill with their training "
               "medians and the tree channels route."),
    "recompute": ("As during the fit, non-finite Flow Bytes/s and Flow Packets/s values were rebuilt from the byte "
                  "and packet totals and the duration; any other gap is filled with training medians (scaled "
                  "channels) or routed (tree channels)."),
}


class MissingColumnsError(ValueError):
    """The uploaded file lacks columns the channels read, so it cannot be scored; the message names every one.

    Attributes:
        missing: the missing column names, in the order the channels read them.
        needed: how many columns the channels read in all.
    """

    def __init__(self, missing: Sequence[str], *, needed: int, source: str = "The file") -> None:
        self.missing = [str(name) for name in missing]
        self.needed = int(needed)
        plural = "s" if len(self.missing) != 1 else ""
        of_all = "s" if self.needed != 1 else ""
        super().__init__(
            f"{source} cannot be scored: it lacks {len(self.missing)} of the {self.needed} column{of_all} the "
            f"channels read: {', '.join(self.missing)}. Header names are matched after trimming spaces; add the "
            f"missing column{plural} and score the file again, or use a run fitted on columns the file has."
        )


@dataclass(frozen=True)
class LineBlock:
    """One block of whole lines of an upload: byte range (after the header), rows parsed and the text encoding."""

    start: int
    stop: int
    rows: int
    encoding: str


@dataclass(eq=False)
class UploadedRows:
    """The uploaded file as written, kept with a scored batch so its download can repeat every row exactly.

    Attributes:
        payload: the uploaded bytes.
        header_stop: offset just past the header line.
        blocks: the blocks of whole lines the rows were parsed from, in file order (their rows add up to the batch's).
        head: the first rows (at most :data:`PREVIEW_ROWS`) as written, for the preview: known feature columns as
            exact numbers, every other column as its text; None when they could not be read back.
        read_frame: only for a file whose lines could not be matched with its rows: the columns as the reader
            returned them (feature columns float32, other columns as text, labels tidied), which its download then
            writes. None otherwise.
    """

    payload: bytes = field(repr=False)
    header_stop: int
    blocks: tuple[LineBlock, ...]
    head: pd.DataFrame | None = None
    read_frame: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def as_written(self) -> bool:
        """True when the download repeats every uploaded line exactly as written."""
        return self.read_frame is None


@dataclass
class ScoredBatch:
    """One uploaded file scored by one channel (or by the consensus of every fitted channel).

    Attributes:
        frame: the readings, one row per uploaded row (the index is each row's 0-based position among the file's
            data rows): ``predicted_label``, one ``prob_<class>`` column per class, ``attack_probability``
            (P(Attack) in binary mode, 1 - P(BENIGN) in multi-class mode), ``alert`` (an attack verdict whose attack
            probability is at least the threshold), ``channels_agreeing`` (consensus only) and ``true_label`` (when
            the file has labels: the run's class for the row, or the label as read when the run never trained on
            it). The uploaded columns themselves are not repeated here; :meth:`to_csv_bytes` writes them in front of
            these, and :meth:`preview` shows the first rows of both.
        channel: the channel key, or ``"consensus"``.
        rows: rows uploaded (= rows scored = rows in ``frame``).
        rows_with_bad_values: rows holding an infinite or missing value in a column the channels read.
        seconds: wall time of the whole assay (reading, checking and scoring).
        labelled: True when the file has a ``Label`` column with at least one non-empty label.
        accuracy, balanced_accuracy: readings over the rows whose label the run knows (None when there are none).
        confusion: counts (true class rows x predicted class columns) over those rows, or None.
        unseen_labels: labels the run never trained on, with their row counts (left out of the readings).
        run_id, classes, mode: the run that scored the file.
        source_name: the uploaded file's name.
        alert_threshold: the attack probability from which an attack verdict raises an alert.
        strategy: the run's bad-value strategy, applied to the uploaded rows.
        score_seconds: the part of ``seconds`` spent predicting.
        rows_without_label: labelled files only: rows whose label cell is empty (left out of the readings).
        rows_measured: rows the readings cover.
        voters: the channels whose probabilities were used (one, or every fitted channel for the consensus).
        extra_columns: uploaded columns that are not known flow features (kept in the download as written).
        notes: plain-language remarks on reading and scoring the file.
        predicted_counts: rows per predicted class, in class-code order.
        run_origin: ``"fitted"`` or ``"loaded"``: where the scoring run came from.
        upload: the uploaded file as written (see :class:`UploadedRows`), or None.
        run_ref: a weak reference to the scoring run object (see :meth:`made_with`), or None.
        rows_seen_in_training: rows whose values over the channels' columns repeat a training row of the run; None
            when the run does not hold its rows (a set loaded without its data folder), so no check was made.
        rows_seen_held_out: rows that repeat one of the run's held-out rows (and no training row); None likewise.
        unseen_accuracy, unseen_balanced_accuracy: readings over the labelled rows that repeat no training row
            (None when not checked, or when no such row has a label the run knows).
        rows_unseen_measured: rows those two readings cover.
        from_sample_file: True when the file's name is one of the files the run's sample was drawn from.
    """

    frame: pd.DataFrame
    channel: str
    rows: int
    rows_with_bad_values: int
    seconds: float
    labelled: bool
    accuracy: float | None
    balanced_accuracy: float | None
    confusion: pd.DataFrame | None
    unseen_labels: dict[str, int]
    run_id: str = ""
    classes: tuple[str, ...] = ()
    mode: str = "binary"
    source_name: str = ""
    alert_threshold: float = 0.9
    strategy: str = "drop"
    score_seconds: float = 0.0
    rows_without_label: int = 0
    rows_measured: int = 0
    voters: tuple[str, ...] = ()
    extra_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    predicted_counts: dict[str, int] = field(default_factory=dict)
    run_origin: str = "fitted"
    upload: UploadedRows | None = field(default=None, repr=False, compare=False)
    run_ref: Any = field(default=None, repr=False, compare=False)
    rows_seen_in_training: int | None = None
    rows_seen_held_out: int | None = None
    unseen_accuracy: float | None = None
    unseen_balanced_accuracy: float | None = None
    rows_unseen_measured: int = 0
    from_sample_file: bool = False

    @property
    def channel_name(self) -> str:
        """The channel as the app names it, e.g. ``"CH2 XGBoost"`` or ``"Consensus of all channels"``."""
        return channel_name(self.channel)

    @property
    def probability_columns(self) -> list[str]:
        """The ``prob_<class>`` column names, in class-code order."""
        return [probability_column(name) for name in self.classes]

    @property
    def result_columns(self) -> list[str]:
        """Every column the scorer added, in the order they appear in ``frame``."""
        added = [PREDICTED, *self.probability_columns, ATTACK_PROBABILITY, ALERT, AGREEMENT, TRUE_LABEL]
        return [name for name in added if name in self.frame.columns]

    @property
    def attacks(self) -> int:
        """Rows whose verdict is not normal traffic."""
        counts = self.verdict_counts()
        return int(self.rows - sum(count for name, count in counts.items() if _is_normal(name)))

    @property
    def alerts(self) -> int:
        """Rows that raised an alert (an attack verdict at or above the alert threshold)."""
        return int(self.frame[ALERT].to_numpy(dtype=bool).sum()) if self.rows else 0

    @property
    def flows_per_second(self) -> float | None:
        """Rows scored per second of prediction time (None when the time is too short to measure)."""
        return float(self.rows / self.score_seconds) if self.score_seconds > 0 else None

    @property
    def file_name(self) -> str:
        """Download name of the scored CSV: ``nids-assay-<run id>-<channel>.csv``."""
        return f"nids-assay-{self.run_id}-{self.channel}.csv"

    @property
    def values_as_written(self) -> bool:
        """True when the scored CSV repeats every uploaded line exactly as written (see :class:`UploadedRows`)."""
        return self.upload is not None and self.upload.as_written

    def made_with(self, run: Any) -> bool:
        """True when ``run`` is the very run object that scored this file.

        A fit and its copy loaded from disk share a run id but are different channel sets (the copy holds CH3 only
        when it was saved by choice, and may lack its held-out rows), so the object is compared, as 06 Sweep
        compares its stream's run.
        A batch that holds no reference (its run could not be referenced weakly) falls back to the id, the run's
        origin and its fitted channels.
        """
        if run is None:
            return False
        if self.run_ref is not None:
            return self.run_ref() is run
        if str(getattr(run, "run_id", "")) != str(self.run_id):
            return False
        if str(getattr(run, "origin", "fitted")) != str(self.run_origin):
            return False
        ok = getattr(run, "ok_channels", None)
        fitted = set(ok()) if callable(ok) else set()
        return set(self.voters) <= fitted

    def verdict_counts(self) -> dict[str, int]:
        """Rows per predicted class, in class-code order (classes never predicted show 0)."""
        if self.predicted_counts:
            return {name: int(self.predicted_counts.get(name, 0)) for name in self.classes}
        counts = self.frame[PREDICTED].value_counts() if self.rows else pd.Series(dtype="int64")
        return {name: int(counts.get(name, 0)) for name in self.classes}

    def uploaded_head(self) -> pd.DataFrame | None:
        """The first uploaded rows (at most :data:`PREVIEW_ROWS`) as written, or None when not available."""
        if self.upload is None:
            return None
        if self.upload.read_frame is not None:
            return self.upload.read_frame.iloc[:PREVIEW_ROWS]
        return self.upload.head

    def preview(self, n: int = PREVIEW_ROWS) -> pd.DataFrame:
        """The first ``n`` scored rows (at most :data:`PREVIEW_ROWS`), led by a ``Verdict`` column with shape cues
        ("○ Normal", "◆ DoS Hulk", "▲ Alert · ◆ Attack"), then the result columns, then the uploaded columns.

        Every column name is unique: an uploaded column whose name the verdict or a result column already uses is
        shown as ``"<name> (uploaded)"``.
        """
        count = max(0, min(int(n), self.rows, PREVIEW_ROWS))
        head = self.frame.iloc[:count]
        verdicts = [
            (f"{GLYPH_ALERT} Alert · " if bool(alert) else "") + verdict_text(str(label))
            for label, alert in zip(head[PREDICTED].tolist(), head[ALERT].tolist())
        ]
        parts = [pd.DataFrame({VERDICT: pd.Series(verdicts, index=head.index, dtype="str")}), head]
        uploaded = self.uploaded_head()
        if uploaded is not None and len(uploaded.columns):
            shown = uploaded.iloc[:count].set_axis(head.index[: min(count, len(uploaded))], axis=0)
            names = unique_names([str(c) for c in shown.columns], taken={VERDICT, *map(str, head.columns)})
            parts.append(shown.set_axis(names, axis=1))
        return pd.concat(parts, axis=1)

    def to_csv_bytes(self) -> bytes:
        """The scored file as CSV bytes: UTF-8 with a byte-order mark, no index column.

        Every uploaded line comes first, exactly as written (re-encoded as UTF-8), followed by the result columns;
        an uploaded column named like a result column gets the suffix ``" (uploaded)"`` in the header. A file read
        whole (see :class:`UploadedRows`) writes its columns as read instead. Result cells are written by Arrow's
        CSV writer (text quoted, the alert flag as ``true``/``false``).
        """
        buffer = io.BytesIO()
        buffer.write(codecs.BOM_UTF8)
        if self.upload is None:
            _write_frame(self.frame, buffer)
        elif self.upload.read_frame is not None:
            uploaded = self.upload.read_frame
            names = unique_names([str(c) for c in uploaded.columns], taken=set(map(str, self.frame.columns)))
            _write_frame(pd.concat([uploaded.set_axis(names, axis=1), self.frame], axis=1), buffer)
        else:
            _write_spliced(self.upload, self.frame, buffer)
        return buffer.getvalue()


# --------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------
def channel_name(channel: str) -> str:
    """How the app names a scoring choice: ``"CH1 Random forest"``, or ``"Consensus of all channels"``."""
    if channel == CONSENSUS:
        return CONSENSUS_NAME
    style = CHANNEL_BY_KEY.get(channel)
    return style.label if style is not None else channel


def channel_choices(run: TrainingRun) -> list[str]:
    """Scoring choices for ``run``: every fitted channel in channel order, then the consensus when there are two
    or more."""
    keys = list(run.ok_channels())
    return keys + [CONSENSUS] if len(keys) >= 2 else keys


def probability_column(class_name: str) -> str:
    """Name of the probability column of a class, e.g. ``"prob_DoS Hulk"``."""
    return f"{PROB_PREFIX}{class_name}"


def unique_names(names: Sequence[str], *, taken: Iterable[str]) -> list[str]:
    """``names`` with every name that is in ``taken`` (or repeats an earlier one) renamed ``"<name> (uploaded)"``
    (then ``"<name> (uploaded 2)"`` and so on), so the result holds no name twice and none of ``taken``."""
    blocked = set(taken)
    used: set[str] = set()
    out: list[str] = []
    for name in names:
        candidate = name
        if candidate in blocked or candidate in used:
            candidate = f"{name}{UPLOADED_SUFFIX}"
            number = 2
            while candidate in blocked or candidate in used:
                candidate = f"{name} (uploaded {number})"
                number += 1
        used.add(candidate)
        out.append(candidate)
    return out


def _is_normal(name: str) -> bool:
    """True for the normal-traffic class under either naming (``BENIGN`` or ``Normal``, any letter case)."""
    return is_normal_traffic(name)


def _normal_index(classes: Sequence[str]) -> int | None:
    """Code of the normal-traffic class, or None when the run has none."""
    return next((i for i, name in enumerate(classes) if _is_normal(str(name))), None)


def _clean_name(name: object) -> str:
    """A header name without a byte-order mark or surrounding spaces (as the reader matches names)."""
    return clean_column_name(name)


def _payload(source: UploadSource, name: str | None) -> tuple[bytes, str]:
    """The raw bytes of the upload (a file on disk is read whole) plus a display name."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DataFileError(f"File not found: {path}")
        try:
            return path.read_bytes(), name or path.name
        except PermissionError as exc:
            raise DataFileError(f"{path.name} cannot be opened: another program may be holding it, or access is "
                                "denied. Close it elsewhere and try again.") from exc
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), name or "uploaded file"
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except (OSError, ValueError):
                pass
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return bytes(data), name or str(getattr(source, "name", "") or "uploaded file")
    raise TypeError(f"Cannot score flows from a {type(source).__name__}.")


def _decode(data: bytes, encoding: str) -> str:
    """Bytes of the upload as text (a byte the encoding cannot read becomes U+FFFD, as the reader shows it)."""
    return data.decode(encoding or "utf-8", errors="replace")


def _text_lines(text: str) -> list[str]:
    """The non-empty lines of a block of text, without their line ends (the parsers skip empty lines too)."""
    return [line for line in (part.rstrip("\r") for part in text.split("\n")) if line]


def _count_lines(data: bytes) -> int:
    """How many non-empty lines a block of bytes holds."""
    return sum(1 for part in data.split(b"\n") if part.rstrip(b"\r"))


def _csv_cells(cells: Sequence[str]) -> str:
    """One CSV line (no line end) holding ``cells``, quoted only where needed."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(list(cells))
    return buffer.getvalue()


# --------------------------------------------------------------------------------------------------------------
# Splitting the upload into blocks of whole lines
# --------------------------------------------------------------------------------------------------------------
def _header_stop(payload: bytes) -> int:
    """Offset just past the header line (the whole payload when it holds a single line)."""
    end = payload.find(b"\n")
    return len(payload) if end < 0 else end + 1


def _bare_carriage_returns(payload: bytes, header_stop: int) -> bool:
    """True when the file's lines end in a bare carriage return (the old Mac format, still written by some
    spreadsheet programs): the "header line" up to the first line feed then holds carriage returns inside it."""
    return b"\r" in payload[:header_stop].rstrip(b"\r\n")


def _line_blocks(payload: bytes, header_stop: int, chunk_rows: int) -> list[tuple[int, int]]:
    """Byte ranges of blocks of whole lines after the header, each about ``chunk_rows`` lines long (bounded by
    :data:`READ_BLOCK_MIN` and :data:`READ_BLOCK_MAX` bytes)."""
    size = len(payload)
    if header_stop >= size:
        return []
    probe = header_stop
    sampled = 0
    while sampled < LINE_SAMPLE and probe < size:
        nxt = payload.find(b"\n", probe)
        probe = size if nxt < 0 else nxt + 1
        sampled += 1
    line_bytes = (probe - header_stop) / max(sampled, 1)
    target = int(min(max(line_bytes * max(int(chunk_rows), 1), READ_BLOCK_MIN), READ_BLOCK_MAX))
    blocks: list[tuple[int, int]] = []
    start = header_stop
    while start < size:
        stop = min(start + target, size)
        if stop < size:
            end = payload.find(b"\n", stop - 1)
            stop = size if end < 0 else end + 1
        blocks.append((start, stop))
        start = stop
    return blocks


def _head_rows(payload: bytes, header_stop: int, encoding: str, rows: int) -> pd.DataFrame | None:
    """The first ``rows`` uploaded rows as written: known feature columns as exact numbers (64-bit), every other
    column as its text; None when they cannot be read back."""
    if rows <= 0:
        return None
    position = header_stop
    found = 0
    size = len(payload)
    while found < rows and position < size:
        end = payload.find(b"\n", position)
        stop = size if end < 0 else end + 1
        if payload[position:stop].strip(b"\r\n"):
            found += 1
        position = stop
    try:
        head = pd.read_csv(io.BytesIO(payload[:position]), engine="c", encoding=encoding or "utf-8",
                           encoding_errors="replace", dtype="str", keep_default_na=False, index_col=False)
    except Exception:  # noqa: BLE001 - the preview of the uploaded columns is a courtesy
        return None
    if len(head) != min(rows, found):
        return None
    head.columns = [_clean_name(c) for c in head.columns]
    for column in head.columns:
        if column in FEATURE_SET:
            head[column] = pd.to_numeric(head[column], errors="coerce")
    return head.reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------
# Writing the scored file
# --------------------------------------------------------------------------------------------------------------
def _write_frame(frame: pd.DataFrame, buffer: BinaryIO) -> None:
    """Write a whole frame as CSV (header included) after the byte-order mark already in ``buffer``.

    Arrow's CSV writer is used (about five times faster than pandas on wide float frames); should Arrow refuse a
    column, pandas writes the file instead (flags as ``true``/``false``, as Arrow writes them).
    """
    start = buffer.tell()
    try:
        import pyarrow as pa
        import pyarrow.csv as pa_csv

        pa_csv.write_csv(pa.Table.from_pandas(frame, preserve_index=False), buffer)
    except Exception:  # noqa: BLE001 - any Arrow conversion problem falls back to the slower writer
        buffer.seek(start)
        buffer.truncate()
        buffer.write(_flags_as_text(frame).to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _flags_as_text(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with its true/false columns written as ``true``/``false`` (as Arrow's writer spells them)."""
    flags = [c for c in frame.columns if pd.api.types.is_bool_dtype(frame[c])]
    if not flags:
        return frame
    out = frame.copy()
    for column in flags:
        out[column] = frame[column].map({True: "true", False: "false"})
    return out


def _result_lines(results: pd.DataFrame) -> list[str]:
    """The result cells of each row as CSV text (one string per row, no line end)."""
    if results.empty:
        return []
    try:
        import pyarrow as pa
        import pyarrow.csv as pa_csv

        sink = io.BytesIO()
        pa_csv.write_csv(pa.Table.from_pandas(results, preserve_index=False), sink,
                         pa_csv.WriteOptions(include_header=False))
        text = sink.getvalue().decode("utf-8")
    except Exception:  # noqa: BLE001 - fall back to pandas
        text = _flags_as_text(results).to_csv(index=False, header=False, lineterminator="\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _write_spliced(upload: UploadedRows, results: pd.DataFrame, buffer: BinaryIO) -> None:
    """Write each uploaded line exactly as written, followed by its result cells."""
    names = [str(c) for c in results.columns]
    first = upload.blocks[0].encoding if upload.blocks else "utf-8"
    header = strip_byte_order_mark(_decode(upload.payload[: upload.header_stop], first)).rstrip("\r\n")
    # Only a name a result column uses is changed; everything else in the header (spaces, a repeated column) stays
    # as written.
    taken = set(names)
    cells = next(csv.reader([header]), [])
    if any(_clean_name(cell) in taken for cell in cells):
        header = _csv_cells([f"{_clean_name(cell)}{UPLOADED_SUFFIX}" if _clean_name(cell) in taken else cell
                             for cell in cells])
    buffer.write(f"{header},{_csv_cells(names)}\n".encode("utf-8"))
    position = 0
    for block in upload.blocks:
        lines = _text_lines(_decode(upload.payload[block.start:block.stop], block.encoding))
        cells_of = _result_lines(results.iloc[position:position + block.rows])
        position += block.rows
        if len(lines) != len(cells_of):  # checked while scoring, so this means the batch was altered since
            raise RuntimeError("The uploaded lines no longer match the scored rows; score the file again.")
        buffer.write("".join(f"{line},{cell}\n" for line, cell in zip(lines, cells_of)).encode("utf-8"))


# --------------------------------------------------------------------------------------------------------------
# Labels, matrices and probabilities
# --------------------------------------------------------------------------------------------------------------
def _extra_columns(payload: bytes, encoding: str, names: Sequence[str], index: pd.Index
                   ) -> tuple[pd.DataFrame | None, str | None]:
    """For a file read whole: the uploaded columns that are not known features, as text exactly as written;
    (frame, problem note).

    The fast reader keeps only the known columns, so these are read in a second, column-limited pass. A column
    whose header cell is empty is named "" by the fast parser and "Unnamed: <n>" by this one; the two are matched.
    """
    wanted = set(names)
    unnamed = "" in wanted

    def keep(column: object) -> bool:
        clean = _clean_name(column)
        return clean in wanted or (unnamed and bool(_UNNAMED.match(clean)))

    try:
        raw = pd.read_csv(io.BytesIO(payload), engine="c", encoding=encoding, encoding_errors="replace",
                          index_col=False, dtype="str", keep_default_na=False, usecols=keep)
    except Exception as exc:  # noqa: BLE001 - extra columns are a courtesy; scoring goes on without them
        first = str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
        return None, f"The other columns could not be kept in the output ({first})."
    if raw.shape[1] == 0:
        return None, None
    raw.columns = ["" if _UNNAMED.match(_clean_name(c)) and unnamed else _clean_name(c) for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.duplicated()]
    if len(raw) != len(index):
        return None, ("The other columns could not be kept in the output: a second reading of the file found "
                      f"{len(raw):,} rows instead of {len(index):,}.")
    raw = raw[[n for n in names if n in raw.columns]]
    raw.index = index
    return raw, None


def _map_labels(labels: pd.Series, run: TrainingRun
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Map uploaded labels onto the run's classes; returns (codes with -1 for unknown, shown true labels, empty,
    rows per label a binary run read as Attack).

    Labels arrive already normalised by the reader. The run's Web Attack merge is applied first; then binary runs
    read BENIGN (or "Normal") as Normal and every other name as Attack, while multi-class runs need the label to
    name one of their classes (an exact match first, then one that differs only in letter case; "Normal" stands for
    BENIGN). In binary mode a label made only of digits is unknown: it says nothing about the kind of traffic (a
    multi-class run reads it only when it is one of the run's own classes).
    """
    classes = [str(c) for c in run.data.classes]
    merged = apply_class_options(labels.astype("str"), merge_web_attacks=bool(run.data_request.merge_web_attacks))
    values = pd.Series(merged.to_numpy(dtype=object, na_value=""), dtype=object)
    code_of = {name: i for i, name in enumerate(classes)}
    folded = {name.casefold(): i for i, name in enumerate(classes)}
    normal = _normal_index(classes)
    binary = run.request.mode == "binary" and ATTACK in code_of and NORMAL in code_of
    lookup: dict[str, int] = {}
    blank: dict[str, bool] = {}
    read_as_attack: list[str] = []
    for value in values.unique():
        text = str(value)
        blank[text] = text.strip() == ""
        if text.strip() == "":
            lookup[text] = -1
        elif binary:
            if _NUMERIC_LABEL.match(text.strip()):
                lookup[text] = -1
            elif _is_normal(text):
                lookup[text] = code_of[NORMAL]
            else:
                lookup[text] = code_of[ATTACK]
                read_as_attack.append(text)
        elif text in code_of:
            lookup[text] = code_of[text]
        elif text.casefold() in folded:
            lookup[text] = folded[text.casefold()]
        elif _is_normal(text) and normal is not None:
            lookup[text] = normal
        else:
            lookup[text] = -1
    # One dictionary lookup per row, done by pandas (each distinct label was mapped once above).
    codes = values.map(lookup).to_numpy(dtype=np.int64)
    empty = values.map(blank).to_numpy(dtype=bool)
    names = np.asarray(classes, dtype=object)
    shown = np.where(codes >= 0, names[np.clip(codes, 0, None)], values.to_numpy(dtype=object))
    as_attack: dict[str, int] = {}
    if read_as_attack:
        counts = values[values.isin(read_as_attack)].value_counts()
        as_attack = {str(k): int(v) for k, v in counts.items()}
    return codes, shown.astype(str), empty, as_attack


def _chunk_matrix(columns: dict[str, np.ndarray], features: Sequence[str], start: int, stop: int,
                  strategy: str, extras: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``start:stop`` of the channels' columns as float32, with the run's bad-value strategy applied.

    Returns (matrix, rows holding a non-finite value before the strategy was applied).
    """
    raw = np.empty((stop - start, len(features)), dtype=np.float32)
    for j, name in enumerate(features):
        raw[:, j] = columns[name][start:stop]
    bad = ~np.isfinite(raw).all(axis=1)
    if strategy != "recompute" or not bad.any():
        return raw, bad
    block = pd.DataFrame(raw, columns=list(features), copy=False)
    for name, values in extras.items():
        if name not in block.columns:
            block[name] = values[start:stop]
    fixed, _ = apply_nonfinite_strategy(block, "recompute", label_col=LABEL)
    return fixed[list(features)].to_numpy(dtype=np.float32, copy=True), bad


def _channel_proba(estimator: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    """One channel's class probabilities (float32, n x K, code order, rows summing to 1)."""
    return _tidy_proba(estimator.predict_proba(X), getattr(estimator, "classes_", None), n_classes)


def _voters(run: TrainingRun, channel: str) -> tuple[str, ...]:
    """The channels whose probabilities make the reading for ``channel`` (validated against the run)."""
    fitted = list(run.ok_channels())
    if channel == CONSENSUS:
        if not fitted:
            raise ValueError(f"Run {run.run_id} has no fitted channel to score with.")
        return tuple(fitted)
    if channel not in fitted:
        raise ValueError(f"{channel_name(channel)} is not a fitted channel of run {run.run_id}; choose one of "
                         f"{', '.join(channel_name(k) for k in channel_choices(run)) or 'none'}.")
    return (channel,)


def _matrix_hashes(X: np.ndarray, features: Sequence[str]) -> np.ndarray:
    """64-bit hashes of the rows of a float32 matrix over ``features`` (as de-duplication hashes rows)."""
    return row_hashes(pd.DataFrame(X, columns=list(features), copy=False), list(features))


#: Attribute of a run holding the hashes of its training and held-out rows (computed once per run object).
ROW_HASHES_ATTR = "assay_row_hashes"


def _run_row_hashes(run: TrainingRun) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique hashes of the run's training rows and of its held-out rows, kept on the run object."""
    kept = run.__dict__.get(ROW_HASHES_ATTR)
    if isinstance(kept, tuple) and len(kept) == 2:
        return kept
    features = [str(f) for f in run.data.feature_names]
    train = np.unique(_matrix_hashes(np.asarray(run.data.X_train, dtype=np.float32), features))
    held_out = getattr(run.data, "X_test", None)
    held_out = np.empty((0, len(features)), dtype=np.float32) if held_out is None else held_out
    test = np.unique(_matrix_hashes(np.asarray(held_out, dtype=np.float32), features))
    run.__dict__[ROW_HASHES_ATTR] = (train, test)
    return train, test


def _is_sample_file(run: TrainingRun, name: str) -> bool:
    """True when ``name`` (an uploaded file's name) is one of the files the run's sample was drawn from."""
    request = run.data_request
    if str(getattr(request, "source", "cicids")) != "cicids":
        return False
    wanted = Path(str(name)).name.strip().lower()
    return any(Path(str(f)).name.lower() == wanted for f in (getattr(request, "files", ()) or ()))


def _weak_reference(run: Any) -> Any:
    """A weak reference to ``run``, or None for an object that cannot be referenced weakly."""
    try:
        return weakref.ref(run)
    except TypeError:
        return None


class _Readings:
    """Scores parsed rows block by block and keeps only the readings (probabilities, verdicts, label codes)."""

    def __init__(self, run: TrainingRun, channel: str, voters: tuple[str, ...], step: int,
                 check: Callable[[], None]) -> None:
        self.run = run
        self.consensus = channel == CONSENSUS
        self.estimators = {key: run.channels[key].estimator for key in voters}
        self.voters = voters
        self.features = [str(f) for f in run.data.feature_names]
        self.classes = tuple(str(c) for c in run.data.classes)
        self.strategy = str(run.data_request.nonfinite_strategy)
        # Rows are matched with the run's own rows only when it holds them (a set loaded without its data folder
        # does not).
        held = getattr(run.data, "X_train", None)
        self.match_rows = held is not None and len(held) > 0
        self.step = step
        self.check = check
        self.reset()

    def reset(self) -> None:
        """Forget every reading taken so far."""
        self.proba: list[np.ndarray] = []
        self.predicted: list[np.ndarray] = []
        self.agreement: list[np.ndarray] = []
        self.codes: list[np.ndarray] = []
        self.shown: list[np.ndarray] = []
        self.empty: list[np.ndarray] = []
        self.read_as_attack: Counter[str] = Counter()
        self.hashes: list[np.ndarray] = []
        self.has_label = False
        self.rows = 0
        self.bad_rows = 0
        self.seconds = 0.0

    def add(self, frame: pd.DataFrame) -> None:
        """Score every row of a parsed block (reader output) and keep its readings."""
        n = len(frame)
        if n == 0:
            return
        n_classes = len(self.classes)
        columns = {f: frame[f].to_numpy() for f in self.features}
        rate_inputs = {c: frame[c].to_numpy() for c in _RATE_INPUTS if c in frame.columns and c not in columns}
        for start in range(0, n, self.step):
            self.check()
            stop = min(n, start + self.step)
            X, bad = _chunk_matrix(columns, self.features, start, stop, self.strategy, rate_inputs)
            self.bad_rows += int(bad.sum())
            if self.match_rows:
                self.hashes.append(_matrix_hashes(X, self.features))
            mark = time.perf_counter()
            if self.consensus:
                verdict = combine({key: _channel_proba(est, X, n_classes) for key, est in self.estimators.items()})
                self.proba.append(np.asarray(verdict.proba, dtype=np.float32))
                self.predicted.append(np.asarray(verdict.label_index, dtype=np.int64))
                self.agreement.append(np.asarray(verdict.agreement, dtype=np.int64))
            else:
                part = _channel_proba(self.estimators[self.voters[0]], X, n_classes)
                self.proba.append(np.asarray(part, dtype=np.float32))
                self.predicted.append(part.argmax(axis=1).astype(np.int64))
            self.seconds += time.perf_counter() - mark
        if LABEL in frame.columns:
            codes, shown, empty, as_attack = _map_labels(frame[LABEL], self.run)
            self.read_as_attack.update(as_attack)
            self.has_label = True
            self.codes.append(codes)
            self.shown.append(shown)
            self.empty.append(empty)
        self.rows += n


def _combined_report(reports: Sequence[FileReadReport], label: str) -> FileReadReport:
    """One account of reading a file block by block (repairs and counts summed over the blocks)."""
    first = reports[0]
    combined = FileReadReport(name=label, encoding=first.encoding, engine=first.engine,
                              fallback_note=next((r.fallback_note for r in reports if r.fallback_note), None))
    combined.rows_read = sum(r.rows_read for r in reports)
    combined.rows_kept = sum(r.rows_kept for r in reports)
    combined.duplicate_columns_dropped = list(first.duplicate_columns_dropped)
    combined.duplicate_columns_mismatched = list(dict.fromkeys(
        name for r in reports for name in r.duplicate_columns_mismatched))
    combined.duplicate_columns_equal = not combined.duplicate_columns_mismatched
    combined.missing_features = list(first.missing_features)
    combined.extra_columns = list(first.extra_columns)
    combined.has_label = first.has_label
    combined.non_numeric_cells = sum(r.non_numeric_cells for r in reports)
    combined.empty_labels = sum(r.empty_labels for r in reports)
    return combined


def _read_notes(reports: Sequence[FileReadReport], label: str) -> list[str]:
    """Plain-language notes on how the file was read (encodings, repairs worth a reader's attention)."""
    notes: list[str] = []
    encodings = list(dict.fromkeys(r.encoding for r in reports if r.encoding))
    if encodings and encodings != ["utf-8"]:
        if len(encodings) == 1:
            notes.append(f"Read as {encodings[0]} text.")
        else:
            notes.append(f"Read as {' and '.join(encodings)} text (parts of the file differ).")
    # The routine repeated column of the published files is left out, as on the sample sheet. The reader's wording
    # for unknown columns and empty labels describes training files (ignored, dropped); here those columns are kept
    # and those rows scored, so both are described elsewhere.
    notes.extend(line for line in _combined_report(reports, label).fixes(include_routine=False)
                 if "unknown column" not in line and "empty label" not in line)
    return notes


def _shown_names(names: Sequence[str]) -> str:
    """Column names for a note (an empty header cell shown as "(unnamed)"), at most six."""
    shown = [name if name else "(unnamed)" for name in names]
    return ", ".join(shown[:6]) + (", ..." if len(shown) > 6 else "")


# --------------------------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------------------------
def score_upload(
    run: TrainingRun,
    source: UploadSource,
    *,
    channel: str,
    alert_threshold: float,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    name: str | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelToken | None = None,
) -> ScoredBatch:
    """Score every row of an uploaded flow CSV with ``channel`` (a fitted channel key, or ``"consensus"``).

    ``source`` is the file's bytes, a binary file object (an upload) or a path. ``alert_threshold`` (0..1) is the
    attack probability from which an attack verdict raises an alert; ``chunk_rows`` rows are predicted at a time
    (and about that many are parsed per block). ``name`` names the file in messages (default: the path's or the
    object's name). ``progress`` receives (message, fraction done) and ``cancel`` is checked between chunks
    (raising :class:`~nids.models.jobs.TrainingCancelled`).

    Raises :class:`MissingColumnsError` when the file lacks a column the channels read (checked on the header
    before the file is read), :class:`~nids.data.reader.DataFileError` when it cannot be read as a flow CSV or
    holds no rows, and ``ValueError`` for a channel the run has not fitted or a threshold outside 0..1. Nothing is
    ever fitted.
    """
    started = time.perf_counter()
    voters = _voters(run, channel)
    threshold = float(alert_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The alert threshold must lie between 0 and 1.")
    step = int(chunk_rows)
    if step < 1:
        raise ValueError("Chunks must hold at least one row.")

    def tell(message: str, fraction: float) -> None:
        if progress is not None:
            progress(message, float(min(max(fraction, 0.0), 1.0)))

    def check() -> None:
        if cancel is not None and cancel.cancelled:
            raise TrainingCancelled("The scoring was cancelled.")

    payload, label = _payload(source, name)
    features = [str(f) for f in run.data.feature_names]
    tell("Checking the header", 0.01)
    _, header = read_flow_csv(payload, require_label=False, nrows=HEADER_CHECK_ROWS, name=label)
    missing = [f for f in features if f in header.missing_features]
    if missing:
        raise MissingColumnsError(missing, needed=len(features), source=label)
    check()

    # Read and score block by block; each block must give exactly one row per non-empty line.
    readings = _Readings(run, channel, voters, step, check)
    header_stop = _header_stop(payload)
    old_mac = _bare_carriage_returns(payload, header_stop)
    spans = [] if old_mac else _line_blocks(payload, header_stop, step)
    head_bytes = payload[:header_stop]
    blocks: list[LineBlock] = []
    reports: list[FileReadReport] = []
    as_written = not old_mac
    what = channel_name(channel)
    for start, stop in spans:
        check()
        tell(f"{what}: reading and scoring {label}", 0.03 + 0.92 * (start - header_stop) / max(
            len(payload) - header_stop, 1))
        piece = payload[start:stop]
        try:
            frame, report = read_flow_csv(head_bytes + piece, require_label=False, name=label)
        except DataFileError:
            as_written = False
            break
        if (report.engine != "pyarrow" or any(f not in frame.columns for f in features)
                or _count_lines(piece) != len(frame)):
            as_written = False
            break
        readings.add(frame)
        blocks.append(LineBlock(start, stop, len(frame), report.encoding))
        reports.append(report)
        del frame, piece

    extras_frame: pd.DataFrame | None = None
    read_frame: pd.DataFrame | None = None
    extra_names: list[str] = []
    notes: list[str] = []
    if as_written:
        if reports:
            extra_names = list(reports[0].extra_columns)
        if extra_names:
            notes.append(f"{len(extra_names)} column{'s' if len(extra_names) != 1 else ''} the channels do not read "
                         f"kept in the download as written ({_shown_names(extra_names)}).")
    else:
        # Lines and rows could not be matched: read the whole file at once and write its columns as read.
        readings.reset()
        tell(f"Reading {label} whole", 0.05)
        frame, report = read_flow_csv(payload, require_label=False, name=label)
        missing = [f for f in features if f not in frame.columns]
        if missing:  # the header check passed, so this only happens when the two readings disagree
            raise MissingColumnsError(missing, needed=len(features), source=label)
        check()
        reports = [report]
        if report.extra_columns:
            tell(f"Keeping {len(report.extra_columns)} other column(s)", 0.18)
            extras_frame, problem = _extra_columns(payload, report.encoding or "utf-8", report.extra_columns,
                                                   frame.index)
            if problem:
                notes.append(problem)
            elif extras_frame is not None:
                extra_names = list(extras_frame.columns)
                notes.append(f"{len(extra_names)} column{'s' if len(extra_names) != 1 else ''} the channels do not "
                             f"read kept in the output as written ({_shown_names(extra_names)}).")
        tell(f"{what}: scoring {label}", 0.2)
        readings.add(frame)
        parts = [frame.drop(columns=[LABEL]) if LABEL in frame.columns else frame]
        if extras_frame is not None and not extras_frame.empty:
            parts.append(extras_frame)
        if LABEL in frame.columns:
            parts.append(frame[[LABEL]])
        read_frame = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
        del frame
        why = ("end in a bare carriage return (the old Mac format)" if old_mac else
               "could not be matched one to one with its rows (a quoted line break, rows of uneven length or a "
               "line of spaces)")
        notes.append(f"The lines of {label} {why}, so the file was read whole. Its download writes the columns as "
                     "read: feature columns as 32-bit numbers, which round integers beyond 16,777,216, and labels "
                     "tidied.")
    n = readings.rows
    if n == 0:
        raise DataFileError(f"{label} holds no flow rows to score.")
    notes[:0] = _read_notes(reports, label)
    check()

    # Result columns.
    tell("Assembling the readings", 0.97)
    classes = readings.classes
    n_classes = len(classes)
    proba = np.concatenate(readings.proba) if len(readings.proba) > 1 else readings.proba[0]
    predicted = np.concatenate(readings.predicted) if len(readings.predicted) > 1 else readings.predicted[0]
    agreement = np.concatenate(readings.agreement) if readings.consensus else None
    normal = _normal_index(classes)
    # The rule every station shares: P(Attack) with two classes, else 1 - P(normal); compared in float32.
    attack_probability = attack_probability_of(proba, normal)
    alert = alert_flags(attack_probability, predicted, normal, threshold)

    index = pd.RangeIndex(n)
    results: dict[str, Any] = {
        PREDICTED: pd.Series(np.asarray(classes, dtype=object)[predicted], index=index, dtype="str"),
    }
    for j, class_name in enumerate(classes):
        results[probability_column(class_name)] = proba[:, j]
    results[ATTACK_PROBABILITY] = attack_probability
    results[ALERT] = alert
    if agreement is not None:
        results[AGREEMENT] = agreement

    # Labels and readings.
    labelled = False
    accuracy: float | None = None
    balanced: float | None = None
    confusion: pd.DataFrame | None = None
    unseen: dict[str, int] = {}
    rows_without_label = 0
    measured = 0
    if readings.has_label:
        codes = np.concatenate(readings.codes)
        shown = np.concatenate(readings.shown)
        empty = np.concatenate(readings.empty)
        labelled = bool((~empty).any())
        rows_without_label = int(empty.sum()) if labelled else 0
        if labelled:
            results[TRUE_LABEL] = pd.Series(shown, index=index, dtype="str")
        unknown = (codes < 0) & ~empty
        if unknown.any():
            counts = pd.Series(shown[unknown]).value_counts()
            unseen = {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))}
        known = codes >= 0
        measured = int(known.sum())
        if measured:
            scores = quick_metrics(codes[known], predicted[known], n_classes)
            accuracy = float(scores["accuracy"])
            balanced = float(scores["balanced_accuracy"])
            matrix = confusion_matrix(codes[known], predicted[known], labels=list(range(n_classes)))
            confusion = pd.DataFrame(matrix.astype(np.int64), index=pd.Index(classes, name="true"),
                                     columns=pd.Index(classes, name="predicted"))
        if rows_without_label:
            notes.append(f"{rows_without_label:,} row{'s' if rows_without_label != 1 else ''} with an empty label "
                         "scored but left out of the readings.")
        if unseen:
            total = sum(unseen.values())
            rows_word = "row carries a label" if total == 1 else "rows carry labels"
            notes.append(f"{total:,} {rows_word} this run never trained on, scored but left out of the readings: "
                         + ", ".join(f"{k} ({v:,})" for k, v in unseen.items()) + ".")
            if any(_NUMERIC_LABEL.match(k.strip()) for k in unseen):
                notes.append("Labels made only of digits (such as 0 and 1) name no class, so they cannot be "
                             "compared with a verdict. Label normal traffic BENIGN (or Normal) and attacks with "
                             "their names to have these rows measured.")
        if readings.read_as_attack and labelled:
            ranked = sorted(readings.read_as_attack.items(), key=lambda kv: (-kv[1], kv[0]))
            listed = ", ".join(f"{k} ({v:,})" for k, v in ranked[:ATTACK_LABELS_SHOWN])
            more = f" and {len(ranked) - ATTACK_LABELS_SHOWN:,} more" if len(ranked) > ATTACK_LABELS_SHOWN else ""
            notes.append("In binary mode every label other than BENIGN or Normal counts as Attack. Read as Attack "
                         f"here: {listed}{more}.")
    # Rows the run has seen: matched with its training and held-out rows over the channels' columns.
    seen_train: int | None = None
    seen_test: int | None = None
    unseen_accuracy: float | None = None
    unseen_balanced: float | None = None
    unseen_measured = 0
    from_sample = _is_sample_file(run, label)
    if readings.match_rows and readings.hashes:
        tell("Matching the rows with the run's own rows", 0.98)
        train_hashes, test_hashes = _run_row_hashes(run)
        hashes = np.concatenate(readings.hashes) if len(readings.hashes) > 1 else readings.hashes[0]
        in_train = np.isin(hashes, train_hashes)
        in_test = np.isin(hashes, test_hashes) & ~in_train
        seen_train, seen_test = int(in_train.sum()), int(in_test.sum())
        if readings.has_label and seen_train:
            fresh = (np.concatenate(readings.codes) >= 0) & ~in_train
            unseen_measured = int(fresh.sum())
            if unseen_measured:
                codes_all = np.concatenate(readings.codes)
                fresh_scores = quick_metrics(codes_all[fresh], predicted[fresh], n_classes)
                unseen_accuracy = float(fresh_scores["accuracy"])
                unseen_balanced = float(fresh_scores["balanced_accuracy"])
        if seen_train or seen_test:
            share = seen_train / n
            verb = "repeats" if seen_train == 1 else "repeat"
            text = (f"{seen_train:,} row{'s' if seen_train != 1 else ''} ({share:.1%}) {verb} a row this run was "
                    f"trained on and {seen_test:,} one of its held-out rows (compared over the channels' "
                    "columns).")
            if unseen_accuracy is not None and unseen_balanced is not None:
                text += (f" Over the {unseen_measured:,} labelled rows it never trained on: accuracy "
                         f"{score_text(unseen_accuracy)}, balanced accuracy {score_text(unseen_balanced)}.")
            elif seen_train and readings.has_label:
                text += " Every labelled row repeats a training row, so these readings are not held-out readings."
            notes.append(text)
    if from_sample:
        notes.append(f"{label} is one of the files this run's sample was drawn from, so it holds rows the channels "
                     "were trained on: its accuracy is not a reading on unseen traffic.")
    if readings.bad_rows:
        rows_word = "row holds" if readings.bad_rows == 1 else "rows hold"
        how = _STRATEGY_NOTES.get(readings.strategy, "They were treated as during the fit.")
        notes.append(f"{readings.bad_rows:,} {rows_word} infinite or missing values in the columns the channels "
                     f"read; scored anyway. {how}")

    out = pd.DataFrame(results, index=index)
    upload = UploadedRows(payload=payload, header_stop=header_stop, blocks=tuple(blocks), read_frame=read_frame)
    if as_written:
        upload.head = _head_rows(payload, header_stop, blocks[0].encoding if blocks else "utf-8",
                                 min(PREVIEW_ROWS, n))
    per_class = np.bincount(predicted, minlength=n_classes)
    tell("Done", 1.0)
    return ScoredBatch(
        frame=out, channel=channel, rows=n, rows_with_bad_values=readings.bad_rows,
        seconds=time.perf_counter() - started, labelled=labelled, accuracy=accuracy, balanced_accuracy=balanced,
        confusion=confusion, unseen_labels=unseen, run_id=str(run.run_id), classes=classes,
        mode=str(run.request.mode), source_name=label, alert_threshold=threshold, strategy=readings.strategy,
        score_seconds=readings.seconds, rows_without_label=rows_without_label, rows_measured=measured,
        voters=voters, extra_columns=extra_names, notes=notes,
        predicted_counts={name: int(per_class[j]) for j, name in enumerate(classes)},
        run_origin=str(getattr(run, "origin", "fitted")), upload=upload, run_ref=_weak_reference(run),
        rows_seen_in_training=seen_train, rows_seen_held_out=seen_test, unseen_accuracy=unseen_accuracy,
        unseen_balanced_accuracy=unseen_balanced, rows_unseen_measured=unseen_measured,
        from_sample_file=from_sample,
    )


__all__ = [
    "AGREEMENT", "ALERT", "ALERT_RULE", "ATTACK_PROBABILITY", "CONSENSUS", "CONSENSUS_NAME", "DEFAULT_CHUNK_ROWS",
    "LineBlock", "MissingColumnsError", "PREDICTED", "PREVIEW_ROWS", "PROB_PREFIX", "ScoredBatch", "TRUE_LABEL",
    "UploadedRows", "VERDICT", "channel_choices", "channel_name", "probability_column", "score_upload",
    "unique_names",
]
