"""CSV exports of a run's readings, and one ZIP holding all of them (07 Record).

Every builder returns the bytes of a CSV file encoded as UTF-8 with a byte-order mark (``"utf-8-sig"``), so
spreadsheet programs on Windows recognise the encoding, with ``.`` as the decimal point and an empty cell for a
missing number. None of them carries flow feature values: the predictions table holds row positions, provenance
(source file and row number, for CIC-IDS2017 data), labels and probabilities only, and the Assay export keeps the
scoring columns of the uploaded file, not its inputs.

:func:`export_items` lists what can be exported for a run (and why the rest cannot yet); each item builds its bytes
only when asked, so a page can offer every download without doing the work up front. :func:`bundle_zip` packs
every available item, plus a ``README.txt`` describing each file, into one ZIP.

The Assay batch and the simulation session are read by duck typing (attributes ``frame``/``channel`` and
``log_frame()``), so this module does not depend on the stations that produce them.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from graticule import APP_NAME, __version__, evaluate
from graticule.models import verdict
from graticule.schema import FEATURE_SET, LABEL

if TYPE_CHECKING:
    from graticule.data.prepare import PreparedDataset
    from graticule.evaluate import ChannelEvaluation
    from graticule.history import RunHistory
    from graticule.models.train import TrainingRun

#: Text encoding of every CSV export (UTF-8 with a byte-order mark).
CSV_ENCODING = "utf-8-sig"
#: Decimals kept for probabilities in the predictions and Assay exports.
PROBABILITY_DECIMALS = 6
#: Columns of the scoring output an Assay export keeps (besides every ``prob_<class>`` column).
ASSAY_OUTPUT_COLUMNS: tuple[str, ...] = (
    "predicted_label", "verdict", "attack_probability", "confidence", "alert", "channels_agreeing", "true_label",
)
#: Prefix of the per-class probability columns of an Assay batch.
ASSAY_PROBABILITY_PREFIX = "prob_"
#: Name of the README inside the ZIP.
README_NAME = "README.txt"
#: Key of the consensus row in the leaderboard export.
CONSENSUS_KEY = "consensus"


# --------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------
def csv_bytes(frame: pd.DataFrame) -> bytes:
    """``frame`` as CSV bytes (UTF-8 with a byte-order mark, no index, ``\\n`` line ends).

    True/false columns are written ``true``/``false``, as the scored file of 05 Assay writes them, so every CSV
    of the app spells a flag the same way (spreadsheets and pandas read both as flags).
    """
    flags = [c for c in frame.columns if pd.api.types.is_bool_dtype(frame[c])]
    if flags:
        frame = frame.copy()
        for column in flags:
            frame[column] = frame[column].map({True: "true", False: "false"})
    return frame.to_csv(index=False, lineterminator="\n").encode(CSV_ENCODING)


def _ordered_keys(keys: Sequence[str]) -> list[str]:
    """Channel keys in CH1..CH5 order (unknown keys after them)."""
    known = [k for k in evaluate.CHANNEL_ORDER if k in keys]
    return known + [k for k in keys if k not in evaluate.CHANNEL_ORDER]


def _evaluations(run: "TrainingRun",
                 evaluations: Mapping[str, "ChannelEvaluation"] | None) -> dict[str, "ChannelEvaluation"]:
    """The readings to export: those given, else the run's own (computed once and kept on the run)."""
    if evaluations is not None:
        return dict(evaluations)
    return evaluate.evaluate_run(run)


def _channel_scores(run: "TrainingRun", key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(probabilities n x K as float64, predicted codes) a channel stored for the held-out rows, or None."""
    result = run.channels.get(key)
    proba = getattr(result, "proba", None)
    if result is None or getattr(result, "status", "") != "ok" or proba is None:
        return None
    values = np.asarray(proba, dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(run.data.y_test), len(run.data.classes)):
        return None
    predicted = getattr(result, "y_pred", None)
    codes = values.argmax(axis=1) if predicted is None else np.asarray(predicted, dtype=np.int64).reshape(-1)
    return values, codes.astype(np.int64)


def consensus_metrics(run: "TrainingRun", keys: Sequence[str] | None = None) -> dict[str, Any] | None:
    """Readings of the combined verdict (equal-weight mean of the channels' probabilities) on the held-out rows.

    ``keys`` picks the channels (default: every fitted one with stored probabilities). Returns None with fewer
    than two such channels or no held-out rows. The dict holds the metrics of
    :func:`graticule.evaluate.classification_metrics` plus ``voters`` (channels combined) and ``unanimous`` (share
    of held-out rows on which every channel read the consensus class).
    """
    if not evaluate.has_test_rows(run):
        return None
    wanted = _ordered_keys(list(keys) if keys is not None else list(run.ok_channels()))
    probas = {}
    for key in wanted:
        scores = _channel_scores(run, key)
        if scores is not None:
            probas[key] = scores[0]
    if len(probas) < 2:
        return None
    combined = verdict.combine(probas)
    out: dict[str, Any] = evaluate.classification_metrics(run.data.y_test, combined.label_index, combined.proba,
                                                          len(run.data.classes))
    out["voters"] = combined.voters
    out["unanimous"] = float(np.mean(combined.agreement == combined.voters))
    out["channels"] = list(probas)
    return out


def belongs_to_run(obj: Any, run: "TrainingRun | None") -> bool:
    """True when an Assay batch or simulation session ``obj`` was made with ``run`` itself.

    A fit and its copy loaded from disk share one run id but are different channel sets (the copy never holds CH3
    and may lack its held-out rows), so the run OBJECT is compared whenever ``obj`` can tell: through its
    ``made_with(run)`` method (a :class:`graticule.scoring.ScoredBatch`) or its ``run`` attribute (a
    :class:`graticule.simulate.SimulationSession`). An object that only names its run (``run_id``) is compared by
    id; else its class names are compared with the run's when they can be read (an Assay batch's ``prob_<class>``
    columns, a session's ``classes``); an object that shows none of these is taken to belong.
    """
    if obj is None or run is None:
        return False
    made_with = getattr(obj, "made_with", None)
    if callable(made_with):
        return bool(made_with(run))
    streamed = getattr(obj, "run", None)
    if streamed is not None and not isinstance(streamed, (str, bytes)):
        return streamed is run
    run_id = getattr(obj, "run_id", None)
    if run_id:
        return str(run_id) == str(run.run_id)
    classes = _object_classes(obj)
    return classes is None or tuple(classes) == tuple(str(c) for c in run.data.classes)


def _object_classes(obj: Any) -> list[str] | None:
    """Class names an Assay batch or a session shows, or None when it shows none."""
    frame = getattr(obj, "frame", None)
    if isinstance(frame, pd.DataFrame):
        names = [str(c)[len(ASSAY_PROBABILITY_PREFIX):] for c in frame.columns
                 if str(c).startswith(ASSAY_PROBABILITY_PREFIX)]
        return names or None
    classes = getattr(obj, "classes", None)
    if isinstance(classes, (list, tuple)) and classes:
        return [str(c) for c in classes]
    return None


# --------------------------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------------------------
def leaderboard_frame(run: "TrainingRun", evaluations: Mapping[str, "ChannelEvaluation"] | None = None, *,
                      consensus: bool = True) -> pd.DataFrame:
    """The leaderboard of 03 Measure (best balanced accuracy first), plus a ``Consensus`` row at the end.

    The consensus row (key ``"consensus"``) carries the scores of the combined verdict of every evaluated channel;
    its timing and row columns stay empty. It is left out with fewer than two channels or ``consensus=False``.
    """
    evals = _evaluations(run, evaluations)
    board = evaluate.leaderboard(evals, run)
    if not consensus or board.empty:
        return board
    readings = consensus_metrics(run, list(evals))
    if readings is None:
        return board
    row: dict[str, Any] = {column: np.nan for column in board.columns}
    row["key"] = CONSENSUS_KEY
    row["Channel"] = f"Consensus ({readings['voters']} channels)"
    for metric, title in evaluate.score_columns(run.request.mode):
        row[title] = float(readings.get(metric, np.nan))
    row["Gap to best"] = float(board["Balanced accuracy"].max()) - float(row["Balanced accuracy"])
    extra = pd.DataFrame([row], columns=board.columns)
    for column in ("Rows used", "Training rows"):
        extra[column] = extra[column].astype("Int64")
        board[column] = board[column].astype("Int64")
    return pd.concat([board, extra], ignore_index=True)


def per_class_table(evaluations: Mapping[str, "ChannelEvaluation"]) -> pd.DataFrame:
    """Every channel's per-class readings, stacked (channel order, then class order)."""
    return evaluate.per_class_frame(evaluations)


def predictions_frame(run: "TrainingRun", *, prepared: "PreparedDataset | None" = None,
                      consensus: bool = True) -> pd.DataFrame:
    """One row per held-out flow: where it came from, its true class and every channel's verdict and probabilities.

    Columns: ``test_index`` (position in the held-out rows), ``sample_row`` (position in the 01 Sample draw), then,
    for CIC-IDS2017 runs whose sample is passed as ``prepared`` (it must be the very sample the run was fitted on),
    ``source_file`` and ``source_row`` (0-based data-row number in that file, the header not counted); then
    ``true_class`` (the class the channels were scored against) and ``true_label`` (the detailed label), and per
    fitted channel ``<key>_predicted`` and one ``<key>_prob_<class>`` column per class. With two or more channels,
    ``consensus_predicted``, ``consensus_agreement`` (channels reading the consensus class) and
    ``consensus_prob_<class>`` follow. No feature value is included.
    """
    data = run.data
    classes = np.asarray([str(c) for c in data.classes], dtype=object)
    n = len(data.y_test) if evaluate.has_test_rows(run) else 0
    columns: dict[str, Any] = {
        "test_index": np.arange(n, dtype=np.int64),
        "sample_row": np.asarray(data.test_rows[:n], dtype=np.int64),
    }
    if (n and prepared is not None and run.data_request.source == "cicids"
            and getattr(prepared, "fingerprint", None) == run.dataset_fingerprint):
        from graticule.data.prepare import FILE_COL, ROW_COL

        rows = np.asarray(data.test_rows, dtype=np.int64)
        columns["source_file"] = prepared.frame[FILE_COL].astype("str").to_numpy()[rows]
        columns["source_row"] = prepared.frame[ROW_COL].to_numpy()[rows].astype(np.int64)
    codes = np.asarray(data.y_test[:n], dtype=np.int64)
    columns["true_class"] = classes[codes] if n else np.asarray([], dtype=object)
    columns["true_label"] = np.asarray(data.detailed_test_labels[:n], dtype=object).astype(str)
    probas: dict[str, np.ndarray] = {}
    for key in (run.ok_channels() if n else []):
        scores = _channel_scores(run, key)
        if scores is None:
            continue
        proba, predicted = scores
        probas[key] = proba
        columns[f"{key}_predicted"] = classes[predicted]
        for j, name in enumerate(classes):
            columns[f"{key}_prob_{name}"] = np.round(proba[:, j], PROBABILITY_DECIMALS)
    if consensus and len(probas) >= 2:
        combined = verdict.combine(probas)
        columns["consensus_predicted"] = classes[combined.label_index]
        columns["consensus_agreement"] = combined.agreement
        for j, name in enumerate(classes):
            columns[f"consensus_prob_{name}"] = np.round(combined.proba[:, j].astype(np.float64),
                                                         PROBABILITY_DECIMALS)
    frame = pd.DataFrame(columns)
    for column in ("true_class", "true_label", "source_file", *[c for c in frame.columns if c.endswith("_predicted")]):
        if column in frame.columns:
            frame[column] = frame[column].astype("str")
    return frame


def cross_validation_table(cv: pd.DataFrame) -> pd.DataFrame:
    """The cross-validation summary of 03 Measure (one row per channel), as kept with the run."""
    return cv.reset_index(drop=True)


def cross_validation_folds(cv: pd.DataFrame) -> pd.DataFrame:
    """Each fold's readings of a cross-validation, with the channel's badge and name."""
    folds = evaluate.cv_fold_frame(cv)
    folds.insert(1, "Channel", [evaluate.channel_label(str(k)) for k in folds["key"]])
    return folds


def history_frame(history: "RunHistory | None" = None, limit: int | None = None) -> pd.DataFrame:
    """Every line of the run history, newest first (see :class:`graticule.history.RunHistory`)."""
    from graticule.history import RunHistory

    return (history or RunHistory()).list(limit)


def _assay_columns(batch: Any, frame: pd.DataFrame) -> list[Any]:
    """The scoring columns of an Assay batch: its own ``result_columns`` when it lists them, else the known output
    columns plus the ``prob_<class>`` columns of its classes (never an uploaded column that merely looks alike)."""
    listed = getattr(batch, "result_columns", None)
    if isinstance(listed, (list, tuple)) and listed:
        return [c for c in listed if c in frame.columns]
    classes = [str(c) for c in (getattr(batch, "classes", None) or ())]
    wanted = {f"{ASSAY_PROBABILITY_PREFIX}{name}" for name in classes}
    keep = [c for c in frame.columns if str(c) in ASSAY_OUTPUT_COLUMNS
            or (str(c).startswith(ASSAY_PROBABILITY_PREFIX) and (not wanted or str(c) in wanted))]
    if not keep:
        keep = [c for c in frame.columns if str(c) not in FEATURE_SET and str(c) != LABEL]
    return keep


def assay_frame(batch: Any) -> pd.DataFrame:
    """The readings of an Assay batch: ``row`` (0-based data row of the uploaded file, the header not counted, as
    in every export and at 05 Assay) and the scoring columns.

    The uploaded columns themselves (flow features, addresses, anything else the file held, including columns of an
    earlier scoring) are left out; the full scored file stays available at 05 Assay. Raises ``ValueError`` for an
    object without a ``frame`` table.
    """
    frame = getattr(batch, "frame", None)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("This Assay result holds no scored table.")
    out = frame[_assay_columns(batch, frame)].reset_index(drop=True)
    for column in out.columns:
        if str(column).startswith(ASSAY_PROBABILITY_PREFIX) or str(column) in ("attack_probability", "confidence"):
            if pd.api.types.is_float_dtype(out[column]):
                out[column] = out[column].astype(np.float64).round(PROBABILITY_DECIMALS)
    index = frame.index.to_numpy()
    rows = index.astype(np.int64) if pd.api.types.is_integer_dtype(index) else np.arange(len(out))
    out.insert(0, "row", np.asarray(rows, dtype=np.int64))
    return out


def sweep_frame(session: Any) -> pd.DataFrame:
    """The flow log of a 06 Sweep simulation session (its ``log_frame()``: one row per emitted flow).

    Raises ``ValueError`` for an object without a ``log_frame`` method.
    """
    log = getattr(session, "log_frame", None)
    if not callable(log):
        raise ValueError("This simulation session keeps no flow log.")
    frame = log()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    bad = [c for c in frame.columns if str(c) in FEATURE_SET]
    return frame.drop(columns=bad).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------
# CSV builders
# --------------------------------------------------------------------------------------------------------------
def leaderboard_csv(run: "TrainingRun", evaluations: Mapping[str, "ChannelEvaluation"] | None = None) -> bytes:
    """The leaderboard (with the consensus row) as CSV bytes."""
    return csv_bytes(leaderboard_frame(run, evaluations))


def per_class_csv(run: "TrainingRun", evaluations: Mapping[str, "ChannelEvaluation"] | None = None) -> bytes:
    """Every channel's per-class readings as CSV bytes."""
    return csv_bytes(per_class_table(_evaluations(run, evaluations)))


def predictions_csv(run: "TrainingRun", *, prepared: "PreparedDataset | None" = None) -> bytes:
    """The held-out predictions (see :func:`predictions_frame`) as CSV bytes."""
    return csv_bytes(predictions_frame(run, prepared=prepared))


def cross_validation_csv(cv: pd.DataFrame) -> bytes:
    """The cross-validation summary as CSV bytes."""
    return csv_bytes(cross_validation_table(cv))


def cross_validation_folds_csv(cv: pd.DataFrame) -> bytes:
    """The per-fold cross-validation readings as CSV bytes."""
    return csv_bytes(cross_validation_folds(cv))


def history_csv(history: "RunHistory | None" = None) -> bytes:
    """The whole run history as CSV bytes."""
    return csv_bytes(history_frame(history, None))


def assay_csv(batch: Any) -> bytes:
    """The readings of an Assay batch (see :func:`assay_frame`) as CSV bytes."""
    return csv_bytes(assay_frame(batch))


def sweep_csv(session: Any) -> bytes:
    """The flow log of a simulation session as CSV bytes."""
    return csv_bytes(sweep_frame(session))


# --------------------------------------------------------------------------------------------------------------
# What can be exported
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ExportItem:
    """One CSV a page can offer: what it is, its file name, and how to build it (or why it is not available).

    ``build`` produces the bytes when called (it may take a moment for large runs); it is None when the table is
    not available, and ``missing`` then says what would make it available.
    """

    key: str
    title: str
    file_name: str
    description: str
    build: Callable[[], bytes] | None
    missing: str = ""

    @property
    def available(self) -> bool:
        """True when the table can be built now."""
        return self.build is not None


#: Order, titles and descriptions of the exports (key -> (title, description)).
EXPORTS: dict[str, tuple[str, str]] = {
    "leaderboard": ("Leaderboard",
                    "One row per channel, best balanced accuracy first: every score on the held-out rows, the gap to "
                    "the best channel, fit time, scoring speed, single-flow latency and training rows used; the "
                    "last row is the consensus of all channels (equal-weight mean of their probabilities)."),
    "per_class": ("Per-class readings",
                  "One row per channel and class: held-out rows (support), precision, recall, F1, one-vs-rest "
                  "ROC-AUC and average precision."),
    "predictions": ("Held-out predictions",
                    "One row per held-out flow: its positions (and, for CIC-IDS2017 data, source file and 0-based "
                    "data-row number), true class and detailed label, each channel's verdict and class "
                    "probabilities, and the consensus. No feature values."),
    "cross_validation": ("Cross-validation",
                         "One row per channel: folds, rows, mean and standard deviation of accuracy, balanced "
                         "accuracy and macro F1 over the folds, mean fit time."),
    "cross_validation_folds": ("Cross-validation folds",
                               "One row per channel and fold: accuracy, balanced accuracy, macro F1 and fit time."),
    "run_history": ("Run history",
                    "One row per finished fit recorded on this machine: when, source, files, mode, feature set, "
                    "rows, channels, best channel, headline metrics and settings (JSON), duration and saved folder."),
    "assay": ("Assay readings",
              "One row per row of the file scored at 05 Assay: its 0-based data-row number (the header not "
              "counted), verdict, class probabilities, attack probability, alert flag (an attack verdict at or "
              "above the alert threshold) and the true label when the file had one. The uploaded columns are not "
              "repeated."),
    "sweep": ("Sweep log",
              "One row per flow streamed at 06 Sweep: tick, row, true label, verdict, attack probability, "
              "confidence, alert flag and whether the verdict was right."),
}


def export_file_name(key: str, run_id: str | None) -> str:
    """File name of an export, e.g. ``graticule-leaderboard-20261001-153012-ab12.csv``."""
    stem = key.replace("_", "-")
    return f"graticule-{stem}-{run_id}.csv" if run_id else f"graticule-{stem}.csv"


def _item(key: str, run_id: str | None, build: Callable[[], bytes] | None, missing: str = "") -> ExportItem:
    """An :class:`ExportItem` with the standard title, description and file name."""
    title, description = EXPORTS[key]
    return ExportItem(key=key, title=title, file_name=export_file_name(key, run_id), description=description,
                      build=build, missing="" if build is not None else missing)


def _history_lines(history: "RunHistory | None") -> int | None:
    """Lines in the run history, or None when the file cannot be read."""
    from graticule.history import RunHistory

    try:
        return (history or RunHistory()).count()
    except Exception:  # noqa: BLE001 - an unreadable history only disables its export
        return None


def export_items(
    run: "TrainingRun | None",
    *,
    evaluations: Mapping[str, "ChannelEvaluation"] | None = None,
    prepared: "PreparedDataset | None" = None,
    history: "RunHistory | None" = None,
    assay: Any = None,
    sweep: Any = None,
    measure_if_needed: bool = True,
) -> list[ExportItem]:
    """Every export of :data:`EXPORTS` for ``run`` (None: no run yet), each available or saying why not.

    Nothing is built here: each available item builds its bytes when its ``build`` is called. The readings come
    from ``evaluations`` when given, else from the run (computed once and kept on it). With
    ``measure_if_needed=False`` and no readings yet, the leaderboard and per-class exports are offered as
    unavailable instead (taking the readings scores every channel and times it, which a page runs in the app's
    work slot, at 03 Measure or while building the PDF record). ``prepared`` adds source files and row numbers to
    the predictions when it is the run's own CIC-IDS2017 sample. ``assay`` (the last Assay batch) and ``sweep``
    (the simulation session) are offered only when they belong to ``run`` (:func:`belongs_to_run`).
    """
    run_id = None if run is None else str(run.run_id)
    items: list[ExportItem] = []
    fitted = run is not None and bool(run.ok_channels())
    rows = fitted and evaluate.has_test_rows(run)  # type: ignore[arg-type]
    need_fit = "Fit channels at 02 Fit to add their readings."
    hollow = ("This run was loaded without its held-out rows, so it has no readings to export; load it again at "
              "the Logbook with the data folder available.")
    unmeasured = ("The readings have not been taken yet: open 03 Measure, or build the PDF record above (either "
                  "takes them once, in the app's work slot), then download this.")
    no_rows = need_fit if not fitted else hollow
    if rows:
        assert run is not None
        evals = evaluations
        if evals is None and not measure_if_needed:
            items.append(_item("leaderboard", run_id, None, unmeasured))
            items.append(_item("per_class", run_id, None, unmeasured))
        else:
            items.append(_item("leaderboard", run_id, lambda: leaderboard_csv(run, evals)))
            items.append(_item("per_class", run_id, lambda: per_class_csv(run, evals)))
        items.append(_item("predictions", run_id, lambda: predictions_csv(run, prepared=prepared)))
    else:
        items += [_item(key, run_id, None, no_rows) for key in ("leaderboard", "per_class", "predictions")]
    cv = evaluate.stored_cross_validation(run) if run is not None else None
    if cv is not None and not cv.empty:
        items.append(_item("cross_validation", run_id, lambda: cross_validation_csv(cv)))
        items.append(_item("cross_validation_folds", run_id, lambda: cross_validation_folds_csv(cv)))
    else:
        why = "Run cross-validation at 03 Measure to add its readings." if fitted else need_fit
        items += [_item("cross_validation", run_id, None, why), _item("cross_validation_folds", run_id, None, why)]
    lines = _history_lines(history)
    if lines:
        items.append(_item("run_history", None, lambda: history_csv(history)))
    else:
        items.append(_item("run_history", None, None, "No fit has been recorded yet." if lines == 0 else
                           "The run history file could not be read."))
    items.append(_optional_item("assay", run, run_id, assay, assay_csv,
                                "Score a file at 05 Assay to add its readings.",
                                "The last file scored at 05 Assay was scored with another run (or with another "
                                "copy of this run, fitted or loaded from disk); score it again with this one to "
                                "add its readings."))
    items.append(_optional_item("sweep", run, run_id, sweep, sweep_csv,
                                "Stream flows at 06 Sweep to add their log.",
                                "The sweep in progress at 06 Sweep uses another run; start one with this run to add "
                                "its log."))
    return items


def _optional_item(key: str, run: "TrainingRun | None", run_id: str | None, obj: Any,
                   builder: Callable[[Any], bytes], absent: str, foreign: str) -> ExportItem:
    """The Assay or Sweep export: available when ``obj`` exists, has content and belongs to ``run``."""
    if obj is None or run is None:
        return _item(key, run_id, None, absent)
    if not belongs_to_run(obj, run):
        return _item(key, run_id, None, foreign)
    if key == "sweep" and not _has_flows(obj):
        return _item(key, run_id, None, absent)
    return _item(key, run_id, lambda: builder(obj))


def _has_flows(session: Any) -> bool:
    """True when a simulation session has emitted at least one flow (or does not say)."""
    stats = getattr(session, "stats", None)
    emitted = getattr(stats, "emitted", None)
    if isinstance(emitted, (int, np.integer)):
        return int(emitted) > 0
    return callable(getattr(session, "log_frame", None))


# --------------------------------------------------------------------------------------------------------------
# ZIP
# --------------------------------------------------------------------------------------------------------------
def readme_text(items: Sequence[ExportItem], run: "TrainingRun | None", *,
                extra_files: Mapping[str, str] | None = None, built_utc: datetime | None = None) -> str:
    """The ``README.txt`` of a ZIP: what the files are, how they are encoded, and what is not included.

    ``items`` are the exports packed (the available ones); ``extra_files`` maps other packed file names (the PDF
    record) to a one-line description.
    """
    when = (built_utc or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"{APP_NAME} {__version__} - measurement record exports", "=" * 48, ""]
    if run is not None:
        channels = ", ".join(evaluate.channel_label(k) for k in run.ok_channels())
        lines += [f"Run:     {run.run_id}", f"Fitted:  {run.created_utc}",
                  f"Data:    {run.data_request.describe()}",
                  f"Mode:    {run.request.mode}; channels: {channels}"]
    lines += [f"Packed:  {when}", "", "Files", "-----"]
    for name, description in (extra_files or {}).items():
        lines += [name, f"    {description}", ""]
    for item in items:
        lines += [item.file_name, f"    {item.title}. {item.description}", ""]
    lines += [
        "Notes", "-----",
        "- CSV files are UTF-8 with a byte-order mark, comma separated, with '.' as the decimal point; an empty "
        "cell is a missing number.",
        "- Readings are measured on the held-out (test) rows of the run, which no fitting step saw.",
        "- No file here holds flow feature values. Source file names and row numbers identify CIC-IDS2017 rows "
        "without copying them. Every row number (source_row, the Assay readings' row) counts data rows from 0, "
        "the header not included.",
        "- Flags (alert, correct) are written true or false.",
        "- Probabilities are rounded to 6 decimals.",
        "",
    ]
    return "\n".join(lines)


def bundle_zip(
    run: "TrainingRun | None" = None,
    *,
    evaluations: Mapping[str, "ChannelEvaluation"] | None = None,
    prepared: "PreparedDataset | None" = None,
    history: "RunHistory | None" = None,
    assay: Any = None,
    sweep: Any = None,
    items: Sequence[ExportItem] | None = None,
    extra_files: Mapping[str, tuple[bytes, str]] | None = None,
) -> bytes:
    """A ZIP of every available export plus ``README.txt``; returns its bytes.

    ``items`` may pass the exports already listed by :func:`export_items` (else they are listed here from the
    other arguments). ``extra_files`` maps a file name to (bytes, one-line description) for other files to pack,
    such as the PDF record. An export whose build fails is left out and named in the README instead.
    """
    chosen = list(items) if items is not None else export_items(
        run, evaluations=evaluations, prepared=prepared, history=history, assay=assay, sweep=sweep)
    stamp = datetime.now(timezone.utc)
    packed: list[ExportItem] = []
    failed: list[str] = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def write(name: str, data: bytes) -> None:
            info = zipfile.ZipInfo(name, date_time=stamp.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)

        for name, (data, _description) in (extra_files or {}).items():
            write(name, data)
        for item in chosen:
            if item.build is None:
                continue
            try:
                data = item.build()
            except Exception as exc:  # noqa: BLE001 - one failing table must not lose the others
                failed.append(f"{item.file_name} ({type(exc).__name__}: {exc})")
                continue
            write(item.file_name, data)
            packed.append(item)
        readme = readme_text(packed, run, extra_files={k: v[1] for k, v in (extra_files or {}).items()},
                             built_utc=stamp)
        if failed:
            readme += "\nNot packed (building failed):\n" + "\n".join(f"- {line}" for line in failed) + "\n"
        write(README_NAME, readme.encode("utf-8"))
    return buffer.getvalue()


def zip_names(data: bytes) -> list[str]:
    """The file names inside a ZIP's bytes (for tests and checks)."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return archive.namelist()


__all__ = [
    "ASSAY_OUTPUT_COLUMNS", "CONSENSUS_KEY", "CSV_ENCODING", "EXPORTS", "ExportItem", "assay_csv", "assay_frame",
    "belongs_to_run", "bundle_zip", "consensus_metrics", "cross_validation_csv", "cross_validation_folds",
    "cross_validation_folds_csv", "cross_validation_table", "csv_bytes", "export_file_name", "export_items",
    "history_csv", "history_frame", "leaderboard_csv", "leaderboard_frame", "per_class_csv", "per_class_table",
    "predictions_csv", "predictions_frame", "readme_text", "sweep_csv", "sweep_frame", "zip_names",
]
