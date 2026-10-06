"""Saved channel sets ("bundles"): write a fitted run to disk, read it back, and prove it still reads the same.

A bundle is one folder, ``saved_models/<run id>/``:

``manifest.json``
    Plain JSON describing the run: app and library versions, when it was fitted and saved, mode, classes, feature
    names and how they were chosen, the data and fit requests (source, files, settings, seed), the dataset, train
    and test fingerprints and digests of their feature values, row counts (per class too), the bad-value strategy,
    every channel's status, notes, timings and metrics, the SHA-256 of every other file, a summary of the probe
    set, and (``manifest_sha256``) the SHA-256 of the manifest's own content.
``<key>.joblib``
    The fitted scikit-learn pipeline of channel ``<key>`` (forest, mlp, logreg, and svm when CH3 is saved by
    choice), as joblib writes it.
``xgboost.ubj`` and ``xgboost.joblib``
    The XGBoost model in XGBoost's own binary format, and the rest of its pipeline (the sanitiser) together with
    the model's constructor parameters, which XGBoost's format does not keep. They are put together again on load.
``probe.npz``
    512 probe vectors and, for every channel kept, the probabilities and labels it gave them when the bundle was
    saved.
    The vectors are synthetic: each value is read off that feature's training quantiles at a random level, chosen
    independently per feature (and 32 vectors carry one missing value each, so the gap-filling steps are checked
    too). They are not dataset rows.
``quantiles.npz``
    The training quantiles of every feature (0 %, 1 %, ..., 100 %): summary statistics only.

Dataset rows stay out of a bundle by default. The run's reference sample (training rows kept for explanations) is
never written; a loaded run draws its background from the quantiles instead, or from the rebuilt training rows.
For the same reason CH3 is left out by default (:data:`UNSAVED_CHANNELS`): a kernel SVM is a weighted
set of its training rows (its support vectors), stored after gap filling, signed logarithm and standardisation,
which the scaler saved with it undoes exactly. The manifest then keeps CH3's readings, timings and support-vector
count under status ``"not_saved"``, and a loaded run lists the channel as not saved; fit it again at 02 Fit when it
is needed.

Saving CH3 is an explicit opt-in (``save_run(..., include_svm=True)``, the "Also save CH3" box at the Logbook): then
``svm.joblib`` is written, checksummed, probe-checked and loaded like every other channel, and the manifest declares
the training rows it holds under :data:`TRAINING_ROWS_KEY` (``{"svm": <support vectors>}``) and says so in its
``contents`` text. A manifest without that key (every bundle saved before the opt-in existed), or with it empty
(every bundle saved without CH3), declares no training rows. The bundle format is the same either way.

Loading (:func:`load_bundle`) first checks the manifest against its own recorded SHA-256, then every other file
against the SHA-256 the manifest records for it, and refuses the bundle when anything is missing or differs
(:class:`BundleIntegrityError`); the fitted channels must also agree with the manifest's classes and feature count.
Then every channel scores the probe vectors again. The bundle is "verified" only when the library versions are
those it was saved with and every channel reproduces its saved labels and probabilities exactly
(``np.array_equal``). With other library versions the largest difference is reported and the bundle is "not
verified"; with the same versions any difference is a failure.

Trust: ``.joblib`` files are pickles, and reading a pickle runs code. The checksums catch damaged files and casual
edits, not a deliberate forgery (whoever can edit a model file can recompute every checksum too), so only load
bundles made on this machine or by someone you trust.

A save works in a hidden ``.saving-*`` folder that is renamed into place at the end, and a delete first renames
the bundle to a hidden ``.deleting-*`` folder and then removes it (any model made of training rows first, the
manifest last). When a file stays held by another program past the retries, such a folder can stay behind: a
failed delete puts the bundle back under its own name when it can, and :func:`find_leftovers` lists whatever is
left (saying when ``svm.joblib``, CH3's training rows, is among it) so the Logbook can show it and
:func:`remove_leftover` can remove it.

Held-out rows are never stored either. :func:`rebuild_training_data` re-runs the recorded 01 Sample and 02 Fit
requests on the data folder (or the generator, for synthetic runs) and checks that the very same rows, with the very
same feature values, come out (a Top-K run reuses its recorded ranking rather than ranking again);
:func:`restore_run` then turns a bundle, with or without those rows, into an ordinary
:class:`~nids.models.train.TrainingRun` by prediction only (nothing is fitted again).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from nids import APP_NAME, __version__
from nids import settings as settings_mod
from nids.data import sampling
from nids.data.prepare import DataRequest, FileReader, FileStager, PreparedDataset, prepare_dataset
from nids.data.reader import DataFileError
from nids.data.sampling import SingleClassError
from nids.evaluate import quick_metrics
from nids.features import FeatureChoice
from nids.models.train import (
    QUANTILE_LEVELS,
    REFERENCE_ROWS,
    ChannelResult,
    TrainingData,
    TrainingRun,
    TrainRequest,
    _tidy_proba,
    build_training_data,
    score_in_blocks,
)
from nids.models.zoo import MODEL_KEYS
from nids.settings import replace_with_retry
from nids.theme import CHANNEL_BY_KEY

#: Version of the bundle layout written by :func:`save_run`; :func:`load_bundle` reads only this version. Format 2
#: added the manifest's own checksum and the digests of the feature values, and leaves CH3 out unless it is saved
#: by choice (the optional :data:`TRAINING_ROWS_KEY` came later within the same format).
BUNDLE_FORMAT = 2
MANIFEST_FILE = "manifest.json"
#: Manifest key holding the SHA-256 of the rest of the manifest (see :func:`manifest_digest`).
MANIFEST_DIGEST_KEY = "manifest_sha256"
#: Optional manifest key: channel key -> training rows its saved model holds (``{"svm": n}`` when CH3 was saved by
#: choice, empty otherwise). Absent in older manifests, which means none.
TRAINING_ROWS_KEY = "training_rows_inside"
#: Status of a fitted channel that was not written to its bundle.
NOT_SAVED = "not_saved"
#: Channels left out of a bundle unless saved by choice, and why (their model is made of dataset rows).
UNSAVED_CHANNELS: dict[str, str] = {
    "svm": ("CH3 RBF SVM is not kept in saved sets unless you choose to save it: a kernel SVM is made of its "
            "training rows (its support vectors, standardised), and by default saved sets hold no dataset rows. "
            "Fit it again at 02 Fit when you need it."),
}
#: Earlier wordings of :data:`UNSAVED_CHANNELS` found in the notes of bundles already on disk (replaced on restore).
_FORMER_UNSAVED_NOTES: dict[str, tuple[str, ...]] = {
    "svm": ("CH3 RBF SVM is not kept in saved sets: a kernel SVM is made of its training rows (its support "
            "vectors, standardised), and saved sets never hold dataset rows. Fit it again at 02 Fit when you need "
            "it.",),
}
#: Model files made of training rows (CH3 saved by choice): a delete removes them first.
_ROW_FILES = frozenset(f"{key}.joblib" for key in UNSAVED_CHANNELS)
#: Name prefixes of the hidden work folders of a save and of a delete (gone when either finishes).
SAVING_PREFIX = ".saving-"
DELETING_PREFIX = ".deleting-"
PROBE_FILE = "probe.npz"
QUANTILE_FILE = "quantiles.npz"
XGB_BOOSTER_FILE = "xgboost.ubj"
XGB_SHELL_FILE = "xgboost.joblib"
#: Probe vectors per bundle, and how many of them carry one missing value.
PROBE_COUNT = 512
PROBE_MISSING_ROWS = 32
#: Added to the run's seed to draw the probe vectors (so they differ from any other seeded draw of the run).
PROBE_SEED_OFFSET = 7_919
#: Background vectors drawn from the quantiles for a run loaded without its training rows.
BACKGROUND_ROWS = 256
#: joblib compression level for model files (zlib level 3: much smaller forests at little cost).
JOBLIB_COMPRESS = 3
#: Libraries whose versions are recorded and compared, besides Python itself.
LIBRARIES: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
#: Forest types whose thread count :func:deterministic pins to one.
_FORESTS = (RandomForestClassifier, ExtraTreesClassifier)
#: Serialises :func:deterministic blocks, so two threads never interleave their changes to one forest's n_jobs.
_DETERMINISTIC_LOCK = threading.RLock()


class BundleIntegrityError(RuntimeError):
    """A bundle is incomplete or one of its files differs from the checksum recorded when it was saved."""


class BundleReadError(RuntimeError):
    """A bundle's files are intact but could not be read (for example with a much newer or older library, or a set
    whose model files were written by an earlier version of the app and name code this version does not have)."""


class BundleDeleteError(OSError):
    """A delete stopped halfway because a file stayed held; the message says where what is left now lies."""


class RebuildError(RuntimeError):
    """The held-out rows of a saved run could not be rebuilt identically (data folder, files or fingerprints)."""


@dataclass(frozen=True)
class BundleSummary:
    """One saved bundle as listed in the Logbook (read from its manifest only; no model is loaded).

    ``problem`` says why :func:`load_bundle` would refuse the bundle when that shows from the manifest alone (an
    older layout, or a manifest changed since it was saved); None otherwise. ``holds_training_rows`` is the number of
    training rows the bundle's models hold (CH3's support vectors when it was saved by choice; 0 when none).
    """

    path: Path
    run_id: str
    created_utc: str
    mode: str
    source: str
    rows: int
    channels: tuple[str, ...]
    best_channel: str | None
    best_balanced_accuracy: float | None
    problem: str | None = None
    holds_training_rows: int = 0


@dataclass(frozen=True)
class LeftoverFolder:
    """A hidden work folder that a save or a delete left in the models folder (see :func:`find_leftovers`).

    ``kind`` is ``"save"`` (a ``.saving-*`` folder) or ``"delete"`` (a ``.deleting-*`` folder); ``files`` are the
    names still in it; ``holds_training_rows`` is True when one of them is a model made of training rows
    (``svm.joblib``, CH3 saved by choice); ``age_seconds`` is the time since the folder or a file in it last changed.
    """

    path: Path
    kind: str
    files: tuple[str, ...]
    holds_training_rows: bool
    age_seconds: float


@dataclass
class VerificationReport:
    """What :func:`load_bundle` found when it checked a bundle.

    ``hashes_ok`` is True once every file matched its checksum (a mismatch raises instead). ``probes_identical``
    and ``max_abs_diff`` hold, per channel, whether the probe labels and probabilities were reproduced exactly and
    the largest absolute probability difference. ``version_mismatches`` maps a library (or ``"nids"``) to its
    (saved, installed) versions. ``verified`` is True only with no version mismatch and every channel identical.
    """

    hashes_ok: bool
    probes_identical: dict[str, bool]
    max_abs_diff: dict[str, float]
    version_mismatches: dict[str, tuple[str, str]]
    verified: bool
    message: str


@dataclass
class LoadedBundle:
    """A bundle read from disk: its manifest, the fitted channels and the verification result.

    ``channels`` maps a channel key to its fitted estimator (only channels that were fitted when the run was
    saved). ``quantiles`` are the per-feature training quantiles (101 x n_features, float32).
    """

    path: Path
    manifest: dict[str, Any]
    channels: dict[str, Any]
    classes: tuple[str, ...]
    feature_names: tuple[str, ...]
    mode: str
    verification: VerificationReport
    quantiles: np.ndarray

    @property
    def run_id(self) -> str:
        """Id of the saved run."""
        return str(self.manifest.get("run_id", self.path.name))

    @property
    def source(self) -> str:
        """Data source of the saved run: ``"cicids"`` or ``"synthetic"``."""
        return str((self.manifest.get("data_request") or {}).get("source", "cicids"))

    @property
    def training_rows_inside(self) -> dict[str, int]:
        """Training rows held per saved channel (``{"svm": n}`` when CH3 was saved by choice; empty otherwise)."""
        return training_rows_inside(self.manifest)


# --------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------
def library_versions() -> dict[str, str]:
    """Installed versions of Python and of the libraries a bundle depends on (``"missing"`` when not installed)."""
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes (read in 1 MB blocks)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    """``value`` as JSON-ready plain values: numpy scalars and arrays become Python ones, non-finite floats None."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def training_rows_inside(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Training rows the models of a bundle hold, per channel, as its manifest declares them.

    Reads :data:`TRAINING_ROWS_KEY` (``{"svm": <support vectors>}`` when CH3 was saved by choice). The key is
    optional: a manifest without it, or with an empty or malformed value, declares none, except that a channel of
    :data:`UNSAVED_CHANNELS` saved with status ``"ok"`` counts its recorded support vectors all the same, so a
    bundle holding CH3 is never listed as holding no rows. Channels holding none are left out.
    """
    declared = manifest.get(TRAINING_ROWS_KEY)
    out: dict[str, int] = {}
    if isinstance(declared, Mapping):
        for key, value in declared.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
                out[str(key)] = int(value)
    entries = manifest.get("channels")
    if isinstance(entries, Mapping):
        for key in UNSAVED_CHANNELS:
            entry = entries.get(key)
            if key in out or not isinstance(entry, Mapping) or entry.get("status") != "ok":
                continue
            extra = entry.get("extra")
            vectors = extra.get("support_vectors") if isinstance(extra, Mapping) else None
            if isinstance(vectors, (int, float)) and not isinstance(vectors, bool) and vectors > 0:
                out[key] = int(vectors)
    return out


def support_vector_count(estimator: Any) -> int:
    """Number of support vectors (training rows) stored inside a fitted CH3 model, 0 when it holds none.

    Looks through the calibration wrapper, the frozen estimator and the pipeline to the kernel SVM itself.
    """
    stack: list[Any] = [estimator]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        vectors = getattr(obj, "support_vectors_", None)
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            return int(vectors.shape[0])
        if isinstance(obj, Pipeline):
            stack.extend(step for _, step in obj.steps)
        for name in ("estimator", "calibrated_classifiers_"):
            inner = getattr(obj, name, None)
            if isinstance(inner, (list, tuple)):
                stack.extend(inner)
            elif inner is not None and not isinstance(inner, (str, int, float)):
                stack.append(inner)
    return 0


def _floats(values: Mapping[str, Any] | None) -> dict[str, float]:
    """Metric values as floats (None in a manifest stands for NaN)."""
    return {str(k): (float("nan") if v is None else float(v)) for k, v in (values or {}).items()}


def split_fingerprints(data: TrainingData) -> dict[str, str]:
    """SHA-256 fingerprints of the training and the test split: row positions in the prepared sample, class codes,
    class names and (for the test rows) detailed labels. Equal fingerprints mean the very same split."""
    classes = json.dumps(list(data.classes)).encode("utf-8")

    def digest(tag: bytes, rows: np.ndarray, codes: np.ndarray, labels: np.ndarray | None) -> str:
        h = hashlib.sha256(tag + b"\x00" + classes + b"\x00")
        h.update(np.ascontiguousarray(np.asarray(rows, dtype="<i8")).tobytes())
        h.update(b"\x00")
        h.update(np.ascontiguousarray(np.asarray(codes, dtype="<i8")).tobytes())
        if labels is not None:
            h.update(b"\x00" + "\n".join(str(v) for v in np.asarray(labels).tolist()).encode("utf-8"))
        return h.hexdigest()

    return {"train": digest(b"train", data.train_rows, data.y_train, None),
            "test": digest(b"test", data.test_rows, data.y_test, data.detailed_test_labels)}


def _values_digest(X: np.ndarray) -> str:
    """SHA-256 of a feature matrix's values (float32, -0.0 as +0.0, every NaN alike), with its shape."""
    values = np.array(X, dtype=np.float32, copy=True)
    if values.ndim != 2:
        values = values.reshape(len(values), -1)
    values[values == 0] = 0.0
    values[np.isnan(values)] = np.nan
    digest = hashlib.sha256(f"float32 {values.shape[0]}x{values.shape[1]}".encode("ascii"))
    digest.update(np.ascontiguousarray(values, dtype="<f4").tobytes())
    return digest.hexdigest()


def content_digests(data: TrainingData) -> dict[str, str]:
    """SHA-256 digests of the feature values of the training and the test matrix (digests only; no row is kept).

    Together with :func:`split_fingerprints` they pin a split down completely: the same rows, with the same labels
    and the very same feature values.
    """
    return {"train": _values_digest(data.X_train), "test": _values_digest(data.X_test)}


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """SHA-256 of a manifest's content apart from its own checksum (keys sorted, compact JSON, UTF-8).

    :func:`save_run` stores it under :data:`MANIFEST_DIGEST_KEY`, and :func:`load_bundle` refuses a bundle whose
    manifest no longer matches it.
    """
    body = {k: v for k, v in manifest.items() if k != MANIFEST_DIGEST_KEY}
    text = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest_text(manifest: Mapping[str, Any]) -> str:
    """The manifest as written to disk, with its checksum set."""
    sealed = {**manifest, MANIFEST_DIGEST_KEY: manifest_digest(manifest)}
    return json.dumps(sealed, indent=2, ensure_ascii=False, allow_nan=False)


def write_manifest(folder: Path, manifest: Mapping[str, Any]) -> None:
    """Write ``manifest`` (with a fresh :func:`manifest_digest`) as the manifest of the bundle in ``folder``.

    The file is replaced atomically. The models are unaffected; their checksums in ``manifest["files"]`` must
    still match them, or the bundle is refused on load. Meant for tools and tests that change a bundle on purpose.
    """
    target = Path(folder) / MANIFEST_FILE
    temporary = target.with_name(f".{MANIFEST_FILE}.{datetime.now(timezone.utc):%H%M%S%f}.tmp")
    temporary.write_text(_manifest_text(manifest), encoding="utf-8", newline="\n")
    replace_with_retry(temporary, target)


def _manifest_problem(manifest: Mapping[str, Any]) -> str | None:
    """Why a manifest makes its bundle unloadable before any file is read (layout version, own checksum)."""
    if manifest.get("bundle_format") != BUNDLE_FORMAT:
        return (f"it uses bundle format {manifest.get('bundle_format')!r}; this version of {APP_NAME} reads format "
                f"{BUNDLE_FORMAT}")
    recorded = manifest.get(MANIFEST_DIGEST_KEY)
    if not isinstance(recorded, str) or not recorded:
        return "its manifest carries no checksum"
    if recorded != manifest_digest(manifest):
        return "its manifest was changed or damaged after saving (it no longer matches its own checksum)"
    return None


def quantile_vectors(quantiles: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` synthetic vectors read off per-feature quantiles (float32, n x n_features).

    Every value is the feature's quantile curve at a random level between 0 and 1 (linear between the stored
    percentiles), the level drawn independently per feature with a seeded generator, so a vector follows each
    feature's training distribution without being, or copying, any row. A feature with no finite quantile gives NaN.
    """
    table = np.asarray(quantiles, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] != len(QUANTILE_LEVELS):
        raise ValueError(f"Expected quantiles of shape ({len(QUANTILE_LEVELS)}, n_features).")
    rng = np.random.default_rng(int(seed))
    levels = rng.random((int(n), table.shape[1]))
    out = np.full((int(n), table.shape[1]), np.nan, dtype=np.float32)
    for j in range(table.shape[1]):
        curve = table[:, j]
        if np.isfinite(curve).all():
            out[:, j] = np.interp(levels[:, j], QUANTILE_LEVELS, curve).astype(np.float32)
    return out


def probe_vectors(quantiles: np.ndarray, seed: int) -> np.ndarray:
    """The probe set of a bundle: :data:`PROBE_COUNT` quantile vectors, the first :data:`PROBE_MISSING_ROWS` of
    them with one value (a different feature each time) set missing."""
    vectors = quantile_vectors(quantiles, PROBE_COUNT, seed)
    n_features = vectors.shape[1]
    if n_features:
        for i in range(min(PROBE_MISSING_ROWS, len(vectors))):
            vectors[i, i % n_features] = np.nan
    return vectors


def _forests(estimator: Any) -> list[Any]:
    """Random forests inside ``estimator`` (the estimator itself or a step of a pipeline)."""
    if isinstance(estimator, _FORESTS):
        return [estimator]
    if isinstance(estimator, Pipeline):
        return [step for _, step in estimator.steps if isinstance(step, _FORESTS)]
    return []


@contextmanager
def deterministic(estimator: Any) -> Iterator[Any]:
    """Score ``estimator`` with reproducible sums: forests inside it predict on one thread meanwhile.

    A forest adds up its trees' probabilities in whichever order its threads finish, which can change the last bit
    of a probability from one call to the next; one thread fixes the order. Other estimators are unaffected.
    The previous ``n_jobs`` is restored afterwards. Blocks on different threads run one after the other.
    """
    forests = _forests(estimator)
    with _DETERMINISTIC_LOCK:
        previous = [forest.n_jobs for forest in forests]
        try:
            for forest in forests:
                forest.n_jobs = 1
            yield estimator
        finally:
            for forest, n_jobs in zip(forests, previous):
                forest.n_jobs = n_jobs


def score_exactly(estimator: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities as float64, predicted class codes) of ``estimator`` on ``X``, computed reproducibly."""
    with deterministic(estimator):
        proba = np.asarray(estimator.predict_proba(np.asarray(X, dtype=np.float32)), dtype=np.float64)
    classes = getattr(estimator, "classes_", None)
    picked = proba.argmax(axis=1)
    labels = np.asarray(classes)[picked] if classes is not None else picked
    return proba, np.asarray(labels, dtype=np.int64)


def _channel_label(key: str) -> str:
    """``"CH2 XGBoost"`` for a channel key (the key itself when unknown)."""
    style = CHANNEL_BY_KEY.get(key)
    return style.label if style is not None else key


def _ordered(keys: Any) -> list[str]:
    """Channel keys in CH1..CH5 order, unknown ones after them."""
    keys = [str(k) for k in keys]
    return [k for k in MODEL_KEYS if k in keys] + [k for k in keys if k not in MODEL_KEYS]


def _check_name(run_id: str) -> str:
    """Refuse run ids that cannot safely name a folder."""
    if not _SAFE_NAME.fullmatch(str(run_id)) or ".." in str(run_id):
        raise ValueError(f"Run id {run_id!r} cannot be used as a folder name.")
    return str(run_id)


def _feature_choice_dict(choice: FeatureChoice) -> dict[str, Any]:
    """A feature choice as plain values."""
    return {
        "mode": choice.mode,
        "columns": list(choice.columns),
        "include_port": bool(choice.include_port),
        "k": None if choice.k is None else int(choice.k),
        "ranking": None if choice.ranking is None else [[str(n), _plain(s)] for n, s in choice.ranking],
        "dropped_degenerate": list(choice.dropped_degenerate),
    }


def _feature_choice_from(values: Mapping[str, Any]) -> FeatureChoice:
    """The inverse of :func:`_feature_choice_dict`."""
    ranking = values.get("ranking")
    return FeatureChoice(
        mode=values["mode"],
        columns=tuple(str(c) for c in values["columns"]),
        include_port=bool(values.get("include_port", False)),
        k=None if values.get("k") is None else int(values["k"]),
        ranking=None if ranking is None else tuple(
            (str(n), float("nan") if s is None else float(s)) for n, s in ranking),
        dropped_degenerate=tuple(str(c) for c in values.get("dropped_degenerate", ())),
    )


def _train_request_from(manifest: Mapping[str, Any]) -> TrainRequest:
    """The recorded :class:`TrainRequest` (unknown keys ignored)."""
    values = dict(manifest.get("train_request") or {})
    known = {f.name for f in fields(TrainRequest)}
    values = {k: v for k, v in values.items() if k in known}
    if "channels" in values:
        values["channels"] = tuple(values["channels"])
    return TrainRequest(**values)


def _data_request_from(manifest: Mapping[str, Any]) -> DataRequest:
    """The recorded :class:`DataRequest` (unknown keys ignored)."""
    values = dict(manifest.get("data_request") or {})
    known = {f.name for f in fields(DataRequest)}
    values = {k: v for k, v in values.items() if k in known}
    if "files" in values:
        values["files"] = tuple(values["files"])
    return DataRequest(**values)


def _read_manifest(folder: Path) -> dict[str, Any]:
    """The manifest of a bundle folder; :class:`BundleIntegrityError` when it is missing or not valid JSON."""
    path = Path(folder) / MANIFEST_FILE
    if not path.is_file():
        raise BundleIntegrityError(f"{folder} is not a saved channel set: it has no {MANIFEST_FILE}.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleIntegrityError(f"The manifest of {folder.name} cannot be read: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BundleIntegrityError(f"The manifest of {folder.name} is not a JSON object.")
    return manifest


# --------------------------------------------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------------------------------------------
def _save_xgboost(pipeline: Any, folder: Path) -> list[str]:
    """Write an XGBoost pipeline as the native booster plus a joblib shell (other steps + constructor parameters)."""
    model = pipeline.steps[-1][1]
    params = dict(model.get_params())
    params["callbacks"] = None
    model.save_model(str(folder / XGB_BOOSTER_FILE))
    shell = {"format": 1, "steps": list(pipeline.steps[:-1]), "model_step": pipeline.steps[-1][0],
             "params": params}
    joblib.dump(shell, folder / XGB_SHELL_FILE, compress=JOBLIB_COMPRESS)
    return [XGB_BOOSTER_FILE, XGB_SHELL_FILE]


def _load_xgboost(folder: Path) -> Pipeline:
    """Put an XGBoost pipeline together again from :func:`_save_xgboost`'s two files."""
    shell = joblib.load(folder / XGB_SHELL_FILE)
    model = XGBClassifier(**shell["params"])
    model.load_model(str(folder / XGB_BOOSTER_FILE))
    return Pipeline([*shell["steps"], (shell["model_step"], model)])


def _is_xgboost_pipeline(estimator: Any) -> bool:
    """True for a pipeline ending in an XGBoost classifier."""
    return isinstance(estimator, Pipeline) and isinstance(estimator.steps[-1][1], XGBClassifier)


def _richer_metrics(run: TrainingRun) -> dict[str, dict[str, Any]]:
    """Full metrics per channel from :mod:`nids.evaluate`, when it offers them (else empty).

    Per fitted channel, the evaluation already cached on the run (03 Measure computes them once per run, through
    :func:`nids.evaluate.evaluate_run`) is used as it is. A channel without one (Measure not visited yet, or
    still computing in another session) gets its metrics computed from the stored test predictions and
    probabilities, when the module offers ``classification_metrics``: no model is asked to predict and nothing is
    timed, so saving stays quick. Both give the same numbers. Any failure leaves that channel with the headline
    metrics only.

    When the run can be weighted to its recorded traffic (:func:`nids.evaluate.traffic_readings`, computed once
    per run from the stored probabilities), each channel's entry also gets the weighted readings under
    ``traffic_<metric>`` keys, their standard errors (``traffic_accuracy_se``, ``traffic_balanced_accuracy_se``),
    the heavy-flow ranges (``traffic_accuracy_low``/``_high``, ``traffic_balanced_accuracy_low``/``_high``, when the
    run records its heavy-flow account) and ``traffic_flows_represented``: optional keys that older bundles simply
    lack.
    """
    from nids import evaluate

    out: dict[str, dict[str, Any]] = {}
    evaluations = getattr(run, "evaluations", None)
    cached = evaluations if isinstance(evaluations, Mapping) else {}
    compute = getattr(evaluate, "classification_metrics", None)
    has_rows = len(run.data.y_test) > 0
    for key in run.ok_channels():
        metrics = getattr(cached.get(key), "metrics", None)  # one lookup per key: safe while another thread adds
        if isinstance(metrics, Mapping):
            out[key] = _plain(metrics)
            continue
        result = run.channels[key]
        if not callable(compute) or not has_rows or result.y_pred is None or result.proba is None:
            continue
        try:
            out[key] = _plain(compute(run.data.y_test, result.y_pred, result.proba, run.data.n_classes))
        except Exception:  # noqa: BLE001 - the headline metrics are enough to save a run
            continue
    try:
        traffic = evaluate.traffic_readings(run) if has_rows else None
    except Exception:  # noqa: BLE001 - the estimate is optional; a run saves without it
        traffic = None
    for key, reading in (traffic or {}).items():
        if key in out:
            out[key].update(_plain({evaluate.TRAFFIC_FLOWS_COLUMN: reading.flows,
                                    **{f"{evaluate.TRAFFIC_PREFIX}{name}": value
                                       for name, value in reading.metrics.items()},
                                    **{column: reading.errors.get(name)
                                       for name, column in evaluate.TRAFFIC_ERROR_EXPORTS},
                                    **(evaluate.bound_values(reading.bounds) if reading.bounds else {})}))
    return out


def _kept_channels(run: TrainingRun, include_svm: bool = False) -> list[str]:
    """The fitted channels of ``run`` that its bundle keeps: every fitted one except :data:`UNSAVED_CHANNELS`, and
    CH3 too when ``include_svm`` (saved by choice)."""
    chosen = {"svm"} if include_svm else set()
    return [key for key in run.ok_channels() if key not in UNSAVED_CHANNELS or key in chosen]


def _best_kept(run: TrainingRun, kept: list[str]) -> tuple[str | None, float | None]:
    """The kept channel with the highest stored balanced accuracy (ties to the lower channel number)."""
    best: tuple[str | None, float | None] = (None, None)
    for key in kept:
        value = ((run.channels[key].extra or {}).get("metrics") or {}).get("balanced_accuracy")
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and (
                best[1] is None or float(value) > best[1]):
            best = (key, float(value))
    return best


def _channel_entry(result: ChannelResult, files: list[str], richer: Mapping[str, Any] | None,
                   kept: bool = True) -> dict[str, Any]:
    """The manifest entry of one channel (a fitted channel the bundle does not keep gets status ``"not_saved"``)."""
    extra = dict(result.extra or {})
    unsaved = result.ok and result.key in UNSAVED_CHANNELS and not kept
    notes = [str(n) for n in result.notes]
    if unsaved:
        notes.append(UNSAVED_CHANNELS[result.key])
    entry: dict[str, Any] = {
        "status": NOT_SAVED if unsaved else result.status,
        "fit_status": result.status,
        "badge": CHANNEL_BY_KEY[result.key].badge if result.key in CHANNEL_BY_KEY else result.key,
        "rows_used": int(result.rows_used),
        "rows_available": int(result.rows_available),
        "notes": notes,
        "error": result.error,
        "fit_seconds": _plain(result.fit_seconds),
        "predict_seconds": _plain(result.predict_seconds),
        "flows_per_second": _plain(extra.get("flows_per_second")),
        "metrics": _plain(extra.get("metrics") or {}),
        "extra": _plain(extra),
        "files": list(files),
    }
    if richer:
        entry["evaluation_metrics"] = _plain(richer)
    return entry


def _contents_text(channel_files: Mapping[str, list[str]], training_rows: Mapping[str, int]) -> str:
    """The manifest's plain-words account of what the bundle holds, dataset rows above all."""
    base = "Models, metadata, synthetic probe vectors and per-feature quantiles"
    if not training_rows:
        return (f"{base}; no dataset rows. CH3 (the kernel SVM) is not kept, because its model is made of training "
                "rows (its support vectors); it is saved only when you choose to.")
    held = []
    for key, count in training_rows.items():
        name = ", ".join(channel_files.get(key, [])) or f"{key}.joblib"
        held.append(f"{name} ({_channel_label(key)}, saved by choice) holds {int(count):,} training rows: its "
                    "support vectors, stored after gap filling, signed logarithm and standardisation, which the "
                    "scaler saved with it turns back into the original values")
    return f"{base}. " + "; ".join(held) + ". No other file holds dataset rows."


def _build_manifest(run: TrainingRun, channel_files: Mapping[str, list[str]], files: Mapping[str, str],
                    probe_seed: int, training_rows: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Everything recorded about ``run`` in its bundle (plain values); ``training_rows`` declares the training rows
    a saved model holds (CH3's support vectors when it is saved by choice)."""
    training_rows = {str(k): int(v) for k, v in (training_rows or {}).items()}
    data = run.data
    reports = data.reports or {}
    richer = _richer_metrics(run)
    prints = split_fingerprints(data)
    values = content_digests(data)
    best_key, best_value = _best_kept(run, [k for k in _ordered(channel_files)])
    svm = run.channels.get("svm")
    rows = dict(reports.get("rows") or {})
    return _plain({
        "bundle_format": BUNDLE_FORMAT,
        "app": {"name": APP_NAME, "version": __version__},
        "run_id": run.run_id,
        "created_utc": run.created_utc,
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": run.request.mode,
        "classes": list(data.classes),
        "feature_names": list(data.feature_names),
        "feature_choice": _feature_choice_dict(data.feature_choice),
        "train_request": run.request.to_dict(),
        "data_request": asdict(run.data_request),
        "data_source": {
            "source": run.data_request.source,
            "files": list(run.data_request.files),
            "data_dir": run.data_request.data_dir,
            "description": run.data_request.describe(),
        },
        "seed": int(run.request.seed),
        "nonfinite_strategy": run.data_request.nonfinite_strategy,
        "dataset_fingerprint": run.dataset_fingerprint,
        "train_fingerprint": prints["train"],
        "test_fingerprint": prints["test"],
        "train_values_sha256": values["train"],
        "test_values_sha256": values["test"],
        "rows": {
            "prepared": rows.get("prepared"),
            "used": rows.get("used"),
            "train": int(len(data.y_train)),
            "test": int(len(data.y_test)),
            "train_by_class": reports.get("class_counts_train") or {},
            "test_by_class": reports.get("class_counts_test") or {},
        },
        "svm_rows_used": (svm.extra.get("svm_rows_used") if svm is not None and svm.ok else None),
        "seconds": float(run.seconds),
        "prep_seconds": float(run.prep_seconds),
        "reports": reports,
        "channels": {key: _channel_entry(run.channels[key], list(channel_files.get(key, [])), richer.get(key),
                                         kept=key in channel_files)
                     for key in _ordered(run.channels)},
        TRAINING_ROWS_KEY: training_rows,
        "best_channel": best_key,
        "best_balanced_accuracy": best_value,
        "library_versions": library_versions(),
        "files": dict(sorted(files.items())),
        "probe": {
            "file": PROBE_FILE,
            "count": PROBE_COUNT,
            "seed": int(probe_seed),
            "missing_value_rows": PROBE_MISSING_ROWS,
            "method": "per-feature training quantiles read at independent random levels (synthetic, not rows)",
            "channels": [k for k in _ordered(channel_files)],
            "check": "labels and float64 probabilities must be identical (forests scored on one thread)",
        },
        "quantiles": {"file": QUANTILE_FILE, "levels": len(QUANTILE_LEVELS), "features": len(data.feature_names)},
        "contents": _contents_text(channel_files, training_rows),
    })


def save_run(run: TrainingRun, directory: Path | None = None, *, include_svm: bool = False) -> Path:
    """Save the fitted channels of ``run`` as a bundle and return its folder (``<directory>/<run id>``).

    ``directory`` defaults to ``settings.MODELS_DIR``. Every file is written into a hidden temporary folder next
    to the target and the folder is renamed into place at the end, so a bundle is either complete or absent.
    Saving a run that is already saved there returns the existing folder as it is (whatever ``include_svm`` says;
    :func:`training_rows_inside` of its manifest tells what it holds). Only channels with status ``"ok"`` are
    written. CH3 (:data:`UNSAVED_CHANNELS`) is left out by default and recorded as ``"not_saved"``; with
    ``include_svm=True`` it is written as ``svm.joblib`` like any scikit-learn channel (checksummed, probe-checked,
    verified on load), and the manifest declares the training rows it holds (its support vectors) under
    :data:`TRAINING_ROWS_KEY` and in its ``contents`` text. Sets ``run.bundle_path``. Raises ``ValueError`` when no
    channel the bundle would keep was fitted (a CH3-only run without ``include_svm``), or when the run has no
    held-out rows in memory (a loaded run without its data cannot be saved again). When a save fails and its hidden
    work folder cannot be removed either, the ``OSError`` raised names that folder (:func:`find_leftovers` lists it).
    """
    run_id = _check_name(run.run_id)
    fitted = run.ok_channels()
    if not fitted:
        raise ValueError("Nothing to save: no channel of this run was fitted.")
    root = Path(directory) if directory is not None else Path(settings_mod.MODELS_DIR)
    final = root / run_id
    if final.exists():
        if (final / MANIFEST_FILE).is_file() and _read_manifest(final).get("run_id") == run_id:
            run.bundle_path = str(final)
            return final
        raise FileExistsError(f"{final} already exists and is not the saved bundle of run {run_id}.")
    kept = _kept_channels(run, include_svm)
    if not kept:
        names = ", ".join(_channel_label(key) for key in fitted)
        raise ValueError(f"Nothing to save: the only fitted channel is {names}, which saved sets leave out unless it "
                         "is saved by choice (include_svm=True; the \"Also save CH3 (RBF SVM)\" box at the Logbook), "
                         "because a kernel SVM is made of its training rows (its support vectors, standardised).")
    if not len(run.data.y_test):  # (not the property: runs made before an in-place code reload lack it)
        raise ValueError("This run has no held-out rows in memory, so its fingerprints cannot be recorded. Rebuild "
                         "them first (load the bundle again with the data folder available).")
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{SAVING_PREFIX}{run_id}-", dir=root))
    try:
        channel_files: dict[str, list[str]] = {}
        for key in kept:
            estimator = run.channels[key].estimator
            if _is_xgboost_pipeline(estimator):
                channel_files[key] = _save_xgboost(estimator, work)
            else:
                name = f"{key}.joblib"
                joblib.dump(estimator, work / name, compress=JOBLIB_COMPRESS)
                channel_files[key] = [name]
        quantiles = np.asarray(run.feature_quantiles, dtype=np.float32)
        np.savez(work / QUANTILE_FILE, quantiles=quantiles, levels=np.asarray(QUANTILE_LEVELS, dtype=np.float64))
        probe_seed = int(run.request.seed) + PROBE_SEED_OFFSET
        vectors = probe_vectors(quantiles, probe_seed)
        arrays: dict[str, np.ndarray] = {"vectors": vectors}
        for key in kept:
            proba, labels = score_exactly(run.channels[key].estimator, vectors)
            arrays[f"proba__{key}"] = proba
            arrays[f"labels__{key}"] = labels
        np.savez(work / PROBE_FILE, **arrays)
        files = {p.name: sha256_file(p) for p in sorted(work.iterdir()) if p.is_file()}
        training_rows: dict[str, int] = {}
        for key in kept:
            if key in UNSAVED_CHANNELS:  # saved by choice: declare the training rows its model holds
                result = run.channels[key]
                count = support_vector_count(result.estimator) or int(
                    (result.extra or {}).get("support_vectors") or 0)
                training_rows[key] = int(count)
        manifest = _build_manifest(run, channel_files, files, probe_seed, training_rows)
        (work / MANIFEST_FILE).write_text(_manifest_text(manifest), encoding="utf-8", newline="\n")
        try:
            replace_with_retry(work, final)
        except OSError:
            # Another session may have saved the same run a moment ago: then its bundle stands (whatever of the
            # work folder cannot be removed is listed by find_leftovers).
            if (final / MANIFEST_FILE).is_file() and _read_manifest(final).get("run_id") == run_id:
                _discard_work(work)
                run.bundle_path = str(final)
                return final
            raise
    except BaseException as exc:
        left = _discard_work(work)
        if left and isinstance(exc, Exception):
            raise OSError(f"{_one_line(exc)} {left}") from exc
        raise
    run.bundle_path = str(final)
    return final


# --------------------------------------------------------------------------------------------------------------
# Loading and verifying
# --------------------------------------------------------------------------------------------------------------
def _check_manifest(folder: Path, manifest: Mapping[str, Any]) -> None:
    """Refuse a bundle of another layout version, or one whose manifest no longer matches its own checksum."""
    problem = _manifest_problem(manifest)
    if problem is not None:
        raise BundleIntegrityError(f"The saved channel set {folder.name} is refused: {problem}.")


def _check_shapes(folder: Path, manifest: Mapping[str, Any], channels: Mapping[str, Any],
                  quantiles: np.ndarray) -> None:
    """The manifest's mode, classes and feature names must fit the checksummed files.

    The number of feature names must equal the columns of the stored quantiles and what every model was fitted
    on, and the number of classes must equal what every model predicts (two in binary mode).
    """
    classes = [str(c) for c in manifest.get("classes") or []]
    names = [str(n) for n in manifest.get("feature_names") or []]
    mode = manifest.get("mode")
    problems: list[str] = []
    if mode not in ("binary", "multiclass"):
        problems.append(f"unknown mode {mode!r}")
    if len(classes) < 2 or (mode == "binary" and len(classes) != 2) or len(set(classes)) != len(classes):
        problems.append(f"{len(classes)} class names for a {mode} run")
    columns = int(quantiles.shape[1]) if quantiles.ndim == 2 else 0
    if len(set(names)) != len(names) or columns != len(names):
        problems.append(f"{len(names)} feature names for {columns} stored feature quantiles")
    for key, estimator in channels.items():
        n_in = getattr(estimator, "n_features_in_", None)
        if n_in is not None and int(n_in) != len(names):
            problems.append(f"{_channel_label(key)} was fitted on {int(n_in)} columns, not {len(names)}")
        model_classes = getattr(estimator, "classes_", None)
        if model_classes is not None and len(model_classes) != len(classes):
            problems.append(f"{_channel_label(key)} reads {len(model_classes)} classes, not {len(classes)}")
    if problems:
        raise BundleIntegrityError(f"The manifest of {folder.name} does not fit its models: " + "; ".join(problems)
                                   + ". The bundle is refused.")


def _held_file_message(name: str, folder: Path) -> str:
    """The message for a bundle file the operating system refused to open."""
    return (f"{name} in {folder.name} cannot be opened: another program may be holding it, or access is denied. "
            "Close it elsewhere and load the set again.")


def _missing_code_message(key: str, folder: Path) -> str:
    """The message for a model file that names program code this installation does not have.

    A saved model records where its classes live in the program. A set written by an earlier version of the app,
    whose code was arranged differently, or with a library that is not installed here, cannot be put together
    again. The message leaves the missing module's name out: it is no use to the reader.
    """
    return (f"{_channel_label(key)} could not be read from {folder.name}: its model file was written by an earlier "
            f"version of {APP_NAME}, or with a library that is not installed here, and this version cannot open it. "
            "Fit the channels again at 02 Fit and save a new set.")


def _check_files(folder: Path, manifest: Mapping[str, Any]) -> None:
    """Every file the bundle needs is listed, present and matches its recorded SHA-256."""
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise BundleIntegrityError(f"The manifest of {folder.name} lists no files to check.")
    needed = {PROBE_FILE, QUANTILE_FILE}
    for entry in (manifest.get("channels") or {}).values():
        if isinstance(entry, Mapping) and entry.get("status") == "ok":
            needed.update(str(name) for name in entry.get("files") or [])
    unlisted = sorted(needed - set(files))
    if unlisted:
        raise BundleIntegrityError(f"The manifest of {folder.name} has no checksum for {', '.join(unlisted)}.")
    for name, recorded in files.items():
        if Path(str(name)).name != name or name == MANIFEST_FILE:
            raise BundleIntegrityError(f"The manifest of {folder.name} lists an unexpected file: {name!r}.")
        path = folder / str(name)
        if not path.is_file():
            raise BundleIntegrityError(f"{name} is missing from the saved channel set {folder.name}.")
        try:
            digest = sha256_file(path)
        except PermissionError as exc:
            raise BundleReadError(_held_file_message(str(name), folder)) from exc
        if digest != str(recorded):
            raise BundleIntegrityError(f"{name} in {folder.name} does not match the checksum recorded when it was "
                                       "saved: the file was changed or damaged afterwards. The bundle is refused.")


def _load_channels(folder: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The fitted estimators of every channel saved with status ``"ok"``, in channel order."""
    channels: dict[str, Any] = {}
    entries = manifest.get("channels") or {}
    for key in _ordered(entries):
        entry = entries[key]
        if entry.get("status") != "ok":
            continue
        names = [str(n) for n in entry.get("files") or []]
        try:
            if XGB_BOOSTER_FILE in names:
                channels[key] = _load_xgboost(folder)
            elif names:
                channels[key] = joblib.load(folder / names[0])
        except PermissionError as exc:
            held = exc.filename if isinstance(exc.filename, str) else (names[0] if names else key)
            raise BundleReadError(_held_file_message(Path(held).name, folder)) from exc
        except ModuleNotFoundError as exc:
            raise BundleReadError(_missing_code_message(key, folder)) from exc
        except Exception as exc:  # noqa: BLE001 - reported with the channel and file named
            raise BundleReadError(f"{_channel_label(key)} could not be read from {folder.name} "
                                  f"({type(exc).__name__}: {exc}).") from exc
    return channels


def _verification_message(identical: Mapping[str, bool], diffs: Mapping[str, float],
                          mismatches: Mapping[str, tuple[str, str]]) -> tuple[bool, str]:
    """(verified, sentence for the Logbook) from the probe comparison and the version comparison."""
    n = len(identical)
    failed = [k for k, same in identical.items() if not same]
    largest = max(diffs.values(), default=0.0)
    if not mismatches:
        if n and not failed:
            what = "the only channel reproduced its" if n == 1 else f"all {n} channels reproduced their"
            return True, f"Verified: {what} saved probe readings exactly."
        if not n:
            return False, "Not verified: the bundle holds no fitted channel."
        names = ", ".join(f"{_channel_label(k)} (largest difference {diffs[k]:.3g})" for k in failed)
        return False, (f"Verification failed: {names} did not reproduce the saved probe readings although the "
                       "library versions are the ones the bundle was saved with. Treat these readings with "
                       "suspicion and fit again.")
    changed = "; ".join(f"{name} {saved} -> {now}" for name, (saved, now) in mismatches.items())
    if failed:
        outcome = (f"{len(failed)} of {n} channels read the probes differently (largest difference "
                   f"{largest:.3g}: {', '.join(_channel_label(k) for k in failed)})")
    else:
        outcome = f"all {n} channels still reproduce their saved probe readings exactly"
    return False, f"Not verified: the libraries differ from those the bundle was saved with ({changed}); {outcome}."


def _verify(folder: Path, manifest: Mapping[str, Any], channels: Mapping[str, Any]) -> VerificationReport:
    """Re-score the probe vectors with every loaded channel and compare versions."""
    saved = {str(k): str(v) for k, v in (manifest.get("library_versions") or {}).items()}
    current = library_versions()
    mismatches: dict[str, tuple[str, str]] = {}
    for name in [*current, *[k for k in saved if k not in current]]:
        before, now = saved.get(name, "unknown"), current.get(name, "unknown")
        if before != now:
            mismatches[name] = (before, now)
    saved_app = str((manifest.get("app") or {}).get("version", "unknown"))
    if saved_app != __version__:
        mismatches["nids"] = (saved_app, __version__)
    identical: dict[str, bool] = {}
    diffs: dict[str, float] = {}
    try:
        with np.load(folder / PROBE_FILE, allow_pickle=False) as probes:
            vectors = probes["vectors"]
            stored = {name: probes[name] for name in probes.files if name != "vectors"}
    except Exception as exc:  # noqa: BLE001 - an unreadable probe file is a damaged bundle
        raise BundleReadError(f"The probe set of {folder.name} cannot be read ({exc}).") from exc
    for key, estimator in channels.items():
        expected_proba = stored.get(f"proba__{key}")
        expected_labels = stored.get(f"labels__{key}")
        if expected_proba is None or expected_labels is None:
            identical[key], diffs[key] = False, float("inf")
            continue
        try:
            proba, labels = score_exactly(estimator, vectors)
        except Exception:  # noqa: BLE001 - a channel that cannot score its probes is not verified
            identical[key], diffs[key] = False, float("inf")
            continue
        if proba.shape != expected_proba.shape:
            identical[key], diffs[key] = False, float("inf")
            continue
        identical[key] = bool(np.array_equal(proba, expected_proba) and np.array_equal(labels, expected_labels))
        diffs[key] = float(np.max(np.abs(proba - expected_proba))) if proba.size else 0.0
    verified, message = _verification_message(identical, diffs, mismatches)
    return VerificationReport(hashes_ok=True, probes_identical=identical, max_abs_diff=diffs,
                              version_mismatches=mismatches, verified=verified, message=message)


def load_bundle(path: Path) -> LoadedBundle:
    """Read the bundle in ``path``: check every file, load the channels, re-score the probes, compare versions.

    Raises :class:`BundleIntegrityError` when the folder is not a bundle, uses another layout version, its manifest
    differs from its own recorded checksum, or a file is missing or differs from its recorded checksum (the bundle
    is refused before any model file is read), or when the models do not fit the manifest's classes and feature
    count; and :class:`BundleReadError` when an intact file cannot be read. Version differences and probe
    differences do not raise: they are described in ``verification`` (see :class:`VerificationReport`).
    """
    folder = Path(path)
    manifest = _read_manifest(folder)
    _check_manifest(folder, manifest)
    _check_files(folder, manifest)
    channels = _load_channels(folder, manifest)
    try:
        with np.load(folder / QUANTILE_FILE, allow_pickle=False) as stored:
            quantiles = np.asarray(stored["quantiles"], dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - an unreadable quantile file is a damaged bundle
        raise BundleReadError(f"The quantiles of {folder.name} cannot be read ({exc}).") from exc
    _check_shapes(folder, manifest, channels, quantiles)
    verification = _verify(folder, manifest, channels)
    return LoadedBundle(
        path=folder, manifest=manifest, channels=channels,
        classes=tuple(str(c) for c in manifest.get("classes") or ()),
        feature_names=tuple(str(c) for c in manifest.get("feature_names") or ()),
        mode=str(manifest.get("mode", "binary")), verification=verification, quantiles=quantiles,
    )


# --------------------------------------------------------------------------------------------------------------
# Listing and deleting
# --------------------------------------------------------------------------------------------------------------
def _summary(folder: Path, manifest: Mapping[str, Any]) -> BundleSummary:
    """The Logbook line of one bundle."""
    request = manifest.get("data_request") or {}
    source = str(request.get("source", "cicids"))
    files = list(request.get("files") or [])
    if source == "synthetic":
        shown = "synthetic flows"
    else:
        shown = f"CIC-IDS2017 ({len(files)} file{'s' if len(files) != 1 else ''})"
    rows = manifest.get("rows") or {}
    entries = manifest.get("channels") or {}
    best = manifest.get("best_balanced_accuracy")
    best_channel = manifest.get("best_channel")
    return BundleSummary(
        path=folder, run_id=str(manifest.get("run_id", folder.name)),
        created_utc=str(manifest.get("created_utc", "")), mode=str(manifest.get("mode", "")), source=shown,
        rows=int(rows.get("train", 0) or 0) + int(rows.get("test", 0) or 0),
        channels=tuple(k for k in _ordered(entries) if (entries[k] or {}).get("status") == "ok"),
        best_channel=None if best_channel is None else str(best_channel),
        best_balanced_accuracy=None if best is None else float(best),
        problem=_manifest_problem(manifest),
        holds_training_rows=sum(training_rows_inside(manifest).values()),
    )


def read_bundle_summary(path: Path) -> BundleSummary:
    """The Logbook line of the bundle in ``path``, read from its manifest only (no model is loaded).

    Raises :class:`BundleIntegrityError` when the folder holds no readable manifest.
    """
    folder = Path(path)
    return _summary(folder, _read_manifest(folder))


def scan_bundles(directory: Path | None = None) -> tuple[list[BundleSummary], list[tuple[Path, str]]]:
    """The bundles in ``directory`` (default ``settings.MODELS_DIR``), newest first, and the folders that hold a
    manifest which cannot be read, each with the reason.

    Only the manifests are read. Hidden folders and folders without a manifest are not bundles and are left out
    here; the hidden work folders a save or a delete left behind are listed by :func:`find_leftovers`. A manifest
    that cannot be read, or whose content is malformed, never raises: its folder goes to the second list.
    """
    root = Path(directory) if directory is not None else Path(settings_mod.MODELS_DIR)
    if not root.is_dir():
        return [], []
    found: list[BundleSummary] = []
    unreadable: list[tuple[Path, str]] = []
    for folder in sorted(root.iterdir()):
        try:
            if not folder.is_dir() or folder.name.startswith(".") or not (folder / MANIFEST_FILE).is_file():
                continue
            found.append(_summary(folder, _read_manifest(folder)))
        except Exception as exc:  # noqa: BLE001 - one bad folder must not hide the others
            text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            unreadable.append((folder, text))
    return sorted(found, key=lambda s: (s.created_utc, s.run_id), reverse=True), unreadable


def list_bundles(directory: Path | None = None) -> list[BundleSummary]:
    """Every bundle in ``directory`` (default ``settings.MODELS_DIR``), newest first (see :func:`scan_bundles`;
    folders whose manifest cannot be read are left out)."""
    return scan_bundles(directory)[0]


def _remove_tree(folder: Path, retries: int = 3) -> None:
    """``shutil.rmtree`` that clears read-only flags and retries briefly (antivirus scans can hold files)."""
    def clear_readonly(function: Callable[..., Any], path: str, exc: BaseException) -> None:
        Path(path).chmod(0o700)
        function(path)

    for attempt in range(retries + 1):
        try:
            shutil.rmtree(folder, onexc=clear_readonly)
            return
        except PermissionError:
            if attempt == retries:
                raise
            time.sleep(0.2 * (attempt + 1))


def _remove_file(path: Path, retries: int = 3) -> None:
    """Delete one file, clearing a read-only flag and retrying briefly (antivirus scans can hold files)."""
    for attempt in range(retries + 1):
        try:
            Path(path).unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == retries:
                raise
            try:
                Path(path).chmod(0o700)
            except OSError:
                pass
            time.sleep(0.2 * (attempt + 1))


def _remove_bundle_files(folder: Path) -> None:
    """Remove a bundle folder: models made of training rows first, the manifest last, then the folder itself.

    So whatever a held file leaves behind holds no training rows when it can be helped, and still describes itself
    (its manifest) while any file of it remains.
    """
    entries = sorted(Path(folder).iterdir())
    first = [p for p in entries if p.name in _ROW_FILES]
    last = [p for p in entries if p.name == MANIFEST_FILE]
    for entry in first + [p for p in entries if p not in first and p not in last] + last:
        if entry.is_dir() and not entry.is_symlink():
            _remove_tree(entry)
        else:
            _remove_file(entry)
    _remove_tree(folder)


def _one_line(exc: BaseException) -> str:
    """The first line of an exception's message (its type name when it has none)."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return lines[0] if lines else type(exc).__name__


def _leftover_sentence(folder: Path) -> str:
    """One sentence naming a hidden work folder that could not be removed, and what is still in it."""
    try:
        names = sorted(p.name for p in Path(folder).iterdir())
    except OSError:
        names = []
    held = ", ".join(names) if names else "no files"
    rows = (" svm.joblib among them still holds CH3's training rows (its support vectors)."
            if any(name in _ROW_FILES for name in names) else "")
    return (f"The hidden folder {folder} is left over, holding {held}.{rows} The Logbook lists such leftovers and "
            "removes them once nothing holds their files.")


def _discard_work(work: Path) -> str:
    """Remove the work folder of a save that did not finish; returns "" when it is gone, else a sentence naming
    what is left (see :func:`find_leftovers`)."""
    try:
        _remove_tree(work)
    except FileNotFoundError:
        return ""
    except OSError:
        return _leftover_sentence(work)
    return ""


def delete_bundle(path: Path) -> None:
    """Delete a saved bundle folder for good.

    Only a folder holding a ``manifest.json`` is deleted. It is first renamed to a hidden name (one atomic step, so
    the Logbook never lists a half-deleted bundle) and then removed, any model made of training rows
    (``svm.joblib``) first and the manifest last. Raises ``FileNotFoundError`` when the folder does not exist and
    :class:`BundleIntegrityError` when it is not a bundle. When a file stays held past the retries, the bundle is
    renamed back to its own name when possible (so it is listed again and can be deleted again), and a
    :class:`BundleDeleteError` says so, or names the hidden folder left over (:func:`find_leftovers` lists it).
    """
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"No saved channel set at {folder}.")
    if not (folder / MANIFEST_FILE).is_file():
        raise BundleIntegrityError(f"{folder} is not a saved channel set (no {MANIFEST_FILE}); nothing was deleted.")
    doomed = folder.parent / f"{DELETING_PREFIX}{folder.name}-{datetime.now(timezone.utc):%H%M%S%f}"
    replace_with_retry(folder, doomed)
    try:
        _remove_bundle_files(doomed)
    except OSError as exc:
        reason = f"The saved channel set {folder.name} could not be deleted completely ({_one_line(exc)})."
        if (doomed / MANIFEST_FILE).is_file() and not folder.exists():
            try:
                replace_with_retry(doomed, folder)
            except OSError:
                pass
            else:
                raise BundleDeleteError(f"{reason} It was put back under its own name, so it is listed again, "
                                        "though some of its files may be gone: delete it again once nothing holds "
                                        "them.") from exc
        raise BundleDeleteError(f"{reason} {_leftover_sentence(doomed)}") from exc


def find_leftovers(directory: Path | None = None, *, min_age_seconds: float = 0.0) -> list[LeftoverFolder]:
    """The hidden work folders (``.saving-*``, ``.deleting-*``) in ``directory`` (default ``settings.MODELS_DIR``).

    A save or a delete removes its work folder when it finishes, so such a folder is either still in use or left
    over by one that failed (a file held by another program past the retries). Only folders whose content has not
    changed for ``min_age_seconds`` are listed, so a caller can leave out a save or a delete still running in
    another session. Never raises for a folder that vanishes meanwhile; returns ``[]`` without a models folder.
    """
    root = Path(directory) if directory is not None else Path(settings_mod.MODELS_DIR)
    if not root.is_dir():
        return []
    now = time.time()
    found: list[LeftoverFolder] = []
    for folder in sorted(root.iterdir()):
        name = folder.name
        kind = "save" if name.startswith(SAVING_PREFIX) else "delete" if name.startswith(DELETING_PREFIX) else ""
        if not kind:
            continue
        try:
            if not folder.is_dir():
                continue
            entries = list(folder.iterdir())
            changed = max([folder.stat().st_mtime, *(entry.stat().st_mtime for entry in entries)])
        except OSError:
            continue
        age = max(now - changed, 0.0)
        if age < float(min_age_seconds):
            continue
        names = tuple(sorted(entry.name for entry in entries))
        found.append(LeftoverFolder(path=folder, kind=kind, files=names,
                                    holds_training_rows=any(n in _ROW_FILES for n in names), age_seconds=age))
    return found


def remove_leftover(path: Path) -> None:
    """Remove a hidden work folder listed by :func:`find_leftovers` for good (nothing happens when it is gone).

    Raises ``ValueError`` for any folder not named like a save's or a delete's work folder, and ``OSError`` when a
    file in it is still held after the retries.
    """
    folder = Path(path)
    if not folder.name.startswith((SAVING_PREFIX, DELETING_PREFIX)):
        raise ValueError(f"{folder} is not the work folder of a save or a delete; nothing was removed.")
    if not folder.exists():
        return
    try:
        _remove_tree(folder)
    except FileNotFoundError:
        return


# --------------------------------------------------------------------------------------------------------------
# Rebuilding the held-out rows and restoring the run
# --------------------------------------------------------------------------------------------------------------
def rebuild_prepared(
    bundle: LoadedBundle,
    *,
    data_dir: str | None,
    read_file: FileReader | None = None,
    stage_file: FileStager | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> PreparedDataset:
    """Draw the saved run's sample again (01 Sample with the recorded request) and check its fingerprint.

    Synthetic runs are regenerated from their seed. For CIC-IDS2017 runs the files are read from ``data_dir``, or,
    when it is None, from the folder recorded when the run was saved. ``read_file``/``stage_file``/``progress``
    are passed to :func:`~nids.data.prepare.prepare_dataset` (the UI passes its cached readers). Raises
    :class:`RebuildError` with a readable message when the folder or a file is missing, or when the sample drawn
    differs from the one the run was fitted on.
    """
    request = _data_request_from(bundle.manifest)
    if request.source == "cicids":
        folder_text = (data_dir or "").strip() or (request.data_dir or "")
        if not folder_text:
            raise RebuildError("The run was fitted on CIC-IDS2017 files and no data folder is set. Point the Bench "
                               f"at the folder holding {', '.join(request.files)}, then load the run again.")
        folder = Path(folder_text)
        if not folder.is_dir():
            raise RebuildError(f"The data folder {folder} does not exist. Point the Bench at the folder holding "
                               f"{', '.join(request.files)}, then load the run again.")
        missing = [name for name in request.files if not (folder / name).is_file()]
        if missing:
            raise RebuildError(f"The data folder {folder} lacks {', '.join(missing)}, which the run was fitted on.")
        request = replace(request, data_dir=str(folder))
    try:
        prepared = prepare_dataset(request, read_file=read_file, stage_file=stage_file, progress=progress)
    except DataFileError as exc:
        raise RebuildError(f"The sample could not be drawn again: {exc}") from exc
    recorded = str(bundle.manifest.get("dataset_fingerprint", ""))
    if prepared.fingerprint != recorded:
        raise RebuildError(
            f"The sample drawn again differs from the one the run was fitted on (fingerprint "
            f"{prepared.fingerprint[:12]} instead of {recorded[:12]}): the files or the program have changed since. "
            "The held-out rows cannot be rebuilt identically.")
    return prepared


def rebuild_training_data(
    bundle: LoadedBundle,
    *,
    data_dir: str | None,
    prepared: PreparedDataset | None = None,
    read_file: FileReader | None = None,
    stage_file: FileStager | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> TrainingData:
    """Rebuild the training and test matrices the saved run was fitted and measured on.

    Re-runs 01 Sample (:func:`rebuild_prepared`, unless the matching ``prepared`` sample is passed) and the matrix
    building of 02 Fit (:func:`~nids.models.train.build_training_data`) with the recorded requests, then
    checks that classes, feature columns, both split fingerprints (rows and labels) and both value digests
    (:func:`content_digests`) equal the recorded ones. Nothing is fitted: a Top-K run reuses the feature ranking
    recorded in its manifest instead of ranking again. Raises :class:`RebuildError` when the data are unavailable
    or anything differs.
    """
    manifest = bundle.manifest
    if prepared is None:
        prepared = rebuild_prepared(bundle, data_dir=data_dir, read_file=read_file, stage_file=stage_file,
                                    progress=progress)
    elif prepared.fingerprint != manifest.get("dataset_fingerprint"):
        raise RebuildError("The sample given is not the one the run was fitted on (different fingerprint).")
    request = _train_request_from(manifest)
    ranking = None
    if request.feature_mode == "topk" and (manifest.get("feature_choice") or {}).get("ranking"):
        ranking = _feature_choice_from(manifest["feature_choice"]).ranking
    try:
        data = build_training_data(prepared, request, ranking=ranking)
    except SingleClassError as exc:
        raise RebuildError(f"The training rows could not be built again: {exc}") from exc
    if tuple(data.classes) != bundle.classes:
        raise RebuildError(f"The rebuilt classes ({', '.join(data.classes)}) differ from the saved ones "
                           f"({', '.join(bundle.classes)}).")
    if tuple(data.feature_names) != bundle.feature_names:
        raise RebuildError("The rebuilt feature columns differ from the saved ones.")
    prints = split_fingerprints(data)
    for part in ("train", "test"):
        recorded = str(manifest.get(f"{part}_fingerprint", ""))
        if prints[part] != recorded:
            raise RebuildError(f"The rebuilt {part} rows differ from those of the saved run (fingerprint "
                               f"{prints[part][:12]} instead of {recorded[:12]}).")
    values = content_digests(data)
    for part in ("train", "test"):
        recorded = str(manifest.get(f"{part}_values_sha256", ""))
        if values[part] != recorded:
            raise RebuildError(
                f"The rebuilt {part} rows are the same rows with the same labels, but their feature values differ "
                f"from those the run was fitted on (digest {values[part][:12]} instead of {recorded[:12]}): a data "
                "file, or the way the program reads or cleans it, has changed since.")
    return data


def _empty_data(bundle: LoadedBundle) -> TrainingData:
    """Training data without rows, carrying the bundle's classes, feature names, feature choice and reports."""
    n_features = len(bundle.feature_names)
    manifest = bundle.manifest
    return TrainingData(
        X_train=np.empty((0, n_features), dtype=np.float32), X_test=np.empty((0, n_features), dtype=np.float32),
        y_train=np.empty(0, dtype=np.int64), y_test=np.empty(0, dtype=np.int64), classes=bundle.classes,
        detailed_test_labels=np.empty(0, dtype=str), feature_names=bundle.feature_names,
        feature_choice=_feature_choice_from(manifest.get("feature_choice") or {
            "mode": "curated", "columns": list(bundle.feature_names)}),
        train_rows=np.empty(0, dtype=np.int64), test_rows=np.empty(0, dtype=np.int64),
        reports=dict(manifest.get("reports") or {}),
    )


def _score_rows(estimator: Any, X: np.ndarray, n_classes: int, *,
                after_block: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, float]:
    """Probabilities (float32, n x K, code order) of ``estimator`` on ``X``; returns (proba, seconds).

    The rows are scored in the same fixed blocks as at fit time (:func:`nids.models.train.score_in_blocks`),
    so a model whose arithmetic depends on the batch size (the neural net) reads exactly as it did then. Forests
    score on one thread (:func:`deterministic`), so restoring the same bundle twice gives the very same readings.
    ``after_block`` receives (rows scored so far, rows in all) after each block; it does not change the blocks.
    """
    started = time.perf_counter()
    with deterministic(estimator):
        raw, _ = score_in_blocks(estimator, X, after_block=after_block)
    raw = raw if len(X) else np.empty((0, n_classes))
    return _tidy_proba(raw, getattr(estimator, "classes_", None), n_classes), time.perf_counter() - started


def _rows_reporter(progress: Callable[[str, float], None], label: str, index: int,
                   count: int) -> Callable[[int, int], None]:
    """A :func:`score_in_blocks` ``after_block`` callback for channel ``index`` of ``count``: reports the rows
    scored so far as (message, fraction of the whole restore)."""
    def report(done: int, total: int) -> None:
        if 0 < done < total:
            progress(f"{label} reads the held-out rows: {done:,} of {total:,}", (index + done / total) / count)

    return report


def restore_run(bundle: LoadedBundle, data: TrainingData | None = None, *,
                progress: Callable[[str, float], None] | None = None) -> TrainingRun:
    """Turn a loaded bundle into an ordinary :class:`~nids.models.train.TrainingRun` (``origin="loaded"``).

    With ``data`` (from :func:`rebuild_training_data`) every fitted channel scores the held-out rows again (by
    prediction only: nothing is fitted, so ``FIT_CALLS`` does not move), the readings are checked against the saved
    ones, and the reference sample is drawn from the rebuilt training rows exactly as the fit drew it (in memory
    only). Without ``data`` the run carries empty matrices with the bundle's classes, feature names and feature
    choice (``has_test_rows`` is False), channels have empty predictions, and the reference sample is made of
    quantile vectors. The feature quantiles always come from the bundle; timings, metrics and the reports of the
    fit (row counts, de-duplication, Top-K ranking, how long the matrices took) are those saved (a bundle saved
    before the 01 Sample shares and the heavy-flow account were recorded takes them from the rebuilt sample, so its
    recorded-traffic estimate can be computed as well). CH3 saved by choice
    comes back as an ordinary ``"ok"`` channel like the others; left out (the default), it comes back with status
    ``"not_saved"`` and a note saying why. ``progress`` receives (message, fraction) before each channel scores the
    held-out rows and after every block of rows it scores (the kernel SVM can take as long as it did at fit time).
    """
    manifest = bundle.manifest
    request = _train_request_from(manifest)
    data_request = _data_request_from(manifest)
    n_classes = len(bundle.classes)
    if data is None:
        data = _empty_data(bundle)
        reference = quantile_vectors(bundle.quantiles, BACKGROUND_ROWS, int(request.seed))
    else:
        if tuple(data.classes) != bundle.classes or tuple(data.feature_names) != bundle.feature_names:
            raise ValueError("These training data do not belong to the saved run (classes or columns differ).")
        n_ref = min(REFERENCE_ROWS, len(data.y_train))
        ref_rows, _ = sampling.sample_positions(data.y_train, n_ref, int(request.seed))
        reference = np.array(data.X_train[ref_rows], dtype=np.float32, copy=True)
        saved_reports = manifest.get("reports")
        if isinstance(saved_reports, Mapping) and saved_reports:
            kept = dict(saved_reports)
            # A bundle saved before the 01 Sample shares (or the heavy-flow account) were recorded: the rebuilt
            # sample is the very same one (its fingerprint was checked), so its reports are those of the fit.
            for name in ("sampling", "heavy_flows"):
                if name not in kept and isinstance((data.reports or {}).get(name), Mapping):
                    kept[name] = dict(data.reports[name])
            data = replace(data, reports=kept)
    entries = manifest.get("channels") or {}
    keys = _ordered(entries)
    channels: dict[str, ChannelResult] = {}
    for index, key in enumerate(keys):
        entry = entries[key] or {}
        extra = dict(entry.get("extra") or {})
        extra["metrics"] = _floats(entry.get("metrics") or extra.get("metrics"))
        result = ChannelResult(
            key=key, status=entry.get("status", "failed"), estimator=bundle.channels.get(key),
            fit_seconds=float(entry.get("fit_seconds") or 0.0), rows_used=int(entry.get("rows_used") or 0),
            rows_available=int(entry.get("rows_available") or 0), notes=[str(n) for n in entry.get("notes") or []],
            error=entry.get("error"), predict_seconds=float(entry.get("predict_seconds") or 0.0), extra=extra,
        )
        if result.status == "ok" and result.estimator is None:
            result.status = "failed"
            result.error = "The model of this channel is missing from the saved bundle."
        if result.status == NOT_SAVED and key in UNSAVED_CHANNELS:
            former = _FORMER_UNSAVED_NOTES.get(key, ())
            result.notes = [n for n in result.notes if n not in former]  # an older bundle's wording of the reason
            if UNSAVED_CHANNELS[key] not in result.notes:
                result.notes.append(UNSAVED_CHANNELS[key])
        if result.status == "ok":
            if len(data.y_test):
                tick = None
                if progress is not None:
                    progress(f"{_channel_label(key)} reads the {len(data.y_test):,} held-out rows", index / len(keys))
                    tick = _rows_reporter(progress, _channel_label(key), index, len(keys))
                result.proba, _ = _score_rows(result.estimator, data.X_test, n_classes, after_block=tick)
                result.y_pred = result.proba.argmax(axis=1).astype(np.int64)
                again = quick_metrics(data.y_test, result.y_pred, n_classes)
                saved = extra["metrics"]
                if any(not math.isclose(again[m], saved.get(m, float("nan")), rel_tol=0, abs_tol=1e-9)
                       for m in again):
                    result.notes.append("Readings on the rebuilt held-out rows differ from the saved readings; the "
                                        "rebuilt ones are shown.")
                    result.extra["metrics_saved"] = dict(saved)
                    result.extra["metrics"] = again
            else:
                result.proba = np.empty((0, n_classes), dtype=np.float32)
                result.y_pred = np.empty(0, dtype=np.int64)
        channels[key] = result
    return TrainingRun(
        run_id=str(manifest.get("run_id", bundle.path.name)), created_utc=str(manifest.get("created_utc", "")),
        request=request, data_request=data_request, dataset_fingerprint=str(manifest.get("dataset_fingerprint", "")),
        data=data, channels=channels, seconds=float(manifest.get("seconds") or 0.0), reference_sample=reference,
        feature_quantiles=np.array(bundle.quantiles, dtype=np.float32, copy=True), origin="loaded",
        bundle_path=str(bundle.path),
    )


__all__ = [
    "BUNDLE_FORMAT", "DELETING_PREFIX", "MANIFEST_DIGEST_KEY", "NOT_SAVED", "SAVING_PREFIX", "TRAINING_ROWS_KEY",
    "UNSAVED_CHANNELS", "BundleDeleteError", "BundleIntegrityError", "BundleReadError", "BundleSummary",
    "LeftoverFolder", "LoadedBundle", "RebuildError", "VerificationReport", "content_digests", "delete_bundle",
    "deterministic", "find_leftovers", "library_versions", "list_bundles", "load_bundle", "manifest_digest",
    "probe_vectors", "quantile_vectors", "read_bundle_summary", "rebuild_prepared", "rebuild_training_data",
    "remove_leftover", "restore_run", "save_run", "scan_bundles", "score_exactly", "sha256_file",
    "split_fingerprints", "support_vector_count", "training_rows_inside", "write_manifest",
]
