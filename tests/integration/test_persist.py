"""Saved channel sets: layout and manifest, identical predictions after a real process restart, tamper and version
checks, no dataset rows on disk by default (CH3 is left out), the opt-in that saves CH3 with the training rows it is
made of (declared in the manifest, held by ``svm.joblib`` only), and rebuilding/restoring a run without refitting.

Every fit uses ``profile="test"`` (tiny models) on about 3,000 generated flows, without the destination port.
Bundles are written under pytest's temporary folders only.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from graticule import __version__, evaluate, persist
from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.models import train as train_mod
from graticule.models.train import FIT_CALLS, TrainingData, TrainingRun, TrainRequest, build_training_data, train_all
from graticule.models.zoo import MODEL_KEYS
from graticule.persist import (
    BundleIntegrityError,
    RebuildError,
    delete_bundle,
    list_bundles,
    load_bundle,
    rebuild_training_data,
    restore_run,
    save_run,
    scan_bundles,
    score_exactly,
    write_manifest,
)
from graticule.schema import FEATURES

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SEED = 11
#: The channels a bundle keeps: every one but CH3, whose model is made of training rows.
KEPT = tuple(k for k in MODEL_KEYS if k != "svm")
MANIFEST_KEYS = {
    "bundle_format", "app", "run_id", "created_utc", "saved_utc", "mode", "classes", "feature_names",
    "feature_choice", "train_request", "data_request", "data_source", "seed", "nonfinite_strategy",
    "dataset_fingerprint", "train_fingerprint", "test_fingerprint", "train_values_sha256", "test_values_sha256",
    "rows", "svm_rows_used", "seconds", "prep_seconds", "reports", "channels", "best_channel",
    "best_balanced_accuracy", "library_versions", "files", "probe", "quantiles", "contents", "manifest_sha256",
}
CHANNEL_KEYS = {"status", "fit_status", "badge", "rows_used", "rows_available", "notes", "error", "fit_seconds",
                "predict_seconds", "flows_per_second", "metrics", "extra", "files"}


@pytest.fixture(scope="module")
def prepared() -> PreparedDataset:
    """About 2,500 synthetic flows (six classes)."""
    return prepare_dataset(DataRequest(source="synthetic", synthetic_flows=2_500, seed=SEED))


def _fit(prepared: PreparedDataset, **changes: Any) -> TrainingRun:
    request = TrainRequest(**{"profile": "test", "seed": SEED, **changes})
    data = build_training_data(prepared, request)
    return train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint)


@pytest.fixture(scope="module")
def runs(prepared: PreparedDataset) -> dict[str, TrainingRun]:
    """All five channels fitted once per mode."""
    return {mode: _fit(prepared, mode=mode) for mode in ("binary", "multiclass")}


@pytest.fixture(scope="module")
def bundles(runs: dict[str, TrainingRun], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Each run saved once into a module-wide temporary folder (the default: CH3 left out)."""
    root = tmp_path_factory.mktemp("saved_models")
    return {mode: save_run(run, root) for mode, run in runs.items()}


@pytest.fixture(scope="module")
def svm_bundles(runs: dict[str, TrainingRun], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Each run saved once more, with CH3 by choice, into another module-wide temporary folder."""
    root = tmp_path_factory.mktemp("saved_models_with_ch3")
    return {mode: save_run(replace(run, bundle_path=None), root, include_svm=True) for mode, run in runs.items()}


def _copy(bundle: Path, tmp_path: Path) -> Path:
    """A private copy of a bundle that a test may damage."""
    return Path(shutil.copytree(bundle, tmp_path / bundle.name))


def _manifest(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


def _fresh_inputs(run: TrainingRun) -> np.ndarray:
    """Flows no channel has seen: quantile vectors from another seed, some missing values, and the test rows."""
    vectors = persist.quantile_vectors(run.feature_quantiles, 300, seed=2024)
    vectors[:10, 0] = np.nan
    return np.vstack([vectors, run.data.X_test]).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------
# Layout and manifest
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures("reloaded")  # starts the second process early, so it loads the bundles meanwhile
def test_bundle_layout_and_manifest_record_every_required_field(runs: dict[str, TrainingRun],
                                                               bundles: dict[str, Path]) -> None:
    run, folder = runs["binary"], bundles["binary"]
    assert folder.name == run.run_id and run.bundle_path == str(folder)
    names = sorted(p.name for p in folder.iterdir())
    assert names == sorted(["manifest.json", "forest.joblib", "xgboost.ubj", "xgboost.joblib", "mlp.joblib",
                            "logreg.joblib", "probe.npz", "quantiles.npz"])
    manifest = _manifest(folder)
    assert MANIFEST_KEYS <= set(manifest)
    assert manifest["bundle_format"] == persist.BUNDLE_FORMAT == 2 and manifest["app"]["version"] == __version__
    assert manifest["manifest_sha256"] == persist.manifest_digest(manifest)
    assert manifest["run_id"] == run.run_id and manifest["created_utc"] == run.created_utc
    assert manifest["mode"] == "binary" and manifest["classes"] == ["Normal", "Attack"]
    assert manifest["feature_names"] == list(run.data.feature_names)
    assert manifest["feature_choice"]["mode"] == "curated" and manifest["feature_choice"]["include_port"] is False
    assert manifest["train_request"] == run.request.to_dict()
    assert manifest["data_request"]["source"] == "synthetic" and manifest["data_request"]["seed"] == SEED
    assert manifest["data_source"]["description"] == run.data_request.describe()
    assert manifest["dataset_fingerprint"] == run.dataset_fingerprint
    assert manifest["test_fingerprint"] == persist.split_fingerprints(run.data)["test"]
    assert manifest["test_values_sha256"] == persist.content_digests(run.data)["test"]
    assert manifest["train_values_sha256"] == persist.content_digests(run.data)["train"]
    assert manifest["rows"]["train"] == len(run.data.y_train) and manifest["rows"]["test"] == len(run.data.y_test)
    assert sum(manifest["rows"]["test_by_class"].values()) == len(run.data.y_test)
    assert manifest["rows"]["prepared"] == run.data.reports["rows"]["prepared"]
    assert manifest["nonfinite_strategy"] == "drop" and manifest["seed"] == SEED
    assert manifest["svm_rows_used"] == run.channels["svm"].extra["svm_rows_used"]
    assert manifest["library_versions"] == persist.library_versions()
    assert set(manifest["library_versions"]) == {"python", "numpy", "pandas", "scikit-learn", "scipy", "xgboost",
                                                 "joblib"}
    assert set(manifest["files"]) == set(names) - {"manifest.json"}
    for name, digest in manifest["files"].items():
        assert persist.sha256_file(folder / name) == digest
    assert manifest["probe"]["count"] == persist.PROBE_COUNT and manifest["probe"]["channels"] == list(KEPT)
    assert manifest["quantiles"] == {"file": "quantiles.npz", "levels": 101, "features": len(run.data.feature_names)}
    for key in MODEL_KEYS:
        entry = manifest["channels"][key]
        assert CHANNEL_KEYS <= set(entry), key
        assert entry["fit_status"] == "ok" and entry["metrics"] == pytest.approx(run.channels[key].extra["metrics"])
        assert entry["rows_used"] == run.channels[key].rows_used and entry["fit_seconds"] > 0
        if hasattr(evaluate, "classification_metrics"):  # the richer readings, when the evaluation module has them
            assert {"precision", "recall", "f1", "roc_auc", "average_precision"} <= set(entry["evaluation_metrics"])
    assert all(manifest["channels"][key]["status"] == "ok" for key in KEPT)
    # CH3 was fitted, but its model is made of training rows: recorded (readings, support vectors), not written
    # (the default), and the manifest declares no training rows inside.
    svm = manifest["channels"]["svm"]
    assert svm["status"] == persist.NOT_SAVED and svm["files"] == []
    assert svm["extra"]["support_vectors"] > 0 and persist.UNSAVED_CHANNELS["svm"] in svm["notes"]
    assert manifest["best_channel"] in KEPT and 0 < manifest["best_balanced_accuracy"] <= 1
    assert "no dataset rows" in manifest["contents"] and "CH3" in manifest["contents"]
    assert "only when you choose to" in manifest["contents"]
    assert not manifest.get(persist.TRAINING_ROWS_KEY) and persist.training_rows_inside(manifest) == {}
    assert persist.read_bundle_summary(folder).holds_training_rows == 0


@pytest.mark.usefixtures("reloaded")
def test_saving_ch3_by_choice_writes_it_and_declares_the_training_rows_it_holds(
        runs: dict[str, TrainingRun], svm_bundles: dict[str, Path]) -> None:
    for mode, run in runs.items():
        folder = svm_bundles[mode]
        names = sorted(p.name for p in folder.iterdir())
        assert names == sorted(["manifest.json", "forest.joblib", "xgboost.ubj", "xgboost.joblib", "svm.joblib",
                                "mlp.joblib", "logreg.joblib", "probe.npz", "quantiles.npz"])
        manifest = _manifest(folder)
        assert manifest["bundle_format"] == persist.BUNDLE_FORMAT == 2
        assert set(manifest["files"]) == set(names) - {"manifest.json"}
        assert manifest["files"]["svm.joblib"] == persist.sha256_file(folder / "svm.joblib")
        vectors = run.channels["svm"].extra["support_vectors"]
        assert vectors == persist.support_vector_count(run.channels["svm"].estimator) > 0
        assert manifest[persist.TRAINING_ROWS_KEY] == {"svm": vectors}
        assert persist.training_rows_inside(manifest) == {"svm": vectors}
        svm = manifest["channels"]["svm"]
        assert svm["status"] == "ok" and svm["files"] == ["svm.joblib"]
        assert persist.UNSAVED_CHANNELS["svm"] not in svm["notes"]
        assert manifest["probe"]["channels"] == list(MODEL_KEYS)
        assert f"svm.joblib (CH3 RBF SVM, saved by choice) holds {vectors:,} training rows" in manifest["contents"]
        assert "turns back into the original values" in manifest["contents"]
        assert "No other file holds dataset rows." in manifest["contents"]
        summary = persist.read_bundle_summary(folder)
        assert summary.channels == tuple(MODEL_KEYS) and summary.holds_training_rows == vectors
        assert summary.problem is None
    # Saving again where the run already has a bundle returns that set as it is (here: with CH3 inside).
    run = runs["binary"]
    again = save_run(replace(run, bundle_path=None), svm_bundles["binary"].parent)
    assert again == svm_bundles["binary"] and (again / "svm.joblib").is_file()


def test_an_older_bundle_without_the_training_rows_key_still_loads(runs: dict[str, TrainingRun],
                                                                   bundles: dict[str, Path], tmp_path: Path) -> None:
    """Bundles saved before the opt-in existed have no ``training_rows_inside`` key and an older wording of why CH3
    is left out: they load and verify unchanged, hold no training rows, and the reason is given once, as worded now."""
    folder = _copy(bundles["binary"], tmp_path)
    manifest = _manifest(folder)
    manifest.pop(persist.TRAINING_ROWS_KEY)
    former = persist._FORMER_UNSAVED_NOTES["svm"][0]
    notes = manifest["channels"]["svm"]["notes"]
    notes[notes.index(persist.UNSAVED_CHANNELS["svm"])] = former
    write_manifest(folder, manifest)  # sealed again, as the older version sealed it
    bundle = load_bundle(folder)
    assert persist.TRAINING_ROWS_KEY not in bundle.manifest
    assert bundle.verification.verified, bundle.verification.message
    assert bundle.training_rows_inside == {} and set(bundle.channels) == set(KEPT)
    summary = persist.read_bundle_summary(folder)
    assert summary.holds_training_rows == 0 and summary.problem is None and summary.channels == KEPT
    restored = restore_run(bundle, None)
    svm = restored.channels["svm"]
    assert svm.status == persist.NOT_SAVED and former not in svm.notes
    assert svm.notes.count(persist.UNSAVED_CHANNELS["svm"]) == 1
    assert restored.channels["forest"].notes == runs["binary"].channels["forest"].notes


def test_the_manifest_keeps_the_readings_03_measure_computed(runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    run = replace(runs["multiclass"], run_id=f"{runs['multiclass'].run_id}-measured", bundle_path=None)
    evaluations = evaluate.evaluate_run(run)
    assert list(evaluations) == list(MODEL_KEYS)
    # One channel's evaluation not cached (another session still computing it): computed from the stored readings.
    setattr(run, evaluate.EVALUATIONS_ATTR, {k: v for k, v in evaluations.items() if k != "svm"})
    calls = sum(FIT_CALLS.values())
    manifest = _manifest(save_run(run, tmp_path))
    assert sum(FIT_CALLS.values()) == calls
    for key, evaluation in evaluations.items():
        expected = {name: (value if np.isfinite(value) else None) for name, value in evaluation.metrics.items()}
        assert manifest["channels"][key]["evaluation_metrics"] == expected, key
    assert {"f1_macro", "f1_weighted", "precision_macro", "recall_weighted", "roc_auc",
            "average_precision"} <= set(manifest["channels"]["svm"]["evaluation_metrics"])


def test_saving_twice_returns_the_same_folder_and_leaves_the_models_as_they_were(
        runs: dict[str, TrainingRun], bundles: dict[str, Path]) -> None:
    run = runs["binary"]
    stamp = (bundles["binary"] / "manifest.json").stat().st_mtime_ns
    assert save_run(run, bundles["binary"].parent) == bundles["binary"]
    assert (bundles["binary"] / "manifest.json").stat().st_mtime_ns == stamp
    assert run.channels["forest"].estimator[-1].n_jobs == -1  # the one-thread probe scoring was undone
    assert run.channels["xgboost"].estimator[-1].get_params()["callbacks"] is None
    assert not [p for p in bundles["binary"].parent.iterdir() if p.name.startswith(".")]


# --------------------------------------------------------------------------------------------------------------
# Identity across a process restart
# --------------------------------------------------------------------------------------------------------------
LOADER = """
import json, sys
from pathlib import Path
import numpy as np
from graticule import persist

inputs = np.load(sys.argv[1])
report = {}
for argument in sys.argv[3:]:
    tag, folder = argument.split("=", 1)
    bundle = persist.load_bundle(Path(folder))
    out = {}
    for key, estimator in bundle.channels.items():
        X = inputs[bundle.mode]
        proba, labels = persist.score_exactly(estimator, X)
        with persist.deterministic(estimator):
            predicted = np.asarray(estimator.predict(X))
        out[f"{key}__proba"] = proba
        out[f"{key}__labels"] = labels
        out[f"{key}__predict"] = predicted
    np.savez(Path(sys.argv[2]) / f"{tag}.npz", **out)
    report[tag] = {"mode": bundle.mode, "verified": bundle.verification.verified,
                   "message": bundle.verification.message, "channels": list(bundle.channels),
                   "mismatches": bundle.verification.version_mismatches,
                   "training_rows": bundle.training_rows_inside}
print(json.dumps(report))
"""


@pytest.fixture(scope="module")
def reloaded(runs: dict[str, TrainingRun], bundles: dict[str, Path], svm_bundles: dict[str, Path],
             tmp_path_factory: pytest.TempPathFactory) -> Iterator[Callable[[], tuple[dict[str, Any], Path, dict]]]:
    """A separate Python process that loads all four bundles (each mode saved as by default, tagged with the mode,
    and with CH3 by choice, tagged ``<mode>-svm``) and scores fresh inputs with every channel.

    It starts as soon as the bundles exist and runs while the other tests of this module do; the returned function
    waits for it and gives (its report, the folder holding its outputs, the inputs).
    """
    folder = tmp_path_factory.mktemp("reloaded")
    inputs = {mode: _fresh_inputs(run) for mode, run in runs.items()}
    np.savez(folder / "inputs.npz", **inputs)
    tagged = [f"{mode}={bundles[mode]}" for mode in runs] + [f"{mode}-svm={svm_bundles[mode]}" for mode in runs]
    process = subprocess.Popen(
        [sys.executable, "-c", LOADER, str(folder / "inputs.npz"), str(folder), *tagged],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    def result() -> tuple[dict[str, Any], Path, dict]:
        out, err = process.communicate(timeout=300)
        assert process.returncode == 0, err
        return json.loads(out.strip().splitlines()[-1]), folder, inputs

    yield result
    if process.poll() is None:
        process.kill()
        process.communicate()


# --------------------------------------------------------------------------------------------------------------
# Tampering and version drift
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["forest.joblib", "xgboost.ubj", "probe.npz"])
def test_changing_one_byte_of_a_file_is_refused(bundles: dict[str, Path], tmp_path: Path, name: str) -> None:
    folder = _copy(bundles["binary"], tmp_path)
    target = folder / name
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0x01
    target.write_bytes(bytes(data))
    with pytest.raises(BundleIntegrityError, match=name.replace(".", r"\.")):
        load_bundle(folder)


def test_a_changed_svm_joblib_is_refused(svm_bundles: dict[str, Path], tmp_path: Path) -> None:
    folder = _copy(svm_bundles["multiclass"], tmp_path)
    target = folder / "svm.joblib"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0x01
    target.write_bytes(bytes(data))
    with pytest.raises(BundleIntegrityError, match=r"svm\.joblib in .* does not match the checksum"):
        load_bundle(folder)
    target.unlink()
    with pytest.raises(BundleIntegrityError, match=r"svm\.joblib is missing"):
        load_bundle(folder)


def test_missing_or_unlisted_files_and_foreign_folders_are_refused(bundles: dict[str, Path], tmp_path: Path) -> None:
    folder = _copy(bundles["binary"], tmp_path)
    (folder / "mlp.joblib").unlink()
    with pytest.raises(BundleIntegrityError, match="mlp.joblib is missing"):
        load_bundle(folder)
    other = _copy(bundles["multiclass"], tmp_path / "second")
    manifest = _manifest(other)
    del manifest["files"]["mlp.joblib"]
    write_manifest(other, manifest)  # a consistent manifest that no longer covers a model file
    with pytest.raises(BundleIntegrityError, match="no checksum for mlp.joblib"):
        load_bundle(other)
    (tmp_path / "plain").mkdir()
    with pytest.raises(BundleIntegrityError, match="not a saved channel set"):
        load_bundle(tmp_path / "plain")


@pytest.mark.parametrize("edit", ["swap classes", "reverse features", "raise a metric", "older format"])
def test_an_edited_manifest_is_refused(bundles: dict[str, Path], tmp_path: Path, edit: str) -> None:
    """Every hand edit of the manifest (names, metrics, layout) is caught by the manifest's own checksum, before any
    model file is read: the edited readings never reach the page as "verified"."""
    folder = _copy(bundles["multiclass"], tmp_path)
    manifest = _manifest(folder)
    if edit == "swap classes":
        manifest["classes"][1], manifest["classes"][2] = manifest["classes"][2], manifest["classes"][1]
    elif edit == "reverse features":
        manifest["feature_names"] = manifest["feature_names"][::-1]
    elif edit == "raise a metric":
        manifest["channels"]["forest"]["metrics"]["balanced_accuracy"] = 0.999
        manifest["best_balanced_accuracy"] = 0.999
    else:
        manifest["bundle_format"] = 1
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    match = "bundle format 1" if edit == "older format" else "changed or damaged after saving"
    with pytest.raises(BundleIntegrityError, match=match):
        load_bundle(folder)
    summary = next(s for s in list_bundles(folder.parent) if s.run_id == manifest["run_id"])
    assert summary.problem is not None and ("format 1" in summary.problem or "changed" in summary.problem)


def test_a_manifest_that_does_not_fit_its_models_is_refused(bundles: dict[str, Path], tmp_path: Path) -> None:
    """Even a re-sealed manifest must agree with the checksummed files: class count, mode and feature count."""
    folder = _copy(bundles["multiclass"], tmp_path)
    manifest = _manifest(folder)
    write_manifest(folder, {**manifest, "mode": "binary", "classes": manifest["classes"][:2]})
    with pytest.raises(BundleIntegrityError, match="does not fit its models"):
        load_bundle(folder)
    write_manifest(folder, {**manifest, "feature_names": manifest["feature_names"][:-1]})
    with pytest.raises(BundleIntegrityError, match="feature names for"):
        load_bundle(folder)
    write_manifest(folder, manifest)  # the original content, sealed again: accepted and verified
    assert load_bundle(folder).verification.verified


def test_a_library_version_change_is_reported_and_the_bundle_is_not_verified(
        bundles: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    real = persist.library_versions()
    monkeypatch.setattr(persist, "library_versions", lambda: {**real, "scipy": "0.0.0-test"})
    bundle = load_bundle(bundles["binary"])
    report = bundle.verification
    assert report.hashes_ok and not report.verified
    assert report.version_mismatches == {"scipy": (real["scipy"], "0.0.0-test")}
    assert all(report.probes_identical.values()) and max(report.max_abs_diff.values()) == 0.0
    assert report.message.startswith("Not verified: the libraries differ")
    assert f"scipy {real['scipy']} -> 0.0.0-test" in report.message


def test_a_changed_reading_with_the_same_versions_fails_verification(
        bundles: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    real = persist.score_exactly

    def drifting(estimator: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proba, labels = real(estimator, X)
        if type(estimator[-1]).__name__ == "LogisticRegression":
            proba = proba + 1e-9
        return proba, labels

    monkeypatch.setattr(persist, "score_exactly", drifting)
    report = load_bundle(bundles["binary"]).verification
    assert not report.verified and not report.version_mismatches
    assert report.probes_identical == {"forest": True, "xgboost": True, "mlp": True, "logreg": False}
    assert report.max_abs_diff["logreg"] == pytest.approx(1e-9, rel=0.1)
    assert report.message.startswith("Verification failed: CH5 Logistic regression")


# --------------------------------------------------------------------------------------------------------------
# No dataset rows on disk
# --------------------------------------------------------------------------------------------------------------
def _row_set(X: np.ndarray) -> set[bytes]:
    """Every row of ``X`` as bytes (float32), for membership tests."""
    return {row.tobytes() for row in np.ascontiguousarray(X, dtype=np.float32)}


def _objects_inside(obj: Any, depth: int = 0, seen: set[int] | None = None) -> list[Any]:
    """Every object reachable through the attributes, lists and dicts of a fitted model (bounded depth)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 10:
        return []
    seen.add(id(obj))
    found = [obj]
    if isinstance(obj, np.ndarray):
        return found
    if isinstance(obj, dict):
        items = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        items = list(obj)
    elif hasattr(obj, "__dict__"):
        items = list(vars(obj).values())
    else:
        items = []
    for item in items:
        found.extend(_objects_inside(item, depth + 1, seen))
    return found


def _arrays_inside(obj: Any) -> list[np.ndarray]:
    """Every float array reachable inside ``obj``."""
    return [o for o in _objects_inside(obj) if isinstance(o, np.ndarray) and o.dtype.kind == "f"]


def _recoverable_rows(estimator: Any, X_train: np.ndarray) -> int:
    """How many rows of arrays inside ``estimator`` are training rows, either as they are or as the model saw them.

    For every pipeline with a standardising step inside the estimator, the training rows are put through the steps
    before the model (gap filling, signed logarithm, standardisation): an array row equal (to 1e-6 relative) to one
    of those transformed rows can be turned back into that training row with the parameters stored next to it.
    """
    found = 0
    raw = _row_set(X_train)
    for array in _arrays_inside(estimator):
        if array.ndim == 2 and array.shape[1] == X_train.shape[1]:
            found += len(_row_set(array) & raw)
    for pipeline in [o for o in _objects_inside(estimator) if isinstance(o, Pipeline)]:
        if not any(isinstance(step, StandardScaler) for _, step in pipeline.steps[:-1]):
            continue
        seen_by_model = np.asarray(pipeline[:-1].transform(X_train), dtype=np.float64)
        for array in _arrays_inside(pipeline.steps[-1][1]):
            if array.ndim != 2 or array.shape[1] != seen_by_model.shape[1]:
                continue
            for row in np.asarray(array, dtype=np.float64):
                close = np.abs(seen_by_model - row) <= 1e-6 * np.maximum(np.abs(seen_by_model), 1.0)
                found += int(close.all(axis=1).any())
    return found


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_the_bundle_holds_no_dataset_rows(runs: dict[str, TrainingRun], bundles: dict[str, Path], mode: str) -> None:
    run, folder = runs[mode], bundles[mode]
    rows = _row_set(run.data.X_train) | _row_set(run.data.X_test)
    n_features = len(run.data.feature_names)
    assert not (folder / "svm.joblib").exists()
    with np.load(folder / "probe.npz", allow_pickle=False) as probe:
        assert set(probe.files) == {"vectors", *(f"{p}__{k}" for k in KEPT for p in ("proba", "labels"))}
        vectors = probe["vectors"]
        assert vectors.shape == (persist.PROBE_COUNT, n_features)
        assert not (_row_set(vectors) & rows)
        assert not (_row_set(vectors) & _row_set(run.reference_sample))
    with np.load(folder / "quantiles.npz", allow_pickle=False) as stored:
        assert set(stored.files) == {"quantiles", "levels"}
        assert stored["quantiles"].shape == (101, n_features)
        assert np.array_equal(stored["quantiles"], run.feature_quantiles, equal_nan=True)
    # The detector finds rows where they are: the in-memory CH3 (not written by default) holds its support vectors,
    # which its own pipeline's parameters turn back into training rows.
    in_memory_svm = run.channels["svm"].estimator
    assert _recoverable_rows(in_memory_svm, run.data.X_train) >= run.channels["svm"].extra["support_vectors"] > 0
    # No model in the bundle holds a training row, neither as stored nor after undoing its pipeline's transforms.
    bundle = load_bundle(folder)
    assert set(bundle.channels) == set(KEPT)
    for key, estimator in bundle.channels.items():
        assert _recoverable_rows(estimator, run.data.X_train) == 0, key
        assert _recoverable_rows(estimator, run.data.X_test) == 0, key
    assert bundle.manifest["channels"]["svm"]["status"] == persist.NOT_SAVED


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_with_ch3_by_choice_the_rows_are_in_svm_joblib_only_and_declared(
        runs: dict[str, TrainingRun], svm_bundles: dict[str, Path], mode: str) -> None:
    """The opt-in counterpart: CH3's support vectors (training rows, recoverable through its saved scaler) are in
    ``svm.joblib`` and nowhere else, no held-out row is anywhere, and the manifest declares how many there are."""
    run, folder = runs[mode], svm_bundles[mode]
    rows = _row_set(run.data.X_train) | _row_set(run.data.X_test)
    declared = _manifest(folder)[persist.TRAINING_ROWS_KEY]
    vectors = run.channels["svm"].extra["support_vectors"]
    assert declared == {"svm": vectors} and vectors > 0
    with np.load(folder / "probe.npz", allow_pickle=False) as probe:
        assert set(probe.files) == {"vectors", *(f"{p}__{k}" for k in MODEL_KEYS for p in ("proba", "labels"))}
        assert not (_row_set(probe["vectors"]) & rows)
    with np.load(folder / "quantiles.npz", allow_pickle=False) as stored:
        assert set(stored.files) == {"quantiles", "levels"}
    bundle = load_bundle(folder)
    assert set(bundle.channels) == set(MODEL_KEYS) and bundle.training_rows_inside == {"svm": vectors}
    for key, estimator in bundle.channels.items():
        if key == "svm":
            assert persist.support_vector_count(estimator) == vectors
            assert _recoverable_rows(estimator, run.data.X_train) >= vectors, key
        else:
            assert _recoverable_rows(estimator, run.data.X_train) == 0, key
        assert _recoverable_rows(estimator, run.data.X_test) == 0, key


# --------------------------------------------------------------------------------------------------------------
# Rebuilding the held-out rows and restoring the run
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_rebuilding_a_synthetic_run_gives_the_exact_rows_and_labels(runs: dict[str, TrainingRun],
                                                                   bundles: dict[str, Path], mode: str) -> None:
    original = runs[mode].data
    data = rebuild_training_data(load_bundle(bundles[mode]), data_dir=None)
    assert np.array_equal(data.X_test, original.X_test, equal_nan=True)
    assert np.array_equal(data.X_train, original.X_train, equal_nan=True)
    assert np.array_equal(data.y_test, original.y_test) and np.array_equal(data.y_train, original.y_train)
    assert np.array_equal(data.test_rows, original.test_rows)
    assert np.array_equal(data.detailed_test_labels, original.detailed_test_labels)
    assert data.classes == original.classes and data.feature_names == original.feature_names


def test_restoring_with_rebuilt_rows_reproduces_the_run_without_fitting(runs: dict[str, TrainingRun],
                                                                       bundles: dict[str, Path]) -> None:
    original = runs["multiclass"]
    bundle = load_bundle(bundles["multiclass"])
    calls = sum(FIT_CALLS.values())
    steps: list[tuple[str, float]] = []
    restored = restore_run(bundle, rebuild_training_data(bundle, data_dir=None),
                           progress=lambda message, fraction: steps.append((message, fraction)))
    assert sum(FIT_CALLS.values()) == calls
    assert isinstance(restored, TrainingRun)
    assert [m for m, _ in steps] == [f"{evaluate.channel_label(k)} reads the {len(original.data.y_test):,} held-out "
                                     "rows" for k in KEPT]
    assert restored.origin == "loaded" and restored.bundle_path == str(bundles["multiclass"])
    assert restored.has_test_rows and original.has_test_rows and original.origin == "fitted"
    assert restored.run_id == original.run_id and restored.created_utc == original.created_utc
    assert restored.request == original.request and restored.data_request == original.data_request
    assert restored.dataset_fingerprint == original.dataset_fingerprint
    assert original.ok_channels() == list(MODEL_KEYS) and restored.ok_channels() == list(KEPT)
    svm = restored.channels["svm"]
    assert svm.status == persist.NOT_SAVED and svm.estimator is None and svm.proba is None
    assert persist.UNSAVED_CHANNELS["svm"] in svm.notes and not restored.cancelled
    assert np.array_equal(restored.reference_sample, original.reference_sample, equal_nan=True)
    assert np.array_equal(restored.feature_quantiles, original.feature_quantiles, equal_nan=True)
    assert restored.data.feature_choice == original.data.feature_choice
    # The reports of the fit (row counts, timings) are the saved ones, not the rebuild's.
    assert restored.data.reports["rows"] == original.data.reports["rows"]
    assert restored.prep_seconds == pytest.approx(original.prep_seconds)
    for key in KEPT:
        before, after = original.channels[key], restored.channels[key]
        assert np.array_equal(after.y_pred, before.y_pred), key
        assert after.proba is not None and after.proba.dtype == np.float32
        if key == "forest":  # threads may add tree votes in another order: equal to the last bit or two
            np.testing.assert_allclose(after.proba, before.proba, rtol=0, atol=1e-6)
        else:
            assert np.array_equal(after.proba, before.proba), key
        assert after.extra["metrics"] == pytest.approx(before.extra["metrics"])
        assert after.rows_used == before.rows_used and after.notes == before.notes


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_restoring_a_set_with_ch3_brings_it_back_as_an_ordinary_channel(runs: dict[str, TrainingRun],
                                                                        svm_bundles: dict[str, Path],
                                                                        mode: str) -> None:
    original = runs[mode]
    bundle = load_bundle(svm_bundles[mode])
    assert bundle.verification.verified and bundle.verification.probes_identical["svm"]
    calls = sum(FIT_CALLS.values())
    restored = restore_run(bundle, rebuild_training_data(bundle, data_dir=None))
    assert sum(FIT_CALLS.values()) == calls
    assert restored.ok_channels() == original.ok_channels() == list(MODEL_KEYS)
    before, after = original.channels["svm"], restored.channels["svm"]
    assert after.status == "ok" and after.estimator is bundle.channels["svm"]
    assert np.array_equal(after.y_pred, before.y_pred) and np.array_equal(after.proba, before.proba)
    assert after.extra["metrics"] == pytest.approx(before.extra["metrics"]) and "metrics_saved" not in after.extra
    assert after.rows_used == before.rows_used and after.notes == before.notes
    assert persist.UNSAVED_CHANNELS["svm"] not in after.notes
    for key in KEPT:
        assert np.array_equal(restored.channels[key].y_pred, original.channels[key].y_pred), key
    # Without the rows CH3 still scores fresh flows exactly as the fitted model does.
    bare = restore_run(bundle, None)
    assert bare.channels["svm"].ok and bare.channels["svm"].proba.shape == (0, len(original.data.classes))
    fresh = _fresh_inputs(original)
    loaded, fitted = score_exactly(bare.channels["svm"].estimator, fresh), score_exactly(before.estimator, fresh)
    assert np.array_equal(loaded[0], fitted[0]) and np.array_equal(loaded[1], fitted[1])


def test_restore_scores_in_the_same_blocks_as_the_fit(runs: dict[str, TrainingRun],
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """The fit and a restore call each model on the same row ranges, so batch-size rounding cannot differ."""

    class Recorder:
        def __init__(self, inner: Any) -> None:
            self.inner, self.sizes = inner, []
            self.classes_ = inner.classes_

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            self.sizes.append(len(X))
            return self.inner.predict_proba(X)

    monkeypatch.setattr(train_mod, "SCORE_BLOCK", 70)
    run = runs["multiclass"]
    X = run.data.X_test
    assert len(X) > 140
    at_fit, at_restore = Recorder(run.channels["mlp"].estimator), Recorder(run.channels["mlp"].estimator)
    fitted_proba, _ = train_mod._score_in_batches("mlp", at_fit, X, None, None)
    restored_proba, _ = persist._score_rows(at_restore, X, len(run.data.classes))
    assert at_fit.sizes == at_restore.sizes and set(at_fit.sizes[:-1]) == {70}
    assert np.array_equal(train_mod._tidy_proba(fitted_proba, at_fit.classes_, len(run.data.classes)),
                          restored_proba)


def test_restoring_reports_every_block_of_rows_a_channel_scores(svm_bundles: dict[str, Path],
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded CH3 can take as long to read the held-out rows as it did at fit time: the progress moves between
    blocks, never backwards, and only within that channel's share."""
    bundle = load_bundle(svm_bundles["binary"])
    data = rebuild_training_data(bundle, data_dir=None)
    n, block = len(data.y_test), 100
    assert n > 2 * block
    monkeypatch.setattr(train_mod, "SCORE_BLOCK", block)
    seen: list[tuple[str, float]] = []
    restored = restore_run(bundle, data, progress=lambda message, fraction: seen.append((message, fraction)))
    keys = list(bundle.manifest["channels"])
    assert restored.channels["svm"].ok and keys.index("svm") == 2
    fractions = [fraction for _, fraction in seen]
    assert fractions == sorted(fractions) and 0 <= fractions[0] and fractions[-1] < 1
    ch3 = [(m, f) for m, f in seen if m.startswith("CH3 RBF SVM reads the held-out rows: ")]
    assert len(ch3) == math.ceil(n / block) - 1
    assert ch3[0][0] == f"CH3 RBF SVM reads the held-out rows: {block:,} of {n:,}"
    assert all(2 / len(keys) < f < 3 / len(keys) for _, f in ch3)


def test_restoring_without_data_keeps_the_channels_usable(runs: dict[str, TrainingRun],
                                                         bundles: dict[str, Path]) -> None:
    original = runs["binary"]
    bundle = load_bundle(bundles["binary"])
    calls = sum(FIT_CALLS.values())
    restored = restore_run(bundle, None)
    assert sum(FIT_CALLS.values()) == calls
    n_features = len(original.data.feature_names)
    data: TrainingData = restored.data
    assert not restored.has_test_rows and restored.origin == "loaded"
    assert data.X_train.shape == data.X_test.shape == (0, n_features) and data.X_test.dtype == np.float32
    assert data.y_test.dtype == np.int64 and len(data.detailed_test_labels) == 0
    assert data.classes == original.classes and data.feature_names == original.data.feature_names
    assert data.feature_choice == original.data.feature_choice
    assert data.reports["rows"] == original.data.reports["rows"]
    assert restored.reference_sample.shape == (persist.BACKGROUND_ROWS, n_features)
    assert np.isfinite(restored.reference_sample).all()
    assert not (_row_set(restored.reference_sample) & _row_set(original.data.X_train))
    assert restored.channels["svm"].status == persist.NOT_SAVED
    fresh = _fresh_inputs(original)
    for key in KEPT:
        result = restored.channels[key]
        assert result.ok and result.proba is not None and result.proba.shape == (0, 2) and len(result.y_pred) == 0
        assert result.extra["metrics"] == pytest.approx(original.channels[key].extra["metrics"])
        after, before = score_exactly(result.estimator, fresh), score_exactly(original.channels[key].estimator, fresh)
        assert np.array_equal(after[0], before[0]) and np.array_equal(after[1], before[1]), key
    with pytest.raises(ValueError, match="no held-out rows"):
        save_run(restored, bundles["binary"].parent / "elsewhere")


def test_rebuilding_refuses_a_different_split_or_missing_data(bundles: dict[str, Path], tmp_path: Path) -> None:
    bundle = load_bundle(bundles["binary"])
    changed = replace(bundle, manifest={**bundle.manifest, "test_fingerprint": "0" * 64})
    with pytest.raises(RebuildError, match="rebuilt test rows differ"):
        rebuild_training_data(changed, data_dir=None)
    other_sample = replace(bundle, manifest={**bundle.manifest, "dataset_fingerprint": "1" * 64})
    with pytest.raises(RebuildError, match="differs from the one the run was fitted on"):
        rebuild_training_data(other_sample, data_dir=None)
    real_request = {**bundle.manifest["data_request"], "source": "cicids", "data_dir": None,
                    "files": ["Monday-WorkingHours.pcap_ISCX.csv"]}
    real = replace(bundle, manifest={**bundle.manifest, "data_request": real_request})
    with pytest.raises(RebuildError, match="no data folder is set"):
        rebuild_training_data(real, data_dir=None)
    with pytest.raises(RebuildError, match="does not exist"):
        rebuild_training_data(real, data_dir=str(tmp_path / "nowhere"))
    with pytest.raises(RebuildError, match="lacks Monday-WorkingHours.pcap_ISCX.csv"):
        rebuild_training_data(real, data_dir=str(tmp_path))


def test_rebuilding_refuses_the_same_rows_with_other_feature_values(prepared: PreparedDataset,
                                                                    bundles: dict[str, Path]) -> None:
    """Same rows, labels and sample fingerprint, but every feature value doubled (a changed file, or a change in
    how files are read or cleaned): the value digests catch it, so "identical held-out rows" is never claimed."""
    bundle = load_bundle(bundles["binary"])
    frame = prepared.frame.copy()
    for name in FEATURES:
        frame[name] = (frame[name] * np.float32(2.0)).astype(np.float32)
    altered = replace(prepared, frame=frame)
    assert altered.fingerprint == bundle.manifest["dataset_fingerprint"]
    with pytest.raises(RebuildError, match="feature values differ"):
        rebuild_training_data(bundle, data_dir=None, prepared=altered)
    assert rebuild_training_data(bundle, data_dir=None, prepared=prepared).X_test.shape[0] > 0


def test_a_top_k_run_survives_the_round_trip_without_ranking_again(prepared: PreparedDataset, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    run = _fit(prepared, feature_mode="topk", top_k=8, channels=("forest", "logreg"))
    folder = save_run(run, tmp_path)
    bundle = load_bundle(folder)
    assert bundle.verification.verified, bundle.verification.message
    assert bundle.channels.keys() == {"forest", "logreg"}

    def no_ranking(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a saved Top-K run must reuse its recorded ranking")

    monkeypatch.setattr(train_mod, "rank_features", no_ranking)
    data = rebuild_training_data(bundle, data_dir=None)
    assert data.reports["topk_overlap"]["ranking_reused"] is True
    restored = restore_run(bundle, data)
    assert restored.data.feature_choice.columns == run.data.feature_choice.columns
    assert restored.data.feature_choice.k == 8 and restored.data.feature_choice.ranking is not None
    assert [n for n, _ in restored.data.feature_choice.ranking] == [n for n, _ in run.data.feature_choice.ranking]
    assert np.array_equal(restored.channels["logreg"].y_pred, run.channels["logreg"].y_pred)
    # The fit's own report (with how long the ranking took) is what the restored run carries.
    assert restored.data.reports["topk_overlap"]["ranking_seconds"] == run.data.reports["topk_overlap"][
        "ranking_seconds"] > 0
    assert set(bundle.manifest["channels"]) == {"forest", "logreg"}
    assert bundle.manifest["feature_choice"]["mode"] == "topk"


# --------------------------------------------------------------------------------------------------------------
# Listing and deleting
# --------------------------------------------------------------------------------------------------------------
def test_listing_and_deleting_bundles(runs: dict[str, TrainingRun], bundles: dict[str, Path], tmp_path: Path) -> None:
    root = tmp_path / "models"
    for folder in bundles.values():
        _copy(folder, root)
    (root / ".saving-unfinished").mkdir()
    (root / "notes").mkdir()
    (unfinished,) = persist.find_leftovers(root)  # not a bundle, but named as what it is
    assert unfinished.path.name == ".saving-unfinished" and unfinished.kind == "save"
    assert unfinished.files == () and not unfinished.holds_training_rows
    listed = list_bundles(root)
    assert {s.run_id for s in listed} == {run.run_id for run in runs.values()}
    assert [s.created_utc for s in listed] == sorted((s.created_utc for s in listed), reverse=True)
    summary = next(s for s in listed if s.run_id == runs["multiclass"].run_id)
    assert summary.mode == "multiclass" and summary.source == "synthetic flows"
    assert summary.channels == KEPT and summary.problem is None
    assert summary.rows == len(runs["multiclass"].data.y_train) + len(runs["multiclass"].data.y_test)
    assert summary.best_channel in KEPT and summary.best_balanced_accuracy is not None
    delete_bundle(summary.path)
    assert not summary.path.exists()
    assert [s.run_id for s in list_bundles(root)] == [runs["binary"].run_id]
    with pytest.raises(BundleIntegrityError, match="not a saved channel set"):
        delete_bundle(root / "notes")
    assert (root / "notes").is_dir()
    with pytest.raises(FileNotFoundError):
        delete_bundle(root / "gone")
    assert list_bundles(tmp_path / "absent") == []


def test_malformed_manifests_are_reported_never_raised(bundles: dict[str, Path], tmp_path: Path) -> None:
    root = tmp_path / "models"
    good = _copy(bundles["binary"], root)
    for name, text in {"rows-text": '{"run_id": "x-rows", "rows": "x"}',
                       "channels-text": '{"run_id": "x-ch", "channels": {"forest": "ok"}}',
                       "not-json": "{ not json", "a-list": "[1, 2]"}.items():
        (root / name).mkdir()
        (root / name / "manifest.json").write_text(text, encoding="utf-8")
    found, unreadable = scan_bundles(root)
    assert [s.run_id for s in found] == [good.name]
    assert sorted(folder.name for folder, _ in unreadable) == ["a-list", "channels-text", "not-json", "rows-text"]
    assert all(reason for _, reason in unreadable)
    assert [s.run_id for s in list_bundles(root)] == [good.name]


def test_saving_needs_a_channel_the_bundle_keeps(runs: dict[str, TrainingRun], tmp_path: Path) -> None:
    run = runs["binary"]
    failed = {k: replace(r, status="failed", estimator=None) for k, r in run.channels.items()}
    with pytest.raises(ValueError, match="no channel of this run was fitted"):
        save_run(replace(run, channels=failed, bundle_path=None), tmp_path)
    only_svm = {k: (r if k == "svm" else replace(r, status="failed", estimator=None)) for k, r in run.channels.items()}
    with pytest.raises(ValueError, match="the only fitted channel is CH3 RBF SVM, which saved sets leave out unless "
                                         "it is saved by choice"):
        save_run(replace(run, channels=only_svm, bundle_path=None), tmp_path)
    assert not list(tmp_path.iterdir())
    # With the opt-in, a run whose only fitted channel is CH3 is saved, and loads back verified.
    folder = save_run(replace(run, channels=only_svm, bundle_path=None), tmp_path, include_svm=True)
    assert sorted(p.name for p in folder.iterdir()) == ["manifest.json", "probe.npz", "quantiles.npz", "svm.joblib"]
    manifest = _manifest(folder)
    assert manifest["best_channel"] == "svm" and manifest["probe"]["channels"] == ["svm"]
    assert manifest[persist.TRAINING_ROWS_KEY] == {"svm": run.channels["svm"].extra["support_vectors"]}
    bundle = load_bundle(folder)
    assert bundle.verification.verified, bundle.verification.message
    assert bundle.verification.message == "Verified: the only channel reproduced its saved probe readings exactly."
    assert list(bundle.channels) == ["svm"]
    # Saved already: saving again returns that set as it is, whatever include_svm says.
    assert save_run(replace(run, channels=only_svm, bundle_path=None), tmp_path) == folder


# --------------------------------------------------------------------------------------------------------------
# Saves and deletes that stop halfway (a file held by another program)
# --------------------------------------------------------------------------------------------------------------
def test_a_delete_that_stops_halfway_puts_the_set_back_or_names_what_is_left(
        runs: dict[str, TrainingRun], svm_bundles: dict[str, Path], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "models"
    folder = _copy(svm_bundles["binary"], root)
    names = sorted(p.name for p in folder.iterdir())
    vectors = runs["binary"].channels["svm"].extra["support_vectors"]
    real_remove, real_replace = persist._remove_file, persist.replace_with_retry

    def held(path: Path, retries: int = 3) -> None:
        if Path(path).name == "svm.joblib":
            raise PermissionError(32, "The file is held by another program", str(path))
        real_remove(path, retries)

    monkeypatch.setattr(persist, "_remove_file", held)
    # CH3's rows go first, so nothing else was removed yet: the set is put back whole and listed again.
    with pytest.raises(persist.BundleDeleteError, match="put back under its own name"):
        delete_bundle(folder)
    assert sorted(p.name for p in folder.iterdir()) == names and persist.find_leftovers(root) == []
    (summary,) = list_bundles(root)
    assert summary.path == folder and summary.holds_training_rows == vectors

    # When it cannot even be put back, the hidden folder is named and listed as a leftover holding CH3's rows.
    def no_way_back(source: Path, target: Path, retries: int = 3) -> None:
        if Path(source).name.startswith(persist.DELETING_PREFIX):
            raise PermissionError(5, "Access is denied", str(source))
        real_replace(source, target, retries)

    monkeypatch.setattr(persist, "replace_with_retry", no_way_back)
    with pytest.raises(persist.BundleDeleteError, match="is left over, holding .*svm.joblib") as caught:
        delete_bundle(folder)
    assert "svm.joblib among them still holds CH3's training rows" in str(caught.value)
    assert not folder.exists() and list_bundles(root) == []
    (leftover,) = persist.find_leftovers(root)
    assert leftover.kind == "delete" and leftover.holds_training_rows and list(leftover.files) == names
    assert leftover.path.name.startswith(f"{persist.DELETING_PREFIX}{folder.name}-")
    assert persist.find_leftovers(root, min_age_seconds=3_600) == []  # a folder in use now is not offered
    persist.remove_leftover(leftover.path)
    assert persist.find_leftovers(root) == [] and not list(root.iterdir())
    persist.remove_leftover(leftover.path)  # gone already: nothing to do
    (root / "notes").mkdir()
    with pytest.raises(ValueError, match="not the work folder"):
        persist.remove_leftover(root / "notes")
    assert (root / "notes").is_dir()


def test_a_save_that_fails_and_cannot_clear_up_names_what_is_left(runs: dict[str, TrainingRun], tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    run = replace(runs["binary"], bundle_path=None)
    real_remove_tree = persist._remove_tree

    def refused(source: Path, target: Path, retries: int = 3) -> None:
        raise PermissionError(5, "Access is denied", str(source))

    def stuck(folder: Path, retries: int = 3) -> None:
        raise PermissionError(32, "The file is held by another program", str(folder))

    monkeypatch.setattr(persist, "replace_with_retry", refused)
    monkeypatch.setattr(persist, "_remove_tree", stuck)
    with pytest.raises(OSError, match="is left over, holding") as caught:
        save_run(run, tmp_path, include_svm=True)
    assert "svm.joblib among them still holds CH3's training rows" in str(caught.value)
    assert isinstance(caught.value.__cause__, PermissionError) and run.bundle_path is None
    monkeypatch.setattr(persist, "_remove_tree", real_remove_tree)
    (leftover,) = persist.find_leftovers(tmp_path)
    assert leftover.kind == "save" and leftover.holds_training_rows and "svm.joblib" in leftover.files
    assert leftover.path.name.startswith(f"{persist.SAVING_PREFIX}{run.run_id}-") and list_bundles(tmp_path) == []
    persist.remove_leftover(leftover.path)
    assert not list(tmp_path.iterdir())
    # When the clean-up works, a failed save leaves nothing behind and raises its own error.
    with pytest.raises(PermissionError, match="Access is denied"):
        save_run(run, tmp_path, include_svm=True)
    assert not list(tmp_path.iterdir())


def test_quantile_vectors_follow_the_quantiles() -> None:
    quantiles = np.stack([np.linspace(0, 10, 101), np.full(101, 3.0), np.full(101, np.nan)], axis=1)
    quantiles = quantiles.astype(np.float32)
    vectors = persist.quantile_vectors(quantiles, 400, seed=5)
    assert vectors.shape == (400, 3) and vectors.dtype == np.float32
    assert (vectors[:, 0] >= 0).all() and (vectors[:, 0] <= 10).all() and np.unique(vectors[:, 0]).size > 300
    assert (vectors[:, 1] == 3.0).all() and np.isnan(vectors[:, 2]).all()
    assert np.array_equal(vectors, persist.quantile_vectors(quantiles, 400, seed=5), equal_nan=True)
    probes = persist.probe_vectors(quantiles, seed=5)
    assert probes.shape == (persist.PROBE_COUNT, 3)
    # Each of the first 32 probes has one missing value, in column i % 3 (column 2 is missing throughout).
    assert np.isnan(probes[: persist.PROBE_MISSING_ROWS, :2]).sum() == sum(i % 3 < 2 for i in range(32))
    assert not np.isnan(probes[persist.PROBE_MISSING_ROWS :, :2]).any()


# --------------------------------------------------------------------------------------------------------------
# Identity across a process restart (the second process has been loading the bundles while the tests above ran)
# --------------------------------------------------------------------------------------------------------------
def test_a_reloaded_bundle_predicts_identically_in_a_separate_process(
        runs: dict[str, TrainingRun], reloaded: Callable[[], tuple[dict[str, Any], Path, dict]]) -> None:
    report, tmp_path, inputs = reloaded()
    for mode, run in runs.items():
        # As saved by default (four channels), and with CH3 by choice (all five, CH3 verified like the others).
        for tag, channels in ((mode, list(KEPT)), (f"{mode}-svm", list(MODEL_KEYS))):
            assert report[tag]["mode"] == mode and report[tag]["verified"], report[tag]
            assert report[tag]["message"] == (f"Verified: all {len(channels)} channels reproduced their saved probe "
                                              "readings exactly.")
            assert report[tag]["channels"] == channels and report[tag]["mismatches"] == {}
            expected_rows = {"svm": run.channels["svm"].extra["support_vectors"]} if "svm" in channels else {}
            assert report[tag]["training_rows"] == expected_rows
            with np.load(tmp_path / f"{tag}.npz") as reloaded:
                for key in channels:
                    estimator = run.channels[key].estimator
                    proba, labels = score_exactly(estimator, inputs[mode])
                    with persist.deterministic(estimator):
                        predicted = np.asarray(estimator.predict(inputs[mode]))
                    assert np.array_equal(reloaded[f"{key}__proba"], proba), (tag, key)
                    assert np.array_equal(reloaded[f"{key}__labels"], labels), (tag, key)
                    assert np.array_equal(reloaded[f"{key}__predict"], predicted), (tag, key)


# --------------------------------------------------------------------------------------------------------------
# Real data (skipped when the CIC-IDS2017 folder is not available)
# --------------------------------------------------------------------------------------------------------------
THURSDAY_WEB = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"


@pytest.mark.realdata
@pytest.mark.slow  # reads a CIC-IDS2017 file twice (a few seconds); run with -m realdata
def test_a_real_data_run_rebuilds_its_held_out_rows_from_the_data_folder(real_data_dir: Path, tmp_path: Path) -> None:
    if not (real_data_dir / THURSDAY_WEB).is_file():
        pytest.skip(f"{THURSDAY_WEB} is not in the data folder")
    prepared = prepare_dataset(DataRequest(source="cicids", data_dir=str(real_data_dir), files=(THURSDAY_WEB,),
                                           row_budget=6_000, seed=42))
    run = _fit(prepared, mode="multiclass", min_class_count=10)
    folder = save_run(run, tmp_path / "models")
    assert not (folder / "svm.joblib").exists()  # real rows never reach the disk, not even as support vectors
    bundle = load_bundle(folder)
    assert bundle.verification.verified, bundle.verification.message
    assert bundle.manifest["data_source"]["files"] == [THURSDAY_WEB]
    with np.load(folder / "probe.npz", allow_pickle=False) as probe:
        assert not (_row_set(probe["vectors"]) & (_row_set(run.data.X_train) | _row_set(run.data.X_test)))
    for key, estimator in bundle.channels.items():
        assert _recoverable_rows(estimator, run.data.X_train) == 0, key
    calls = sum(FIT_CALLS.values())
    data = rebuild_training_data(bundle, data_dir=str(real_data_dir))
    assert np.array_equal(data.test_rows, run.data.test_rows) and np.array_equal(data.y_test, run.data.y_test)
    assert np.array_equal(data.X_test, run.data.X_test, equal_nan=True)
    assert np.array_equal(data.detailed_test_labels, run.data.detailed_test_labels)
    restored = restore_run(bundle, data)
    assert sum(FIT_CALLS.values()) == calls
    for key in KEPT:
        assert np.array_equal(restored.channels[key].y_pred, run.channels[key].y_pred), key
    # With no folder given, the folder recorded in the bundle is used; another folder without the file is refused.
    assert np.array_equal(rebuild_training_data(bundle, data_dir=None).test_rows, run.data.test_rows)
    with pytest.raises(RebuildError, match="lacks"):
        rebuild_training_data(bundle, data_dir=str(tmp_path))
