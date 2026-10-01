"""Headless checks of the Logbook: save the current fit (CH3 stays out), list it, load it back verified, delete it,
the confirmation before a load replaces an unsaved fit, robustness against damaged folders, and the history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import graticule.settings as settings_mod
from graticule import persist
from graticule.data.prepare import DataRequest, PreparedDataset, prepare_dataset
from graticule.history import COLUMNS, RunHistory, run_summary
from graticule.models import train
from graticule.models.jobs import get_job
from graticule.models.train import TrainingRun, TrainRequest, build_training_data, train_all
from graticule.models.zoo import MODEL_KEYS
from tests.ui.harness import draw_synthetic_sample, errors, fresh_caches, goto, new_app  # noqa: F401
from ui import state
from ui.pages import logbook

pytestmark = pytest.mark.ui
#: CH3 is fitted but never saved (its model is made of training rows), so a full fit loads back with four.
KEPT = [k for k in MODEL_KEYS if k != "svm"]
VERIFIED = "Verified: all 4 channels reproduced their saved probe readings exactly."
QUICK = ["forest", "logreg"]


def _text(at: AppTest) -> str:
    """Every markdown, caption and alert text of the last run."""
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [a.value for kind in ("success", "info", "warning", "error") for a in getattr(at, kind)]
    return " ".join(parts)


def _table(at: AppTest, column: str) -> pd.DataFrame:
    frames = [d.value for d in at.dataframe if column in d.value.columns]
    assert frames, f"no table with a {column!r} column"
    return frames[0]


def _fit_calls() -> int:
    return sum(train.FIT_CALLS.values())


def _fitted_app() -> tuple[AppTest, str]:
    """A session that drew a small synthetic sample at 01 Sample and fitted every channel at 02 Fit, now at the
    Logbook (the whole way through the stations' forms)."""
    at = new_app().run()
    draw_synthetic_sample(at)
    goto(at, "fit")
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    run_id = at.session_state[state.LAST_RUN_ID]
    goto(at, "logbook")
    assert not errors(at), errors(at)
    return at, run_id


@pytest.fixture(scope="module")
def quick_fit() -> tuple[PreparedDataset, TrainingRun]:
    """A small synthetic sample and a fit of the two quick channels, made once for the tests that need any fit."""
    prepared = prepare_dataset(DataRequest(source="synthetic", synthetic_flows=2_000, row_budget=1_600, seed=42))
    request = TrainRequest(profile="test", channels=tuple(QUICK))
    run = train_all(build_training_data(prepared, request), request, data_request=prepared.request,
                    dataset_fingerprint=prepared.fingerprint)
    return prepared, run


def _quick_app(quick_fit: tuple[PreparedDataset, TrainingRun]) -> tuple[AppTest, str]:
    """A Logbook session whose 01 Sample and current run are those of ``quick_fit``, as right after a fit at 02 Fit
    (the run is a fresh copy, recorded in the history, not saved)."""
    prepared, fitted = quick_fit
    run = replace(fitted, bundle_path=None)
    at = new_app("logbook")
    at.session_state[state.PREPARED] = prepared
    at.session_state[state.RUN] = run
    at.session_state[state.LAST_RUN_ID] = run.run_id
    state.run_registry().put(run.run_id, run)
    RunHistory().record(run)
    at.run()
    assert not errors(at), errors(at)
    return at, run.run_id


def test_without_a_run_the_logbook_points_to_02_fit(fresh_caches: None) -> None:
    at = new_app("logbook").run()
    assert not errors(at), errors(at)
    text = _text(at)
    assert "No fitted channels in this session yet." in text
    assert "No saved channel sets yet." in text and "No fits recorded yet." in text
    assert "lb_save" not in [b.key for b in at.button]


def test_save_list_load_delete_and_clear(fresh_caches: None) -> None:
    at, run_id = _fitted_app()

    # Every finished fit is in the history (recorded once, when the run was adopted).
    history = _table(at, "Train rows")
    assert history["Run"].tolist() == [run_id]
    assert history["Channels"].tolist() == ["CH1 CH2 CH3 CH4 CH5"]
    assert "fitted in this session" in _text(at)
    # Saving says plainly what is written, and that CH3 stays out.
    assert "No dataset rows are written, so CH3" in _text(at) and "support vectors" in _text(at)

    # Save the current fit: everything but CH3.
    at.button(key="lb_save").click().run()
    assert not errors(at), errors(at)
    folder = settings_mod.MODELS_DIR / run_id
    assert (folder / "manifest.json").is_file() and not (folder / "svm.joblib").exists()
    saved_message = " ".join(s.value for s in at.success)
    assert saved_message.startswith(f"Saved run {run_id} to {folder}.")
    assert "CH3 RBF SVM is not kept in saved sets" in saved_message
    assert "lb_save" not in [b.key for b in at.button]
    assert str(folder) in _text(at)
    assert _table(at, "Train rows")["Saved to"].tolist() == [str(folder)]
    saved = _table(at, "Folder")
    assert saved["Run"].tolist() == [run_id] and saved["Channels"].tolist() == ["CH1 CH2 CH4 CH5"]
    assert saved["Check"].tolist() == ["manifest intact"]

    # Load it back: verified, held-out rows rebuilt from the session's own sample, nothing fitted, marked loaded.
    before = _fit_calls()
    assert at.selectbox(key="lb_pick").value == run_id
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    assert _fit_calls() == before
    message = " ".join(s.value for s in at.success)
    assert f"Loaded run {run_id} from disk." in message and VERIFIED in message
    assert "Held-out rows rebuilt" in message and "same rows, labels and feature values" in message
    assert "01 Sample already held this sample" in message and "CH3 RBF SVM is not kept" in message
    run = at.session_state[state.RUN]
    assert run.origin == "loaded" and run.bundle_path == str(folder) and run.has_test_rows
    assert run.ok_channels() == KEPT and run.channels["svm"].status == persist.NOT_SAVED
    assert at.session_state[state.LAST_RUN_ID] == run_id
    assert "loaded from disk" in _text(at) and "verified" in _text(at)
    assert at.session_state[state.PREPARED].fingerprint == run.dataset_fingerprint
    assert len(_table(at, "Train rows")) == 1  # a load is not a new fit

    # The other stations read the loaded run like any other (02 Fit shows its readings without refitting; CH3 is
    # listed as not saved).
    goto(at, "fit")
    assert not errors(at), errors(at)
    assert _fit_calls() == before
    assert "Readings" in [h.value for h in at.subheader]
    readings = _table(at, "Balanced accuracy")
    assert readings.set_index("Channel").loc["CH3 RBF SVM", "Status"] == "not saved"
    goto(at, "sample")  # the rebuilt sample is the session's sample now
    assert not errors(at), errors(at)
    goto(at, "logbook")

    # Delete asks first; "Keep it" keeps it.
    at.button(key="lb_delete").click().run()
    assert any("for good?" in w.value for w in at.warning)
    at.button(key="lb_delete_no").click().run()
    assert folder.is_dir() and "lb_delete_yes" not in [b.key for b in at.button]
    at.button(key="lb_delete").click().run()
    at.button(key="lb_delete_yes").click().run()
    assert not errors(at), errors(at)
    assert not folder.exists()
    assert f"Deleted the saved channel set {run_id}." in [s.value for s in at.success]
    assert "No saved channel sets yet." in _text(at)
    assert "has since been deleted" in _text(at)

    # Clear the history, again after a confirmation.
    at.button(key="lb_clear").click().run()
    assert any("Clear every line" in w.value for w in at.warning)
    at.button(key="lb_clear_yes").click().run()
    assert not errors(at), errors(at)
    assert "No fits recorded yet." in _text(at)


def test_loading_over_an_unsaved_fit_asks_first(fresh_caches: None,
                                                quick_fit: tuple[PreparedDataset, TrainingRun]) -> None:
    at, run_id = _quick_app(quick_fit)
    fitted = at.session_state[state.RUN]
    copy_id = f"{run_id}-copy"
    persist.save_run(replace(fitted, run_id=copy_id, bundle_path=None))  # another saved set of the same sample
    goto(at, "logbook")
    at.selectbox(key="lb_pick").set_value(copy_id).run()

    # Load: the unsaved fit is not replaced without a word.
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    assert any(f"Run {run_id} on the bench was fitted in this session and is not saved" in w.value
               for w in at.warning)
    assert at.session_state[state.RUN] is fitted
    at.button(key="lb_load_no").click().run()
    assert at.session_state[state.RUN] is fitted and "lb_load_no" not in [b.key for b in at.button]

    # Save it first, then load: both happen, and the fit can be loaded back later.
    at.button(key="lb_load").click().run()
    at.button(key="lb_load_save").click().run()
    assert not errors(at), errors(at)
    assert (settings_mod.MODELS_DIR / run_id / "manifest.json").is_file()
    message = " ".join(s.value for s in at.success)
    assert message.startswith(f"Saved run {run_id}") and f"Loaded run {copy_id} from disk." in message
    assert at.session_state[state.RUN].run_id == copy_id and at.session_state[state.RUN].origin == "loaded"

    # A loaded run is on disk already: loading another set asks nothing.
    at.selectbox(key="lb_pick").set_value(run_id).run()
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    assert at.session_state[state.RUN].run_id == run_id and "lb_load_save" not in [b.key for b in at.button]

    # A different sample drawn since: loading rebuilds the run's own sample and says which one it replaced.
    draw_synthetic_sample(at, flows=2_000, budget=1_200)
    drawn = at.session_state[state.PREPARED].fingerprint
    goto(at, "logbook")
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    message = " ".join(s.value for s in at.success)
    assert f"(fingerprint {drawn[:12]}) was replaced" in message
    assert at.session_state[state.PREPARED].fingerprint == fitted.dataset_fingerprint


def test_a_damaged_bundle_is_refused_and_changes_nothing(fresh_caches: None,
                                                         quick_fit: tuple[PreparedDataset, TrainingRun]) -> None:
    at, run_id = _quick_app(quick_fit)
    at.button(key="lb_save").click().run()
    folder: Path = settings_mod.MODELS_DIR / run_id
    target = folder / "logreg.joblib"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    refusal = " ".join(e.value for e in at.error)
    assert refusal.startswith("Refused:") and "logreg.joblib" in refusal
    assert at.session_state[state.RUN].origin == "fitted"

    # A hand-edited manifest is listed as refused and refused on load.
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["best_balanced_accuracy"] = 0.999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    at.run()
    assert _table(at, "Folder")["Check"].iloc[0].startswith("refused: its manifest was changed")
    at.button(key="lb_load").click().run()
    assert "changed or damaged after saving" in " ".join(e.value for e in at.error)


def test_broken_folders_never_crash_the_station(fresh_caches: None,
                                                quick_fit: tuple[PreparedDataset, TrainingRun]) -> None:
    at, run_id = _quick_app(quick_fit)
    # Malformed manifests in the models folder are named, not listed, and never raised.
    for name, text in {"x-rows": '{"run_id": "x-rows", "rows": "x"}', "x-broken": "{ not json"}.items():
        (settings_mod.MODELS_DIR / name).mkdir(parents=True)
        (settings_mod.MODELS_DIR / name / "manifest.json").write_text(text, encoding="utf-8")
    at.run()
    assert not errors(at), errors(at)
    assert "Not listed, because their manifest cannot be read: x-broken" in _text(at)
    # A damaged folder under the run's own id: Save reports it instead of raising into the page.
    (settings_mod.MODELS_DIR / run_id).mkdir()
    (settings_mod.MODELS_DIR / run_id / "manifest.json").write_text("{ not json", encoding="utf-8")
    at.button(key="lb_save").click().run()
    assert not errors(at), errors(at)
    assert any(e.value.startswith(f"Run {run_id} could not be saved:") for e in at.error)


def test_a_version_change_loads_but_is_marked_not_verified(fresh_caches: None,
                                                           quick_fit: tuple[PreparedDataset, TrainingRun]) -> None:
    at, run_id = _quick_app(quick_fit)
    at.button(key="lb_save").click().run()
    folder = settings_mod.MODELS_DIR / run_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["library_versions"]["xgboost"] = "0.0.1"
    persist.write_manifest(folder, manifest)  # a consistent manifest, as a bundle saved with that version has
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    warning = " ".join(w.value for w in at.warning)
    assert "Not verified: the libraries differ" in warning and "xgboost 0.0.1 ->" in warning
    assert at.session_state[state.RUN].origin == "loaded"
    assert "not verified" in _text(at)


def test_a_fit_no_page_adopts_is_still_in_the_history(fresh_caches: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A background fit records itself when it ends, even if its tab is gone before any page adopts it."""
    at = new_app().run()
    draw_synthetic_sample(at, flows=2_000, budget=1_200)
    goto(at, "fit")
    at.multiselect(key="fit_channels").set_value(QUICK)
    monkeypatch.setenv("GRATICULE_SYNC_TRAINING", "0")
    at.button(key="fit_submit").click().run()
    job = get_job(at.session_state[state.JOB_ID])
    assert job is not None and job.wait(120)
    # Nothing has rendered since the fit ended; the run is in the history all the same.
    assert job.hook_error is None
    assert RunHistory().list()["run_id"].tolist() == [job.run_id]


def test_the_history_counts_every_run_and_the_csv_holds_them_all(fresh_caches: None,
                                                                 quick_fit: tuple[PreparedDataset, TrainingRun],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    at, run_id = _quick_app(quick_fit)
    history = RunHistory()
    line = run_summary(at.session_state[state.RUN])
    lines = [{**line, "run_id": f"20200101-{i:06d}-0000", "created_utc": f"2020-01-01T00:00:{i % 60:02d}+00:00"}
             for i in range(204)]
    with closing(sqlite3.connect(history.path)) as conn, conn:
        conn.executemany(f"INSERT INTO runs ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})",
                         [[row[c] for c in COLUMNS] for row in lines])
    exported: list[int] = []
    real_csv = logbook.history_csv

    def spy(frame: pd.DataFrame) -> bytes:
        exported.append(len(frame))
        return real_csv(frame)

    monkeypatch.setattr(logbook, "history_csv", spy)
    at.run()
    assert not errors(at), errors(at)
    assert len(_table(at, "Train rows")) == logbook.HISTORY_SHOWN
    assert "205 runs recorded" in _text(at) and "the CSV holds all 205" in _text(at)
    assert exported == [205]
