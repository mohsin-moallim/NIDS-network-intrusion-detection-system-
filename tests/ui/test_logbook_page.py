"""Headless checks of the Logbook: save the current fit (CH3 stays out by default), list it, load it back verified,
delete it; the "Also save CH3" opt-in (unticked for every new run, then saved, counted and loaded back as an
ordinary channel); the confirmation before a load replaces an unsaved fit, robustness against damaged folders, and
the history."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

import nids.settings as settings_mod
from nids import persist
from nids.data.prepare import PreparedDataset
from nids.history import COLUMNS, RunHistory, run_summary
from nids.models import train
from nids.models.jobs import get_job
from nids.models.train import TrainingRun
from nids.models.zoo import MODEL_KEYS
from nids.report import exports
from tests.ui.harness import (  # noqa: F401
    app_with_run,
    app_with_sample,
    drawn_sample,
    errors,
    fit_synthetic,
    fresh_caches,
    goto,
    new_app,
)
from ui import state
from ui.pages import logbook

pytestmark = pytest.mark.ui
#: CH3 is fitted but not saved by default (its model is made of training rows), so a full fit loads back with four.
KEPT = [k for k in MODEL_KEYS if k != "svm"]
VERIFIED = "Verified: all 4 channels reproduced their saved probe readings exactly."
QUICK = ["forest", "logreg"]
#: A quick fit that includes CH3, for the "Also save CH3" opt-in.
WITH_SVM = ["forest", "svm", "logreg"]


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
    """A Logbook session holding a small synthetic sample and a fit of every channel, as right after a fit at 02 Fit:
    the run is in the history (as 02 Fit records it when the run is adopted), not saved.

    The fit is made once per session (``fit_synthetic``, shared with the 03 Measure and 04 Probe tests); a fit
    through the 02 Fit form, and the single history line it records, are checked by
    ``tests/ui/test_fit_page.py::test_a_fit_through_the_form_stores_one_run``.
    """
    prepared, run = fit_synthetic()
    RunHistory().record(run)
    at = app_with_run(run, prepared, key="logbook").run()
    assert not errors(at), errors(at)
    return at, run.run_id


@pytest.fixture(scope="module")
def quick_fit() -> tuple[PreparedDataset, TrainingRun]:
    """The small synthetic sample 01 Sample draws and a fit of the two quick channels, made once for the tests
    that need any fit."""
    return fit_synthetic(flows=2_000, budget=1_600, channels=tuple(QUICK))


@pytest.fixture(scope="module")
def svm_fit() -> tuple[PreparedDataset, TrainingRun]:
    """The small synthetic sample 01 Sample draws and a quick fit that includes CH3 (made once for the opt-in
    tests)."""
    return fit_synthetic(flows=2_000, budget=1_600, channels=tuple(WITH_SVM))


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

    # The fit's history line is listed.
    history = _table(at, "Train rows")
    assert history["Run"].tolist() == [run_id]
    assert history["Channels"].tolist() == ["CH1 CH2 CH3 CH4 CH5"]
    assert "fitted in this session" in _text(at)
    # Saving says plainly what is written, and that CH3 stays out unless chosen (the box is there, unticked).
    assert "No dataset rows are written unless you save CH3" in _text(at) and "support vectors" in _text(at)
    box = at.checkbox(key=logbook.save_svm_key(at.session_state[state.RUN]))
    assert box.label == "Also save CH3 (RBF SVM)" and box.value is False

    # Save the current fit with the box left unticked: everything but CH3.
    at.button(key="lb_save").click().run()
    assert not errors(at), errors(at)
    folder = settings_mod.MODELS_DIR / run_id
    assert (folder / "manifest.json").is_file() and not (folder / "svm.joblib").exists()
    saved_message = " ".join(s.value for s in at.success)
    assert saved_message.startswith(f"Saved run {run_id} to {folder}.")
    # CH3 is still fitted on the bench: the notice and the caption say so and how to include it, not "fit it again".
    assert "CH3 RBF SVM was left out of the saved set, as by default" in saved_message
    assert ("It stays fitted on the bench for this session; to include it in the set, delete the set at the Logbook "
            "and save again with \"Also save CH3 (RBF SVM)\" ticked.") in saved_message
    assert "Fit it again" not in saved_message and "Fit it again" not in _text(at)
    assert "lb_save" not in [b.key for b in at.button] and box.key not in [c.key for c in at.checkbox]
    assert str(folder) in _text(at)
    assert _table(at, "Train rows")["Saved"].tolist() == ["yes"]
    assert RunHistory().get(run_id)["saved_path"] == str(folder)  # the full path is kept (and in the CSV)
    saved = _table(at, "Check")
    assert "Folder" not in saved.columns  # the folder is named after the run, said once under the table
    assert f"Saved sets are kept in {folder.parent}, each in a folder named after its run." in _text(at)
    assert saved["Run"].tolist() == [run_id] and saved["Channels"].tolist() == ["CH1 CH2 CH4 CH5"]
    assert saved["Check"].tolist() == ["manifest intact"] and saved["Training rows inside"].tolist() == [0]

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


def test_ch3_is_saved_only_when_ticked_and_loads_back_as_an_ordinary_channel(
        fresh_caches: None, svm_fit: tuple[PreparedDataset, TrainingRun]) -> None:
    at, run_id = _quick_app(svm_fit)
    run = at.session_state[state.RUN]
    vectors = run.channels["svm"].extra["support_vectors"]
    assert vectors > 0
    fits = _fit_calls()

    # The opt-in: drawn unticked, with a plain account of what ticking it writes, and where.
    box = at.checkbox(key=logbook.save_svm_key(run))
    assert box.label == "Also save CH3 (RBF SVM)" and box.value is False
    captions = " ".join(c.value for c in at.caption)
    assert f"CH3's model is its {vectors:,} support vectors, which are training rows after gap filling" in captions
    assert "its saved scaler turns them back into the original values" in captions
    assert f"saved_models\\{run_id}\\ on this machine (git ignores the folder)" in captions
    assert "Leave it unticked to keep dataset rows out of the project, as by default" in captions

    # Ticked and saved: CH3 is written, and the notice says how many training rows the set now holds.
    box.check().run()
    assert at.checkbox(key=logbook.save_svm_key(run)).value is True
    at.button(key="lb_save").click().run()
    assert not errors(at), errors(at)
    folder = settings_mod.MODELS_DIR / run_id
    assert (folder / "svm.joblib").is_file()
    message = " ".join(s.value for s in at.success)
    assert message.startswith(f"Saved run {run_id} to {folder}.")
    assert f"CH3 (RBF SVM) was saved with it, by choice: the set holds {vectors:,} training rows" in message
    assert "is not kept in saved sets" not in message
    assert f"This set holds {vectors:,} training rows (CH3)" in _text(at)
    assert state.unsaved_channel_note(run) == ""
    saved = _table(at, "Check")
    assert saved["Channels"].tolist() == ["CH1 CH3 CH5"] and saved["Training rows inside"].tolist() == [vectors]

    # Loaded back: CH3 is verified and restored like the other channels, nothing is fitted, and the set's rows
    # are named.
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    message = " ".join(s.value for s in at.success)
    assert f"Loaded run {run_id} from disk." in message
    assert "Verified: all 3 channels reproduced their saved probe readings exactly." in message
    assert f"This set holds {vectors:,} training rows" in message and "is not kept in saved sets" not in message
    loaded = at.session_state[state.RUN]
    assert loaded.origin == "loaded" and loaded.has_test_rows and loaded.ok_channels() == WITH_SVM
    assert loaded.channels["svm"].status == "ok"
    assert np.array_equal(loaded.channels["svm"].y_pred, run.channels["svm"].y_pred)
    assert np.array_equal(loaded.channels["svm"].proba, run.channels["svm"].proba)
    text = _text(at)
    assert f"holds {vectors:,} training rows (CH3)" in text and "3 channels: CH1 CH3 CH5" in text
    assert state.unsaved_channel_note(loaded) == "" and state.saved_training_rows(loaded) == vectors
    assert _fit_calls() == fits
    goto(at, "fit")  # 02 Fit lists the loaded CH3 as fitted, not as "not saved"
    assert not errors(at), errors(at)
    assert _table(at, "Balanced accuracy").set_index("Channel").loc["CH3 RBF SVM", "Status"] == "fitted"
    assert _fit_calls() == fits

    # The set is deleted while its run is on the bench: the panel no longer says the rows stay on disk.
    goto(at, "logbook")
    at.button(key="lb_delete").click().run()
    at.button(key="lb_delete_yes").click().run()
    assert not errors(at), errors(at)
    assert not folder.exists() and state.saved_training_rows(at.session_state[state.RUN]) == 0
    text = _text(at)
    assert "(that folder has since been deleted)" in text and "No saved channel sets yet." in text
    assert f"The set this run was loaded from held CH3's {vectors:,} training rows (its support vectors)" in text
    assert f"holds {vectors:,} training rows (CH3)" not in text and "They stay in this machine's" not in text
    assert at.session_state[state.RUN].channels["svm"].ok

    # The choice is never carried over: the next fit on the bench gets its own box, unticked.
    following = replace(run, run_id=f"{run_id}-next", bundle_path=None)
    state.run_registry().put(following.run_id, following)
    at.session_state[state.RUN] = following
    at.session_state[state.LAST_RUN_ID] = following.run_id
    goto(at, "logbook")
    assert not errors(at), errors(at)
    assert at.checkbox(key=logbook.save_svm_key(following)).value is False


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

    # A different sample drawn since (as Draw sample leaves it): loading rebuilds the run's own sample and says
    # which one it replaced.
    at.session_state[state.PREPARED] = drawn_sample(flows=2_000, budget=1_200)
    drawn = at.session_state[state.PREPARED].fingerprint
    assert drawn != fitted.dataset_fingerprint
    goto(at, "logbook")
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    message = " ".join(s.value for s in at.success)
    assert f"(fingerprint {drawn[:12]}) was replaced" in message
    assert at.session_state[state.PREPARED].fingerprint == fitted.dataset_fingerprint


def test_save_then_load_follows_the_ch3_box(fresh_caches: None,
                                            svm_fit: tuple[PreparedDataset, TrainingRun]) -> None:
    """"Save <run>, then load" saves the fit on the bench as the "Also save CH3" box says, and says so first."""
    at, run_id = _quick_app(svm_fit)
    fitted = at.session_state[state.RUN]
    vectors = fitted.channels["svm"].extra["support_vectors"]
    copy_id = f"{run_id}-copy"
    persist.save_run(replace(fitted, run_id=copy_id, bundle_path=None))  # another set of the same sample, no CH3
    goto(at, "logbook")
    at.selectbox(key="lb_pick").set_value(copy_id).run()
    fits = _fit_calls()

    # Unticked: the confirmation says the save would leave CH3 out.
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    assert "Saving first leaves CH3 (RBF SVM) out, as by default" in _text(at)
    at.button(key="lb_load_no").click().run()

    # Ticked: the confirmation says CH3 is included, and the save writes it before the other set is loaded.
    at.checkbox(key=logbook.save_svm_key(fitted)).check().run()
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    assert "Saving first includes CH3 (RBF SVM) and its training rows, as ticked above." in _text(at)
    at.button(key="lb_load_save").click().run()
    assert not errors(at), errors(at)
    folder = settings_mod.MODELS_DIR / run_id
    assert (folder / "svm.joblib").is_file()
    assert persist.read_bundle_summary(folder).holds_training_rows == vectors
    message = " ".join(s.value for s in at.success)
    assert message.startswith(f"Saved run {run_id}")
    assert f"CH3 (RBF SVM) was saved with it, by choice: the set holds {vectors:,} training rows" in message
    assert f"Loaded run {copy_id} from disk." in message
    loaded = at.session_state[state.RUN]
    assert loaded.run_id == copy_id and loaded.channels["svm"].status == persist.NOT_SAVED
    table = _table(at, "Check").set_index("Run")
    assert table.loc[run_id, "Training rows inside"] == vectors and table.loc[copy_id, "Training rows inside"] == 0
    assert _fit_calls() == fits


def test_a_ch3_only_fit_points_to_the_box_and_is_saved_with_it(fresh_caches: None,
                                                               svm_fit: tuple[PreparedDataset, TrainingRun]) -> None:
    prepared, fitted = svm_fit
    only = {k: (r if k == "svm" else replace(r, status="failed", estimator=None)) for k, r in fitted.channels.items()}
    at, run_id = _quick_app((prepared, replace(fitted, run_id=f"{fitted.run_id}-svm", channels=only)))
    vectors = fitted.channels["svm"].extra["support_vectors"]
    assert "CH3 is the only fitted channel, so without the box there is nothing to save." in _text(at)

    # Unticked: nothing is saved, and the refusal names the box (CH3 is fitted; fitting it again would not help).
    at.button(key="lb_save").click().run()
    assert not errors(at), errors(at)
    refusal = " ".join(e.value for e in at.error)
    assert refusal.startswith(f"Run {run_id} was not saved: its only fitted channel is CH3 (RBF SVM)")
    assert f"Tick the box to save CH3 with its {vectors:,} training rows" in refusal and "Fit it again" not in refusal
    assert not (settings_mod.MODELS_DIR / run_id).exists()

    # "Save, then load" says so before anything is pressed.
    persist.save_run(replace(fitted, run_id=f"{run_id}-copy", bundle_path=None))
    at.run()
    at.selectbox(key="lb_pick").set_value(f"{run_id}-copy").run()
    at.button(key="lb_load").click().run()
    assert "Saving first would save nothing: CH3 (RBF SVM) is the only fitted channel" in _text(at)
    at.button(key="lb_load_no").click().run()

    # Ticked: saved, with the rows counted.
    at.checkbox(key=logbook.save_svm_key(at.session_state[state.RUN])).check().run()
    at.button(key="lb_save").click().run()
    assert not errors(at), errors(at)
    assert (settings_mod.MODELS_DIR / run_id / "svm.joblib").is_file()
    assert f"the set holds {vectors:,} training rows" in " ".join(s.value for s in at.success)


def test_a_fit_without_ch3_writes_no_rows_and_leftover_folders_are_named(
        fresh_caches: None, quick_fit: tuple[PreparedDataset, TrainingRun]) -> None:
    at, _ = _quick_app(quick_fit)
    text = _text(at)
    assert ("No dataset rows are written: CH3, the one channel whose model is made of training rows (and which is "
            "saved only by choice), has no fitted model in this run.") in text
    assert "unless you save CH3" not in text and len(at.checkbox) == 0

    # Work folders a save or a delete left behind: an old one is named (with CH3's rows), a fresh one is not (a save
    # in another session may still be writing it). The files are placeholders, not models.
    models = settings_mod.MODELS_DIR
    old = models / ".deleting-20260101-000000-abcd-120000000000"
    fresh = models / ".saving-20260101-000000-abcd-x1y2"
    for folder in (old, fresh):
        folder.mkdir(parents=True)
        for name in ("svm.joblib", "manifest.json"):
            (folder / name).write_bytes(b"placeholder")
    hour_ago = time.time() - 3_600
    for path in (old / "svm.joblib", old / "manifest.json", old):
        os.utime(path, (hour_ago, hour_ago))
    at.run()
    assert not errors(at), errors(at)
    warning = " ".join(w.value for w in at.warning)
    assert f"{old.name} (unfinished delete, 2 files, svm.joblib: CH3's training rows)" in warning
    assert "1 of them still holds CH3's training rows" in warning and fresh.name not in warning
    at.button(key="lb_leftovers").click().run()
    assert not errors(at), errors(at)
    assert not old.exists() and fresh.is_dir()
    assert "Removed 1 leftover folder." in [s.value for s in at.success]
    assert "lb_leftovers" not in [b.key for b in at.button]


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
    assert _table(at, "Check")["Check"].iloc[0].startswith("refused: its manifest was changed")
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
    at = app_with_sample(drawn_sample(flows=2_000, budget=1_200), "fit").run()
    at.pills(key="fit_channels").set_value(QUICK)
    monkeypatch.setenv("NIDS_SYNC_TRAINING", "0")
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
    names: dict[str, str] = {}
    real_button = DeltaGenerator.download_button

    def button_spy(self: DeltaGenerator, *args: Any, **kwargs: Any) -> bool:
        names[str(kwargs.get("key"))] = str(kwargs.get("file_name"))
        return real_button(self, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", button_spy)
    at.run()
    assert not errors(at), errors(at)
    assert len(_table(at, "Train rows")) == logbook.HISTORY_SHOWN
    assert "205 runs recorded" in _text(at) and "the CSV holds all 205" in _text(at)
    assert exported == [205]
    # The download carries the same file name as 07 Record's run-history export.
    assert names == {"lb_history_csv": "nids-run-history.csv"}
    assert logbook.HISTORY_FILE_NAME == exports.export_file_name("run_history", None) == "nids-run-history.csv"
