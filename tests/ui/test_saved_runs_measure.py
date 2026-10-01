"""03 Measure on channel sets loaded at the Logbook.

* A fit saved at the Logbook and loaded again, in the same session and (``-m slow``) in a FRESH PROCESS (a new
  Python interpreter running its own headless session), reads at 03 Measure exactly as the original run did in
  memory: the same tables (leaderboard, per-class), the same captions and the same charts, data included. Only the
  single-flow latency is measured afresh on every visit, so that column and its chart are left out of the
  comparison. The fit uses every channel a saved set keeps (CH3 never is: its model is made of training rows).
* Saving, loading and working 03 Measure's views and permutation importance on the loaded run fit nothing; only
  the explicit cross-validation button fits (k folds per chosen channel), on the rebuilt training rows.
* The manifest keeps the full readings 03 Measure shows (``tests/integration/test_persist.py`` checks that the
  ones Measure already computed are taken as they are).
* A run loaded without its held-out rows (its data folder has gone) explains itself at 03 Measure, 02 Fit and the
  Logbook, shows the readings saved with it, and never raises.
* With the real files (``-m realdata``): a CIC-IDS2017 run's held-out rows are rebuilt from the folder set on the
  Bench, and 03 Measure reads as the fit did.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import graticule.settings as settings_mod
from graticule import evaluate, persist
from graticule.data.prepare import DataRequest, prepare_dataset
from graticule.models import train
from graticule.models.train import TrainRequest, build_training_data, train_all
from tests.ui.harness import (  # noqa: F401
    draw_synthetic_sample,
    errors,
    fresh_caches,
    goto,
    new_app,
    touch_every_widget,
)
from ui import state
from ui.pages import measure as measure_page

pytestmark = pytest.mark.ui
ROOT = Path(__file__).resolve().parents[2]
#: The one reading 03 Measure takes afresh on every visit (median of 30 single-row calls), and its chart.
TIMED_COLUMN = "Single-flow ms"
TIMED_CHART = "Single-flow latency"
#: The permutation-importance estimate, which counts the single-flow latency as the cost of one call.
TIMED_CAPTION = "Shuffles each of the"
SECTIONS = ["Readings overview", "Confusion matrices", "Curves", "Channel detail", "Timing", "Cross-validation",
            "Downloads"]
#: Every channel a saved set keeps.
SAVED_CHANNELS = ["forest", "xgboost", "mlp", "logreg"]


def _fits() -> int:
    return sum(train.FIT_CALLS.values())


def _text(at: AppTest) -> str:
    """Every markdown, caption and alert text of the last run."""
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [a.value for kind in ("success", "info", "warning", "error") for a in getattr(at, kind)]
    return " ".join(parts)


def _readings(at: AppTest) -> pd.DataFrame:
    """The 02 Fit readings table."""
    return next(d.value for d in at.dataframe if "Balanced accuracy" in d.value.columns)


def measure_snapshot(at: AppTest) -> dict[str, Any]:
    """What 03 Measure shows that must not depend on where the run came from.

    Section titles, every table (without the timed single-flow column), every caption (except the permutation
    estimate, which builds on that latency), and every chart except the single-flow latency one, as its Vega-Lite
    spec plus the serialised datasets it draws.
    """
    charts: dict[str, tuple[str, list[bytes]]] = {}
    for chart in at.get("vega_lite_chart"):
        spec = json.loads(chart.proto.spec)
        title = spec.get("title")
        name = title.get("text") if isinstance(title, dict) else str(title or "")
        if name != TIMED_CHART:
            charts[name] = (chart.proto.spec, [d.SerializeToString() for d in chart.proto.datasets])
    return {
        "subheaders": [h.value for h in at.subheader],
        "tables": [d.value.drop(columns=[TIMED_COLUMN], errors="ignore") for d in at.dataframe],
        "captions": [c.value for c in at.caption if not c.value.startswith(TIMED_CAPTION)],
        "charts": charts,
    }


def _assert_same_measure(seen: dict[str, Any], expected: dict[str, Any], where: str) -> None:
    assert seen["subheaders"] == expected["subheaders"] == SECTIONS, where
    assert len(seen["tables"]) == len(expected["tables"]) >= 2, where
    for got, want in zip(seen["tables"], expected["tables"]):
        pd.testing.assert_frame_equal(got, want, check_exact=True, obj=f"{where}: table")
    assert seen["captions"] == expected["captions"], where
    assert sorted(seen["charts"]) == sorted(expected["charts"]), where
    for name, drawn in expected["charts"].items():
        assert seen["charts"][name] == drawn, f"{where}: chart {name!r} differs"


# Run in a new interpreter: point the settings at the test's folders, load the saved set at the Logbook, then
# visit 03 Measure and hand back what it shows.
FRESH_PROCESS = r"""
import pickle, sys
from pathlib import Path

import graticule.settings as settings_mod

settings_file, models_dir, history_dir, out = (Path(a) for a in sys.argv[1:5])
settings_mod.SETTINGS_FILE, settings_mod.MODELS_DIR, settings_mod.HISTORY_DIR = settings_file, models_dir, history_dir

from graticule import evaluate
from graticule.models import train
from tests.ui.harness import goto, new_app
from tests.ui.test_saved_runs_measure import measure_snapshot
from ui import state

at = new_app("logbook").run()
at.button(key="lb_load").click().run()
problems = [e.value for e in at.exception]
messages = [s.value for s in at.success] + [w.value for w in at.warning] + [e.value for e in at.error]
run = at.session_state[state.RUN]
goto(at, "measure")
problems += [e.value for e in at.exception]
evaluations = evaluate.cached_evaluations(run) or {}
out.write_bytes(pickle.dumps({
    "errors": problems, "messages": messages, "fits": sum(train.FIT_CALLS.values()), "origin": run.origin,
    "run_id": run.run_id, "has_test_rows": run.has_test_rows, "snapshot": measure_snapshot(at),
    "metrics": {key: ev.metrics for key, ev in evaluations.items()},
    "confusion": {key: ev.confusion for key, ev in evaluations.items()},
}))
"""


class FreshProcess:
    """A new Python process that loads the only saved set at the Logbook and visits 03 Measure (see
    :data:`FRESH_PROCESS`). It starts at once and runs alongside the test; :meth:`result` waits for it."""

    def __init__(self, tmp_path: Path) -> None:
        self.out = tmp_path / "fresh_process.pickle"
        env = {k: v for k, v in os.environ.items() if k != settings_mod.ENV_DATA_DIR}
        env.update(GRATICULE_SYNC_TRAINING="1", GRATICULE_TEST_PROFILE="1", PYTHONUTF8="1")
        self.process = subprocess.Popen(
            [sys.executable, "-c", FRESH_PROCESS, str(settings_mod.SETTINGS_FILE), str(settings_mod.MODELS_DIR),
             str(settings_mod.HISTORY_DIR), str(self.out)],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
        )

    def result(self) -> dict[str, Any]:
        """Wait for the process and return what its 03 Measure showed."""
        _, stderr = self.process.communicate(timeout=300)
        assert self.process.returncode == 0, stderr[-4000:]
        return pickle.loads(self.out.read_bytes())

    def stop(self) -> None:
        """End the process if it is still running (a failed test does not leave it behind)."""
        if self.process.poll() is None:
            self.process.kill()
            self.process.communicate()


def _plain_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Metrics as a manifest stores them (non-finite values become None)."""
    return {k: (None if not math.isfinite(float(v)) else float(v)) for k, v in metrics.items()}


def _fit_save_and_measure(at: AppTest) -> dict[str, Any]:
    """Fit every channel a saved set keeps on a small sample, save the run at the Logbook (nothing is fitted), and
    take 03 Measure's readings of the run in memory: the reference every reload is compared with."""
    draw_synthetic_sample(at, flows=2_000, budget=1_200)
    goto(at, "fit")
    at.multiselect(key="fit_channels").set_value(SAVED_CHANNELS)
    at.button(key="fit_submit").click().run()
    assert not errors(at), errors(at)
    original = at.session_state[state.RUN]
    fit_readings = _readings(at)
    fits = _fits()
    goto(at, "logbook")
    at.button(key="lb_save").click().run()
    assert not errors(at), errors(at)
    assert _fits() == fits
    manifest = json.loads((settings_mod.MODELS_DIR / original.run_id / "manifest.json").read_text(encoding="utf-8"))
    goto(at, "measure")
    assert not errors(at), errors(at)
    evaluations = evaluate.cached_evaluations(original)
    assert evaluations is not None and list(evaluations) == original.ok_channels() == SAVED_CHANNELS
    return {"original": original, "fit_readings": fit_readings, "fits": fits, "manifest": manifest,
            "reference": measure_snapshot(at), "evaluations": evaluations}


@pytest.mark.slow  # save, reload and measure twice through the pages (about 8 s); the round trip itself is
# covered on every run by tests/integration/test_persist.py
def test_a_saved_fit_reloaded_in_this_session_measures_exactly_like_the_original(fresh_caches: None) -> None:
    at = new_app().run()
    fitted = _fit_save_and_measure(at)
    # The manifest holds the full readings 03 Measure shows (the same numbers, computed from the stored ones).
    for key, evaluation in fitted["evaluations"].items():
        assert fitted["manifest"]["channels"][key]["evaluation_metrics"] == _plain_metrics(evaluation.metrics), key
    assert {"precision", "recall", "f1", "roc_auc", "average_precision",
            "precision_macro"} <= set(fitted["manifest"]["channels"]["forest"]["evaluation_metrics"])
    _load_and_measure_here(at, fitted["original"], fitted["reference"], fitted["fit_readings"], fitted["fits"])


@pytest.mark.slow  # a second interpreter loads the set (about 10 s); run with -m slow
def test_a_saved_fit_reloaded_in_a_fresh_process_measures_exactly_like_the_original(fresh_caches: None,
                                                                                   tmp_path: Path) -> None:
    at = new_app().run()
    fitted = _fit_save_and_measure(at)
    run_id = fitted["original"].run_id
    fresh_process = FreshProcess(tmp_path)
    try:
        fresh = fresh_process.result()
    finally:
        fresh_process.stop()
    assert fresh["errors"] == []
    message = " ".join(fresh["messages"])
    assert f"Loaded run {run_id} from disk." in message and "Verified: all 4 channels" in message, message
    assert "Held-out rows rebuilt" in message, message
    assert fresh["fits"] == 0 and fresh["origin"] == "loaded" and fresh["has_test_rows"]
    assert fresh["run_id"] == run_id
    _assert_same_measure(fresh["snapshot"], fitted["reference"], "fresh process")
    np.testing.assert_equal(fresh["metrics"], {k: ev.metrics for k, ev in fitted["evaluations"].items()})
    np.testing.assert_equal(fresh["confusion"], {k: ev.confusion for k, ev in fitted["evaluations"].items()})


def _load_and_measure_here(at: AppTest, original: Any, reference: dict[str, Any], fit_readings: pd.DataFrame,
                           fits: int) -> None:
    """Load the saved set in the test's own session, check 03 Measure against ``reference``, then work its views,
    permutation importance and cross-validation (nothing but the cross-validation button may fit), and check 02 Fit
    reads as it did after the fit. (Every widget of every station is changed on a fitted run by
    ``tests/ui/test_no_retrain.py``, and on a loaded run without held-out rows below; a loaded run with its rows is
    the same kind of run to 03 Measure.)"""
    run_id = original.run_id
    goto(at, "logbook")
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    assert _fits() == fits
    loaded = at.session_state[state.RUN]
    assert loaded.origin == "loaded" and loaded is not original and loaded.has_test_rows
    goto(at, "measure")
    assert not errors(at), errors(at)
    _assert_same_measure(measure_snapshot(at), reference, "this session")
    assert "loaded from disk" in _text(at)

    # The views, then permutation importance: nothing is fitted and the readings are never recomputed.
    first = evaluate.cached_evaluations(loaded)
    assert first is not None

    def nothing_fitted(label: str) -> None:
        assert not errors(at), (label, errors(at))
        assert _fits() == fits, f"{label} fitted something"
        assert at.session_state[state.RUN] is loaded, f"{label} replaced the run"
        again = evaluate.cached_evaluations(loaded)
        assert again is not None and all(again[k] is first[k] for k in first), f"{label} recomputed readings"

    at.radio(key="ms_cm_show").set_value("Counts").run()
    nothing_fitted("the confusion view")
    at.radio(key="ms_roc_zoom").set_value(measure_page.ROC_OPTIONS[1]).run()
    nothing_fitted("the ROC view")
    at.selectbox(key=f"ms_detail_channel-{run_id}").set_value("logreg").run()  # the quickest to measure
    nothing_fitted("the detail channel")
    at.button(key="ms_perm_run").click().run()
    assert not errors(at), errors(at)
    nothing_fitted("permutation importance")
    assert evaluate.stored_permutations(loaded)

    # Only the explicit cross-validation button fits: k folds for each chosen channel, on the rebuilt training rows.
    at.number_input(key="ms_cv_k").set_value(3).run()
    at.multiselect(key=f"ms_cv_channels-{run_id}").set_value(["forest", "logreg"]).run()
    nothing_fitted("choosing the folds and channels")
    at.button(key="ms_cv_run").click().run()
    assert not errors(at), errors(at)
    assert _fits() - fits == 3 * 2
    cv = evaluate.stored_cross_validation(loaded)
    assert cv is not None and list(cv["key"]) == ["forest", "logreg"]
    assert cv.attrs["rows_available"] == len(original.data.y_train)
    assert evaluate.stored_cross_validation(original) is None  # the run in memory before the load is untouched

    # 02 Fit reads the loaded run exactly as it read the fit (same table), marked as loaded; still no new fit.
    goto(at, "fit")
    assert not errors(at), errors(at)
    assert _readings(at).equals(fit_readings)
    assert "loaded from disk" in _text(at) and at.session_state[state.RUN] is loaded
    assert _fits() - fits == 3 * 2 and state.JOB_ID not in at.session_state


THURSDAY_WEB = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"


@pytest.mark.realdata
@pytest.mark.slow  # reads a CIC-IDS2017 file twice (about 6 s); run with -m realdata
def test_a_real_data_set_is_rebuilt_from_the_bench_folder_and_measures_like_the_fit(
        fresh_caches: None, real_data_dir: Path, tmp_path: Path) -> None:
    """The Logbook rebuilds a real run's held-out rows through the app's cached file readers, from the folder set
    on the Bench (the one recorded with the run has moved), and 03 Measure then reads exactly as the fit did."""
    if not (real_data_dir / THURSDAY_WEB).is_file():
        pytest.skip(f"{THURSDAY_WEB} is not in the data folder")
    prepared = prepare_dataset(DataRequest(source="cicids", data_dir=str(real_data_dir), files=(THURSDAY_WEB,),
                                           row_budget=6_000, seed=42))
    request = TrainRequest(profile="test", mode="multiclass", min_class_count=10,
                           channels=("forest", "xgboost", "logreg"))
    run = train_all(build_training_data(prepared, request), request, data_request=prepared.request,
                    dataset_fingerprint=prepared.fingerprint)
    expected = evaluate.leaderboard(evaluate.evaluate_run(run), run).drop(columns=["key", "Training rows",
                                                                                   TIMED_COLUMN])
    folder = persist.save_run(run)
    # The folder recorded with the run has moved since; the Bench points at where the files are now.
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["data_request"]["data_dir"] = str(tmp_path / "moved")
    persist.write_manifest(folder, manifest)
    settings_mod.save_settings(settings_mod.AppSettings(data_dir=str(real_data_dir)))
    fits = _fits()

    at = new_app("logbook").run()
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    message = " ".join(s.value for s in at.success)
    assert "Verified: all 3 channels" in message and "Held-out rows rebuilt" in message, message
    loaded = at.session_state[state.RUN]
    assert loaded.origin == "loaded" and loaded.has_test_rows
    assert np.array_equal(loaded.data.test_rows, run.data.test_rows)
    assert at.session_state[state.PREPARED].fingerprint == run.dataset_fingerprint
    goto(at, "measure")
    assert not errors(at), errors(at)
    board = next(d.value for d in at.dataframe if "Gap to best" in d.value.columns)
    pd.testing.assert_frame_equal(board.drop(columns=[TIMED_COLUMN]), expected, check_exact=True)
    assert _fits() == fits


def test_a_run_loaded_without_its_held_out_rows_explains_itself(fresh_caches: None, tmp_path: Path) -> None:
    prepared = prepare_dataset(DataRequest(source="synthetic", synthetic_flows=1_500, seed=5))
    request = TrainRequest(profile="test", seed=5, channels=("forest", "xgboost", "logreg"))
    data = build_training_data(prepared, request)
    run = train_all(data, request, data_request=prepared.request, dataset_fingerprint=prepared.fingerprint)
    folder = persist.save_run(run)  # into the test's own saved_models folder
    # As if the run had been fitted on CIC-IDS2017 files in a folder that has gone since (the manifest is sealed
    # again after the edit, as a bundle saved that way would be; the request is only read to rebuild the rows).
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["data_request"].update(source="cicids", files=["Monday-WorkingHours.pcap_ISCX.csv"],
                                    data_dir=str(tmp_path / "gone"))
    persist.write_manifest(folder, manifest)
    n_train, n_test = len(run.data.y_train), len(run.data.y_test)
    saved = {key: run.channels[key].extra["metrics"] for key in run.ok_channels()}
    fits = _fits()
    # Another session (another browser tab) is reading the run as it was fitted.
    fitting_tab = new_app("fit")
    state.run_registry().put(run.run_id, run)
    fitting_tab.session_state[state.LAST_RUN_ID] = run.run_id
    fitting_tab.session_state[state.RUN] = run
    fitting_tab.run()
    assert not errors(fitting_tab), errors(fitting_tab)

    at = new_app("logbook").run()
    at.button(key="lb_load").click().run()
    assert not errors(at), errors(at)
    # The loaded copy shares the run id (and replaces the process registry's entry), yet the other tab keeps the
    # run it chose, with its held-out rows.
    fitting_tab.run()
    assert not errors(fitting_tab), errors(fitting_tab)
    assert "loaded from disk" not in _text(fitting_tab)
    assert f"Measured on {n_test:,} held-out rows: ○ Normal" in _text(fitting_tab)
    loaded = at.session_state[state.RUN]
    assert loaded.origin == "loaded" and not loaded.has_test_rows and loaded.ok_channels() == run.ok_channels()
    message = " ".join(s.value for s in at.success)
    assert "Verified: all 3 channels" in message and "Held-out rows not rebuilt" in message, message
    assert "does not exist" in message and "Set the data folder on the Bench" in message
    assert "no held-out rows" in _text(at) and "lb_rebuild" in [b.key for b in at.button]
    # Trying again loads the set again; the folder is still missing, so the rows stay unavailable.
    at.button(key="lb_rebuild").click().run()
    assert not errors(at), errors(at)
    loaded = at.session_state[state.RUN]
    assert loaded.origin == "loaded" and not loaded.has_test_rows and _fits() == fits
    assert "Held-out rows not rebuilt" in " ".join(s.value for s in at.success)

    # 03 Measure: says why nothing can be measured, links the way back, lists the saved readings; no charts.
    goto(at, "measure")
    assert not errors(at), errors(at)
    assert "was loaded from disk without its held-out rows" in _text(at)
    links = [link.proto.label for link in at.get("page_link")]
    assert "Go to Logbook" in links and "Go to Bench" in links
    assert not at.get("vega_lite_chart") and "ms_cv_run" not in [b.key for b in at.button]
    recorded = next(d.value for d in at.dataframe if "Balanced accuracy" in d.value.columns)
    assert recorded["Channel"].tolist() == ["CH1 Random forest", "CH2 XGBoost", "CH5 Logistic regression"]
    assert recorded["Balanced accuracy"].tolist() == [saved[k]["balanced_accuracy"] for k in run.ok_channels()]
    touch_every_widget(at, lambda label: None)
    assert evaluate.cached_evaluations(loaded) is None

    # 02 Fit: the split sizes and held-out classes saved with the run, not "0 rows".
    goto(at, "fit")
    assert not errors(at), errors(at)
    text = _text(at)
    assert f"train {n_train:,} · test {n_test:,} rows" in text and "loaded from disk" in text
    assert f"Measured on {n_test:,} held-out rows when fitted: ○ Normal" in text
    readings = next(d.value for d in at.dataframe if "Balanced accuracy" in d.value.columns)
    assert readings["Balanced accuracy"].tolist() == [saved[k]["balanced_accuracy"] for k in run.ok_channels()]

    # Every other station draws without an exception, and nothing anywhere fitted a channel.
    for key in ("sample", "probe", "assay", "sweep", "record", "bench", "logbook"):
        goto(at, key)
        assert not errors(at), (key, errors(at))
    assert _fits() == fits
