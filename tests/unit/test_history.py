"""Run history: recording, listing, reading back, deleting and clearing, across instances and processes."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

import nids.settings as settings_mod
from nids.data.prepare import DataRequest
from nids.features import FeatureChoice
from nids.history import COLUMNS, RunHistory, best_reading, run_summary
from nids.models.train import ChannelResult, TrainingData, TrainingRun, TrainRequest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
NAMES = ("Flow Duration", "Total Fwd Packets")


def make_run(run_id: str = "20261001-120000-ab12", created: str = "2026-10-01T12:00:00+00:00",
             scores: dict[str, float | None] | None = None, source: str = "synthetic",
             files: tuple[str, ...] = ()) -> TrainingRun:
    """A small finished run built by hand (no fitting): ``scores`` maps channel keys to balanced accuracy, or
    None for a channel that failed."""
    scores = {"forest": 0.91, "xgboost": 0.95, "svm": None} if scores is None else scores
    choice = FeatureChoice(mode="curated", columns=NAMES, include_port=False, k=None, ranking=None,
                           dropped_degenerate=())
    data = TrainingData(
        X_train=np.zeros((6, 2), dtype=np.float32), X_test=np.zeros((4, 2), dtype=np.float32),
        y_train=np.array([0, 1, 0, 1, 0, 1]), y_test=np.array([0, 1, 0, 1]), classes=("Normal", "Attack"),
        detailed_test_labels=np.array(["BENIGN", "Flood", "BENIGN", "Flood"]), feature_names=NAMES,
        feature_choice=choice, train_rows=np.arange(6), test_rows=np.arange(6, 10), reports={"seconds": 1.5},
    )
    channels = {}
    for key, value in scores.items():
        if value is None:
            channels[key] = ChannelResult(key=key, status="failed", error="RuntimeError: boom")
        else:
            channels[key] = ChannelResult(key=key, status="ok", extra={"metrics": {
                "accuracy": value, "balanced_accuracy": value, "f1_macro": value, "f1_weighted": value}})
    data_request = (DataRequest(source="synthetic") if source == "synthetic"
                    else DataRequest(source="cicids", data_dir="D:/flows", files=files))
    return TrainingRun(
        run_id=run_id, created_utc=created, request=TrainRequest(profile="test", channels=tuple(scores)),
        data_request=data_request, dataset_fingerprint="f" * 64, data=data, channels=channels, seconds=2.0,
        reference_sample=np.zeros((0, 2), dtype=np.float32), feature_quantiles=np.zeros((101, 2), np.float32),
    )


def test_record_list_get_delete_and_clear(tmp_path: Path) -> None:
    history = RunHistory(tmp_path / "nested" / "runs.sqlite3")
    assert history.list().empty and list(history.list().columns) == list(COLUMNS)
    older = make_run("20261001-120000-aaaa", "2026-10-01T12:00:00+00:00")
    newer = make_run("20261001-130000-bbbb", "2026-10-01T13:00:00+00:00", {"logreg": 0.8, "mlp": 0.85})
    history.record(older)
    history.record(newer)
    frame = history.list()
    assert frame["run_id"].tolist() == [newer.run_id, older.run_id]
    assert list(frame.columns) == list(COLUMNS)
    assert frame["rows_train"].dtype == np.int64 and frame["best_balanced_accuracy"].dtype == np.float64
    line = history.get(older.run_id)
    assert line is not None
    assert line["best_channel"] == "xgboost" and line["best_balanced_accuracy"] == pytest.approx(0.95)
    assert line["channels"] == "forest, xgboost" and line["source"] == "synthetic" and line["files"] == "generated"
    assert (line["rows_train"], line["rows_test"]) == (6, 4)
    assert line["seconds"] == pytest.approx(3.5)  # matrices 1.5 s + channels 2.0 s
    metrics = json.loads(line["metrics_json"])
    assert metrics["svm"] == {"status": "failed"} and metrics["forest"]["balanced_accuracy"] == pytest.approx(0.91)
    settings_blob = json.loads(line["settings_json"])
    assert settings_blob["train_request"]["profile"] == "test" and settings_blob["data_request"]["source"] == "synthetic"
    assert history.list(limit=1)["run_id"].tolist() == [newer.run_id]
    assert history.list(limit=None)["run_id"].tolist() == [newer.run_id, older.run_id]  # None: every line
    history.delete(older.run_id)
    assert history.get(older.run_id) is None and history.count() == 1
    history.delete("never-recorded")
    history.clear()
    assert history.count() == 0 and history.list().empty
    assert history.path.is_file()


def test_record_is_idempotent_and_notes_where_a_run_was_saved(tmp_path: Path) -> None:
    history = RunHistory(tmp_path / "runs.sqlite3")
    run = make_run()
    history.record(run)
    history.record(run)
    assert history.count() == 1
    assert history.get(run.run_id)["saved_path"] is None  # type: ignore[index]
    history.record(run, saved_path=tmp_path / "bundle")
    assert history.count() == 1
    assert history.get(run.run_id)["saved_path"] == str(tmp_path / "bundle")  # type: ignore[index]
    history.mark_saved(run.run_id, tmp_path / "elsewhere")
    assert history.get(run.run_id)["saved_path"] == str(tmp_path / "elsewhere")  # type: ignore[index]
    history.mark_saved("never-recorded", tmp_path)  # no line, no error
    assert history.count() == 1


def test_default_location_follows_the_settings_module(isolated_settings: Path) -> None:
    history = RunHistory()
    assert history.path == settings_mod.HISTORY_DIR / "runs.sqlite3"
    assert history.path.parent == isolated_settings / "run_history"
    history.record(make_run())
    assert history.path.is_file()


def test_history_survives_a_new_instance(tmp_path: Path) -> None:
    """Each instance opens its own connections: what one recorded, a new one reads back from the file."""
    path = tmp_path / "runs.sqlite3"
    RunHistory(path).record(make_run("20261001-120000-aaaa"))
    RunHistory(path).record(make_run("20261001-130000-bbbb", "2026-10-01T13:00:00+00:00"))
    fresh = RunHistory(path)
    assert fresh.list()["run_id"].tolist() == ["20261001-130000-bbbb", "20261001-120000-aaaa"]
    assert fresh.get("20261001-120000-aaaa")["best_channel"] == "xgboost"


def test_history_survives_a_new_process(tmp_path: Path) -> None:
    """A second interpreter reads the history file back (the history survives a restart of the app)."""
    path = tmp_path / "runs.sqlite3"
    RunHistory(path).record(make_run("20261001-120000-aaaa"))
    RunHistory(path).record(make_run("20261001-130000-bbbb", "2026-10-01T13:00:00+00:00"))
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from nids.history import RunHistory\n"
        "history = RunHistory(Path(sys.argv[1]))\n"
        "frame = history.list()\n"
        "print(json.dumps({'ids': frame['run_id'].tolist(), 'best': history.get('20261001-120000-aaaa')['best_channel']}))\n"
    )
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout.strip().splitlines()[-1])
    assert seen == {"ids": ["20261001-130000-bbbb", "20261001-120000-aaaa"], "best": "xgboost"}


def test_connections_are_closed_and_no_write_ahead_log_is_used(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite3"
    history = RunHistory(path)
    history.record(make_run())
    history.list()
    history.get("20261001-120000-ab12")
    assert not (tmp_path / "runs.sqlite3-wal").exists()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    path.unlink()  # Windows refuses this while any connection is still open
    assert not path.exists()


def test_best_reading_prefers_the_lower_channel_number_on_ties() -> None:
    assert best_reading(make_run(scores={"svm": 0.9, "forest": 0.9, "logreg": 0.7})) == ("forest", 0.9)
    assert best_reading(make_run(scores={"forest": None, "xgboost": None})) == (None, None)


def test_summary_of_a_cicids_run_lists_its_files() -> None:
    run = make_run(source="cicids", files=("Tuesday-WorkingHours.pcap_ISCX.csv", "Monday-WorkingHours.pcap_ISCX.csv"))
    line = run_summary(run, saved_path=Path("saved") / "x")
    assert line["source"] == "cicids"
    assert line["files"] == "Monday-WorkingHours.pcap_ISCX.csv, Tuesday-WorkingHours.pcap_ISCX.csv"
    assert line["saved_path"] == str(Path("saved") / "x")
    assert json.loads(line["settings_json"])["data_dir"] == "D:/flows"
    assert set(line) == set(COLUMNS)
