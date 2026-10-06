"""The 06 Sweep engine: seeded flow sources (replay and synthetic) and the live session's readings.

Source tests use small made-up arrays. Session tests use one tiny synthetic sample fitted once per mode with the
"test" profile (two channels), so the module runs in a few seconds. Nothing in the engine may fit a model.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from nids import simulate
from nids.data.prepare import DataRequest, PreparedDataset
from nids.data.synthetic import SYNTHETIC_CLASSES
from nids.models.train import FIT_CALLS, TrainingRun, TrainRequest
from nids.simulate import ReplaySource, SimulationSession, SyntheticSource
from tests.helpers import shared_fit, shared_sample

pytestmark = pytest.mark.unit
SEED = 23


# --------------------------------------------------------------------------------------------------------------
# Made-up held-out rows for the source tests
# --------------------------------------------------------------------------------------------------------------
def _held_out(n_normal: int = 800, attacks: dict[str, int] | None = None, n_features: int = 4,
              mode: str = "binary") -> dict[str, object]:
    """Arrays shaped like a run's held-out rows; every row's first feature is its row id (so rows are traceable)."""
    attacks = attacks if attacks is not None else {"DoS Hulk": 150, "PortScan": 40, "Bot": 10}
    detailed = ["BENIGN"] * n_normal + [name for name, count in attacks.items() for _ in range(count)]
    n = len(detailed)
    if mode == "binary":
        classes = ("Normal", "Attack")
        y = np.array([0 if d == "BENIGN" else 1 for d in detailed], dtype=np.int64)
    else:
        classes = ("BENIGN", *sorted(attacks))
        y = np.array([classes.index(d) for d in detailed], dtype=np.int64)
    row_ids = np.arange(n, dtype=np.int64) * 3 + 1000
    X = np.zeros((n, n_features), dtype=np.float32)
    X[:, 0] = row_ids
    return {"X_test": X, "y_test": y, "detailed_labels": np.asarray(detailed, dtype=str), "row_ids": row_ids,
            "classes": classes, "mode": mode}


def _source(**options: object) -> ReplaySource:
    arrays = _held_out(**{k: v for k, v in options.items() if k in ("n_normal", "attacks", "mode")})
    kwargs = {k: v for k, v in options.items() if k in ("attack_share", "mix", "seed")}
    return ReplaySource(**arrays, **kwargs)  # type: ignore[arg-type]


def _same(a: simulate.FlowBatch, b: simulate.FlowBatch) -> bool:
    return (np.array_equal(a.X, b.X) and np.array_equal(a.y_true, b.y_true) and np.array_equal(a.row_ids, b.row_ids)
            and np.array_equal(a.detailed, b.detailed))


@pytest.mark.parametrize("options", [{}, {"attack_share": 0.5}, {"mix": {"PortScan": 2.0, "Bot": 1.0}},
                                     {"attack_share": 0.3, "mix": {"DoS Hulk": 1.0, "Bot": 1.0}}],
                         ids=["natural", "share", "mix", "share+mix"])
def test_replay_is_deterministic_by_seed(options: dict[str, object]) -> None:
    sizes = (1, 7, 250, 1, 900)
    first, second, other = (_source(seed=s, **options) for s in (5, 5, 6))
    for size in sizes:
        a, b, c = first.draw(size), second.draw(size), other.draw(size)
        assert _same(a, b)
        assert a.X.dtype == np.float32 and len(a) == size
    assert not _same(first.draw(300), other.draw(300))
    first.reset()
    replay = _source(seed=5, **options)
    assert _same(first.draw(500), replay.draw(500))


def test_natural_replay_streams_every_test_row_once_per_pass() -> None:
    source = _source(seed=1)
    arrays = _held_out()
    batch = source.draw(source.size)
    assert sorted(batch.row_ids.tolist()) == sorted(np.asarray(arrays["row_ids"]).tolist())
    assert source.repeated == 0
    assert np.array_equal(batch.X[:, 0].astype(np.int64), batch.row_ids)  # features travel with their row
    source.draw(10)
    assert source.repeated == 10


@pytest.mark.parametrize("share", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_attack_share_is_held_within_three_points(share: float) -> None:
    source = _source(attack_share=share, seed=4)
    batch = source.draw(5_000)
    seen = float(np.mean(batch.y_true == 1))
    assert abs(seen - share) <= 0.03
    # The shuffled bags make every block of 1,000 flows exact to one flow.
    blocks = batch.y_true[:5_000].reshape(5, 1_000)
    assert np.all(np.abs(blocks.sum(axis=1) - share * 1_000) <= 1)
    # The pools ran dry (200 attack rows, 800 normal rows), so rows repeat, and that is counted.
    assert source.repeated > 0


def test_natural_share_follows_the_test_set() -> None:
    source = _source(seed=2)
    assert source.natural_share == pytest.approx(200 / 1_000)
    batch = source.draw(5_000)
    assert abs(float(np.mean(batch.y_true == 1)) - 0.2) <= 0.03


def test_mix_weights_the_attack_types() -> None:
    source = _source(attack_share=0.6, mix={"DoS Hulk": 3.0, "PortScan": 1.0, "Bot": 0.0}, seed=9)
    batch = source.draw(5_000)
    attacks = batch.detailed[batch.y_true == 1]
    assert "Bot" not in set(attacks.tolist())
    assert abs(float(np.mean(attacks == "DoS Hulk")) - 0.75) <= 0.03
    assert abs(float(np.mean(batch.y_true == 1)) - 0.6) <= 0.03
    # Natural share with a custom mix: the test-set share of attacks, split by the weights.
    mixed = _source(mix={"PortScan": 1.0, "Bot": 1.0}, seed=9).draw(5_000)
    assert abs(float(np.mean(mixed.y_true == 1)) - 0.2) <= 0.03
    assert abs(float(np.mean(mixed.detailed[mixed.y_true == 1] == "Bot")) - 0.5) <= 0.03


def test_multiclass_replay_keeps_class_codes_and_detailed_labels() -> None:
    source = _source(mode="multiclass", attack_share=0.5, seed=3)
    batch = source.draw(2_000)
    classes = source.classes
    assert classes[0] == "BENIGN"
    assert all(classes[code] == label for code, label in zip(batch.y_true, batch.detailed))
    assert source.attack_types == ("DoS Hulk", "PortScan", "Bot")  # most rows first
    assert sum(source.natural_mix.values()) == pytest.approx(1.0)


def test_bad_options_are_refused() -> None:
    with pytest.raises(ValueError, match="Unknown attack type"):
        _source(mix={"Heartbleed": 1.0})
    with pytest.raises(ValueError, match="weight above 0"):
        _source(mix={"Bot": 0.0})
    with pytest.raises(ValueError, match="between 0 and 1"):
        _source(attack_share=1.5)
    with pytest.raises(ValueError, match="no held-out rows"):
        _source(n_normal=0, attacks={})


def test_a_share_the_rows_cannot_give_is_explained() -> None:
    source = _source(n_normal=0, attacks={"DoS Hulk": 50}, attack_share=0.5, seed=1)
    assert source.effective_share == 1.0 and source.notes
    assert np.all(source.draw(100).y_true == 1)


# --------------------------------------------------------------------------------------------------------------
# Sessions on a tiny fitted run
# --------------------------------------------------------------------------------------------------------------
#: Seed of the sample and the fits below: the sample 01 Sample draws in the UI tests, so the binary fit is shared.
FIT_SEED = 42


@pytest.fixture(scope="module")
def prepared() -> PreparedDataset:
    """2,000 synthetic flows kept to 1,200 rows (six classes), shared with the other modules that use this sample."""
    return shared_sample(DataRequest(source="synthetic", synthetic_flows=2_000, row_budget=1_200, seed=FIT_SEED))


@pytest.fixture(scope="module")
def runs(prepared: PreparedDataset) -> dict[str, TrainingRun]:
    """Two small channels fitted once per mode (each fit made once per session, see shared_fit)."""
    return {mode: shared_fit(prepared, TrainRequest(mode=mode, profile="test", seed=FIT_SEED,  # type: ignore[arg-type]
                                                    channels=("xgboost", "logreg")))
            for mode in ("binary", "multiclass")}


def _fits() -> int:
    return sum(FIT_CALLS.values())


def _stream(session: SimulationSession, sizes: tuple[int, ...]) -> list[simulate.FlowEvent]:
    events: list[simulate.FlowEvent] = []
    for size in sizes:
        events.extend(session.step(size))
    return events


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["binary", "multiclass"])
@pytest.mark.parametrize("kind", ["replay", "synthetic"])
def test_live_readings_equal_scores_over_every_emitted_flow(runs: dict[str, TrainingRun], mode: str,
                                                            kind: str) -> None:
    run = runs[mode]
    fits = _fits()
    for channel in run.ok_channels():
        source = simulate.make_source(run, kind, attack_share=0.4, seed=2)  # type: ignore[arg-type]
        session = SimulationSession(run, channel, source, alert_threshold=0.8)
        events = _stream(session, (1, 40, 300, 7, 500))
        stats = session.stats
        truth = [e.true_label for e in events]
        verdicts = [e.predicted for e in events]
        assert stats.emitted == len(events) == 848 and stats.ticks == 5
        assert stats.live_accuracy == pytest.approx(accuracy_score(truth, verdicts), abs=1e-12)
        assert stats.live_balanced_accuracy == pytest.approx(balanced_accuracy_score(truth, verdicts), abs=1e-12)
        assert stats.correct == sum(e.correct for e in events)
        assert int(stats.confusion.sum()) == 848
        assert sum(stats.per_class_seen.values()) == 848
        frame = session.log_frame()
        assert accuracy_score(frame["true_label"], frame["predicted"]) == pytest.approx(stats.live_accuracy)
        assert [e.seq for e in events] == list(range(1, 849))
    assert _fits() == fits  # streaming never fits


@pytest.mark.integration
def test_replayed_rows_are_held_out_rows_with_their_true_labels(runs: dict[str, TrainingRun]) -> None:
    run = runs["multiclass"]
    data = run.data
    session = SimulationSession(run, "xgboost", ReplaySource.from_run(run, attack_share=0.7, seed=4),
                                alert_threshold=0.9)
    events = _stream(session, (500, 500, 500))
    ids = np.array([e.row_id for e in events], dtype=np.int64)
    assert set(ids.tolist()) <= set(data.test_rows.tolist())
    assert not set(ids.tolist()) & set(data.train_rows.tolist())
    where = {int(r): i for i, r in enumerate(data.test_rows)}
    for event in events[:200]:
        i = where[int(event.row_id)]  # type: ignore[arg-type]
        assert event.true_label == data.classes[int(data.y_test[i])]
        assert event.detail == str(data.detailed_test_labels[i])
    # Scoring a replayed row gives the very verdict the fit stored for that test row.
    stored = run.channels["xgboost"].y_pred
    assert stored is not None
    rows = [where[int(event.row_id)] for event in events]  # type: ignore[arg-type]
    assert [event.predicted for event in events] == [data.classes[int(stored[i])] for i in rows]


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_alert_rule(runs: dict[str, TrainingRun], mode: str) -> None:
    run = runs[mode]
    session = SimulationSession(run, "logreg", simulate.make_source(run, "replay", attack_share=0.5, seed=8),
                                alert_threshold=0.85)
    events = _stream(session, (400, 400))
    normal = simulate.normal_index(run.classes)
    assert normal is not None
    for event in events:
        expected = event.attack_probability >= 0.85 and event.predicted != run.classes[normal]
        assert event.alert == expected
        assert 0.0 <= event.attack_probability <= 1.0
    assert session.stats.alerts_total == sum(e.alert for e in events) > 0
    # The attack probability is P(Attack) in binary mode and 1 - P(BENIGN) in multi-class mode.
    batch_rows = np.array([np.flatnonzero(run.data.test_rows == e.row_id)[0] for e in events[:50]])
    proba = np.asarray(run.channels["logreg"].estimator.predict_proba(run.data.X_test[batch_rows]))
    expected_p = proba[:, 1] if mode == "binary" else 1.0 - proba[:, 0]
    assert np.allclose([e.attack_probability for e in events[:50]], expected_p, atol=1e-6)
    # A threshold of 0 turns every attack verdict into an alert; the threshold can change between ticks.
    session.alert_threshold = 0.0
    later = session.step(300)
    assert all(e.alert == (e.predicted != run.classes[normal]) for e in later)
    with pytest.raises(ValueError):
        session.alert_threshold = 1.5


@pytest.mark.integration
def test_feed_alert_and_log_caps(runs: dict[str, TrainingRun], monkeypatch: pytest.MonkeyPatch) -> None:
    run = runs["binary"]
    monkeypatch.setattr(simulate, "LOG_CAP", 1_000)
    session = SimulationSession(run, "xgboost", simulate.make_source(run, "replay", attack_share=0.8, seed=1),
                                alert_threshold=0.5, feed_size=30, max_alerts=10)
    events = _stream(session, (250, 250, 250, 250, 333))
    feed = session.feed
    assert len(feed) == 30
    assert [e.seq for e in feed] == list(range(1_333, 1_303, -1))  # newest first
    assert session.stats.alerts_total > 10 and len(session.alerts) == 10
    assert [e.seq for e in session.alerts] == sorted((e.seq for e in events if e.alert), reverse=True)[:10]
    log = session.log_frame()
    assert len(log) == 1_000 and log["seq"].iloc[0] == 334 and log["seq"].iloc[-1] == 1_333
    assert list(log.columns) == list(simulate.LOG_COLUMNS)
    timeline = session.timeline
    assert list(timeline["tick"]) == [1, 2, 3, 4, 5] and int(timeline["flows"].sum()) == 1_333
    assert (timeline["normal"] + timeline["attack_predicted"] == timeline["flows"]).all()
    assert int(timeline["alerts"].sum()) == session.stats.alerts_total


@pytest.mark.integration
def test_reset_clears_the_readings_and_rewinds_the_stream(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    for kind in ("replay", "synthetic"):
        session = SimulationSession(run, "logreg", simulate.make_source(run, kind, seed=3),  # type: ignore[arg-type]
                                    alert_threshold=0.9)
        first = _stream(session, (60, 60))
        session.reset()
        stats = session.stats
        assert stats.emitted == stats.ticks == stats.alerts_total == 0 and not session.feed and not session.alerts
        assert np.isnan(stats.live_accuracy) and int(stats.confusion.sum()) == 0
        assert session.timeline.empty and session.log_frame().empty
        assert _stream(session, (60, 60)) == first  # same seed, same stream, same readings
        twin = SimulationSession(run, "logreg", simulate.make_source(run, kind, seed=3),  # type: ignore[arg-type]
                                 alert_threshold=0.9)
        assert _stream(twin, (60, 60)) == first


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_synthetic_source_yields_the_run_columns_and_classes(runs: dict[str, TrainingRun], mode: str) -> None:
    run = runs[mode]
    source = SyntheticSource.from_run(run, attack_share=0.5, seed=12)
    assert source.feature_names == run.data.feature_names
    assert source.classes == run.classes
    batch = source.draw(3_000)
    assert batch.X.shape == (3_000, len(run.data.feature_names)) and batch.X.dtype == np.float32
    assert set(batch.detailed.tolist()) <= set(SYNTHETIC_CLASSES)
    assert np.all(batch.row_ids == -1)
    assert abs(float(np.mean(batch.detailed != "BENIGN")) - 0.5) <= 0.03
    if mode == "binary":
        assert np.array_equal(batch.y_true, (batch.detailed != "BENIGN").astype(np.int64))
    else:
        assert all(run.classes[code] == label for code, label in zip(batch.y_true, batch.detailed))
    # The run's bad-value strategy holds (drop: no infinite or missing values reach the channel).
    if run.data_request.nonfinite_strategy == "drop":
        assert np.isfinite(batch.X).all()
    # Seeded: the same seed gives the same flows, another seed other flows.
    again = SyntheticSource.from_run(run, attack_share=0.5, seed=12).draw(3_000)
    assert _same(batch, again)
    assert not _same(batch, SyntheticSource.from_run(run, attack_share=0.5, seed=13).draw(3_000))
    # A custom mix: one attack type only.
    only = source.attack_types[0]
    single = SyntheticSource.from_run(run, attack_share=1.0, mix={only: 1.0}, seed=1).draw(400)
    assert set(single.detailed.tolist()) == {only}
    # Natural share: the share the run's own sample was generated with.
    assert SyntheticSource.from_run(run).effective_share == run.data_request.synthetic_attack_share


@pytest.mark.integration
def test_real_data_runs_only_replay_real_held_out_rows(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    assert simulate.stream_kinds(run) == ["synthetic", "replay"]
    real_request = replace(run.data_request, source="cicids", files=("Monday-WorkingHours.pcap_ISCX.csv",))
    real = replace(run, data_request=real_request)
    assert simulate.stream_kinds(real) == ["replay"]
    with pytest.raises(ValueError, match="synthetic data"):
        simulate.make_source(real, "synthetic")
    data = run.data
    empty = replace(data, X_test=np.empty((0, data.n_features), dtype=np.float32), y_test=np.empty(0, dtype=np.int64),
                    test_rows=np.empty(0, dtype=np.int64), detailed_test_labels=np.empty(0, dtype=str))
    assert simulate.stream_kinds(replace(real, data=empty)) == []
    assert simulate.stream_kinds(replace(run, data=empty)) == ["synthetic"]  # generated flows need no rows
    with pytest.raises(ValueError, match="no held-out rows"):
        simulate.make_source(replace(real, data=empty), "replay")


@pytest.mark.integration
def test_session_refuses_a_channel_without_a_model(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    with pytest.raises(ValueError, match="no fitted model"):
        SimulationSession(run, "forest", simulate.make_source(run, "replay"), alert_threshold=0.9)
    with pytest.raises(ValueError):
        SimulationSession(run, "xgboost", simulate.make_source(run, "replay"),
                          alert_threshold=0.9).step(0)


@pytest.mark.integration
def test_summary_and_log_hold_plain_values_and_no_features(runs: dict[str, TrainingRun]) -> None:
    run = runs["binary"]
    session = SimulationSession(run, "xgboost", simulate.make_source(run, "synthetic", seed=5), alert_threshold=0.9)
    _stream(session, (100, 100))
    summary = session.summary()
    assert summary["flows"] == 200 and summary["ticks"] == 2 and summary["source_kind"] == "synthetic"
    assert summary["channel_label"] == "CH2 XGBoost" and summary["run_id"] == run.run_id
    log = session.log_frame()
    assert not set(run.data.feature_names) & set(log.columns)
    assert log["row_id"].isna().all()  # generated flows have no test-row id
    assert pd.api.types.is_string_dtype(log["true_label"]) and log["alert"].dtype == bool
