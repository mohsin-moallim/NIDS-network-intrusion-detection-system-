"""Headless checks of the 01 Sample station: synthetic fallback, drawing a sample, and no accidental re-runs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from graticule import theme, viz
from graticule.data import prepare
from graticule.data.synthetic import SYNTHETIC_CLASSES
from graticule.settings import AppSettings, save_settings
from tests.helpers import fake_generator, flow_row, make_rows, with_values, write_cic_csv
from ui import data_cache, state

pytestmark = pytest.mark.ui
MON = "Monday-WorkingHours.pcap_ISCX.csv"
TUE = "Tuesday-WorkingHours.pcap_ISCX.csv"
WED = "Wednesday-workingHours.pcap_ISCX.csv"


def _sample_script() -> None:
    """AppTest script body: render 01 Sample through the shell."""
    from ui.shell import main

    main(force_key="sample")


def _app() -> AppTest:
    return AppTest.from_function(_sample_script, default_timeout=60)


@pytest.fixture
def prepare_calls(monkeypatch: pytest.MonkeyPatch) -> list[prepare.DataRequest]:
    """Record every call of prepare_dataset (the real function still runs)."""
    calls: list[prepare.DataRequest] = []
    real = prepare.prepare_dataset

    def spy(request: prepare.DataRequest, **kwargs: object) -> prepare.PreparedDataset:
        calls.append(request)
        return real(request, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(prepare, "prepare_dataset", spy)
    return calls


def _subheaders(at: AppTest) -> list[str]:
    return [s.value for s in at.subheader]


def test_without_a_folder_only_synthetic_flows_are_offered() -> None:
    at = _app().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio[0].options == ["Synthetic flows"]
    notes = " ".join(m.value for m in at.markdown)
    assert "synthetic flows" in notes and "Bench" in notes
    assert "Sample sheet" not in _subheaders(at)


def test_drawing_a_sample_from_a_folder(tmp_path: Path, prepare_calls: list[prepare.DataRequest]) -> None:
    folder = tmp_path / "data"
    write_cic_csv(folder / MON, make_rows({"BENIGN": 60}))
    write_cic_csv(folder / TUE, make_rows({"BENIGN": 20, "FTP-Patator": 40}, start=100))
    save_settings(AppSettings(data_dir=str(folder)))

    at = _app().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio[0].options == ["CIC-IDS2017 files", "Synthetic flows"]
    assert set(at.multiselect(key="smp_files").value) == {MON, TUE}
    at.number_input(key="smp_budget").set_value(50_000)
    at.button(key="smp_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(prepare_calls) == 1
    assert prepare_calls[0].files == (MON, TUE) and prepare_calls[0].row_budget == 50_000
    assert "Sample sheet" in _subheaders(at)
    assert not at.warning  # two classes: no single-class warning
    assert len(at.dataframe) == 2  # class table and per-file table

    # Unrelated interaction afterwards: nothing is prepared again, the sheet stays.
    at.number_input(key="smp_budget").set_value(1_000).run()
    at.radio[0].set_value("Synthetic flows").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(prepare_calls) == 1
    assert "Sample sheet" in _subheaders(at)


def test_benign_only_sample_shows_the_single_class_warning(tmp_path: Path,
                                                           prepare_calls: list[prepare.DataRequest]) -> None:
    folder = tmp_path / "data"
    write_cic_csv(folder / MON, make_rows({"BENIGN": 30}))
    save_settings(AppSettings(data_dir=str(folder)))
    at = _app().run()
    at.button(key="smp_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    warnings = [w.value for w in at.warning]
    assert any(w.startswith("Only one class in this sample: every row is BENIGN") for w in warnings)


def test_synthetic_draw(monkeypatch: pytest.MonkeyPatch, prepare_calls: list[prepare.DataRequest]) -> None:
    monkeypatch.setattr(prepare, "_load_generator", lambda: fake_generator)
    at = _app().run()
    at.number_input(key="smp_syn_flows").set_value(4_000)
    at.button(key="smp_syn_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(prepare_calls) == 1 and prepare_calls[0].source == "synthetic"
    assert "Sample sheet" in _subheaders(at)
    at.slider(key="smp_syn_share").set_value(0.5).run()
    assert len(prepare_calls) == 1


def test_synthetic_draw_with_the_real_generator(prepare_calls: list[prepare.DataRequest]) -> None:
    """No folder: the real generator feeds the sheet, the result is stored for later stations, the station is ticked."""
    at = _app().run()
    assert at.radio[0].options == ["Synthetic flows"]
    at.number_input(key="smp_syn_flows").set_value(4_000)
    at.number_input(key="smp_syn_budget").set_value(3_000)
    at.radio(key="smp_syn_strategy").set_value("recompute")
    at.button(key="smp_syn_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(prepare_calls) == 1
    request = prepare_calls[0]
    assert request.source == "synthetic" and request.synthetic_flows == 4_000
    assert request.row_budget == 3_000 and request.nonfinite_strategy == "recompute"
    assert "Sample sheet" in _subheaders(at)
    assert not at.warning and not at.error
    assert len(at.dataframe) == 2
    classes = at.dataframe[0].value["Class"].tolist()
    assert classes[0] == f"{theme.GLYPH_NORMAL} BENIGN"
    assert {c.split(" ", 1)[1] for c in classes} == set(SYNTHETIC_CLASSES)
    assert all(c.startswith(theme.GLYPH_ATTACK) for c in classes[1:])
    sources = at.dataframe[1].value
    assert sources["Session"].tolist() == ["generated"] and sources["Engine"].tolist() == ["generator"]
    notes = " ".join(m.value for m in at.markdown)
    assert "Rates were rebuilt" in notes and "Degenerate columns" in notes
    stored = at.session_state[state.PREPARED]
    assert stored.request == request and stored.rows_sampled == 3_000
    assert state.DONE in at.session_state and "sample" in at.session_state[state.DONE]

    # Changing an option afterwards redraws the stored sheet without preparing again.
    at.radio(key="smp_syn_strategy").set_value("drop").run()
    assert not at.exception and len(prepare_calls) == 1
    assert "Sample sheet" in _subheaders(at)


def test_an_unusable_folder_falls_back_to_synthetic_flows(tmp_path: Path) -> None:
    save_settings(AppSettings(data_dir=str(tmp_path / "missing")))
    at = _app().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio[0].options == ["Synthetic flows"]
    notes = " ".join(m.value for m in at.markdown)
    assert "Folder not found" in notes and "Synthetic flows are available" in notes

    empty = tmp_path / "empty"
    empty.mkdir()
    save_settings(AppSettings(data_dir=str(empty)))
    at = _app().run()
    assert at.radio[0].options == ["Synthetic flows"]
    assert "contains no CSV files" in " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_class_chart_uses_shapes_and_theme_colours(mode: str) -> None:
    before = {"BENIGN": 400_000, "DoS Hulk": 150_000, "Heartbleed": 11}
    after = {"BENIGN": 120_000, "DoS Hulk": 40_000, "Heartbleed": 11}
    spec = viz.class_distribution_chart(before, after, mode).to_dict()  # type: ignore[arg-type]
    tokens = theme.palette(mode)  # type: ignore[arg-type]
    layers = spec["layer"]
    point_layers = [layer for layer in layers if layer["mark"]["type"] == "point"]
    assert len(point_layers) == 2
    shapes = point_layers[1]["encoding"]["shape"]["scale"]
    assert shapes["domain"] == ["Normal", "Attack"] and shapes["range"] == ["circle", "diamond"]
    colours = point_layers[1]["encoding"]["color"]["scale"]["range"]
    assert colours == [tokens.benign, tokens.attack]
    assert layers[0]["encoding"]["x"]["scale"]["type"] == "log"
    assert spec["config"]["axis"]["labelFont"] == theme.FONT_MONO
    data = next(iter(spec["datasets"].values()))
    assert [row["class"] for row in data] == ["BENIGN", "DoS Hulk", "Heartbleed"]
    # The written count is the sampled one, so it sits beside the filled (sampled) mark, on its left.
    text = next(layer for layer in layers if layer["mark"]["type"] == "text")
    assert text["encoding"]["x"]["field"] == "after" and text["encoding"]["text"]["field"] == "label"
    assert text["mark"]["align"] == "right" and text["mark"]["dx"] < 0
    domain_low = layers[0]["encoding"]["x"]["scale"]["domain"][0]
    assert domain_low < 11 / 1.8  # room left of the smallest mark for its label


def test_class_chart_exports_to_png() -> None:
    chart = viz.class_distribution_chart({"BENIGN": 900, "Bot": 12}, {"BENIGN": 300, "Bot": 12}, "light")
    png = viz.to_png(chart, scale=1, background=theme.LIGHT.surface)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 2_000


def test_a_bad_file_is_reported_not_raised(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    write_cic_csv(folder / "odd.csv", make_rows({"BENIGN": 5}), drop_columns=["Idle Min"])
    save_settings(AppSettings(data_dir=str(folder)))
    at = _app().run()
    at.button(key="smp_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("lacks 1 of the 77" in e.value for e in at.error)
    assert "Sample sheet" not in _subheaders(at)


def _markdown(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_sheet_explains_every_repair_and_the_readings_add_up(tmp_path: Path) -> None:
    """Empty labels, a mismatched repeated column, an unknown column and a text cell all reach the sheet."""
    folder = tmp_path / "data"
    rows = make_rows({"BENIGN": 30, "DoS Hulk": 36}, start=50)
    rows[4] = with_values(rows[4], {"Flow IAT Mean": "abc"})
    rows += [flow_row("", 900), flow_row("", 901)]
    write_cic_csv(folder / TUE, rows, duplicate_mismatch=True, extra_columns={"Timestamp": "4/7/2017 9:00"})
    save_settings(AppSettings(data_dir=str(folder)))
    at = _app().run()
    at.selectbox(key="smp_policy").set_value("drop")
    at.button(key="smp_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    warnings = " ".join(w.value for w in at.warning)
    assert "repeated column Fwd Header Length does not match its first copy" in warnings
    notes = _markdown(at)
    assert "**File fixes.**" in notes and "Timestamp" in notes and "text cell" in notes
    assert "2 rows with an empty label dropped" in notes
    # Reading cards come from the shared component (CSS classes, mono values) and include the empty labels.
    assert 'class="g-cards"' in notes and '<div class="g-card-label">Empty labels</div>' in notes
    assert '<div class="g-card-label">Conflicting rows</div>' in notes
    ds = at.session_state[state.PREPARED]
    readings = {r["Reading"]: r for r in ds.summary_rows()}
    removed = sum(int(r["Value"]) for r in readings.values() if str(r["Note"]).startswith("rows removed"))
    assert readings["Rows read"]["Value"] - removed == readings["Rows kept"]["Value"]
    sources = at.dataframe[1].value
    assert "differed from the first copy" in sources["File fixes"].iloc[0]


def test_a_sample_with_no_rows_left_is_an_error_not_a_crash(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    rows = [with_values(flow_row("BENIGN", k), {"Flow Bytes/s": float("inf")}) for k in range(10)]
    write_cic_csv(folder / MON, rows)
    save_settings(AppSettings(data_dir=str(folder)))
    at = _app().run()
    at.button(key="smp_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("No rows are left to sample" in e.value and "impute or rebuild" in e.value for e in at.error)
    assert "Sample sheet" not in _subheaders(at)
    assert state.PREPARED not in at.session_state
    assert "sample" not in (at.session_state[state.DONE] if state.DONE in at.session_state else set())
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_a_failed_draw_says_the_sheet_is_the_previous_sample(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    write_cic_csv(folder / TUE, make_rows({"BENIGN": 30, "FTP-Patator": 30}))
    write_cic_csv(folder / "odd.csv", make_rows({"BENIGN": 5}), drop_columns=["Idle Min"])
    save_settings(AppSettings(data_dir=str(folder)))
    at = _app().run()
    at.button(key="smp_draw").click().run()
    assert not at.exception and "Sample sheet" in _subheaders(at)
    assert any(t.value.startswith("Sample drawn in") for t in at.toast)
    fingerprint = at.session_state[state.PREPARED].fingerprint
    at.checkbox(key="smp_all").check()
    at.number_input(key="smp_budget").set_value(5_000)
    at.button(key="smp_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("lacks 1 of the 77" in e.value for e in at.error)
    assert "Sample sheet" in _subheaders(at)
    notes = " ".join(i.value for i in at.info)
    assert "The new draw failed" in notes and fingerprint[:12] in notes
    assert at.session_state[state.PREPARED].fingerprint == fingerprint


def test_the_form_keeps_the_last_draw_when_the_source_is_switched_back(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    write_cic_csv(folder / MON, make_rows({"BENIGN": 40}))
    write_cic_csv(folder / TUE, make_rows({"BENIGN": 20, "FTP-Patator": 40}, start=100))
    write_cic_csv(folder / WED, make_rows({"BENIGN": 20, "DoS Hulk": 40}, start=300))
    save_settings(AppSettings(data_dir=str(folder)))
    at = _app().run()
    assert at.multiselect(key="smp_files").value == [MON, TUE, WED]
    at.multiselect(key="smp_files").set_value([TUE, WED])
    at.number_input(key="smp_budget").set_value(30_000)
    at.radio(key="smp_strategy").set_value("impute")
    at.selectbox(key="smp_policy").set_value("majority")
    at.checkbox(key="smp_merge").check()
    at.button(key="smp_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    request = at.session_state[state.PREPARED].request
    assert request.files == (TUE, WED) and request.row_budget == 30_000

    at.radio[0].set_value("Synthetic flows").run()
    at.radio[0].set_value("CIC-IDS2017 files").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.multiselect(key="smp_files").value == [TUE, WED]
    assert at.number_input(key="smp_budget").value == 30_000
    assert at.radio(key="smp_strategy").value == "impute"
    assert at.selectbox(key="smp_policy").value == "majority"
    assert at.checkbox(key="smp_merge").value is True
    captions = " ".join(c.value for c in at.caption)
    assert "Options repeat your last draw" in captions
    assert "2 CIC-IDS2017 files" in captions  # the sheet below still matches the form


@pytest.fixture
def cache_spies(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, list[str]]]:
    """Empty the file caches and count the real reads and per-file stages computed behind them."""
    calls: dict[str, list[str]] = {"read": [], "stage": []}
    real_read, real_stage = data_cache.read_source_file, data_cache.stage_rows

    def read(path: Path) -> object:
        calls["read"].append(Path(path).name)
        return real_read(path)

    def stage(frame: object, report: object, strategy: str, **options: object) -> object:
        calls["stage"].append(strategy)
        return real_stage(frame, report, strategy, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(data_cache, "read_source_file", read)
    monkeypatch.setattr(data_cache, "stage_rows", stage)
    data_cache.clear_file_cache()
    yield calls
    data_cache.clear_file_cache()


def test_files_are_read_once_for_every_strategy(tmp_path: Path, cache_spies: dict[str, list[str]]) -> None:
    folder = tmp_path / "data"
    write_cic_csv(folder / MON, make_rows({"BENIGN": 40}))
    write_cic_csv(folder / TUE, make_rows({"BENIGN": 20, "FTP-Patator": 40}, start=100))
    save_settings(AppSettings(data_dir=str(folder)))
    at = _app().run()
    at.button(key="smp_draw").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert sorted(cache_spies["read"]) == [MON, TUE] and cache_spies["stage"] == ["drop", "drop"]
    assert not any(e.endswith("(cached)") for e in at.dataframe[1].value["Engine"])

    at.button(key="smp_draw").click().run()  # the same draw again: nothing is read or staged
    assert len(cache_spies["read"]) == 2 and len(cache_spies["stage"]) == 2
    assert all(e.endswith("(cached)") for e in at.dataframe[1].value["Engine"])

    at.radio(key="smp_strategy").set_value("impute")
    at.button(key="smp_draw").click().run()  # another strategy: staged again from the same reads
    assert len(cache_spies["read"]) == 2 and cache_spies["stage"][2:] == ["impute", "impute"]

    at.radio(key="smp_strategy").set_value("drop")
    at.button(key="smp_draw").click().run()  # back to the first strategy: still cached
    assert len(cache_spies["read"]) == 2 and len(cache_spies["stage"]) == 4
    assert not at.exception, [e.value for e in at.exception]

    at.button(key="smp_release").click().run()
    assert any("released" in t.value for t in at.toast)
    at.button(key="smp_draw").click().run()
    assert len(cache_spies["read"]) == 4 and len(cache_spies["stage"]) == 6
    assert not at.exception, [e.value for e in at.exception]
