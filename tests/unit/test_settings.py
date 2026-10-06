"""Settings persistence and data-folder resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

import nids.settings as s
from nids.schema import EXPECTED_FILES

pytestmark = pytest.mark.unit


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    loaded = s.load_settings(tmp_path / "nope.json")
    assert loaded == s.AppSettings()
    assert loaded.row_budget == 200_000 and loaded.svm_cap == 20_000 and loaded.min_class_count == 50


def test_round_trip_and_validation(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    s.save_settings(s.AppSettings(data_dir="  ", svm_cap=999_999, test_share=0.9, nonfinite_strategy="bogus"), target)
    loaded = s.load_settings(target)
    assert loaded.data_dir is None
    assert loaded.svm_cap == s.SVM_CAP_RANGE[1]
    assert loaded.test_share == 0.5
    assert loaded.nonfinite_strategy == "drop"


def test_corrupt_file_gives_defaults(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text("{not json", encoding="utf-8")
    assert s.load_settings(target) == s.AppSettings()
    target.write_text('{"row_budget": "lots"}', encoding="utf-8")
    assert s.load_settings(target) == s.AppSettings()


def test_unknown_keys_ignored(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text('{"seed": 7, "mystery": true}', encoding="utf-8")
    assert s.load_settings(target).seed == 7


def _make_folder(root: Path) -> Path:
    folder = root / "csvs"
    (folder / "nested").mkdir(parents=True)
    (folder / EXPECTED_FILES[0].name).write_text("x\n", encoding="utf-8")
    (folder / "extra.csv").write_text("x\n", encoding="utf-8")
    (folder / "nested" / EXPECTED_FILES[1].name).write_text("x\n", encoding="utf-8")
    (folder / "notes.txt").write_text("x\n", encoding="utf-8")
    return folder


def test_setting_beats_environment(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    res = s.resolve_data_dir(s.AppSettings(data_dir=str(folder)), environ={s.ENV_DATA_DIR: str(other)})
    assert res.source == "setting" and res.path == folder


def test_environment_used_when_setting_empty(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    res = s.resolve_data_dir(s.AppSettings(), environ={s.ENV_DATA_DIR: str(folder)})
    assert res.source == "env" and res.path == folder and res.usable


def test_csv_discovery_is_not_recursive(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path)
    res = s.resolve_data_dir(s.AppSettings(data_dir=str(folder)), environ={})
    assert [p.name for p in res.found] == [EXPECTED_FILES[0].name]
    assert [p.name for p in res.other_csvs] == ["extra.csv"]
    assert EXPECTED_FILES[1].name in res.missing


def test_no_folder_means_synthetic(tmp_path: Path) -> None:
    res = s.resolve_data_dir(s.AppSettings(), environ={})
    assert res.source == "none" and res.path is None and not res.usable


def test_missing_folder_reports_problem(tmp_path: Path) -> None:
    res = s.resolve_data_dir(s.AppSettings(data_dir=str(tmp_path / "absent")), environ={})
    assert res.path is None and res.problem and "not found" in res.problem


def test_empty_folder_reports_problem(tmp_path: Path) -> None:
    res = s.resolve_data_dir(s.AppSettings(data_dir=str(tmp_path)), environ={})
    assert res.path == tmp_path and not res.usable and res.problem
