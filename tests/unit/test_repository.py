"""Repository hygiene: the app's name, licence, ignore rules, and theme config that agrees with nids.theme."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import nids
from nids import theme

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_the_name_is_set_in_one_place_and_the_project_files_agree() -> None:
    assert (nids.APP_NAME, nids.APP_FULL_NAME) == ("NIDS", "Network Intrusion Detection System")
    assert nids.APP_TITLE == "NIDS — Network Intrusion Detection System"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == nids.__name__ == "nids"
    assert project["version"] == nids.__version__ and project["description"] == nids.TAGLINE
    assert f'aria-label="{nids.APP_NAME}"' in (ROOT / "static" / "nids-icon.svg").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(f"# {nids.APP_TITLE}\n\n*{nids.TAGLINE}*\n")


def test_license_names_owner_and_year() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text and "mohsin-moallim" in text and "2026" in text


@pytest.mark.parametrize(
    "pattern", [".venv/", "__pycache__/", "*.csv", "saved_models/*", "run_history/*", "local_settings.json", "data/*"]
)
def test_gitignore_covers(pattern: str) -> None:
    assert pattern in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_readme_design_and_data_notes_exist() -> None:
    for name in ("README.md", "DESIGN.md", "data/README.md", "requirements.txt"):
        assert (ROOT / name).is_file(), name
    assert "Sharafaldin" in (ROOT / "data" / "README.md").read_text(encoding="utf-8")


def test_streamlit_theme_matches_tokens() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    for mode in ("light", "dark"):
        section = config["theme"][mode]
        tokens = theme.palette(mode)  # type: ignore[arg-type]
        assert section["primaryColor"] == tokens.primary
        assert section["backgroundColor"] == tokens.background
        assert section["secondaryBackgroundColor"] == tokens.surface
        assert section["textColor"] == tokens.text
        assert section["redColor"] == tokens.attack
        assert section["blueColor"] == tokens.benign
        assert list(section["chartSequentialColors"]) == list(theme.SEQUENTIAL[mode])
        assert list(section["chartDivergingColors"]) == list(theme.DIVERGING[mode])
        assert len(section["chartSequentialColors"]) == 10 == len(section["chartDivergingColors"])
        assert list(section["chartCategoricalColors"]) == [c.colour(mode) for c in theme.CHANNELS]  # type: ignore[arg-type]


def test_font_files_are_bundled_with_licences() -> None:
    fonts = ROOT / "static" / "fonts"
    for filename in theme.FONT_FILES.values():
        assert (fonts / filename).is_file(), filename
    assert len(list(fonts.glob("OFL-*.txt"))) == 3
