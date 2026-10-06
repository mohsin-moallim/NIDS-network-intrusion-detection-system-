"""The README and its companion notes stay true to the repository: tree, links, anchors, screenshots, citation."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DATA_NOTE = ROOT / "data" / "README.md"
SHOTS_NOTE = ROOT / "docs" / "screenshots" / "README.md"
DOCS = (README, DATA_NOTE, SHOTS_NOTE)


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _tree_paths() -> list[str]:
    """Repository-relative paths named in the README's project tree, parents joined to children."""
    block = _readme().split("## Project structure", 1)[1].split("```text", 1)[1].split("```", 1)[0]
    paths: list[str] = []
    stack: list[str] = []
    for line in block.splitlines():
        match = re.match(r"^((?:│   |    )*)(?:├── |└── )(\S+)", line)
        if match is None:
            continue
        depth = len(match.group(1)) // 4
        name = match.group(2)
        stack = stack[:depth]
        paths.append("".join(stack) + name)
        stack.append(name if name.endswith("/") else name + "/")
    return paths


def test_every_path_in_the_project_tree_exists() -> None:
    paths = _tree_paths()
    assert len(paths) > 30
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert missing == []


def test_project_tree_names_every_core_and_ui_module() -> None:
    listed = set(_tree_paths())
    expected = {
        f"{package}/{item.name}" + ("/" if item.is_dir() else "")
        for package in ("nids", "ui")
        for item in (ROOT / package).iterdir()
        if item.name not in {"__pycache__", "__init__.py"} or (package == "nids" and item.name == "__init__.py")
        if item.is_dir() or item.suffix == ".py"
    }
    assert expected - listed == set()


def test_relative_links_resolve_and_screenshot_names_agree() -> None:
    shots = set(re.findall(r"`([\w-]+\.png)`", SHOTS_NOTE.read_text(encoding="utf-8")))
    linked_images: set[str] = set()
    for doc in DOCS:
        for target in re.findall(r"\]\(([^)\s]+)\)", doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            if target.endswith(".png"):
                linked_images.add(Path(target).name)
                assert (doc.parent / target).resolve().parent == SHOTS_NOTE.parent.resolve(), target
                continue
            assert (doc.parent / target).resolve().exists(), f"{doc.name} -> {target}"
    assert linked_images == shots
    assert len(shots) == 9  # one per station and utility shown, plus 02 Fit while it fits


def test_contents_anchors_match_headings() -> None:
    text = _readme()
    headings = {
        re.sub(r"[^a-z0-9 -]", "", heading.strip().lower()).replace(" ", "-")
        for heading in re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    }
    anchors = re.findall(r"\]\(#([^)]+)\)", text)
    assert anchors and set(anchors) <= headings


def test_citation_is_identical_in_readme_and_data_note() -> None:
    quote = DATA_NOTE.read_text(encoding="utf-8").split("> Iman", 1)[1].split("\n\n", 1)[0]
    assert "Sharafaldin" in quote and "ICISSP" in quote
    assert _readme().count("> Iman" + quote) == 2


def test_readme_markers_and_commands_match_the_project() -> None:
    text = _readme()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = {entry.split(":", 1)[0] for entry in config["tool"]["pytest"]["ini_options"]["markers"]}
    for marker in re.findall(r"pytest -q -m (\w+)", text):
        assert marker in markers, marker
    assert "py -3.13 -m venv .venv" in text
    assert r".\.venv\Scripts\python.exe -m pip install -r requirements.txt" in text
    assert r".\.venv\Scripts\python.exe -m streamlit run app.py" in text
    assert '"--data-dir=' in text and '"--data-dir=' in DATA_NOTE.read_text(encoding="utf-8")
    for name in ("requirements.txt", "requirements-lock.txt", "app.py", "scripts/bench.py"):
        assert (ROOT / name).is_file(), name


def test_docs_carry_no_emoji() -> None:
    for doc in DOCS:
        odd = {ch for ch in doc.read_text(encoding="utf-8") if ord(ch) >= 0x1F000 or 0x2600 <= ord(ch) <= 0x27BF}
        assert odd <= {"✓"}, (doc.name, odd)
