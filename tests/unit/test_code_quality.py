"""Structural rules: the core never imports Streamlit, and every public API has type hints and a docstring.

The core's sources are checked on every run; the import of every core module in a fresh interpreter, which also
catches an indirect import but takes a few seconds, is marked ``slow``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import graticule
import ui

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _modules(package: ModuleType) -> list[ModuleType]:
    names = [package.__name__] + [
        info.name for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + ".")
    ]
    return [importlib.import_module(name) for name in names]


def test_core_sources_never_import_streamlit_or_the_ui() -> None:
    """No module under graticule/ imports Streamlit, or the ui package built on it (read from the sources)."""
    problems: list[str] = []
    for path in sorted((ROOT / "graticule").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names = [node.value] if node.value.split(".")[0] == "streamlit" else []
            else:
                continue
            problems += [f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)} names {name}"
                         for name in names if name.split(".")[0] in ("streamlit", "ui")]
    assert problems == []


@pytest.mark.slow  # a fresh interpreter importing every core module (scikit-learn, XGBoost...) takes a few seconds
def test_core_never_imports_streamlit() -> None:
    """Import every core module in a fresh interpreter and check Streamlit was not pulled in."""
    code = (
        "import importlib, pkgutil, sys, graticule\n"
        "for m in pkgutil.walk_packages(graticule.__path__, prefix='graticule.'):\n"
        "    importlib.import_module(m.name)\n"
        "bad = sorted(n for n in sys.modules if n == 'streamlit' or n.startswith('streamlit.'))\n"
        "print('LEAK' if bad else 'CLEAN')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("CLEAN")


def _public_callables(module: ModuleType) -> list[tuple[str, object]]:
    found: list[tuple[str, object]] = []
    for name, obj in vars(module).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj):
            found.append((f"{module.__name__}.{name}", obj))
        elif inspect.isclass(obj):
            found.append((f"{module.__name__}.{name}", obj))
            for attr, member in vars(obj).items():
                if attr.startswith("_"):
                    continue
                func = member.fget if isinstance(member, property) else member
                if isinstance(func, (staticmethod, classmethod)):
                    func = func.__func__
                if inspect.isfunction(func):
                    found.append((f"{module.__name__}.{name}.{attr}", func))
    return found


@pytest.mark.parametrize("package", [graticule, ui], ids=["graticule", "ui"])
def test_public_api_is_documented_and_typed(package: ModuleType) -> None:
    problems: list[str] = []
    for module in _modules(package):
        if not (module.__doc__ or "").strip():
            problems.append(f"{module.__name__}: module docstring missing")
        for qualname, obj in _public_callables(module):
            if not (inspect.getdoc(obj) or "").strip():
                problems.append(f"{qualname}: docstring missing")
            if inspect.isfunction(obj):
                sig = inspect.signature(obj)
                for pname, param in sig.parameters.items():
                    if pname in ("self", "cls") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                        continue
                    if param.annotation is inspect.Parameter.empty:
                        problems.append(f"{qualname}: parameter '{pname}' has no type hint")
                if sig.return_annotation is inspect.Signature.empty:
                    problems.append(f"{qualname}: return type missing")
    assert not problems, "\n".join(problems)
