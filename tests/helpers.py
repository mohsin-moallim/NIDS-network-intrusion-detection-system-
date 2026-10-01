"""Test helpers: tiny CIC-IDS2017-style CSV files written into a test's ``tmp_path`` (never into the repository).

The files copy the real header layout (79 columns: stray leading spaces, and ``Fwd Header Length`` repeated after
``Avg Bwd Segment Size``) and let a test control every awkward detail: text encoding, the stand-in character in
Web Attack labels, a byte-order mark, "Infinity"/"NaN"/empty cells, duplicate rows and conflicting labels. The
values are made up; no dataset rows are involved.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from graticule.schema import FEATURES, LABEL

LabelStyle = Literal["fffd", "cp1252", "clean"]
REPEATED = "Fwd Header Length"
REPEAT_AFTER = "Avg Bwd Segment Size"
FIELDS = list(FEATURES)


def raw_header(*, with_duplicate_column: bool = True, leading_spaces: bool = True,
               drop_columns: Sequence[str] = (), extra_columns: Sequence[str] = (),
               include_label: bool = True) -> list[str]:
    """Header names in file order, spaced like the published files (most names carry a leading space)."""
    names: list[str] = []
    for i, name in enumerate(FIELDS):
        if name in drop_columns:
            continue
        names.append(name)
        if with_duplicate_column and name == REPEAT_AFTER:
            names.append(REPEATED)
    names.extend(extra_columns)
    if include_label:
        names.append(LABEL)
    if leading_spaces:
        names = [n if i % 7 == 3 else f" {n}" for i, n in enumerate(names)]
    return names


def flow_row(label: str = "BENIGN", key: int = 0, **overrides: object) -> dict[str, object]:
    """A made-up flow whose values depend on ``key`` (different keys give different rows).

    Overrides use feature names with spaces replaced by underscores and slashes by ``_per_``, or pass a mapping
    to :func:`with_values` instead.
    """
    row: dict[str, object] = {name: float((i + 1) * 3 + key * 7) for i, name in enumerate(FIELDS)}
    row["Destination Port"] = float(80 + key % 50)
    row["Flow Duration"] = float(1_000 + key * 13)
    row["Total Fwd Packets"] = float(2 + key % 5)
    row["Total Backward Packets"] = float(1 + key % 3)
    row["Total Length of Fwd Packets"] = float(100 + key)
    row["Total Length of Bwd Packets"] = float(50 + key)
    row["Flow Bytes/s"] = (150 + 2 * key) / (1_000 + key * 13) * 1e6
    row["Flow Packets/s"] = (3 + key % 5 + key % 3) / (1_000 + key * 13) * 1e6
    row[LABEL] = label
    for name, value in overrides.items():
        row[name.replace("_per_", "/").replace("_", " ")] = value
    return row


def with_values(row: Mapping[str, object], values: Mapping[str, object]) -> dict[str, object]:
    """Copy of ``row`` with some columns replaced (keys are exact column names)."""
    out = dict(row)
    out.update(values)
    return out


def make_rows(spec: Mapping[str, int], start: int = 0) -> list[dict[str, object]]:
    """Distinct rows: ``spec`` maps a label to how many rows of it to make; keys run from ``start``."""
    rows = []
    key = start
    for label, count in spec.items():
        for _ in range(count):
            rows.append(flow_row(label, key))
            key += 1
    return rows


def _cell(value: object) -> str:
    """How one value is written in the CSV: inf as "Infinity", NaN as "NaN", None as an empty cell."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return repr(int(value)) if value.is_integer() else repr(value)
    return str(value)


def _render_label(label: object, style: LabelStyle) -> str:
    """Write a Web Attack label with the stand-in character a given copy of the files uses."""
    text = "" if label is None else str(label)
    if text.startswith("Web Attack - "):
        if style == "fffd":
            return text.replace(" - ", " � ", 1)
        if style == "cp1252":
            return text.replace(" - ", " – ", 1)  # encodes to byte 0x96 in Windows-1252
    return text


def write_cic_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]] | None = None,
    *,
    encoding: str = "utf-8",
    label_style: LabelStyle = "fffd",
    with_duplicate_column: bool = True,
    leading_spaces: bool = True,
    bom: bool = False,
    duplicate_mismatch: bool = False,
    drop_columns: Sequence[str] = (),
    extra_columns: Mapping[str, object] | None = None,
    include_label: bool = True,
    extra_rows: int = 0,
) -> Path:
    """Write a small CIC-style CSV to ``path`` and return the path.

    ``rows`` are dicts keyed by clean column names (missing features get made-up values). ``duplicate_mismatch``
    makes the repeated column differ from the first copy in one row. ``extra_rows`` appends that many distinct
    BENIGN filler rows.
    """
    rows = [dict(r) for r in (rows if rows is not None else default_rows())]
    rows += make_rows({"BENIGN": extra_rows}, start=10_000) if extra_rows else []
    extras = dict(extra_columns or {})
    header = raw_header(with_duplicate_column=with_duplicate_column, leading_spaces=leading_spaces,
                        drop_columns=drop_columns, extra_columns=list(extras), include_label=include_label)
    clean_names = [h.strip() for h in header]
    lines = [",".join(header)]
    for index, source in enumerate(rows):
        filled = flow_row(str(source.get(LABEL, "BENIGN")), index)
        filled.update(source)
        filled.update({k: v for k, v in extras.items() if k not in source})
        cells = []
        seen_repeat = False
        for name in clean_names:
            if name == REPEATED and seen_repeat:
                value = filled[REPEATED]
                if duplicate_mismatch and index == 0:
                    value = (float(value) if value not in (None, "") else 0.0) + 1.0
                cells.append(_cell(value))
                continue
            if name == REPEATED:
                seen_repeat = True
            if name == LABEL:
                cells.append(_render_label(filled.get(LABEL), label_style))
            else:
                cells.append(_cell(filled.get(name)))
        lines.append(",".join(cells))
    data = ("\n".join(lines) + "\n").encode(encoding)
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def default_rows() -> list[dict[str, object]]:
    """A handful of rows: normal traffic, one DoS class and two Web Attack classes, plus one infinite rate."""
    rows = make_rows({"BENIGN": 6, "DoS Hulk": 3, "Web Attack - Brute Force": 2, "Web Attack - XSS": 1})
    rows[1] = with_values(rows[1], {"Flow Bytes/s": float("inf"), "Flow Packets/s": float("inf")})
    return rows


def feature_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Rows as an in-memory frame shaped like :func:`graticule.data.reader.read_flow_csv` output."""
    filled = []
    for index, source in enumerate(rows):
        row = flow_row(str(source.get(LABEL, "BENIGN")), index)
        row.update(source)
        filled.append(row)
    block = np.array([[float(r[f]) if r[f] is not None else np.nan for f in FIELDS] for r in filled],
                     dtype=np.float32).reshape(len(filled), len(FIELDS))
    frame = pd.DataFrame(block, columns=FIELDS)
    frame[LABEL] = pd.Series([str(r[LABEL]) for r in filled], dtype="str")
    return frame


def fake_generator(n_flows: int, *, seed: int, attack_share: float = 0.35, blur: float = 0.04) -> pd.DataFrame:
    """Stand-in for ``graticule.data.synthetic.generate`` with the agreed output shape (for isolated tests)."""
    rng = np.random.default_rng(seed)
    kinds = ["Flood", "Slow Drip", "Sweep", "Credential Guess", "Web Injection"]
    is_attack = rng.random(n_flows) < attack_share
    labels = np.where(is_attack, rng.choice(kinds, size=n_flows), "BENIGN")
    base = rng.gamma(2.0, 50.0, size=(n_flows, len(FIELDS))).astype(np.float32)
    base[is_attack] *= 3.0
    frame = pd.DataFrame(base, columns=FIELDS)
    frame.loc[frame.index[:3], "Flow Bytes/s"] = np.inf
    frame[LABEL] = pd.Series(labels, dtype="str")
    return frame
