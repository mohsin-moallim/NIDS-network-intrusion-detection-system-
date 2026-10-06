"""The PDF measurement record: a real five-channel binary build (offline, uncompressed so its text can be read back),
plus faster builds with a stand-in chart renderer for the multi-class, no-extras and no-held-out-rows cases.

Text is read back from the uncompressed page streams through each font's ToUnicode map, and the outline (bookmarks)
is read from the document catalogue, so the checks see what a PDF reader sees.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import socket
import time
import zlib
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nids import evaluate, viz
from nids.data.prepare import DataRequest, PreparedDataset
from nids.models import train
from nids.models.jobs import CancelToken, TrainingCancelled
from nids.models.train import TrainingRun, TrainRequest
from nids.report import pdf as rp
from tests.helpers import shared_fit, shared_sample

pytestmark = pytest.mark.unit
SEED = 7
#: The mark glyphs (circle, diamond, triangle) and the micro sign, which the bundled fonts cannot draw.
FORBIDDEN = "".join(map(chr, (0x25CB, 0x25C6, 0x25B2, 0x00B5)))


# --------------------------------------------------------------------------------------------------------------
# Reading a PDF back
# --------------------------------------------------------------------------------------------------------------
def _unescape(raw: bytes) -> bytes:
    """The bytes of a PDF literal string body (backslash escapes and octal codes resolved)."""
    out = bytearray()
    i = 0
    simple = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12, ord("("): 40, ord(")"): 41,
              ord("\\"): 92}
    while i < len(raw):
        byte = raw[i]
        if byte != 92:
            out.append(byte)
            i += 1
            continue
        nxt = raw[i + 1]
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif 48 <= nxt <= 55:
            digits = re.match(rb"[0-7]{1,3}", raw[i + 1:i + 4]).group(0)  # type: ignore[union-attr]
            out.append(int(digits, 8) & 0xFF)
            i += 1 + len(digits)
        elif nxt in (10, 13):
            i += 2
        else:
            out.append(nxt)
            i += 2
    return bytes(out)


def _objects(data: bytes) -> dict[int, bytes]:
    """Object number -> object body."""
    return {int(m.group(1)): m.group(2) for m in re.finditer(rb"(\d+) 0 obj(.*?)endobj", data, re.S)}


def _cmaps(objects: dict[int, bytes]) -> dict[str, dict[int, str]]:
    """Font resource name (``F1``...) -> glyph id -> text, from each font's ToUnicode stream."""
    fonts: dict[str, dict[int, str]] = {}
    names: dict[str, int] = {}
    for body in objects.values():
        for m in re.finditer(rb"/(F\d+) (\d+) 0 R", body):
            names[m.group(1).decode()] = int(m.group(2))
    for name, number in names.items():
        found = re.search(rb"/ToUnicode (\d+) 0 R", objects.get(number, b""))
        if not found:
            continue
        stream = objects[int(found.group(1))]
        table: dict[int, str] = {}
        for block in re.findall(rb"beginbfchar(.*?)endbfchar", stream, re.S):
            for src, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
                table[int(src, 16)] = bytes.fromhex(dst.decode()).decode("utf-16-be")
        for block in re.findall(rb"beginbfrange(.*?)endbfrange", stream, re.S):
            for lo, hi, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
                start = int(dst, 16)
                for offset, gid in enumerate(range(int(lo, 16), int(hi, 16) + 1)):
                    table[gid] = chr(start + offset)
        fonts[name] = table
    return fonts


_TEXT_OPS = re.compile(rb"/(F\d+) [\d.]+ Tf|\[((?:\((?:\\.|[^\\)])*\)|[^\]])*)\]\s*TJ|\(((?:\\.|[^\\)])*)\)\s*Tj|"
                       rb"(?<![A-Za-z])(ET)(?![A-Za-z])", re.S)


def pdf_text(data: bytes) -> str:
    """All text drawn on the pages of an uncompressed PDF: one line per text object (``BT`` ... ``ET``)."""
    objects = _objects(data)
    fonts = _cmaps(objects)
    lines: list[str] = []
    for body in objects.values():
        if b"stream" not in body or b"BT" not in body:
            continue
        current: dict[int, str] = {}
        line: list[str] = []
        for m in _TEXT_OPS.finditer(body):
            if m.group(1):
                current = fonts.get(m.group(1).decode(), {})
                continue
            if m.group(4):
                if line:
                    lines.append("".join(line))
                line = []
                continue
            pieces = [m.group(3)] if m.group(3) is not None else re.findall(rb"\(((?:\\.|[^\\)])*)\)", m.group(2))
            for piece in pieces:
                raw = _unescape(piece)
                for i in range(0, len(raw) - 1, 2):
                    line.append(current.get(int.from_bytes(raw[i:i + 2], "big"), ""))
        if line:
            lines.append("".join(line))
    return "\n".join(lines)


def _pdf_string(raw: bytes) -> str:
    """A PDF string object (literal or hex) as text."""
    if raw.startswith(b"<"):
        data = bytes.fromhex(raw[1:-1].decode())
    else:
        data = _unescape(raw[1:-1])
    return data[2:].decode("utf-16-be") if data.startswith(b"\xfe\xff") else data.decode("latin-1")


_OBJECT = re.compile(rb"(\d+) 0 obj")
_STREAM = re.compile(rb"stream\r?\n")
_LENGTH = re.compile(rb"/Length (\d+)(?! \d+ R)")


def inflated(data: bytes) -> bytes:
    """A compressed PDF with its text streams inflated (read without fpdf2), for :func:`pdf_text` and :func:`outline`.

    Each stream is cut at its declared ``/Length``: compressed bytes may well end in a CR or LF byte (or hold the
    text "endobj"), so the end markers alone cannot delimit them. Pictures and embedded font programs hold no text
    and are left out, so their binary bytes cannot confuse the parsing that follows.
    """
    parts = []
    position = 0
    while (found := _OBJECT.search(data, position)) is not None:
        start = found.end()
        end = data.find(b"endobj", start)
        if end < 0:
            break
        stream = _STREAM.search(data, start)
        if stream is None or stream.start() > end:  # no stream (a stream's dictionary never holds "endobj")
            body = data[start:end]
        else:
            head = data[start:stream.start()]
            declared = _LENGTH.search(head)
            if declared is not None:
                raw = data[stream.end():stream.end() + int(declared.group(1))]
            else:  # no direct length: drop the single line end before "endstream"
                stop = data.find(b"endstream", stream.end())
                raw = data[stream.end():stop]
                raw = raw[:-2] if raw.endswith(b"\r\n") else raw[:-1] if raw.endswith((b"\n", b"\r")) else raw
            end = data.find(b"endobj", stream.end() + len(raw))
            if b"/Subtype /Image" in head or b"/Length1" in head:
                raw = b""
                head = head.replace(b"/Filter /FlateDecode", b"")
            elif b"/FlateDecode" in head:
                head = head.replace(b"/Filter /FlateDecode", b"")
                raw = zlib.decompress(raw)
            body = head + b"stream\n" + raw + b"\nendstream\n"
        parts.append(found.group(1) + b" 0 obj" + body + b"endobj\n")
        position = end + len(b"endobj")
    return b"".join(parts)


def outline(data: bytes) -> list[str]:
    """Titles of the document outline (bookmarks), in order."""
    objects = _objects(data)
    titles = []
    for number in sorted(objects):
        body = objects[number]
        if b"/Parent" not in body:
            continue
        m = re.search(rb"/Title\s*(\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f]*>)", body)
        if m:
            titles.append(_pdf_string(m.group(1)))
    return titles


# --------------------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prepared() -> PreparedDataset:
    """About 1,500 synthetic flows (six classes), shared with the other modules that use this sample."""
    return shared_sample(DataRequest(source="synthetic", synthetic_flows=1_500, seed=SEED))


def _fit(prepared: PreparedDataset, **changes: Any) -> TrainingRun:
    """A fresh copy of a ``profile="test"`` fit of ``prepared`` (each fit made once per session, see shared_fit)."""
    return shared_fit(prepared, TrainRequest(**{"profile": "test", "seed": SEED, **changes}))


@pytest.fixture(scope="module")
def binary_run(prepared: PreparedDataset) -> TrainingRun:
    """All five channels, binary mode."""
    return _fit(prepared, mode="binary")


@pytest.fixture(scope="module")
def multiclass_run(prepared: PreparedDataset) -> TrainingRun:
    """Two channels, multi-class mode."""
    return _fit(prepared, mode="multiclass", channels=("forest", "logreg"))


def fake_cv(run: TrainingRun) -> pd.DataFrame:
    """A cross-validation table shaped like :func:`nids.evaluate.cross_validate_run` output."""
    rows = []
    folds = []
    for key in ("forest", "logreg"):
        rows.append({"key": key, "Channel": evaluate.channel_label(key), "Folds": 3, "Rows": 900,
                     "Accuracy mean": 0.97, "Accuracy std": 0.01, "Balanced accuracy mean": 0.95,
                     "Balanced accuracy std": 0.02, "F1 macro mean": 0.94, "F1 macro std": 0.02, "Fit s mean": 0.1,
                     "Error": None})
        folds += [{"key": key, "fold": i + 1, "accuracy": 0.96 + i / 100, "balanced_accuracy": 0.94 + i / 100,
                   "f1_macro": 0.93, "fit_seconds": 0.1} for i in range(3)]
    frame = pd.DataFrame(rows, columns=list(evaluate.CV_COLUMNS))
    frame.attrs.update({"k": 3, "k_requested": 3, "rows": 900, "rows_available": len(run.data.y_train), "note": "",
                        "cancelled": False, "seconds": 1.0, "seed": SEED, "folds": folds})
    return frame


def fake_permutation(run: TrainingRun) -> pd.DataFrame:
    """A permutation-importance table shaped like :func:`nids.evaluate.permutation_importance_for` output."""
    names = list(run.data.feature_names)
    frame = pd.DataFrame({"feature": pd.Series(names, dtype="str"),
                          "importance": np.linspace(0.2, 0.0, len(names)), "std": np.full(len(names), 0.01)})
    frame.attrs.update({"key": "logreg", "rows": 400, "repeats": 2, "baseline": 0.9, "seconds": 0.5, "seed": SEED})
    return frame


def fake_assay(run: TrainingRun, run_id: str | None = None) -> SimpleNamespace:
    """An Assay result shaped like :class:`nids.scoring.ScoredBatch` (uploaded columns plus scoring columns)."""
    n = 40
    rng = np.random.default_rng(1)
    attack = rng.random(n)
    frame = pd.DataFrame({"Flow Duration": rng.random(n).astype(np.float32), "Label": ["BENIGN"] * n,
                          "predicted_label": np.where(attack > 0.6, "Attack", "Normal"),
                          "prob_Normal": 1 - attack, "prob_Attack": attack, "attack_probability": attack,
                          "alert": attack >= 0.9, "true_label": np.where(attack > 0.5, "Attack", "Normal")})
    confusion = pd.DataFrame([[20, 3], [2, 15]], index=["Normal", "Attack"], columns=["Normal", "Attack"])
    return SimpleNamespace(frame=frame, channel="xgboost", rows=n, rows_with_bad_values=2, seconds=0.4,
                           labelled=True, accuracy=0.875, balanced_accuracy=0.86, confusion=confusion,
                           unseen_labels={"Heartbleed": 3}, run_id=run.run_id if run_id is None else run_id,
                           classes=run.data.classes, source_name="upload.csv", alert_threshold=0.9)


def fake_sweep(run: TrainingRun) -> SimpleNamespace:
    """A simulation session shaped like :class:`nids.simulate.SimulationSession`."""
    stats = SimpleNamespace(emitted=500, correct=470, live_accuracy=0.94, live_balanced_accuracy=0.92,
                            confusion=np.array([[300, 10], [20, 170]]), alerts_total=150, ticks=10)
    source = SimpleNamespace(describe=lambda: "Replay of 748 held-out rows (natural attack share)")
    log = pd.DataFrame({"seq": np.arange(3), "tick": [1, 1, 2], "row_id": [5, 9, 11], "true_label": ["Normal"] * 3,
                        "predicted": ["Normal"] * 3, "attack_probability": [0.1, 0.2, 0.3],
                        "confidence": [0.9, 0.8, 0.7], "alert": [False] * 3, "correct": [True] * 3})
    return SimpleNamespace(run=run, run_id=run.run_id, channel="forest", classes=run.data.classes, stats=stats,
                           source=source, alert_threshold=0.9, log_frame=lambda: log)


class _Glyphs(logging.Handler):
    """Collects fpdf2's warnings about characters a font cannot draw."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _build_full_record(prepared: PreparedDataset, run: TrainingRun) -> dict[str, Any]:
    """The full record of ``run`` with every extra present, built with the network blocked (uncompressed, so its
    text can be read back), plus what was observed while building it."""
    evals = evaluate.evaluate_run(run)
    extras = rp.ReportExtras(cross_validation=fake_cv(run), permutations={"logreg": fake_permutation(run)},
                             assay=rp.summarise_assay(fake_assay(run)), sweep=rp.summarise_sweep(fake_sweep(run)))
    attempts: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        attempts.append(args)
        raise OSError("network access is blocked in this test")

    progress: list[tuple[str, float]] = []
    handler = _Glyphs()
    logger = logging.getLogger("fpdf")
    logger.addHandler(handler)
    fits = sum(train.FIT_CALLS.values())
    started = time.perf_counter()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(socket, "socket", refuse)
            patch.setattr(socket, "create_connection", refuse)
            report = rp.render_report(run, evals, prepared_summary=rp.summarise_prepared(prepared),
                                      settings={"alert_threshold": 0.999}, extras=extras, compress=False,
                                      progress=lambda message, fraction: progress.append((message, fraction)))
    finally:
        logger.removeHandler(handler)
    return {"report": report, "text": pdf_text(report.data), "outline": outline(report.data), "attempts": attempts,
            "progress": progress, "glyph_warnings": [m for m in handler.messages if "missing" in m],
            "seconds": time.perf_counter() - started, "fits": sum(train.FIT_CALLS.values()) - fits}


@pytest.fixture(scope="module")
def binary_record(prepared: PreparedDataset, binary_run: TrainingRun) -> dict[str, Any]:
    """The full record of the five-channel binary run, every chart built (and its spec checked) but drawn by the
    stand-in renderer, which keeps the default suite quick; ``test_binary_record_with_real_charts`` (slow) and the
    real-data record draw them for real."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(viz, "to_png", fake_png)
        return _build_full_record(prepared, binary_run)


_FAKE_PNGS: list[int] = [0]


def fake_png(chart: Any, scale: float = 2, *, background: str | None = None) -> bytes:
    """Stand-in for :func:`nids.viz.to_png`: checks the chart's spec, returns a small PNG of the chart's width.

    Every picture differs (one pixel carries a counter), so the PDF embeds each one rather than sharing one image.
    """
    spec = chart.to_dict()
    width = spec.get("width") if isinstance(spec.get("width"), (int, float)) else 400
    _FAKE_PNGS[0] += 1
    image = Image.new("RGB", (int(width) * 2, 300), "#FFFFFF")
    image.putpixel((0, 0), (_FAKE_PNGS[0] % 256, (_FAKE_PNGS[0] // 256) % 256, 7))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def quick_charts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Charts are built (and their specs validated) but not rendered, to keep these builds quick."""
    monkeypatch.setattr(viz, "to_png", fake_png)


# --------------------------------------------------------------------------------------------------------------
# The full binary record
# --------------------------------------------------------------------------------------------------------------
def test_binary_record_is_a_complete_pdf_of_sensible_size(binary_record: dict[str, Any]) -> None:
    report: rp.RenderedReport = binary_record["report"]
    data = report.data
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    assert 6 <= report.pages <= 24 and report.pages == rp.count_pages(data)
    # Readings, confusion matrices (5), curves (2), importance (3), timing, CV, sample, assay and sweep matrices.
    assert report.images >= 14 and report.images == rp.count_images(data)
    assert report.size_bytes < 3 * 2**20
    assert not report.problems
    assert binary_record["fits"] == 0  # building a record never fits anything


def test_binary_record_has_every_section_in_its_outline(binary_record: dict[str, Any]) -> None:
    expected = ["Sample sheet", "Fit settings", "Readings", "Confusion matrices", "Curves", "Feature importance",
                "Timing", "Cross-validation", "Assay", "Sweep", "Notes and limitations"]
    assert binary_record["outline"] == expected
    assert list(binary_record["report"].sections) == expected


def test_key_text_is_on_the_pages(binary_record: dict[str, Any], binary_run: TrainingRun) -> None:
    text = binary_record["text"]
    pages = binary_record["report"].pages
    for phrase in ("MEASUREMENT RECORD", "NIDS", "Network Intrusion Detection System", "NIDS measurement record",
                   "A measuring bench for training", "Contents", "Sample sheet", "Fit settings", "Readings",
                   "Confusion matrices", "Curves", "Feature importance", "Timing", "Cross-validation", "Assay",
                   "Sweep", "Notes and limitations",
                   binary_run.run_id, "Consensus", "Balanced accuracy", "Bal. acc.", "CH1 Random forest",
                   "CH5 Logistic regression", "Sharafaldin", "ICISSP", "upload.csv", "Heartbleed",
                   f"page 1 of {pages}", f"page {pages} of {pages}", "Normal traffic", "Attack",
                   "Alert (high-confidence attack)", "Classes before and after sampling", "Per class:"):
        assert phrase in text, phrase
    assert "{nb}" not in text


def test_no_glyph_is_missing_and_marks_are_never_text(binary_record: dict[str, Any]) -> None:
    assert binary_record["glyph_warnings"] == []
    assert not set(FORBIDDEN) & set(binary_record["text"])


def test_the_record_is_built_without_any_network_access(binary_record: dict[str, Any]) -> None:
    assert binary_record["attempts"] == []
    assert binary_record["seconds"] < 120


def test_a_chart_renders_to_png_without_any_network_access(binary_run: TrainingRun) -> None:
    """The real renderer (vl-convert, fonts registered locally) draws a record chart with the network blocked."""
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise OSError("network access is blocked in this test")

    evaluation = evaluate.evaluate_run(binary_run)["logreg"]
    chart = viz.confusion_chart(evaluation.confusion, list(binary_run.data.classes), "light", show="share",
                                title="CH5 Logistic regression")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket, "socket", refuse)
        patch.setattr(socket, "create_connection", refuse)
        png = viz.to_png(chart)
    assert png.startswith(b"\x89PNG") and Image.open(io.BytesIO(png)).size[0] > 100


@pytest.mark.slow
def test_binary_record_with_real_charts(prepared: PreparedDataset, binary_run: TrainingRun) -> None:
    """The full five-channel record with every chart drawn by the real renderer, offline (slow: about 15 charts)."""
    built = _build_full_record(prepared, binary_run)
    report: rp.RenderedReport = built["report"]
    assert built["attempts"] == [] and not report.problems and built["glyph_warnings"] == []
    assert report.images >= 14 and report.images == rp.count_images(report.data)
    assert 6 <= report.pages <= 24 and report.size_bytes < 3 * 2**20 and built["seconds"] < 60
    assert built["outline"][-1] == "Notes and limitations" and "Sharafaldin" in built["text"]


def test_alert_thresholds_print_with_three_decimals(binary_record: dict[str, Any]) -> None:
    """0.999 (the highest threshold the app allows) must not read 1.00; the alert rule is spelled out."""
    text = " ".join(binary_record["text"].split())  # wrapped lines joined
    assert "0.999" in text and "1.00 (" not in text and "least 1.00" not in text
    assert "attack verdicts with attack probability at least 0.900" in text  # the Assay and Sweep thresholds
    assert "an attack verdict whose attack probability is at least the alert threshold" in text


def test_inflating_a_compressed_record_cuts_each_stream_at_its_length(prepared: PreparedDataset,
                                                                      binary_run: TrainingRun,
                                                                      hollow_record: rp.RenderedReport,
                                                                      quick_charts: None) -> None:
    """Compressed bytes may end in a line-feed byte; :func:`inflated` must not cut them short."""
    candidates = (hashlib.sha256(str(i).encode()).digest() * 4 for i in range(10_000))
    payload = next(data for data in candidates if zlib.compress(data)[-1:] in (b"\n", b"\r"))
    packed = zlib.compress(payload)
    document = (b"%PDF-1.4\n1 0 obj\n<< /Filter /FlateDecode /Length " + str(len(packed)).encode() + b" >>\nstream\n"
                + packed + b"\nendstream\nendobj\n")
    assert payload in inflated(document)
    # A real compressed record reads back as its uncompressed twin (and is smaller).
    hollow = _without_held_out_rows(binary_run)
    plain = hollow_record.data  # the same record, built uncompressed
    compressed = rp.build_report(hollow, {}, prepared_summary=_another_samples_account(prepared), settings={})
    assert compressed.startswith(b"%PDF-") and len(compressed) < len(plain)
    assert rp.count_pages(compressed) == rp.count_pages(plain)

    def stamped(data: bytes) -> str:
        # The build time (footer and cover) moves on between the two builds, possibly into the next minute.
        return re.sub(r"\d{4}-\d\d-\d\d \d\d:\d\d UTC", "<built>", pdf_text(data))

    assert stamped(inflated(compressed)) == stamped(plain)
    assert outline(inflated(compressed)) == outline(plain)


def test_progress_runs_to_done(binary_record: dict[str, Any]) -> None:
    fractions = [f for _, f in binary_record["progress"]]
    assert binary_record["progress"][-1] == ("Done", 1.0)
    assert fractions == sorted(fractions) and 0.0 < fractions[0] < 1.0
    assert any(message.startswith("Drawing") for message, _ in binary_record["progress"])


@pytest.mark.slow
def test_multiclass_record_with_real_charts(prepared: PreparedDataset) -> None:
    """Every chart of a five-channel multi-class record renders (slow: about fifteen charts)."""
    run = _fit(prepared, mode="multiclass")
    report = rp.render_report(run, evaluate.evaluate_run(run), prepared_summary=rp.summarise_prepared(prepared),
                              settings={})
    assert not report.problems and report.images >= 12 and report.size_bytes < 3 * 2**20


def _another_samples_account(prepared: PreparedDataset) -> dict[str, Any]:
    """The sample sheet's account of ``prepared`` with a fingerprint no run was fitted on."""
    return dict(rp.summarise_prepared(prepared), fingerprint="0" * 64)


@pytest.fixture(scope="module")
def hollow_record(prepared: PreparedDataset, binary_run: TrainingRun) -> rp.RenderedReport:
    """The uncompressed record of the binary run as loaded without its held-out rows, handed the account of another
    sample (which it must not use) and no settings; built once for the tests that read it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(viz, "to_png", fake_png)
        return rp.render_report(_without_held_out_rows(binary_run), {}, prepared_summary=_another_samples_account(
            prepared), settings={}, compress=False)


def _without_held_out_rows(run: TrainingRun) -> TrainingRun:
    """A copy of ``run`` as loaded from disk without its held-out rows."""
    data = run.data
    empty = replace(data, X_test=np.empty((0, data.n_features), dtype=np.float32), y_test=np.empty(0, dtype=np.int64),
                    test_rows=np.empty(0, dtype=np.int64), detailed_test_labels=np.empty(0, dtype=str))
    return replace(run, data=empty, origin="loaded", bundle_path="saved_models/20261001-000000-abcd")


# --------------------------------------------------------------------------------------------------------------
# Other runs and missing inputs
# --------------------------------------------------------------------------------------------------------------
def test_multiclass_record_without_extras_or_sample_account(multiclass_run: TrainingRun, quick_charts: None) -> None:
    evals = evaluate.evaluate_run(multiclass_run)
    report = rp.render_report(multiclass_run, evals, prepared_summary=None, settings={}, compress=False)
    titles = outline(report.data)
    assert titles == ["Sample sheet", "Fit settings", "Readings", "Confusion matrices", "Curves",
                      "Feature importance", "Timing", "Notes and limitations"]
    text = pdf_text(report.data)
    assert "Multi-class" in text and "One-vs-rest curves" in text
    assert "not in memory" in text  # the sample sheet says where its figures come from
    assert "Consensus (2)" in text
    for name in multiclass_run.data.classes:
        assert name in text
    assert report.images >= 1 and not report.problems


def test_a_two_class_multiclass_record_keeps_headings_over_their_numbers(
        prepared: PreparedDataset, quick_charts: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-class run left with two classes (one attack type above the smallest class size) has the multi-class
    leaderboard columns; the record must head them as such, one heading per cell."""
    counts = prepared.frame["Label"].astype(str).value_counts()
    assert counts.iloc[1] > counts.iloc[2]
    run = _fit(prepared, mode="multiclass", min_class_count=int(counts.iloc[1]), channels=("forest", "logreg"))
    assert run.request.mode == "multiclass" and len(run.data.classes) == 2
    tables: list[tuple[list[str], list[list[str]]]] = []
    real = rp._Builder.table

    def spy(self: Any, headers: Any, rows: Any, **options: Any) -> None:
        tables.append((list(headers), [list(row) for row in rows]))
        real(self, headers, rows, **options)

    monkeypatch.setattr(rp._Builder, "table", spy)
    token = CancelToken()

    def stop_after_readings(message: str, fraction: float) -> None:
        if any(heads[:2] == ["Channel", "Bal. acc."] for heads, _ in tables):
            token.cancel()  # the rest of the record is not needed here

    with pytest.raises(TrainingCancelled):
        rp.render_report(run, evaluate.evaluate_run(run), prepared_summary=None, settings={}, compress=False,
                         progress=stop_after_readings, cancel=token)
    heads, rows = next(t for t in tables if t[0][:2] == ["Channel", "Bal. acc."])
    assert heads == ["Channel", "Bal. acc.", "Accuracy", "F1 macro", "F1 wtd", "Prec. macro", "Rec. macro",
                     "ROC-AUC", "Avg prec.", "Gap"]
    assert len(rows) == 3 and all(len(row) == len(heads) for row in rows)  # two channels and the consensus
    board = evaluate.leaderboard(evaluate.evaluate_run(run), run)
    first = board.iloc[0]
    assert rows[0][1:-1] == [f"{float(first[c]):.4f}" for c in ("Balanced accuracy", "Accuracy", "F1 macro",
                                                                 "F1 weighted", "Precision macro", "Recall macro",
                                                                 "ROC-AUC", "Average precision")]


def test_a_sample_account_of_another_sample_is_not_used(hollow_record: rp.RenderedReport) -> None:
    # The hollow record was handed the account of another sample (the sample sheet is the same with or without
    # readings).
    text = pdf_text(hollow_record.data)
    assert "not the one this run was fitted on" in text
    assert "Classes before and after sampling" not in text


def test_a_run_without_held_out_rows_gets_its_recorded_readings(hollow_record: rp.RenderedReport) -> None:
    report = hollow_record
    titles = outline(report.data)
    assert "Confusion matrices" not in titles and "Curves" not in titles
    assert {"Readings", "Timing", "Notes and limitations"} <= set(titles)
    text = pdf_text(report.data)
    assert "without its held-out rows" in text and "Loaded from the saved channel set 20261001-000000-abcd" in text
    assert "CH1 Random forest" in text


def test_a_cancelled_build_stops_at_once(binary_run: TrainingRun) -> None:
    token = CancelToken()
    token.cancel()
    with pytest.raises(TrainingCancelled):
        rp.render_report(binary_run, evaluate.evaluate_run(binary_run), prepared_summary=None, settings={},
                         cancel=token)


def test_a_chart_that_fails_is_reported_not_fatal(multiclass_run: TrainingRun, binary_run: TrainingRun,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(chart: Any, scale: float = 2, *, background: str | None = None) -> bytes:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(viz, "to_png", broken)
    # The full multi-class record: every chart site (readings, confusion grid, per-class curves, importance,
    # timing) fails, and the build still finishes with each failure named.
    report = rp.render_report(multiclass_run, evaluate.evaluate_run(multiclass_run), prepared_summary=None,
                              settings={}, compress=False)
    assert report.images == 0 and len(report.problems) >= len(multiclass_run.ok_channels()) + 3
    assert all("renderer unavailable" in p for p in report.problems)
    assert "could not be drawn" in pdf_text(report.data)
    # The short record of a run loaded without its held-out rows (and with no sample account) as well.
    hollow = rp.render_report(_without_held_out_rows(binary_run), {}, prepared_summary=None, settings={},
                              compress=False)
    assert hollow.images == 0 and hollow.problems and all("renderer unavailable" in p for p in hollow.problems)
    assert {"Readings", "Timing", "Notes and limitations"} <= set(outline(hollow.data))


# --------------------------------------------------------------------------------------------------------------
# Small parts
# --------------------------------------------------------------------------------------------------------------
def test_printable_replaces_characters_the_fonts_lack() -> None:
    circle, diamond, triangle, micro, plus_minus = map(chr, (0x25CB, 0x25C6, 0x25B2, 0x00B5, 0x00B1))
    text = rp.printable(f"{circle} Normal {diamond} Attack {triangle} Alert 5 {micro}s {plus_minus} 0.01", rp.BODY)
    assert not set(FORBIDDEN) & set(text)
    assert "Normal" in text and "5 us" in text and chr(0x00B1) in text  # the body face has the plus-minus sign
    assert rp.printable(chr(0x00B1), rp.HEAD) == "+/-"  # the heading face does not


def test_summaries_read_assay_and_sweep_objects(binary_run: TrainingRun) -> None:
    assay = rp.summarise_assay(fake_assay(binary_run))
    assert assay is not None and assay.channel == "CH2 XGBoost" and assay.rows == 40 and assay.labelled
    assert assay.alerts == int((fake_assay(binary_run).frame["attack_probability"] >= 0.9).sum())
    assert assay.unseen_labels == {"Heartbleed": 3} and assay.source_name == "upload.csv"
    sweep = rp.summarise_sweep(fake_sweep(binary_run))
    assert sweep is not None and sweep.flows == 500 and sweep.alerts == 150 and sweep.channel == "CH1 Random forest"
    assert sweep.source.startswith("Replay") and sweep.confusion is not None and sweep.confusion.shape == (2, 2)
    assert rp.summarise_sweep(SimpleNamespace(stats=SimpleNamespace(emitted=0))) is None
    assert rp.summarise_assay(None) is None


def test_extras_only_take_results_made_with_the_run(binary_run: TrainingRun) -> None:
    evaluate.remember_cross_validation(binary_run, fake_cv(binary_run))
    try:
        mine = rp.ReportExtras.from_run(binary_run, assay=fake_assay(binary_run), sweep=fake_sweep(binary_run))
        assert mine.cross_validation is not None and mine.assay is not None and mine.sweep is not None
        foreign = rp.ReportExtras.from_run(binary_run, assay=fake_assay(binary_run, run_id="another-run"))
        assert foreign.assay is None and foreign.sweep is None
    finally:
        setattr(binary_run, evaluate.CROSS_VALIDATION_ATTR, None)


def test_static_font_instances_are_cached(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp.tempfile, "gettempdir", lambda: str(tmp_path))
    first = rp.static_font("AtkinsonHyperlegibleMono-Variable.ttf", {"wght": 700})
    again = rp.static_font("AtkinsonHyperlegibleMono-Variable.ttf", {"wght": 700})
    assert first == again and first.is_file() and first.parent.parent == tmp_path
    assert not list(first.parent.glob("*.tmp"))
