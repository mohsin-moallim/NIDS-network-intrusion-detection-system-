"""03 Measure charts: every builder builds and serialises in both themes, follows the identity (channel colour,
dash and marker; sequential ramp with switching text colour; shapes for normal and attack), stays under the row
cap, and the confusion and ROC charts export to PNG offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_curve, roc_curve

from graticule import theme, viz
from graticule.models.zoo import MODEL_KEYS

pytestmark = pytest.mark.unit
MODES = ["light", "dark"]
PNG = b"\x89PNG\r\n\x1a\n"


def _curve(seed: int, kind: str = "roc", n: int = 3_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    score = np.clip(y * (0.4 + 0.05 * seed) + rng.random(n) * 0.6, 0, 1)
    if kind == "roc":
        fpr, tpr, _ = roc_curve(y, score)
        return pd.DataFrame({"fpr": fpr, "tpr": tpr})
    precision, recall, _ = precision_recall_curve(y, score)
    return pd.DataFrame({"recall": recall, "precision": precision})


@pytest.fixture
def board() -> pd.DataFrame:
    """A leaderboard in the shape :func:`graticule.evaluate.leaderboard` returns (binary columns)."""
    rows = []
    for i, key in enumerate(MODEL_KEYS):
        rows.append({"key": key, "Channel": theme.CHANNEL_BY_KEY[key].label, "Balanced accuracy": 0.999 - i * 0.01,
                     "Accuracy": 0.998 - i * 0.01, "F1 (attack)": 0.99 - i * 0.02, "ROC-AUC": 0.9999 - i * 0.003,
                     "Average precision": float("nan") if key == "mlp" else 0.995, "Gap to best": i * 0.01,
                     "Fit s": 0.5 * (i + 1) ** 2, "Flows/s": 10.0 ** (6 - i), "Single-flow ms": 0.4 * (i + 1),
                     "Rows used": 1000 * (i + 1)})
    return pd.DataFrame(rows)


def _datasets(spec: dict) -> list[list[dict]]:
    return list(spec.get("datasets", {}).values())


def _layers(spec: dict) -> list[dict]:
    return spec.get("layer", [])


def _all_charts(board: pd.DataFrame, mode: str) -> dict[str, object]:
    counts = np.array([[950, 50], [20, 480]])
    multi_counts = np.arange(1, 50).reshape(7, 7)
    classes7 = ["BENIGN", "Bot", "DDoS", "DoS Hulk", "PortScan", "FTP-Patator", "Web Attack - XSS"]
    roc = {key: _curve(i) for i, key in enumerate(MODEL_KEYS)}
    pr = {key: _curve(i, "pr") for i, key in enumerate(MODEL_KEYS)}
    importance = pd.DataFrame({"feature": [f"f{i}" for i in range(30)], "importance": np.linspace(0.2, 0, 30)})
    permutation = importance.assign(std=0.01, importance=importance["importance"] - 0.02)
    cv = pd.DataFrame({"key": ["forest", "svm", "logreg"], "Balanced accuracy mean": [0.99, 0.97, 0.95],
                       "Balanced accuracy std": [0.004, 0.01, float("nan")], "Accuracy mean": [0.99, 0.98, 0.96],
                       "Accuracy std": [0.003, 0.004, 0.005]})
    folds = pd.DataFrame({"key": ["forest"] * 3 + ["svm"] * 3, "fold": [1, 2, 3] * 2,
                          "balanced_accuracy": [0.985, 0.99, 0.995, 0.96, 0.97, 0.98]})
    return {
        "leaderboard": viz.leaderboard_chart(board, mode, subtitle="Measured on 1,500 held-out rows."),  # type: ignore[arg-type]
        "confusion": viz.confusion_chart(counts, ["Normal", "Attack"], mode, title="CH1 Random forest"),  # type: ignore[arg-type]
        "confusion_counts": viz.confusion_chart(multi_counts, classes7, mode, show="count"),  # type: ignore[arg-type]
        "roc": viz.roc_chart(roc, mode, scores={k: 0.99 for k in MODEL_KEYS}),  # type: ignore[arg-type]
        "roc_zoom": viz.roc_chart(roc, mode, max_fpr=0.05),  # type: ignore[arg-type]
        "pr": viz.pr_chart(pr, mode, chance=0.3, scores={"forest": 0.98}),  # type: ignore[arg-type]
        "class_roc": viz.class_curves_chart({c: _curve(i) for i, c in enumerate(classes7)}, mode,  # type: ignore[arg-type]
                                            support={c: 100 - i for i, c in enumerate(classes7)}),
        "class_pr": viz.class_curves_chart({c: _curve(i, "pr") for i, c in enumerate(classes7[:3])}, mode,  # type: ignore[arg-type]
                                           kind="pr", scores={"BENIGN": 0.99}),
        "importance": viz.importance_chart(importance, mode),  # type: ignore[arg-type]
        "permutation": viz.importance_chart(permutation, mode, error="std", number_format=".4f"),  # type: ignore[arg-type]
        "fit": viz.timing_chart(board, mode, measure="Fit s"),  # type: ignore[arg-type]
        "speed": viz.timing_chart(board, mode, measure="Flows/s"),  # type: ignore[arg-type]
        "latency": viz.timing_chart(board, mode, measure="Single-flow ms"),  # type: ignore[arg-type]
        "timing_panels": viz.timing_panels(board, mode),  # type: ignore[arg-type]
        "cv": viz.cv_spread_chart(cv, mode, folds=folds),  # type: ignore[arg-type]
        "cv_accuracy": viz.cv_spread_chart(cv, mode, metric="Accuracy"),  # type: ignore[arg-type]
        "held_out": viz.held_out_classes_chart({"BENIGN": 30_000, "Bot": 20, "Heartbleed": 2}, mode),  # type: ignore[arg-type]
    }


# The dark palette is the same specs with other colours (each colour is checked by the tests above): its full pass
# runs with the slow tests.
@pytest.mark.parametrize("mode", [pytest.param(m, marks=pytest.mark.slow) if m == "dark" else m for m in MODES])
def test_every_chart_builds_and_serialises(board: pd.DataFrame, mode: str) -> None:
    charts = _all_charts(board, mode)
    for name, chart in charts.items():
        spec = chart.to_dict()  # type: ignore[attr-defined]
        assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/"), name
        assert spec["config"]["axis"]["labelFont"] == theme.FONT_MONO, name
        assert all(len(rows) <= viz.MAX_CHART_ROWS for rows in _datasets(spec)), name


@pytest.mark.parametrize("mode", MODES)
def test_channel_lines_keep_their_colour_dash_and_marker(board: pd.DataFrame, mode: str) -> None:
    roc = {key: _curve(i) for i, key in enumerate(MODEL_KEYS)}
    spec = viz.roc_chart(roc, mode, scores={k: 0.99 for k in MODEL_KEYS}).to_dict()  # type: ignore[arg-type]
    line = next(layer for layer in _layers(spec) if layer["mark"]["type"] == "line" and "color" in layer["encoding"])
    labels = [c.label for c in theme.CHANNELS]
    assert line["encoding"]["color"]["scale"]["domain"] == labels
    assert line["encoding"]["color"]["scale"]["range"] == [c.colour(mode) for c in theme.CHANNELS]  # type: ignore[arg-type]
    assert line["encoding"]["strokeDash"]["scale"]["range"] == [list(c.dash) or [1, 0] for c in theme.CHANNELS]
    markers = next(layer for layer in _layers(spec) if layer["mark"]["type"] == "point")
    assert markers["encoding"]["shape"]["scale"]["range"] == [c.marker for c in theme.CHANNELS]
    chance = _layers(spec)[0]
    assert chance["mark"]["strokeDash"] == [4, 4] and chance["mark"]["color"] == theme.palette(mode).muted  # type: ignore[arg-type]
    labels_layer = next(layer for layer in _layers(spec) if layer["mark"]["type"] == "text")
    assert labels_layer["mark"]["align"] == "left" and labels_layer["mark"]["dx"] > 0  # labelled at the line ends
    texts = [row["text"] for rows in _datasets(spec) for row in rows if "text" in row]
    assert "CH1 Random forest 0.9900" in texts
    board_spec = viz.leaderboard_chart(board, mode).to_dict()  # type: ignore[arg-type]
    points = _layers(board_spec)[0]
    assert points["encoding"]["shape"]["scale"]["range"] == [c.marker for c in theme.CHANNELS]
    rows = _datasets(board_spec)[0]
    assert rows[0]["metric"] == "Balanced accuracy"
    assert not any(r["channel"] == "CH4 Neural net (MLP)" and r["metric"] == "Average precision" for r in rows)


@pytest.mark.parametrize("mode", MODES)
def test_confusion_cells_use_the_ramp_and_switch_text_colour(mode: str) -> None:
    counts = np.array([[95, 5], [45, 55]])
    spec = viz.confusion_chart(counts, ["Normal", "Attack"], mode).to_dict()  # type: ignore[arg-type]
    rect, text = _layers(spec)
    assert rect["encoding"]["color"]["scale"]["range"] == list(theme.SEQUENTIAL[mode])
    assert text["encoding"]["color"]["scale"] is None
    cells = {(r["true"], r["predicted"]): r for r in _datasets(spec)[0]}
    p = theme.palette(mode)  # type: ignore[arg-type]
    light_letters = "#FFFFFF" if mode == "light" else theme.DARK.background
    assert cells[("Normal", "Normal")]["step"] == 9 and cells[("Normal", "Normal")]["ink"] == light_letters
    assert cells[("Normal", "Attack")]["step"] == 0 and cells[("Normal", "Attack")]["ink"] == p.text
    assert cells[("Attack", "Normal")]["step"] == 4 and cells[("Attack", "Normal")]["ink"] == p.text
    assert cells[("Attack", "Attack")]["step"] == 5 and cells[("Attack", "Attack")]["ink"] == light_letters
    assert cells[("Normal", "Normal")]["text"] == "95.0%\n95"
    counted = {(r["true"], r["predicted"]): r for r in _datasets(
        viz.confusion_chart(counts, ["Normal", "Attack"], mode, show="count").to_dict())[0]}  # type: ignore[arg-type]
    assert counted[("Normal", "Normal")]["text"] == "95\n95.0%" and counted[("Normal", "Normal")]["step"] == 9
    with pytest.raises(ValueError):
        viz.confusion_chart(counts, ["Normal"], mode)  # type: ignore[arg-type]


def test_class_curves_mark_normal_with_circles_and_attacks_with_diamonds() -> None:
    names = ["BENIGN", "Bot", "DDoS", "DoS Hulk", "PortScan", "FTP-Patator"]
    support = {"BENIGN": 900, "Bot": 5, "DDoS": 400, "DoS Hulk": 300, "PortScan": 200, "FTP-Patator": 100}
    spec = viz.class_curves_chart({n: _curve(i) for i, n in enumerate(names)}, "light", support=support).to_dict()
    line = next(layer for layer in _layers(spec) if layer["mark"]["type"] == "line" and "color" in layer["encoding"])
    colours = dict(zip(line["encoding"]["color"]["scale"]["domain"], line["encoding"]["color"]["scale"]["range"]))
    assert colours["BENIGN"] == theme.LIGHT.benign
    assert [colours[n] for n in ("DDoS", "DoS Hulk", "PortScan", "FTP-Patator")] == list(theme.ATTACK_TYPES["light"])
    assert colours["Bot"] == theme.LIGHT.muted  # fifth attack class by rows: "other attacks"
    shapes = next(layer for layer in _layers(spec) if layer["mark"]["type"] == "point")["encoding"]["shape"]
    assert dict(zip(shapes["scale"]["domain"], shapes["scale"]["range"]))["BENIGN"] == "circle"
    assert set(shapes["scale"]["range"][1:]) == {"diamond"}


def test_many_long_curves_are_thinned_under_the_row_cap() -> None:
    curves = {f"Class {i}": pd.DataFrame({"fpr": np.linspace(0, 1, 400), "tpr": np.linspace(0, 1, 400) ** 0.3})
              for i in range(15)}
    spec = viz.class_curves_chart(curves, "light").to_dict()
    biggest = max(len(rows) for rows in _datasets(spec))
    assert biggest <= viz.MAX_CHART_ROWS


def test_label_spreading_keeps_order_and_gap() -> None:
    placed = viz._spread([1.0, 1.0, 0.99, 0.2], gap=0.05)
    assert placed[0] > placed[1] > placed[2] > placed[3]
    assert min(a - b for a, b in zip(placed, placed[1:])) >= 0.05 - 1e-12
    assert max(placed) <= 1.0 and min(placed) >= 0.0
    crowded = viz._spread([0.01] * 30, gap=0.05)
    assert min(crowded) >= 0.0 and max(crowded) <= 1.0


def test_share_text_never_rounds_to_zero_or_a_hundred() -> None:
    assert [viz.share_text(v) for v in (0, 0.00016, 0.0123, 0.99984, 1.0)] == ["0%", "<0.1%", "1.2%", ">99.9%",
                                                                              "100%"]
    assert viz.share_text(0.003, decimals=0) == "<1%" and viz.share_text(0.5, decimals=0) == "50%"
    cells = viz._confusion_cells(np.array([[30_822, 5], [0, 40]]), ["BENIGN", "Bot"], "light", "share")
    assert cells["text"].tolist()[:2] == [">99.9%\n30,822", "<0.1%\n5"]


def test_score_and_zoom_domains() -> None:
    assert viz.score_domain([0.9991, 0.9995]) == [0.98, 1.0]
    assert viz.score_domain([0.5, 0.97]) == pytest.approx([0.44, 1.0])
    assert viz.score_domain([float("nan")]) == [0.0, 1.0]
    roc = {"forest": pd.DataFrame({"fpr": [0, 0.01, 0.05, 1], "tpr": [0, 0.97, 0.99, 1]}),
           "logreg": pd.DataFrame({"fpr": [0, 0.1, 1], "tpr": [0, 0.9, 1]})}
    low, high = viz.roc_zoom_domain(roc, 0.05)
    assert high == 1.0 and low < 0.9 * 0.05 / 0.1 and low >= 0.0


def test_confusion_and_roc_charts_export_to_png() -> None:
    confusion = viz.confusion_chart(np.array([[950, 50], [20, 480]]), ["Normal", "Attack"], "light")
    roc = viz.roc_chart({key: _curve(i) for i, key in enumerate(MODEL_KEYS)}, "light", scores={"forest": 0.99})
    for chart in (confusion, roc):
        png = viz.to_png(chart, scale=1, background=theme.LIGHT.surface)
        assert png[:8] == PNG and len(png) > 5_000
