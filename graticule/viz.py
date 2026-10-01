"""Chart builders shared by the UI and the PDF record.

Every builder takes plain data plus a theme mode and returns an Altair chart with its data inlined, so the same
definition renders in the browser (``st.altair_chart(chart, theme=None)``) and as a PNG for the PDF
(:func:`to_png`, through vl-convert, with no browser involved). Colours and fonts come from :mod:`graticule.theme`.
Normal traffic is always drawn with circles and attacks with diamonds, so colour is never the only cue.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Union

import altair as alt
import numpy as np
import pandas as pd

from graticule.schema import is_benign
from graticule.theme import (
    ATTACK_TYPES,
    CHANNEL_BY_KEY,
    CHANNELS,
    DARK,
    FONT_BODY,
    FONT_HEADING,
    FONT_MONO,
    SEQUENTIAL,
    ChannelStyle,
    Mode,
    palette,
)

AnyChart = Union[alt.Chart, alt.LayerChart, alt.VConcatChart, alt.HConcatChart, alt.ConcatChart, alt.FacetChart]
FONT_DIR = Path(__file__).resolve().parent.parent / "static" / "fonts"
MAX_CHART_ROWS = 5_000
DEFAULT_WIDTH = 560
KIND_NORMAL = "Normal"
KIND_ATTACK = "Attack"

_fonts_lock = threading.Lock()
_fonts_registered = False


def base_config(chart: AnyChart, mode: Mode = "light", *, background: str | None = None, fit: bool = True) -> AnyChart:
    """Apply Graticule's chart styling: body font for text, Instrument Sans titles, mono axis labels, hairline grid.

    The background stays transparent unless ``background`` is given, so the chart sits on the page colour in both
    themes. Returns a configured copy (``chart`` itself is not changed). With ``fit`` (the default) a single or
    layered chart fits its container's width; ``fit=False`` keeps the chart's own size, for charts such as heatmaps
    whose cells must not be squeezed.
    """
    p = palette(mode)
    if isinstance(chart, (alt.Chart, alt.LayerChart)) and not fit:
        chart = chart.properties(autosize=alt.AutoSizeParams(type="pad"))
    elif isinstance(chart, (alt.Chart, alt.LayerChart)):
        # Fit the width to the container but keep the height: plain "fit" squeezes titles, legends and axes into the
        # plot height when the chart is stretched to a narrow column.
        chart = chart.properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    else:
        chart = chart.copy(deep=False)
    # The whole configuration is set at once on that copy: each chained configure_* call would copy the chart
    # again, and copying a layered chart combines and deep-copies all of its layers every time.
    chart.config = alt.Config(
        font=FONT_BODY, background=background or "transparent", padding=8,
        axis=alt.AxisConfig(
            labelFont=FONT_MONO, labelFontSize=11, labelColor=p.muted,
            titleFont=FONT_BODY, titleFontSize=12, titleFontWeight="normal", titleColor=p.text,
            gridColor=p.border, gridWidth=0.5, gridOpacity=1,
            domainColor=p.border, domainWidth=1, tickColor=p.border, tickSize=4,
        ),
        axisBand=alt.AxisConfig(labelFont=FONT_BODY, labelColor=p.text, labelFontSize=12),
        title=alt.TitleConfig(
            font=FONT_HEADING, fontSize=14, fontWeight=600, color=p.text, anchor="start",
            subtitleFont=FONT_BODY, subtitleFontSize=11, subtitleColor=p.muted, subtitlePadding=4, offset=10,
        ),
        legend=alt.LegendConfig(
            labelFont=FONT_BODY, labelFontSize=12, labelColor=p.text,
            titleFont=FONT_BODY, titleFontSize=11, titleColor=p.muted, titleFontWeight="normal",
            symbolStrokeWidth=1.6,
        ),
        text=alt.MarkConfig(font=FONT_MONO, color=p.text),
        view=alt.ViewConfig(stroke=None),
    )
    return chart


def _layer(*charts: alt.Chart) -> alt.LayerChart:
    """``alt.layer(*charts)``, with the data every layer shares moved up to the layer chart first.

    Altair moves shared data up itself, but only after it has hashed each layer's data (through the frame's
    printed form) to name the layers, which costs more than the rest of building the chart. Moving it here first
    gives the very same spec without that cost. Layers with data of their own are layered as they are.
    """
    shared = charts[0].data if charts else alt.Undefined
    if shared is alt.Undefined or any(chart.data is not shared for chart in charts):
        return alt.layer(*charts)
    bare = []
    for chart in charts:
        copy = chart.copy(deep=False)
        copy.data = alt.Undefined
        bare.append(copy)
    return alt.layer(*bare, data=shared)


def kind_of(label: str) -> str:
    """Return the traffic kind of a class: ``Normal`` for benign labels, ``Attack`` for everything else."""
    return KIND_NORMAL if is_benign(label) or label == KIND_NORMAL else KIND_ATTACK


def kind_encodings(mode: Mode = "light", *, field: str = "kind") -> tuple[alt.Color, alt.Shape]:
    """Colour and shape encodings for the normal/attack distinction (blue circles and vermilion diamonds)."""
    p = palette(mode)
    domain = [KIND_NORMAL, KIND_ATTACK]
    colour = alt.Color(f"{field}:N", title=None,
                       scale=alt.Scale(domain=domain, range=[p.benign, p.attack]),
                       legend=alt.Legend(orient="top", direction="horizontal"))
    shape = alt.Shape(f"{field}:N", title=None, scale=alt.Scale(domain=domain, range=["circle", "diamond"]),
                      legend=alt.Legend(orient="top", direction="horizontal"))
    return colour, shape


_LABEL_GAP_PX = 9
_LABEL_CHAR_PX = 6.5  # advance width of a 10 px mono digit or comma, rounded up


def _log_floor_with_room(rows: list[dict[str, object]], high: float, width: int) -> float:
    """Lower end of the log axis, low enough that each sampled count fits between the axis and its filled mark.

    On a log axis from ``low`` to ``high`` a value ``v`` sits ``width * log(v / low) / log(high / low)`` pixels from
    the left edge; solving that for the label width (plus a small gap) gives the largest ``low`` that works for
    every class. It is never above the smallest value divided by 1.8, as before labels were placed there.
    """
    values = [float(v) for r in rows for v in (r["before"], r["after"]) if v]
    log_high = np.log(high)
    log_low = np.log(min(values) / 1.8)
    for r in rows:
        if not r["after"]:
            continue
        need = _LABEL_GAP_PX + _LABEL_CHAR_PX * len(str(r["label"])) + 4
        if need >= width:
            continue
        log_low = min(log_low, (width * np.log(float(r["after"])) - need * log_high) / (width - need))
    return float(np.exp(log_low))


def class_distribution_chart(
    before: Mapping[str, int],
    after: Mapping[str, int],
    mode: Mode = "light",
    *,
    title: str = "Rows per class",
    width: int = DEFAULT_WIDTH,
) -> alt.LayerChart:
    """Dot plot of rows per class on a log scale: hollow mark before sampling, filled mark in the sample.

    Classes are ordered by rows available. Normal traffic is a blue circle, attacks vermilion diamonds. The sampled
    count is written just left of the filled mark it belongs to (the space between the axis and the filled mark is
    always free, since a sample never exceeds what is available); the scale starts low enough to leave room for it.
    """
    p = palette(mode)
    classes = sorted(set(before) | set(after), key=lambda c: (-int(before.get(c, 0)), -int(after.get(c, 0)), c))
    classes = classes[:MAX_CHART_ROWS]
    rows = []
    for name in classes:
        b, a = int(before.get(name, 0)), int(after.get(name, 0))
        present = [v for v in (b, a) if v > 0]
        rows.append({
            "class": name, "kind": kind_of(name),
            "before": b if b > 0 else None, "after": a if a > 0 else None,
            "low": min(present) if present else None, "high": max(present) if present else None,
            "label": f"{a:,}",
        })
    data = pd.DataFrame(rows)
    positive = [v for r in rows for v in (r["before"], r["after"]) if v]
    high = max(positive) * 5 if positive else 10
    low = _log_floor_with_room(rows, high, width) if positive else 0.5
    x_scale = alt.Scale(type="log", domain=[low, high], nice=False)
    y = alt.Y("class:N", sort=classes, title=None, axis=alt.Axis(labelLimit=240, ticks=False, domain=False))
    colour, shape = kind_encodings(mode)
    tooltip = [
        alt.Tooltip("class:N", title="Class"),
        alt.Tooltip("before:Q", title="Rows available", format=","),
        alt.Tooltip("after:Q", title="Rows in sample", format=","),
    ]
    base = alt.Chart(data)
    span = base.mark_rule(color=p.muted, strokeWidth=1, opacity=0.7).encode(
        x=alt.X("low:Q", scale=x_scale, title="Rows (log scale)", axis=alt.Axis(format="~s", tickCount=6)),
        x2="high:Q", y=y,
    )
    # Hollow marks are filled with the page colour so the connecting rule does not show through them.
    outline = alt.Stroke("kind:N", legend=None,
                         scale=alt.Scale(domain=[KIND_NORMAL, KIND_ATTACK], range=[p.benign, p.attack]))
    hollow = base.mark_point(filled=True, fill=p.background, size=90, strokeWidth=1.6, opacity=1).encode(
        x=alt.X("before:Q", scale=x_scale), y=y, stroke=outline, shape=shape, tooltip=tooltip,
    )
    filled = base.mark_point(filled=True, size=90, opacity=1).encode(
        x=alt.X("after:Q", scale=x_scale), y=y, color=colour, shape=shape, tooltip=tooltip,
    )
    counts = base.mark_text(align="right", baseline="middle", dx=-_LABEL_GAP_PX, fontSize=10, color=p.muted).encode(
        x=alt.X("after:Q", scale=x_scale), y=y, text="label:N",
    )
    chart = _layer(span, hollow, filled, counts).properties(
        width=width,
        height=max(110, 26 * len(classes)),
        title=alt.Title(title, subtitle="Hollow mark: rows available after cleaning. Filled mark and number: rows in "
                                        "the sample."),
    )
    return base_config(chart, mode)


def chart_spec(build: Callable[[], AnyChart]) -> dict[str, Any]:
    """Build a chart with ``build`` and return its Vega-Lite spec, skipping Altair's schema checks.

    Altair checks every object against the Vega-Lite schema as it is created and the whole spec again when it is
    serialised, which costs several times more than building and drawing the chart. The app draws its charts
    through this function; every builder here is checked against the schema by the test suite instead (and
    :func:`to_png`, which the PDF uses, still validates).
    """
    from altair.utils.schemapi import debug_mode

    with debug_mode(False):
        return build().to_dict(validate=False)


def build_chart(build: Callable[[], AnyChart]) -> AnyChart:
    """Build a chart with ``build`` without Altair's schema checks at each object's creation.

    The checks run again, on the whole spec, when the chart is serialised with validation (as :func:`to_png` does),
    so nothing goes unchecked; skipping the first round keeps a PDF record of many charts quick.
    """
    from altair.utils.schemapi import debug_mode

    with debug_mode(False):
        return build()


def _ensure_fonts() -> None:
    """Register the bundled fonts with vl-convert once per process."""
    global _fonts_registered
    with _fonts_lock:
        if _fonts_registered:
            return
        import vl_convert

        if FONT_DIR.is_dir():
            vl_convert.register_font_directory(str(FONT_DIR))
        _fonts_registered = True


def to_png(chart: AnyChart, scale: float = 2, *, background: str | None = None) -> bytes:
    """Render ``chart`` to PNG bytes with vl-convert (offline, bundled fonts).

    ``background`` overrides the chart's (transparent) background, e.g. the light surface colour for a PDF.
    A chart sized to its container gets a fixed width, since there is no container outside the browser.
    """
    import vl_convert

    _ensure_fonts()
    spec = chart.to_dict()
    if background:
        spec["background"] = background
    if spec.get("width") == "container":
        spec["width"] = DEFAULT_WIDTH
    return vl_convert.vegalite_to_png(spec, scale=scale)


# --------------------------------------------------------------------------------------------------------------
# 03 Measure charts: channel comparisons, confusion matrices, curves, importance, timing, cross-validation
# --------------------------------------------------------------------------------------------------------------
#: Score columns a leaderboard may carry, in the order they are plotted (balanced accuracy first).
SCORE_COLUMNS: tuple[str, ...] = (
    "Balanced accuracy", "Accuracy", "Precision (attack)", "Recall (attack)", "F1 (attack)", "F1 macro",
    "F1 weighted", "Precision macro", "Precision weighted", "Recall macro", "Recall weighted", "ROC-AUC",
    "Average precision",
)
#: Timing columns of a leaderboard: column -> (axis title, number format).
TIMING_MEASURES: dict[str, tuple[str, str]] = {
    "Fit s": ("Fit time (s, log scale)", ",.2f"),
    "Flows/s": ("Flows scored per second (log scale)", ",.0f"),
    "Single-flow ms": ("One flow scored (ms, log scale)", ",.2f"),
}
#: Short titles of the timing readings.
TIMING_TITLES: dict[str, str] = {"Fit s": "Fit time", "Flows/s": "Scoring speed",
                                 "Single-flow ms": "Single-flow latency"}
#: Cross-validation metric titles and the per-fold field each one reads.
CV_METRIC_FIELDS: dict[str, str] = {"Accuracy": "accuracy", "Balanced accuracy": "balanced_accuracy",
                                    "F1 macro": "f1_macro"}
#: Dash patterns cycled through when lines are not channels (per-class curves). Vega needs [1, 0] for solid.
DASH_CYCLE: tuple[tuple[int, ...], ...] = ((1, 0), (8, 4), (2, 3), (6, 3, 2, 3), (10, 5))
_LABEL_ROW_PX = 14
_GUTTER_DX = 18
_UNIT_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _channel_styles(keys: Sequence[str]) -> list[ChannelStyle]:
    """Styles of the given channel keys in the fixed CH1..CH5 order (unknown keys are skipped)."""
    wanted = set(keys)
    return [style for style in CHANNELS if style.key in wanted]


def channel_encodings(mode: Mode, keys: Sequence[str], *, field: str = "channel",
                      legend: bool = False) -> tuple[alt.Color, alt.Shape, alt.StrokeDash]:
    """Colour, marker shape and dash encodings for channel lines and points, keyed on badge-and-name labels.

    The domain is the channels' labels (e.g. ``"CH2 XGBoost"``) in channel order, so every chart draws a channel
    with the same colour, marker and dash. With ``legend`` the colour and shape legends (which Vega-Lite merges
    into one) sit above the chart; the dash pattern never gets a legend of its own.
    """
    styles = _channel_styles(keys)
    labels = [s.label for s in styles]
    shown = (alt.Legend(orient="top", direction="horizontal", title=None, symbolSize=70, columns=3,
                        labelLimit=220, columnPadding=16) if legend else None)
    colour = alt.Color(f"{field}:N", title=None, legend=shown,
                       scale=alt.Scale(domain=labels, range=[s.colour(mode) for s in styles]))
    shape = alt.Shape(f"{field}:N", title=None, legend=shown,
                      scale=alt.Scale(domain=labels, range=[s.marker for s in styles]))
    dash = alt.StrokeDash(f"{field}:N", title=None, legend=None,
                          scale=alt.Scale(domain=labels, range=[list(s.dash) or [1, 0] for s in styles]))
    return colour, shape, dash


def score_domain(values: Sequence[float], *, top: float = 1.0, min_span: float = 0.02) -> list[float]:
    """A zoomed axis domain for scores: from a little below the lowest score to ``top`` (never below 0).

    The lower end leaves a margin of 12 % of the spread (at least half of ``min_span``) and is rounded down to a
    hundredth, so readings close to 1 are spread out without exaggerating tiny gaps.
    """
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return [0.0, float(top)]
    low = min(finite)
    pad = max((float(top) - low) * 0.12, min_span / 2)
    start = max(0.0, float(np.floor((low - pad) * 100) / 100))
    return [start, float(max(top, max(finite)))]


def _spread(values: Sequence[float], gap: float, low: float = 0.0, high: float = 1.0) -> list[float]:
    """Nudge label positions apart so neighbours are at least ``gap`` apart, keeping their order and the range.

    Returns positions in the order of ``values``. Labels are placed from the top down, then pushed back up from
    the bottom if they ran past ``low``; with more labels than the range can hold, the spacing shrinks.
    """
    count = len(values)
    if count == 0:
        return []
    span = high - low
    if count > 1 and gap * (count - 1) > span:
        gap = span / (count - 1)
    order = sorted(range(count), key=lambda i: (-float(values[i]), i))
    placed = [0.0] * count
    previous = None
    for i in order:
        y = min(float(values[i]), high)
        if previous is not None:
            y = min(y, previous - gap)
        placed[i] = y
        previous = y
    following = None
    for i in reversed(order):
        y = max(placed[i], low)
        if following is not None:
            y = max(y, following + gap)
        placed[i] = min(y, high)
        following = placed[i]
    return placed


def _thin(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """At most ``limit`` evenly spaced rows of ``frame``, always keeping the first and the last."""
    if len(frame) <= limit:
        return frame
    picks = np.unique(np.linspace(0, len(frame) - 1, max(limit, 2)).round().astype(np.int64))
    return frame.iloc[picks]


def _gutter_labels(anchors: pd.DataFrame, x_scale: alt.Scale, y_scale: alt.Scale, colour: alt.Color,
                   shape: alt.Shape, *, x_value: float, height: int,
                   text_colour: str | None = None) -> list[alt.Chart]:
    """Line labels drawn just right of the plot, beside the line ends: a marker, then the label text.

    ``anchors`` holds one row per line: the line's name (the field the colour and shape encodings use), ``y``
    (where its end is) and ``text``. Labels are nudged apart vertically so they never overlap, keeping the order
    of the line ends. The text takes the line's colour unless ``text_colour`` is given (for pale line colours).
    """
    domain = y_scale.to_dict().get("domain", [0.0, 1.0])
    low, high = float(domain[0]), float(domain[1])
    gap = (high - low) * _LABEL_ROW_PX / max(height, 1)
    data = anchors.copy()
    data["label_y"] = _spread(list(data["y"].astype(float)), gap, low, high)
    data["label_x"] = float(x_value)
    base = alt.Chart(data)
    marker = base.mark_point(filled=True, size=45, opacity=1).encode(
        x=alt.X("label_x:Q", scale=x_scale), y=alt.Y("label_y:Q", scale=y_scale),
        color=colour, shape=shape, xOffset=alt.XOffset(value=_GUTTER_DX - 9),
    )
    text_mark = base.mark_text(align="left", baseline="middle", dx=_GUTTER_DX, fontSize=11, font=FONT_BODY)
    position = {"x": alt.X("label_x:Q", scale=x_scale), "y": alt.Y("label_y:Q", scale=y_scale), "text": "text:N"}
    if text_colour is None:
        text = text_mark.encode(color=colour, **position)
    else:
        text = text_mark.encode(color=alt.value(text_colour), **position)
    return [marker, text]


def leaderboard_chart(
    board: pd.DataFrame,
    mode: Mode = "light",
    *,
    metrics: Sequence[str] | None = None,
    title: str = "Readings by channel",
    subtitle: str | None = None,
    width: int = DEFAULT_WIDTH,
) -> alt.LayerChart:
    """Dot plot of every channel's scores: one row per metric, one mark per channel (its colour and marker).

    ``board`` is a leaderboard table (:func:`graticule.evaluate.leaderboard`): a ``key`` column plus score columns.
    ``metrics`` picks and orders the score columns (default: every known score column present, balanced accuracy
    first). The score axis is zoomed (:func:`score_domain`) and says so.
    """
    columns = [m for m in (metrics or SCORE_COLUMNS) if m in board.columns]
    keys = [str(k) for k in board["key"]] if "key" in board.columns else []
    styles = _channel_styles(keys)
    labels = {s.key: s.label for s in styles}
    rows = []
    for _, record in board.iterrows():
        key = str(record.get("key", ""))
        if key not in labels:
            continue
        for metric in columns:
            value = record[metric]
            if value is None or not np.isfinite(float(value)):
                continue
            rows.append({"channel": labels[key], "metric": metric, "value": float(value),
                         "shown": f"{float(value):.4f}"})
    data = pd.DataFrame(rows, columns=["channel", "metric", "value", "shown"])
    domain = score_domain(list(data["value"]))
    colour, shape, _ = channel_encodings(mode, keys, legend=True)
    order = [s.label for s in styles]
    x = alt.X("value:Q", scale=alt.Scale(domain=domain, nice=False, clamp=True),
              title=f"Score (axis from {domain[0]:.2f} to {domain[1]:.2f})",
              axis=alt.Axis(format=".2f", tickCount=5, labelFlush=True))
    y = alt.Y("metric:N", sort=columns, title=None,
              axis=alt.Axis(ticks=False, domain=False, labelLimit=200, grid=False, labelPadding=8))
    offset = alt.YOffset("channel:N", sort=order, scale=alt.Scale(domain=order, paddingOuter=0.6))
    tooltip = [alt.Tooltip("channel:N", title="Channel"), alt.Tooltip("metric:N", title="Reading"),
               alt.Tooltip("shown:N", title="Value")]
    points = alt.Chart(data).mark_point(filled=True, size=70, opacity=1, clip=True).encode(
        x=x, y=y, yOffset=offset, color=colour, shape=shape, tooltip=tooltip,
    )
    chart = _layer(points).properties(
        width=width, height=max(140, len(columns) * (9 * max(len(order), 1) + 14)),
        title=alt.Title(title, subtitle=subtitle or "Each mark is one channel's reading on the held-out rows."),
    )
    return base_config(chart, mode)


def share_text(share: float, *, decimals: int = 1) -> str:
    """A row share as a percentage that never rounds a non-zero share to 0 or a partial one to 100.

    ``0.00016`` reads ``"<0.1%"`` and ``0.99984`` reads ``">99.9%"`` (with ``decimals=1``); exact 0 and 1 read
    ``"0%"`` and ``"100%"``.
    """
    value = float(share)
    step = 10.0 ** -(decimals + 2)
    if value <= 0:
        return "0%"
    if value >= 1:
        return "100%"
    if value < step:
        return f"<{step:.{decimals}%}"
    if value > 1 - step:
        return f">{1 - step:.{decimals}%}"
    return f"{value:.{decimals}%}"


def _confusion_cells(counts: np.ndarray, classes: Sequence[str], mode: Mode, show: str) -> pd.DataFrame:
    """One row per confusion-matrix cell with its shade step (0..9), text and text colour."""
    p = palette(mode)
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros(counts.shape, dtype=np.float64), where=totals > 0)
    biggest = float(counts.max()) if counts.size else 0.0
    k = len(classes)
    # Ink lettering on the five lightest steps, light lettering on the five darkest (graphite in dark mode).
    on_light, on_dark = (p.text, p.on_primary) if mode != "dark" else (p.text, DARK.background)
    rows = []
    for i, true_name in enumerate(classes):
        for j, predicted in enumerate(classes):
            count = int(counts[i, j])
            row_share = float(share[i, j])
            if show == "count":
                step = int(min(np.floor(10 * np.log1p(count) / np.log1p(biggest)), 9)) if biggest > 0 else 0
            else:
                step = int(min(np.floor(row_share * 10 + 1e-9), 9))
            shown_share = share_text(row_share)
            count_text = f"{count:,}"
            if k <= 6:
                text = f"{shown_share}\n{count_text}" if show != "count" else f"{count_text}\n{shown_share}"
            else:
                text = "" if count == 0 else (count_text if show == "count" else share_text(row_share, decimals=0))
            ink = on_light if step <= 4 else on_dark
            rows.append({"true": str(true_name), "predicted": str(predicted), "count": count, "share": row_share,
                         "step": step, "text": text, "ink": ink, "share_text": shown_share})
    return pd.DataFrame(rows)


def confusion_chart(
    counts: np.ndarray,
    classes: Sequence[str],
    mode: Mode = "light",
    *,
    show: str = "share",
    title: str | None = None,
    subtitle: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> alt.LayerChart:
    """Confusion-matrix heatmap: rows are the true classes, columns the predicted ones.

    ``show="share"`` shades each cell by its share of the true class's rows (row %), ``show="count"`` by its count
    on a log scale; either way the sequential ramp has ten steps and the cell text switches between ink and light
    lettering by step so it stays readable. With up to six classes a cell shows both numbers (the shaded one
    first); with more it shows the shaded one only and leaves empty cells blank (the tooltip has both).
    """
    matrix = np.asarray(counts, dtype=np.int64)
    names = [str(c) for c in classes]
    k = len(names)
    if matrix.shape != (k, k):
        raise ValueError(f"A {k}-class confusion matrix must be {k} x {k}, got {matrix.shape}.")
    cells = _confusion_cells(matrix, names, mode, show)
    ramp = list(SEQUENTIAL["dark" if mode == "dark" else "light"])
    cell = 84 if k <= 2 else 64 if k <= 4 else 54 if k <= 6 else 34
    angle = 0 if k <= 3 else -35
    x = alt.X("predicted:N", sort=names, title="Predicted class",
              axis=alt.Axis(orient="bottom", labelAngle=angle, labelLimit=120, ticks=False, domain=False))
    y = alt.Y("true:N", sort=names, title="True class", axis=alt.Axis(labelLimit=120, ticks=False, domain=False))
    tooltip = [alt.Tooltip("true:N", title="True class"), alt.Tooltip("predicted:N", title="Predicted"),
               alt.Tooltip("count:Q", title="Rows", format=","), alt.Tooltip("share_text:N", title="Row share")]
    base = alt.Chart(cells)
    rect = base.mark_rect(stroke=palette(mode).background, strokeWidth=1).encode(
        x=x, y=y, tooltip=tooltip,
        color=alt.Color("step:O", legend=None, scale=alt.Scale(domain=list(range(10)), range=ramp)),
    )
    text = base.mark_text(baseline="middle", lineBreak="\n", fontSize=11 if k <= 6 else 9).encode(
        x=x, y=y, text="text:N", color=alt.Color("ink:N", scale=None), tooltip=tooltip,
    )
    shading = "row %" if show != "count" else "count (log scale)"
    # The width covers the class labels and the axis title as well as the cells (the chart fits its width).
    labels_px = min(120, 7 * max((len(n) for n in names), default=4)) + 36
    chart = _layer(rect, text).properties(
        width=width or cell * k + labels_px, height=height or cell * k,
        title=alt.Title(title or "Confusion matrix", subtitle=subtitle or f"Shade: {shading}."),
    )
    return base_config(chart, mode, fit=False)


def _curve_frame(curves: Mapping[str, pd.DataFrame], names: Sequence[str], x_col: str, y_col: str,
                 field: str) -> pd.DataFrame:
    """Curves stacked into one long frame (field, x, y, order), thinned so the whole stays under the row cap."""
    limit = max(20, MAX_CHART_ROWS // max(len(names), 1))
    parts = []
    for key, name in zip(curves, names):
        part = _thin(curves[key][[x_col, y_col]].reset_index(drop=True), limit)
        parts.append(pd.DataFrame({field: name, "x": part[x_col].to_numpy(dtype=float),
                                   "y": part[y_col].to_numpy(dtype=float), "order": np.arange(len(part))}))
    if not parts:
        return pd.DataFrame(columns=[field, "x", "y", "order"])
    return pd.concat(parts, ignore_index=True)


def _end_point(frame: pd.DataFrame, x_col: str, y_col: str, x_at: float | None = None) -> float:
    """Where a curve leaves the plot on the right: its y at ``x_at`` when the curve goes past it, else its end.

    At the end (largest x) the highest y there is taken. ``x_at`` is used for zoomed plots, whose right edge cuts
    the curve; the curve's x values must then be non-decreasing (as for ROC curves).
    """
    if frame.empty:
        return 0.0
    xs = frame[x_col].to_numpy(dtype=float)
    ys = frame[y_col].to_numpy(dtype=float)
    if x_at is not None and xs.max() > x_at:
        return float(np.interp(x_at, xs, ys))
    return float(ys[xs >= xs.max()].max())


def _unit_axis(domain: Sequence[float], title: str) -> alt.Axis:
    """Axis for a rate between 0 and 1: ticks every 0.2 on the full range, five nice ticks on a zoomed one."""
    if float(domain[0]) == 0.0 and float(domain[1]) == 1.0:
        return alt.Axis(format=".1f", values=_UNIT_TICKS, title=title)
    span = float(domain[1]) - float(domain[0])
    return alt.Axis(format=".2f" if span >= 0.05 else ".3f", tickCount=5, labelFlush=True, title=title)


def _guide_layer(guide: pd.DataFrame, mode: Mode, x_scale: alt.Scale, y_scale: alt.Scale, x_axis: alt.Axis,
                 y_axis: alt.Axis) -> alt.Chart:
    """The dashed muted reference line (chance) of a curve plot."""
    return alt.Chart(guide).mark_line(color=palette(mode).muted, strokeDash=[4, 4], strokeWidth=1, opacity=0.8,
                                      clip=True).encode(
        x=alt.X("x:Q", scale=x_scale, axis=x_axis), y=alt.Y("y:Q", scale=y_scale, axis=y_axis))


def roc_zoom_domain(curves: Mapping[str, pd.DataFrame], max_fpr: float) -> list[float]:
    """The true-positive-rate range a ROC plot zoomed to false-positive rates up to ``max_fpr`` needs.

    It starts a little below the lowest rate any curve reaches at ``max_fpr`` (rounded down to a hundredth), so
    every curve still crosses the plot.
    """
    reached = [_end_point(frame, "fpr", "tpr", float(max_fpr)) for frame in curves.values() if not frame.empty]
    if not reached:
        return [0.0, 1.0]
    low = min(reached)
    start = max(0.0, float(np.floor((low - max((1.0 - low) * 0.15, 0.01)) * 100) / 100))
    return [start, 1.0]


def _overlay(
    curves: Mapping[str, pd.DataFrame],
    mode: Mode,
    *,
    kind: str,
    scores: Mapping[str, float] | None,
    chance: float | None,
    title: str,
    subtitle: str,
    width: int,
    height: int,
    x_domain: Sequence[float] = (0.0, 1.0),
    y_domain: Sequence[float] = (0.0, 1.0),
) -> alt.LayerChart:
    """ROC (``kind="roc"``) or precision-recall overlay of channel curves, labelled where they leave the plot."""
    keys = [s.key for s in _channel_styles(list(curves))]
    labels = [CHANNEL_BY_KEY[k].label for k in keys]
    x_col, y_col = ("fpr", "tpr") if kind == "roc" else ("recall", "precision")
    data = _curve_frame({k: curves[k] for k in keys}, labels, x_col, y_col, "channel")
    colour, shape, dash = channel_encodings(mode, keys)
    x_scale = alt.Scale(domain=[float(v) for v in x_domain], nice=False)
    y_scale = alt.Scale(domain=[float(v) for v in y_domain], nice=False)
    x_title = "False-positive rate (normal flows flagged)" if kind == "roc" else "Recall (attacks caught)"
    y_title = "True-positive rate (attacks caught)" if kind == "roc" else "Precision (flagged flows that are attacks)"
    x_axis, y_axis = _unit_axis(x_domain, x_title), _unit_axis(y_domain, y_title)
    layers: list[alt.Chart] = []
    if kind == "roc":
        layers.append(_guide_layer(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}), mode, x_scale, y_scale,
                                   x_axis, y_axis))
    elif chance is not None and np.isfinite(chance):
        layers.append(_guide_layer(pd.DataFrame({"x": [0.0, 1.0], "y": [float(chance)] * 2}), mode, x_scale,
                                   y_scale, x_axis, y_axis))
    layers.append(alt.Chart(data).mark_line(strokeWidth=1.8, clip=True).encode(
        x=alt.X("x:Q", scale=x_scale, axis=x_axis), y=alt.Y("y:Q", scale=y_scale, axis=y_axis),
        color=colour, strokeDash=dash, order="order:Q",
        tooltip=[alt.Tooltip("channel:N", title="Channel"), alt.Tooltip("x:Q", title=x_title, format=".4f"),
                 alt.Tooltip("y:Q", title=y_title, format=".4f")],
    ))
    right = float(x_domain[1])
    anchors = pd.DataFrame({
        "channel": labels,
        "y": [_end_point(curves[k], x_col, y_col, right if kind == "roc" else None) for k in keys],
        "text": [f"{label} {float(scores[k]):.4f}" if scores and k in scores and np.isfinite(scores[k]) else label
                 for k, label in zip(keys, labels)],
    })
    layers.extend(_gutter_labels(anchors, x_scale, y_scale, colour, shape, x_value=right, height=height))
    chart = _layer(*layers).properties(width=width, height=height, title=alt.Title(title, subtitle=subtitle))
    return base_config(chart, mode)


def roc_chart(
    curves: Mapping[str, pd.DataFrame],
    mode: Mode = "light",
    *,
    scores: Mapping[str, float] | None = None,
    max_fpr: float = 1.0,
    title: str = "ROC curves",
    subtitle: str | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = 300,
) -> alt.LayerChart:
    """ROC curves of several channels on one plot (keys of ``curves`` are channel keys; frames hold fpr/tpr).

    Each channel keeps its colour and dash pattern; a label with its marker and (when ``scores`` gives it) its
    ROC-AUC sits at the right edge of the plot, beside where the curve leaves it, nudged apart from the others.
    The dashed muted diagonal is chance. ``max_fpr`` below 1 zooms into the low false-positive corner, where
    near-perfect channels differ (the true-positive axis then starts just below the lowest curve).
    """
    zoomed = float(max_fpr) < 1.0
    x_domain = [0.0, float(max_fpr)] if zoomed else [0.0, 1.0]
    y_domain = roc_zoom_domain(curves, float(max_fpr)) if zoomed else [0.0, 1.0]
    if subtitle is None:
        subtitle = ("Dashed diagonal: a channel that guesses. Number after each name: area under the curve."
                    if not zoomed else f"Zoomed: false-positive rate up to {float(max_fpr):.0%}, true-positive "
                    f"rate from {y_domain[0]:.2f}. Number: whole-curve AUC.")
    return _overlay(curves, mode, kind="roc", scores=scores, chance=None, title=title, subtitle=subtitle,
                    width=width, height=height, x_domain=x_domain, y_domain=y_domain)


def pr_chart(
    curves: Mapping[str, pd.DataFrame],
    mode: Mode = "light",
    *,
    scores: Mapping[str, float] | None = None,
    chance: float | None = None,
    title: str = "Precision-recall curves",
    subtitle: str | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = 300,
) -> alt.LayerChart:
    """Precision-recall curves of several channels on one plot (frames hold recall/precision).

    ``chance`` is the attack share of the held-out rows: the precision of a channel that flags flows at random,
    drawn as a dashed muted line. Labels (with average precision from ``scores``) sit at the right ends.
    """
    note = subtitle or ("Dashed line: precision of flagging flows at random. Number after each name: average "
                        "precision.")
    return _overlay(curves, mode, kind="pr", scores=scores, chance=chance, title=title, subtitle=note,
                    width=width, height=height)


def _class_styles(names: Sequence[str], support: Mapping[str, int] | None,
                  mode: Mode) -> tuple[list[str], list[list[int]], list[str]]:
    """Colours, dash patterns and marker shapes for per-class lines.

    Normal traffic is the benign colour with circles; the four attack classes with most rows take the attack-type
    ramp and the rest the muted colour, all with diamonds. Dash patterns cycle so neighbouring classes differ by
    more than colour.
    """
    p = palette(mode)
    ramp = ATTACK_TYPES["dark" if mode == "dark" else "light"]
    attacks = [n for n in names if kind_of(n) == KIND_ATTACK]
    ranked = sorted(attacks, key=lambda n: (-int((support or {}).get(n, 0)), list(names).index(n)))
    colours: list[str] = []
    dashes: list[list[int]] = []
    shapes: list[str] = []
    attack_index = 0
    for name in names:
        if kind_of(name) == KIND_NORMAL:
            colours.append(p.benign)
            dashes.append([1, 0])
            shapes.append("circle")
            continue
        rank = ranked.index(name)
        colours.append(ramp[rank] if rank < len(ramp) else p.muted)
        attack_index += 1
        dashes.append(list(DASH_CYCLE[attack_index % len(DASH_CYCLE)]))
        shapes.append("diamond")
    return colours, dashes, shapes


def class_curves_chart(
    curves: Mapping[str, pd.DataFrame],
    mode: Mode = "light",
    *,
    kind: str = "roc",
    scores: Mapping[str, float] | None = None,
    support: Mapping[str, int] | None = None,
    max_fpr: float = 1.0,
    title: str | None = None,
    subtitle: str | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = 320,
) -> alt.LayerChart:
    """One channel's one-vs-rest curves, one line per class (keys of ``curves`` are class names).

    ``kind`` is ``"roc"`` (frames hold fpr/tpr) or ``"pr"`` (recall/precision). Normal traffic is drawn in the
    benign colour, the four attack classes with most held-out rows (``support``) in the attack-type ramp, the
    others muted; dash patterns and markers (circle for normal, diamond for attacks) separate them as well. Each
    line is labelled at the right edge with the class name and, from ``scores``, its area under the curve or
    average precision. For ROC curves, ``max_fpr`` below 1 zooms into the low false-positive corner as
    :func:`roc_chart` does.
    """
    names = [str(n) for n in curves]
    x_col, y_col = ("fpr", "tpr") if kind == "roc" else ("recall", "precision")
    data = _curve_frame(curves, names, x_col, y_col, "class")
    colours, dashes, shapes = _class_styles(names, support, mode)
    colour = alt.Color("class:N", legend=None, scale=alt.Scale(domain=names, range=colours))
    dash = alt.StrokeDash("class:N", legend=None, scale=alt.Scale(domain=names, range=dashes))
    shape = alt.Shape("class:N", legend=None, scale=alt.Scale(domain=names, range=shapes))
    zoomed = kind == "roc" and float(max_fpr) < 1.0
    x_domain = [0.0, float(max_fpr)] if zoomed else [0.0, 1.0]
    y_domain = roc_zoom_domain(curves, float(max_fpr)) if zoomed else [0.0, 1.0]
    x_scale = alt.Scale(domain=x_domain, nice=False)
    y_scale = alt.Scale(domain=y_domain, nice=False)
    x_title = "False-positive rate" if kind == "roc" else "Recall"
    y_title = "True-positive rate" if kind == "roc" else "Precision"
    layers: list[alt.Chart] = []
    x_axis, y_axis = _unit_axis(x_domain, x_title), _unit_axis(y_domain, y_title)
    if kind == "roc":
        layers.append(_guide_layer(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}), mode, x_scale, y_scale,
                                   x_axis, y_axis))
    layers.append(alt.Chart(data).mark_line(strokeWidth=1.5, clip=True).encode(
        x=alt.X("x:Q", scale=x_scale, axis=x_axis), y=alt.Y("y:Q", scale=y_scale, axis=y_axis),
        color=colour, strokeDash=dash, order="order:Q",
        tooltip=[alt.Tooltip("class:N", title="Class"), alt.Tooltip("x:Q", title=x_title, format=".4f"),
                 alt.Tooltip("y:Q", title=y_title, format=".4f")],
    ))
    number = "AUC" if kind == "roc" else "AP"
    anchors = pd.DataFrame({
        "class": names,
        "y": [_end_point(curves[n], x_col, y_col, x_domain[1] if kind == "roc" else None) for n in names],
        "text": [f"{n} {float(scores[n]):.4f}" if scores and n in scores and np.isfinite(scores[n]) else n
                 for n in names],
    })
    layers.extend(_gutter_labels(anchors, x_scale, y_scale, colour, shape, x_value=x_domain[1], height=height,
                                 text_colour=palette(mode).text))
    default_title = "One-vs-rest ROC curves" if kind == "roc" else "One-vs-rest precision-recall curves"
    if subtitle is not None:
        note = subtitle
    elif zoomed:
        note = (f"Zoomed: false-positive rate up to {float(max_fpr):.0%}, true-positive rate from "
                f"{y_domain[0]:.2f}. Number: whole-curve AUC.")
    else:
        note = (f"Each class against all the others. Number after each name: {number}. Circle: normal "
                "traffic; diamonds: attack classes.")
    chart = _layer(*layers).properties(width=width, height=height,
                                          title=alt.Title(title or default_title, subtitle=note))
    return base_config(chart, mode)


def importance_chart(
    frame: pd.DataFrame,
    mode: Mode = "light",
    *,
    value: str = "importance",
    error: str | None = None,
    top: int = 20,
    title: str = "Feature importance",
    subtitle: str | None = None,
    x_title: str = "Share of total importance",
    colour: str | None = None,
    number_format: str = ".3f",
    width: int = DEFAULT_WIDTH,
) -> alt.LayerChart:
    """Horizontal bars of the ``top`` features by ``value`` (largest first), with the value written at each bar.

    ``error`` names a column of spreads drawn as a thin rule of +/- one spread around the value (permutation
    importance). Bars are clipped to the plot (``clip=True``), and the axis includes 0 and any negative values.
    """
    p = palette(mode)
    data = frame[["feature", value] + ([error] if error else [])].copy()
    data = data.sort_values(value, ascending=False, kind="stable").head(int(top)).reset_index(drop=True)
    data["shown"] = [format(float(v), number_format) for v in data[value]]
    data["v"] = data[value].astype(float)
    spread = data[error].astype(float).fillna(0.0) if error else 0.0
    data["lo"] = data["v"] - spread
    data["hi"] = data["v"] + spread
    low = float(min(0.0, data["lo"].min() if len(data) else 0.0))
    high = float(max(data["hi"].max() if len(data) else 0.0, 0.0))
    high = high + (high - low) * 0.18 if high > low else 1.0
    order = list(data["feature"])
    x_scale = alt.Scale(domain=[low, high], nice=False)
    y = alt.Y("feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, ticks=False, domain=False))
    tooltip = [alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("shown:N", title=x_title)]
    base = alt.Chart(data)
    bars = base.mark_bar(clip=True, color=colour or p.primary, height={"band": 0.7}).encode(
        x=alt.X("v:Q", scale=x_scale, title=x_title), y=y, tooltip=tooltip,
    )
    layers = [bars]
    if error:
        layers.append(base.mark_rule(clip=True, color=p.text, strokeWidth=1).encode(
            x=alt.X("lo:Q", scale=x_scale), x2="hi:Q", y=y))
    layers.append(base.mark_text(align="left", baseline="middle", dx=4, fontSize=10, color=p.muted).encode(
        x=alt.X("hi:Q", scale=x_scale), y=y, text="shown:N"))
    chart = _layer(*layers).properties(
        width=width, height=max(90, 18 * len(order) + 10),
        title=alt.Title(title, subtitle=subtitle or ""),
    )
    return base_config(chart, mode)


def _timing_layer(board: pd.DataFrame, mode: Mode, measure: str, width: int) -> alt.LayerChart:
    """Unconfigured dot plot of one timing column per channel (log scale, value written right of the mark)."""
    p = palette(mode)
    axis_title, number = TIMING_MEASURES.get(measure, (measure, ",.2f"))
    keys = [str(k) for k in board["key"]]
    styles = _channel_styles(keys)
    by_key = {k: float(v) for k, v in zip(keys, board[measure])}
    rows = [{"channel": s.label, "value": by_key[s.key], "shown": format(by_key[s.key], number)}
            for s in styles if np.isfinite(by_key.get(s.key, np.nan)) and by_key[s.key] > 0]
    data = pd.DataFrame(rows, columns=["channel", "value", "shown"])
    order = [s.label for s in styles]
    colour, shape, _ = channel_encodings(mode, keys)
    values = list(data["value"]) or [1.0]
    floor = min(values) / 2.5
    x_scale = alt.Scale(type="log", domain=[floor, max(values) * 6], nice=False)
    y = alt.Y("channel:N", sort=order, title=None, axis=alt.Axis(labelLimit=200, ticks=False, domain=False))
    base = alt.Chart(data)
    tick_format = "~s" if measure == "Flows/s" else "~g"
    span = base.mark_rule(color=p.border, strokeWidth=1).encode(
        x=alt.X("value:Q", scale=x_scale, title=axis_title,
                axis=alt.Axis(format=tick_format, labelOverlap="greedy", tickCount=5)),
        x2=alt.X2(datum=floor), y=y)
    dots = base.mark_point(filled=True, size=80, opacity=1).encode(
        x=alt.X("value:Q", scale=x_scale), y=y, color=colour, shape=shape,
        tooltip=[alt.Tooltip("channel:N", title="Channel"), alt.Tooltip("shown:N", title=axis_title)])
    labels = base.mark_text(align="left", baseline="middle", dx=9, fontSize=10, color=p.muted).encode(
        x=alt.X("value:Q", scale=x_scale), y=y, text="shown:N")
    return _layer(span, dots, labels).properties(width=width, height=max(90, 26 * len(order)))


def timing_chart(
    board: pd.DataFrame,
    mode: Mode = "light",
    *,
    measure: str = "Fit s",
    title: str | None = None,
    subtitle: str | None = None,
    width: int = DEFAULT_WIDTH,
) -> alt.LayerChart:
    """One timing reading per channel (``measure`` is ``"Fit s"``, ``"Flows/s"`` or ``"Single-flow ms"``).

    Times differ by orders of magnitude between channels, so the axis is logarithmic and each value is written
    beside its mark.
    """
    chart = _timing_layer(board, mode, measure, width).properties(
        title=alt.Title(title or TIMING_TITLES.get(measure, measure), subtitle=subtitle or ""))
    return base_config(chart, mode)


def timing_panels(
    board: pd.DataFrame,
    mode: Mode = "light",
    *,
    measures: Sequence[str] = ("Fit s", "Flows/s"),
    title: str = "Timing",
    width: int = 260,
) -> alt.HConcatChart:
    """Several timing readings side by side at fixed widths (for the PDF record)."""
    panels = [_timing_layer(board, mode, m, width).properties(title=TIMING_TITLES.get(m, m)) for m in measures]
    chart = alt.hconcat(*panels, spacing=24).properties(title=alt.Title(title))
    return base_config(chart, mode)


def cv_spread_chart(
    summary: pd.DataFrame,
    mode: Mode = "light",
    *,
    metric: str = "Balanced accuracy",
    folds: pd.DataFrame | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    width: int = DEFAULT_WIDTH,
) -> alt.LayerChart:
    """Cross-validation spread per channel: mean (channel marker) with a bar of +/- one standard deviation.

    ``summary`` is a cross-validation table (:func:`graticule.evaluate.cross_validate_run`) with ``key`` and
    ``"<metric> mean"``/``"<metric> std"`` columns; ``folds`` (optional, :func:`graticule.evaluate.cv_fold_frame`)
    adds each fold's reading as a small hollow mark. The axis is zoomed and says so.
    """
    p = palette(mode)
    mean_col, std_col = f"{metric} mean", f"{metric} std"
    keys = [str(k) for k in summary["key"]] if "key" in summary.columns else []
    styles = _channel_styles(keys)
    by_key = {str(r["key"]): r for _, r in summary.iterrows()}
    rows = []
    for style in styles:
        record = by_key[style.key]
        mean = float(record[mean_col])
        if not np.isfinite(mean):
            continue
        std = float(record[std_col])
        std = std if np.isfinite(std) else 0.0
        rows.append({"channel": style.label, "mean": mean, "lo": mean - std, "hi": min(mean + std, 1.0),
                     "shown": f"{mean:.4f} ± {std:.4f}"})
    data = pd.DataFrame(rows, columns=["channel", "mean", "lo", "hi", "shown"])
    field = CV_METRIC_FIELDS.get(metric)
    fold_data = pd.DataFrame({"channel": pd.Series([], dtype="str"), "value": pd.Series([], dtype="float64")})
    if folds is not None and field is not None and not folds.empty and field in folds.columns:
        labels = {s.key: s.label for s in styles}
        fold_data = pd.DataFrame({"channel": [labels.get(str(k)) for k in folds["key"]],
                                  "value": folds[field].astype(float).to_numpy()}).dropna()
    values = list(data["lo"]) + list(fold_data["value"]) + list(data["mean"])
    domain = score_domain(values)
    order = [s.label for s in styles]
    colour, shape, _ = channel_encodings(mode, keys)
    x_scale = alt.Scale(domain=domain, nice=False, clamp=True)
    y = alt.Y("channel:N", sort=order, title=None, axis=alt.Axis(labelLimit=200, ticks=False, domain=False))
    base = alt.Chart(data)
    spread = base.mark_rule(strokeWidth=2.5, clip=True).encode(
        x=alt.X("lo:Q", scale=x_scale, title=f"{metric} across folds (axis from {domain[0]:.2f})",
                axis=alt.Axis(format=".2f", tickCount=5, labelFlush=True)),
        x2="hi:Q", y=y, color=colour)
    fold_marks = alt.Chart(fold_data).mark_point(filled=False, size=28, strokeWidth=1, opacity=0.7,
                                                 clip=True).encode(
        x=alt.X("value:Q", scale=x_scale), y=y, color=colour)
    means = base.mark_point(filled=True, size=90, opacity=1, clip=True).encode(
        x=alt.X("mean:Q", scale=x_scale), y=y, color=colour, shape=shape,
        tooltip=[alt.Tooltip("channel:N", title="Channel"), alt.Tooltip("shown:N", title=f"{metric} (mean ± sd)")])
    edge = data.assign(edge=domain[1])
    text = alt.Chart(edge).mark_text(align="left", baseline="middle", dx=12, fontSize=10, color=p.text).encode(
        x=alt.X("edge:Q", scale=x_scale), y=y, text="shown:N")
    chart = _layer(spread, fold_marks, means, text).properties(
        width=width, height=max(110, 34 * len(order)),
        title=alt.Title(title or f"Cross-validation: {metric.lower()}",
                        subtitle=subtitle or "Filled mark: mean over folds; bar: one standard deviation each side; "
                                             "hollow marks: single folds. Right: mean ± sd."),
    )
    return base_config(chart, mode)


def held_out_classes_chart(
    counts: Mapping[str, int],
    mode: Mode = "light",
    *,
    title: str = "Held-out rows per class",
    subtitle: str | None = None,
    width: int = DEFAULT_WIDTH,
) -> alt.LayerChart:
    """The test-set class distribution: one mark per class on a log scale, the count written beside it.

    Normal traffic is a blue circle, attack classes vermilion diamonds. Classes are ordered by row count.
    """
    p = palette(mode)
    items = sorted(((str(k), int(v)) for k, v in counts.items() if int(v) > 0), key=lambda t: (-t[1], t[0]))
    items = items[:MAX_CHART_ROWS]
    data = pd.DataFrame({"class": [k for k, _ in items], "rows": [v for _, v in items],
                         "kind": [kind_of(k) for k, _ in items], "label": [f"{v:,}" for _, v in items]})
    high = float(max([v for _, v in items], default=10)) * 8
    x_scale = alt.Scale(type="log", domain=[0.5, high], nice=False)
    order = [k for k, _ in items]
    y = alt.Y("class:N", sort=order, title=None, axis=alt.Axis(labelLimit=240, ticks=False, domain=False))
    colour, shape = kind_encodings(mode)
    base = alt.Chart(data)
    rule = base.mark_rule(color=p.border, strokeWidth=1).encode(
        x=alt.X("rows:Q", scale=x_scale, title="Held-out rows (log scale)", axis=alt.Axis(format="~s")),
        x2=alt.X2(datum=0.5), y=y)
    dots = base.mark_point(filled=True, size=80, opacity=1).encode(
        x=alt.X("rows:Q", scale=x_scale), y=y, color=colour, shape=shape,
        tooltip=[alt.Tooltip("class:N", title="Class"), alt.Tooltip("rows:Q", title="Held-out rows", format=",")])
    text = base.mark_text(align="left", baseline="middle", dx=9, fontSize=10, color=p.muted).encode(
        x=alt.X("rows:Q", scale=x_scale), y=y, text="label:N")
    chart = _layer(rule, dots, text).properties(
        width=width, height=max(80, 24 * len(items)),
        title=alt.Title(title, subtitle=subtitle or "Every reading on this station is measured on these rows."),
    )
    return base_config(chart, mode)


# --------------------------------------------------------------------------------------------------------------
# 04 Probe charts: what moved one channel's reading of one flow
# --------------------------------------------------------------------------------------------------------------
def value_text(value: float) -> str:
    """A feature value for a chart label: whole numbers with thousands separators, other values of 1,000 or more
    with one decimal, smaller ones with four significant digits; ``"missing"`` for NaN and ``"inf"``/``"-inf"``
    for infinities (and scientific notation from 10^15 up)."""
    number = float(value)
    if np.isnan(number):
        return "missing"
    if np.isinf(number):
        return "inf" if number > 0 else "-inf"
    if abs(number) >= 1e15:
        return f"{number:.3e}"
    if number.is_integer():
        return f"{int(number):,}"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    return f"{number:.4g}"


def contribution_chart(
    frame: pd.DataFrame,
    mode: Mode = "light",
    *,
    top: int = 12,
    toward_label: str = "Towards attack",
    away_label: str = "Towards normal",
    toward_is_normal: bool = False,
    title: str = "What moved the reading",
    subtitle: str | None = None,
    x_title: str = "Contribution",
    number_format: str = "+.3f",
    width: int = DEFAULT_WIDTH,
) -> alt.LayerChart:
    """Diverging bars of one flow's feature contributions: the ``top`` largest by size, largest first.

    ``frame`` holds ``feature``, ``value`` (the flow's own value, written after the feature name) and
    ``contribution`` (positive pushes towards the explained class). Bars right of the zero line push towards it
    (``toward_label``), bars left of it push away (``away_label``). Bars pushing towards attack are vermilion and
    bars pushing towards normal blue, the two ends of the diverging ramp (``toward_is_normal`` swaps them when the
    explained class is the normal one). Every bar carries its signed value, so the direction never rests on colour
    alone. Bars are clipped to the plot.
    """
    from graticule.theme import DIVERGING

    p = palette(mode)
    ramp = DIVERGING["dark" if mode == "dark" else "light"]
    towards_normal, towards_attack = ramp[0], ramp[-1]
    toward_colour, away_colour = ((towards_normal, towards_attack) if toward_is_normal
                                  else (towards_attack, towards_normal))
    data = frame[["feature", "value", "contribution"]].copy()
    data["c"] = data["contribution"].astype(float)
    data["size"] = data["c"].abs()
    data = data.sort_values("size", ascending=False, kind="stable").head(max(int(top), 1)).reset_index(drop=True)
    data["label"] = [f"{name} = {value_text(v)}" for name, v in zip(data["feature"], data["value"])]
    data["value_text"] = [value_text(v) for v in data["value"]]
    data["shown"] = [format(float(v), number_format) for v in data["c"]]
    data["direction"] = np.where(data["c"] >= 0, toward_label, away_label)
    data = data.drop(columns=["value", "contribution"])
    low = min(0.0, float(data["c"].min()) if len(data) else 0.0)
    high = max(0.0, float(data["c"].max()) if len(data) else 0.0)
    span = high - low
    if span <= 0:
        low, high = -1.0, 1.0
    else:
        low = low - 0.24 * span if low < 0 else low
        high = high + 0.24 * span if high > 0 else high
    x_scale = alt.Scale(domain=[low, high], nice=False)
    order = list(data["label"])
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300, ticks=False, domain=False))
    colour = alt.Color("direction:N", title=None,
                       scale=alt.Scale(domain=[toward_label, away_label], range=[toward_colour, away_colour]),
                       legend=alt.Legend(orient="top", direction="horizontal", labelLimit=320))
    tooltip = [alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("value_text:N", title="Value in this flow"),
               alt.Tooltip("shown:N", title=x_title)]
    base = alt.Chart(data)
    bars = base.mark_bar(clip=True, height={"band": 0.7}).encode(
        x=alt.X("c:Q", scale=x_scale, title=x_title, axis=alt.Axis(tickCount=5, labelFlush=True)),
        x2=alt.X2(datum=0), y=y, color=colour, tooltip=tooltip,
    )
    right = base.transform_filter(alt.datum.c >= 0).mark_text(
        align="left", baseline="middle", dx=4, fontSize=10, color=p.text).encode(
        x=alt.X("c:Q", scale=x_scale), y=y, text="shown:N")
    left = base.transform_filter(alt.datum.c < 0).mark_text(
        align="right", baseline="middle", dx=-4, fontSize=10, color=p.text).encode(
        x=alt.X("c:Q", scale=x_scale), y=y, text="shown:N")
    zero = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(color=p.text, strokeWidth=1).encode(
        x=alt.X("zero:Q", scale=x_scale))
    chart = alt.layer(_layer(bars, right, left), zero).properties(
        width=width, height=max(90, 22 * len(order) + 10), title=alt.Title(title, subtitle=subtitle or ""),
    )
    return base_config(chart, mode)


# --------------------------------------------------------------------------------------------------------------
# 06 Sweep: detections over time
# --------------------------------------------------------------------------------------------------------------
#: Ticks shown at once by :func:`detections_chart`.
SWEEP_WINDOW = 120
#: The two parts of a detections bar, bottom first.
SWEEP_PARTS: tuple[str, str] = ("Read as attack", "Read as normal")


def _tint(colour: str, background: str, strength: float) -> str:
    """``colour`` laid over ``background`` at ``strength`` (0..1) as an opaque hex colour, e.g. a pale fill."""
    def channels(value: str) -> list[int]:
        text = value.lstrip("#")
        return [int(text[i:i + 2], 16) for i in (0, 2, 4)]

    mixed = [round(b + (c - b) * float(strength)) for c, b in zip(channels(colour), channels(background))]
    return "#" + "".join(f"{min(max(v, 0), 255):02X}" for v in mixed)


def detections_chart(
    timeline: pd.DataFrame,
    mode: Mode = "light",
    *,
    window: int = SWEEP_WINDOW,
    title: str = "Detections over time",
    subtitle: str | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = 220,
) -> alt.LayerChart:
    """Flows per tick of a live stream, split by the channel's verdict, over the latest ``window`` ticks.

    ``timeline`` holds one row per tick with ``tick``, ``normal``, ``attack_predicted`` and ``alerts`` counts (the
    shape of :attr:`graticule.simulate.SimulationSession.timeline`). Each tick is one stacked bar: flows read as
    attack at the bottom (solid vermilion), flows read as normal above them (pale blue with an outline, like the
    hollow normal mark elsewhere). A warning-coloured triangle sits on each tick that raised alerts and a brass rule
    marks the newest tick. The axis always spans ``window`` ticks: it fills from the left, then scrolls.
    """
    p = palette(mode)
    span = max(int(window), 1)
    ticks = timeline.sort_values("tick").tail(span) if len(timeline) else timeline
    now = int(ticks["tick"].max()) if len(ticks) else 0
    first = max(1, now - span + 1)
    last = first + span - 1
    rows: list[dict[str, object]] = []
    marks: list[dict[str, object]] = []
    top = 1.0
    for record in ticks.itertuples(index=False):
        tick = int(getattr(record, "tick"))
        attack = int(getattr(record, "attack_predicted"))
        normal = int(getattr(record, "normal"))
        alerts = int(getattr(record, "alerts"))
        top = max(top, float(attack + normal))
        for part, lo, hi in ((SWEEP_PARTS[0], 0, attack), (SWEEP_PARTS[1], attack, attack + normal)):
            if hi > lo:
                rows.append({"tick": tick, "x0": tick - 0.42, "x1": tick + 0.42, "part": part, "lo": lo, "hi": hi,
                             "flows": hi - lo})
        if alerts > 0:
            marks.append({"tick": tick, "total": attack + normal, "alerts": alerts})
    bars_data = pd.DataFrame(rows, columns=["tick", "x0", "x1", "part", "lo", "hi", "flows"])
    alert_data = pd.DataFrame(marks, columns=["tick", "total", "alerts"])
    x_scale = alt.Scale(domain=[first - 0.5, last + 0.5], nice=False, zero=False)
    y_scale = alt.Scale(domain=[0.0, top * 1.22], nice=False)
    x_axis = alt.Axis(format="d", tickMinStep=1, labelFlush=True, labelOverlap="greedy")
    # The normal part is a pale tint of the benign colour with a full-strength outline (like the hollow normal
    # mark elsewhere); fill and outline share one legend, so its swatches look like the bars.
    legend = alt.Legend(orient="top", direction="horizontal", symbolType="square", symbolStrokeWidth=1.2)
    domain = list(SWEEP_PARTS)
    colour = alt.Color("part:N", title=None, legend=legend,
                       scale=alt.Scale(domain=domain, range=[p.attack, _tint(p.benign, p.background, 0.3)]))
    outline = alt.Stroke("part:N", title=None, legend=legend,
                         scale=alt.Scale(domain=domain, range=[p.attack, p.benign]))
    bars = alt.Chart(bars_data).mark_bar(strokeWidth=1, clip=True).encode(
        x=alt.X("x0:Q", scale=x_scale, title="Tick", axis=x_axis), x2="x1:Q",
        y=alt.Y("lo:Q", scale=y_scale, title="Flows per tick", axis=alt.Axis(format="~s", tickCount=4)),
        y2="hi:Q", color=colour, stroke=outline,
        tooltip=[alt.Tooltip("tick:Q", title="Tick"), alt.Tooltip("part:N", title="Verdict"),
                 alt.Tooltip("flows:Q", title="Flows", format=",")],
    )
    triangles = alt.Chart(alert_data).mark_point(shape="triangle-up", filled=True, size=70, fill=p.warning,
                                                 stroke=p.text, strokeWidth=0.8, opacity=1, yOffset=-8,
                                                 clip=True).encode(
        x=alt.X("tick:Q", scale=x_scale), y=alt.Y("total:Q", scale=y_scale),
        tooltip=[alt.Tooltip("tick:Q", title="Tick"), alt.Tooltip("alerts:Q", title="Alerts", format=",")],
    )
    cursor_data = pd.DataFrame({"tick": [now] if now else [], "label": [f"now: tick {now:,}"] if now else []})
    cursor = alt.Chart(cursor_data).mark_rule(color=p.secondary, strokeWidth=2).encode(
        x=alt.X("tick:Q", scale=x_scale))
    left_side = now - first < span * 0.6
    cursor_text = alt.Chart(cursor_data).mark_text(align="left" if left_side else "right", baseline="top",
                                                   dx=5 if left_side else -5, dy=2, fontSize=10,
                                                   font=FONT_BODY, color=p.secondary).encode(
        x=alt.X("tick:Q", scale=x_scale), y=alt.value(0), text="label:N")
    chart = alt.layer(bars, triangles, cursor, cursor_text).properties(
        width=width, height=int(height),
        title=alt.Title(title, subtitle=subtitle if subtitle is not None else (
            "One bar per tick. Triangles: ticks with alerts. Brass line: the latest tick.")),
    )
    return base_config(chart, mode)
