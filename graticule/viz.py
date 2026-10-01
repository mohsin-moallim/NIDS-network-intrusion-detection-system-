"""Chart builders shared by the UI and the PDF record.

Every builder takes plain data plus a theme mode and returns an Altair chart with its data inlined, so the same
definition renders in the browser (``st.altair_chart(chart, theme=None)``) and as a PNG for the PDF
(:func:`to_png`, through vl-convert, with no browser involved). Colours and fonts come from :mod:`graticule.theme`.
Normal traffic is always drawn with circles and attacks with diamonds, so colour is never the only cue.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import altair as alt
import numpy as np
import pandas as pd

from graticule.schema import is_benign
from graticule.theme import FONT_BODY, FONT_HEADING, FONT_MONO, Mode, palette

AnyChart = Union[alt.Chart, alt.LayerChart, alt.VConcatChart, alt.HConcatChart, alt.ConcatChart, alt.FacetChart]
FONT_DIR = Path(__file__).resolve().parent.parent / "static" / "fonts"
MAX_CHART_ROWS = 5_000
DEFAULT_WIDTH = 560
KIND_NORMAL = "Normal"
KIND_ATTACK = "Attack"

_fonts_lock = threading.Lock()
_fonts_registered = False


def base_config(chart: AnyChart, mode: Mode = "light", *, background: str | None = None) -> AnyChart:
    """Apply Graticule's chart styling: body font for text, Instrument Sans titles, mono axis labels, hairline grid.

    The background stays transparent unless ``background`` is given, so the chart sits on the page colour in both
    themes. Returns a configured copy (``chart`` itself is not changed).
    """
    p = palette(mode)
    if isinstance(chart, (alt.Chart, alt.LayerChart)):
        # Fit the width to the container but keep the height: plain "fit" squeezes titles, legends and axes into the
        # plot height when the chart is stretched to a narrow column.
        chart = chart.properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    return (
        chart.configure(font=FONT_BODY, background=background or "transparent", padding=8)
        .configure_axis(
            labelFont=FONT_MONO, labelFontSize=11, labelColor=p.muted,
            titleFont=FONT_BODY, titleFontSize=12, titleFontWeight="normal", titleColor=p.text,
            gridColor=p.border, gridWidth=0.5, gridOpacity=1,
            domainColor=p.border, domainWidth=1, tickColor=p.border, tickSize=4,
        )
        .configure_axisBand(labelFont=FONT_BODY, labelColor=p.text, labelFontSize=12)
        .configure_title(
            font=FONT_HEADING, fontSize=14, fontWeight=600, color=p.text, anchor="start",
            subtitleFont=FONT_BODY, subtitleFontSize=11, subtitleColor=p.muted, subtitlePadding=4, offset=10,
        )
        .configure_legend(
            labelFont=FONT_BODY, labelFontSize=12, labelColor=p.text,
            titleFont=FONT_BODY, titleFontSize=11, titleColor=p.muted, titleFontWeight="normal",
            symbolStrokeWidth=1.6,
        )
        .configure_text(font=FONT_MONO, color=p.text)
        .configure_view(stroke=None)
    )


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
    chart = alt.layer(span, hollow, filled, counts).properties(
        width=width,
        height=max(110, 26 * len(classes)),
        title=alt.Title(title, subtitle="Hollow mark: rows available after cleaning. Filled mark and number: rows in "
                                        "the sample."),
    )
    return base_config(chart, mode)


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
