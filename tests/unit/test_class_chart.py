"""The 01 Sample class chart keeps every sampled count legible at any width: the counts stand in a column of their
own right of the plot, so they never meet the marks or the class names, in a narrow browser column or in the PDF.

The checks read the layout Vega itself computes (through vl-convert's scenegraph, offline), at the widths a narrow
container hands the chart and at the fixed width the PDF renders it with.
"""

from __future__ import annotations

from typing import Any

import pytest

from graticule import viz

pytestmark = pytest.mark.unit

# Class totals shaped like a whole-week draw: long class names, sampled counts from two to six digits.
BEFORE = {"BENIGN": 2_096_484, "DoS Hulk": 172_849, "PortScan": 90_819, "Web Attack - Brute Force": 1_470,
          "Infiltration": 36, "Web Attack - Sql Injection": 21, "Heartbleed": 11}
AFTER = {"BENIGN": 109_100, "DoS Hulk": 9_010, "PortScan": 4_790, "Web Attack - Brute Force": 1_470,
         "Infiltration": 36, "Web Attack - Sql Injection": 21, "Heartbleed": 11}
DIGIT_PX = 6.5  # advance width of a 10 px mono digit or comma, rounded up (as the chart assumes)


def _text_marks(node: Any, found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every text mark of a Vega scenegraph, depth first."""
    if isinstance(node, dict):
        if node.get("marktype") == "text":
            found.append(node)
        for item in node.get("items", []):
            _text_marks(item, found)
    return found


def _layout(width: int | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Lay the chart out with Vega at a container ``width`` (None: the chart's own width, as in the PDF).

    Returns the scenegraph, the count labels and the class-name labels.
    """
    import vl_convert

    viz._ensure_fonts()  # measure with the bundled fonts, as the browser and the PDF do
    spec = viz.class_distribution_chart(BEFORE, AFTER, "light").to_dict()
    if width is not None:
        spec["width"] = width  # what Streamlit hands a chart drawn with width="stretch" (autosize fit-x)
    scene = vl_convert.vegalite_to_scenegraph(spec)
    marks = _text_marks(scene["scenegraph"], [])
    counts = next(m for m in marks if m.get("role") == "mark")["items"]
    names = [item for m in marks if m.get("role") == "axis-label" for item in m["items"]
             if item.get("text") in BEFORE]
    return scene, counts, names


@pytest.mark.parametrize("width", [240, 320, None], ids=["narrow-240", "column-320", "pdf"])
def test_counts_stand_right_of_the_plot_and_inside_the_chart(width: int | None) -> None:
    scene, counts, names = _layout(width)
    if width is not None:
        assert scene["width"] == width
    plot_width = counts[0]["x"]  # the column is anchored on the plot's right edge
    assert plot_width > 0 and all(item["x"] == plot_width for item in counts)
    assert [item["text"] for item in counts] == [f"{AFTER[c]:,}" for c in BEFORE]
    for item in counts:
        assert item["align"] == "right"
        right = item["x"] + item["dx"]
        left = right - DIGIT_PX * len(item["text"])
        assert left > plot_width, item  # clear of every mark, so of the class names left of the plot too
        assert scene["origin"][0] + right <= scene["width"], item  # inside the chart, not cut off
    # Class names end left of the plot, level with their counts.
    assert len(names) == len(BEFORE)
    for name, count in zip(sorted(names, key=lambda i: i["y"]), counts, strict=True):
        assert name["align"] == "right" and name["x"] <= 0
        assert abs(name["y"] - count["y"]) < 2


def test_a_class_the_sample_lacks_prints_zero() -> None:
    spec = viz.class_distribution_chart({"BENIGN": 100, "Bot": 5}, {"BENIGN": 50}, "light").to_dict()
    data = next(iter(spec["datasets"].values()))
    assert [(row["class"], row["label"], row["after"]) for row in data] == [("BENIGN", "50", 50), ("Bot", "0", None)]


def test_subtitle_is_split_so_it_fits_a_narrow_column() -> None:
    title = viz.class_distribution_chart(BEFORE, AFTER, "light").to_dict()["title"]
    assert isinstance(title["subtitle"], list) and all(len(line) <= 46 for line in title["subtitle"])
