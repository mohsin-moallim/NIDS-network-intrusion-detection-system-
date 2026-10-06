"""Visual tokens for NIDS, kept in one place so the UI, the charts and the PDF agree.

Streamlit only exposes a handful of theme keys, so every colour the app needs (semantic colours for normal and
attack traffic, channel styles, chart ramps) is defined here and read by whichever layer draws something. So is the
rule for printing a score (:func:`score_text`): four decimals, and never a perfect 1.0000 for a reading short of 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Mode = Literal["light", "dark"]


@dataclass(frozen=True)
class Palette:
    """Base colour tokens for one theme mode."""

    background: str
    surface: str
    border: str
    text: str
    muted: str
    primary: str
    on_primary: str
    secondary: str
    benign: str
    attack: str
    warning: str
    warning_tint: str
    warning_text: str


LIGHT = Palette(
    background="#F3F1EA",
    surface="#FBFAF6",
    border="#D5D1C6",
    text="#1B2229",
    muted="#5A6470",
    primary="#2A6B5A",
    on_primary="#FFFFFF",
    secondary="#8C6620",
    benign="#1F5B9C",
    attack="#A5341A",
    warning="#B08200",
    warning_tint="#FBEFC6",
    warning_text="#6B4E00",
)

DARK = Palette(
    background="#12171C",
    surface="#1A2027",
    border="#2E3740",
    text="#E4E8EC",
    muted="#98A3AE",
    primary="#2E7A67",
    on_primary="#FFFFFF",
    secondary="#C49A55",
    benign="#62A2DE",
    attack="#F2845E",
    warning="#F4C63D",
    warning_tint="#3F3A23",
    warning_text="#F4C63D",
)


def palette(mode: Mode = "light") -> Palette:
    """Return the colour tokens for ``mode`` ("light" or "dark"); anything else falls back to light."""
    return DARK if mode == "dark" else LIGHT


@dataclass(frozen=True)
class ChannelStyle:
    """How one model channel is drawn everywhere: badge, colours, dash pattern and marker."""

    key: str
    badge: str
    name: str
    light: str
    dark: str
    dash: tuple[int, ...]
    marker: str

    def colour(self, mode: Mode = "light") -> str:
        """Return this channel's colour for the given theme mode."""
        return self.dark if mode == "dark" else self.light

    @property
    def label(self) -> str:
        """Badge and name together, for example ``"CH2 XGBoost"``."""
        return f"{self.badge} {self.name}"


# Channel order is fixed: it decides badge numbers, legend order and table order across the app.
CHANNELS: tuple[ChannelStyle, ...] = (
    ChannelStyle("forest", "CH1", "Random forest", "#2A6B5A", "#5FB39B", (), "circle"),
    ChannelStyle("xgboost", "CH2", "XGBoost", "#8C6620", "#C49A55", (8, 4), "square"),
    ChannelStyle("svm", "CH3", "RBF SVM", "#A0508A", "#A86A98", (2, 3), "diamond"),
    ChannelStyle("mlp", "CH4", "Neural net (MLP)", "#1B2229", "#E4E8EC", (6, 3, 2, 3), "cross"),
    ChannelStyle("logreg", "CH5", "Logistic regression", "#5A6470", "#98A3AE", (10, 5), "triangle-down"),
)
CHANNEL_BY_KEY: dict[str, ChannelStyle] = {c.key: c for c in CHANNELS}
CONSENSUS_LABEL = "Consensus"

# Ten-step ramps. Streamlit requires exactly ten colours for its sequential and diverging chart settings.
SEQUENTIAL: dict[str, tuple[str, ...]] = {
    "light": ("#F4F3EC", "#D2E0D5", "#AFCEBF", "#8DBBA9", "#6CA08E", "#46806E", "#2A6B5A", "#215648", "#184237", "#0F2F27"),
    "dark": ("#1B2328", "#213534", "#264740", "#2A5A4C", "#367160", "#4D9480", "#5FB39B", "#8BC7B3", "#B4DCCC", "#DCF0E6"),
}
# Diverging ramp for feature contributions: blue pushes towards normal, vermilion towards attack.
DIVERGING: dict[str, tuple[str, ...]] = {
    "light": ("#1F5B9C", "#5F84B9", "#91B1D6", "#B9CCE1", "#DDE2E6", "#EFE0D7", "#EFC7B6", "#EAA88F", "#CC6241", "#A5341A"),
    "dark": ("#62A2DE", "#4D80B0", "#3A6085", "#334A62", "#2E3944", "#3D3536", "#603F36", "#874D3A", "#BB684C", "#F2845E"),
}
# Attack-type ramp for multi-class traffic views (top four types; the rest are grouped as "Other attacks").
ATTACK_TYPES: dict[str, tuple[str, ...]] = {
    "light": ("#6B230D", "#A5341A", "#DA7956", "#EAA88F"),
    "dark": ("#F2845E", "#BB684C", "#F7B39A", "#874D3A"),
}

# Text glyphs that carry meaning without colour.
GLYPH_NORMAL = "○"
GLYPH_ATTACK = "◆"
GLYPH_ALERT = "▲"
GLYPH_DONE = "✓"

FONT_HEADING = "Instrument Sans"
FONT_BODY = "Atkinson Hyperlegible Next"
FONT_MONO = "Atkinson Hyperlegible Mono"
FONT_FILES: dict[str, str] = {
    FONT_HEADING: "InstrumentSans-Variable.ttf",
    FONT_BODY: "AtkinsonHyperlegibleNext-Variable.ttf",
    FONT_MONO: "AtkinsonHyperlegibleMono-Variable.ttf",
}


def verdict_text(label: str, benign_label: str = "BENIGN") -> str:
    """Return a verdict with its shape glyph, e.g. ``"○ Normal"`` or ``"◆ DoS Hulk"``."""
    if label in (benign_label, "Normal") or str(label).strip().upper() in ("BENIGN", "NORMAL"):
        return f"{GLYPH_NORMAL} Normal"
    return f"{GLYPH_ATTACK} {label}"


# Scores (accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision) are printed with four
# decimals. Rounding alone would print every value from 0.99995 up as a perfect 1.0000, so a score short of 1 is
# held at 0.9999 instead: 1.0000 on screen or in the record always means exactly 1.
SCORE_DECIMALS = 4
#: The largest score shown for a reading that is not perfect.
BELOW_PERFECT = 0.9999


def shown_score(value: float) -> float:
    """The value a score is printed as: ``value`` itself, except that a score in [0.9999, 1) becomes 0.9999.

    Printed with four decimals, the result never reads 1.0000 unless ``value`` is exactly 1 (or more). NaN and
    infinities pass through unchanged.
    """
    number = float(value)
    return BELOW_PERFECT if BELOW_PERFECT <= number < 1.0 else number


def shown_scores(values: Any) -> np.ndarray:
    """:func:`shown_score` for every value of an array (float64 copy; NaN stays NaN)."""
    array = np.array(values, dtype=np.float64, copy=True)
    with np.errstate(invalid="ignore"):
        array[(array >= BELOW_PERFECT) & (array < 1.0)] = BELOW_PERFECT
    return array


def score_text(value: Any, missing: str = "n/a") -> str:
    """A score with four decimals (see :func:`shown_score`), or ``missing`` for None, NaN or a non-number."""
    if value is None or isinstance(value, bool):
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(number):
        return missing
    return f"{shown_score(number):.{SCORE_DECIMALS}f}"
