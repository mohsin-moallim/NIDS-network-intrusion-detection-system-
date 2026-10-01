"""Bench: the instrument settings (data folder and the defaults every station starts from)."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import graticule.settings as settings_mod
from graticule.models.verdict import ALERT_RULE
from graticule.schema import EXPECTED_FILES
from graticule.settings import ENV_DATA_DIR, NONFINITE_STRATEGIES, SVM_CAP_RANGE, AppSettings, resolve_data_dir
from ui import components, state

#: Session key of a message about settings that could not be written to disk (shown once, after the rerun).
SAVE_PROBLEM = "g_bench_save_problem"

_STRATEGY_HELP = {
    "drop": "Drop rows that contain an infinite or missing value (the counts are reported per class).",
    "impute": "Keep those rows; infinities become missing and are filled with the training median inside each channel.",
    "recompute": "Rebuild Flow Bytes/s and Flow Packets/s from the byte and packet totals, with a 1 µs floor on duration.",
}


def _data_folder_status(current: AppSettings) -> None:
    """Show which folder is in use, where the choice came from, and which of the eight files were found."""
    res = resolve_data_dir(current)
    env_value = os.environ.get(ENV_DATA_DIR, "").strip()
    if res.source == "none":
        st.info("No data folder is set, so every station runs on synthetic flows. "
                f"Set one below, or start the app with the {ENV_DATA_DIR} environment variable.")
    elif res.problem:
        st.warning(res.problem)
    else:
        where = "this setting" if res.source == "setting" else f"the {ENV_DATA_DIR} environment variable"
        st.success(f"Reading CSV files from `{res.path}` (chosen by {where}).")
    if env_value and res.source == "setting":
        st.caption(f"{ENV_DATA_DIR} is also set (`{env_value}`), but the folder above takes precedence.")
    if res.path is not None:
        found = {p.name.lower() for p in res.found}
        rows = [
            {"File": f.name, "Session": f.day, "Contents": f.contents,
             "Status": "found" if f.name.lower() in found else "missing"}
            for f in EXPECTED_FILES
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if res.other_csvs:
            st.caption("Other CSV files in the folder: " + ", ".join(p.name for p in res.other_csvs))


def _save_problem_text(exc: OSError) -> str:
    """What to tell the viewer when the settings file could not be written."""
    name = settings_mod.SETTINGS_FILE.name
    if isinstance(exc, PermissionError):
        why = "another program may be holding it (a spreadsheet or a sync tool, say), or access is denied"
    else:
        why = state.first_line(str(exc)) or type(exc).__name__
    return (f"The new settings apply in this session, but they could not be saved to {name}: {why}. Close the "
            "file elsewhere and press Save settings again to keep them for the next start.")


def render() -> None:
    """Draw the Bench settings page."""
    components.station_header("bench")
    problem = st.session_state.pop(SAVE_PROBLEM, None)
    if problem:
        st.error(problem)
    current = state.settings()
    st.subheader("Data folder", anchor=False)
    _data_folder_status(current)

    with st.form("bench_form", border=True):
        data_dir = st.text_input(
            "Data folder",
            value=current.data_dir or "",
            placeholder=r"C:\datasets\CIC-IDS2017\MachineLearningCSV",
            help="Folder holding the CIC-IDS2017 MachineLearningCSV files. Leave empty to use "
            f"{ENV_DATA_DIR} if it is set, or synthetic flows otherwise.",
        )
        st.subheader("Cleaning and sampling", anchor=False)
        left, right = st.columns(2)
        with left:
            strategy = st.radio(
                "Infinite and missing values",
                NONFINITE_STRATEGIES,
                index=NONFINITE_STRATEGIES.index(current.nonfinite_strategy),
                format_func=lambda s: {"drop": "Drop the row", "impute": "Impute (median)", "recompute": "Recompute rates"}[s],
                help=" ".join(f"{k}: {v}" for k, v in _STRATEGY_HELP.items()),
            )
            row_budget = st.number_input("Row budget (sample size)", 1_000, 5_000_000, current.row_budget, step=10_000,
                                         help="Upper limit on rows kept after cleaning. Rare classes are protected.")
            min_count = st.number_input("Minimum rows per class (multi-class mode)", 10, 100_000,
                                        current.min_class_count, step=10,
                                        help="Classes with fewer rows are left out of multi-class fits and reported.")
            merge_web = st.checkbox("Merge the three Web Attack classes", value=current.merge_web_attacks)
        with right:
            svm_cap = st.number_input("SVM training cap (rows)", SVM_CAP_RANGE[0], SVM_CAP_RANGE[1], current.svm_cap,
                                      step=1_000, help="Kernel SVM fit time grows at least quadratically with rows, "
                                      "so CH3 trains on at most this many rows.")
            test_share = st.slider("Test share", 0.10, 0.50, current.test_share, 0.05,
                                   help="Fraction of the sample held out for measuring the channels.")
            alert = st.slider("Alert confidence threshold", 0.50, 0.999, current.alert_threshold, 0.005,
                              help=f"An alert is {ALERT_RULE}. 04 Probe uses this value; 05 Assay and 06 Sweep "
                              "start from it and can change it, and the PDF record reports the values they used.")
            seed = st.number_input("Random seed", 0, 2**31 - 2, current.seed, step=1,
                                   help="Makes sampling, splitting and fitting repeatable.")
        saved = st.form_submit_button("Save settings", type="primary")
    if saved:
        try:
            state.update_settings(AppSettings(
                data_dir=data_dir, nonfinite_strategy=strategy, row_budget=int(row_budget), svm_cap=int(svm_cap),
                seed=int(seed), alert_threshold=float(alert), min_class_count=int(min_count),
                test_share=float(test_share), merge_web_attacks=bool(merge_web),
            ))
        except OSError as exc:
            st.session_state[SAVE_PROBLEM] = _save_problem_text(exc)
        else:
            st.toast("Settings saved.")
        st.rerun()

    st.subheader("Appearance", anchor=False)
    st.caption("Choose the light (paper) or dark (graphite) theme, or let the system decide, under Theme in the ⋮ "
               "menu at the top right. The PDF record always uses the light theme.")
