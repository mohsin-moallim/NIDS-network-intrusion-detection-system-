"""Logbook station: saved channel sets (save, load with verification, delete) and the history of every fit.

Saving writes the session's current run as a bundle (:mod:`graticule.persist`). CH3 is left out unless the viewer
ticks "Also save CH3 (RBF SVM)", a box drawn unticked for every new run: a kernel SVM is made of training rows, so
saving it writes them (its support vectors) into the run's folder under saved_models. Loading checks the bundle's
files, re-scores its probe vectors and, where the data allow, rebuilds the held-out rows, with a progress bar
through every stage; the loaded run then becomes the current run for every station, marked "loaded from disk".
Loading over a fit that is not saved, deleting a saved set and clearing the history all ask for a confirmation
first. Hidden folders that a save or a delete left behind (a file held by another program) are named, with a
note when one still holds CH3's training rows, and can be removed. Nothing here fits a channel.
"""

from __future__ import annotations

import html
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from graticule import persist, theme
from graticule.history import RunHistory
from graticule.models.train import TrainingRun
from ui import components, state

# One message about the last Logbook action (kind, text), shown once.
NOTICE = "lb_notice"
# Run id of the saved set waiting for a delete confirmation; True while a history clear waits for one.
CONFIRM_DELETE = "lb_confirm_delete"
CONFIRM_CLEAR = "lb_confirm_clear"
# Run id of the saved set waiting to be loaded over an unsaved fit.
CONFIRM_LOAD = "lb_confirm_load"
PICK = "lb_pick"
#: Key prefix of the "Also save CH3" box; the run id completes it, so every new run starts unticked.
SAVE_SVM = "lb_save_svm"
#: History lines shown in the table (the CSV holds every line).
HISTORY_SHOWN = 200
#: Seconds a hidden work folder in saved_models must stand unchanged before it is shown as left over (a save or a
#: delete still running in another session keeps changing its folder).
LEFTOVER_MIN_AGE = 30.0

BUNDLE_COLUMNS: tuple[str, ...] = ("Run", "Fitted (UTC)", "Mode", "Source", "Rows", "Channels",
                                   "Training rows inside", "Best channel", "Best balanced accuracy", "Check",
                                   "Folder")
#: History columns shown in the table (the CSV holds every column). The source is left out because the files say it
#: ("generated" for synthetic flows); whether the run was saved is a yes or no (its folder is named after the run).
HISTORY_COLUMNS: dict[str, str] = {
    "run_id": "Run", "created_utc": "Fitted (UTC)", "files": "Files", "mode": "Mode",
    "feature_mode": "Features", "rows_train": "Train rows", "rows_test": "Test rows", "channels": "Channels",
    "best_channel": "Best channel", "best_balanced_accuracy": "Best balanced accuracy", "seconds": "Fit s",
    "saved_path": "Saved",
}


def _notice(kind: str, text: str) -> None:
    """Leave one message for the next draw of this station."""
    st.session_state[NOTICE] = (kind, text)


def _show_notice() -> None:
    """Show (once) the message left by the last Logbook action."""
    notice = st.session_state.pop(NOTICE, None)
    if notice is None:
        return
    kind, text = notice
    {"success": st.success, "info": st.info, "warning": st.warning}.get(kind, st.error)(text)


def _badges(keys: tuple[str, ...] | list[str]) -> str:
    """``CH1 CH2 CH5`` for channel keys."""
    return " ".join(theme.CHANNEL_BY_KEY[k].badge if k in theme.CHANNEL_BY_KEY else k for k in keys)


def _channel_name(key: str | None) -> str:
    """``CH2 XGBoost`` for a channel key ("" for None)."""
    if not key:
        return ""
    style = theme.CHANNEL_BY_KEY.get(str(key))
    return style.label if style is not None else str(key)


def _mode_text(mode: str) -> str:
    """Readable mode name."""
    return {"binary": "binary", "multiclass": "multi-class"}.get(mode, mode)


def when_text(created_utc: object) -> str:
    """A recorded UTC time for a table, to the minute: ``"2026-10-03T16:15:36+00:00"`` -> ``"2026-10-03 16:15"``
    (text that is not such a time is shown as it is)."""
    text = str(created_utc or "")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return text
    return moment.strftime("%Y-%m-%d %H:%M")


def files_text(files: object) -> str:
    """The data files of a run, short enough for a table: the day and part of each CIC-IDS2017 file
    (``"Wednesday"``, ``"Thursday-Morning-WebAttacks"``), or how many files when there are more than two."""
    names = [name.strip() for name in str(files or "").split(",") if name.strip()]
    if len(names) > 2:
        return f"{len(names)} files"
    short = []
    for name in names:
        stem = name.removesuffix(".csv").removesuffix(".pcap_ISCX")
        short.append("-".join(part for part in stem.split("-") if part.lower() != "workinghours") or stem)
    return ", ".join(short)


def save_svm_key(run: TrainingRun) -> str:
    """Widget key of the "Also save CH3" box for ``run`` (one per run, so it is unticked for every new run)."""
    return f"{SAVE_SVM}-{run.run_id}"


def _held_rows_text(count: int) -> str:
    """Chip text for a set that holds CH3's training rows."""
    return f"holds {count:,} training rows (CH3)"


# --------------------------------------------------------------------------------------------------------------
# The run on the bench
# --------------------------------------------------------------------------------------------------------------
def _loaded_panel(run: TrainingRun) -> None:
    """What is known about a run loaded from disk: where from, verification, held-out rows."""
    info = state.loaded_info(run.run_id)
    folder = state.bundle_on_disk(run)
    chips = [f"run {run.run_id}", "loaded from disk", _mode_text(run.request.mode),
             f"{len(run.ok_channels())} channels: {_badges(run.ok_channels())}"]
    if info is not None:
        chips.append("verified" if info.verified else "not verified")
    chips.append("held-out rows rebuilt" if run.has_test_rows else "no held-out rows")
    held = state.saved_training_rows(run)  # rows in the set on disk now: 0 once its folder is deleted
    if held:
        chips.append(_held_rows_text(held))
    components.chips(chips)
    where = run.bundle_path or "an unknown folder"
    st.markdown(f'Loaded from <span class="g-mono">{html.escape(where)}</span>'
                + ("" if folder is not None else " (that folder has since been deleted)") + ".",
                unsafe_allow_html=True)
    if info is not None:
        tone = {"success": st.success, "warning": st.warning}.get(info.kind, st.error)
        tone(info.verification)
        st.caption(info.rows_note)
    if held:
        st.caption(f"This set {_held_rows_text(held)}: CH3 was saved with it by choice, and its model is its "
                   f"{held:,} support vectors, training rows stored after gap filling, signed logarithm and "
                   "standardisation, which its saved scaler turns back into the original values. They stay in this "
                   "machine's saved_models folder, which git ignores.")
    elif folder is None and (vectors := state.ch3_support_vectors(run)):
        st.caption(f"The set this run was loaded from held CH3's {vectors:,} training rows (its support vectors). "
                   "That folder has since been deleted, and with it the rows (a delete that stopped halfway would "
                   "be listed under Saved channel sets below); CH3 stays on the bench, in memory, like the other "
                   "channels.")
    if not run.has_test_rows:
        st.caption("Without the held-out rows, the stations can score single flows and uploaded files with these "
                   "channels, but cannot show readings on the test rows or replay them.")
        if folder is not None and st.button("Try again to rebuild the held-out rows", key="lb_rebuild"):
            _load(folder, run.run_id)


def _current_section(run: TrainingRun | None) -> None:
    """The run on the bench and, for a fresh fit, the Save button."""
    st.subheader("On the bench", anchor=False)
    if run is None:
        components.needs("No fitted channels in this session yet. Fit them at 02 Fit, or load a saved set below.",
                         "fit")
        return
    if getattr(run, "origin", "fitted") == "loaded":
        _loaded_panel(run)
        return
    components.chips([f"run {run.run_id}", "fitted in this session", _mode_text(run.request.mode),
                      f"{len(run.ok_channels())} channels: {_badges(run.ok_channels())}",
                      f"fitted {run.created_utc}"])
    folder = state.bundle_on_disk(run)
    if folder is not None:
        st.markdown(f'Saved as <span class="g-mono">{html.escape(str(folder))}</span>.', unsafe_allow_html=True)
        held = state.saved_training_rows(run)
        if held:
            st.caption(f"This set {_held_rows_text(held)}: CH3 was saved with it by choice (its support vectors, in "
                       "svm.joblib).")
        left_out = state.unsaved_channel_note(run)
        if left_out:
            st.caption(left_out)
        return
    what = ("Saving writes the fitted channels, their settings and readings, a set of synthetic probe flows with the "
            "channels' readings of them, and per-feature quantiles.")
    include_svm = False
    vectors = state.ch3_support_vectors(run)
    if vectors is None:
        st.caption(f"{what} No dataset rows are written: CH3, the one channel whose model is made of training rows "
                   "(and which is saved only by choice), has no fitted model in this run.")
    else:
        st.caption(f"{what} No dataset rows are written unless you save CH3: the kernel SVM's model is made of "
                   "training rows, so it is not saved unless you choose to.")
        include_svm = st.checkbox("Also save CH3 (RBF SVM)", value=False, key=save_svm_key(run))
        seconds = float(run.channels["svm"].predict_seconds or 0.0)
        cost = (f" (it took {seconds:,.1f} s at this fit)" if seconds >= 0.1 else "")
        alone = (" CH3 is the only fitted channel, so without the box there is nothing to save."
                 if all(key in persist.UNSAVED_CHANNELS for key in run.ok_channels()) else "")
        st.caption(f"CH3's model is its {vectors:,} support vectors, which are training rows after gap filling, "
                   "signed logarithm and standardisation; its saved scaler turns them back into the original "
                   f"values. Ticking writes them into saved_models\\{run.run_id}\\ on this machine (git ignores the "
                   "folder). Leave it unticked to keep dataset rows out of the project, as by default; CH3's "
                   "readings are recorded with the set either way. Loading a set that holds CH3 has it read the "
                   f"held-out rows again, which takes about as long as its scoring did{cost}.{alone}")
    if st.button("Save the current fit", key="lb_save", type="primary"):
        with st.spinner("Saving the channel set"):
            kind, text = state.save_current_run(include_svm=bool(include_svm))
        _notice(kind, text)
        st.rerun()


# --------------------------------------------------------------------------------------------------------------
# Saved channel sets
# --------------------------------------------------------------------------------------------------------------
def bundle_table(bundles: list[persist.BundleSummary]) -> pd.DataFrame:
    """The saved sets as a display table, newest first (``Check`` says why Load would refuse a set, if it would).

    Compact enough for a 1440-pixel page: the time to the minute and the folder by its name (every set sits in the
    saved-sets folder, named under the table).
    """
    rows = [{
        "Run": b.run_id, "Fitted (UTC)": when_text(b.created_utc), "Mode": _mode_text(b.mode), "Source": b.source,
        "Rows": int(b.rows), "Channels": _badges(b.channels),
        "Training rows inside": int(getattr(b, "holds_training_rows", 0) or 0),
        "Best channel": _channel_name(b.best_channel),
        "Best balanced accuracy": b.best_balanced_accuracy,
        "Check": "manifest intact" if getattr(b, "problem", None) is None else f"refused: {b.problem}",
        "Folder": Path(b.path).name,
    } for b in bundles]
    return pd.DataFrame(rows, columns=list(BUNDLE_COLUMNS))


def _bundle_label(summary: persist.BundleSummary) -> str:
    """Option text for one saved set."""
    best = (f", best {theme.score_text(summary.best_balanced_accuracy)}"
            if summary.best_balanced_accuracy is not None else "")
    refused = ", will be refused" if getattr(summary, "problem", None) else ""
    held = int(getattr(summary, "holds_training_rows", 0) or 0)
    rows = f", {_held_rows_text(held)}" if held else ""
    return f"{summary.run_id} ({_mode_text(summary.mode)}, {summary.source}{best}{rows}{refused})"


def _load(path: Path | str, run_id: str, prefix: str = "") -> None:
    """Load a saved set with a progress bar through every stage, leave the outcome as the notice (after
    ``prefix``, when given), and rerun.

    Reading and cleaning the data files can take as long as drawing the sample at 01 Sample, so each stage (the
    file checks, every data file, the matrices, each channel reading the held-out rows) moves the bar.
    """
    started = time.perf_counter()
    with st.status(f"Loading {run_id}", expanded=True) as status:
        bar = st.progress(0.0, text="Starting")

        def report(message: str, fraction: float) -> None:
            bar.progress(min(max(float(fraction), 0.0), 1.0),
                         text=f"{message} · {time.perf_counter() - started:.1f} s")

        result = state.load_bundle_into_session(path, progress=report)
        status.update(label=f"Loaded {run_id} in {time.perf_counter() - started:.1f} s"
                      if result.run_id else f"{run_id} was not loaded",
                      state="complete" if result.run_id else "error", expanded=False)
    _notice(result.kind, f"{prefix} {result.message}".strip())
    st.rerun()


def _needs_load_confirmation(current: TrainingRun | None, summary: persist.BundleSummary) -> bool:
    """True when loading ``summary`` would replace a fit of this session that was never saved."""
    return (current is not None and getattr(current, "origin", "fitted") == "fitted"
            and state.bundle_on_disk(current) is None and current.run_id != summary.run_id)


def _load_confirmation(summary: persist.BundleSummary, current: TrainingRun) -> None:
    """The confirmation step of a load that would replace an unsaved fit."""
    with st.container(border=True, key="lb_load_box"):
        st.warning(f"Run {current.run_id} on the bench was fitted in this session and is not saved. Loading "
                   f"{summary.run_id} replaces it, and this session cannot go back to it afterwards.")
        # The "Also save CH3" box above applies to this save too (unticked unless the viewer ticked it).
        include_svm = bool(st.session_state.get(save_svm_key(current), False))
        if state.ch3_support_vectors(current) is not None:
            if include_svm:
                st.caption("Saving first includes CH3 (RBF SVM) and its training rows, as ticked above.")
            elif all(key in persist.UNSAVED_CHANNELS for key in current.ok_channels()):
                st.caption("Saving first would save nothing: CH3 (RBF SVM) is the only fitted channel, and it is "
                           "left out unless \"Also save CH3 (RBF SVM)\" above is ticked.")
            else:
                st.caption("Saving first leaves CH3 (RBF SVM) out, as by default; tick \"Also save CH3 (RBF SVM)\" "
                           "above to include it.")
        save_first, anyway, keep = st.columns(3)
        if save_first.button(f"Save {current.run_id}, then load", key="lb_load_save", type="primary",
                             width="stretch"):
            st.session_state.pop(CONFIRM_LOAD, None)
            kind, text = state.save_current_run(include_svm=include_svm)
            if kind != "success":
                _notice(kind, f"{text} Nothing was loaded.")
                st.rerun()
            _load(summary.path, summary.run_id, prefix=text)
        if anyway.button("Load without saving", key="lb_load_anyway", width="stretch"):
            st.session_state.pop(CONFIRM_LOAD, None)
            _load(summary.path, summary.run_id)
        if keep.button("Keep the current run", key="lb_load_no", width="stretch"):
            st.session_state.pop(CONFIRM_LOAD, None)
            st.rerun()


def _delete_confirmation(summary: persist.BundleSummary, current: TrainingRun | None) -> None:
    """The confirmation step of a delete."""
    with st.container(border=True, key="lb_delete_box"):
        st.warning(f"Delete the saved channel set {summary.run_id} for good? Its folder {summary.path} is removed "
                   "from disk; this cannot be undone.")
        if current is not None and current.run_id == summary.run_id:
            st.caption("The run stays on the bench until the session ends; it just can no longer be loaded again.")
        yes, no = st.columns(2)
        if yes.button("Delete for good", key="lb_delete_yes", type="primary"):
            st.session_state.pop(CONFIRM_DELETE, None)
            try:
                persist.delete_bundle(summary.path)
            except persist.BundleDeleteError as exc:  # stopped halfway: the message says where the rest is
                _notice("error", str(exc))
            except (OSError, persist.BundleIntegrityError) as exc:
                _notice("error", f"The saved set {summary.run_id} could not be deleted: {exc}")
            else:
                _notice("success", f"Deleted the saved channel set {summary.run_id}.")
            st.rerun()
        if no.button("Keep it", key="lb_delete_no"):
            st.session_state.pop(CONFIRM_DELETE, None)
            st.rerun()


def _unreadable_note(unreadable: list[tuple[Path, str]]) -> None:
    """Name the folders whose manifest could not be read (they are not listed)."""
    if unreadable:
        names = "; ".join(f"{folder.name} ({reason})" for folder, reason in unreadable[:5])
        more = f"; {len(unreadable) - 5} more" if len(unreadable) > 5 else ""
        st.caption(f"Not listed, because their manifest cannot be read: {names}{more}.")


def _leftover_text(item: persist.LeftoverFolder) -> str:
    """``.deleting-<id>-<time> (unfinished delete, 2 files, svm.joblib: CH3's training rows)``."""
    files = f"{len(item.files)} file{'s' if len(item.files) != 1 else ''}"
    rows = ", svm.joblib: CH3's training rows" if item.holds_training_rows else ""
    return f"{item.path.name} (unfinished {item.kind}, {files}{rows})"


def _leftovers_section() -> None:
    """Name the hidden work folders a save or a delete left in saved_models, and offer to remove them.

    Only folders unchanged for :data:`LEFTOVER_MIN_AGE` seconds are shown, so a save or a delete still running in
    another session is never offered for removal.
    """
    try:
        leftovers = persist.find_leftovers(min_age_seconds=LEFTOVER_MIN_AGE)
    except Exception:  # noqa: BLE001 - an unreadable models folder is reported by the listing above
        return
    if not leftovers:
        return
    n = len(leftovers)
    names = "; ".join(_leftover_text(item) for item in leftovers[:5])
    more = f"; {n - 5} more" if n > 5 else ""
    rows = sum(1 for item in leftovers if item.holds_training_rows)
    held = (f" {rows} of them still hold{'s' if rows == 1 else ''} CH3's training rows (svm.joblib, its support "
            "vectors)." if rows else "")
    st.warning(f"A save or a delete that did not finish left {n} hidden folder{'s' if n != 1 else ''} in "
               f"{leftovers[0].path.parent}: {names}{more}.{held} They are not saved sets and cannot be loaded; "
               "removing them deletes them for good.")
    if st.button("Remove the leftover folders", key="lb_leftovers"):
        failed: list[str] = []
        for item in leftovers:
            try:
                persist.remove_leftover(item.path)
            except (OSError, ValueError) as exc:
                failed.append(f"{item.path.name} ({state.first_line(str(exc)) or type(exc).__name__})")
        if failed:
            _notice("error", f"Not every leftover folder could be removed (another program may still hold a file "
                             f"in it; try again in a moment): {'; '.join(failed)}.")
        else:
            _notice("success", f"Removed {n} leftover folder{'s' if n != 1 else ''}.")
        st.rerun()


def _bundles_section(current: TrainingRun | None) -> None:
    """Table of saved sets with Load and Delete."""
    st.subheader("Saved channel sets", anchor=False)
    try:
        bundles, unreadable = persist.scan_bundles()
    except Exception as exc:  # noqa: BLE001 - an unreadable models folder must not take the station down
        st.error(f"The saved channel sets cannot be listed: {state.first_line(str(exc)) or type(exc).__name__}")
        return
    if not bundles:
        st.caption("No saved channel sets yet. Save a fit above; sets are kept in the saved_models folder of the "
                   "project.")
        _unreadable_note(unreadable)
        _leftovers_section()
        return
    table = bundle_table(bundles)
    if (table["Folder"] == table["Run"]).all():  # every set sits in a folder named after its run: say it once
        table = table.drop(columns=["Folder"])
    st.dataframe(
        components.shown_scores(table, ("Best balanced accuracy",)), hide_index=True,
        width="stretch",
        column_config={
            "Rows": st.column_config.NumberColumn("Rows", format="localized", help="Training plus test rows."),
            "Training rows inside": st.column_config.NumberColumn(
                "Rows inside", format="localized",
                help="Training rows inside: dataset rows written into the set, CH3's support vectors when it was "
                     "saved by choice, else 0."),
            "Best balanced accuracy": st.column_config.NumberColumn(
                "Best bal. acc.", format="%.4f", help="Best balanced accuracy of the set's channels."),
            "Check": st.column_config.TextColumn("Check", help="Whether the manifest still matches the checksum "
                                                 "recorded in it, read without loading any model."),
            "Folder": st.column_config.TextColumn("Folder", help="The set's folder inside the saved-sets folder."),
        },
    )
    st.caption(f"Saved sets are kept in {Path(bundles[0].path).parent}, each in a folder named after its run.")
    _unreadable_note(unreadable)
    _leftovers_section()
    by_id = {b.run_id: b for b in bundles}
    if st.session_state.get(PICK) not in by_id:
        st.session_state[PICK] = bundles[0].run_id
    picked_id = st.selectbox("Saved set", options=list(by_id), key=PICK,
                             format_func=lambda run_id: _bundle_label(by_id[run_id]))
    summary = by_id[picked_id]
    load_col, delete_col, _ = st.columns([1, 1, 2])
    if load_col.button("Load", key="lb_load", type="primary", width="stretch"):
        if _needs_load_confirmation(current, summary):
            st.session_state[CONFIRM_LOAD] = summary.run_id
        else:
            st.session_state.pop(CONFIRM_LOAD, None)
            _load(summary.path, summary.run_id)
    if delete_col.button("Delete", key="lb_delete", width="stretch"):
        st.session_state[CONFIRM_DELETE] = summary.run_id
    waiting = st.session_state.get(CONFIRM_LOAD)
    if waiting is not None:
        if waiting in by_id and current is not None and _needs_load_confirmation(current, by_id[waiting]):
            _load_confirmation(by_id[waiting], current)
        else:
            st.session_state.pop(CONFIRM_LOAD, None)
    pending = st.session_state.get(CONFIRM_DELETE)
    if pending is not None:
        if pending in by_id:
            _delete_confirmation(by_id[pending], current)
        else:
            st.session_state.pop(CONFIRM_DELETE, None)
    st.caption("Loading checks the manifest and every file against the checksums recorded when the set was saved "
               "(a changed file is refused), then has each channel read the saved probe flows again: the set is "
               "verified when every reading is reproduced exactly with the same library versions. The held-out rows "
               "are then rebuilt from the data and checked row by row, label by label and value by value.")


# --------------------------------------------------------------------------------------------------------------
# Run history
# --------------------------------------------------------------------------------------------------------------
def history_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The history as a display table: readable column names, channel badges, the time to the minute, short file
    names and whether the run was saved (the CSV keeps every column, the saved folder's full path included)."""
    shown = frame[list(HISTORY_COLUMNS)].copy()
    shown["channels"] = [_badges([k.strip() for k in str(v).split(",") if k.strip()]) for v in shown["channels"]]
    shown["best_channel"] = [_channel_name(v) for v in shown["best_channel"]]
    shown["mode"] = [_mode_text(str(v)) for v in shown["mode"]]
    shown["created_utc"] = [when_text(v) for v in shown["created_utc"]]
    shown["files"] = [files_text(v) for v in shown["files"]]
    shown["saved_path"] = ["yes" if isinstance(v, str) and v else "no" for v in shown["saved_path"]]
    return shown.rename(columns=HISTORY_COLUMNS)


def history_csv(frame: pd.DataFrame) -> bytes:
    """The full history (every column) as CSV bytes with a byte-order mark, so spreadsheet programs read UTF-8."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def _history_section() -> None:
    """The run history table, its CSV download and Clear history."""
    st.subheader("Run history", anchor=False)
    problem = state.history_error()
    if problem:
        st.warning(problem)
    history = RunHistory()
    try:
        total = history.count()
        frame = history.list(limit=HISTORY_SHOWN)
    except Exception as exc:  # noqa: BLE001 - an unreadable history file must not take the station down
        st.error(f"The run history at {history.path} cannot be read: {exc}")
        return
    if frame.empty:
        st.caption("No fits recorded yet. Every fit finished at 02 Fit adds a line here.")
        return
    st.dataframe(
        components.shown_scores(history_table(frame), ("Best balanced accuracy",)), hide_index=True,
        width="stretch",
        column_config={
            "Files": st.column_config.TextColumn("Files", help="The data files, by day and part (the CSV holds "
                                                 "their full names)."),
            "Train rows": st.column_config.NumberColumn("Train rows", format="localized"),
            "Test rows": st.column_config.NumberColumn("Test rows", format="localized"),
            "Best balanced accuracy": st.column_config.NumberColumn(
                "Best bal. acc.", format="%.4f", help="Best balanced accuracy of the run's channels."),
            "Fit s": st.column_config.NumberColumn("Fit s", format="%.1f",
                                                   help="Seconds for the whole fit (matrices and channels)."),
            "Saved": st.column_config.TextColumn("Saved", help="Whether the run was saved as a channel set (its "
                                                 "folder is named after the run; the CSV gives the full path)."),
        },
    )
    shown = (f" The table shows the newest {len(frame):,}; the CSV holds all {total:,}."
             if total > len(frame) else "")
    st.caption(f"{total:,} run{'s' if total != 1 else ''} recorded in {history.path}.{shown} The CSV holds every "
               "column, including each channel's metrics and the settings as JSON.")
    try:
        everything = frame if total <= len(frame) else history.list(limit=None)
    except Exception:  # noqa: BLE001 - fall back to the lines already read
        everything = frame
    left, right, _ = st.columns([1, 1, 2])
    left.download_button("Download the history (CSV)", data=history_csv(everything),
                         file_name="graticule_run_history.csv", mime="text/csv", key="lb_history_csv",
                         width="stretch")
    if right.button("Clear history", key="lb_clear", width="stretch"):
        st.session_state[CONFIRM_CLEAR] = True
    if st.session_state.get(CONFIRM_CLEAR):
        with st.container(border=True, key="lb_clear_box"):
            st.warning("Clear every line of the run history? Saved channel sets are not affected.")
            yes, no = st.columns(2)
            if yes.button("Clear for good", key="lb_clear_yes", type="primary"):
                st.session_state.pop(CONFIRM_CLEAR, None)
                history.clear()
                _notice("success", "The run history was cleared.")
                st.rerun()
            if no.button("Keep it", key="lb_clear_no"):
                st.session_state.pop(CONFIRM_CLEAR, None)
                st.rerun()


def render() -> None:
    """Draw the Logbook station."""
    components.station_header("logbook")
    _show_notice()
    run = state.current_run()
    _current_section(run)
    st.divider()
    _bundles_section(run)
    st.divider()
    _history_section()

