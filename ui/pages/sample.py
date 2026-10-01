"""01 Sample station: choose a source, draw a cleaned rare-aware sample and read its sample sheet.

Preparation runs only when the "Draw sample" button of the form is pressed; every other interaction just redraws
the stored result, so nothing is re-read or re-sampled by accident. The forms start from the options of the last
draw from the same source (or from the Bench defaults before the first one), so leaving the page or switching the
source and back does not silently reset them.
"""

from __future__ import annotations

import time

import streamlit as st

from graticule import theme, viz
from graticule.data import prepare
from graticule.data.clean import CONFLICT_POLICIES
from graticule.data.prepare import DataRequest, PreparedDataset
from graticule.data.reader import DataFileError
from graticule.schema import EXPECTED_BY_NAME, is_benign
from graticule.settings import NONFINITE_STRATEGIES, AppSettings, DataDirResolution, resolve_data_dir
from ui import components, data_cache, state

SOURCE_LABELS = {"cicids": "CIC-IDS2017 files", "synthetic": "Synthetic flows"}
STRATEGY_LABELS = {
    "drop": "Drop the row",
    "impute": "Keep the row, impute per channel",
    "recompute": "Rebuild the two rate columns",
}
POLICY_LABELS = {
    "keep": "Keep them all (default)",
    "majority": "Keep the majority label",
    "drop": "Drop every conflicting row",
}
SYNTHETIC_FLOWS = (2_000, 500_000, 40_000)
SYNTHETIC_SHARE = (0.05, 0.90, 0.35)
NOTICE_KEY = "smp_notice"


def _file_caption(name: str, sizes: dict[str, int]) -> str:
    """Multiselect text for one file: session and contents for the known files, the name otherwise."""
    known = EXPECTED_BY_NAME.get(name.lower())
    size = sizes.get(name, 0) / 2**20
    if known is None:
        return f"{name} · {size:,.0f} MB"
    return f"{known.day} · {known.contents} · {size:,.0f} MB"


def _last_request(source: str, data_dir: str | None = None) -> DataRequest | None:
    """The request behind the stored sample when it came from ``source`` (and, for files, the same folder)."""
    dataset = state.get_prepared()
    if dataset is None or dataset.request.source != source:
        return None
    if source == "cicids" and dataset.request.data_dir != data_dir:
        return None
    return dataset.request


def _shared_options(current: AppSettings, prefix: str, last: DataRequest | None) -> tuple[int, str, str]:
    """Row budget, bad-value strategy and conflict policy widgets (inside the current form)."""
    budget_default = int(last.row_budget) if last else int(current.row_budget)
    strategy_default = last.nonfinite_strategy if last else current.nonfinite_strategy
    policy_default = last.conflict_policy if last else CONFLICT_POLICIES[0]
    left, right = st.columns(2)
    with left:
        budget = st.number_input(
            "Row budget (rows in the sample)", min_value=1_000, max_value=5_000_000,
            value=min(max(budget_default, 1_000), 5_000_000), step=10_000, key=f"{prefix}_budget",
            help="Upper limit on sampled rows. Every class first gets up to max(1,000, 2% of the budget) rows; "
            "the rest is shared in proportion to class size.",
        )
        strategy = st.radio(
            "Infinite and missing values", NONFINITE_STRATEGIES,
            index=NONFINITE_STRATEGIES.index(strategy_default),
            format_func=lambda s: STRATEGY_LABELS[s], key=f"{prefix}_strategy",
            help="Drop removes the row. Impute keeps it; each channel fills the gap with its training median. "
            "Rebuild recomputes Flow Bytes/s and Flow Packets/s from the totals and the duration (1 µs floor).",
        )
    with right:
        policy = st.selectbox(
            "Identical flows with different labels", CONFLICT_POLICIES, index=CONFLICT_POLICIES.index(policy_default),
            format_func=lambda s: POLICY_LABELS[s], key=f"{prefix}_policy",
            help="Keeping them is the honest default: removing hard cases would flatter every score.",
        )
    return int(budget), str(strategy), str(policy)


def _origin_caption(current: AppSettings, last: DataRequest | None, extra: str) -> None:
    """Caption under a form: the seed, where the options came from, and one source-specific remark."""
    origin = ("Options repeat your last draw; the Bench holds the defaults." if last is not None
              else "Options start from the Bench defaults.")
    st.caption(f"Seed {current.seed}, set on the Bench. {origin} {extra}")


def _files_form(current: AppSettings, resolution: DataDirResolution) -> DataRequest | None:
    """Form for sampling the CIC-IDS2017 files; returns a request when "Draw sample" is pressed."""
    paths = sorted(list(resolution.found) + list(resolution.other_csvs), key=lambda p: prepare.file_order_key(p.name))
    sizes = {p.name: p.stat().st_size for p in paths if p.exists()}
    names = [p.name for p in paths]
    found = {p.name for p in resolution.found}
    last = _last_request("cicids", str(resolution.path))
    default = [n for n in names if n in found] or names
    if last is not None:
        default = [n for n in names if n in last.files] or default
    with st.form("smp_files_form", border=True):
        chosen = st.multiselect(
            "Files", names, default=default, format_func=lambda n: _file_caption(n, sizes), key="smp_files",
            help="Files are read one at a time and cleaned on their own before duplicates across files are "
            "removed. Monday holds normal traffic only.",
        )
        use_all = st.checkbox(f"Use every CSV file in the folder ({len(names)})", value=False, key="smp_all")
        budget, strategy, policy = _shared_options(current, "smp", last)
        merge_default = last.merge_web_attacks if last is not None else current.merge_web_attacks
        merge = st.checkbox("Merge the three Web Attack classes into one", value=merge_default,
                            key="smp_merge", help="Brute Force, XSS and Sql Injection become one class, "
                            "Web Attack. The detailed label stays on every row.")
        _origin_caption(current, last, f"Folder: {resolution.path}")
        submitted = st.form_submit_button("Draw sample", type="primary", key="smp_draw")
    if not submitted:
        return None
    files = tuple(names if use_all else chosen)
    if not files:
        st.warning("Choose at least one file, or tick the box to use every file.")
        return None
    return DataRequest(
        source="cicids", data_dir=str(resolution.path), files=files, row_budget=budget,
        nonfinite_strategy=strategy, merge_web_attacks=bool(merge), seed=int(current.seed), conflict_policy=policy,
    )


def _synthetic_form(current: AppSettings) -> DataRequest | None:
    """Form for sampling synthetic flows; returns a request when "Draw sample" is pressed."""
    low, high, flows_default = SYNTHETIC_FLOWS
    share_low, share_high, share_default = SYNTHETIC_SHARE
    last = _last_request("synthetic")
    if last is not None:
        flows_default = min(max(int(last.synthetic_flows), low), high)
        share_default = min(max(round(float(last.synthetic_attack_share), 2), share_low), share_high)
    with st.form("smp_synth_form", border=True):
        left, right = st.columns(2)
        with left:
            flows = st.number_input("Flows to generate", min_value=low, max_value=high, value=flows_default,
                                    step=2_000, key="smp_syn_flows")
        with right:
            share = st.slider("Attack share", min_value=share_low, max_value=share_high, value=share_default,
                              step=0.05, key="smp_syn_share", help="Fraction of generated flows that are attacks.")
        budget, strategy, policy = _shared_options(current, "smp_syn", last)
        _origin_caption(current, last, "Generated flows use the same 77 columns as the real files.")
        submitted = st.form_submit_button("Draw sample", type="primary", key="smp_syn_draw")
    if not submitted:
        return None
    return DataRequest(
        source="synthetic", row_budget=budget, nonfinite_strategy=strategy, seed=int(current.seed),
        synthetic_flows=int(flows), synthetic_attack_share=float(share), conflict_policy=policy,
    )


def _release_cache_row() -> None:
    """A small control that frees the memory held by files read in earlier draws."""
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.caption("Files read by earlier draws stay in memory, so drawing from them again takes seconds whatever "
                   "the options. Release them to free that memory; the next draw reads the files again.")
    with right:
        if st.button("Release cached files", key="smp_release", width="stretch"):
            data_cache.clear_file_cache()
            st.toast("Cached files released.")


def _draw(request: DataRequest) -> PreparedDataset | None:
    """Run preparation inside a status panel that shows per-file progress and the elapsed time."""
    with st.status("Drawing the sample", expanded=True) as status:
        bar = st.progress(0.0, text="Starting")
        log = st.container()
        started = time.perf_counter()

        def report(message: str, fraction: float) -> None:
            bar.progress(min(max(float(fraction), 0.0), 1.0), text=f"{message} · {time.perf_counter() - started:.1f} s")
            if "rows kept" in message:
                log.caption(message)

        try:
            dataset = prepare.prepare_dataset(request, read_file=data_cache.file_reader(),
                                              stage_file=data_cache.file_stager(), progress=report)
        except MemoryError:
            status.update(label="The sample could not be drawn", state="error", expanded=True)
            st.error("Not enough memory for this selection. Choose fewer files or a smaller row budget.")
            return None
        except (DataFileError, ValueError, OSError) as exc:
            status.update(label="The sample could not be drawn", state="error", expanded=True)
            st.error(str(exc))
            return None
        status.update(label=f"Sample drawn in {dataset.seconds:.1f} s", state="complete", expanded=False)
    return dataset


def _class_table(dataset: PreparedDataset) -> None:
    """Per-class rows available and sampled, with the ○/◆ shape marks."""
    table = dataset.class_table()
    table["Class"] = [
        f"{theme.GLYPH_NORMAL if is_benign(c) else theme.GLYPH_ATTACK} {c}" for c in table["Class"]
    ]
    st.dataframe(
        table.drop(columns=["Kind"]), hide_index=True, width="stretch",
        column_config={
            "Available": st.column_config.NumberColumn("Available", format="localized",
                                                       help="Rows left after cleaning and de-duplication"),
            "In sample": st.column_config.NumberColumn("In sample", format="localized"),
            "Share of sample": st.column_config.NumberColumn("Share of sample", format="percent"),
        },
    )


def _counts_text(counts: dict[str, int], limit: int = 6) -> str:
    """Summarise a count dictionary as text, e.g. ``BENIGN 1,234; DoS Hulk 56``."""
    items = list(counts.items())
    text = "; ".join(f"{name} {count:,}" for name, count in items[:limit])
    if len(items) > limit:
        text += f"; {len(items) - limit} more"
    return text


def _notable_fixes(read: object) -> list[str]:
    """Non-routine repairs of one read report (none for a report made by older code still held in the session)."""
    fixes = getattr(read, "fixes", None)
    return fixes(include_routine=False) if callable(fixes) else []


def _file_fix_notes(dataset: PreparedDataset) -> None:
    """Repairs made while reading that deserve attention; a warning when a repeated column did not match."""
    for report in dataset.file_reports:
        mismatched = getattr(report.read, "duplicate_columns_mismatched", [])
        if mismatched:
            st.warning(f"{report.name}: the repeated column {', '.join(mismatched)} does not match its first copy "
                       "in every row. The first copy was kept; check that the file is an unaltered export.")
    notable = [(r.name, _notable_fixes(r.read)) for r in dataset.file_reports]
    notable = [(name, fixes) for name, fixes in notable if fixes]
    if notable:
        lines = " ".join(f"{name}: {'; '.join(fixes)}." for name, fixes in notable)
        st.markdown(f"**File fixes.** {lines}")
    else:
        routine = sorted({c for r in dataset.file_reports for c in r.read.duplicate_columns_dropped})
        if routine:
            st.markdown(f"**File fixes.** Only the routine one: the repeated {', '.join(routine)} column was "
                        "identical to its first copy and was dropped.")
        else:
            st.markdown("**File fixes.** None were needed.")


def _notes(dataset: PreparedDataset) -> None:
    """Short notes on file repairs, bad values, duplicates, conflicting labels and degenerate columns."""
    _file_fix_notes(dataset)
    nf = dataset.nonfinite
    if nf.rows_affected:
        columns = ", ".join(f"{c} ({n:,})" for c, n in list(nf.by_column.items())[:4])
        action = {
            "drop": "They were dropped.",
            "impute": "They were kept; infinities became gaps that each channel fills with its training median.",
            "recompute": (f"Rates were rebuilt for {sum(nf.recomputed.values()):,} cells; "
                          f"{nf.rows_left_with_gaps:,} rows still hold gaps, filled per channel."),
        }[nf.strategy]
        st.markdown(f"**Bad values.** {nf.rows_affected:,} rows held infinite or missing values "
                    f"(cells per column: {columns}). {action} By class: {_counts_text(nf.by_class)}.")
    else:
        st.markdown("**Bad values.** No infinite or missing values were found.")
    within, across = dataset.within_duplicates, dataset.across_duplicates
    dup_text = f"**Duplicates.** {within.rows_removed:,} exact repeats removed within files"
    if within.by_class:
        dup_text += f" ({_counts_text(within.by_class, 4)})"
    dup_text += f", and {across.rows_removed:,} across files"
    if across.by_class:
        dup_text += f" ({_counts_text(across.by_class, 4)})"
    st.markdown(dup_text + ". The earliest file keeps its copy.")
    conflicts = dataset.conflicts
    if conflicts.groups:
        fate = {"keep": "They are kept, so the channels are measured on these hard cases too.",
                "majority": (f"The majority label was kept (a group without one loses every row); "
                             f"{conflicts.rows_removed:,} rows were removed."),
                "drop": f"All {conflicts.rows_removed:,} of them were removed."}[conflicts.policy]
        groups = (f"{conflicts.groups:,} groups of identical flows carry" if conflicts.groups != 1
                  else "1 group of identical flows carries")
        st.markdown(f"**Conflicting labels.** {groups} different labels "
                    f"({conflicts.rows:,} rows: {_counts_text(conflicts.by_class, 4)}). {fate}")
    else:
        st.markdown("**Conflicting labels.** No identical flows carry different labels.")
    deg = dataset.degenerate
    if deg.constant or deg.duplicate_of:
        parts = []
        if deg.constant:
            parts.append(f"{len(deg.constant)} constant in this sample ({', '.join(deg.constant)})")
        if deg.duplicate_of:
            pairs = ", ".join(f"{a} = {b}" for a, b in deg.duplicate_of.items())
            parts.append(f"{len(deg.duplicate_of)} repeat an earlier column ({pairs})")
        st.markdown(f"**Degenerate columns.** {' and '.join(parts)}. They are left out of the "
                    "\"all numeric\" feature set.")


def _file_table(dataset: PreparedDataset) -> None:
    """Per-source table: rows, bad values, duplicates, encoding, engine, this draw's time and the repairs."""
    table = dataset.file_table()
    config = {c: st.column_config.NumberColumn(c, format="localized") for c in
              ("Rows read", "Bad-value rows", "Duplicates in file", "Rows after file stage", "Rows in sample")}
    config["Seconds"] = st.column_config.NumberColumn(
        "Seconds", format="%.2f", help="Time this draw spent on the file. A file read by an earlier draw comes "
        "from memory; its engine is marked cached.")
    config["File fixes"] = st.column_config.TextColumn(
        "File fixes", help="Repairs made while reading: repeated columns, unknown columns, text in number "
        "columns, empty labels, a fallback parser.")
    st.dataframe(table, hide_index=True, width="stretch", column_config=config)


def _sample_sheet(dataset: PreparedDataset) -> None:
    """Everything known about the stored sample."""
    st.subheader("Sample sheet", anchor=False)
    details = [dataset.request.describe(), f"drawn in {dataset.seconds:.1f} s"]
    if dataset.peak_memory_mb is not None:
        memory = f"process peak working set {dataset.peak_memory_mb:,.0f} MB"
        rise = getattr(dataset, "memory_rise_mb", None)
        if rise is not None:
            memory += f" (+{rise:,.0f} MB during this draw)"
        details.append(memory)
    details.append(f"fingerprint {dataset.fingerprint[:12]}")
    st.caption(" · ".join(details))
    if dataset.single_class:
        st.warning(dataset.single_class_message or "Only one class in this sample.")
    components.reading_cards(dataset.summary_rows())
    left, right = st.columns([1, 1.25], gap="medium")
    with left:
        st.markdown("**Classes before and after sampling**")
        _class_table(dataset)
        if dataset.sampling.sampled:
            lowered = " (floor lowered to fit the budget)" if dataset.sampling.floor_shrunk else ""
            st.caption(f"Each class first received up to {dataset.sampling.floor:,} rows{lowered}; the rest of "
                       "the budget was shared in proportion to class size.")
        else:
            st.caption("The data fit within the budget, so every row was kept.")
    with right:
        # Drawn from the spec without Altair's schema checks (see viz.chart_spec): this sheet is redrawn on every
        # rerun of the station, and the checks cost more than the drawing.
        spec = viz.chart_spec(lambda: viz.class_distribution_chart(
            dataset.sampling.before, dataset.sampling.after, components.current_mode()))
        st.vega_lite_chart(spec=spec, width="stretch", theme=None)
    st.markdown("**Sources**")
    _file_table(dataset)
    _notes(dataset)


def render() -> None:
    """Draw the 01 Sample station."""
    components.station_header("sample")
    notice = st.session_state.pop(NOTICE_KEY, None)
    if notice:
        st.toast(notice)
    current = state.settings()
    resolution = resolve_data_dir(current)
    options = ["cicids", "synthetic"] if resolution.usable else ["synthetic"]
    if not resolution.usable:
        if resolution.source == "none":
            components.needs("No data folder is set, so this station draws synthetic flows. To sample the "
                             "CIC-IDS2017 files, set their folder on the Bench.", "bench")
        else:
            components.needs(f"{resolution.problem or 'The data folder cannot be used.'} Synthetic flows are "
                             "available meanwhile; correct the folder on the Bench.", "bench")
    stored = state.get_prepared()
    start = options.index(stored.request.source) if stored is not None and stored.request.source in options else 0
    source = st.radio("Data source", options, index=start, format_func=lambda s: SOURCE_LABELS[s], horizontal=True,
                      key=f"smp_source_{len(options)}")
    request = _files_form(current, resolution) if source == "cicids" else _synthetic_form(current)
    if source == "cicids":
        _release_cache_row()
    failed = False
    if request is not None:
        dataset = _draw(request)
        if dataset is not None:
            state.set_prepared(dataset)
            state.mark_done("sample")
            st.session_state[NOTICE_KEY] = f"Sample drawn in {dataset.seconds:.1f} s."
            st.rerun()
        failed = True

    dataset = state.get_prepared()
    if dataset is None:
        st.caption("No sample drawn in this session yet. Choose a source and press Draw sample.")
        return
    if failed:
        st.info(f"The new draw failed, so nothing was replaced: the sheet below is still the previous sample "
                f"(fingerprint {dataset.fingerprint[:12]}), and the later stations keep using it.")
    _sample_sheet(dataset)
