"""06 Sweep station: stream flows through one fitted channel and watch the live readings, the feed and the alerts.

A CIC-IDS2017 run replays its real held-out flows with their true labels (never generated ones), so the live
readings measure the channel exactly as 03 Measure does, one flow at a time. A run fitted on synthetic data streams
fresh generator flows, or replays its held-out rows. The engine is :mod:`nids.simulate`; this page only keeps
one :class:`~nids.simulate.SimulationSession` per browser session (under :data:`SESSION_KEY`) and draws it.

Settings change nothing until they are used: the first Start or Step once builds the stream from them, and later
changes wait for Apply (a new channel, stream, attack share, mix or seed starts a fresh stream; pace, interval and
alert threshold change in place). While the stream runs, a fragment refreshes ONLY the live panel once per tick
interval, stepping the session by one tick each time; the rest of the page does not rerun. Pause keeps every
reading, Reset clears the stream. Nothing here ever fits a model.

07 Record reads the stream through :func:`current_session` (its log is
:meth:`~nids.simulate.SimulationSession.log_frame`).
"""

from __future__ import annotations

import html
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from nids import simulate, theme, viz
from nids.models.jobs import busy_message, claim_slot, release_slot, slot_holder
from nids.models.train import TrainingRun
from nids.simulate import FlowEvent, SimulationSession, SourceKind
from nids.theme import Mode
from ui import components, state
from ui.stations import BY_KEY, PAGE_OBJECTS
from ui.training_ui import channel_label

#: Session keys: the simulation session, the settings it runs with, whether it runs, when it last stepped, a notice.
SESSION_KEY = "g_sweep"
CONFIG_KEY = "g_sweep_config"
RUNNING_KEY = "g_sweep_running"
LAST_STEP_KEY = "g_sweep_last_step"
NOTICE_KEY = "g_sweep_notice"
#: The button pressed, carried over to the script run its click starts (see :func:`_request`).
ACTION_KEY = "g_sweep_action"
#: Copies of the settings widgets' values, kept while the station is not drawn (see :func:`_restore`).
SHADOW_KEY = "g_sweep_kept_widgets"
#: The last detections chart built, with what it shows (see :func:`_timeline_spec`).
CHART_KEY = "g_sweep_chart"
ACTIONS: tuple[str, ...] = ("start", "pause", "step", "reset", "apply")
#: Pace (flows per tick) and tick interval choices.
PACE_RANGE = (1, 500)
PACE_DEFAULT = 25
INTERVALS: tuple[float, ...] = tuple(x / 2 for x in range(1, 11))
INTERVAL_DEFAULT = 1.0
#: Rows shown in the feed and in the alert list.
FEED_ROWS = 50
ALERT_ROWS = 20
#: A running stream steps when at least this share of the tick interval has passed since its last step, so a full
#: rerun of the page between two ticks never adds an extra step.
STEP_GUARD = 0.5
#: Alert threshold limits (as on the Bench).
THRESHOLD_RANGE = (0.5, 0.999)
STREAM_LABELS: dict[str, str] = {
    "synthetic": "Fresh synthetic flows",
    "replay": "Held-out rows of this run",
}
SHARE_OPTIONS = ("Natural", "Set the share")
MIX_OPTIONS = ("Natural mix", "Custom mix")
#: Attribute of a run holding the stream facts computed here (attack types, natural share and mix).
FACTS_ATTR = "sweep_stream_facts"


@dataclass(frozen=True)
class SweepConfig:
    """The settings of a stream. ``share`` None is the natural attack share; ``mix`` None the natural attack mix,
    else (attack type, weight) pairs. ``problem`` says why the settings cannot start a stream (None when they can)."""

    run_id: str
    channel: str
    stream: SourceKind
    pace: int
    interval: float
    share: float | None
    mix: tuple[tuple[str, float], ...] | None
    threshold: float
    seed: int
    problem: str | None = None

    @property
    def flows_per_second(self) -> float:
        """Flows streamed per second at this pace and interval."""
        return self.pace / self.interval

    def restart_key(self) -> tuple[Any, ...]:
        """The settings that define the stream itself: changing any of them starts a fresh stream."""
        return (self.run_id, self.channel, self.stream, self.share, self.mix, self.seed)


@dataclass(frozen=True)
class StreamFacts:
    """What a stream of a run can hold: its attack types and the natural share and mix (shares sum to 1)."""

    attack_types: tuple[str, ...]
    natural_share: float
    natural_mix: dict[str, float]


# --------------------------------------------------------------------------------------------------------------
# Session access
# --------------------------------------------------------------------------------------------------------------
def _stored_session() -> SimulationSession | None:
    """The simulation session stored in this browser session, whichever run it belongs to."""
    value = st.session_state.get(SESSION_KEY)
    # Duck-typed: after a code reload the stored object's class is an older copy of SimulationSession.
    return value if value is not None and hasattr(value, "step") and hasattr(value, "log_frame") else None


def _streams_run(session: Any, run: TrainingRun | None) -> bool:
    """True when ``session`` streams ``run`` itself.

    The run object is compared, not only its id: a fit and its copy loaded from disk share one run id, but the
    loaded copy may lack held-out rows or a channel (CH3 is not saved unless chosen), so a stream of the fit must
    not carry over to it. A stored session without a ``run`` attribute falls back to the id.
    """
    if session is None or run is None:
        return False
    streamed = getattr(session, "run", None)
    if streamed is not None:
        return streamed is run
    return getattr(session, "run_id", None) == run.run_id


def current_session() -> SimulationSession | None:
    """This session's stream of the CURRENT run, or None (no stream yet, or it belongs to an earlier run, or to
    the same run before it was loaded again from disk)."""
    session = _stored_session()
    if session is None:
        return None
    return session if _streams_run(session, state.current_run()) else None


def _applied() -> SweepConfig | None:
    """The settings the stored stream runs with."""
    value = st.session_state.get(CONFIG_KEY)
    return value if value is not None and hasattr(value, "restart_key") else None


def _running() -> bool:
    """True while the stream advances by itself."""
    return bool(st.session_state.get(RUNNING_KEY, False))


def _clear(note: str | None = None) -> None:
    """Forget the stream and its settings (and stop it); ``note`` is shown once."""
    for key in (SESSION_KEY, CONFIG_KEY, RUNNING_KEY, LAST_STEP_KEY, CHART_KEY):
        st.session_state.pop(key, None)
    if note:
        _notice("info", note)


def _notice(kind: str, text: str) -> None:
    """Leave a message for the next drawing of this station (shown once)."""
    st.session_state[NOTICE_KEY] = (kind, text)


def _show_notice() -> None:
    """Show the pending message, if any."""
    notice = st.session_state.pop(NOTICE_KEY, None)
    if notice is None:
        return
    kind, text = notice
    {"success": st.success, "info": st.info, "warning": st.warning}.get(kind, st.error)(text)


# --------------------------------------------------------------------------------------------------------------
# Building and driving the stream
# --------------------------------------------------------------------------------------------------------------
def stream_facts(run: TrainingRun, kind: SourceKind) -> StreamFacts:
    """Attack types and the natural share and mix of stream ``kind`` of ``run`` (computed once, kept on the run)."""
    store = run.__dict__.setdefault(FACTS_ATTR, {})
    facts = store.get(kind)
    if facts is None:
        source = simulate.make_source(run, kind)
        share = (float(source.natural_share) if isinstance(source, simulate.ReplaySource)
                 else float(source.effective_share))
        facts = StreamFacts(tuple(source.attack_types), share, dict(source.natural_mix))
        store[kind] = facts
    return facts


def build_session(run: TrainingRun, config: SweepConfig) -> SimulationSession:
    """A fresh simulation session for ``run`` with ``config`` (raises ValueError when it cannot be built)."""
    if config.problem:
        raise ValueError(config.problem)
    source = simulate.make_source(run, config.stream, attack_share=config.share,
                                  mix=dict(config.mix) if config.mix is not None else None, seed=config.seed)
    return SimulationSession(run, config.channel, source, alert_threshold=config.threshold)


def _ensure_session(config: SweepConfig) -> SimulationSession | None:
    """The current stream, built from ``config`` first when there is none; None (with a notice) on failure."""
    session = current_session()
    if session is not None:
        return session
    run = state.current_run()
    if run is None:
        _notice("warning", "There is no fitted run to stream. Fit channels at 02 Fit first.")
        return None
    try:
        session = build_session(run, config)
    except (ValueError, RuntimeError) as exc:
        _notice("error", f"The stream could not start: {state.first_line(str(exc))}")
        return None
    st.session_state[SESSION_KEY] = session
    st.session_state[CONFIG_KEY] = config
    st.session_state[LAST_STEP_KEY] = 0.0
    return session


def _advance(session: SimulationSession, config: SweepConfig) -> bool:
    """One tick: score ``config.pace`` flows; True when it worked.

    A failure stops the stream and leaves a notice instead of raising. A tick marks 06 Sweep done on the stepper.
    """
    try:
        session.step(int(config.pace))
    except Exception as exc:  # noqa: BLE001 - a failing channel must stop the stream, not break the page
        st.session_state[RUNNING_KEY] = False
        _notice("error", f"The stream stopped: {type(exc).__name__}: {state.first_line(str(exc))}")
        return False
    finally:
        st.session_state[LAST_STEP_KEY] = time.monotonic()
    state.mark_done("sweep")
    return True


def _due(config: SweepConfig) -> bool:
    """True when a running stream should take its next step (see :data:`STEP_GUARD`)."""
    last = float(st.session_state.get(LAST_STEP_KEY, 0.0) or 0.0)
    return last <= 0.0 or time.monotonic() - last >= float(config.interval) * STEP_GUARD


def _request(action: str) -> None:
    """Button callback: remember which button was pressed. The page performs it once the settings are read
    (:func:`perform`), so a value typed just before the click is the one used."""
    st.session_state[ACTION_KEY] = action


def perform(action: str | None, config: SweepConfig) -> None:
    """Carry out a button press (``start``, ``pause``, ``step``, ``reset`` or ``apply``) with the settings shown."""
    if action == "start":
        _start(config)
    elif action == "pause":
        _pause()
    elif action == "step":
        _step_once(config)
    elif action == "reset":
        _reset()
    elif action == "apply":
        _apply(config)


def _start(config: SweepConfig) -> None:
    """Start: build the stream if needed, then let it run (its first tick comes with this redraw)."""
    if _ensure_session(config) is None:
        return
    st.session_state[RUNNING_KEY] = True
    st.session_state[LAST_STEP_KEY] = 0.0


def _pause() -> None:
    """Pause: stop advancing; every reading stays."""
    st.session_state[RUNNING_KEY] = False


def _step_once(config: SweepConfig) -> None:
    """Step once: score a single tick (building the stream first if needed)."""
    session = _ensure_session(config)
    applied = _applied()
    if session is not None and applied is not None:
        _advance(session, applied)


def _reset() -> None:
    """Reset: forget the stream (the next Start or Step once begins a fresh one)."""
    _clear("The stream was cleared. Start or Step once begins a fresh one with the settings shown.")


def _apply(config: SweepConfig) -> None:
    """Apply: use the changed settings (a fresh stream when the stream itself changed)."""
    session = current_session()
    applied = _applied()
    if session is None or applied is None:
        return
    if config.restart_key() != applied.restart_key():
        run = state.current_run()
        if run is None:
            return
        try:
            fresh = build_session(run, config)
        except (ValueError, RuntimeError) as exc:
            _notice("error", f"The new settings could not start a stream: {state.first_line(str(exc))}")
            return
        st.session_state[SESSION_KEY] = fresh
        st.session_state[LAST_STEP_KEY] = 0.0
        _notice("info", "A fresh stream started with the new settings; the earlier readings were cleared.")
    else:
        session.alert_threshold = config.threshold
        _notice("info", "Pace, interval and alert threshold now apply; the stream and its readings carry on (the "
                        "threshold applies to flows from now on).")
    st.session_state[CONFIG_KEY] = config


# --------------------------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------------------------
def _percent(value: float) -> str:
    """A share as a percentage with one decimal, e.g. ``"34.5 %"``."""
    return f"{100 * float(value):.1f} %"


def _attack_text(name: str) -> str:
    """An attack type with its shape cue (``"◆ DoS Hulk"``)."""
    return f"{theme.GLYPH_ATTACK} {name}"


def _restore(key: str, default: Any, options: list[Any] | None = None) -> None:
    """Before drawing a settings widget: give it back its last value (else ``default``).

    Streamlit forgets the state of a widget that is not drawn in a run, so after a visit to another station the
    settings would come back at their defaults: the panel would no longer show the settings the stream runs with,
    Apply would look pending, and pressing it would restart the stream. :func:`_keep` saves a copy of each value
    after it is drawn. A kept value that is no longer allowed (``options``) gives way to ``default``.
    """
    if key in st.session_state:
        if options is not None and st.session_state[key] not in options:
            st.session_state[key] = default
        return
    kept = st.session_state.get(SHADOW_KEY, {})
    value = kept.get(key, default)
    if options is not None and value not in options:
        value = default
    st.session_state[key] = value


def _keep(key: str) -> None:
    """After drawing a settings widget: save a copy of its value for :func:`_restore`."""
    st.session_state.setdefault(SHADOW_KEY, {})[key] = st.session_state.get(key)


def _mix_inputs(run: TrainingRun, stream: SourceKind, facts: StreamFacts,
                applied: SweepConfig | None) -> tuple[tuple[str, float], ...] | str:
    """The custom attack-mix widgets: the types and one weight each. Returns the mix, or a problem text."""
    if not facts.attack_types:
        return "This stream holds no attack types to mix."
    base = f"{run.run_id}-{stream}"
    types_key = f"sw_mix_types-{base}"
    used = dict(applied.mix) if applied is not None and applied.mix is not None else None
    _restore(types_key, [t for t in facts.attack_types if used is None or t in used])
    known = [t for t in st.session_state[types_key] if t in facts.attack_types]
    if known != list(st.session_state[types_key]):
        st.session_state[types_key] = known
    chosen = st.multiselect("Attack types to stream", list(facts.attack_types), format_func=_attack_text,
                            key=types_key,
                            help="Types left out are not streamed. The weights below share the attack flows.")
    _keep(types_key)
    if not chosen:
        return "Choose at least one attack type for the custom mix."
    st.caption("Relative weights (the defaults are each type's natural share of the attacks, in %):")
    pairs: list[tuple[str, float]] = []
    columns = st.columns(4, gap="small")
    for i, name in enumerate(t for t in facts.attack_types if t in chosen):
        index = facts.attack_types.index(name)
        natural = max(round(100.0 * facts.natural_mix.get(name, 0.0), 2), 0.01)
        weight_key = f"sw_w-{base}-{index}"
        _restore(weight_key, float(used[name]) if used is not None and name in used else float(natural))
        with columns[i % 4]:
            weight = st.number_input(name, min_value=0.0, max_value=1_000.0, step=1.0, format="%.2f",
                                     key=weight_key)
        _keep(weight_key)
        pairs.append((name, float(weight)))
    if not any(weight > 0 for _, weight in pairs):
        return "Give at least one attack type a weight above 0."
    return tuple(pairs)


def _settings(run: TrainingRun, kinds: list[SourceKind]) -> SweepConfig:
    """Draw the stream settings and return them (nothing is applied here).

    Every widget starts from its last value in this session (see :func:`_restore`), else from the settings the
    current stream runs with, else from the defaults (the Bench's threshold and seed).
    """
    defaults = state.settings()
    applied = _applied() if current_session() is not None else None
    low, high = THRESHOLD_RANGE
    channels = list(run.ok_channels())
    with st.container(border=True, key="sw_settings"):
        left, middle, right = st.columns([3, 2, 2], gap="medium")
        with left:
            channel_key = f"sw_channel-{run.run_id}"
            _restore(channel_key, applied.channel if applied is not None and applied.channel in channels
                     else channels[0], channels)
            channel = st.selectbox("Channel", channels, format_func=channel_label, key=channel_key)
            _keep(channel_key)
            if len(kinds) > 1:
                stream_key = f"sw_stream-{run.run_id}"
                _restore(stream_key, applied.stream if applied is not None and applied.stream in kinds else kinds[0],
                         list(kinds))
                stream: SourceKind = st.radio("Stream", kinds, format_func=lambda k: STREAM_LABELS.get(k, k),
                                              horizontal=True, key=stream_key,
                                              help="Fresh flows come from the generator the run's sample came "
                                                   "from; held-out rows are the very rows 03 Measure reads.")
                _keep(stream_key)
            else:
                stream = kinds[0]
        with middle:
            _restore("sw_pace", int(applied.pace) if applied is not None else PACE_DEFAULT)
            pace = int(st.number_input("Pace (flows per tick)", min_value=PACE_RANGE[0], max_value=PACE_RANGE[1],
                                       step=5, key="sw_pace"))
            _restore("sw_interval", float(applied.interval) if applied is not None else INTERVAL_DEFAULT,
                     list(INTERVALS))
            interval = float(st.select_slider("Tick interval (s)", options=list(INTERVALS), key="sw_interval"))
            _keep("sw_pace")
            _keep("sw_interval")
            st.markdown(f'<span class="g-mono">{pace / interval:,.1f}</span> flows/s '
                        f'({pace:,} every {interval:g} s)', unsafe_allow_html=True)
        with right:
            _restore("sw_threshold", float(applied.threshold) if applied is not None
                     else float(min(max(defaults.alert_threshold, low), high)))
            threshold = float(st.number_input(
                "Alert threshold (attack probability)", min_value=low, max_value=high, step=0.01, format="%.3f",
                key="sw_threshold", help="A flow read as an attack raises an alert when its attack probability "
                "(binary: P(Attack); multi-class: 1 - P(BENIGN)) reaches this value, as at 04 Probe and 05 Assay. "
                "Default from the Bench."))
            _restore("sw_seed", int(applied.seed) if applied is not None else int(defaults.seed))
            seed = int(st.number_input("Seed", min_value=0, max_value=2**31 - 2, step=1, key="sw_seed",
                                       help="The same seed and settings replay the same stream."))
            _keep("sw_threshold")
            _keep("sw_seed")
        facts = stream_facts(run, stream)
        base = f"{run.run_id}-{stream}"
        same_stream = applied if applied is not None and applied.stream == stream else None
        share_left, mix_right = st.columns([2, 3], gap="medium")
        with share_left:
            natural_label = ("Natural (the held-out share)" if stream == "replay"
                             else "Natural (the share the run's sample was generated with)")
            share_mode_key, share_key = f"sw_share_mode-{base}", f"sw_share-{base}"
            set_share = same_stream is not None and same_stream.share is not None
            _restore(share_mode_key, SHARE_OPTIONS[1] if set_share else SHARE_OPTIONS[0], list(SHARE_OPTIONS))
            share_mode = st.radio(f"Attack share: natural is {_percent(facts.natural_share)}", SHARE_OPTIONS,
                                  horizontal=True, key=share_mode_key, help=natural_label)
            _restore(share_key, int(round(100 * (same_stream.share if set_share else facts.natural_share))))
            share_value = st.slider("Attack share (%)", min_value=0, max_value=100, step=1, key=share_key,
                                    disabled=share_mode == SHARE_OPTIONS[0])
            _keep(share_mode_key)
            _keep(share_key)
        share = None if share_mode == SHARE_OPTIONS[0] else share_value / 100.0
        with mix_right:
            mix_mode_key = f"sw_mix_mode-{base}"
            custom = same_stream is not None and same_stream.mix is not None
            _restore(mix_mode_key, MIX_OPTIONS[1] if custom else MIX_OPTIONS[0], list(MIX_OPTIONS))
            mix_mode = st.radio(f"Attack mix: {len(facts.attack_types)} attack types", MIX_OPTIONS, horizontal=True,
                                key=mix_mode_key)
            _keep(mix_mode_key)
            mix: tuple[tuple[str, float], ...] | None = None
            problem: str | None = None
            if mix_mode == MIX_OPTIONS[1]:
                picked = _mix_inputs(run, stream, facts, same_stream)
                if isinstance(picked, str):
                    problem = picked
                    st.caption(picked)
                else:
                    mix = picked
            else:
                top = ", ".join(f"{name} {_percent(share_of)}" for name, share_of in
                                list(facts.natural_mix.items())[:4])
                more = len(facts.natural_mix) - 4
                where = "As held out" if stream == "replay" else "As the generator makes them"
                tail = f" and {more} more." if more > 0 else "."
                st.caption(f"{where}: {top}{tail}" if top else "This stream holds no attack types.")
    return SweepConfig(run_id=run.run_id, channel=str(channel), stream=stream, pace=pace, interval=interval,
                       share=share, mix=mix, threshold=threshold, seed=seed, problem=problem)


def _buttons(config: SweepConfig) -> None:
    """Start, Pause, Step once, Reset and Apply, with their states."""
    session = current_session()
    applied = _applied()
    running = _running()
    pending = session is not None and applied is not None and config != applied
    blocked = session is None and config.problem is not None
    columns = st.columns(5, gap="small")
    with columns[0]:
        st.button("Start", key="sw_start", type="primary", on_click=_request, args=("start",),
                  disabled=running or blocked, width="stretch")
    with columns[1]:
        st.button("Pause", key="sw_pause", on_click=_request, args=("pause",), disabled=not running,
                  width="stretch")
    with columns[2]:
        st.button("Step once", key="sw_step", on_click=_request, args=("step",), disabled=running or blocked,
                  width="stretch")
    with columns[3]:
        st.button("Reset", key="sw_reset", on_click=_request, args=("reset",), disabled=session is None,
                  width="stretch")
    with columns[4]:
        st.button("Apply settings", key="sw_apply", on_click=_request, args=("apply",),
                  disabled=not pending or config.problem is not None, width="stretch")
    if pending:
        st.caption("Changed settings wait for Apply. A new channel, stream, attack share, mix or seed starts a fresh "
                   "stream; pace, interval and alert threshold change in place.")
    elif session is None:
        st.caption("Start streams one tick per interval; Step once scores a single tick. Both use the settings above.")


# --------------------------------------------------------------------------------------------------------------
# The live panel
# --------------------------------------------------------------------------------------------------------------
def _verdict(label: str) -> str:
    """A verdict with its shape cue (``"○ Normal"``, ``"◆ Attack"``, ``"◆ DoS Hulk"``)."""
    return theme.verdict_text(str(label))


def _truth(event: FlowEvent) -> str:
    """The true class with its shape cue, plus the detailed label when it says more (``"◆ Attack (DoS Hulk)"``)."""
    text = _verdict(event.true_label)
    detail = str(event.detail or "")
    if detail and detail != event.true_label and not (simulate.is_normal_class(detail)
                                                     and simulate.is_normal_class(event.true_label)):
        text += f" ({detail})"
    return text


def _reading(event: FlowEvent) -> str:
    """Whether the verdict matched the true class, in words (as at 04 Probe; the only glyphs are the class and
    alert marks)."""
    return "Correct" if event.correct else "Incorrect"


def feed_frame(events: list[FlowEvent]) -> pd.DataFrame:
    """The feed as a display table (newest first): verdict, attack probability, truth, reading, alert."""
    rows = [{
        "Flow": e.seq, "Tick": e.tick, "Verdict": _verdict(e.predicted), "P(attack)": e.attack_probability,
        "True class": _truth(e), "Reading": _reading(e),
        "Alert": f"{theme.GLYPH_ALERT} Alert" if e.alert else "",
    } for e in events]
    return pd.DataFrame(rows, columns=["Flow", "Tick", "Verdict", "P(attack)", "True class", "Reading", "Alert"])


def _styled_feed(frame: pd.DataFrame, mode: Mode) -> Any:
    """The feed table with alert rows on the warning tint (text in the warning text colour)."""
    p = theme.palette(mode)
    alert_style = f"background-color: {p.warning_tint}; color: {p.warning_text}"

    def paint(row: pd.Series) -> list[str]:
        return [alert_style if row["Alert"] else ""] * len(row)

    return frame.style.apply(paint, axis=1).format({"P(attack)": "{:.3f}"})


def _readings(session: SimulationSession, config: SweepConfig) -> None:
    """The headline readings of the stream as cards."""
    stats = session.stats

    def score(value: float) -> str:
        return theme.score_text(value, missing="–")

    seen = stats.classes_seen
    components.reading_cards([
        {"Reading": "Flows seen", "Value": stats.emitted,
         "Note": f"{stats.ticks:,} tick{'s' if stats.ticks != 1 else ''}"},
        {"Reading": "Live accuracy", "Value": score(stats.live_accuracy),
         "Note": f"{stats.correct:,} read correctly"},
        {"Reading": "Live balanced accuracy", "Value": score(stats.live_balanced_accuracy),
         "Note": f"mean recall over {seen} class{'es' if seen != 1 else ''} seen"},
        {"Reading": f"{theme.GLYPH_ALERT} Alerts", "Value": stats.alerts_total,
         "Note": f"attack verdicts at P(attack) ≥ {session.alert_threshold:.3f}"},
        {"Reading": "Flows/s", "Value": f"{config.flows_per_second:,.1f}",
         "Note": f"{config.pace:,} per tick, every {config.interval:g} s"},
        {"Reading": "Attack share seen", "Value": "–" if not stats.emitted else _percent(stats.attack_share_seen),
         "Note": f"{stats.attacks_seen:,} true attacks"},
    ])


def _alert_list(session: SimulationSession) -> None:
    """The newest high-confidence alerts."""
    stats = session.stats
    alerts = session.alerts[:ALERT_ROWS]
    st.markdown(f"**{theme.GLYPH_ALERT} High-confidence alerts**")
    if not alerts:
        st.caption(f"No alerts yet: no flow read as attack has reached P(attack) ≥ {session.alert_threshold:.3f}.")
        return
    frame = pd.DataFrame([{
        "Flow": e.seq, "Verdict": _verdict(e.predicted), "P(attack)": e.attack_probability,
        "True class": _truth(e), "Reading": _reading(e),
    } for e in alerts])
    st.dataframe(frame, hide_index=True, width="stretch", key="sw_alerts_table",
                 column_config={"P(attack)": st.column_config.NumberColumn("P(attack)", format="%.3f"),
                                "Flow": st.column_config.NumberColumn("Flow", format="%d")})
    st.caption(f"{stats.alerts_total:,} alert{'s' if stats.alerts_total != 1 else ''} so far; the newest "
               f"{len(alerts)} shown (the list keeps the newest {simulate.MAX_ALERTS:,}).")


def _confusion(session: SimulationSession) -> None:
    """The running confusion counts (rows: true class, columns: verdict)."""
    stats = session.stats
    names = [_verdict(c) for c in stats.classes]
    frame = pd.DataFrame(stats.confusion, index=names, columns=names)
    frame.index.name = "True class"
    with st.expander("Live confusion counts"):
        st.dataframe(frame, width="stretch", key="sw_confusion_table")
        st.caption("Rows: true class. Columns: the channel's verdict. Counts over every flow streamed so far.")


def _status_line(session: SimulationSession, config: SweepConfig) -> None:
    """Running or paused, which channel, which stream."""
    stats = session.stats
    state_text = "Running" if _running() else ("Paused" if stats.ticks else "Ready")
    cost = (f" · scoring {1000 * stats.scoring_seconds / stats.ticks:,.1f} ms per tick" if stats.ticks else "")
    st.markdown(f"**{state_text}** · tick <span class=\"g-mono\">{stats.ticks:,}</span> · "
                f"{html.escape(session.channel_label)} · {html.escape(session.source.describe())}{cost}",
                unsafe_allow_html=True)
    notes = list(stats.notes)
    if stats.repeated:
        notes.append(f"Every held-out row of a pool has been replayed at least once, so {stats.repeated:,} of the "
                     f"{stats.emitted:,} flows so far repeat a row (a used-up pool is reshuffled).")
    for note in notes:
        st.caption(note)


def _timeline_spec(session: SimulationSession, mode: Mode) -> dict[str, Any]:
    """The detections-over-time chart of ``session``, built again only after a new tick (or in another theme):
    a paused stream, or a settings change, redraws the page without rebuilding the chart."""
    stats = session.stats
    key = (stats.ticks, stats.emitted, mode)
    kept = st.session_state.get(CHART_KEY)
    if kept is not None and kept[0] is session and kept[1] == key:
        return kept[2]
    spec = viz.chart_spec(lambda: viz.detections_chart(session.timeline, mode))
    st.session_state[CHART_KEY] = (session, key, spec)  # the session itself, so a new stream never matches
    return spec


def _live_panel() -> None:
    """The live panel: take a tick when one is due, then draw readings, chart, feed and alerts.

    Run as a fragment: while the stream runs it refreshes once per tick interval without rerunning the page.
    """
    session = current_session()
    config = _applied()
    if session is None and _stored_session() is not None:
        # The current run changed (a fit finished, a set was loaded) while the stream ran: redraw the whole page,
        # which clears the stream of the earlier run.
        st.session_state[RUNNING_KEY] = False
        st.rerun(scope="app")
    if session is None or config is None:
        st.caption("No stream yet. Start streams one tick per interval; Step once scores a single tick.")
        return
    if _running() and _due(config) and not _advance(session, config):
        st.rerun(scope="app")  # show why the stream stopped, and stop refreshing this panel
    mode = components.current_mode()
    with st.container(key="sw_live"):
        _status_line(session, config)
        _readings(session, config)
        st.vega_lite_chart(spec=_timeline_spec(session, mode), width="stretch", theme=None)
        feed_col, alert_col = st.columns([3, 2], gap="medium")
        with feed_col:
            st.markdown(f"**Feed** (newest first, latest {FEED_ROWS})")
            frame = feed_frame(session.feed[:FEED_ROWS])
            if frame.empty:
                st.caption("Nothing streamed yet.")
            else:
                st.dataframe(_styled_feed(frame, mode), hide_index=True, width="stretch", key="sw_feed_table")
        with alert_col:
            _alert_list(session)
        _confusion(session)


# --------------------------------------------------------------------------------------------------------------
# Runs without held-out rows
# --------------------------------------------------------------------------------------------------------------
def _rebuild(folder: Path, run_id: str) -> None:
    """Rebuild a loaded run's held-out rows (the saved set is loaded again; nothing is fitted), then rerun.

    The rebuild reads the data files, so it takes the app's work slot: it never overlaps a fit or a measurement.
    """
    holder = "a rebuild of held-out rows"
    if not claim_slot(holder):
        _notice("warning", busy_message())
        st.rerun()
    started = time.perf_counter()
    try:
        with st.status(f"Rebuilding the held-out rows of {run_id}", expanded=True) as status:
            bar = st.progress(0.0, text="Starting")

            def report(message: str, fraction: float) -> None:
                bar.progress(min(max(float(fraction), 0.0), 1.0),
                             text=f"{message} · {time.perf_counter() - started:.1f} s")

            result = state.load_bundle_into_session(folder, progress=report)
            status.update(label=f"Finished in {time.perf_counter() - started:.1f} s", state="complete",
                          expanded=False)
    finally:
        release_slot()
    _notice(result.kind if result.rebuilt else "warning", result.message)
    st.rerun()


def _no_rows(run: TrainingRun) -> None:
    """A CIC-IDS2017 run without its held-out rows: explain, offer the rebuild, and link to the Logbook and Bench."""
    components.needs(
        f"Run {run.run_id} was loaded from disk without its held-out rows, so there are no real flows to replay. "
        "This station replays only real held-out CIC-IDS2017 flows with their true labels, never generated ones. "
        "Point the Bench at the data folder the run was fitted on, then rebuild the rows here (or load the set "
        "again at the Logbook): the very same rows are rebuilt and checked against the saved fingerprints.",
        "logbook")
    bench = PAGE_OBJECTS.get("bench")
    if bench is not None:
        st.page_link(bench, label=f"Go to {BY_KEY['bench'].label}", icon=":material/arrow_forward:")
    info = state.loaded_info(run.run_id)
    if info is not None and info.rows_note:
        st.caption(info.rows_note)
    folder = state.bundle_on_disk(run)
    if folder is None:
        st.caption("The folder this set was loaded from is gone, so it cannot be loaded again from here.")
        return
    holder = slot_holder()
    if st.button("Rebuild the held-out rows", key="sw_rebuild", disabled=holder is not None,
                 help="Loads the saved set again with the Bench's data folder: checks it, rebuilds the held-out "
                      "rows and scores them. Nothing is fitted."):
        _rebuild(folder, run.run_id)
    if holder is not None:
        st.caption(f"{holder[:1].upper()}{holder[1:]} is running in this app; the rebuild waits until it ends.")


# --------------------------------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------------------------------
def _source_badge(run: TrainingRun, stream: SourceKind) -> str:
    """The badge naming where the flows come from."""
    if stream == "synthetic":
        return "Synthetic stream: fresh generator flows with their true labels"
    if run.data_request.source == "synthetic":
        return "Replaying held-out synthetic rows with their true labels"
    return "Replaying held-out CIC-IDS2017 flows with their true labels"


def _run_chips(run: TrainingRun) -> list[str]:
    """Short facts about the run being streamed."""
    chips = [f"run {run.run_id}",
             "binary: normal vs attack" if run.request.mode == "binary"
             else f"multi-class: {len(run.data.classes)} classes",
             f"{len(run.ok_channels())} channel{'s' if len(run.ok_channels()) != 1 else ''}"]
    if len(run.data.y_test):
        chips.append(f"{len(run.data.y_test):,} held-out rows")
    if getattr(run, "origin", "fitted") == "loaded":
        chips.append("loaded from disk")
    return chips


def render() -> None:
    """Draw the 06 Sweep station."""
    components.station_header("sweep")
    action = st.session_state.pop(ACTION_KEY, None)  # taken now, so a press never lingers into a later visit
    run = state.current_run()
    stored = _stored_session()
    if stored is not None and not _streams_run(stored, run):
        if run is not None and run.run_id == stored.run_id:
            _clear(f"The stream of run {stored.run_id} was cleared: the run was loaded again from disk, and the "
                   "loaded copy streams afresh.")
        else:
            _clear(f"The stream of run {stored.run_id} was cleared: the current run is now "
                   f"{run.run_id if run is not None else 'none'}.")
    if run is None or not run.ok_channels():
        _show_notice()
        components.needs("Needs a fitted channel: fit at least one at 02 Fit.", "fit")
        return
    kinds = simulate.stream_kinds(run)
    components.chips(_run_chips(run))
    if not kinds:
        _show_notice()
        _no_rows(run)
        return
    applied = _applied()
    if applied is not None and applied.stream not in kinds:
        _clear("The stream was cleared: its source is no longer available for this run.")
        applied = None
    config = _settings(run, kinds)
    perform(action, config)
    _show_notice()
    shown_stream = applied.stream if applied is not None and current_session() is not None else config.stream
    st.markdown(f'<div class="g-note">{html.escape(_source_badge(run, shown_stream))}. '
                "Nothing is fitted here: every flow is scored by the chosen channel as it arrives.</div>",
                unsafe_allow_html=True)
    _buttons(config)
    st.subheader("Live readings", anchor=False)
    config_now = _applied()
    run_every = config_now.interval if _running() and config_now is not None else None
    st.fragment(_live_panel, run_every=run_every)()
