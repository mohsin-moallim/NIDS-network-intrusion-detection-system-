"""06 Sweep engine: stream flows through one fitted channel and keep live readings, a feed and alerts.

Two flow sources feed a :class:`SimulationSession`:

:class:`ReplaySource`
    Replays the run's HELD-OUT test rows with their true labels, so the live readings measure the channel on flows
    it never saw while fitting, exactly as 03 Measure does. With no options it streams a seeded permutation of the
    test rows (the natural class mix; one full pass reads every test row once). An ``attack_share`` fixes the share
    of attack flows, and a ``mix`` weights the attack types (the detailed labels of the attack rows) against each
    other. Rows are drawn without replacement from each pool until it runs dry, then the pool is reshuffled and
    rows repeat; :attr:`ReplaySource.repeated` counts the repeats.
:class:`SyntheticSource`
    Fresh flows from the packet simulator (:func:`graticule.data.synthetic.generate`), for channels fitted on
    synthetic data. The flows pass the run's bad-value strategy (as at 01 Sample) and only flows of the run's
    classes are emitted.

How shares and mixes are kept. Each draw first decides, slot by slot, whether a flow is normal or an attack (and
which attack type), from a shuffled "bag" of :data:`BAG_SLOTS` slots filled in exact proportion to the weights.
So every 1,000 flows hold the requested share to within one flow, while the order inside the bag stays random.

The session scores each batch with the channel's estimator in one call (nothing is ever fitted here), then updates
running confusion counts, live accuracy (always equal to the accuracy over every emitted flow), live balanced
accuracy (mean recall over the classes seen so far), a per-tick timeline, a bounded feed of the latest flows, a
bounded list of alerts and a columnar log of the last :data:`LOG_CAP` flows (:meth:`SimulationSession.log_frame`).

Alert rule (shared with 04 Probe and 05 Assay, :func:`graticule.models.verdict.alert_flags`): a flow raises an
alert when the channel's verdict is not normal traffic AND its attack probability (binary: P(Attack); multi-class:
1 - P(BENIGN)) reaches the threshold.

Everything is seeded: the same source options, seed, channel and batch sizes give the very same stream and the very
same readings (forests score on one thread, see :func:`graticule.persist.deterministic`).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from graticule.data import synthetic
from graticule.data.clean import apply_nonfinite_strategy
from graticule.models.train import _tidy_proba, score_in_blocks
from graticule.models.verdict import alert_flags
from graticule.persist import deterministic
from graticule.schema import FEATURE_SET, LABEL, NORMAL, is_benign
from graticule.theme import CHANNEL_BY_KEY

if TYPE_CHECKING:
    from graticule.models.train import TrainingRun

SourceKind = Literal["replay", "synthetic"]

#: Slots in one refill of a shuffled bag (shares are exact to one flow in this many).
BAG_SLOTS = 1_000
#: Newest flows kept for the feed, and alerts kept for the alert list (the total is counted regardless).
FEED_SIZE = 200
MAX_ALERTS = 500
#: Most flows kept in the log (:meth:`SimulationSession.log_frame`): the newest ones.
LOG_CAP = 50_000
#: Most ticks kept in the timeline (the newest ones).
TIMELINE_CAP = 10_000
#: Largest batch :meth:`SimulationSession.step` accepts in one call.
MAX_STEP = 100_000
#: Flows the synthetic source generates at a time, and the most it keeps in reserve per category.
SYNTHETIC_BLOCK = 2_048
SYNTHETIC_RESERVE = 20_000
#: Generated blocks in a row without a single flow of the label being drawn before the synthetic source gives up.
SYNTHETIC_MAX_BLOCKS = 20
#: Columns of :meth:`SimulationSession.timeline` and :meth:`SimulationSession.log_frame`.
TIMELINE_COLUMNS: tuple[str, ...] = ("tick", "flows", "normal", "attack_predicted", "alerts", "correct",
                                     "true_attacks")
LOG_COLUMNS: tuple[str, ...] = ("seq", "tick", "row_id", "true_label", "detailed_label", "predicted",
                                "attack_probability", "confidence", "alert", "correct")


def is_normal_class(name: str) -> bool:
    """True for the normal-traffic class under either naming (``BENIGN`` in multi-class, ``Normal`` in binary)."""
    return is_benign(str(name)) or str(name) == NORMAL


def normal_index(classes: Sequence[str]) -> int | None:
    """Position of the normal-traffic class in ``classes`` (None when the target has no normal class)."""
    for i, name in enumerate(classes):
        if is_normal_class(name):
            return i
    return None


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Whole counts summing to ``total`` in proportion to ``weights`` (largest-remainder rounding, ties by order)."""
    share = np.asarray(weights, dtype=np.float64)
    share = share / share.sum()
    exact = share * int(total)
    counts = np.floor(exact).astype(np.int64)
    short = int(total) - int(counts.sum())
    if short > 0:
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:short]] += 1
    return counts


def _clean_weights(mix: Mapping[str, float], allowed: Sequence[str]) -> dict[str, float]:
    """Validated attack-mix weights: known types only, finite and non-negative, at least one above zero."""
    known = set(allowed)
    unknown = [str(name) for name in mix if str(name) not in known]
    if unknown:
        raise ValueError(f"Unknown attack type(s) in the mix: {', '.join(unknown)}. Choose from: "
                         f"{', '.join(allowed) or 'none'}.")
    weights: dict[str, float] = {}
    for name, value in mix.items():
        weight = float(value)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"The weight of {name} must be a finite number of at least 0.")
        if weight > 0:
            weights[str(name)] = weight
    if not weights:
        raise ValueError("Give at least one attack type a weight above 0.")
    return weights


def _check_share(share: float | None) -> float | None:
    """A validated attack share (None stays None)."""
    if share is None:
        return None
    value = float(share)
    if not 0.0 <= value <= 1.0 or not np.isfinite(value):
        raise ValueError("The attack share must lie between 0 and 1.")
    return value


class _Bag:
    """A shuffled bag of category slots, refilled in exact proportion to fixed weights.

    Each refill holds :data:`BAG_SLOTS` slots shared between the categories by largest-remainder rounding and is
    shuffled with the source's generator; :meth:`take` hands the slots out in that order.
    """

    def __init__(self, weights: Sequence[float], rng: np.random.Generator, slots: int = BAG_SLOTS) -> None:
        self._counts = _largest_remainder(np.asarray(weights, dtype=np.float64), int(slots))
        self._rng = rng
        self._slots = np.empty(0, dtype=np.int64)
        self._at = 0

    def take(self, n: int) -> np.ndarray:
        """The categories (indices into the weights) of the next ``n`` slots."""
        out = np.empty(int(n), dtype=np.int64)
        filled = 0
        while filled < n:
            if self._at >= len(self._slots):
                self._slots = self._rng.permutation(np.repeat(np.arange(len(self._counts)), self._counts))
                self._at = 0
            k = min(int(n) - filled, len(self._slots) - self._at)
            out[filled:filled + k] = self._slots[self._at:self._at + k]
            filled += k
            self._at += k
        return out


class _Pool:
    """A seeded stream over a fixed set of row indices: a permutation, then another one once it is used up.

    ``repeats`` counts the indices handed out after the first full pass (rows replayed a second time or more).
    """

    def __init__(self, indices: np.ndarray, rng: np.random.Generator) -> None:
        self._indices = np.asarray(indices, dtype=np.int64)
        self._rng = rng
        self._order = self._rng.permutation(self._indices)
        self._at = 0
        self.passes = 0
        self.repeats = 0

    @property
    def size(self) -> int:
        """Number of distinct indices in the pool."""
        return len(self._indices)

    def take(self, k: int) -> np.ndarray:
        """The next ``k`` indices (reshuffling whenever the current permutation is used up)."""
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        if self.size == 0:
            raise ValueError("This pool has no rows to draw from.")
        out = np.empty(int(k), dtype=np.int64)
        filled = 0
        while filled < k:
            if self._at >= len(self._order):
                self._order = self._rng.permutation(self._indices)
                self._at = 0
                self.passes += 1
            n = min(int(k) - filled, len(self._order) - self._at)
            out[filled:filled + n] = self._order[self._at:self._at + n]
            if self.passes > 0:
                self.repeats += n
            filled += n
            self._at += n
        return out


@dataclass(frozen=True)
class FlowBatch:
    """Flows drawn from a source: features (float32, n x features), true class codes, detailed labels, row ids.

    ``row_ids`` are the run's test-row ids (positions in the prepared sample) for replayed flows, and -1 for
    generated ones.
    """

    X: np.ndarray
    y_true: np.ndarray
    detailed: np.ndarray
    row_ids: np.ndarray

    def __len__(self) -> int:
        """Number of flows in the batch."""
        return len(self.y_true)


def _empty_batch(n_features: int) -> FlowBatch:
    """A batch without flows."""
    return FlowBatch(X=np.empty((0, n_features), dtype=np.float32), y_true=np.empty(0, dtype=np.int64),
                     detailed=np.empty(0, dtype=object), row_ids=np.empty(0, dtype=np.int64))


# --------------------------------------------------------------------------------------------------------------
# Replaying held-out rows
# --------------------------------------------------------------------------------------------------------------
class ReplaySource:
    """Held-out TEST rows of a run, replayed with their true labels (seeded, deterministic).

    ``attack_share`` None keeps the natural test-set proportion; a value between 0 and 1 draws normal and attack rows
    to hold that share. ``mix`` (None for the natural mix) gives relative weights to the attack types present among
    the test rows; the types are the rows' detailed labels (in binary mode "Attack" splits into its detailed types).
    Types left out of the mix, or weighted 0, are not replayed. With neither option the source streams a seeded
    permutation of every test row, so each full pass replays each held-out row exactly once.
    """

    kind: SourceKind = "replay"

    def __init__(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        detailed_labels: np.ndarray,
        row_ids: np.ndarray,
        classes: Sequence[str],
        mode: str,
        *,
        attack_share: float | None = None,
        mix: Mapping[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        X = np.asarray(X_test)
        y = np.asarray(y_test, dtype=np.int64)
        detailed = np.asarray(detailed_labels, dtype=object)
        ids = np.asarray(row_ids, dtype=np.int64)
        if X.ndim != 2 or not len(y) == len(X) == len(detailed) == len(ids):
            raise ValueError("X_test, y_test, detailed_labels and row_ids must describe the same rows.")
        if len(y) == 0:
            raise ValueError("There are no held-out rows to replay.")
        self.classes: tuple[str, ...] = tuple(str(c) for c in classes)
        if y.min() < 0 or y.max() >= len(self.classes):
            raise ValueError("y_test holds codes outside the class list.")
        self.mode = str(mode)
        self.seed = int(seed)
        self._X = X if X.dtype == np.float32 else X.astype(np.float32)
        self._y = y
        self._detailed = detailed
        self._row_ids = ids
        normal_codes = [i for i, name in enumerate(self.classes) if is_normal_class(name)]
        self._normal_rows = np.flatnonzero(np.isin(y, normal_codes))
        self._attack_rows = np.flatnonzero(~np.isin(y, normal_codes))
        names, counts = np.unique(detailed[self._attack_rows].astype(str), return_counts=True)
        order = sorted(range(len(names)), key=lambda i: (-int(counts[i]), str(names[i])))
        self._type_counts: dict[str, int] = {str(names[i]): int(counts[i]) for i in order}
        self.attack_share = _check_share(attack_share)
        self.mix: dict[str, float] | None = (None if mix is None
                                             else _clean_weights(mix, tuple(self._type_counts)))
        self.notes: list[str] = []
        self.reset()

    @classmethod
    def from_run(cls, run: "TrainingRun", *, attack_share: float | None = None,
                 mix: Mapping[str, float] | None = None, seed: int = 0) -> "ReplaySource":
        """A replay of ``run``'s held-out rows (raises ValueError when the run holds none)."""
        data = run.data
        if len(data.y_test) == 0:
            raise ValueError(f"Run {run.run_id} holds no held-out rows to replay.")
        return cls(data.X_test, data.y_test, data.detailed_test_labels, data.test_rows, data.classes,
                   run.request.mode, attack_share=attack_share, mix=mix, seed=seed)

    # -- facts -----------------------------------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of held-out rows available."""
        return len(self._y)

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self._X.shape[1])

    @property
    def attack_types(self) -> tuple[str, ...]:
        """Attack types (detailed labels) among the held-out rows, most rows first."""
        return tuple(self._type_counts)

    @property
    def attack_type_counts(self) -> dict[str, int]:
        """Held-out rows per attack type, most rows first."""
        return dict(self._type_counts)

    @property
    def natural_mix(self) -> dict[str, float]:
        """Each attack type's share of the held-out attack rows (sums to 1; empty without attack rows)."""
        total = sum(self._type_counts.values())
        return {name: count / total for name, count in self._type_counts.items()} if total else {}

    @property
    def natural_share(self) -> float:
        """Share of attack rows among the held-out rows."""
        return len(self._attack_rows) / self.size

    @property
    def effective_share(self) -> float:
        """The attack share this source streams (the natural one unless a share was set or forced)."""
        return self._share

    @property
    def repeated(self) -> int:
        """Flows drawn so far that replay a row already replayed in this stream."""
        return sum(pool.repeats for pool in self._pools())

    def _pools(self) -> list[_Pool]:
        """Every pool in use."""
        pools = [p for p in (self._all, self._normal, self._attacks) if p is not None]
        return pools + list(self._by_type.values())

    def describe(self) -> str:
        """One line naming the stream and its options."""
        share = ("natural attack share" if self.attack_share is None
                 else f"attack share {self.attack_share:.0%}")
        mix = "natural attack mix" if self.mix is None else f"custom mix of {len(self.mix)} attack types"
        return f"Replay of {self.size:,} held-out rows ({share}, {mix}, seed {self.seed})"

    # -- drawing ---------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Rewind the stream: the next draws repeat the very first ones."""
        rng = np.random.default_rng(self.seed)
        self.notes = []
        self._all: _Pool | None = None
        self._normal: _Pool | None = None
        self._attacks: _Pool | None = None
        self._by_type: dict[str, _Pool] = {}
        self._share_bag: _Bag | None = None
        self._type_bag: _Bag | None = None
        self._type_names: list[str] = []
        share = self.natural_share if self.attack_share is None else self.attack_share
        if share < 1.0 and len(self._normal_rows) == 0:
            self.notes.append("No normal rows are held out, so every replayed flow is an attack.")
            share = 1.0
        if share > 0.0 and len(self._attack_rows) == 0:
            self.notes.append("No attack rows are held out, so every replayed flow is normal traffic.")
            share = 0.0
        self._share = float(share)
        if self.attack_share is None and self.mix is None:
            self._all = _Pool(np.arange(self.size, dtype=np.int64), rng)
            return
        self._share_bag = _Bag([1.0 - share, share], rng)
        if len(self._normal_rows):
            self._normal = _Pool(self._normal_rows, rng)
        if self.mix is None:
            if len(self._attack_rows):
                self._attacks = _Pool(self._attack_rows, rng)
            return
        types = self._detailed[self._attack_rows].astype(str)
        self._type_names = [name for name in self._type_counts if name in self.mix]
        for name in self._type_names:
            self._by_type[name] = _Pool(self._attack_rows[types == name], rng)
        self._type_bag = _Bag([self.mix[name] for name in self._type_names], rng)

    def _positions(self, n: int) -> np.ndarray:
        """Row positions (into the held-out rows) of the next ``n`` flows."""
        if self._all is not None:
            return self._all.take(n)
        assert self._share_bag is not None
        slots = self._share_bag.take(n)
        out = np.empty(n, dtype=np.int64)
        normal_at = np.flatnonzero(slots == 0)
        attack_at = np.flatnonzero(slots == 1)
        if len(normal_at):
            assert self._normal is not None
            out[normal_at] = self._normal.take(len(normal_at))
        if len(attack_at):
            if self._type_bag is None:
                assert self._attacks is not None
                out[attack_at] = self._attacks.take(len(attack_at))
            else:
                kinds = self._type_bag.take(len(attack_at))
                for j, name in enumerate(self._type_names):
                    where = attack_at[kinds == j]
                    if len(where):
                        out[where] = self._by_type[name].take(len(where))
        return out

    def draw(self, n: int) -> FlowBatch:
        """The next ``n`` held-out flows with their true class codes, detailed labels and test-row ids."""
        n = int(n)
        if n < 0:
            raise ValueError("Draw a non-negative number of flows.")
        if n == 0:
            return _empty_batch(self.n_features)
        at = self._positions(n)
        return FlowBatch(X=self._X[at], y_true=self._y[at].copy(), detailed=self._detailed[at].astype(str),
                         row_ids=self._row_ids[at].copy())


# --------------------------------------------------------------------------------------------------------------
# Fresh synthetic flows
# --------------------------------------------------------------------------------------------------------------
class SyntheticSource:
    """Fresh flows from the packet simulator, for channels fitted on synthetic data (seeded, deterministic).

    Flows carry the run's feature columns (``feature_names``) after the run's bad-value strategy, and true class
    codes into ``classes``: in binary mode BENIGN is the normal class and every attack type is "Attack"; in
    multi-class mode only flows of the run's classes are emitted. ``attack_share`` is the share of attack flows and
    ``mix`` (None: the generator's own mix) weights the attack types. Flows are generated in blocks of
    :data:`SYNTHETIC_BLOCK` and never repeat.
    """

    kind: SourceKind = "synthetic"

    def __init__(
        self,
        feature_names: Sequence[str],
        classes: Sequence[str],
        mode: str,
        *,
        attack_share: float = 0.35,
        mix: Mapping[str, float] | None = None,
        seed: int = 0,
        nonfinite_strategy: str = "drop",
        blur: float = 0.04,
    ) -> None:
        self.feature_names: tuple[str, ...] = tuple(str(f) for f in feature_names)
        unknown = [f for f in self.feature_names if f not in FEATURE_SET]
        if unknown or not self.feature_names:
            raise ValueError(f"The synthetic generator does not produce these columns: {', '.join(unknown)}.")
        self.classes: tuple[str, ...] = tuple(str(c) for c in classes)
        if len(self.classes) < 2:
            raise ValueError("A stream needs at least two classes.")
        self.mode = str(mode)
        self.seed = int(seed)
        self.nonfinite_strategy = str(nonfinite_strategy)
        self.blur = float(blur)
        self._normal_code = normal_index(self.classes)
        generated = list(synthetic.SYNTHETIC_CLASSES[1:])
        if self.mode == "binary":
            attack_codes = [i for i, name in enumerate(self.classes) if not is_normal_class(name)]
            if len(attack_codes) != 1:
                raise ValueError("A binary run needs exactly one attack class.")
            self._codes = {name: attack_codes[0] for name in generated}
            types = generated
        else:
            self._codes = {name: self.classes.index(name) for name in generated if name in self.classes}
            types = [name for name in generated if name in self.classes]
        if self._normal_code is not None:
            self._codes[synthetic.SYNTHETIC_CLASSES[0]] = self._normal_code
        self._types: tuple[str, ...] = tuple(types)
        self.attack_share = _check_share(attack_share)
        if self.attack_share is None:
            raise ValueError("A synthetic stream needs an attack share.")
        self.mix: dict[str, float] | None = None if mix is None else _clean_weights(mix, self._types)
        self.notes: list[str] = []
        self.reset()

    @classmethod
    def from_run(cls, run: "TrainingRun", *, attack_share: float | None = None,
                 mix: Mapping[str, float] | None = None, seed: int = 0) -> "SyntheticSource":
        """A synthetic stream for ``run`` (share None: the share the run's own sample was generated with)."""
        share = run.data_request.synthetic_attack_share if attack_share is None else attack_share
        return cls(run.data.feature_names, run.data.classes, run.request.mode, attack_share=share, mix=mix,
                   seed=seed, nonfinite_strategy=run.data_request.nonfinite_strategy)

    # -- facts -----------------------------------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return len(self.feature_names)

    @property
    def attack_types(self) -> tuple[str, ...]:
        """Attack types this stream can emit (generator order)."""
        return self._types

    @property
    def natural_mix(self) -> dict[str, float]:
        """The generator's own split between the attack types this stream emits (sums to 1)."""
        weights = {name: float(synthetic.ATTACK_MIX.get(name, 0.0)) for name in self._types}
        total = sum(weights.values()) or 1.0
        return {name: value / total for name, value in weights.items()}

    @property
    def effective_share(self) -> float:
        """The attack share this source streams."""
        return self._share

    @property
    def repeated(self) -> int:
        """Always 0: generated flows never repeat."""
        return 0

    @property
    def blocks_generated(self) -> int:
        """Blocks of flows generated since the last reset."""
        return self._blocks

    def describe(self) -> str:
        """One line naming the stream and its options."""
        mix = "the generator's attack mix" if self.mix is None else f"a custom mix of {len(self.mix)} attack types"
        return (f"Synthetic stream ({self._share:.0%} attacks, {mix}, bad values: {self.nonfinite_strategy}, "
                f"seed {self.seed})")

    # -- drawing ---------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Rewind the stream: the next draws repeat the very first ones."""
        rng = np.random.default_rng(self.seed)
        self.notes = []
        share = float(self.attack_share or 0.0)
        if share < 1.0 and self._normal_code is None:
            self.notes.append("The run has no normal class, so every generated flow is an attack.")
            share = 1.0
        if share > 0.0 and not self._types:
            self.notes.append("The run has no attack class the generator makes, so every flow is normal traffic.")
            share = 0.0
        self._share = share
        self._share_bag = _Bag([1.0 - share, share], rng)
        weights = self.mix if self.mix is not None else self.natural_mix
        self._type_names = [name for name in self._types if weights.get(name, 0.0) > 0]
        self._type_bag = _Bag([weights[name] for name in self._type_names], rng) if self._type_names else None
        self._queues: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._queued: dict[str, int] = {}
        self._blocks = 0

    def _block_seed(self) -> int:
        """Seed of the next generated block (depends only on the source seed and the block number)."""
        state = np.random.SeedSequence([self.seed & 0xFFFFFFFF, self._blocks, 0x53EE9]).generate_state(1)
        return int(state[0])

    def _generate(self, attacks: bool, needed: str) -> int:
        """Generate one block of normal flows (or of attack flows) and queue them by label.

        Labels the run does not use are dropped, and so are flows of a label whose reserve is already full (except
        ``needed``, the label being drawn). Returns how many ``needed`` flows were queued.
        """
        frame = synthetic.generate(SYNTHETIC_BLOCK, seed=self._block_seed(), attack_share=1.0 if attacks else 0.0,
                                   blur=self.blur)
        self._blocks += 1
        frame, _ = apply_nonfinite_strategy(frame, self.nonfinite_strategy)  # type: ignore[arg-type]
        labels = frame[LABEL].to_numpy(dtype=object).astype(str)
        X = np.empty((len(frame), self.n_features), dtype=np.float32)
        for j, name in enumerate(self.feature_names):
            X[:, j] = frame[name].to_numpy(dtype=np.float32, na_value=np.nan)
        added = 0
        for name in np.unique(labels).tolist():
            if name not in self._codes:
                continue
            if name != needed and self._queued.get(name, 0) >= SYNTHETIC_RESERVE:
                continue
            rows = labels == name
            self._queues.setdefault(name, []).append((X[rows], labels[rows]))
            self._queued[name] = self._queued.get(name, 0) + int(rows.sum())
            if name == needed:
                added = int(rows.sum())
        return added

    def _take(self, name: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The next ``k`` queued flows labelled ``name``, generating blocks as needed."""
        attacks = name != synthetic.SYNTHETIC_CLASSES[0]
        barren = 0
        while self._queued.get(name, 0) < k:
            if self._generate(attacks, needed=name):
                barren = 0
            else:
                barren += 1
                if barren >= SYNTHETIC_MAX_BLOCKS:
                    raise RuntimeError(f"The generator made no {name} flows in {barren} blocks in a row.")
        parts_X: list[np.ndarray] = []
        parts_label: list[np.ndarray] = []
        need = k
        queue = self._queues[name]
        while need > 0:
            X, labels = queue[0]
            if len(labels) <= need:
                parts_X.append(X)
                parts_label.append(labels)
                queue.pop(0)
                need -= len(labels)
            else:
                parts_X.append(X[:need])
                parts_label.append(labels[:need])
                queue[0] = (X[need:], labels[need:])
                need = 0
        self._queued[name] -= k
        return np.concatenate(parts_X), np.concatenate(parts_label)

    def draw(self, n: int) -> FlowBatch:
        """The next ``n`` generated flows with their true class codes and labels (row ids are -1)."""
        n = int(n)
        if n < 0:
            raise ValueError("Draw a non-negative number of flows.")
        if n == 0:
            return _empty_batch(self.n_features)
        slots = self._share_bag.take(n)
        names = np.empty(n, dtype=object)
        names[slots == 0] = synthetic.SYNTHETIC_CLASSES[0]
        attack_at = np.flatnonzero(slots == 1)
        if len(attack_at):
            assert self._type_bag is not None
            kinds = self._type_bag.take(len(attack_at))
            names[attack_at] = np.asarray(self._type_names, dtype=object)[kinds]
        X = np.empty((n, self.n_features), dtype=np.float32)
        detailed = np.empty(n, dtype=object)
        for name in dict.fromkeys(names.tolist()):
            where = np.flatnonzero(names == name)
            X[where], detailed[where] = self._take(str(name), len(where))
        codes = np.array([self._codes[str(name)] for name in detailed], dtype=np.int64)
        return FlowBatch(X=X, y_true=codes, detailed=detailed.astype(str), row_ids=np.full(n, -1, dtype=np.int64))


Source = ReplaySource | SyntheticSource


# --------------------------------------------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FlowEvent:
    """One scored flow of the stream.

    ``true_label`` and ``predicted`` are class names of the run's target (in binary mode "Normal" or "Attack");
    ``detail`` is the flow's detailed label (e.g. "DoS Hulk"). ``row_id`` is the test-row id of a replayed flow and
    None for a generated one. ``confidence`` is the probability of the predicted class.
    """

    seq: int
    tick: int
    row_id: int | None
    true_label: str
    predicted: str
    attack_probability: float
    confidence: float
    alert: bool
    correct: bool
    detail: str = ""


@dataclass(frozen=True)
class LiveStats:
    """Running readings of a session (a snapshot; the arrays are copies).

    ``confusion`` is K x K (rows: true class, columns: verdict). ``live_accuracy`` and ``live_balanced_accuracy``
    are NaN before the first flow; balanced accuracy is the mean recall over the classes seen so far.
    ``repeated`` counts replayed flows that repeat a row (0 for generated flows).
    """

    emitted: int
    correct: int
    live_accuracy: float
    live_balanced_accuracy: float
    confusion: np.ndarray
    per_class_seen: dict[str, int]
    alerts_total: int
    ticks: int
    classes: tuple[str, ...] = ()
    attacks_seen: int = 0
    predicted_attacks: int = 0
    repeated: int = 0
    scoring_seconds: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def classes_seen(self) -> int:
        """Number of classes with at least one emitted flow."""
        return sum(1 for count in self.per_class_seen.values() if count > 0)

    @property
    def attack_share_seen(self) -> float:
        """Share of true attacks among the emitted flows (NaN before the first flow)."""
        return self.attacks_seen / self.emitted if self.emitted else float("nan")


class SimulationSession:
    """Live replay of a flow source through one fitted channel of a run.

    ``alert_threshold`` is the attack probability from which a flow whose verdict is an attack raises an alert.
    The feed keeps the newest ``feed_size`` flows and the alert list the newest ``max_alerts`` alerts (the total is
    counted regardless); the log keeps the newest :data:`LOG_CAP` flows. Nothing is fitted: each step scores the
    drawn flows with the channel's estimator in one call.
    """

    def __init__(self, run: "TrainingRun", channel: str, source: Source, *, alert_threshold: float,
                 feed_size: int = FEED_SIZE, max_alerts: int = MAX_ALERTS) -> None:
        if channel not in run.channels or not run.channels[channel].ok or run.channels[channel].estimator is None:
            raise ValueError(f"Channel {channel!r} has no fitted model in run {run.run_id}.")
        if tuple(source.classes) != tuple(run.classes):
            raise ValueError("The source's classes differ from the run's classes.")
        if source.n_features != len(run.data.feature_names):
            raise ValueError("The source's feature columns differ from the run's.")
        self.run = run
        self.channel = str(channel)
        self.source = source
        self.classes: tuple[str, ...] = tuple(run.classes)
        self.normal_index = normal_index(self.classes)
        self._estimator = run.channels[channel].estimator
        self._feed_size = max(int(feed_size), 1)
        self._max_alerts = max(int(max_alerts), 1)
        self._lock = threading.RLock()
        self.alert_threshold = float(alert_threshold)
        self.reset()

    # -- settings --------------------------------------------------------------------------------------------
    @property
    def alert_threshold(self) -> float:
        """Attack probability from which an attack verdict raises an alert."""
        return self._threshold

    @alert_threshold.setter
    def alert_threshold(self, value: float) -> None:
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0 or not np.isfinite(threshold):
            raise ValueError("The alert threshold must lie between 0 and 1.")
        self._threshold = threshold

    @property
    def run_id(self) -> str:
        """Id of the run whose channel is streamed."""
        return self.run.run_id

    @property
    def channel_label(self) -> str:
        """Badge and name of the streamed channel, e.g. ``"CH2 XGBoost"``."""
        style = CHANNEL_BY_KEY.get(self.channel)
        return style.label if style is not None else self.channel

    # -- stepping --------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Clear every reading and rewind the source (the next steps replay the same stream again)."""
        with self._lock:
            self.source.reset()
            k = len(self.classes)
            self._confusion = np.zeros((k, k), dtype=np.int64)
            self._emitted = 0
            self._correct = 0
            self._alerts_total = 0
            self._ticks = 0
            self._predicted_attacks = 0
            self._attacks_seen = 0
            self._scoring_seconds = 0.0
            self._timeline: deque[tuple[int, ...]] = deque(maxlen=TIMELINE_CAP)
            self._feed: deque[FlowEvent] = deque(maxlen=self._feed_size)
            self._alerts: deque[FlowEvent] = deque(maxlen=self._max_alerts)
            self._log: deque[dict[str, np.ndarray]] = deque()
            self._logged = 0

    def _score(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Class probabilities (n x K, code order) of the channel on ``X``, and the seconds the call took."""
        started = time.perf_counter()
        with deterministic(self._estimator):
            raw, _ = score_in_blocks(self._estimator, X)
        proba = _tidy_proba(raw, getattr(self._estimator, "classes_", None), len(self.classes))
        return proba, time.perf_counter() - started

    def step(self, n: int) -> list[FlowEvent]:
        """Draw ``n`` flows, score them, update every reading and return their events (oldest first).

        One call is one tick of the timeline. ``n`` is between 1 and :data:`MAX_STEP`.
        """
        n = int(n)
        if not 1 <= n <= MAX_STEP:
            raise ValueError(f"A step takes between 1 and {MAX_STEP:,} flows.")
        with self._lock:
            batch = self.source.draw(n)
            proba, seconds = self._score(batch.X)
            predicted = proba.argmax(axis=1).astype(np.int64)
            confidence = proba[np.arange(n), predicted].astype(np.float64)
            if self.normal_index is None:
                attack_p = np.ones(n, dtype=np.float64)
                verdict_attack = np.ones(n, dtype=bool)
                true_attack = np.ones(n, dtype=bool)
            else:
                if len(self.classes) == 2:  # binary: the attack class's own probability
                    attack_p = proba[:, 1 - self.normal_index].astype(np.float64)
                else:  # multi-class: everything that is not normal traffic
                    attack_p = np.clip(1.0 - proba[:, self.normal_index].astype(np.float64), 0.0, 1.0)
                verdict_attack = predicted != self.normal_index
                true_attack = batch.y_true != self.normal_index
            alert = alert_flags(attack_p, predicted, self.normal_index, self._threshold)
            correct = predicted == batch.y_true
            np.add.at(self._confusion, (batch.y_true, predicted), 1)
            first = self._emitted + 1
            self._emitted += n
            self._correct += int(correct.sum())
            self._alerts_total += int(alert.sum())
            self._predicted_attacks += int(verdict_attack.sum())
            self._attacks_seen += int(true_attack.sum())
            self._scoring_seconds += seconds
            self._ticks += 1
            tick = self._ticks
            self._timeline.append((tick, n, int((~verdict_attack).sum()), int(verdict_attack.sum()),
                                   int(alert.sum()), int(correct.sum()), int(true_attack.sum())))
            names = np.asarray(self.classes, dtype=object)
            seq = np.arange(first, first + n, dtype=np.int64)
            true_names = names[batch.y_true]
            predicted_names = names[predicted]
            self._remember({
                "seq": seq, "tick": np.full(n, tick, dtype=np.int64), "row_id": batch.row_ids.astype(np.int64),
                "true_label": true_names, "detailed_label": batch.detailed.astype(object),
                "predicted": predicted_names, "attack_probability": attack_p, "confidence": confidence,
                "alert": alert, "correct": correct,
            })
            events = [
                FlowEvent(seq=int(seq[i]), tick=tick, row_id=int(batch.row_ids[i]) if batch.row_ids[i] >= 0 else None,
                          true_label=str(true_names[i]), predicted=str(predicted_names[i]),
                          attack_probability=float(attack_p[i]), confidence=float(confidence[i]),
                          alert=bool(alert[i]), correct=bool(correct[i]), detail=str(batch.detailed[i]))
                for i in range(n)
            ]
            self._feed.extend(events[-self._feed_size:])
            self._alerts.extend(e for e in events if e.alert)
            return events

    def _remember(self, part: dict[str, np.ndarray]) -> None:
        """Add one step's flows to the log and drop the oldest beyond :data:`LOG_CAP`."""
        self._log.append(part)
        self._logged += len(part["seq"])
        while self._logged > LOG_CAP:
            oldest = self._log[0]
            extra = self._logged - LOG_CAP
            size = len(oldest["seq"])
            if size <= extra:
                self._log.popleft()
                self._logged -= size
            else:
                self._log[0] = {name: values[extra:] for name, values in oldest.items()}
                self._logged -= extra

    # -- readings --------------------------------------------------------------------------------------------
    @property
    def stats(self) -> LiveStats:
        """The running readings (a snapshot)."""
        with self._lock:
            confusion = self._confusion.copy()
            seen = confusion.sum(axis=1)
            emitted = self._emitted
            accuracy = self._correct / emitted if emitted else float("nan")
            present = seen > 0
            balanced = (float(np.mean(np.diag(confusion)[present] / seen[present])) if present.any()
                        else float("nan"))
            notes = tuple(getattr(self.source, "notes", ()) or ())
            return LiveStats(
                emitted=emitted, correct=self._correct, live_accuracy=float(accuracy),
                live_balanced_accuracy=balanced, confusion=confusion,
                per_class_seen={name: int(count) for name, count in zip(self.classes, seen)},
                alerts_total=self._alerts_total, ticks=self._ticks, classes=self.classes,
                attacks_seen=self._attacks_seen, predicted_attacks=self._predicted_attacks,
                repeated=int(self.source.repeated), scoring_seconds=float(self._scoring_seconds), notes=notes,
            )

    @property
    def timeline(self) -> pd.DataFrame:
        """One row per tick: flows, verdicts read as normal and as attack, alerts, correct verdicts, true attacks."""
        with self._lock:
            rows = list(self._timeline)
        if not rows:
            return pd.DataFrame({name: pd.Series([], dtype="int64") for name in TIMELINE_COLUMNS})
        return pd.DataFrame(np.asarray(rows, dtype=np.int64), columns=list(TIMELINE_COLUMNS))

    @property
    def feed(self) -> list[FlowEvent]:
        """The newest flows (at most ``feed_size``), newest first."""
        with self._lock:
            return list(reversed(self._feed))

    @property
    def alerts(self) -> list[FlowEvent]:
        """The newest alerts (at most ``max_alerts``), newest first."""
        with self._lock:
            return list(reversed(self._alerts))

    def log_frame(self) -> pd.DataFrame:
        """Every emitted flow still in the log (the newest :data:`LOG_CAP`), oldest first, as a table.

        Columns: seq, tick, row_id (test-row id; missing for generated flows), true_label, detailed_label,
        predicted, attack_probability, confidence, alert, correct. No feature values are included.
        """
        with self._lock:
            parts = list(self._log)
        if not parts:
            frame = pd.DataFrame({name: [] for name in LOG_COLUMNS})
        else:
            frame = pd.DataFrame({name: np.concatenate([p[name] for p in parts]) for name in LOG_COLUMNS})
        ids = frame["row_id"].to_numpy(dtype=np.int64) if len(frame) else np.empty(0, dtype=np.int64)
        frame["row_id"] = pd.arrays.IntegerArray(np.where(ids >= 0, ids, 0), ids < 0)
        for name in ("true_label", "detailed_label", "predicted"):
            frame[name] = frame[name].astype("str")
        for name in ("alert", "correct"):
            frame[name] = frame[name].astype(bool)
        for name in ("seq", "tick"):
            frame[name] = frame[name].astype("int64")
        for name in ("attack_probability", "confidence"):
            frame[name] = frame[name].astype("float64")
        return frame

    def summary(self) -> dict[str, Any]:
        """Plain values describing the session and its readings (for reports and exports)."""
        stats = self.stats
        mix = getattr(self.source, "mix", None)
        return {
            "run_id": self.run_id,
            "channel": self.channel,
            "channel_label": self.channel_label,
            "source_kind": self.source.kind,
            "source": self.source.describe(),
            "seed": int(self.source.seed),
            "attack_share": float(self.source.effective_share),
            "attack_share_set": getattr(self.source, "attack_share", None),
            "mix": dict(mix) if mix else None,
            "alert_threshold": float(self._threshold),
            "ticks": stats.ticks,
            "flows": stats.emitted,
            "correct": stats.correct,
            "live_accuracy": stats.live_accuracy,
            "live_balanced_accuracy": stats.live_balanced_accuracy,
            "alerts": stats.alerts_total,
            "attacks_seen": stats.attacks_seen,
            "predicted_attacks": stats.predicted_attacks,
            "repeated": stats.repeated,
            "per_class_seen": dict(stats.per_class_seen),
            "classes": list(self.classes),
            "confusion": stats.confusion.tolist(),
            "notes": list(stats.notes),
        }


def stream_kinds(run: "TrainingRun") -> list[SourceKind]:
    """The flow sources a run can stream, preferred first.

    A CIC-IDS2017 run replays its held-out rows (and has none to offer without them). A synthetic run streams fresh
    generator flows, and can also replay its held-out rows when they are in memory.
    """
    has_rows = len(run.data.y_test) > 0
    if run.data_request.source == "synthetic":
        return ["synthetic", "replay"] if has_rows else ["synthetic"]
    return ["replay"] if has_rows else []


def make_source(run: "TrainingRun", kind: SourceKind, *, attack_share: float | None = None,
                mix: Mapping[str, float] | None = None, seed: int = 0) -> Source:
    """The flow source ``kind`` for ``run`` (see :func:`stream_kinds`); raises ValueError when it cannot stream.

    Real-data runs only ever replay real held-out rows: asking for a synthetic stream of a CIC-IDS2017 run fails.
    """
    if kind not in stream_kinds(run):
        if kind == "synthetic":
            raise ValueError("Only runs fitted on synthetic data can stream generated flows.")
        raise ValueError(f"Run {run.run_id} holds no held-out rows to replay.")
    if kind == "synthetic":
        return SyntheticSource.from_run(run, attack_share=attack_share, mix=mix, seed=seed)
    return ReplaySource.from_run(run, attack_share=attack_share, mix=mix, seed=seed)


__all__ = [
    "BAG_SLOTS", "FEED_SIZE", "LOG_CAP", "LOG_COLUMNS", "MAX_ALERTS", "TIMELINE_COLUMNS", "FlowBatch", "FlowEvent",
    "LiveStats", "ReplaySource", "SimulationSession", "Source", "SourceKind", "SyntheticSource", "is_normal_class",
    "make_source", "normal_index", "stream_kinds",
]
