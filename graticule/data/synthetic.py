"""Synthetic network flows: a small packet simulator feeding one shared flow meter.

Graticule has to work before anyone points it at the CIC-IDS2017 files, so this module manufactures labelled flow
records with exactly the same 77 feature columns. It works in two stages, the way a capture pipeline does.

1. Packet simulation. Every flow is given a traffic profile (an interactive web session, a DNS lookup, a port
   sweep, ...). The profile decides how many packets the conversation has, which side sends each one, how many
   payload bytes it carries, which TCP flags it sets and how long its sender waited since the previous packet.
   Profiles are vectorised numpy code working on all flows of that profile at once; no Python loop runs per
   packet or per flow.
2. Flow metering. All packets are gathered into one table sorted by (flow, time) and a single meter
   (:func:`measure`) reduces it to per-flow statistics. Because every column is computed from the same packets,
   derived columns always agree: a byte rate is the byte total over the duration, the average forward segment
   size is the forward mean length, the subflow counts are the direction totals, and so on.

Units follow the recorded files: times in microseconds, rates per second, lengths in payload bytes. As in the
recorded files, every flow has at least two packets (a lone packet that gets no answer is retried), and a flow
whose packets share one timestamp has zero duration, so its two whole-flow rates (Flow Bytes/s, Flow Packets/s) are
+inf (or NaN when it carried no bytes at all) while the per-direction packet rates read 0. Such flows are rare here,
as they are in the recorded files (about one flow in a thousand): a small share of floods fire a whole burst within
one clock tick. The cleaning stage has to deal with them exactly as it does for real traffic.

Class mix. ``attack_share`` of the flows are attacks, split between the five attack profiles by
:data:`ATTACK_MIX`; normal flows are split between four everyday behaviours by :data:`BENIGN_MIX`.

Realism knobs. ``blur`` is the share of attack flows that copy a normal behaviour (drawn by :data:`COVER_MIX`):
their timing, sizes and flags look ordinary and only their destination port and their label give them away.
About ``blur / 4`` of normal flows become refused connections (one SYN answered by a reset), which look much like
a port sweep. Together they keep the classes clearly separable but never perfectly so.

Memory. Flows are simulated in blocks of at most :data:`BLOCK_FLOWS`, each with its own random stream spawned from
the seed, and every block is metered before the next one starts. Peak memory therefore stays near that of one
block however many flows are asked for, and the output still depends only on the arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from graticule import schema

SYNTHETIC_CLASSES: tuple[str, ...] = ("BENIGN", "Flood", "Slow Drip", "Sweep", "Credential Guess", "Web Injection")

#: Share of the attack flows given to each attack profile (sums to 1).
ATTACK_MIX: dict[str, float] = {
    "Flood": 0.30,
    "Slow Drip": 0.10,
    "Sweep": 0.25,
    "Credential Guess": 0.20,
    "Web Injection": 0.15,
}
#: Share of the normal flows given to each everyday behaviour (before ``blur`` turns some into refusals).
BENIGN_MIX: dict[str, float] = {"web": 0.55, "dns": 0.20, "bulk": 0.12, "keepalive": 0.13}
#: Normal behaviours a disguised attack flow may copy, with their odds.
COVER_MIX: dict[str, float] = {"web": 0.60, "bulk": 0.15, "keepalive": 0.25}

#: Longest conversation the simulator emits, in packets.
MAX_PACKETS = 300
#: Flows simulated and metered together; part of the definition of the output for a given seed.
BLOCK_FLOWS = 50_000
#: Share of floods whose whole burst falls within one clock tick (zero duration, as in a few recorded flows).
SAME_TICK_FLOODS = 0.01
#: A silence longer than this (microseconds) splits a flow into busy periods separated by idle gaps.
IDLE_GAP_US = 5_000_000.0

# TCP flag bits used in the ``flags`` column of a PacketTable.
FIN = 1
SYN = 2
RST = 4
PSH = 8
ACK = 16
URG = 32
ECE = 64
CWE = 128

_CLIENT_WINDOWS = np.array([8192, 14600, 26883, 29200, 64240, 65535], dtype=np.int64)
_SERVER_WINDOWS = np.array([5792, 14480, 28960, 43440, 65160], dtype=np.int64)
_COMMON_PORTS = np.array(
    [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 993, 995, 1433, 3306, 3389, 5900, 8080], dtype=np.int64
)


# ----------------------------------------------------------------------------------------------------------------
# The flow meter
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PacketTable:
    """Packets of many flows, one entry per packet, sorted by flow and then by time.

    Packet-level arrays (all the same length):
        flow: flow index of each packet, ``0 .. n_flows - 1``, non-decreasing; every flow has at least one packet.
        time_us: capture time in microseconds, non-decreasing within a flow.
        backward: True for packets sent by the responder, False for packets sent by the initiator.
        payload: payload bytes carried by the packet.
        header: header bytes of the packet (e.g. 20 or 32 for TCP, 8 for UDP).
        flags: TCP flag bitmask built from this module's FIN/SYN/RST/PSH/ACK/URG/ECE/CWE constants (0 for UDP).
        window: TCP window advertised by the packet.

    Flow-level arrays (length ``n_flows``):
        port: destination port of each flow.
        udp: True for flows carried over UDP.
    """

    flow: np.ndarray
    time_us: np.ndarray
    backward: np.ndarray
    payload: np.ndarray
    header: np.ndarray
    flags: np.ndarray
    window: np.ndarray
    port: np.ndarray
    udp: np.ndarray

    @property
    def n_flows(self) -> int:
        """Number of flows described by the table."""
        return int(np.asarray(self.port).size)


@dataclass(frozen=True)
class _Stats:
    """Per-group summary of one quantity: sum, count, mean, population std, max and min (all zero when empty)."""

    total: np.ndarray
    count: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    max: np.ndarray
    min: np.ndarray


def _stats(values: np.ndarray, groups: np.ndarray, n_groups: int) -> _Stats:
    """Summarise ``values`` per group. ``groups`` must be sorted ascending (groups are contiguous runs)."""
    values = np.asarray(values, dtype=np.float64)
    count = np.bincount(groups, minlength=n_groups).astype(np.float64)
    total = np.bincount(groups, weights=values, minlength=n_groups)
    has = count > 0
    mean = np.divide(total, count, out=np.zeros(n_groups), where=has)
    deviation = values - mean[groups]
    spread = np.bincount(groups, weights=deviation * deviation, minlength=n_groups)
    std = np.sqrt(np.divide(spread, count, out=np.zeros(n_groups), where=has))
    high = np.zeros(n_groups)
    low = np.zeros(n_groups)
    if values.size:
        starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
        present = groups[starts]
        high[present] = np.maximum.reduceat(values, starts)
        low[present] = np.minimum.reduceat(values, starts)
    return _Stats(total, count, mean, std, high, low)


def _rate(amount: np.ndarray, duration_us: np.ndarray, empty_value: float) -> np.ndarray:
    """``amount`` per second of flow time. Zero duration gives +inf for a positive amount, else ``empty_value``."""
    out = np.empty(amount.shape, dtype=np.float64)
    timed = duration_us > 0
    out[timed] = amount[timed] / (duration_us[timed] / 1e6)
    instant = ~timed
    out[instant] = np.where(amount[instant] > 0, np.inf, empty_value)
    return out


def _direction_rate(amount: np.ndarray, duration_us: np.ndarray) -> np.ndarray:
    """Packets per second in one direction; 0 for a zero-duration flow, matching the recorded files."""
    out = np.zeros(amount.shape, dtype=np.float64)
    timed = duration_us > 0
    out[timed] = amount[timed] / (duration_us[timed] / 1e6)
    return out


def _check_table(packets: PacketTable) -> None:
    """Reject tables the meter cannot reduce correctly."""
    n = packets.n_flows
    size = np.asarray(packets.flow).size
    for name in ("time_us", "backward", "payload", "header", "flags", "window"):
        if np.asarray(getattr(packets, name)).size != size:
            raise ValueError(f"PacketTable.{name} must have one entry per packet ({size}).")
    if np.asarray(packets.udp).size != n:
        raise ValueError("PacketTable.udp must have one entry per flow.")
    flow = np.asarray(packets.flow)
    if size == 0:
        if n:
            raise ValueError("Every flow needs at least one packet.")
        return
    if flow.min() < 0 or flow.max() >= n:
        raise ValueError("PacketTable.flow values must lie in 0 .. n_flows - 1.")
    if np.any(np.diff(flow) < 0):
        raise ValueError("PacketTable must be sorted by flow.")
    if np.any(np.bincount(flow, minlength=n) == 0):
        raise ValueError("Every flow needs at least one packet.")
    same_flow = flow[1:] == flow[:-1]
    if np.any(np.diff(np.asarray(packets.time_us, dtype=np.float64))[same_flow] < 0):
        raise ValueError("Packet times must not decrease within a flow.")


def measure(packets: PacketTable) -> pd.DataFrame:
    """Reduce a packet table to one row of the 77 flow features per flow (float32, ``schema.FEATURES`` order).

    Column semantics, in the meter's own terms:

    * Forward means sent by the initiator, backward by the responder. Lengths are payload bytes.
    * Flow Duration is the time from the first to the last packet. Rates divide by the duration in seconds. For a
      zero-duration flow the whole-flow rates are +inf when it carried something and NaN (bytes) otherwise, and the
      per-direction packet rates are 0, which is what the recorded files hold for such flows.
    * Standard deviations are population deviations; Packet Length Variance is the square of Packet Length Std.
    * Inter-arrival times (IAT) are gaps between consecutive packets of the whole flow (Flow IAT) or of one
      direction (Fwd/Bwd IAT, where Total is the sum of the gaps). Fewer than two packets gives zeros.
    * Per-direction PSH/URG columns and the flag counts are numbers of packets carrying the flag.
    * Header Length columns sum per-packet header sizes; min_seg_size_forward is the smallest forward header.
    * Init_Win_bytes_* is the window of the first packet in that direction, -1 for an empty direction or UDP.
    * Active/Idle: gaps longer than :data:`IDLE_GAP_US` are idle periods and split the flow into busy periods,
      whose lengths (first to last packet of the period) are the active times. All zero when there is no such gap.
    * The six bulk columns are always 0; Subflow columns repeat the direction totals.
    """
    _check_table(packets)
    n = packets.n_flows
    if n == 0:
        return pd.DataFrame({name: np.zeros(0, dtype=np.float32) for name in schema.FEATURES})

    flow = np.asarray(packets.flow, dtype=np.int64)
    t = np.asarray(packets.time_us, dtype=np.float64)
    backward = np.asarray(packets.backward, dtype=bool)
    payload = np.asarray(packets.payload, dtype=np.float64)
    header = np.asarray(packets.header, dtype=np.float64)
    flags = np.asarray(packets.flags, dtype=np.int64)
    window = np.asarray(packets.window, dtype=np.float64)
    udp = np.asarray(packets.udp, dtype=bool)
    n_packets = flow.size

    counts = np.bincount(flow, minlength=n)
    starts = np.cumsum(counts) - counts
    ends = starts + counts - 1
    first = np.zeros(n_packets, dtype=bool)
    first[starts] = True
    duration = t[ends] - t[starts]

    # Group 2*i holds the forward packets of flow i, group 2*i + 1 its backward packets.
    group = flow * 2 + backward
    by_dir = np.argsort(group, kind="stable")
    g_sorted = group[by_dir]

    overall = _stats(payload, flow, n)
    direction = _stats(payload[by_dir], g_sorted, 2 * n)
    fwd_len = _Stats(*(getattr(direction, f)[0::2] for f in ("total", "count", "mean", "std", "max", "min")))
    bwd_len = _Stats(*(getattr(direction, f)[1::2] for f in ("total", "count", "mean", "std", "max", "min")))
    fwd_pkts, bwd_pkts = fwd_len.count, bwd_len.count
    total_bytes = overall.total
    total_pkts = overall.count

    gaps = np.zeros(n_packets)
    gaps[1:] = np.diff(t)
    inner = ~first
    flow_iat = _stats(gaps[inner], flow[inner], n)

    t_dir = t[by_dir]
    same_dir = g_sorted[1:] == g_sorted[:-1]
    dir_iat = _stats((t_dir[1:] - t_dir[:-1])[same_dir], g_sorted[1:][same_dir], 2 * n)

    def flag_count(bit: int) -> np.ndarray:
        return np.bincount(flow, weights=(flags & bit) != 0, minlength=n)

    def flag_by_dir(bit: int) -> np.ndarray:
        return np.bincount(group, weights=(flags & bit) != 0, minlength=2 * n)

    psh_dir = flag_by_dir(PSH)
    urg_dir = flag_by_dir(URG)
    header_dir = _stats(header[by_dir], g_sorted, 2 * n)

    group_starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    init_window = np.full(2 * n, -1.0)
    init_window[g_sorted[group_starts]] = window[by_dir][group_starts]
    init_window = init_window.reshape(n, 2)
    init_window[udp] = -1.0

    idle_packet = inner & (gaps > IDLE_GAP_US)
    has_idle = np.bincount(flow[idle_packet], minlength=n) > 0
    period_starts = np.flatnonzero(first | idle_packet)
    period_ends = np.r_[period_starts[1:] - 1, n_packets - 1]
    period_flow = flow[period_starts]
    period_len = t[period_ends] - t[period_starts]
    keep = has_idle[period_flow]
    active = _stats(period_len[keep], period_flow[keep], n)
    idle = _stats(gaps[idle_packet], flow[idle_packet], n)

    down_up = np.floor(np.divide(bwd_pkts, fwd_pkts, out=np.zeros(n), where=fwd_pkts > 0))
    zeros = np.zeros(n)
    columns: dict[str, np.ndarray] = {
        "Destination Port": np.asarray(packets.port, dtype=np.float64),
        "Flow Duration": duration,
        "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_len.total,
        "Total Length of Bwd Packets": bwd_len.total,
        "Fwd Packet Length Max": fwd_len.max,
        "Fwd Packet Length Min": fwd_len.min,
        "Fwd Packet Length Mean": fwd_len.mean,
        "Fwd Packet Length Std": fwd_len.std,
        "Bwd Packet Length Max": bwd_len.max,
        "Bwd Packet Length Min": bwd_len.min,
        "Bwd Packet Length Mean": bwd_len.mean,
        "Bwd Packet Length Std": bwd_len.std,
        "Flow Bytes/s": _rate(total_bytes, duration, np.nan),
        "Flow Packets/s": _rate(total_pkts, duration, np.nan),
        "Flow IAT Mean": flow_iat.mean,
        "Flow IAT Std": flow_iat.std,
        "Flow IAT Max": flow_iat.max,
        "Flow IAT Min": flow_iat.min,
        "Fwd IAT Total": dir_iat.total[0::2],
        "Fwd IAT Mean": dir_iat.mean[0::2],
        "Fwd IAT Std": dir_iat.std[0::2],
        "Fwd IAT Max": dir_iat.max[0::2],
        "Fwd IAT Min": dir_iat.min[0::2],
        "Bwd IAT Total": dir_iat.total[1::2],
        "Bwd IAT Mean": dir_iat.mean[1::2],
        "Bwd IAT Std": dir_iat.std[1::2],
        "Bwd IAT Max": dir_iat.max[1::2],
        "Bwd IAT Min": dir_iat.min[1::2],
        "Fwd PSH Flags": psh_dir[0::2],
        "Bwd PSH Flags": psh_dir[1::2],
        "Fwd URG Flags": urg_dir[0::2],
        "Bwd URG Flags": urg_dir[1::2],
        "Fwd Header Length": header_dir.total[0::2],
        "Bwd Header Length": header_dir.total[1::2],
        "Fwd Packets/s": _direction_rate(fwd_pkts, duration),
        "Bwd Packets/s": _direction_rate(bwd_pkts, duration),
        "Min Packet Length": overall.min,
        "Max Packet Length": overall.max,
        "Packet Length Mean": overall.mean,
        "Packet Length Std": overall.std,
        "Packet Length Variance": overall.std**2,
        "FIN Flag Count": flag_count(FIN),
        "SYN Flag Count": flag_count(SYN),
        "RST Flag Count": flag_count(RST),
        "PSH Flag Count": flag_count(PSH),
        "ACK Flag Count": flag_count(ACK),
        "URG Flag Count": flag_count(URG),
        "CWE Flag Count": flag_count(CWE),
        "ECE Flag Count": flag_count(ECE),
        "Down/Up Ratio": down_up,
        "Average Packet Size": total_bytes / total_pkts,
        "Avg Fwd Segment Size": fwd_len.mean,
        "Avg Bwd Segment Size": bwd_len.mean,
        "Fwd Avg Bytes/Bulk": zeros,
        "Fwd Avg Packets/Bulk": zeros,
        "Fwd Avg Bulk Rate": zeros,
        "Bwd Avg Bytes/Bulk": zeros,
        "Bwd Avg Packets/Bulk": zeros,
        "Bwd Avg Bulk Rate": zeros,
        "Subflow Fwd Packets": fwd_pkts,
        "Subflow Fwd Bytes": fwd_len.total,
        "Subflow Bwd Packets": bwd_pkts,
        "Subflow Bwd Bytes": bwd_len.total,
        "Init_Win_bytes_forward": init_window[:, 0],
        "Init_Win_bytes_backward": init_window[:, 1],
        "act_data_pkt_fwd": np.bincount(flow, weights=~backward & (payload >= 1), minlength=n),
        "min_seg_size_forward": header_dir.min[0::2],
        "Active Mean": active.mean,
        "Active Std": active.std,
        "Active Max": active.max,
        "Active Min": active.min,
        "Idle Mean": idle.mean,
        "Idle Std": idle.std,
        "Idle Max": idle.max,
        "Idle Min": idle.min,
    }
    return pd.DataFrame({name: columns[name].astype(np.float32) for name in schema.FEATURES})


# ----------------------------------------------------------------------------------------------------------------
# Packet simulation: traffic profiles
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class _Packets:
    """Packets emitted by one profile: flow id, order key within the flow, gap before the packet (µs), direction,
    payload bytes and flag bits. Gaps are measured from the previous packet of the same flow."""

    fid: np.ndarray
    key: np.ndarray
    gap: np.ndarray
    bwd: np.ndarray
    size: np.ndarray
    flags: np.ndarray


class _FlowAttrs:
    """Flow-level properties the profiles fill in: port, transport, header size and the two opening windows."""

    def __init__(self, n: int) -> None:
        self.port = np.zeros(n, dtype=np.int64)
        self.udp = np.zeros(n, dtype=bool)
        self.header = np.full(n, 20, dtype=np.int64)
        self.win_fwd = np.full(n, -1, dtype=np.int64)
        self.win_bwd = np.full(n, -1, dtype=np.int64)


def _ln(rng: np.random.Generator, median: float, sigma: float, size: int) -> np.ndarray:
    """Lognormal draws described by their median and log-scale spread."""
    return rng.lognormal(np.log(median), sigma, size)


def _expand(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For flows with ``counts`` packets each, return per packet the owning flow (local index) and its position."""
    owner = np.repeat(np.arange(counts.size), counts)
    starts = np.cumsum(counts) - counts
    pos = np.arange(owner.size) - starts[owner]
    return owner, pos


def _web(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """Interactive web session (TCP 80/443/8080).

    Handshake (SYN, SYN-ACK, ACK), then a lognormal number of data packets (median about 10) in which the client
    sends small requests (median 350 bytes, often pure ACKs) and the server larger responses (median 900 bytes),
    millisecond-scale gaps (median 2.5 ms, capped at 4 s), and usually a FIN exchange at the end. A few flows
    negotiate ECN in the handshake and a very few data packets carry URG.
    """
    m = ids.size
    attrs.port[ids] = rng.choice([80, 443, 8080], size=m, p=[0.30, 0.65, 0.05])
    attrs.header[ids] = rng.choice([20, 32], size=m, p=[0.35, 0.65])
    attrs.win_fwd[ids] = rng.choice(_CLIENT_WINDOWS, size=m)
    attrs.win_bwd[ids] = rng.choice(_SERVER_WINDOWS, size=m)
    n_data = np.minimum(1 + np.floor(_ln(rng, 9.0, 0.8, m)), MAX_PACKETS - 5).astype(np.int64)
    closed = rng.random(m) < 0.85
    counts = 3 + n_data + 2 * closed
    owner, pos = _expand(counts)
    k = owner.size
    last_data = 2 + n_data[owner]
    data = (pos >= 3) & (pos <= last_data)
    teardown = pos > last_data
    first_request = pos == 3
    reply_odds = rng.uniform(0.45, 0.70, m)[owner]
    bwd = (pos == 1) | (data & ~first_request & (rng.random(k) < reply_odds)) | (teardown & (pos == counts[owner] - 1))
    request = data & ~bwd & (first_request | (rng.random(k) < 0.45))
    response = data & bwd & (rng.random(k) < 0.85)
    size = np.select(
        [request, response],
        [np.clip(np.rint(_ln(rng, 350, 0.9, k)), 1, 1460), np.clip(np.rint(_ln(rng, 900, 0.7, k)), 1, 1460)],
        0.0,
    )
    flags = np.full(k, ACK, dtype=np.int64)
    flags[pos == 0] = SYN
    flags[pos == 1] = SYN | ACK
    flags[(size > 0) & (rng.random(k) < 0.6)] |= PSH
    flags[teardown] |= FIN
    flags[data & (rng.random(k) < 0.002)] |= URG
    ecn = (rng.random(m) < 0.03)[owner]
    flags[(pos == 0) & ecn] |= ECE | CWE
    flags[(pos == 1) & ecn] |= ECE
    gap = np.select(
        [pos == 1, pos == 2, data, teardown],
        [_ln(rng, 800, 0.8, k), _ln(rng, 60, 0.5, k), np.minimum(_ln(rng, 2500, 1.5, k), 4e6), _ln(rng, 300, 0.8, k)],
        0.0,
    )
    return [_Packets(ids[owner], pos, gap, bwd, size, flags)]


def _dns(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """DNS lookup (UDP 53, 8-byte headers, no flags).

    90 % are one query and one answer (query 28-60 bytes, answer median 120 bytes, answer after a median 15 ms);
    4 % go unanswered (the query is repeated after 1-2 s and still gets no answer); 6 % repeat the query after
    1-2 s before the answer.
    """
    m = ids.size
    attrs.port[ids] = 53
    attrs.udp[ids] = True
    attrs.header[ids] = 8
    shape = rng.choice(3, size=m, p=[0.90, 0.04, 0.06])  # answered, unanswered, retried
    counts = np.array([2, 2, 3])[shape]
    owner, pos = _expand(counts)
    k = owner.size
    bwd = (pos == counts[owner] - 1) & (shape[owner] != 1)
    query = rng.integers(28, 61, m)[owner]
    answer = np.clip(np.rint(_ln(rng, 120, 0.5, m)), 40, 512)[owner]
    size = np.where(bwd, answer, query).astype(np.float64)
    gap = np.where(
        pos == 0, 0.0, np.where(bwd, np.clip(_ln(rng, 15_000, 1.0, k), 200, 2e6), rng.uniform(1e6, 2e6, k))
    )
    return [_Packets(ids[owner], pos, gap, bwd, size, np.zeros(k, dtype=np.int64))]


def _bulk(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """Bulk download (TCP 443/80).

    Handshake, one request (median 400 bytes), then 25-294 data packets of which 65-80 % come from the server,
    mostly full 1460-byte segments, while the client only acknowledges; sub-millisecond gaps; FIN exchange.
    """
    m = ids.size
    attrs.port[ids] = rng.choice([443, 80], size=m, p=[0.8, 0.2])
    attrs.header[ids] = rng.choice([20, 32], size=m, p=[0.3, 0.7])
    attrs.win_fwd[ids] = rng.choice(_CLIENT_WINDOWS, size=m)
    attrs.win_bwd[ids] = rng.choice(_SERVER_WINDOWS, size=m)
    n_data = rng.integers(25, MAX_PACKETS - 5, m)
    counts = 4 + n_data + 2
    owner, pos = _expand(counts)
    k = owner.size
    data = (pos >= 4) & (pos < 4 + n_data[owner])
    teardown = pos >= 4 + n_data[owner]
    server_odds = rng.uniform(0.65, 0.80, m)[owner]
    bwd = (pos == 1) | (data & (rng.random(k) < server_odds)) | (teardown & (pos == counts[owner] - 1))
    segment = np.where(rng.random(k) < 0.88, 1460, rng.integers(100, 1461, k))
    size = np.select(
        [pos == 3, data & bwd], [np.clip(np.rint(_ln(rng, 400, 0.5, k)), 40, 1460), segment], 0.0
    ).astype(np.float64)
    flags = np.full(k, ACK, dtype=np.int64)
    flags[pos == 0] = SYN
    flags[pos == 1] = SYN | ACK
    flags[pos == 3] |= PSH
    flags[data & bwd & (rng.random(k) < 0.1)] |= PSH
    flags[teardown] |= FIN
    gap = np.select(
        [pos == 1, pos == 2, pos == 3, data, teardown],
        [
            _ln(rng, 1500, 0.7, k),
            _ln(rng, 80, 0.5, k),
            _ln(rng, 150, 0.5, k),
            np.minimum(_ln(rng, 250, 1.2, k), 2e6),
            _ln(rng, 400, 0.8, k),
        ],
        0.0,
    )
    return [_Packets(ids[owner], pos, gap, bwd, size, flags)]


def _keepalive(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """Long-lived keep-alive connection seen mid-life (TCP 443/993/5223, no handshake).

    3-9 small packets (median 45 bytes, some pure ACKs) from both sides, in bursts separated by idle gaps of
    5.5-45 s; at least one such gap is guaranteed, so Active/Idle columns are populated. Windows are the scaled
    values of an established connection.
    """
    m = ids.size
    attrs.port[ids] = rng.choice([443, 993, 5223], size=m, p=[0.75, 0.15, 0.10])
    attrs.header[ids] = rng.choice([20, 32], size=m, p=[0.3, 0.7])
    attrs.win_fwd[ids] = rng.choice([255, 501, 1026, 2053], size=m)
    attrs.win_bwd[ids] = rng.choice([83, 235, 501, 1002], size=m)
    counts = rng.integers(3, 10, m)
    owner, pos = _expand(counts)
    k = owner.size
    bwd = (pos > 0) & (rng.random(k) < 0.5)
    size = np.where(rng.random(k) < 0.8, np.clip(np.rint(_ln(rng, 45, 0.6, k)), 1, 300), 0.0)
    flags = np.where(size > 0, ACK | PSH, ACK).astype(np.int64)
    forced = (1 + np.floor(rng.random(m) * (counts - 1))).astype(np.int64)[owner]
    idle = (pos > 0) & ((rng.random(k) < 0.35) | (pos == forced))
    gap = np.where(
        pos == 0, 0.0, np.where(idle, rng.uniform(5.5e6, 4.5e7, k), np.minimum(_ln(rng, 900, 1.0, k), 1e6))
    )
    return [_Packets(ids[owner], pos, gap, bwd, size, flags)]


def _refused(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """Refused connection (normal traffic that fails): a SYN answered by RST-ACK after a median 400 µs.

    70 % target 80/443, the rest a random port; 15 % retransmit the SYN about a second later first. Ordinary
    operating-system windows and header sizes, which is the main thing telling them apart from a sweep.
    """
    m = ids.size
    attrs.port[ids] = np.where(rng.random(m) < 0.7, rng.choice([80, 443], size=m), rng.integers(1, 65536, m))
    attrs.header[ids] = rng.choice([20, 32], size=m)
    attrs.win_fwd[ids] = rng.choice(_CLIENT_WINDOWS, size=m)
    attrs.win_bwd[ids] = 0
    counts = 2 + (rng.random(m) < 0.15)
    owner, pos = _expand(counts)
    k = owner.size
    bwd = pos == counts[owner] - 1
    flags = np.where(bwd, RST | ACK, SYN).astype(np.int64)
    gap = np.where(pos == 0, 0.0, np.where(bwd, _ln(rng, 400, 0.8, k), rng.uniform(0.9e6, 1.1e6, k)))
    return [_Packets(ids[owner], pos, gap, bwd, np.zeros(k), flags)]


def _flood(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """Flood (DoS/DDoS-like) aimed at port 80: very short, one-sided and fast.

    Two or more forward packets (one plus a geometric number; mean about 4, at most 40) sent tens of microseconds
    apart; in :data:`SAME_TICK_FLOODS` of flows the whole burst lands in one clock tick (zero duration). 55 % of
    flows are SYN floods with empty payloads; the rest push ACK packets with 0-12 byte payloads. The server
    answers with a single RST-ACK in 25 % (SYN) or 40 % (ACK) of flows; unanswered ACK floods often end with the
    attacker's own RST; a tenth of the ACK floods also set URG on every packet. Tool-chosen small windows
    (512-3072).
    """
    m = ids.size
    attrs.port[ids] = 80
    attrs.header[ids] = 20
    attrs.win_fwd[ids] = rng.choice([512, 1024, 2048, 3072], size=m)
    attrs.win_bwd[ids] = 0
    syn_style = rng.random(m) < 0.55
    n_fwd = np.minimum(1 + rng.geometric(0.35, m), 40)
    answered = rng.random(m) < np.where(syn_style, 0.25, 0.40)
    counts = n_fwd + answered
    owner, pos = _expand(counts)
    k = owner.size
    syn = syn_style[owner]
    bwd = pos >= n_fwd[owner]
    carries = ~syn & ~bwd & (rng.random(k) < 0.5)
    size = np.where(carries, rng.integers(1, 13, k), 0).astype(np.float64)
    flags = np.select([bwd, syn], [RST | ACK, SYN], ACK).astype(np.int64)
    flags[carries] |= PSH
    urgent = (~syn_style & (rng.random(m) < 0.1))[owner] & ~bwd
    flags[urgent] |= URG
    last_fwd = (pos == n_fwd[owner] - 1) & ~syn & ~answered[owner] & (rng.random(k) < 0.6)
    flags[last_fwd] |= RST
    same_tick = (rng.random(m) < SAME_TICK_FLOODS)[owner]
    gap = np.where((pos == 0) | same_tick, 0.0, np.clip(_ln(rng, 35, 0.9, k), 0, 5000))
    return [_Packets(ids[owner], pos, gap, bwd, size, flags)]


def _slow_drip(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """Slow drip (slowloris-like) against port 80: keep a connection open by trickling a request.

    Handshake, then a partial request (60-220 bytes), then tiny forward fragments (1-24 bytes, PSH) every 10-15 s
    (steady per flow, +-3 % jitter) until 60-120 s have passed. The server acknowledges only about a quarter of
    the fragments, so replies are few.
    """
    m = ids.size
    attrs.port[ids] = 80
    attrs.header[ids] = rng.choice([20, 32], size=m)
    attrs.win_fwd[ids] = rng.choice(_CLIENT_WINDOWS, size=m)
    attrs.win_bwd[ids] = rng.choice(_SERVER_WINDOWS, size=m)
    target = rng.uniform(60e6, 120e6, m)
    cadence = rng.uniform(10e6, 15e6, m)
    n_frag = np.maximum(1, np.rint(target / cadence)).astype(np.int64)

    owner, pos = _expand(np.full(m, 3))
    k = owner.size
    opening = _Packets(
        ids[owner],
        pos,
        np.select([pos == 1, pos == 2], [_ln(rng, 700, 0.7, k), _ln(rng, 60, 0.5, k)], 0.0),
        pos == 1,
        np.where(pos == 2, rng.integers(60, 221, k), 0).astype(np.float64),
        np.select([pos == 0, pos == 1, pos == 2], [SYN, SYN | ACK, ACK | PSH], ACK).astype(np.int64),
    )

    f_owner, j = _expand(n_frag)
    f = f_owner.size
    acked = rng.random(f) < 0.25
    fragments = _Packets(
        ids[f_owner],
        3 + 2 * j,
        cadence[f_owner] * rng.uniform(0.97, 1.03, f),
        np.zeros(f, dtype=bool),
        rng.integers(1, 25, f).astype(np.float64),
        np.full(f, ACK | PSH, dtype=np.int64),
    )
    a_owner = f_owner[acked]
    a = a_owner.size
    replies = _Packets(
        ids[a_owner],
        4 + 2 * j[acked],
        _ln(rng, 400, 0.5, a),
        np.ones(a, dtype=bool),
        np.zeros(a),
        np.full(a, ACK, dtype=np.int64),
    )
    return [opening, fragments, replies]


def _sweep(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """Sweep (port scan): one empty SYN to a port, then a reset, an open-port reply or nothing.

    60 % of targets are well-known service ports, the rest random. 70 % are half-open probes (24-byte header,
    windows 1024-4096); 30 % use the full operating-system stack. Replies: RST-ACK 55 %, SYN-ACK 10 % (the
    scanner then resets), silence 35 % (the scanner sends the SYN once more after its timeout, median 150 ms,
    then gives up). Answered probes last tens of microseconds.
    """
    m = ids.size
    attrs.port[ids] = np.where(rng.random(m) < 0.6, rng.choice(_COMMON_PORTS, size=m), rng.integers(1, 65536, m))
    half_open = rng.random(m) < 0.7
    attrs.header[ids] = np.where(half_open, 24, rng.choice([20, 32], size=m))
    attrs.win_fwd[ids] = np.where(
        half_open, rng.choice([1024, 2048, 3072, 4096], size=m), rng.choice([8192, 29200, 64240], size=m)
    )
    reply = rng.choice(3, size=m, p=[0.55, 0.10, 0.35])  # reset, open port, silence
    attrs.win_bwd[ids] = np.where(reply == 1, rng.choice(_SERVER_WINDOWS, size=m), 0)
    counts = np.array([2, 3, 2])[reply]
    owner, pos = _expand(counts)
    k = owner.size
    r = reply[owner]
    retry = (pos == 1) & (r == 2)
    bwd = (pos == 1) & ~retry
    closing = np.where(half_open[owner], RST, RST | ACK)
    flags = np.select(
        [pos == 0, retry, bwd & (r == 0), bwd & (r == 1), pos == 2], [SYN, SYN, RST | ACK, SYN | ACK, closing], 0
    ).astype(np.int64)
    gap = np.select(
        [retry, pos == 1, pos == 2],
        [np.clip(_ln(rng, 150_000, 0.6, k), 20_000, 1.5e6), _ln(rng, 60, 0.6, k), _ln(rng, 25, 0.5, k)],
        0.0,
    )
    return [_Packets(ids[owner], pos, gap, bwd, np.zeros(k), flags)]


def _credential(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """Credential guessing against FTP (21) or SSH (22): a scripted, metronome-like conversation.

    Handshake and a server greeting, then 10-25 exchanges of a small forward attempt (FTP 8-40 bytes, SSH 36-100)
    and a small reply (20-70 bytes). Attempts follow a steady per-flow cadence of 0.25-2.5 s with only +-4 %
    jitter and the server answers after a steady ~4 ms, so the forward inter-arrival spread is small. FIN at end.
    """
    m = ids.size
    ssh = rng.random(m) < 0.5
    attrs.port[ids] = np.where(ssh, 22, 21)
    attrs.header[ids] = 32
    attrs.win_fwd[ids] = rng.choice([29200, 64240], size=m, p=[0.7, 0.3])
    attrs.win_bwd[ids] = rng.choice([26847, 28960, 65160], size=m)
    exchanges = rng.integers(10, 26, m)
    counts = 4 + 2 * exchanges + 2
    owner, pos = _expand(counts)
    k = owner.size
    is_ssh = ssh[owner]
    step = pos - 4
    exchange = (pos >= 4) & (pos < counts[owner] - 2)
    teardown = pos >= counts[owner] - 2
    bwd = (pos == 1) | (pos == 3) | (exchange & (step % 2 == 1)) | (pos == counts[owner] - 1)
    greeting = np.where(is_ssh, rng.integers(21, 46, k), rng.integers(20, 61, k))
    attempt = np.where(is_ssh, rng.integers(36, 101, k), rng.integers(8, 41, k))
    size = np.select(
        [pos == 3, exchange & ~bwd, exchange & bwd], [greeting, attempt, rng.integers(20, 71, k)], 0
    ).astype(np.float64)
    flags = np.select([pos == 0, pos == 1, teardown], [SYN, SYN | ACK, FIN | ACK], ACK).astype(np.int64)
    flags[size > 0] |= PSH
    cadence = rng.uniform(0.25e6, 2.5e6, m)[owner]
    gap = np.select(
        [pos == 1, pos == 2, pos == 3, exchange & ~bwd, exchange & bwd, teardown],
        [
            _ln(rng, 300, 0.4, k),
            _ln(rng, 60, 0.4, k),
            _ln(rng, 3000, 0.4, k),
            cadence * rng.uniform(0.96, 1.04, k),
            _ln(rng, 4000, 0.25, k),
            _ln(rng, 250, 0.5, k),
        ],
        0.0,
    )
    return [_Packets(ids[owner], pos, gap, bwd, size, flags)]


def _injection(rng: np.random.Generator, ids: np.ndarray, attrs: _FlowAttrs) -> list[_Packets]:
    """Web injection (SQL/script injection style) on port 80 (15 % on 8080).

    Handshake, then one unusually large forward request (500-1460 bytes, median 1000) with PSH, a server ACK,
    a pause while the server works (median 20 ms), a moderate response of 1-4 packets (median 450 bytes each,
    PSH on the last), the client's ACK and a FIN exchange after a median half-second.
    """
    m = ids.size
    attrs.port[ids] = np.where(rng.random(m) < 0.85, 80, 8080)
    attrs.header[ids] = rng.choice([20, 32], size=m)
    attrs.win_fwd[ids] = rng.choice([29200, 64240, 65535], size=m)
    attrs.win_bwd[ids] = rng.choice([28960, 65160], size=m)
    n_resp = rng.integers(1, 5, m)
    counts = 8 + n_resp
    owner, pos = _expand(counts)
    k = owner.size
    last_resp = 4 + n_resp[owner]
    response = (pos >= 5) & (pos <= last_resp)
    teardown = pos >= counts[owner] - 2
    bwd = (pos == 1) | (pos == 4) | response | (pos == counts[owner] - 1)
    size = np.select(
        [pos == 3, response],
        [np.clip(np.rint(_ln(rng, 1000, 0.35, k)), 500, 1460), np.clip(np.rint(_ln(rng, 450, 0.5, k)), 60, 1460)],
        0.0,
    )
    flags = np.select([pos == 0, pos == 1, teardown], [SYN, SYN | ACK, FIN | ACK], ACK).astype(np.int64)
    flags[(pos == 3) | (pos == last_resp)] |= PSH
    gap = np.select(
        [pos == 1, pos == 2, pos == 3, pos == 4, pos == 5, response, pos == last_resp + 1, teardown],
        [
            _ln(rng, 500, 0.6, k),
            _ln(rng, 60, 0.5, k),
            _ln(rng, 200, 0.5, k),
            _ln(rng, 150, 0.5, k),
            _ln(rng, 20_000, 0.8, k),
            _ln(rng, 80, 0.5, k),
            _ln(rng, 100, 0.5, k),
            np.minimum(_ln(rng, 5e5, 1.0, k), 4e6),
        ],
        0.0,
    )
    return [_Packets(ids[owner], pos, gap, bwd, size, flags)]


_Builder = Callable[[np.random.Generator, np.ndarray, _FlowAttrs], list[_Packets]]
# Profile codes: 0-3 everyday normal behaviours (same order as BENIGN_MIX), 4 refused connection, and
# 5-9 the attacks in SYNTHETIC_CLASSES order (class code c uses profile 4 + c).
_BUILDERS: tuple[_Builder, ...] = (
    _web,
    _dns,
    _bulk,
    _keepalive,
    _refused,
    _flood,
    _slow_drip,
    _sweep,
    _credential,
    _injection,
)
_REFUSED = 4
_COVER_PROFILES = {"web": 0, "bulk": 2, "keepalive": 3}


def _draw_classes(rng: np.random.Generator, n_flows: int, attack_share: float) -> np.ndarray:
    """Class code per flow: exactly round(n * attack_share) attacks, split between attack classes by ATTACK_MIX."""
    n_attack = int(round(n_flows * attack_share))
    weights = np.array([ATTACK_MIX[name] for name in SYNTHETIC_CLASSES[1:]])
    per_class = rng.multinomial(n_attack, weights / weights.sum())
    codes = np.concatenate(
        [np.zeros(n_flows - n_attack, dtype=np.int64), np.repeat(np.arange(1, len(SYNTHETIC_CLASSES)), per_class)]
    )
    return rng.permutation(codes)


def _assign_profiles(rng: np.random.Generator, codes: np.ndarray, blur: float) -> tuple[np.ndarray, np.ndarray]:
    """Profile code per flow, and a mask of the attack flows disguised as normal behaviour."""
    profile = np.empty(codes.size, dtype=np.int64)
    benign = codes == 0
    n_benign = int(benign.sum())
    everyday = rng.choice(len(BENIGN_MIX), size=n_benign, p=list(BENIGN_MIX.values()))
    refused = rng.random(n_benign) < blur / 4
    profile[benign] = np.where(refused, _REFUSED, everyday)
    attack = ~benign
    n_attack = int(attack.sum())
    covers = np.array([_COVER_PROFILES[name] for name in COVER_MIX])
    cover = rng.choice(covers, size=n_attack, p=list(COVER_MIX.values()))
    hidden = rng.random(n_attack) < blur
    profile[attack] = np.where(hidden, cover, _REFUSED + codes[attack])
    disguised = np.zeros(codes.size, dtype=bool)
    disguised[attack] = hidden
    return profile, disguised


def _attack_ports(rng: np.random.Generator, codes: np.ndarray) -> np.ndarray:
    """Destination port a disguised attack keeps from its own profile (the one thing it does not copy)."""
    m = codes.size
    sweep = SYNTHETIC_CLASSES.index("Sweep")
    guess = SYNTHETIC_CLASSES.index("Credential Guess")
    return np.select([codes == sweep, codes == guess], [rng.integers(1, 65536, m), rng.choice([21, 22], size=m)], 80)


def _assemble(parts: list[_Packets], attrs: _FlowAttrs, n_flows: int) -> PacketTable:
    """Merge profile output into one PacketTable sorted by (flow, time), with times, headers and windows filled in.

    Packets beyond :data:`MAX_PACKETS` in a flow are dropped. Each direction's first packet carries the flow's
    opening window; later packets carry a scaled-down value, as established TCP connections do.
    """
    fid = np.concatenate([p.fid for p in parts])
    key = np.concatenate([p.key for p in parts])
    order = np.lexsort((key, fid))
    fid = fid[order]
    gap = np.concatenate([p.gap for p in parts])[order]
    bwd = np.concatenate([p.bwd for p in parts]).astype(bool)[order]
    size = np.concatenate([p.size for p in parts])[order]
    flags = np.concatenate([p.flags for p in parts]).astype(np.int64)[order]

    counts = np.bincount(fid, minlength=n_flows)
    starts = np.cumsum(counts) - counts
    pos = np.arange(fid.size) - starts[fid]
    keep = pos < MAX_PACKETS
    if not keep.all():
        fid, gap, bwd, size, flags = fid[keep], gap[keep], bwd[keep], size[keep], flags[keep]
        counts = np.bincount(fid, minlength=n_flows)
        starts = np.cumsum(counts) - counts

    gap = np.rint(np.maximum(gap, 0.0))
    gap[starts] = 0.0
    elapsed = np.cumsum(gap)
    time_us = elapsed - elapsed[starts][fid]

    bwd_seen = np.cumsum(bwd)
    fwd_seen = np.cumsum(~bwd)
    nth_bwd = bwd_seen - (bwd_seen[starts] - bwd[starts])[fid]
    nth_fwd = fwd_seen - (fwd_seen[starts] - ~bwd[starts])[fid]
    opening = np.where(bwd, nth_bwd == 1, nth_fwd == 1)
    initial = np.where(bwd, attrs.win_bwd[fid], attrs.win_fwd[fid])
    window = np.where(opening, initial, np.maximum(initial, 0) >> 7)

    return PacketTable(
        flow=fid,
        time_us=time_us,
        backward=bwd,
        payload=np.rint(size),
        header=attrs.header[fid].astype(np.float64),
        flags=flags,
        window=window,
        port=attrs.port,
        udp=attrs.udp,
    )


def _simulate_block(rng: np.random.Generator, profile: np.ndarray, ports: np.ndarray) -> pd.DataFrame:
    """Simulate and meter one block of flows.

    ``profile`` holds each flow's profile code; ``ports`` a destination port that overrides the profile's choice
    (used by disguised attacks, which keep their own port), or -1 to keep the profile's port.
    """
    n = profile.size
    attrs = _FlowAttrs(n)
    parts: list[_Packets] = []
    for code, build in enumerate(_BUILDERS):
        ids = np.flatnonzero(profile == code)
        if ids.size:
            parts.extend(build(rng, ids, attrs))
    override = ports >= 0
    attrs.port[override] = ports[override]
    return measure(_assemble(parts, attrs, n))


def generate(
    n_flows: int,
    *,
    seed: int,
    attack_share: float = 0.35,
    blur: float = 0.04,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Simulate ``n_flows`` labelled network flows.

    Returns the 77 ``schema.FEATURES`` columns (float32, same order) plus ``Label`` (pandas string dtype) with
    values from :data:`SYNTHETIC_CLASSES`. Exactly ``round(n_flows * attack_share)`` flows are attacks, split
    between attack classes by :data:`ATTACK_MIX`. ``blur`` is the share of attack flows disguised as normal
    traffic (and a quarter of it the share of normal flows that are refused connections). The result depends
    only on the arguments: the same arguments always give an identical frame.

    Flows are simulated in blocks of :data:`BLOCK_FLOWS` (see the module notes); ``progress``, when given, is
    called after each block with (flows done, ``n_flows``).
    """
    if isinstance(n_flows, bool) or not isinstance(n_flows, (int, np.integer)) or n_flows < 0:
        raise ValueError("n_flows must be a non-negative integer.")
    if not 0.0 <= attack_share <= 1.0:
        raise ValueError("attack_share must lie between 0 and 1.")
    if not 0.0 <= blur <= 1.0:
        raise ValueError("blur must lie between 0 and 1.")
    n_flows = int(n_flows)
    n_blocks = -(-n_flows // BLOCK_FLOWS)
    plan_seed, *block_seeds = np.random.SeedSequence(seed).spawn(1 + n_blocks)

    # The whole run's class and profile choices come first (a few bytes per flow), so they do not depend on blocks.
    rng = np.random.default_rng(plan_seed)
    codes = _draw_classes(rng, n_flows, attack_share)
    profile, disguised = _assign_profiles(rng, codes, blur)
    ports = np.full(n_flows, -1, dtype=np.int64)
    if disguised.any():
        ports[disguised] = _attack_ports(rng, codes[disguised])

    values = np.empty((len(schema.FEATURES), n_flows), dtype=np.float32)
    for block, child in enumerate(block_seeds):
        lo, hi = block * BLOCK_FLOWS, min(n_flows, (block + 1) * BLOCK_FLOWS)
        metered = _simulate_block(np.random.default_rng(child), profile[lo:hi], ports[lo:hi])
        for i, name in enumerate(schema.FEATURES):
            values[i, lo:hi] = metered[name].to_numpy()
        del metered
        if progress is not None:
            progress(hi, n_flows)
    frame = pd.DataFrame(values.T, columns=list(schema.FEATURES), copy=False)
    labels = np.asarray(SYNTHETIC_CLASSES, dtype=object)[codes]
    frame[schema.LABEL] = pd.Series(labels, index=frame.index, dtype="str")
    return frame
