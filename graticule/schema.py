"""Column names, label vocabulary and file catalogue for CIC-IDS2017 flow records.

The raw CSV headers carry stray leading spaces and one repeated column; everything in Graticule works with the
cleaned names defined here. The synthetic generator emits exactly the same columns, so the rest of the pipeline
never needs to know where a flow came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LABEL = "Label"
BENIGN = "BENIGN"
ATTACK = "Attack"
NORMAL = "Normal"
DESTINATION_PORT = "Destination Port"
WEB_ATTACK = "Web Attack"
WEB_ATTACK_PREFIX = "Web Attack -"

# The 77 distinct numeric features, in the order they appear in the source files
# (the repeated "Fwd Header Length" column is removed when files are read).
FEATURES: tuple[str, ...] = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
)
FEATURE_SET = frozenset(FEATURES)
RATE_COLUMNS: tuple[str, ...] = ("Flow Bytes/s", "Flow Packets/s")

# Curated feature set: 28 columns grouped by the behaviour they describe. Chosen from what each column measures,
# not from any fitted score, so the choice cannot leak information from the test data.
CURATED_GROUPS: dict[str, tuple[str, ...]] = {
    "Volume": (
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
    ),
    "Payload shape": (
        "Fwd Packet Length Max",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Max",
        "Bwd Packet Length Mean",
        "Packet Length Std",
    ),
    "Tempo": (
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Fwd IAT Mean",
    ),
    "TCP flags": (
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
    ),
    "TCP setup": (
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "act_data_pkt_fwd",
    ),
    "Rhythm": ("Active Mean", "Idle Mean"),
    "Balance": ("Down/Up Ratio",),
}
CURATED_WHY: dict[str, str] = {
    "Volume": "Floods and scans are small and one-sided; bulk transfers are large and two-sided.",
    "Payload shape": "Injected requests and scripted logins have characteristic packet sizes.",
    "Tempo": "Separates bursts (floods), slow drip-feeding (slow DoS) and scripted regularity (brute force, bots).",
    "TCP flags": "Half-open connections and resets leave distinctive flag counts.",
    "TCP setup": "Tool-specific window sizes and connections that never carry data.",
    "Rhythm": "Long idle periods mark slow attacks and beaconing.",
    "Balance": "Reply-heavy versus request-heavy conversations.",
}
CURATED: tuple[str, ...] = tuple(col for group in CURATED_GROUPS.values() for col in group)

# Canonical attack labels as they appear after normalisation.
KNOWN_LABELS: tuple[str, ...] = (
    BENIGN,
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - Sql Injection",
    "Web Attack - XSS",
)


@dataclass(frozen=True)
class ExpectedFile:
    """One of the eight CIC-IDS2017 MachineLearningCSV files and what it contains."""

    name: str
    day: str
    contents: str


EXPECTED_FILES: tuple[ExpectedFile, ...] = (
    ExpectedFile("Monday-WorkingHours.pcap_ISCX.csv", "Monday", "normal traffic only"),
    ExpectedFile("Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday", "FTP and SSH password guessing"),
    ExpectedFile("Wednesday-workingHours.pcap_ISCX.csv", "Wednesday", "DoS variants and Heartbleed"),
    ExpectedFile(
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Thursday morning", "web attacks"
    ),
    ExpectedFile(
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Thursday afternoon", "infiltration"
    ),
    ExpectedFile("Friday-WorkingHours-Morning.pcap_ISCX.csv", "Friday morning", "botnet"),
    ExpectedFile("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Friday afternoon", "port scan"),
    ExpectedFile("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Friday afternoon", "DDoS"),
)
EXPECTED_BY_NAME: dict[str, ExpectedFile] = {f.name.lower(): f for f in EXPECTED_FILES}

# Characters that stand in for the dash inside the Web Attack labels, depending on how a copy was saved:
# the Unicode replacement character, the raw Windows-1252 en dash byte decoded as latin-1, and real dashes.
_DASH_STANDINS = re.compile("[�\x96–—]")
_SPACES = re.compile(r"\s+")


def normalize_label(raw: object) -> str:
    """Return a clean class label: stand-in characters become " - " and whitespace is collapsed.

    >>> normalize_label("Web Attack � Brute Force")
    'Web Attack - Brute Force'
    """
    text = "" if raw is None else str(raw)
    text = _DASH_STANDINS.sub(" - ", text)
    text = _SPACES.sub(" ", text).strip()
    text = text.replace(" - - ", " - ")
    return text


def is_benign(label: str) -> bool:
    """True when ``label`` names normal traffic."""
    return label.strip().upper() == BENIGN
