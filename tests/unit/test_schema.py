"""Column catalogue and label normalisation."""

from __future__ import annotations

import pytest

from graticule import schema

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Web Attack � Brute Force", "Web Attack - Brute Force"),
        ("Web Attack \x96 XSS", "Web Attack - XSS"),
        ("Web Attack – Sql Injection", "Web Attack - Sql Injection"),
        ("Web Attack - Brute Force", "Web Attack - Brute Force"),
        ("  BENIGN ", "BENIGN"),
        ("FTP-Patator", "FTP-Patator"),
        ("DoS   Hulk", "DoS Hulk"),
        (None, ""),
    ],
)
def test_normalize_label(raw: object, expected: str) -> None:
    assert schema.normalize_label(raw) == expected


def test_feature_catalogue_is_consistent() -> None:
    assert len(schema.FEATURES) == 77
    assert len(set(schema.FEATURES)) == 77
    assert schema.DESTINATION_PORT in schema.FEATURE_SET
    assert all(col in schema.FEATURE_SET for col in schema.CURATED)
    assert len(schema.CURATED) == 28 == len(set(schema.CURATED))
    assert schema.DESTINATION_PORT not in schema.CURATED
    assert set(schema.CURATED_GROUPS) == set(schema.CURATED_WHY)


def test_known_labels_are_already_normalised() -> None:
    assert all(schema.normalize_label(lbl) == lbl for lbl in schema.KNOWN_LABELS)
    assert len(schema.EXPECTED_FILES) == 8
