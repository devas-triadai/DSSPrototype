"""Tests for LicenseManager."""

from backend.dataset_catalog.license_manager import KNOWN_LICENSES, LicenseManager


def test_classify_license_by_id() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("mit")
    assert lic.license_id == "mit"
    assert lic.is_open is True


def test_classify_license_by_spdx() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("CC0-1.0")
    assert lic.license_id == "cc0"


def test_classify_license_by_name() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("Apache License 2.0")
    assert lic.license_id == "apache_2"


def test_classify_cc_attribution() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("Creative Commons Attribution 4.0")
    assert lic.license_id == "cc_by_4"


def test_classify_cc_non_commercial() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("Creative Commons NonCommercial")
    assert lic.license_id == "cc_nc"


def test_classify_cc_zero() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("CC0 1.0 Universal")
    assert lic.license_id == "cc0"


def test_classify_proprietary() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("Proprietary License")
    assert lic.license_id == "proprietary"


def test_classify_unknown() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("Some random license text")
    assert lic.license_id == "unknown"


def test_classify_public_domain() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("Public Domain")
    assert lic.license_id == "cc0"


def test_classify_all_rights_reserved() -> None:
    lm = LicenseManager()
    lic = lm.classify_license("All Rights Reserved")
    assert lic.license_id == "proprietary"


def test_is_allowed() -> None:
    lm = LicenseManager()
    assert lm.is_allowed(KNOWN_LICENSES["mit"]) is True
    assert lm.is_allowed(KNOWN_LICENSES["cc0"]) is True
    assert lm.is_allowed(KNOWN_LICENSES["proprietary"]) is False


def test_is_restricted() -> None:
    lm = LicenseManager()
    assert lm.is_restricted(KNOWN_LICENSES["cc_nc"]) is True
    assert lm.is_restricted(KNOWN_LICENSES["unknown"]) is True
    assert lm.is_restricted(KNOWN_LICENSES["mit"]) is False


def test_compute_risk_score() -> None:
    lm = LicenseManager()
    assert lm.compute_risk_score(KNOWN_LICENSES["cc0"]) == 0.0
    assert lm.compute_risk_score(KNOWN_LICENSES["proprietary"]) == 1.0
    assert lm.compute_risk_score(KNOWN_LICENSES["mit"]) == 0.05


def test_get_license_compatibility_same() -> None:
    lm = LicenseManager()
    assert lm.get_license_compatibility(KNOWN_LICENSES["mit"], KNOWN_LICENSES["mit"]) == 1.0


def test_get_license_compatibility_compatible() -> None:
    lm = LicenseManager()
    score = lm.get_license_compatibility(KNOWN_LICENSES["mit"], KNOWN_LICENSES["cc0"])
    assert score > 0.5


def test_get_license_compatibility_incompatible() -> None:
    lm = LicenseManager()
    score = lm.get_license_compatibility(KNOWN_LICENSES["proprietary"], KNOWN_LICENSES["mit"])
    assert score == 0.0


def test_get_license_compatibility_share_alike() -> None:
    lm = LicenseManager()
    score = lm.get_license_compatibility(KNOWN_LICENSES["cc_by_sa_4"], KNOWN_LICENSES["mit"])
    assert score < 0.5


def test_known_licenses_are_consistent() -> None:
    for lid, info in KNOWN_LICENSES.items():
        assert info.license_id == lid
        assert 0.0 <= info.risk_score <= 1.0
