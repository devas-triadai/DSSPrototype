"""License management and risk assessment for datasets."""

from __future__ import annotations

import logging

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.interfaces import LicenseManagerInterface
from backend.dataset_catalog.models import LicenseInfo

logger = logging.getLogger("dss.dataset_catalog.license_manager")

# Known license reference data
KNOWN_LICENSES: dict[str, LicenseInfo] = {
    "cc0": LicenseInfo(
        license_id="cc0",
        name="CC0 1.0 Universal",
        spdx_identifier="CC0-1.0",
        is_open=True,
        allows_commercial=True,
        allows_modification=True,
        requires_attribution=False,
        requires_share_alike=False,
        risk_score=0.0,
        compatible_licenses=["cc_by_4", "mit", "apache_2"],
    ),
    "cc_by_4": LicenseInfo(
        license_id="cc_by_4",
        name="Creative Commons Attribution 4.0",
        spdx_identifier="CC-BY-4.0",
        is_open=True,
        allows_commercial=True,
        allows_modification=True,
        requires_attribution=True,
        requires_share_alike=False,
        risk_score=0.1,
        compatible_licenses=["cc0"],
    ),
    "cc_by_sa_4": LicenseInfo(
        license_id="cc_by_sa_4",
        name="Creative Commons Attribution-ShareAlike 4.0",
        spdx_identifier="CC-BY-SA-4.0",
        is_open=True,
        allows_commercial=True,
        allows_modification=True,
        requires_attribution=True,
        requires_share_alike=True,
        risk_score=0.2,
        compatible_licenses=["cc0", "cc_by_4"],
    ),
    "cc_nc": LicenseInfo(
        license_id="cc_nc",
        name="Creative Commons Non-Commercial",
        spdx_identifier="CC-BY-NC-4.0",
        is_open=True,
        allows_commercial=False,
        allows_modification=True,
        requires_attribution=True,
        requires_share_alike=False,
        risk_score=0.6,
        compatible_licenses=[],
    ),
    "mit": LicenseInfo(
        license_id="mit",
        name="MIT License",
        spdx_identifier="MIT",
        is_open=True,
        allows_commercial=True,
        allows_modification=True,
        requires_attribution=True,
        requires_share_alike=False,
        risk_score=0.05,
        compatible_licenses=["cc0", "apache_2"],
    ),
    "apache_2": LicenseInfo(
        license_id="apache_2",
        name="Apache License 2.0",
        spdx_identifier="Apache-2.0",
        is_open=True,
        allows_commercial=True,
        allows_modification=True,
        requires_attribution=True,
        requires_share_alike=False,
        risk_score=0.05,
        compatible_licenses=["cc0", "mit"],
    ),
    "odbl": LicenseInfo(
        license_id="odbl",
        name="Open Database License",
        spdx_identifier="ODbL",
        is_open=True,
        allows_commercial=True,
        allows_modification=True,
        requires_attribution=True,
        requires_share_alike=True,
        risk_score=0.3,
        compatible_licenses=["cc0"],
    ),
    "pddl": LicenseInfo(
        license_id="pddl",
        name="Public Domain Dedication and License",
        spdx_identifier="PDDL",
        is_open=True,
        allows_commercial=True,
        allows_modification=True,
        requires_attribution=False,
        requires_share_alike=False,
        risk_score=0.0,
        compatible_licenses=["cc0", "mit", "apache_2", "cc_by_4"],
    ),
    "proprietary": LicenseInfo(
        license_id="proprietary",
        name="Proprietary License",
        spdx_identifier="LicenseRef-Proprietary",
        is_open=False,
        allows_commercial=False,
        allows_modification=False,
        requires_attribution=True,
        requires_share_alike=False,
        risk_score=1.0,
        compatible_licenses=[],
    ),
    "unknown": LicenseInfo(
        license_id="unknown",
        name="Unknown License",
        is_open=False,
        allows_commercial=False,
        allows_modification=False,
        requires_attribution=True,
        requires_share_alike=False,
        risk_score=0.8,
        compatible_licenses=[],
    ),
}


class LicenseManager(LicenseManagerInterface):
    """Manages dataset license classification, risk assessment, and compatibility."""

    def __init__(self) -> None:
        self._allowed = set(dc_config.allowed_licenses)
        self._restricted = set(dc_config.restricted_licenses)
        self._known = KNOWN_LICENSES

    def classify_license(self, license_text: str) -> LicenseInfo:
        """Classify a license string into a structured LicenseInfo."""
        text_lower = license_text.strip().lower()

        # Direct lookup by ID or SPDX
        for lid, info in self._known.items():
            if (
                lid == text_lower
                or info.spdx_identifier.lower() == text_lower
                or info.name.lower() == text_lower
            ):
                return info

        # Fuzzy matching
        if "creative commons" in text_lower or "cc" in text_lower:
            if "noncommercial" in text_lower or "nc" in text_lower:
                return self._known["cc_nc"]
            if "zero" in text_lower or "cc0" in text_lower:
                return self._known["cc0"]
            if "attribution" in text_lower:
                if "sharealike" in text_lower or "sa" in text_lower:
                    return self._known["cc_by_sa_4"]
                return self._known["cc_by_4"]
            return self._known["cc_by_4"]

        if "mit" in text_lower:
            return self._known["mit"]
        if "apache" in text_lower:
            return self._known["apache_2"]
        if "public domain" in text_lower:
            return self._known["cc0"]
        if "proprietary" in text_lower or "all rights reserved" in text_lower:
            return self._known["proprietary"]

        return self._known["unknown"]

    def is_allowed(self, license_info: LicenseInfo) -> bool:
        return license_info.license_id in self._allowed

    def is_restricted(self, license_info: LicenseInfo) -> bool:
        return license_info.license_id in self._restricted

    def compute_risk_score(self, license_info: LicenseInfo) -> float:
        return license_info.risk_score

    def get_license_compatibility(
        self, license_a: LicenseInfo, license_b: LicenseInfo
    ) -> float:
        """Return compatibility score (0.0–1.0) between two licenses."""
        if license_a.license_id == license_b.license_id:
            return 1.0

        # Both must be open
        if not license_a.is_open or not license_b.is_open:
            return 0.0

        # Check explicit compatibility
        if license_b.license_id in license_a.compatible_licenses:
            return 0.9
        if license_a.license_id in license_b.compatible_licenses:
            return 0.9

        # Share-alike creates incompatibility with non-SA licenses
        if license_a.requires_share_alike and not license_b.requires_share_alike:
            return 0.3
        if license_b.requires_share_alike and not license_a.requires_share_alike:
            return 0.3

        # Both allow commercial use
        if license_a.allows_commercial and license_b.allows_commercial:
            return 0.7

        return 0.2
