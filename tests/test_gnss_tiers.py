"""
@file test_gnss_tiers.py
@brief Unit tests for the visualiser's GNSS tier presentation loader.

CPU-only. No CARLA, ROS 2, or GPU required. Covers the real profiles YAML as
well as the malformed-input paths, since the viewer must keep rendering when it
is launched from a directory without the config.
"""

from pathlib import Path

import yaml

from scripts.visualise.gnss_tiers import (
    display_label,
    load_gnss_tiers,
    resolve_tier,
    tier_colours_in_order,
    unknown_colour,
)

_PROFILES = Path("configs/deployment/sim/gnss_noise_profiles.yaml")


class TestLoadGnssTiers:
    """
    @class TestLoadGnssTiers
    @brief Loading tier presentation data from YAML.
    """

    def test_loads_repo_profiles(self) -> None:
        """
        @brief The shipped profiles file must yield the four RTK fix states.
        """
        tiers = load_gnss_tiers(_PROFILES)
        assert set(tiers) == {"rtk_fixed", "rtk_float", "standalone", "degraded"}

    def test_severity_follows_declaration_order(self) -> None:
        """
        @brief Severity must rank best fix to worst, driving the colour ramp.
        """
        tiers = load_gnss_tiers(_PROFILES)
        ranked = sorted(tiers.values(), key=lambda t: t.severity)
        assert [t.name for t in ranked] == [
            "rtk_fixed",
            "rtk_float",
            "standalone",
            "degraded",
        ]

    def test_sigma_increases_with_severity(self) -> None:
        """
        @brief A worse fix must carry a larger 1-sigma, since it sets the ring.
        """
        tiers = load_gnss_tiers(_PROFILES)
        sigmas = [t.stddev_m for t in sorted(tiers.values(), key=lambda t: t.severity)]
        assert sigmas == sorted(sigmas)
        assert sigmas[0] < sigmas[-1]

    def test_descriptions_are_populated(self) -> None:
        """
        @brief Every tier needs the plain-English line shown on the panel.
        """
        tiers = load_gnss_tiers(_PROFILES)
        assert all(t.description for t in tiers.values())

    def test_missing_file_returns_empty(self) -> None:
        """
        @brief A missing config must degrade to empty, never raise.
        """
        assert load_gnss_tiers(Path("does/not/exist.yaml")) == {}

    def test_malformed_yaml_returns_empty(self, tmp_path: Path) -> None:
        """
        @brief Unparseable YAML must degrade to empty, never raise.
        """
        bad = tmp_path / "bad.yaml"
        bad.write_text("tiers: [unclosed", encoding="utf-8")
        assert load_gnss_tiers(bad) == {}

    def test_non_mapping_tiers_returns_empty(self, tmp_path: Path) -> None:
        """
        @brief A tiers key that is not a mapping must yield no tiers.
        """
        odd = tmp_path / "odd.yaml"
        odd.write_text(yaml.safe_dump({"tiers": ["a", "b"]}), encoding="utf-8")
        assert load_gnss_tiers(odd) == {}

    def test_missing_fields_fall_back(self, tmp_path: Path) -> None:
        """
        @brief A tier lacking sigma/description must still load with defaults.
        """
        sparse = tmp_path / "sparse.yaml"
        sparse.write_text(yaml.safe_dump({"tiers": {"odd": {}}}), encoding="utf-8")
        tiers = load_gnss_tiers(sparse)
        assert tiers["odd"].stddev_m == 0.0
        assert tiers["odd"].description == ""

    def test_extra_tiers_reuse_worst_colour(self, tmp_path: Path) -> None:
        """
        @brief More tiers than ramp entries must not raise on the colour lookup.
        """
        many = tmp_path / "many.yaml"
        many.write_text(
            yaml.safe_dump({"tiers": {f"t{i}": {} for i in range(7)}}),
            encoding="utf-8",
        )
        tiers = load_gnss_tiers(many)
        assert len(tiers) == 7
        assert tiers["t6"].colour == tiers["t3"].colour


class TestTierLookup:
    """
    @class TestTierLookup
    @brief Resolving a frame's tier name and rendering its label.
    """

    def test_resolve_known_tier(self) -> None:
        """
        @brief A known name resolves to the matching tier.
        """
        tiers = load_gnss_tiers(_PROFILES)
        assert resolve_tier(tiers, "degraded") is tiers["degraded"]

    def test_resolve_unknown_and_empty(self) -> None:
        """
        @brief Unknown and empty names must resolve to None, not raise.
        """
        tiers = load_gnss_tiers(_PROFILES)
        assert resolve_tier(tiers, "nonesuch") is None
        assert resolve_tier(tiers, "") is None

    def test_display_label_formats_name(self) -> None:
        """
        @brief The panel label upper-cases and de-underscores the tier key.
        """
        tiers = load_gnss_tiers(_PROFILES)
        assert display_label(tiers["rtk_fixed"], "rtk_fixed") == "RTK FIXED"

    def test_display_label_without_tier(self) -> None:
        """
        @brief An unknown name still renders; a blank one says so explicitly.
        """
        assert display_label(None, "mystery") == "MYSTERY"
        assert display_label(None, "") == "NO FIX DATA"

    def test_colours_are_distinct_per_severity(self) -> None:
        """
        @brief Adjacent tiers must differ in colour or the ramp conveys nothing.
        """
        colours = tier_colours_in_order(load_gnss_tiers(_PROFILES))
        assert len(set(colours)) == len(colours)

    def test_unknown_colour_is_neutral(self) -> None:
        """
        @brief The fallback colour must not collide with a real tier colour.
        """
        assert unknown_colour() not in tier_colours_in_order(load_gnss_tiers(_PROFILES))
