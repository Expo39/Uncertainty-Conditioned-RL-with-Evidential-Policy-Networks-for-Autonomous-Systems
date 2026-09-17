"""
@file gnss_tiers.py
@brief Load GNSS fix-state tier presentation data for the visualiser.

Reads tier sigma and description from gnss_noise_profiles.yaml once, so
adding a tier to the YAML is enough to make it render. Severity is
declaration order (best fix to worst), driving the colour ramp.
"""

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import yaml

_DEFAULT_PROFILE_PATH = Path("configs/deployment/sim/gnss_noise_profiles.yaml")

# Green -> amber -> orange -> red, indexed by severity rank. A tier beyond the
# last entry reuses the final (worst) colour.
_SEVERITY_COLOURS: Tuple[Tuple[int, int, int], ...] = (
    (0, 200, 83),
    (255, 193, 7),
    (255, 112, 0),
    (229, 57, 53),
)

# Shown when a frame carries a tier name absent from the YAML, so an unknown
# tier degrades to a readable neutral panel instead of crashing the viewer.
_UNKNOWN_COLOUR = (150, 150, 150)


class GnssTier(NamedTuple):
    """
    @class GnssTier
    @brief Presentation data for one GNSS fix-state tier.
    """

    name: str
    stddev_m: float
    description: str
    colour: Tuple[int, int, int]
    severity: int


def _format_label(name: str) -> str:
    """
    @brief Render a YAML tier key as a display label.
    @param name: Tier key, e.g. "rtk_fixed".
    @return Upper-case label with underscores as spaces, e.g. "RTK FIXED".
    """
    return name.replace("_", " ").upper()


def load_gnss_tiers(path: Optional[Path] = None) -> Dict[str, GnssTier]:
    """
    @brief Load tier presentation data keyed by tier name.

    Missing or malformed YAML yields an empty mapping rather than raising: the
    viewer must keep running (showing the bare tier name) when it is launched
    from a directory without the config.

    @param path: Profile YAML path. Defaults to the repo-relative profiles file.
    @return Mapping of tier name to GnssTier, in declaration order.
    """
    profile_path = path or _DEFAULT_PROFILE_PATH
    try:
        with open(profile_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}

    tiers_raw = raw.get("tiers", {})
    if not isinstance(tiers_raw, dict):
        return {}

    tiers: Dict[str, GnssTier] = {}
    for severity, (name, spec) in enumerate(tiers_raw.items()):
        if not isinstance(spec, dict):
            continue
        colour_idx = min(severity, len(_SEVERITY_COLOURS) - 1)
        tiers[str(name)] = GnssTier(
            name=str(name),
            stddev_m=float(spec.get("metric_stddev_m", 0.0)),
            description=str(spec.get("description", "")),
            colour=_SEVERITY_COLOURS[colour_idx],
            severity=severity,
        )
    return tiers


def tier_colours_in_order(tiers: Dict[str, GnssTier]) -> List[Tuple[int, int, int]]:
    """
    @brief List tier colours from best to worst fix, for a legend or ramp.
    @param tiers: Mapping from load_gnss_tiers.
    @return Colours ordered by severity.
    """
    return [t.colour for t in sorted(tiers.values(), key=lambda t: t.severity)]


def resolve_tier(tiers: Dict[str, GnssTier], name: str) -> Optional[GnssTier]:
    """
    @brief Look up a tier by the name carried in a visualiser frame.
    @param tiers: Mapping from load_gnss_tiers.
    @param name: Tier name from the frame; may be empty or unknown.
    @return The matching GnssTier, or None when the name is empty/unknown.
    """
    return tiers.get(name) if name else None


def unknown_colour() -> Tuple[int, int, int]:
    """@brief Neutral panel colour for an unrecognised tier name."""
    return _UNKNOWN_COLOUR


def display_label(tier: Optional[GnssTier], raw_name: str) -> str:
    """
    @brief Build the on-screen tier label.
    @param tier: Resolved tier, or None when unknown.
    @param raw_name: Tier name as it appeared in the frame.
    @return Display label; "NO FIX DATA" when the frame carried no tier.
    """
    if tier is not None:
        return _format_label(tier.name)
    return _format_label(raw_name) if raw_name else "NO FIX DATA"
