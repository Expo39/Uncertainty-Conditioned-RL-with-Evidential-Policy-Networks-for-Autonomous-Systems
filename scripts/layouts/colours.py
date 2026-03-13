"""
@file colours.py
@brief Centralised colour palette for all parking lot visualisations.

Single source of truth for bay type colours, used by:
  - scripts/layouts/common.py  (generate_layouts PNG output)
  - scripts/inspect_layout.py  (CARLA debug overlay)
  - scripts/visualise_training.py  (live training bird's-eye view)

Colours are defined as hex strings (#RRGGBB). CARLA consumers convert to
carla.Color via hex_to_carla_color(). Matplotlib consumers use the hex strings
directly.
"""

from typing import Tuple

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# Bay types
HEX_PERP_BAY = "#0000DC"       # Blue
HEX_ANGLED_BAY = "#FFD700"     # Yellow
HEX_PARALLEL_BAY = "#B400FF"   # Violet
HEX_TARGET_BAY = "#00FF00"     # Bright green

# Lot features
HEX_PEDESTRIAN_ZONE = "#00CED1"        # Dark turquoise
HEX_PEDESTRIAN_ZONE_EDGE = "#008B8B"   # Dark cyan
HEX_PATROL_PATH = "#DC0000"            # Red
HEX_LOT = "#DDDDDD"                    # Light grey

# Bay type lookup (hex, for matplotlib)
BAY_HEX: dict = {
    "perpendicular": HEX_PERP_BAY,
    "angled": HEX_ANGLED_BAY,
    "parallel": HEX_PARALLEL_BAY,
}


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------


def hex_to_rgb(hex_colour: str) -> Tuple[int, int, int]:
    """
    @brief Convert a hex colour string to an (r, g, b) integer tuple.
    @param hex_colour: Colour string in #RRGGBB format.
    @return Tuple of (r, g, b) each in range 0-255.
    """
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_carla_color(hex_colour: str) -> "carla.Color":  # type: ignore[name-defined]
    """
    @brief Convert a hex colour string to a carla.Color instance.
    @param hex_colour: Colour string in #RRGGBB format.
    @return carla.Color with matching r, g, b values.
    @note Importing carla is deferred so this module is importable outside
          the CARLA container (e.g. during generate-layouts on the host).
    """
    import carla  # noqa: PLC0415

    r, g, b = hex_to_rgb(hex_colour)
    return carla.Color(r=r, g=g, b=b)
