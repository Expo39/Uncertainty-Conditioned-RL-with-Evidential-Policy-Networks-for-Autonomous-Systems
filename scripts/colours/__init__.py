"""
@file __init__.py
@brief Centralised colour palette for all parking lot and sensor visualisations.

Single source of truth for all visualisation colours, used by:
  - scripts/layouts/common.py          (generate_layouts PNG output)
  - scripts/inspect/lot_inspector.py   (CARLA lot + sensor debug overlay)
  - scripts/visualise/visualiser.py    (live training bird's-eye view)

Bay and lot colours are defined as hex strings (#RRGGBB). CARLA consumers convert
to carla.Color via hex_to_carla_color(). Matplotlib consumers use hex strings directly.

Sensor overlay colours are defined as carla.Color instances (CARLA-only consumers).
The carla import is deferred so this module remains importable on the host (e.g.
during make generate-layouts) where the carla package is not installed.
"""

from typing import Tuple

# ---------------------------------------------------------------------------
# Bay and lot feature colours (hex, matplotlib-compatible)
# ---------------------------------------------------------------------------

HEX_PERP_BAY = "#0000DC"  # Blue
HEX_ANGLED_BAY = "#FFD700"  # Yellow
HEX_PARALLEL_BAY = "#B400FF"  # Violet
HEX_TARGET_BAY = "#00FF00"  # Bright green

HEX_PEDESTRIAN_ZONE = "#00CED1"  # Dark turquoise
HEX_PEDESTRIAN_ZONE_EDGE = "#008B8B"  # Dark cyan
HEX_PATROL_PATH = "#DC0000"  # Red
HEX_LOT = "#DDDDDD"  # Light grey

# Visualiser actor colours (used by visualise_training.py)
HEX_EGO = "#00CFFF"  # Cyan (ego vehicle + trajectory trail)
HEX_STATIC_VEHICLE = "#FF9000"  # Orange (parked NPC vehicles)
HEX_PATROL_VEHICLE = "#FF3030"  # Red (moving patrol NPC)
HEX_CONE = "#FF6600"  # Orange-red (perimeter cones)

# Bay type lookup (hex, for matplotlib)
BAY_HEX: dict = {
    "perpendicular": HEX_PERP_BAY,
    "angled": HEX_ANGLED_BAY,
    "parallel": HEX_PARALLEL_BAY,
}

# ---------------------------------------------------------------------------
# Sensor overlay colours (hex, used by inspect_sensors.py via hex_to_carla_color)
# ---------------------------------------------------------------------------

HEX_SENSOR_IMU = "#FFDC00"  # Yellow
HEX_SENSOR_LIDAR_2D = "#00B4FF"  # Cyan (sensor mount dot)
HEX_SENSOR_LIDAR_3D = "#00FF50"  # Green (sensor mount dot)
HEX_SENSOR_CAMERA = "#FF0080"  # Hot pink/orange (g=0 avoids CARLA yellow shift)
HEX_SENSOR_FOV_LIDAR = "#FF0000"  # Pure red (LiDAR FOV arc, 2D and 3D)
HEX_SENSOR_FOV_BLIND = "#505050"  # Dark grey (LiDAR blind sector arc)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def hex_to_rgb(hex_colour: str) -> Tuple[int, int, int]:
    """
    @brief Convert a hex colour string to an (r, g, b) integer tuple.
    @param hex_colour: Colour string in #RRGGBB format.
    @return Tuple of (r, g, b) each in range 0-255.
    """
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_carla_color(
    hex_colour: str,
) -> "carla.Color":  # type: ignore[name-defined]  # noqa: F821
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
