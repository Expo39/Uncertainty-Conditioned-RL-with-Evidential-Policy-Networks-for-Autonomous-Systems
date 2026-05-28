"""
@file builder.py
@brief Declarative DSL for parking lot floor plan layouts.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from uncertainty_rl.utils.geometry import point_in_polygon

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Bay-dimension constants (German EAR 05) and structural defaults
# ---------------------------------------------------------------------------


BAY_DIMS: Dict[str, Dict[str, float]] = {
    "perpendicular": {"width": 2.75, "depth": 5.25, "aisle": 6.0},
    "angled": {"width": 2.75, "depth": 5.65, "aisle": 3.6},
    "parallel": {"width": 2.75, "depth": 6.25, "aisle": 4.0},
}

# Reference target for bay counts (5 per type per layout where geometry permits).
BAYS_PER_TYPE: int = 5

# Minimum clearance between any bay corner and the perimeter wall/cones.
WALL_GAP: float = 0.5

# Default width and end-margin of a pedestrian strip alongside an aisle face.
PED_STRIP: float = 3.0
PED_MARGIN: float = 0.5


# ---------------------------------------------------------------------------
# Bay rectangle primitive (used by validators, BayGroup.bbox, and warn helpers)
# ---------------------------------------------------------------------------


def _bay_corners(
    cx: float,
    cy: float,
    yaw_deg: float,
    width: float,
    depth: float,
) -> List[Point]:
    """
    @brief Return the four corner points of a bay rectangle in local frame.

    The bay rectangle has depth along the vehicle's heading axis and width
    perpendicular to it. Corners are ordered CCW starting from (-d/2, -w/2).
    """
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    hw = width / 2.0
    hd = depth / 2.0
    local = [(-hd, -hw), (hd, -hw), (hd, hw), (-hd, hw)]
    return [
        (cx + cos_y * lx - sin_y * ly, cy + sin_y * lx + cos_y * ly) for lx, ly in local
    ]


def _yaw_to_normal(yaw_deg: float) -> Point:
    """
    @brief Convert a bay yaw to its nose-direction unit vector.
    """
    rad = math.radians(yaw_deg)
    return (math.cos(rad), math.sin(rad))


# ---------------------------------------------------------------------------
# Validators (called automatically inside LotBuilder.build())
# ---------------------------------------------------------------------------


def validate_bays_in_polygon(
    bays: List[Dict[str, Any]],
    corners: List[Dict[str, float]],
    shape_name: str,
    margin: float = 0.05,
) -> None:
    """
    @brief Raise ValueError if any bay corner lies outside the lot polygon.

    @param bays: Bay dicts with local_x, local_y, local_yaw_deg, width, depth.
    @param corners: Lot perimeter as list of {x, y} dicts (local frame).
    @param shape_name: Floor plan name for error messages.
    @param margin: Outward expansion of polygon for boundary-touching bays
                   (allows for floating-point tolerance on flush bays).
    @raises ValueError: If any bay corner is outside the expanded polygon.
    """
    poly = [(c["x"], c["y"]) for c in corners]
    n = len(poly)
    cx_avg = sum(p[0] for p in poly) / n
    cy_avg = sum(p[1] for p in poly) / n
    scale = 1.0 + margin
    expanded = [
        (cx_avg + scale * (px - cx_avg), cy_avg + scale * (py - cy_avg))
        for px, py in poly
    ]
    for bay in bays:
        for corner in _bay_corners(
            bay["local_x"],
            bay["local_y"],
            bay["local_yaw_deg"],
            bay["width"],
            bay["depth"],
        ):
            if not point_in_polygon(corner[0], corner[1], expanded):
                raise ValueError(
                    f"[{shape_name}] Bay '{bay.get('bay_type', '?')}' at "
                    f"({bay['local_x']:.2f}, {bay['local_y']:.2f}) has a corner "
                    f"at ({corner[0]:.2f}, {corner[1]:.2f}) outside the lot "
                    "boundary."
                )


def warn_narrow_corridors(
    bays: List[Dict[str, Any]],
    shape_name: str,
    min_width: float = 6.0,
) -> None:
    """
    @brief Warn if the minimum gap between any two facing bay clusters is
           narrower than min_width (EAR 05 minimum corridor = 6.0 m).
    """
    for i in range(len(bays)):
        for j in range(i + 1, len(bays)):
            a = bays[i]
            b = bays[j]
            same_type = a.get("bay_type") == b.get("bay_type")
            yaw_diff = abs(a["local_yaw_deg"] - b["local_yaw_deg"]) % 360.0
            same_yaw = yaw_diff < 1.0 or abs(yaw_diff - 360.0) < 1.0
            if same_type and same_yaw:
                continue
            gap_x = (
                abs(a["local_x"] - b["local_x"]) - a["depth"] / 2.0 - b["depth"] / 2.0
            )
            gap_y = (
                abs(a["local_y"] - b["local_y"]) - a["width"] / 2.0 - b["width"] / 2.0
            )
            gap = max(gap_x, gap_y, 0.0)
            if gap < min_width:
                print(
                    f"  WARNING [{shape_name}]: corridor between "
                    f"'{a.get('bay_type', '?')}'"
                    f" ({a['local_x']:.1f}, {a['local_y']:.1f}) "
                    f"and '{b.get('bay_type', '?')}'"
                    f" ({b['local_x']:.1f}, {b['local_y']:.1f}) "
                    f"is {gap:.2f} m (min {min_width:.1f} m)."
                )


# ---------------------------------------------------------------------------
# BayGroup - a placed row of bays plus geometry summary
# ---------------------------------------------------------------------------


class BayGroup:
    """
    @class BayGroup
    @brief A placed row of bays with face/bbox metadata.

    Returned by LotBuilder.row(), row_centred(), row_along_wall(), place_bay().
    Passed into PedestrianZone and PatrolPath helpers so zones and waypoints
    can be specified relative to the row's faces rather than as raw coordinates.
    """

    def __init__(
        self,
        bay_type: str,
        bays: List[Dict[str, Any]],
        direction: Point,
        normal: Point,
        bay_width: float,
        bay_depth: float,
        spacing: float,
    ) -> None:
        self.bay_type = bay_type
        self.bays = bays
        self.direction = direction
        self.normal = normal
        self.bay_width = bay_width
        self.bay_depth = bay_depth
        self.spacing = spacing

    @property
    def count(self) -> int:
        """@brief Number of bays in the group."""
        return len(self.bays)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """
        @brief Axis-aligned bounding box of all bay rectangles.
        @return (x_min, x_max, y_min, y_max).
        """
        all_corners = [
            pt
            for bay in self.bays
            for pt in _bay_corners(
                bay["local_x"],
                bay["local_y"],
                bay["local_yaw_deg"],
                bay["width"],
                bay["depth"],
            )
        ]
        xs = [p[0] for p in all_corners]
        ys = [p[1] for p in all_corners]
        return (min(xs), max(xs), min(ys), max(ys))

    def _is_axis_aligned(self) -> bool:
        """@brief True if every bay yaw is a multiple of 90 degrees."""
        return all(
            abs(((bay["local_yaw_deg"] + 0.5) % 90.0) - 0.5) <= 0.01
            for bay in self.bays
        )

    @property
    def nose_y(self) -> float:
        """
        @brief y-coordinate of the row's nose face.
        @raises ValueError: If nose normal is not along the y-axis.
        """
        nx, ny = self.normal
        if abs(nx) > 1e-6 or abs(abs(ny) - 1.0) > 1e-6:
            raise ValueError(
                "nose_y is only defined when nose normal is along the y-axis."
            )
        cy = self.bays[0]["local_y"]
        return cy + ny * self.bay_depth / 2.0

    @property
    def back_y(self) -> float:
        """
        @brief y-coordinate of the row's back face.
        @raises ValueError: If nose normal is not along the y-axis.
        """
        nx, ny = self.normal
        if abs(nx) > 1e-6 or abs(abs(ny) - 1.0) > 1e-6:
            raise ValueError(
                "back_y is only defined when nose normal is along the y-axis."
            )
        cy = self.bays[0]["local_y"]
        return cy - ny * self.bay_depth / 2.0

    @property
    def nose_x(self) -> float:
        """
        @brief x-coordinate of the row's nose face.
        @raises ValueError: If nose normal is not along the x-axis.
        """
        nx, ny = self.normal
        if abs(ny) > 1e-6 or abs(abs(nx) - 1.0) > 1e-6:
            raise ValueError(
                "nose_x is only defined when nose normal is along the x-axis."
            )
        cx = self.bays[0]["local_x"]
        return cx + nx * self.bay_depth / 2.0

    @property
    def back_x(self) -> float:
        """
        @brief x-coordinate of the row's back face.
        @raises ValueError: If nose normal is not along the x-axis.
        """
        nx, ny = self.normal
        if abs(ny) > 1e-6 or abs(abs(nx) - 1.0) > 1e-6:
            raise ValueError(
                "back_x is only defined when nose normal is along the x-axis."
            )
        cx = self.bays[0]["local_x"]
        return cx - nx * self.bay_depth / 2.0


# ---------------------------------------------------------------------------
# PedestrianZone - axis-aligned strip in local frame
# ---------------------------------------------------------------------------


class PedestrianZone:
    """
    @class PedestrianZone
    @brief Axis-aligned pedestrian strip (x_min/x_max/y_min/y_max).

    Construct directly with explicit bounds, or use the classmethod helpers
    (along_row, between_rows, beside_wall) to derive bounds from a BayGroup
    or a wall reference.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"PedestrianZone has zero or negative extent: "
                f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]"
            )
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def to_dict(self) -> Dict[str, float]:
        """@brief Convert to the dict shape consumed by common.to_world_frame."""
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

    @classmethod
    def along_row(
        cls,
        group: BayGroup,
        side: str,
        strip: float = PED_STRIP,
        margin: float = PED_MARGIN,
    ) -> "PedestrianZone":
        """
        @brief Strip alongside one face of a row's bbox.

        Works for any row (axis-aligned or angled). The strip is always
        axis-aligned (matching the YAML schema); for angled rows it hugs the
        bbox edge in the requested cardinal direction.

        @param group: Source bay group.
        @param side: One of "north", "south", "east", "west" (cardinal -
                     bbox-aligned), or "nose", "back", "left", "right"
                     (relative to the row's nose direction). The relative names
                     require an axis-aligned row.
        @param strip: Strip width (m) outward from the chosen face.
        @param margin: Inset applied to each end of the strip.
        """
        x_min, x_max, y_min, y_max = group.bbox
        cardinal = side
        if side in ("nose", "back", "left", "right"):
            if not group._is_axis_aligned():
                raise ValueError(
                    f"PedestrianZone.along_row(side='{side}') requires an "
                    "axis-aligned row; use side='north'|'south'|'east'|'west' "
                    "for angled rows."
                )
            cardinal = _row_side_to_cardinal(group, side)
        if cardinal == "north":
            return cls(x_min + margin, x_max - margin, y_max + margin, y_max + strip)
        if cardinal == "south":
            return cls(x_min + margin, x_max - margin, y_min - strip, y_min - margin)
        if cardinal == "east":
            return cls(x_max + margin, x_max + strip, y_min + margin, y_max - margin)
        if cardinal == "west":
            return cls(x_min - strip, x_min - margin, y_min + margin, y_max - margin)
        raise ValueError(
            f"Unknown side '{side}' (expected nose/back/left/right or "
            "north/south/east/west)."
        )

    @classmethod
    def between_rows(
        cls,
        group_a: BayGroup,
        group_b: BayGroup,
        margin: float = PED_MARGIN,
    ) -> "PedestrianZone":
        """
        @brief Strip in the gap between two facing rows (typically back-to-back).

        Auto-detects whether the rows are stacked vertically or horizontally
        based on bbox overlap. Works for any pair of rows whose bboxes overlap
        in one axis.
        """
        ax_min, ax_max, ay_min, ay_max = group_a.bbox
        bx_min, bx_max, by_min, by_max = group_b.bbox
        x_overlap_min = max(ax_min, bx_min)
        x_overlap_max = min(ax_max, bx_max)
        y_overlap_min = max(ay_min, by_min)
        y_overlap_max = min(ay_max, by_max)
        if x_overlap_max - x_overlap_min > y_overlap_max - y_overlap_min:
            # Rows are stacked vertically (gap is along y).
            y_low = min(ay_max, by_max)
            y_high = max(ay_min, by_min)
            return cls(
                x_overlap_min + margin,
                x_overlap_max - margin,
                y_low + margin,
                y_high - margin,
            )
        # Rows are stacked horizontally (gap is along x).
        x_low = min(ax_max, bx_max)
        x_high = max(ax_min, bx_min)
        return cls(
            x_low + margin,
            x_high - margin,
            y_overlap_min + margin,
            y_overlap_max - margin,
        )

    @classmethod
    def beside_wall(
        cls,
        wall_p0: Point,
        wall_p1: Point,
        strip: float = PED_STRIP,
        inward: bool = True,
        along_range: Optional[Tuple[float, float]] = None,
        margin: float = PED_MARGIN,
    ) -> "PedestrianZone":
        """
        @brief Strip along an axis-aligned wall segment.

        @param wall_p0: Wall start point.
        @param wall_p1: Wall end point (must share x or y with wall_p0).
        @param strip: Strip width (m).
        @param inward: True places the strip on the inward side; False, outward.
        @param along_range: Optional (start, end) clip in absolute coordinates
                            along the wall direction.
        @param margin: Inset applied to each end of the strip.
        @note Only axis-aligned walls are supported; sloped walls require
              a hand-built PedestrianZone with explicit bounds.
        """
        x0, y0 = wall_p0
        x1, y1 = wall_p1
        sign = 1.0 if inward else -1.0
        if abs(x0 - x1) < 1e-6:
            wall_x = x0
            y_lo, y_hi = sorted((y0, y1))
            if along_range is not None:
                y_lo = max(y_lo, along_range[0])
                y_hi = min(y_hi, along_range[1])
            return cls(
                wall_x + margin if sign > 0 else wall_x - strip,
                wall_x + strip if sign > 0 else wall_x - margin,
                y_lo + margin,
                y_hi - margin,
            )
        if abs(y0 - y1) < 1e-6:
            wall_y = y0
            x_lo, x_hi = sorted((x0, x1))
            if along_range is not None:
                x_lo = max(x_lo, along_range[0])
                x_hi = min(x_hi, along_range[1])
            return cls(
                x_lo + margin,
                x_hi - margin,
                wall_y + margin if sign > 0 else wall_y - strip,
                wall_y + strip if sign > 0 else wall_y - margin,
            )
        raise ValueError("PedestrianZone.beside_wall only supports axis-aligned walls.")


def _row_side_to_cardinal(group: BayGroup, side: str) -> str:
    """
    @brief Map a row-relative side ("nose"/"back"/"left"/"right") to a cardinal
           direction ("north"/"south"/"east"/"west") using the row's normal +
           direction unit vectors.
    """
    nx, ny = group.normal
    dx, dy = group.direction
    if side == "nose":
        return _vec_to_cardinal(nx, ny)
    if side == "back":
        return _vec_to_cardinal(-nx, -ny)
    if side == "left":
        # "Left" of the direction of travel: 90 deg CCW from direction.
        return _vec_to_cardinal(-dy, dx)
    if side == "right":
        return _vec_to_cardinal(dy, -dx)
    raise ValueError(f"Unknown row side '{side}'.")


def _vec_to_cardinal(vx: float, vy: float) -> str:
    """@brief Closest cardinal direction to a 2D vector."""
    if abs(vx) > abs(vy):
        return "east" if vx > 0 else "west"
    return "north" if vy > 0 else "south"


# ---------------------------------------------------------------------------
# PatrolPath - waypoint container with face-midpoint helpers
# ---------------------------------------------------------------------------


# Edge type accepted by aisle_x / aisle_y. Either a single BayGroup, a list
# of them (for clusters of multiple rows), or a raw float coordinate (acting
# as a virtual axis-aligned edge).
Edge = Any  # Union[BayGroup, List[BayGroup], float]


def _edge_y_max(side: Edge) -> float:
    """@brief Highest y of any bay corner in `side` (or float as raw y)."""
    if isinstance(side, BayGroup):
        return side.bbox[3]
    if isinstance(side, list):
        return max(_edge_y_max(s) for s in side)
    return float(side)


def _edge_y_min(side: Edge) -> float:
    """@brief Lowest y of any bay corner in `side` (or float as raw y)."""
    if isinstance(side, BayGroup):
        return side.bbox[2]
    if isinstance(side, list):
        return min(_edge_y_min(s) for s in side)
    return float(side)


def _edge_x_max(side: Edge) -> float:
    """@brief Rightmost x of any bay corner in `side` (or float as raw x)."""
    if isinstance(side, BayGroup):
        return side.bbox[1]
    if isinstance(side, list):
        return max(_edge_x_max(s) for s in side)
    return float(side)


def _edge_x_min(side: Edge) -> float:
    """@brief Leftmost x of any bay corner in `side` (or float as raw x)."""
    if isinstance(side, BayGroup):
        return side.bbox[0]
    if isinstance(side, list):
        return min(_edge_x_min(s) for s in side)
    return float(side)


class PatrolPath:
    """
    @class PatrolPath
    @brief Ordered list of patrol waypoints in local frame.

    Waypoint order is the user's responsibility. Helper methods compute
    coordinates from BayGroup faces so the user does not need to extract
    aisle midpoints or diagonal endpoints by hand.

    Each helper accepts BayGroup, list of BayGroups, or a raw float for the
    "side" arguments. A raw float is treated as a virtual axis-aligned edge
    at that coordinate (useful for "midpoint to a virtual line at x=0",
    common when patrol corridors are referenced against a notional internal
    boundary rather than a physical wall).
    """

    def __init__(self) -> None:
        self.waypoints: List[Point] = []

    def add(self, x: float, y: float) -> "PatrolPath":
        """@brief Append a raw (x, y) waypoint. Returns self for chaining."""
        self.waypoints.append((float(x), float(y)))
        return self

    def aisle_y(self, below: Edge, above: Edge) -> float:
        """
        @brief Y-midpoint of the aisle between a lower stack and upper stack.
        @param below: Bay group(s) on the south side of the aisle, or a raw y.
        @param above: Bay group(s) on the north side of the aisle, or a raw y.
        @return Midpoint y between the upper edge of `below` and the lower
                edge of `above`.
        """
        return (_edge_y_max(below) + _edge_y_min(above)) / 2.0

    def aisle_x(self, left: Edge, right: Edge) -> float:
        """
        @brief X-midpoint of the aisle between a left stack and right stack.
        @param left: Bay group(s) on the west side of the aisle, or a raw x.
        @param right: Bay group(s) on the east side of the aisle, or a raw x.
        @return Midpoint x between the right edge of `left` and the left
                edge of `right`.
        """
        return (_edge_x_max(left) + _edge_x_min(right)) / 2.0

    def add_diag_from_prev(
        self,
        x_direction: str,
        target_y: float,
    ) -> "PatrolPath":
        """
        @brief Append a waypoint connected to the previous one by a 45-deg line.


        @param x_direction: "left" (decreasing x) or "right" (increasing x).
        @param target_y: y of the new waypoint.
        @raises ValueError: If the path is empty.
        """
        if not self.waypoints:
            raise ValueError(
                "add_diag_from_prev requires at least one previous waypoint."
            )
        prev_x, prev_y = self.waypoints[-1]
        dy = target_y - prev_y
        if x_direction == "left":
            dx = -abs(dy)
        elif x_direction == "right":
            dx = abs(dy)
        else:
            raise ValueError(
                f"x_direction must be 'left' or 'right', got '{x_direction}'."
            )
        return self.add(prev_x + dx, target_y)

    def to_list(self) -> List[Dict[str, float]]:
        """@brief Convert to the list-of-dicts shape expected by to_world_frame."""
        return [{"x": x, "y": y} for x, y in self.waypoints]


# ---------------------------------------------------------------------------
# LotBuilder
# ---------------------------------------------------------------------------


# Direction names map to unit vectors. "east"/"west" run along +x/-x, etc.
_DIR_VECTORS: Dict[str, Point] = {
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
}

# Default yaw pairs for back-to-back rows in row_pair_back_to_back.
# When direction runs along x (east/west), the rows stack along y so they face
# along y too: (row_a_yaw, row_b_yaw) = (90, 270). When direction runs along y
# (north/south), the rows stack along x: (yaw_a, yaw_b) = (0, 180).
_DEFAULT_BACK_TO_BACK_YAWS: Dict[str, Tuple[float, float]] = {
    "east": (90.0, 270.0),
    "west": (90.0, 270.0),
    "north": (0.0, 180.0),
    "south": (0.0, 180.0),
}


def _check_direction_and_type(
    direction: str,
    bay_type: str,
    dims: Dict[str, Dict[str, float]],
) -> None:
    """
    @brief Raise ValueError if direction or bay_type are not recognised.
    @param direction: Direction string to validate.
    @param bay_type: Bay type string to validate.
    @param dims: Bay dimensions dict (typically LotBuilder.dims).
    """
    if direction not in _DIR_VECTORS:
        raise ValueError(
            f"direction must be one of {list(_DIR_VECTORS)}, got '{direction}'."
        )
    if bay_type not in dims:
        raise ValueError(f"Unknown bay_type '{bay_type}'.")


def angled_corner_clearance(lot: "LotBuilder", angle_deg: float = 45.0) -> float:
    """
    @brief Default along-wall clearance for an angled bay's leading corner.

    Returns the distance from the wall start point to the first bay centre
    such that the nearest bay corner clears the wall end by wall_gap.

    @param lot: LotBuilder instance (provides dims and wall_gap).
    @param angle_deg: Bay angle relative to the inward wall normal (degrees).
    @return Clearance distance in metres.
    """
    dims_ang = lot.dims["angled"]
    return (dims_ang["depth"] / 2.0 + dims_ang["width"] / 2.0) * math.cos(
        math.radians(angle_deg)
    ) + lot.wall_gap


class LotBuilder:
    """
    @class LotBuilder
    @brief Top-level builder for a parking lot floor plan.

    Use one builder per layout. Add bays via row(), row_centred(),
    row_pair_back_to_back(), row_along_wall(), place_bay(); spawns via spawn();
    pedestrian zones via add_zone(); a patrol path via set_patrol(); obstacles
    via add_obstacle(). Call build() at the end to validate and return the
    layout dict.
    """

    def __init__(
        self,
        name: str,
        corners: Sequence[Point],
        wall_gap: float = WALL_GAP,
    ) -> None:
        """
        @param name: Floor plan name (used in validator messages).
        @param corners: Lot perimeter polygon as a sequence of (x, y) points (CCW).
        @param wall_gap: Minimum clearance from any bay corner to the perimeter.
        """
        self.name = name
        self.corners: List[Dict[str, float]] = [
            {"x": float(x), "y": float(y)} for x, y in corners
        ]
        self.wall_gap = wall_gap
        self.dims = BAY_DIMS

        self._groups: List[BayGroup] = []
        self._spawns: List[Dict[str, float]] = []
        self._primary_spawn: Optional[Dict[str, float]] = None
        self._zones: List[PedestrianZone] = []
        self._patrol: Optional[PatrolPath] = None
        self._obstacles: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Lot polygon helpers
    # ------------------------------------------------------------------

    def wall_y(self, wall: int) -> float:
        """
        @brief Y-coordinate of an axis-aligned horizontal perimeter wall.
        @param wall: Wall index (0..len(corners)-1).
        @raises ValueError: If the wall is not horizontal.
        """
        p0 = self.corners[wall]
        p1 = self.corners[(wall + 1) % len(self.corners)]
        if abs(p0["y"] - p1["y"]) > 1e-6:
            raise ValueError(f"wall_y({wall}) is undefined for a non-horizontal wall.")
        return p0["y"]

    def wall_x(self, wall: int) -> float:
        """
        @brief X-coordinate of an axis-aligned vertical perimeter wall.
        @param wall: Wall index (0..len(corners)-1).
        @raises ValueError: If the wall is not vertical.
        """
        p0 = self.corners[wall]
        p1 = self.corners[(wall + 1) % len(self.corners)]
        if abs(p0["x"] - p1["x"]) > 1e-6:
            raise ValueError(f"wall_x({wall}) is undefined for a non-vertical wall.")
        return p0["x"]

    # ------------------------------------------------------------------
    # Bay placement primitives
    # ------------------------------------------------------------------

    def row(
        self,
        bay_type: str,
        n: int,
        anchor: Point,
        direction: str,
        yaw_deg: float,
        spacing: Optional[float] = None,
        bay_extras: Optional[Dict[str, Any]] = None,
    ) -> BayGroup:
        """
        @brief Place an axis-aligned row of n bays starting at anchor.
        @param bay_type: One of perpendicular, angled, parallel.
        @param n: Number of bays.
        @param anchor: Centre of the first bay (local frame).
        @param direction: "east"/"west"/"north"/"south" - axis bays advance along.
        @param yaw_deg: Bay heading (vehicle nose direction), degrees.
        @param spacing: Centre-to-centre spacing (m). Defaults to bay width.
        @param bay_extras: Extra fields merged into every bay dict.
        """
        _check_direction_and_type(direction, bay_type, self.dims)
        dx, dy = _DIR_VECTORS[direction]
        bay_w = self.dims[bay_type]["width"]
        bay_d = self.dims[bay_type]["depth"]
        if spacing is None:
            spacing = bay_w
        extras = bay_extras or {}
        step_x = dx * spacing
        step_y = dy * spacing
        ax, ay = anchor
        yaw_norm = yaw_deg % 360.0
        bays: List[Dict[str, Any]] = [
            {
                "bay_type": bay_type,
                "local_x": ax + step_x * i,
                "local_y": ay + step_y * i,
                "local_yaw_deg": yaw_norm,
                "width": bay_w,
                "depth": bay_d,
                **extras,
            }
            for i in range(n)
        ]
        group = BayGroup(
            bay_type=bay_type,
            bays=bays,
            direction=(dx, dy),
            normal=_yaw_to_normal(yaw_deg),
            bay_width=bay_w,
            bay_depth=bay_d,
            spacing=spacing,
        )
        self._groups.append(group)
        return group

    def row_centred(
        self,
        bay_type: str,
        n: int,
        centre: Point,
        direction: str,
        yaw_deg: float,
        spacing: Optional[float] = None,
        bay_extras: Optional[Dict[str, Any]] = None,
    ) -> BayGroup:
        """
        @brief Place an axis-aligned row of n bays centred at `centre`.

        Equivalent to calling row() with the anchor pre-shifted by -(n-1)/2
        along direction so the cluster's mid-point lands at `centre`. Use this
        whenever the natural reference for the cluster is its centre rather
        than its leftmost/topmost bay.
        """
        _check_direction_and_type(direction, bay_type, self.dims)
        dx, dy = _DIR_VECTORS[direction]
        bay_w = self.dims[bay_type]["width"]
        if spacing is None:
            spacing = bay_w
        offset = (n - 1) / 2.0 * spacing
        cx, cy = centre
        anchor = (cx - dx * offset, cy - dy * offset)
        return self.row(
            bay_type=bay_type,
            n=n,
            anchor=anchor,
            direction=direction,
            yaw_deg=yaw_deg,
            spacing=spacing,
            bay_extras=bay_extras,
        )

    def row_pair_back_to_back(
        self,
        bay_type: str,
        n: int,
        centre: Point,
        direction: str,
        gap: float,
        yaws: Optional[Tuple[float, float]] = None,
        spacing: Optional[float] = None,
        bay_extras: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BayGroup, BayGroup]:
        """
        @brief Place two facing rows of n bays each, back-to-back across `gap`.

        @param centre: Midpoint of the cluster (cluster centre x along direction
                       axis, mid-aisle y along the perpendicular axis).
        @param direction: Axis the rows extend along ("east"/"west" stacks them
                          vertically; "north"/"south" stacks horizontally).
        @param gap: Aisle width between the two rows' back faces (m).
        @param yaws: Optional (yaw_a, yaw_b). Defaults to the canonical
                     back-to-back pair for the chosen direction:
                       east/west  -> (90, 270)  rows facing +Y and -Y
                       north/south -> (0, 180)   rows facing +X and -X
        @return (row_a, row_b) tuple. row_a is the row on the -axis side
                of the centre (south or west); row_b is on the +axis side.
        """
        _check_direction_and_type(direction, bay_type, self.dims)
        bay_d = self.dims[bay_type]["depth"]
        if yaws is None:
            yaws = _DEFAULT_BACK_TO_BACK_YAWS[direction]
        yaw_a, yaw_b = yaws
        offset = gap / 2.0 + bay_d / 2.0
        cx, cy = centre
        # Rows are placed perpendicular to `direction`. For east/west direction,
        # rows stack along y; for north/south, along x.
        if direction in ("east", "west"):
            centre_a = (cx, cy - offset)
            centre_b = (cx, cy + offset)
        else:
            centre_a = (cx - offset, cy)
            centre_b = (cx + offset, cy)
        row_a = self.row_centred(
            bay_type=bay_type,
            n=n,
            centre=centre_a,
            direction=direction,
            yaw_deg=yaw_a,
            spacing=spacing,
            bay_extras=bay_extras,
        )
        row_b = self.row_centred(
            bay_type=bay_type,
            n=n,
            centre=centre_b,
            direction=direction,
            yaw_deg=yaw_b,
            spacing=spacing,
            bay_extras=bay_extras,
        )
        return row_a, row_b

    def row_along_wall(
        self,
        bay_type: str,
        n: int,
        wall_p0: Point,
        wall_p1: Point,
        bay_angle_deg: float,
        side: str = "ccw",
        start_along: Optional[float] = None,
        pack_from: str = "start",
        centred: bool = False,
        bay_extras: Optional[Dict[str, Any]] = None,
    ) -> BayGroup:
        """
        @brief Place a row of n bays along an arbitrary wall segment.

        @param bay_type: One of perpendicular, angled, parallel.
        @param n: Number of bays.
        @param wall_p0: Wall start point.
        @param wall_p1: Wall end point.
        @param bay_angle_deg: Bay yaw relative to inward normal:
                                0   - perpendicular (nose along inward normal).
                              +/-45 - angled, leaning toward wall_p1 / wall_p0.
                              +/-90 - parallel-to-wall, nose along wall direction
                                       (toward wall_p1) or against it (wall_p0).
        @param side: "ccw" places bays on the CCW (left) side of wall direction;
                     "cw" picks the inward side when traversing the wall in
                     reverse polygon order.
        @param start_along: Distance from wall_p0 (or wall_p1 if pack_from="end")
                            to the first bay centre. Defaults to a clearance
                            value matching bay_angle_deg.
        @param pack_from: "start" packs left-to-right from wall_p0;
                          "end" packs right-to-left from wall_p1.
        @param centred: True ignores start_along and centres the cluster of
                        n bays in the wall span.
        @param bay_extras: Extra fields merged into every bay dict.
        """
        if bay_type not in self.dims:
            raise ValueError(f"Unknown bay_type '{bay_type}'.")
        x0, y0 = wall_p0
        x1, y1 = wall_p1
        wall_dx = x1 - x0
        wall_dy = y1 - y0
        wall_len = math.hypot(wall_dx, wall_dy)
        if wall_len < 1e-6:
            raise ValueError("row_along_wall needs a wall of non-zero length.")
        wdx = wall_dx / wall_len
        wdy = wall_dy / wall_len

        wall_yaw_deg = math.degrees(math.atan2(wdy, wdx))
        if side == "ccw":
            nx = -wdy
            ny = wdx
            inward_normal_yaw_deg = wall_yaw_deg + 90.0
        elif side == "cw":
            nx = wdy
            ny = -wdx
            inward_normal_yaw_deg = wall_yaw_deg - 90.0
        else:
            raise ValueError(f"side must be 'ccw' or 'cw', got '{side}'.")

        bay_yaw_deg = (inward_normal_yaw_deg + bay_angle_deg) % 360.0

        bay_w = self.dims[bay_type]["width"]
        bay_d = self.dims[bay_type]["depth"]
        normalised_angle = abs(((bay_angle_deg + 0.5) % 180.0) - 0.5)
        if normalised_angle < 1.0:
            spacing = bay_w
            inward = bay_d / 2.0 + self.wall_gap
            default_start_along = bay_w / 2.0 + self.wall_gap
        elif abs(normalised_angle - 90.0) < 1.0:
            spacing = bay_d
            inward = bay_w / 2.0 + self.wall_gap
            default_start_along = bay_d / 2.0 + self.wall_gap
        else:
            spacing = bay_w / math.sin(math.radians(normalised_angle))
            half_diag = (bay_d / 2.0 + bay_w / 2.0) * math.sin(
                math.radians(normalised_angle)
            )
            inward = half_diag + self.wall_gap
            default_start_along = (bay_d / 2.0 + bay_w / 2.0) * math.cos(
                math.radians(normalised_angle)
            ) + self.wall_gap

        if centred:
            cluster_span = (n - 1) * spacing
            start_along = (wall_len - cluster_span) / 2.0
        elif start_along is None:
            start_along = default_start_along

        if pack_from == "start":
            anchor_along = start_along
            advance_sign = 1.0
        elif pack_from == "end":
            anchor_along = wall_len - start_along
            advance_sign = -1.0
        else:
            raise ValueError(f"pack_from must be 'start' or 'end', got '{pack_from}'.")

        extras = bay_extras or {}
        step = advance_sign * spacing
        bays: List[Dict[str, Any]] = [
            {
                "bay_type": bay_type,
                "local_x": x0 + wdx * (anchor_along + step * i) + nx * inward,
                "local_y": y0 + wdy * (anchor_along + step * i) + ny * inward,
                "local_yaw_deg": bay_yaw_deg,
                "width": bay_w,
                "depth": bay_d,
                **extras,
            }
            for i in range(n)
        ]

        group = BayGroup(
            bay_type=bay_type,
            bays=bays,
            direction=(wdx * advance_sign, wdy * advance_sign),
            normal=_yaw_to_normal(bay_yaw_deg),
            bay_width=bay_w,
            bay_depth=bay_d,
            spacing=spacing,
        )
        self._groups.append(group)
        return group

    def row_along_perimeter(
        self,
        bay_type: str,
        n: int,
        wall: int,
        bay_angle_deg: float = 0.0,
        centred: bool = False,
        start_along: Optional[float] = None,
        pack_from: str = "start",
        bay_extras: Optional[Dict[str, Any]] = None,
    ) -> BayGroup:
        """
        @brief Place a row of bays alongside one perimeter wall.

        Wall index follows CCW polygon order: wall 0 is from corners[0] to
        corners[1], wall 1 is corners[1] to corners[2], and so on. The
        inward direction is auto-detected (CCW polygon convention).

        @param bay_type: One of perpendicular, angled, parallel.
        @param n: Number of bays.
        @param wall: Perimeter wall index (0 to len(corners)-1).
        @param bay_angle_deg: Bay yaw relative to the inward normal:
                                0   - perpendicular, back to wall (typical).
                              +/-45 - angled, leaning toward next/previous corner.
                              +/-90 - parallel-to-wall.
                              180   - perpendicular, head-in to wall.
        @param centred: True centres the cluster of n bays in the wall span.
        @param start_along: Distance from corners[wall] to the first bay
                            centre. Defaults to the natural corner clearance.
        @param pack_from: "start" packs from corners[wall] toward
                          corners[wall+1]; "end" packs the other way.
        @param bay_extras: Extra fields merged into every bay dict.
        """
        if not 0 <= wall < len(self.corners):
            raise ValueError(
                f"wall index {wall} out of range " f"[0, {len(self.corners) - 1}]."
            )
        p0_dict = self.corners[wall]
        p1_dict = self.corners[(wall + 1) % len(self.corners)]
        return self.row_along_wall(
            bay_type=bay_type,
            n=n,
            wall_p0=(p0_dict["x"], p0_dict["y"]),
            wall_p1=(p1_dict["x"], p1_dict["y"]),
            bay_angle_deg=bay_angle_deg,
            side="ccw",
            start_along=start_along,
            pack_from=pack_from,
            centred=centred,
            bay_extras=bay_extras,
        )

    def row_along_obstacle_face(
        self,
        bay_type: str,
        n: int,
        obstacle: Tuple[float, float, float, float],
        face: str,
        bay_angle_deg: float = 180.0,
        centred: bool = True,
        start_along: Optional[float] = None,
        pack_from: str = "start",
        bay_extras: Optional[Dict[str, Any]] = None,
    ) -> BayGroup:
        """
        @brief Place a row of bays on the outside of one face of a rectangular
               obstacle.

        Bays are placed in the lot aisle adjacent to the chosen obstacle face
        (never inside the obstacle). The default bay_angle_deg=180 produces
        head-in parking with the bay nose toward the obstacle, which is the
        typical layout for parking around a central obstruction. Use
        bay_angle_deg=0 for back-in parking with bay back toward the obstacle.

        @param bay_type: One of perpendicular, angled, parallel.
        @param n: Number of bays.
        @param obstacle: (x_min, x_max, y_min, y_max) of the obstacle rectangle.
        @param face: "north", "south", "east", or "west" - which face of the
                     obstacle the bays sit alongside.
        @param bay_angle_deg: Bay yaw relative to the obstacle-outward normal.
                              180 (default) = head-in (nose toward obstacle).
                              0 = back-in (back toward obstacle).
        @param centred: True centres the cluster along the face extent.
        @param start_along: Distance from the face start to the first bay.
        @param pack_from: "start" or "end" along the face direction.
        @param bay_extras: Extra fields merged into every bay dict.
        """
        x_min, x_max, y_min, y_max = obstacle
        # Walk each face in obstacle-CCW order (bottom-left start). side="cw"
        # then picks the "outward" normal (away from obstacle interior, into
        # the surrounding aisle where bays sit).
        if face == "south":
            wall_p0, wall_p1 = (x_min, y_min), (x_max, y_min)
        elif face == "east":
            wall_p0, wall_p1 = (x_max, y_min), (x_max, y_max)
        elif face == "north":
            wall_p0, wall_p1 = (x_max, y_max), (x_min, y_max)
        elif face == "west":
            wall_p0, wall_p1 = (x_min, y_max), (x_min, y_min)
        else:
            raise ValueError(
                f"face must be 'north'|'south'|'east'|'west', got '{face}'."
            )
        return self.row_along_wall(
            bay_type=bay_type,
            n=n,
            wall_p0=wall_p0,
            wall_p1=wall_p1,
            bay_angle_deg=bay_angle_deg,
            side="cw",
            centred=centred,
            start_along=start_along,
            pack_from=pack_from,
            bay_extras=bay_extras,
        )

    def facing_row(
        self,
        twin: BayGroup,
        gap: float,
        n: Optional[int] = None,
        bay_extras: Optional[Dict[str, Any]] = None,
    ) -> BayGroup:
        """
        @brief Place a row that faces an existing row across an aisle of `gap`.

        The new row sits on the NOSE side of `twin` at distance `gap` (nose
        face to nose face, with the aisle between them), with its own nose
        pointing back toward `twin`. Same bay type, same spacing, same number
        of bays as `twin` unless `n` is given. Useful for the opposing row
        when the first row is placed directly against a wall (backs to wall,
        noses into the lot interior).

        @param twin: An existing BayGroup to mirror across an aisle.
        @param gap: Aisle width between the two rows' nose faces (m).
        @param n: Override number of bays in the new row (defaults to twin's).
        @param bay_extras: Extra fields merged into every bay dict.
        """
        n = n if n is not None else twin.count
        nx, ny = twin.normal  # nose direction of twin
        offset = gap + twin.bay_depth  # centre-to-centre distance
        first = twin.bays[0]
        dx, dy = twin.direction
        new_first_x = first["local_x"] + nx * offset
        new_first_y = first["local_y"] + ny * offset
        new_yaw_deg = (first["local_yaw_deg"] + 180.0) % 360.0
        bay_w = twin.bay_width
        bay_d = twin.bay_depth
        extras = bay_extras or {}
        step_x = dx * twin.spacing
        step_y = dy * twin.spacing
        bays: List[Dict[str, Any]] = [
            {
                "bay_type": twin.bay_type,
                "local_x": new_first_x + step_x * i,
                "local_y": new_first_y + step_y * i,
                "local_yaw_deg": new_yaw_deg,
                "width": bay_w,
                "depth": bay_d,
                **extras,
            }
            for i in range(n)
        ]
        group = BayGroup(
            bay_type=twin.bay_type,
            bays=bays,
            direction=(dx, dy),
            normal=(-nx, -ny),
            bay_width=bay_w,
            bay_depth=bay_d,
            spacing=twin.spacing,
        )
        self._groups.append(group)
        return group

    def place_bay(
        self,
        bay_type: str,
        x: float,
        y: float,
        yaw_deg: float,
        width: Optional[float] = None,
        depth: Optional[float] = None,
        bay_extras: Optional[Dict[str, Any]] = None,
    ) -> BayGroup:
        """
        @brief Place a single bay (returned as a one-bay BayGroup).
        @param bay_type: Bay type. Custom types (e.g. "motorcycle") are
                         allowed when width and depth are also supplied.
        """
        if bay_type in self.dims:
            bay_w = width if width is not None else self.dims[bay_type]["width"]
            bay_d = depth if depth is not None else self.dims[bay_type]["depth"]
        else:
            if width is None or depth is None:
                raise ValueError(
                    f"bay_type '{bay_type}' is not in BAY_DIMS; "
                    "width and depth must be provided explicitly."
                )
            bay_w = width
            bay_d = depth
        extras = bay_extras or {}
        bay = {
            "bay_type": bay_type,
            "local_x": float(x),
            "local_y": float(y),
            "local_yaw_deg": yaw_deg % 360.0,
            "width": bay_w,
            "depth": bay_d,
            **extras,
        }
        group = BayGroup(
            bay_type=bay_type,
            bays=[bay],
            direction=(1.0, 0.0),
            normal=_yaw_to_normal(yaw_deg),
            bay_width=bay_w,
            bay_depth=bay_d,
            spacing=bay_w,
        )
        self._groups.append(group)
        return group

    # ------------------------------------------------------------------
    # Spawns
    # ------------------------------------------------------------------

    def spawn(
        self,
        x: float,
        y: float,
        yaw_deg: float,
        primary: bool = False,
    ) -> None:
        """
        @brief Add a spawn transform.
        @param primary: True marks the entrance gate. Exactly one primary
                        spawn is required.
        """
        spawn_dict = {"x": float(x), "y": float(y), "yaw_deg": float(yaw_deg)}
        if primary:
            if self._primary_spawn is not None:
                raise ValueError("Only one primary spawn is allowed per layout.")
            self._primary_spawn = spawn_dict
        else:
            self._spawns.append(spawn_dict)

    # ------------------------------------------------------------------
    # Pedestrian zones, patrol, obstacles
    # ------------------------------------------------------------------

    def add_zone(self, zone: PedestrianZone) -> None:
        """@brief Append a pedestrian zone."""
        self._zones.append(zone)

    def set_patrol(self, patrol: PatrolPath) -> None:
        """@brief Set the patrol path for this lot."""
        self._patrol = patrol

    def add_obstacle(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        """@brief Add an axis-aligned interior obstacle rectangle."""
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Obstacle has zero or negative extent: "
                f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]"
            )
        self._obstacles.append(
            {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
            }
        )

    # ------------------------------------------------------------------
    # Validation + build
    # ------------------------------------------------------------------

    def _all_bays(self) -> List[Dict[str, Any]]:
        bays: List[Dict[str, Any]] = []
        for group in self._groups:
            bays.extend(group.bays)
        return bays

    def _warn_bays_close_to_wall(self) -> None:
        """
        @brief Warn if any bay corner is within wall_gap of the perimeter.
        """
        poly: List[Point] = [(c["x"], c["y"]) for c in self.corners]
        n = len(poly)
        # Pre-compute edge vectors and squared lengths once.
        edges = []
        for i in range(n):
            p0 = poly[i]
            p1 = poly[(i + 1) % n]
            ex = p1[0] - p0[0]
            ey = p1[1] - p0[1]
            seg_len_sq = ex * ex + ey * ey
            edges.append((p0, ex, ey, seg_len_sq))

        for bay in self._all_bays():
            corners = _bay_corners(
                bay["local_x"],
                bay["local_y"],
                bay["local_yaw_deg"],
                bay["width"],
                bay["depth"],
            )
            for cx, cy in corners:
                min_d = float("inf")
                for p0, ex, ey, seg_len_sq in edges:
                    if seg_len_sq < 1e-12:
                        continue
                    t = ((cx - p0[0]) * ex + (cy - p0[1]) * ey) / seg_len_sq
                    t = max(0.0, min(1.0, t))
                    px = p0[0] + t * ex
                    py = p0[1] + t * ey
                    d = math.hypot(cx - px, cy - py)
                    if d < min_d:
                        min_d = d
                if min_d < self.wall_gap - 1e-3:
                    print(
                        f"  WARNING [{self.name}]: bay corner of "
                        f"'{bay.get('bay_type', '?')}' "
                        f"at ({cx:.2f}, {cy:.2f}) is {min_d:.3f} m from the "
                        f"perimeter (wall_gap={self.wall_gap:.2f} m)."
                    )

    def build(self) -> Dict[str, Any]:
        """
        @brief Validate and emit the layout dict.
        @raises ValueError: If validation fails (missing primary or extra spawn,
                bay outside polygon).
        """
        if self._primary_spawn is None:
            raise ValueError(
                f"[{self.name}] No primary spawn set "
                "(call spawn(..., primary=True))."
            )
        if not self._spawns:
            raise ValueError(
                f"[{self.name}] No extra spawn set; every layout must provide "
                "at least one secondary spawn."
            )
        all_bays = self._all_bays()
        validate_bays_in_polygon(all_bays, self.corners, self.name)
        warn_narrow_corridors(all_bays, self.name)
        self._warn_bays_close_to_wall()

        result: Dict[str, Any] = {
            "corners": self.corners,
            "bays": all_bays,
            "spawn": self._primary_spawn,
            "extra_spawns": list(self._spawns),
            "patrol_waypoints": self._patrol.to_list() if self._patrol else [],
            "pedestrian_zones": [zone.to_dict() for zone in self._zones],
        }
        if self._obstacles:
            result["obstacles"] = list(self._obstacles)
        return result
