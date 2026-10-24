"""Half-court geometry.

Units are feet. Origin is the left baseline corner as seen from half court:
x runs along the baseline (0..50), y runs from the baseline toward the
half-court line (0..47). The basket is at (25, 5.25).
"""

from __future__ import annotations

import math

import numpy as np

WIDTH = 50.0
HALF_LENGTH = 47.0
BASKET = np.array([25.0, 5.25])

THREE_ARC_RADIUS = 23.75
THREE_CORNER_DX = 22.0  # corner lines sit 22 ft from the basket center
THREE_BREAK_Y = BASKET[1] + math.sqrt(THREE_ARC_RADIUS**2 - THREE_CORNER_DX**2)  # ~14.2

PAINT_HALF_WIDTH = 8.0
PAINT_LENGTH = 19.0
RESTRICTED_RADIUS = 4.0
PLAYER_RADIUS = 1.25

# Named tactical spots. Macro actions move players between these.
REGIONS: dict[str, tuple[float, float]] = {
    "left_corner": (3.0, 3.5),
    "right_corner": (47.0, 3.5),
    "left_short_corner": (10.0, 6.0),
    "right_short_corner": (40.0, 6.0),
    "left_block": (18.0, 7.0),
    "right_block": (32.0, 7.0),
    "rim": (25.0, 8.0),
    "left_elbow": (17.0, 19.0),
    "free_throw": (25.0, 19.0),
    "right_elbow": (33.0, 19.0),
    "left_wing": (7.5, 23.0),
    "right_wing": (42.5, 23.0),
    "left_slot": (16.0, 29.5),
    "top": (25.0, 31.0),
    "right_slot": (34.0, 29.5),
}
REGION_NAMES = list(REGIONS)
REGION_XY = np.array([REGIONS[n] for n in REGION_NAMES])


def dist_to_basket(p) -> float:
    return float(math.hypot(p[0] - BASKET[0], p[1] - BASKET[1]))


def is_three(p) -> bool:
    """True if a shot from p is worth 3. Standing on the line is a 2."""
    dx = abs(float(p[0]) - BASKET[0])
    if p[1] <= THREE_BREAK_Y:
        return bool(dx > THREE_CORNER_DX)
    return dist_to_basket(p) > THREE_ARC_RADIUS


def in_paint(p) -> bool:
    return abs(p[0] - BASKET[0]) <= PAINT_HALF_WIDTH and 0.0 <= p[1] <= PAINT_LENGTH


def shot_zone(p) -> str:
    """One of: rim, paint, mid, corner3, above3."""
    if is_three(p):
        return "corner3" if p[1] <= THREE_BREAK_Y else "above3"
    if dist_to_basket(p) <= RESTRICTED_RADIUS:
        return "rim"
    if in_paint(p):
        return "paint"
    return "mid"


def clamp_to_court(pos: np.ndarray) -> np.ndarray:
    """Keep bodies inside the half court (in place)."""
    np.clip(pos[..., 0], PLAYER_RADIUS, WIDTH - PLAYER_RADIUS, out=pos[..., 0])
    np.clip(pos[..., 1], PLAYER_RADIUS, HALF_LENGTH - PLAYER_RADIUS, out=pos[..., 1])
    return pos
