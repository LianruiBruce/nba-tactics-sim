"""Full-court geometry.

Global frame, in feet: x runs along the court length (0..94), y across it
(0..50). Baskets sit 5.25 ft from each baseline at (5.25, 25) and (88.75, 25).

Each team attacks one basket, given by `direction`: +1 attacks the basket at
x = 88.75, -1 the one at x = 5.25. Tactics are written in a local frame seen
from half court facing the attacked basket: `lx` runs across the court from
the offense's left sideline (0) to its right sideline (50), and `ly` is the
distance from the attacked baseline (0 at the baseline, 47 at half court,
94 at the far baseline). Geometry below takes local coordinates.
"""

from __future__ import annotations

import math

import numpy as np

LENGTH = 94.0
WIDTH = 50.0
HALF = 47.0
BASKET_FROM_BASELINE = 5.25
BASKET_LOCAL = np.array([25.0, BASKET_FROM_BASELINE])

THREE_ARC_RADIUS = 23.75
THREE_CORNER_DX = 22.0
THREE_BREAK_LY = BASKET_FROM_BASELINE + math.sqrt(THREE_ARC_RADIUS**2 - THREE_CORNER_DX**2)

PAINT_HALF_WIDTH = 8.0
PAINT_LENGTH = 19.0  # baseline to free-throw line
RIM_ZONE = 4.0  # "at the rim" shots, matches Cleaning the Glass
SHORT_MID_MAX = 14.0
PLAYER_RADIUS = 1.25
FT_SPOT_LOCAL = np.array([25.0, 19.0])


def to_global(local, direction: int) -> np.ndarray:
    local = np.asarray(local, dtype=float)
    lx, ly = local[..., 0], local[..., 1]
    if direction > 0:
        return np.stack([LENGTH - ly, WIDTH - lx], axis=-1)
    return np.stack([ly, lx], axis=-1)


def to_local(glob, direction: int) -> np.ndarray:
    glob = np.asarray(glob, dtype=float)
    x, y = glob[..., 0], glob[..., 1]
    if direction > 0:
        return np.stack([WIDTH - y, LENGTH - x], axis=-1)
    return np.stack([y, x], axis=-1)


def basket(direction: int) -> np.ndarray:
    return to_global(BASKET_LOCAL, direction)


# Named half-court spots in the local frame.
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
REGION_LOCAL = np.array([REGIONS[n] for n in REGION_NAMES])


def dist_to_basket(local) -> float:
    return float(math.hypot(local[0] - BASKET_LOCAL[0], local[1] - BASKET_LOCAL[1]))


def is_three(local) -> bool:
    """True if a shot from `local` is worth 3. Standing on the line is a 2."""
    dx = abs(float(local[0]) - BASKET_LOCAL[0])
    if local[1] <= THREE_BREAK_LY:
        return bool(dx > THREE_CORNER_DX)
    return dist_to_basket(local) > THREE_ARC_RADIUS


def in_paint(local) -> bool:
    return bool(abs(local[0] - BASKET_LOCAL[0]) <= PAINT_HALF_WIDTH and 0.0 <= local[1] <= PAINT_LENGTH)


def in_frontcourt(local) -> bool:
    return bool(local[1] < HALF)


def shot_zone(local) -> str:
    """rim, short_mid, long_mid, corner3 or above3 (Cleaning the Glass buckets)."""
    if is_three(local):
        return "corner3" if local[1] <= THREE_BREAK_LY else "above3"
    d = dist_to_basket(local)
    if d <= RIM_ZONE:
        return "rim"
    if d <= SHORT_MID_MAX:
        return "short_mid"
    return "long_mid"


def clamp_inbounds(pos: np.ndarray, margin: float = PLAYER_RADIUS) -> np.ndarray:
    """Keep bodies on the floor (global frame, in place)."""
    np.clip(pos[..., 0], margin, LENGTH - margin, out=pos[..., 0])
    np.clip(pos[..., 1], margin, WIDTH - margin, out=pos[..., 1])
    return pos


def out_of_bounds(glob) -> bool:
    return bool(glob[0] < 0 or glob[0] > LENGTH or glob[1] < 0 or glob[1] > WIDTH)
