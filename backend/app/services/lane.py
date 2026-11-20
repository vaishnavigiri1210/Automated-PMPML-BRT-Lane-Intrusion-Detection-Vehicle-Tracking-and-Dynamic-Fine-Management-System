"""Lane geometry, ground-contact localization, and metric distance estimation."""
from typing import List, Optional, Tuple

Point = Tuple[float, float]


def point_in_polygon(p: Point, poly: List[Point]) -> bool:
    """Determine if point p lies inside polygon poly using ray casting."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def ground_point(bbox: Tuple[float, float, float, float]) -> Point:
    """Calculate the bottom-center ground-contact point of a vehicle bounding box."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def estimate_step_distance_m(
    prev_point: Optional[Point],
    curr_point: Point,
    frame_height: int,
    fallback_step_m: float = 12.0,
) -> float:
    """Estimate forward corridor travel in meters from ground-point displacement."""
    if prev_point is None or frame_height <= 0:
        return fallback_step_m

    dx = curr_point[0] - prev_point[0]
    dy = curr_point[1] - prev_point[1]
    pixel_displacement = (dx * dx + dy * dy) ** 0.5

    # Normalize displacement relative to frame height and scale to corridor meters
    meters_per_pixel = 35.0 / float(frame_height)
    estimated_m = pixel_displacement * meters_per_pixel

    # Clamp extreme jumps/jitter so distance progression remains realistic
    if estimated_m < 0.5:
        return fallback_step_m
    return min(estimated_m, 60.0)