"""Utility helpers for posture analysis."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple, Union

Number = Union[int, float]
PointLike = Union[Sequence[Number], Iterable[Number]]


def _normalize_point(point: Union[PointLike, object]) -> Tuple[float, float]:
    """Return the ``(x, y)`` representation of a Mediapipe landmark or sequence."""

    if hasattr(point, "x") and hasattr(point, "y"):
        return float(getattr(point, "x")), float(getattr(point, "y"))

    if isinstance(point, Sequence):
        if len(point) < 2:
            raise ValueError("Point sequences must contain at least two elements.")
        return float(point[0]), float(point[1])

    try:
        iterator = iter(point)  # type: ignore[arg-type]
        x = float(next(iterator))
        y = float(next(iterator))
    except TypeError as exc:
        raise TypeError("Unsupported point representation provided.") from exc
    except StopIteration as exc:
        raise ValueError("Point iterables must yield at least two values.") from exc

    return x, y


def calculate_angle(a: Union[PointLike, object],
                    b: Union[PointLike, object],
                    c: Union[PointLike, object]) -> float:
    """Calculate the angle ABC (in degrees) formed by three 2D points."""

    ax, ay = _normalize_point(a)
    bx, by = _normalize_point(b)
    cx, cy = _normalize_point(c)

    ba_x, ba_y = ax - bx, ay - by
    bc_x, bc_y = cx - bx, cy - by

    dot = ba_x * bc_x + ba_y * bc_y
    cross = ba_x * bc_y - ba_y * bc_x

    angle = abs(math.degrees(math.atan2(cross, dot)))

    return angle
