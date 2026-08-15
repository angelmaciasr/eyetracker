from __future__ import annotations

import math
from typing import Any

from ..domain import HeadPose


def angular_distance(first: float, second: float) -> float:
    """Return the shortest distance between two angles, including -180/180 wraparound."""
    return abs((first - second + 180.0) % 360.0 - 180.0)


def signed_angular_delta(angle: float, reference: float) -> float:
    """Return the signed angular difference within [-180, 180)."""
    return (angle - reference + 180.0) % 360.0 - 180.0


def head_pose_from_transform(matrix: Any) -> HeadPose:
    """Extract XYZ Euler angles from MediaPipe's 4x4 facial transform."""
    import numpy as np

    transform = np.asarray(matrix, dtype=float)
    if transform.shape not in ((4, 4), (3, 3)):
        raise ValueError("The pose matrix must be 3x3 or 4x4")
    raw_rotation = transform[:3, :3]
    # The matrix may contain a small scale component. SVD finds the closest rotation.
    left, _, right = np.linalg.svd(raw_rotation)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right

    horizontal = math.hypot(rotation[0, 0], rotation[1, 0])
    if horizontal > 1e-6:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(-rotation[2, 0], horizontal)
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = math.atan2(-rotation[2, 0], horizontal)
        roll = 0.0
    return HeadPose(*(math.degrees(angle) for angle in (pitch, yaw, roll)))
