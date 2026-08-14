from __future__ import annotations

import math
import statistics
from collections import deque

from ..config import MeasurementConfig
from ..domain import FaceObservation, Point2D, RawEyeMeasurement, TrackingStatus


def eye_aspect_ratio(points: tuple[Point2D, ...]) -> float:
    """Calcula EAR con seis puntos ordenados: esquinas, dos pares verticales."""
    if len(points) != 6:
        raise ValueError("EAR necesita exactamente seis puntos")
    p1, p2, p3, p4, p5, p6 = points

    def distance(a: Point2D, b: Point2D) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    horizontal = distance(p1, p4)
    if horizontal <= 1e-9:
        return 0.0
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * horizontal)


class EAREyeMeasurementService:
    def __init__(self, config: MeasurementConfig) -> None:
        self.config = config
        self._left: deque[float] = deque(maxlen=config.smoothing_window)
        self._right: deque[float] = deque(maxlen=config.smoothing_window)

    def reset(self) -> None:
        self._left.clear()
        self._right.clear()

    def measure(self, observation: FaceObservation) -> RawEyeMeasurement:
        if observation.status is not TrackingStatus.VALID:
            return RawEyeMeasurement(observation.timestamp, 0.0, 0.0, False)
        left = eye_aspect_ratio(observation.left_eye)
        right = eye_aspect_ratio(observation.right_eye)
        reliable = all(
            self.config.minimum_ear <= value <= self.config.maximum_ear for value in (left, right)
        )
        if not reliable:
            return RawEyeMeasurement(
                observation.timestamp, left, right, False, observation.head_pose
            )
        self._left.append(left)
        self._right.append(right)
        return RawEyeMeasurement(
            observation.timestamp,
            statistics.median(self._left),
            statistics.median(self._right),
            True,
            observation.head_pose,
        )
