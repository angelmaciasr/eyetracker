from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class TrackingStatus(StrEnum):
    VALID = "valid"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    LOW_CONFIDENCE = "low_confidence"
    ERROR = "error"


class CalibrationPhase(StrEnum):
    OPEN = "open"
    BLINKING = "blinking"
    CLOSED = "closed"


class DrowsinessState(StrEnum):
    AWAKE = "awake"
    BLINKING = "blinking"
    EYES_CLOSED = "eyes_closed"
    ALERT = "alert"
    TRACKING_LOST = "tracking_lost"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class VideoFrame:
    frame_id: int
    timestamp: float
    width: int
    height: int
    image: Any


@dataclass(frozen=True)
class FaceObservation:
    timestamp: float
    status: TrackingStatus
    left_eye: tuple[Point2D, ...] = ()
    right_eye: tuple[Point2D, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class RawEyeMeasurement:
    timestamp: float
    left_ear: float
    right_ear: float
    reliable: bool

    @property
    def combined_ear(self) -> float:
        return (self.left_ear + self.right_ear) / 2.0


@dataclass(frozen=True)
class RelativeEyeMeasurement:
    timestamp: float
    left_openness: float
    right_openness: float
    combined_openness: float
    reliable: bool


@dataclass(frozen=True)
class CalibrationProfile:
    left_open_ear: float
    left_closed_ear: float
    right_open_ear: float
    right_closed_ear: float
    typical_blink_seconds: float
    maximum_normal_blink_seconds: float
    left_blink_min_ear: float
    right_blink_min_ear: float
    measurement_noise: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CalibrationProfile:
        return cls(**{key: float(values[key]) for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DrowsinessAssessment:
    timestamp: float
    state: DrowsinessState
    current_closure_seconds: float
    last_blink_seconds: float | None
    blink_count: int
    should_alert: bool
    reason: str | None = None
