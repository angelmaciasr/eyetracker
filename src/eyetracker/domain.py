from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass
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
    LOOK_DOWN = "look_down"
    LOOK_UP = "look_up"
    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    BLINKING = "blinking"
    CLOSED = "closed"


class DrowsinessState(StrEnum):
    AWAKE = "awake"
    BLINKING = "blinking"
    EYES_CLOSED = "eyes_closed"
    ALERT = "alert"
    HEAD_TILT = "head_tilt"
    HEAD_TILT_ALERT = "head_tilt_alert"
    CALIBRATION_RANGE_ALERT = "calibration_range_alert"
    TRACKING_LOST = "tracking_lost"


class DrowsinessLevel(StrEnum):
    NORMAL = "normal"
    POSSIBLE_DROWSINESS = "possible_drowsiness"
    DROWSY = "drowsy"
    IMMEDIATE_ALERT = "immediate_alert"
    TRACKING_LOST = "tracking_lost"


class AlarmSeverity(StrEnum):
    NONE = "none"
    WARNING = "warning"
    URGENT = "urgent"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class HeadPose:
    pitch: float
    yaw: float
    roll: float


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
    head_pose: HeadPose | None = None


@dataclass(frozen=True)
class RawEyeMeasurement:
    timestamp: float
    left_ear: float
    right_ear: float
    reliable: bool
    head_pose: HeadPose | None = None

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
    head_pose: HeadPose | None = None
    pose_valid: bool = True
    pitch_delta: float | None = None
    roll_delta: float | None = None
    pose_confidence: float = 1.0


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
    neutral_pitch: float = 0.0
    neutral_yaw: float = 0.0
    neutral_roll: float = 0.0
    pitch_lower_delta: float = -15.0
    pitch_upper_delta: float = 15.0
    left_open_ear_at_pitch_lower: float = 0.0
    right_open_ear_at_pitch_lower: float = 0.0
    left_open_ear_at_pitch_upper: float = 0.0
    right_open_ear_at_pitch_upper: float = 0.0
    yaw_lower_delta: float = -20.0
    yaw_upper_delta: float = 20.0
    left_open_ear_at_yaw_lower: float = 0.0
    right_open_ear_at_yaw_lower: float = 0.0
    left_open_ear_at_yaw_upper: float = 0.0
    right_open_ear_at_yaw_upper: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CalibrationProfile:
        parsed: dict[str, float] = {}
        for key, field in cls.__dataclass_fields__.items():
            if key in values:
                parsed[key] = float(values[key])
            elif field.default is not MISSING:
                parsed[key] = float(field.default)
        return cls(**parsed)


@dataclass(frozen=True)
class DrowsinessAssessment:
    timestamp: float
    state: DrowsinessState
    current_closure_seconds: float
    last_blink_seconds: float | None
    blink_count: int
    should_alert: bool
    reason: str | None = None
    current_head_tilt_seconds: float = 0.0
    head_pitch_delta: float | None = None
    head_roll_delta: float | None = None
    level: DrowsinessLevel = DrowsinessLevel.NORMAL
    confidence: float = 1.0
    alarm_severity: AlarmSeverity = AlarmSeverity.NONE
    perclos_30_seconds: float = 0.0
    perclos_60_seconds: float = 0.0
    valid_observation_seconds: float = 0.0
    slow_blinks_last_minute: int = 0
    average_closure_seconds: float = 0.0
    longest_closure_seconds: float = 0.0
    current_tracking_lost_seconds: float = 0.0
