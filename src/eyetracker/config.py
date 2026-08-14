from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30


@dataclass(frozen=True)
class TrackerConfig:
    model_path: str = "models/face_landmarker.task"
    min_face_detection_confidence: float = 0.55
    min_face_presence_confidence: float = 0.55
    min_tracking_confidence: float = 0.55


@dataclass(frozen=True)
class MeasurementConfig:
    smoothing_window: int = 3
    minimum_ear: float = 0.02
    maximum_ear: float = 1.00


@dataclass(frozen=True)
class HeadPoseConfig:
    pitch_range_margin: float = 5.0
    yaw_range_margin: float = 5.0
    maximum_roll_delta: float = 20.0
    minimum_calibration_pitch_span: float = 12.0
    minimum_calibration_yaw_span: float = 15.0


@dataclass(frozen=True)
class CalibrationConfig:
    open_seconds: float = 3.0
    pose_seconds: float = 2.0
    pose_settle_seconds: float = 1.0
    required_blinks: int = 5
    blink_timeout_seconds: float = 20.0
    closed_seconds: float = 1.0
    closed_timeout_seconds: float = 10.0
    provisional_closed_ratio: float = 0.60
    provisional_reopened_ratio: float = 0.80
    minimum_samples: int = 12


@dataclass(frozen=True)
class DetectorConfig:
    closed_threshold: float = 0.75
    reopened_threshold: float = 0.85
    minimum_blink_seconds: float = 0.08
    maximum_blink_seconds: float = 0.40
    alert_after_closed_seconds: float = 1.00


@dataclass(frozen=True)
class AlarmConfig:
    repeat_seconds: float = 0.9


@dataclass(frozen=True)
class StorageConfig:
    calibration_path: str = "data/calibration.json"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    tracker: TrackerConfig
    measurement: MeasurementConfig
    head_pose: HeadPoseConfig
    calibration: CalibrationConfig
    detector: DetectorConfig
    alarm: AlarmConfig
    storage: StorageConfig


def _section(data: dict[str, Any], name: str, cls: type[Any]) -> Any:
    return cls(**data.get(name, {}))


def load_config(path: Path) -> AppConfig:
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    return AppConfig(
        camera=_section(data, "camera", CameraConfig),
        tracker=_section(data, "tracker", TrackerConfig),
        measurement=_section(data, "measurement", MeasurementConfig),
        head_pose=_section(data, "head_pose", HeadPoseConfig),
        calibration=_section(data, "calibration", CalibrationConfig),
        detector=_section(data, "detector", DetectorConfig),
        alarm=_section(data, "alarm", AlarmConfig),
        storage=_section(data, "storage", StorageConfig),
    )
