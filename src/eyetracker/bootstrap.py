from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .adapters.alarm import SoundAlarm
from .adapters.camera import OpenCVCamera, VideoFileCamera
from .adapters.display import OpenCVDisplay
from .adapters.event_logger import JsonlEventLogger
from .adapters.face_tracker import MediaPipeFaceTracker
from .adapters.repository import JsonCalibrationRepository
from .application import CalibrationController, MonitoringController
from .config import AppConfig
from .ports import CameraPort
from .services.alert_manager import AlertManager
from .services.calibration import PersonalCalibrationService
from .services.detector import TemporalDrowsinessDetector
from .services.eye_measurement import EAREyeMeasurementService

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_model(path: Path) -> None:
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    print(f"Downloading the MediaPipe model to {path}...")
    try:
        with (
            urllib.request.urlopen(MODEL_URL, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("The model download is incomplete")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@dataclass
class Application:
    camera: CameraPort
    tracker: MediaPipeFaceTracker
    display: OpenCVDisplay
    alarm: SoundAlarm
    calibrator: PersonalCalibrationService
    repository: JsonCalibrationRepository
    calibration_controller: CalibrationController
    monitoring_controller: MonitoringController

    def close(self) -> None:
        self.alarm.close()
        self.tracker.close()
        self.camera.stop()
        self.display.close()


def build_application(
    config: AppConfig,
    root: Path,
    video_path: Path | None = None,
    data_root: Path | None = None,
) -> Application:
    data_root = data_root or root
    bundled_model_path = (root / config.tracker.model_path).resolve()
    writable_model_path = (data_root / config.tracker.model_path).resolve()
    model_path = bundled_model_path if bundled_model_path.exists() else writable_model_path
    calibration_path = (data_root / config.storage.calibration_path).resolve()
    ensure_model(model_path)
    events_path = (data_root / config.logging.events_path).resolve()
    camera: CameraPort = (
        VideoFileCamera(str(video_path.resolve()))
        if video_path is not None
        else OpenCVCamera(config.camera)
    )
    tracker = MediaPipeFaceTracker(config.tracker, model_path)
    display = OpenCVDisplay(config.detector.closed_threshold)
    alarm = SoundAlarm(config.alarm)
    alert_manager = AlertManager(alarm, config.alarm)
    event_logger = JsonlEventLogger(events_path, config.logging.enabled)
    measurement = EAREyeMeasurementService(config.measurement)
    calibrator = PersonalCalibrationService(config.calibration, config.head_pose)
    detector = TemporalDrowsinessDetector(config.detector)
    repository = JsonCalibrationRepository(calibration_path)
    calibration_controller = CalibrationController(
        camera,
        tracker,
        measurement,
        calibrator,
        alarm,
        display,
        repository,
        config.calibration,
    )
    monitoring_controller = MonitoringController(
        camera,
        tracker,
        measurement,
        calibrator,
        detector,
        alert_manager,
        display,
        event_logger,
    )
    return Application(
        camera,
        tracker,
        display,
        alarm,
        calibrator,
        repository,
        calibration_controller,
        monitoring_controller,
    )
