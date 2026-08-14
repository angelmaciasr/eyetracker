import pytest

from eyetracker.config import CalibrationConfig
from eyetracker.domain import CalibrationPhase, RawEyeMeasurement
from eyetracker.services.calibration import PersonalCalibrationService


def sample(timestamp: float, left: float, right: float) -> RawEyeMeasurement:
    return RawEyeMeasurement(timestamp, left, right, True)


def calibrated_service() -> PersonalCalibrationService:
    service = PersonalCalibrationService(CalibrationConfig(minimum_samples=3))
    for index, value in enumerate((0.30, 0.31, 0.29)):
        service.add_sample(CalibrationPhase.OPEN, sample(index / 30, value, value + 0.01))
    service.add_blink(0.18, 0.10, 0.11)
    service.add_blink(0.22, 0.09, 0.10)
    for index, value in enumerate((0.10, 0.09, 0.11)):
        service.add_sample(CalibrationPhase.CLOSED, sample(1 + index / 30, value, value + 0.01))
    service.finish()
    return service


def test_calibration_normalizes_each_eye_independently():
    service = calibrated_service()
    opened = service.normalize(sample(2.0, 0.30, 0.31))
    closed = service.normalize(sample(2.1, 0.10, 0.11))
    assert opened.left_openness == pytest.approx(1.0)
    assert opened.right_openness == pytest.approx(1.0)
    assert closed.combined_openness == pytest.approx(0.0)


def test_blink_limit_is_personalized_but_bounded():
    profile = calibrated_service().profile
    assert profile is not None
    assert profile.typical_blink_seconds == pytest.approx(0.20)
    assert profile.maximum_normal_blink_seconds == pytest.approx(0.40)
