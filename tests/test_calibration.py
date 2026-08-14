import pytest

from eyetracker.config import CalibrationConfig
from eyetracker.domain import CalibrationPhase, HeadPose, RawEyeMeasurement
from eyetracker.services.calibration import PersonalCalibrationService


def sample(
    timestamp: float,
    left: float,
    right: float,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> RawEyeMeasurement:
    return RawEyeMeasurement(timestamp, left, right, True, HeadPose(pitch, yaw, 0.0))


def calibrated_service() -> PersonalCalibrationService:
    service = PersonalCalibrationService(CalibrationConfig(minimum_samples=3))
    for index, value in enumerate((0.30, 0.31, 0.29)):
        service.add_sample(CalibrationPhase.OPEN, sample(index / 30, value, value + 0.01))
    for index, value in enumerate((0.24, 0.25, 0.23)):
        service.add_sample(
            CalibrationPhase.LOOK_DOWN,
            sample(0.2 + index / 30, value, value + 0.01, -20.0),
        )
    for index, value in enumerate((0.36, 0.37, 0.35)):
        service.add_sample(
            CalibrationPhase.LOOK_UP,
            sample(0.4 + index / 30, value, value + 0.01, 20.0),
        )
    for index in range(3):
        service.add_sample(
            CalibrationPhase.LOOK_LEFT,
            sample(0.6 + index / 30, 0.27, 0.34, yaw=-25.0),
        )
        service.add_sample(
            CalibrationPhase.LOOK_RIGHT,
            sample(0.8 + index / 30, 0.33, 0.28, yaw=25.0),
        )
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


def test_pose_outside_calibrated_range_makes_measurement_unreliable():
    service = calibrated_service()
    tilted = RawEyeMeasurement(3.0, 0.30, 0.31, True, HeadPose(26.0, 0.0, 0.0))
    normalized = service.normalize(tilted)
    assert not normalized.pose_valid
    assert not normalized.reliable


@pytest.mark.parametrize(
    ("pitch", "left_ear", "right_ear"),
    [(-20.0, 0.24, 0.25), (20.0, 0.36, 0.37)],
)
def test_pitch_calibration_compensates_apparent_eye_aperture(
    pitch: float, left_ear: float, right_ear: float
):
    service = calibrated_service()
    normalized = service.normalize(sample(3.0, left_ear, right_ear, pitch))
    assert normalized.pose_valid
    assert normalized.left_openness == pytest.approx(1.0)
    assert normalized.right_openness == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("yaw", "left_ear", "right_ear"),
    [(-25.0, 0.27, 0.34), (25.0, 0.33, 0.28)],
)
def test_yaw_calibration_compensates_each_eye_independently(
    yaw: float, left_ear: float, right_ear: float
):
    service = calibrated_service()
    normalized = service.normalize(sample(3.0, left_ear, right_ear, yaw=yaw))
    assert normalized.pose_valid
    assert normalized.left_openness == pytest.approx(1.0)
    assert normalized.right_openness == pytest.approx(1.0)
