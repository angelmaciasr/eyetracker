import pytest

from eyetracker.config import MeasurementConfig
from eyetracker.domain import FaceObservation, Point2D, TrackingStatus
from eyetracker.services.eye_measurement import EAREyeMeasurementService, eye_aspect_ratio


def test_eye_aspect_ratio_uses_vertical_over_horizontal_distance():
    eye = (
        Point2D(0.0, 0.0),
        Point2D(0.25, 0.2),
        Point2D(0.75, 0.2),
        Point2D(1.0, 0.0),
        Point2D(0.75, -0.2),
        Point2D(0.25, -0.2),
    )
    assert eye_aspect_ratio(eye) == pytest.approx(0.4)


def test_eye_aspect_ratio_rejects_wrong_number_of_points():
    with pytest.raises(ValueError):
        eye_aspect_ratio((Point2D(0, 0),))


def test_wide_open_eyes_are_not_treated_as_tracking_loss():
    eye = (
        Point2D(0.0, 0.0),
        Point2D(0.25, 0.35),
        Point2D(0.75, 0.35),
        Point2D(1.0, 0.0),
        Point2D(0.75, -0.35),
        Point2D(0.25, -0.35),
    )
    observation = FaceObservation(0.0, TrackingStatus.VALID, eye, eye, 1.0)
    result = EAREyeMeasurementService(MeasurementConfig()).measure(observation)
    assert result.left_ear == pytest.approx(0.7)
    assert result.reliable
