import pytest

from eyetracker.config import DetectorConfig
from eyetracker.domain import DrowsinessState, RelativeEyeMeasurement
from eyetracker.services.detector import TemporalDrowsinessDetector


def measurement(timestamp: float, openness: float, reliable: bool = True):
    return RelativeEyeMeasurement(timestamp, openness, openness, openness, reliable)


def test_normal_blink_does_not_alert_and_is_counted():
    detector = TemporalDrowsinessDetector(DetectorConfig())
    detector.update(measurement(0.00, 1.0))
    detector.update(measurement(1.00, 0.1))
    closing = detector.update(measurement(1.15, 0.0))
    reopened = detector.update(measurement(1.25, 0.9))
    assert closing.state is DrowsinessState.BLINKING
    assert not closing.should_alert
    assert reopened.state is DrowsinessState.AWAKE
    assert reopened.blink_count == 1


def test_prolonged_closure_alerts_using_time_not_frame_count():
    detector = TemporalDrowsinessDetector(DetectorConfig(alert_after_closed_seconds=1.5))
    detector.update(measurement(10.0, 0.1))
    assessment = detector.update(measurement(11.51, 0.1))
    assert assessment.state is DrowsinessState.ALERT
    assert assessment.should_alert
    assert assessment.current_closure_seconds == pytest.approx(1.51)


def test_hysteresis_prevents_state_oscillation():
    detector = TemporalDrowsinessDetector(DetectorConfig())
    detector.update(measurement(0.0, 0.1))
    still_closed = detector.update(measurement(0.2, 0.80))
    reopened = detector.update(measurement(0.3, 0.86))
    assert still_closed.state is DrowsinessState.BLINKING
    assert reopened.state is DrowsinessState.AWAKE


def test_tracking_loss_never_becomes_an_eye_closure():
    detector = TemporalDrowsinessDetector(DetectorConfig())
    detector.update(measurement(0.0, 0.1))
    lost = detector.tracking_lost(5.0)
    assert lost.state is DrowsinessState.TRACKING_LOST
    assert not lost.should_alert
    assert lost.current_closure_seconds == 0.0
