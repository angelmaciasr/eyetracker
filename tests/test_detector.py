import pytest

from eyetracker.config import DetectorConfig
from eyetracker.domain import (
    AlarmSeverity,
    DrowsinessLevel,
    DrowsinessState,
    RelativeEyeMeasurement,
)
from eyetracker.services.detector import TemporalDrowsinessDetector


def measurement(
    timestamp: float,
    openness: float,
    reliable: bool = True,
    pitch_delta: float | None = None,
    pose_valid: bool = True,
    roll_delta: float | None = None,
):
    return RelativeEyeMeasurement(
        timestamp,
        openness,
        openness,
        openness,
        reliable,
        pose_valid=pose_valid,
        pitch_delta=pitch_delta,
        roll_delta=roll_delta,
    )


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


def test_pose_outside_calibrated_range_alerts_immediately():
    detector = TemporalDrowsinessDetector(DetectorConfig())
    assessment = detector.update(
        measurement(2.0, 1.0, reliable=False, pitch_delta=12.0, pose_valid=False)
    )
    assert assessment.state is DrowsinessState.CALIBRATION_RANGE_ALERT
    assert assessment.should_alert
    assert assessment.reason == "head_pose_outside_calibrated_range"


def test_sustained_head_tilt_alerts_even_outside_valid_eye_pose():
    detector = TemporalDrowsinessDetector(
        DetectorConfig(head_tilt_threshold=25.0, head_tilt_alert_seconds=0.75)
    )
    started = detector.update(measurement(0.0, 1.0, reliable=False, pitch_delta=30.0))
    alerted = detector.update(measurement(0.76, 1.0, reliable=False, pitch_delta=30.0))
    assert started.state is DrowsinessState.HEAD_TILT
    assert not started.should_alert
    assert alerted.state is DrowsinessState.HEAD_TILT_ALERT
    assert alerted.should_alert
    assert alerted.current_head_tilt_seconds == pytest.approx(0.76)


def test_head_tilt_hysteresis_requires_return_near_neutral():
    detector = TemporalDrowsinessDetector(
        DetectorConfig(head_tilt_threshold=25.0, head_tilt_recovered_threshold=18.0)
    )
    detector.update(measurement(0.0, 1.0, pitch_delta=-30.0))
    still_tilted = detector.update(measurement(0.2, 1.0, pitch_delta=-20.0))
    recovered = detector.update(measurement(0.3, 1.0, pitch_delta=-17.0))
    assert still_tilted.state is DrowsinessState.HEAD_TILT
    assert recovered.state is DrowsinessState.AWAKE


def test_sustained_side_tilt_alerts():
    detector = TemporalDrowsinessDetector(
        DetectorConfig(head_side_tilt_threshold=15.0, head_tilt_alert_seconds=0.75)
    )
    started = detector.update(measurement(0.0, 1.0, roll_delta=16.0))
    alerted = detector.update(measurement(0.76, 1.0, roll_delta=16.0))
    assert started.state is DrowsinessState.HEAD_TILT
    assert alerted.state is DrowsinessState.HEAD_TILT_ALERT
    assert alerted.should_alert
    assert alerted.head_roll_delta == pytest.approx(16.0)


def test_head_tilt_thresholds_must_have_valid_hysteresis():
    with pytest.raises(ValueError, match="head_tilt_recovered_threshold"):
        TemporalDrowsinessDetector(
            DetectorConfig(head_tilt_threshold=20.0, head_tilt_recovered_threshold=20.0)
        )


def test_perclos_excludes_a_normal_blink():
    detector = TemporalDrowsinessDetector(DetectorConfig())
    detector.update(measurement(0.0, 1.0))
    detector.update(measurement(0.1, 0.0))
    assessment = detector.update(measurement(0.3, 1.0))
    assert assessment.blink_count == 1
    assert assessment.perclos_60_seconds == 0.0
    assert assessment.slow_blinks_last_minute == 0


def test_perclos_raises_drowsy_warning_after_enough_valid_observation():
    detector = TemporalDrowsinessDetector(
        DetectorConfig(
            perclos_minimum_observation_seconds=0.5,
            perclos_drowsy_threshold=0.20,
            alert_after_closed_seconds=2.0,
            maximum_sample_gap_seconds=1.0,
        )
    )
    detector.update(measurement(0.0, 1.0))
    detector.update(measurement(0.1, 0.0))
    assessment = detector.update(measurement(0.7, 0.0))
    assert assessment.perclos_60_seconds > 0.20
    assert assessment.level is DrowsinessLevel.DROWSY
    assert assessment.alarm_severity is AlarmSeverity.WARNING
    assert assessment.should_alert


def test_perclos_removes_closed_time_outside_window():
    detector = TemporalDrowsinessDetector(
        DetectorConfig(
            perclos_short_window_seconds=1.0,
            perclos_window_seconds=1.0,
            recent_closure_window_seconds=1.0,
            alert_after_closed_seconds=2.0,
        )
    )
    detector.update(measurement(0.0, 1.0))
    detector.update(measurement(0.1, 0.0))
    detector.update(measurement(0.7, 0.0))
    detector.update(measurement(0.8, 1.0))
    detector.update(measurement(1.2, 1.0))
    assessment = detector.update(measurement(1.8, 1.0))
    assert assessment.perclos_60_seconds == pytest.approx(0.0)


def test_tracking_loss_is_confirmed_only_after_configured_duration():
    detector = TemporalDrowsinessDetector(DetectorConfig(tracking_lost_seconds=2.0))
    first = detector.tracking_lost(10.0)
    confirmed = detector.tracking_lost(12.1)
    assert first.level is DrowsinessLevel.NORMAL
    assert confirmed.level is DrowsinessLevel.TRACKING_LOST
    assert confirmed.current_tracking_lost_seconds == pytest.approx(2.1)
    assert confirmed.confidence == 0.0


def test_perclos_is_time_based_and_independent_of_frame_rate():
    config = DetectorConfig(
        alert_after_closed_seconds=5.0,
        maximum_sample_gap_seconds=0.2,
        perclos_minimum_observation_seconds=10.0,
    )

    def run(step):
        detector = TemporalDrowsinessDetector(config)
        timestamp = 0.0
        assessment = None
        while timestamp <= 2.0 + 1e-9:
            openness = 0.0 if 0.5 <= timestamp < 1.2 else 1.0
            assessment = detector.update(measurement(timestamp, openness))
            timestamp += step
        assert assessment is not None
        return assessment.perclos_60_seconds

    assert run(0.1) == pytest.approx(run(0.05), abs=0.03)
