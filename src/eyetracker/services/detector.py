from __future__ import annotations

from ..config import DetectorConfig
from ..domain import (
    CalibrationProfile,
    DrowsinessAssessment,
    DrowsinessState,
    RelativeEyeMeasurement,
)


class TemporalDrowsinessDetector:
    def __init__(self, config: DetectorConfig) -> None:
        if config.reopened_threshold <= config.closed_threshold:
            raise ValueError("reopened_threshold must be greater than closed_threshold")
        if config.head_tilt_recovered_threshold >= config.head_tilt_threshold:
            raise ValueError("head_tilt_recovered_threshold must be lower than head_tilt_threshold")
        if config.head_side_tilt_recovered_threshold >= config.head_side_tilt_threshold:
            raise ValueError(
                "head_side_tilt_recovered_threshold must be lower than head_side_tilt_threshold"
            )
        self.config = config
        self.maximum_blink_seconds = config.maximum_blink_seconds
        self.reset()

    def apply_calibration(self, profile: CalibrationProfile) -> None:
        self.maximum_blink_seconds = profile.maximum_normal_blink_seconds

    def reset(self) -> None:
        self._closure_started: float | None = None
        self._closed = False
        self._last_blink: float | None = None
        self._blink_count = 0
        self._head_tilt_started: float | None = None
        self._head_tilt_active = False

    def update(self, measurement: RelativeEyeMeasurement) -> DrowsinessAssessment:
        if not measurement.pose_valid:
            self._closure_started = None
            self._closed = False
            self._head_tilt_started = None
            self._head_tilt_active = False
            return self._assessment(
                measurement.timestamp,
                DrowsinessState.CALIBRATION_RANGE_ALERT,
                0.0,
                True,
                "head_pose_outside_calibrated_range",
                pitch_delta=measurement.pitch_delta,
                roll_delta=measurement.roll_delta,
            )
        head_assessment = self._update_head_tilt(measurement)
        if head_assessment is not None:
            return head_assessment
        if not measurement.reliable:
            return self.tracking_lost(measurement.timestamp)

        openness = measurement.combined_openness
        if self._closed:
            is_closed = openness <= self.config.reopened_threshold
        else:
            is_closed = openness < self.config.closed_threshold

        if is_closed:
            if self._closure_started is None:
                self._closure_started = measurement.timestamp
            self._closed = True
            duration = max(0.0, measurement.timestamp - self._closure_started)
            if duration >= self.config.alert_after_closed_seconds:
                state = DrowsinessState.ALERT
                alert = True
                reason = "eyes_closed_too_long"
            elif duration > self.maximum_blink_seconds:
                state = DrowsinessState.EYES_CLOSED
                alert = False
                reason = None
            else:
                state = DrowsinessState.BLINKING
                alert = False
                reason = None
            return self._assessment(measurement.timestamp, state, duration, alert, reason)

        if self._closure_started is not None:
            duration = max(0.0, measurement.timestamp - self._closure_started)
            if self.config.minimum_blink_seconds <= duration <= self.maximum_blink_seconds:
                self._last_blink = duration
                self._blink_count += 1
        self._closure_started = None
        self._closed = False
        return self._assessment(measurement.timestamp, DrowsinessState.AWAKE, 0.0, False, None)

    def tracking_lost(self, timestamp: float) -> DrowsinessAssessment:
        self._closure_started = None
        self._closed = False
        self._head_tilt_started = None
        self._head_tilt_active = False
        return self._assessment(
            timestamp,
            DrowsinessState.TRACKING_LOST,
            0.0,
            False,
            "face_or_eyes_not_reliable",
        )

    def _update_head_tilt(self, measurement: RelativeEyeMeasurement) -> DrowsinessAssessment | None:
        pitch_delta = measurement.pitch_delta
        roll_delta = measurement.roll_delta
        if pitch_delta is None and roll_delta is None:
            self._head_tilt_started = None
            self._head_tilt_active = False
            return None
        pitch_threshold = (
            self.config.head_tilt_recovered_threshold
            if self._head_tilt_active
            else self.config.head_tilt_threshold
        )
        roll_threshold = (
            self.config.head_side_tilt_recovered_threshold
            if self._head_tilt_active
            else self.config.head_side_tilt_threshold
        )
        pitch_tilted = pitch_delta is not None and abs(pitch_delta) >= pitch_threshold
        roll_tilted = roll_delta is not None and abs(roll_delta) >= roll_threshold
        if not pitch_tilted and not roll_tilted:
            self._head_tilt_started = None
            self._head_tilt_active = False
            return None

        if self._head_tilt_started is None:
            self._head_tilt_started = measurement.timestamp
        self._head_tilt_active = True
        self._closure_started = None
        self._closed = False
        duration = max(0.0, measurement.timestamp - self._head_tilt_started)
        should_alert = duration >= self.config.head_tilt_alert_seconds
        state = DrowsinessState.HEAD_TILT_ALERT if should_alert else DrowsinessState.HEAD_TILT
        return self._assessment(
            measurement.timestamp,
            state,
            0.0,
            should_alert,
            "head_tilt_too_long" if should_alert else None,
            head_tilt=duration,
            pitch_delta=pitch_delta,
            roll_delta=roll_delta,
        )

    def _assessment(
        self,
        timestamp: float,
        state: DrowsinessState,
        closure: float,
        alert: bool,
        reason: str | None,
        head_tilt: float = 0.0,
        pitch_delta: float | None = None,
        roll_delta: float | None = None,
    ) -> DrowsinessAssessment:
        return DrowsinessAssessment(
            timestamp=timestamp,
            state=state,
            current_closure_seconds=closure,
            last_blink_seconds=self._last_blink,
            blink_count=self._blink_count,
            should_alert=alert,
            reason=reason,
            current_head_tilt_seconds=head_tilt,
            head_pitch_delta=pitch_delta,
            head_roll_delta=roll_delta,
        )
