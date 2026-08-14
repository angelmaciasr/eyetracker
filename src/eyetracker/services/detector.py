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
            raise ValueError("reopened_threshold debe superar closed_threshold")
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

    def update(self, measurement: RelativeEyeMeasurement) -> DrowsinessAssessment:
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
        return self._assessment(
            timestamp,
            DrowsinessState.TRACKING_LOST,
            0.0,
            False,
            "face_or_eyes_not_reliable",
        )

    def _assessment(
        self,
        timestamp: float,
        state: DrowsinessState,
        closure: float,
        alert: bool,
        reason: str | None,
    ) -> DrowsinessAssessment:
        return DrowsinessAssessment(
            timestamp=timestamp,
            state=state,
            current_closure_seconds=closure,
            last_blink_seconds=self._last_blink,
            blink_count=self._blink_count,
            should_alert=alert,
            reason=reason,
        )
