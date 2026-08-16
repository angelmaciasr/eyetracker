from __future__ import annotations

from collections import deque

from ..config import DetectorConfig
from ..domain import (
    AlarmSeverity,
    CalibrationProfile,
    DrowsinessAssessment,
    DrowsinessLevel,
    DrowsinessState,
    RelativeEyeMeasurement,
)


class TemporalDrowsinessDetector:
    """Time-based eye/head detector with rolling drowsiness metrics."""

    def __init__(self, config: DetectorConfig) -> None:
        if config.reopened_threshold <= config.closed_threshold:
            raise ValueError("reopened_threshold must be greater than closed_threshold")
        if config.head_tilt_recovered_threshold >= config.head_tilt_threshold:
            raise ValueError("head_tilt_recovered_threshold must be lower than head_tilt_threshold")
        if config.head_side_tilt_recovered_threshold >= config.head_side_tilt_threshold:
            raise ValueError(
                "head_side_tilt_recovered_threshold must be lower than head_side_tilt_threshold"
            )
        if config.perclos_short_window_seconds > config.perclos_window_seconds:
            raise ValueError("perclos_short_window_seconds must not exceed perclos_window_seconds")
        if not 0.0 <= config.perclos_drowsy_threshold <= 1.0:
            raise ValueError("perclos_drowsy_threshold must be between zero and one")
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
        self._tracking_lost_started: float | None = None
        self._last_valid_timestamp: float | None = None
        self._valid_intervals: deque[tuple[float, float]] = deque()
        self._closed_intervals: deque[tuple[float, float]] = deque()
        self._closure_events: deque[tuple[float, float]] = deque()
        self._perclos_closed_started: float | None = None
        self._perclos_candidate_intervals: list[tuple[float, float]] = []
        self._measurement_confidence = 1.0

    def update(self, measurement: RelativeEyeMeasurement) -> DrowsinessAssessment:
        self._measurement_confidence = measurement.pose_confidence
        if not measurement.pose_valid:
            self._tracking_lost_started = None
            self._last_valid_timestamp = None
            self._reset_active_events()
            return self._assessment(
                measurement.timestamp,
                DrowsinessState.CALIBRATION_RANGE_ALERT,
                0.0,
                True,
                "head_pose_outside_calibrated_range",
                pitch_delta=measurement.pitch_delta,
                roll_delta=measurement.roll_delta,
                level=DrowsinessLevel.IMMEDIATE_ALERT,
                severity=AlarmSeverity.URGENT,
                confidence=0.0,
            )

        head_assessment = self._update_head_tilt(measurement)
        if head_assessment is not None:
            self._tracking_lost_started = None
            self._last_valid_timestamp = None
            return head_assessment
        if not measurement.reliable:
            return self.tracking_lost(measurement.timestamp)

        self._tracking_lost_started = None
        self._record_valid_interval(measurement.timestamp)
        openness = measurement.combined_openness
        self._update_perclos_candidate(measurement.timestamp, openness)
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
                return self._assessment(
                    measurement.timestamp,
                    DrowsinessState.ALERT,
                    duration,
                    True,
                    "eyes_closed_too_long",
                    level=DrowsinessLevel.IMMEDIATE_ALERT,
                    severity=AlarmSeverity.URGENT,
                )
            state = (
                DrowsinessState.EYES_CLOSED
                if duration > self.maximum_blink_seconds
                else DrowsinessState.BLINKING
            )
            return self._assessment(measurement.timestamp, state, duration, False, None)

        if self._closure_started is not None:
            duration = max(0.0, measurement.timestamp - self._closure_started)
            if self.config.minimum_blink_seconds <= duration <= self.maximum_blink_seconds:
                self._last_blink = duration
                self._blink_count += 1
            elif duration >= self.config.slow_blink_seconds:
                self._closure_events.append((measurement.timestamp, duration))
                self._closed_intervals.extend(self._perclos_candidate_intervals)
            self._perclos_candidate_intervals.clear()
        self._closure_started = None
        self._closed = False
        return self._assessment(measurement.timestamp, DrowsinessState.AWAKE, 0.0, False, None)

    def tracking_lost(self, timestamp: float) -> DrowsinessAssessment:
        self._reset_active_events()
        self._last_valid_timestamp = None
        if self._tracking_lost_started is None:
            self._tracking_lost_started = timestamp
        duration = max(0.0, timestamp - self._tracking_lost_started)
        confirmed = duration >= self.config.tracking_lost_seconds
        confidence = max(0.0, 1.0 - duration / max(self.config.tracking_lost_seconds, 1e-6))
        return self._assessment(
            timestamp,
            DrowsinessState.TRACKING_LOST,
            0.0,
            False,
            "face_or_eyes_not_reliable",
            level=DrowsinessLevel.TRACKING_LOST if confirmed else DrowsinessLevel.NORMAL,
            confidence=confidence,
            tracking_lost=duration,
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
            level=DrowsinessLevel.IMMEDIATE_ALERT if should_alert else None,
            severity=AlarmSeverity.URGENT if should_alert else AlarmSeverity.NONE,
        )

    def _record_valid_interval(self, timestamp: float) -> None:
        if self._last_valid_timestamp is not None:
            gap = timestamp - self._last_valid_timestamp
            if 0.0 < gap <= self.config.maximum_sample_gap_seconds:
                self._valid_intervals.append((self._last_valid_timestamp, timestamp))
        self._last_valid_timestamp = timestamp
        self._prune(timestamp)

    def _update_perclos_candidate(self, timestamp: float, openness: float) -> None:
        if openness < self.config.perclos_closed_threshold:
            if self._perclos_closed_started is None:
                self._perclos_closed_started = timestamp
        elif self._perclos_closed_started is not None:
            self._perclos_candidate_intervals.append((self._perclos_closed_started, timestamp))
            self._perclos_closed_started = None

    def _prune(self, timestamp: float) -> None:
        cutoff = timestamp - max(
            self.config.perclos_window_seconds,
            self.config.recent_closure_window_seconds,
        )
        for intervals in (self._valid_intervals, self._closed_intervals):
            while intervals and intervals[0][1] <= cutoff:
                intervals.popleft()
        while self._closure_events and self._closure_events[0][0] <= cutoff:
            self._closure_events.popleft()

    @staticmethod
    def _overlap(intervals: deque[tuple[float, float]], start: float, end: float) -> float:
        return sum(max(0.0, min(last, end) - max(first, start)) for first, last in intervals)

    def _metrics(self, timestamp: float) -> tuple[float, float, float, int, float, float]:
        self._prune(timestamp)
        long_window_start = timestamp - self.config.perclos_window_seconds
        short_window_start = timestamp - self.config.perclos_short_window_seconds
        valid_long = self._overlap(self._valid_intervals, long_window_start, timestamp)
        valid_short = self._overlap(self._valid_intervals, short_window_start, timestamp)
        closed_long = self._overlap(self._closed_intervals, long_window_start, timestamp)
        closed_short = self._overlap(self._closed_intervals, short_window_start, timestamp)

        if self._closure_started is not None:
            ongoing = timestamp - self._closure_started
            if ongoing > self.maximum_blink_seconds:
                candidates = deque(self._perclos_candidate_intervals)
                if self._perclos_closed_started is not None:
                    candidates.append((self._perclos_closed_started, timestamp))
                closed_long += self._overlap(candidates, long_window_start, timestamp)
                closed_short += self._overlap(candidates, short_window_start, timestamp)

        recent_cutoff = timestamp - self.config.recent_closure_window_seconds
        durations = [duration for ended, duration in self._closure_events if ended > recent_cutoff]
        slow_blinks = len(durations)
        average = sum(durations) / slow_blinks if slow_blinks else 0.0
        longest = max(durations, default=0.0)
        return (
            closed_short / valid_short if valid_short > 0.0 else 0.0,
            closed_long / valid_long if valid_long > 0.0 else 0.0,
            valid_long,
            slow_blinks,
            average,
            longest,
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
        level: DrowsinessLevel | None = None,
        confidence: float | None = None,
        severity: AlarmSeverity = AlarmSeverity.NONE,
        tracking_lost: float = 0.0,
    ) -> DrowsinessAssessment:
        perclos_30, perclos_60, valid_seconds, slow_blinks, average, longest = self._metrics(
            timestamp
        )
        if level is None:
            if (
                valid_seconds >= self.config.perclos_minimum_observation_seconds
                and perclos_60 >= self.config.perclos_drowsy_threshold
            ):
                level = DrowsinessLevel.DROWSY
            elif slow_blinks >= self.config.possible_drowsiness_slow_blinks:
                level = DrowsinessLevel.POSSIBLE_DROWSINESS
            else:
                level = DrowsinessLevel.NORMAL
        if severity is AlarmSeverity.NONE and level in (
            DrowsinessLevel.POSSIBLE_DROWSINESS,
            DrowsinessLevel.DROWSY,
        ):
            severity = AlarmSeverity.WARNING
            alert = True
            reason = "perclos_high" if level is DrowsinessLevel.DROWSY else "slow_blinks_detected"
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
            level=level,
            confidence=self._measurement_confidence if confidence is None else confidence,
            alarm_severity=severity,
            perclos_30_seconds=perclos_30,
            perclos_60_seconds=perclos_60,
            valid_observation_seconds=valid_seconds,
            slow_blinks_last_minute=slow_blinks,
            average_closure_seconds=average,
            longest_closure_seconds=longest,
            current_tracking_lost_seconds=tracking_lost,
        )

    def _reset_active_events(self) -> None:
        self._closure_started = None
        self._closed = False
        self._head_tilt_started = None
        self._head_tilt_active = False
        self._perclos_closed_started = None
        self._perclos_candidate_intervals.clear()
