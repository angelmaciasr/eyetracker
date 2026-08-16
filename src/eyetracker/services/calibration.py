from __future__ import annotations

import statistics

from ..config import CalibrationConfig, HeadPoseConfig
from ..domain import (
    CalibrationPhase,
    CalibrationProfile,
    RawEyeMeasurement,
    RelativeEyeMeasurement,
)
from .head_pose import signed_angular_delta


class CalibrationError(RuntimeError):
    pass


class PersonalCalibrationService:
    def __init__(
        self,
        config: CalibrationConfig,
        head_pose_config: HeadPoseConfig | None = None,
    ) -> None:
        self.config = config
        self.head_pose_config = head_pose_config or HeadPoseConfig()
        self.profile: CalibrationProfile | None = None
        self.begin()

    def begin(self) -> None:
        self._open: list[RawEyeMeasurement] = []
        self._pose_open: dict[CalibrationPhase, list[RawEyeMeasurement]] = {
            CalibrationPhase.LOOK_DOWN: [],
            CalibrationPhase.LOOK_UP: [],
            CalibrationPhase.LOOK_LEFT: [],
            CalibrationPhase.LOOK_RIGHT: [],
        }
        self._closed: list[RawEyeMeasurement] = []
        self._blink_samples: list[RawEyeMeasurement] = []
        self._blink_durations: list[float] = []
        self._blink_minima: list[tuple[float, float]] = []

    def add_sample(self, phase: CalibrationPhase, sample: RawEyeMeasurement) -> None:
        if not sample.reliable:
            return
        if phase is CalibrationPhase.OPEN:
            self._open.append(sample)
        elif phase in self._pose_open:
            self._pose_open[phase].append(sample)
        elif phase is CalibrationPhase.CLOSED:
            self._closed.append(sample)
        else:
            self._blink_samples.append(sample)

    def add_blink(self, duration: float, left_min: float, right_min: float) -> None:
        if duration > 0:
            self._blink_durations.append(duration)
            self._blink_minima.append((left_min, right_min))

    def clear_closed_samples(self) -> None:
        self._closed.clear()

    def clear_phase(self, phase: CalibrationPhase) -> None:
        if phase is CalibrationPhase.OPEN:
            self._open.clear()
        elif phase in self._pose_open:
            self._pose_open[phase].clear()
        elif phase is CalibrationPhase.BLINKING:
            self._blink_samples.clear()
            self._blink_durations.clear()
            self._blink_minima.clear()
        elif phase is CalibrationPhase.CLOSED:
            self._closed.clear()

    def provisional_open_ear(self) -> tuple[float, float]:
        if len(self._open) < self.config.minimum_samples:
            raise CalibrationError("Not enough open-eye samples")
        pose_samples = sum(sample.head_pose is not None for sample in self._open)
        if pose_samples < self.config.minimum_samples:
            raise CalibrationError("Head pose was not detected reliably")
        return (
            statistics.median(sample.left_ear for sample in self._open),
            statistics.median(sample.right_ear for sample in self._open),
        )

    def validate_pose_phase(
        self,
        phase: CalibrationPhase,
        counterpart: CalibrationPhase | None = None,
    ) -> None:
        mapping = {
            CalibrationPhase.LOOK_DOWN: (
                "pitch",
                self.head_pose_config.minimum_calibration_pitch_span,
                "up and down",
            ),
            CalibrationPhase.LOOK_UP: (
                "pitch",
                self.head_pose_config.minimum_calibration_pitch_span,
                "up and down",
            ),
            CalibrationPhase.LOOK_LEFT: (
                "yaw",
                self.head_pose_config.minimum_calibration_yaw_span,
                "left and right",
            ),
            CalibrationPhase.LOOK_RIGHT: (
                "yaw",
                self.head_pose_config.minimum_calibration_yaw_span,
                "left and right",
            ),
        }
        if phase not in mapping:
            raise ValueError(f"{phase.value} is not a head-pose phase")
        samples = [sample for sample in self._pose_open[phase] if sample.head_pose is not None]
        if len(samples) < self.config.minimum_samples:
            raise CalibrationError("Not enough reliable head-pose samples")
        axis, minimum_span, instruction = mapping[phase]
        neutral_poses = [sample.head_pose for sample in self._open if sample.head_pose is not None]
        if not neutral_poses:
            raise CalibrationError("The neutral head pose is missing")
        neutral = statistics.median(getattr(pose, axis) for pose in neutral_poses)
        angle = statistics.median(
            getattr(sample.head_pose, axis) for sample in samples if sample.head_pose
        )
        if abs(signed_angular_delta(angle, neutral)) < max(3.0, minimum_span / 2.0):
            raise CalibrationError(f"Move your head farther {instruction}")
        if counterpart is not None:
            left_open, right_open = self.provisional_open_ear()
            self._axis_anchors(
                (counterpart, phase),
                axis,
                neutral,
                minimum_span,
                instruction,
                left_open,
                right_open,
            )

    def validate_blink_phase(self, required_blinks: int) -> None:
        if len(self._blink_durations) < required_blinks:
            raise CalibrationError(
                f"Only {len(self._blink_durations)}/{required_blinks} blinks were detected"
            )

    def validate_closed_phase(self) -> None:
        left_open, right_open = self.provisional_open_ear()
        if len(self._closed) < self.config.minimum_samples:
            raise CalibrationError("Not enough reliable closed-eye samples")
        left_closed = statistics.median(sample.left_ear for sample in self._closed)
        right_closed = statistics.median(sample.right_ear for sample in self._closed)
        if left_open - left_closed < 0.025 or right_open - right_closed < 0.025:
            raise CalibrationError("Open and closed eye references are too similar")
        if left_closed >= left_open * 0.82 or right_closed >= right_open * 0.82:
            raise CalibrationError("The closed-eye reference is invalid")

    def finish(self) -> CalibrationProfile:
        if len(self._open) < self.config.minimum_samples:
            raise CalibrationError("Not enough reliable open-eye samples")
        if len(self._closed) < self.config.minimum_samples:
            raise CalibrationError("Not enough reliable closed-eye samples")
        if not self._blink_durations:
            raise CalibrationError("No blinks were detected")

        left_open = statistics.median(s.left_ear for s in self._open)
        right_open = statistics.median(s.right_ear for s in self._open)
        left_closed = statistics.median(s.left_ear for s in self._closed)
        right_closed = statistics.median(s.right_ear for s in self._closed)
        if left_open - left_closed < 0.025 or right_open - right_closed < 0.025:
            raise CalibrationError("Open and closed eye references are too similar")
        if left_closed >= left_open * 0.82 or right_closed >= right_open * 0.82:
            raise CalibrationError("The closed-eye reference is invalid")

        typical = statistics.median(self._blink_durations)
        max_normal = min(max(typical * 2.0, 0.25), 0.60)
        left_min = statistics.median(pair[0] for pair in self._blink_minima)
        right_min = statistics.median(pair[1] for pair in self._blink_minima)
        medians = (left_open, right_open)
        deviations = [
            abs(sample.left_ear - medians[0]) + abs(sample.right_ear - medians[1])
            for sample in self._open
        ]
        open_poses = [sample.head_pose for sample in self._open if sample.head_pose is not None]
        neutral_pitch = statistics.median(pose.pitch for pose in open_poses) if open_poses else 0.0
        neutral_yaw = statistics.median(pose.yaw for pose in open_poses) if open_poses else 0.0
        neutral_roll = statistics.median(pose.roll for pose in open_poses) if open_poses else 0.0

        pitch_lower, pitch_upper = self._axis_anchors(
            (CalibrationPhase.LOOK_DOWN, CalibrationPhase.LOOK_UP),
            "pitch",
            neutral_pitch,
            self.head_pose_config.minimum_calibration_pitch_span,
            "up and down",
            left_open,
            right_open,
        )
        yaw_lower, yaw_upper = self._axis_anchors(
            (CalibrationPhase.LOOK_LEFT, CalibrationPhase.LOOK_RIGHT),
            "yaw",
            neutral_yaw,
            self.head_pose_config.minimum_calibration_yaw_span,
            "left and right",
            left_open,
            right_open,
        )
        self.profile = CalibrationProfile(
            left_open_ear=left_open,
            left_closed_ear=left_closed,
            right_open_ear=right_open,
            right_closed_ear=right_closed,
            typical_blink_seconds=typical,
            maximum_normal_blink_seconds=max_normal,
            left_blink_min_ear=left_min,
            right_blink_min_ear=right_min,
            measurement_noise=statistics.median(deviations) / 2.0,
            neutral_pitch=neutral_pitch,
            neutral_yaw=neutral_yaw,
            neutral_roll=neutral_roll,
            pitch_lower_delta=pitch_lower[0],
            pitch_upper_delta=pitch_upper[0],
            left_open_ear_at_pitch_lower=pitch_lower[1],
            right_open_ear_at_pitch_lower=pitch_lower[2],
            left_open_ear_at_pitch_upper=pitch_upper[1],
            right_open_ear_at_pitch_upper=pitch_upper[2],
            yaw_lower_delta=yaw_lower[0],
            yaw_upper_delta=yaw_upper[0],
            left_open_ear_at_yaw_lower=yaw_lower[1],
            right_open_ear_at_yaw_lower=yaw_lower[2],
            left_open_ear_at_yaw_upper=yaw_upper[1],
            right_open_ear_at_yaw_upper=yaw_upper[2],
        )
        return self.profile

    def _axis_anchors(
        self,
        phases: tuple[CalibrationPhase, CalibrationPhase],
        axis: str,
        neutral_angle: float,
        minimum_span: float,
        instruction: str,
        left_open: float,
        right_open: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        anchors: list[tuple[float, float, float]] = []
        for phase in phases:
            samples = [sample for sample in self._pose_open[phase] if sample.head_pose is not None]
            if samples:
                angle = statistics.median(
                    getattr(sample.head_pose, axis) for sample in samples if sample.head_pose
                )
                anchors.append(
                    (
                        signed_angular_delta(angle, neutral_angle),
                        statistics.median(sample.left_ear for sample in samples),
                        statistics.median(sample.right_ear for sample in samples),
                    )
                )
        if not anchors:
            return (-20.0, left_open, right_open), (20.0, left_open, right_open)
        if len(anchors) != 2:
            raise CalibrationError(f"A head-pose sample is missing: look {instruction}")
        anchors.sort(key=lambda anchor: anchor[0])
        lower, upper = anchors
        if upper[0] - lower[0] < minimum_span or lower[0] >= -3.0 or upper[0] <= 3.0:
            raise CalibrationError(f"Move your head farther {instruction} during calibration")
        return lower, upper

    def load_profile(self, profile: CalibrationProfile) -> None:
        self.profile = profile

    def is_calibrated(self) -> bool:
        return self.profile is not None

    def normalize(self, sample: RawEyeMeasurement) -> RelativeEyeMeasurement:
        if self.profile is None:
            raise CalibrationError("The system has not been calibrated yet")

        def relative(value: float, closed: float, opened: float) -> float:
            normalized = (value - closed) / (opened - closed)
            return min(1.2, max(0.0, normalized))

        pose_valid = True
        pose_confidence = 1.0
        pitch_delta: float | None = None
        roll_delta: float | None = None
        left_ear = sample.left_ear
        right_ear = sample.right_ear
        if sample.head_pose is not None:
            pitch_delta = signed_angular_delta(sample.head_pose.pitch, self.profile.neutral_pitch)
            yaw_delta = signed_angular_delta(sample.head_pose.yaw, self.profile.neutral_yaw)
            roll_delta = signed_angular_delta(sample.head_pose.roll, self.profile.neutral_roll)
            pitch_lower = self.profile.pitch_lower_delta
            pitch_upper = self.profile.pitch_upper_delta
            yaw_lower = self.profile.yaw_lower_delta
            yaw_upper = self.profile.yaw_upper_delta
            pose_valid = (
                pitch_lower - self.head_pose_config.pitch_range_margin
                <= pitch_delta
                <= pitch_upper + self.head_pose_config.pitch_range_margin
                and yaw_lower - self.head_pose_config.yaw_range_margin
                <= yaw_delta
                <= yaw_upper + self.head_pose_config.yaw_range_margin
                and abs(roll_delta) <= self.head_pose_config.maximum_roll_delta
            )
            pitch_limit = max(
                abs(pitch_lower - self.head_pose_config.pitch_range_margin),
                abs(pitch_upper + self.head_pose_config.pitch_range_margin),
                1e-6,
            )
            yaw_limit = max(
                abs(yaw_lower - self.head_pose_config.yaw_range_margin),
                abs(yaw_upper + self.head_pose_config.yaw_range_margin),
                1e-6,
            )
            pose_ratio = max(
                abs(pitch_delta) / pitch_limit,
                abs(yaw_delta) / yaw_limit,
                abs(roll_delta) / max(self.head_pose_config.maximum_roll_delta, 1e-6),
            )
            pose_confidence = max(0.0, 1.0 - 0.5 * pose_ratio) if pose_valid else 0.0
            pitch_left = self._expected_open_ear(
                pitch_delta,
                self.profile.left_open_ear,
                self.profile.left_open_ear_at_pitch_lower,
                self.profile.left_open_ear_at_pitch_upper,
                pitch_lower,
                pitch_upper,
            )
            pitch_right = self._expected_open_ear(
                pitch_delta,
                self.profile.right_open_ear,
                self.profile.right_open_ear_at_pitch_lower,
                self.profile.right_open_ear_at_pitch_upper,
                pitch_lower,
                pitch_upper,
            )
            yaw_left = self._expected_open_ear(
                yaw_delta,
                self.profile.left_open_ear,
                self.profile.left_open_ear_at_yaw_lower,
                self.profile.left_open_ear_at_yaw_upper,
                yaw_lower,
                yaw_upper,
            )
            yaw_right = self._expected_open_ear(
                yaw_delta,
                self.profile.right_open_ear,
                self.profile.right_open_ear_at_yaw_lower,
                self.profile.right_open_ear_at_yaw_upper,
                yaw_lower,
                yaw_upper,
            )
            expected_left = max(
                pitch_left + yaw_left - self.profile.left_open_ear,
                self.profile.left_open_ear * 0.40,
            )
            expected_right = max(
                pitch_right + yaw_right - self.profile.right_open_ear,
                self.profile.right_open_ear * 0.40,
            )
            left_ear *= self.profile.left_open_ear / expected_left
            right_ear *= self.profile.right_open_ear / expected_right

        left = relative(left_ear, self.profile.left_closed_ear, self.profile.left_open_ear)
        right = relative(
            right_ear,
            self.profile.right_closed_ear,
            self.profile.right_open_ear,
        )
        return RelativeEyeMeasurement(
            timestamp=sample.timestamp,
            left_openness=left,
            right_openness=right,
            combined_openness=(left + right) / 2.0,
            reliable=sample.reliable and pose_valid,
            head_pose=sample.head_pose,
            pose_valid=pose_valid,
            pitch_delta=pitch_delta,
            roll_delta=roll_delta,
            pose_confidence=pose_confidence,
        )

    @staticmethod
    def _expected_open_ear(
        pitch_delta: float,
        neutral_ear: float,
        lower_ear: float,
        upper_ear: float,
        lower_pitch: float,
        upper_pitch: float,
    ) -> float:
        lower_ear = lower_ear if lower_ear > 0 else neutral_ear
        upper_ear = upper_ear if upper_ear > 0 else neutral_ear
        if pitch_delta < 0 and lower_pitch < 0:
            ratio = pitch_delta / lower_pitch
            expected = neutral_ear + (lower_ear - neutral_ear) * ratio
        elif pitch_delta > 0 and upper_pitch > 0:
            ratio = pitch_delta / upper_pitch
            expected = neutral_ear + (upper_ear - neutral_ear) * ratio
        else:
            expected = neutral_ear
        return max(expected, neutral_ear * 0.40, 1e-6)
