from __future__ import annotations

from dataclasses import dataclass

from .adapters.repository import JsonCalibrationRepository
from .config import CalibrationConfig
from .domain import CalibrationPhase, CalibrationProfile, TrackingStatus
from .ports import AlarmPort, CameraPort, DisplayPort, EventLoggerPort, FaceTrackerPort
from .services.alert_manager import AlertManager
from .services.calibration import CalibrationError, PersonalCalibrationService
from .services.detector import TemporalDrowsinessDetector
from .services.eye_measurement import EAREyeMeasurementService


class UserQuit(RuntimeError):
    pass


@dataclass
class CalibrationController:
    camera: CameraPort
    tracker: FaceTrackerPort
    measurement: EAREyeMeasurementService
    calibrator: PersonalCalibrationService
    alarm: AlarmPort
    display: DisplayPort
    repository: JsonCalibrationRepository
    config: CalibrationConfig

    def run(self) -> CalibrationProfile:
        self.measurement.reset()
        self.calibrator.begin()

        left_open, right_open = self._collect_open_phase()
        combined_open = (left_open + right_open) / 2.0
        closed_threshold = combined_open * self.config.provisional_closed_ratio
        reopened_threshold = combined_open * self.config.provisional_reopened_ratio

        self._collect_pose_phase(
            CalibrationPhase.LOOK_DOWN,
            2,
            "Step 2/7: Keep your eyes naturally open and tilt your head DOWN. "
            "Press SPACE to start and wait for the sound that marks the end.",
            "Step 2/7 in progress: Keep your eyes open and hold your head tilted DOWN.",
        )
        self._collect_pose_phase(
            CalibrationPhase.LOOK_UP,
            3,
            "Step 3/7: Keep your eyes naturally open and tilt your head UP. "
            "Press SPACE to start and wait for the sound that marks the end.",
            "Step 3/7 in progress: Keep your eyes open and hold your head tilted UP.",
            counterpart=CalibrationPhase.LOOK_DOWN,
        )
        self._collect_pose_phase(
            CalibrationPhase.LOOK_LEFT,
            4,
            "Step 4/7: Keep your eyes naturally open and turn your head to the LEFT. "
            "Press SPACE to start and wait for the sound that marks the end.",
            "Step 4/7 in progress: Keep your eyes open and hold your head turned LEFT.",
        )
        self._collect_pose_phase(
            CalibrationPhase.LOOK_RIGHT,
            5,
            "Step 5/7: Keep your eyes naturally open and turn your head to the RIGHT. "
            "Press SPACE to start and wait for the sound that marks the end.",
            "Step 5/7 in progress: Keep your eyes open and hold your head turned RIGHT.",
            counterpart=CalibrationPhase.LOOK_LEFT,
        )
        self._collect_blink_phase(left_open, right_open, closed_threshold, reopened_threshold)
        self._collect_closed_phase(closed_threshold)

        profile = self.calibrator.finish()
        self.repository.save(profile)
        return profile

    def _collect_open_phase(self) -> tuple[float, float]:
        self._wait_for_space(
            "Step 1/7: Keep your eyes naturally open and look straight ahead. "
            "Press SPACE to start and wait for the sound that marks the end."
        )
        while True:
            self.calibrator.clear_phase(CalibrationPhase.OPEN)
            open_started: float | None = None
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue
                face = self.tracker.track(frame)
                raw = (
                    self.measurement.measure(face) if face.status is TrackingStatus.VALID else None
                )
                if raw is not None and raw.reliable:
                    open_started = open_started or raw.timestamp
                    self.calibrator.add_sample(CalibrationPhase.OPEN, raw)
                    elapsed = raw.timestamp - open_started
                else:
                    elapsed = 0.0
                self.display.render_calibration(
                    frame,
                    face,
                    raw,
                    "Step 1/7 in progress: Keep your eyes naturally open and look straight ahead.",
                    elapsed / self.config.open_seconds,
                )
                self._check_quit()
                if open_started is not None and elapsed >= self.config.open_seconds:
                    break
            try:
                result = self.calibrator.provisional_open_ear()
            except CalibrationError as error:
                self._wait_to_retry_step(1, error)
                continue
            self.alarm.notify()
            return result

    def _collect_pose_phase(
        self,
        phase: CalibrationPhase,
        step: int,
        preview_message: str,
        active_message: str,
        counterpart: CalibrationPhase | None = None,
    ) -> None:
        self._wait_for_space(preview_message)
        while True:
            self.calibrator.clear_phase(phase)
            settle_started: float | None = None
            collect_started: float | None = None
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue
                face = self.tracker.track(frame)
                raw = (
                    self.measurement.measure(face) if face.status is TrackingStatus.VALID else None
                )
                settle_started = settle_started or frame.timestamp
                settled = frame.timestamp - settle_started >= self.config.pose_settle_seconds
                if settled and raw is not None and raw.reliable and raw.head_pose is not None:
                    collect_started = collect_started or raw.timestamp
                    self.calibrator.add_sample(phase, raw)
                    elapsed = raw.timestamp - collect_started
                else:
                    elapsed = 0.0
                self.display.render_calibration(
                    frame,
                    face,
                    raw,
                    active_message,
                    elapsed / self.config.pose_seconds,
                )
                self._check_quit()
                if collect_started is not None and elapsed >= self.config.pose_seconds:
                    break
            try:
                self.calibrator.validate_pose_phase(phase, counterpart)
            except CalibrationError as error:
                self._wait_to_retry_step(step, error)
                continue
            self.alarm.notify()
            return

    def _collect_blink_phase(
        self,
        left_open: float,
        right_open: float,
        closed_threshold: float,
        reopened_threshold: float,
    ) -> None:
        self._wait_for_space(
            f"Step 6/7: Blink naturally {self.config.required_blinks} times while looking "
            "straight ahead. Press SPACE to start and wait for the sound that marks the end."
        )
        while True:
            self.calibrator.clear_phase(CalibrationPhase.BLINKING)
            phase_started: float | None = None
            in_blink = False
            blink_started = 0.0
            left_min = left_open
            right_min = right_open
            blink_count = 0
            timed_out = False
            while blink_count < self.config.required_blinks:
                frame = self.camera.read()
                if frame is None:
                    continue
                phase_started = phase_started or frame.timestamp
                face = self.tracker.track(frame)
                raw = (
                    self.measurement.measure(face) if face.status is TrackingStatus.VALID else None
                )
                if raw is not None and raw.reliable:
                    self.calibrator.add_sample(CalibrationPhase.BLINKING, raw)
                    if not in_blink and raw.combined_ear < closed_threshold:
                        in_blink = True
                        blink_started = raw.timestamp
                        left_min, right_min = raw.left_ear, raw.right_ear
                    elif in_blink:
                        left_min = min(left_min, raw.left_ear)
                        right_min = min(right_min, raw.right_ear)
                        if raw.combined_ear > reopened_threshold:
                            duration = raw.timestamp - blink_started
                            if 0.05 <= duration <= 1.0:
                                self.calibrator.add_blink(duration, left_min, right_min)
                                blink_count += 1
                            in_blink = False
                self.display.render_calibration(
                    frame,
                    face,
                    raw,
                    f"Step 6/7 in progress: Natural blinks {blink_count}/"
                    f"{self.config.required_blinks}.",
                    blink_count / self.config.required_blinks,
                )
                self._check_quit()
                if frame.timestamp - phase_started > self.config.blink_timeout_seconds:
                    timed_out = True
                    break
            try:
                if timed_out:
                    raise CalibrationError("The required blinks were not detected in time")
                self.calibrator.validate_blink_phase(self.config.required_blinks)
            except CalibrationError as error:
                self._wait_to_retry_step(6, error)
                continue
            self.alarm.notify()
            return

    def _collect_closed_phase(self, closed_threshold: float) -> None:
        self._wait_for_space(
            "Step 7/7: Close both eyes and keep them closed. Press SPACE to start and "
            "wait for the sound that marks the end."
        )
        while True:
            self.calibrator.clear_phase(CalibrationPhase.CLOSED)
            phase_started: float | None = None
            closed_started: float | None = None
            timed_out = False
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue
                phase_started = phase_started or frame.timestamp
                face = self.tracker.track(frame)
                raw = (
                    self.measurement.measure(face) if face.status is TrackingStatus.VALID else None
                )
                if raw is not None and raw.reliable and raw.combined_ear < closed_threshold:
                    closed_started = closed_started or raw.timestamp
                    self.calibrator.add_sample(CalibrationPhase.CLOSED, raw)
                    closed_elapsed = raw.timestamp - closed_started
                else:
                    if closed_started is not None:
                        self.calibrator.clear_closed_samples()
                    closed_started = None
                    closed_elapsed = 0.0
                self.display.render_calibration(
                    frame,
                    face,
                    raw,
                    "Step 7/7 in progress: Keep both eyes fully closed.",
                    closed_elapsed / self.config.closed_seconds,
                )
                self._check_quit()
                if closed_elapsed >= self.config.closed_seconds:
                    break
                if frame.timestamp - phase_started > self.config.closed_timeout_seconds:
                    timed_out = True
                    break
            try:
                if timed_out:
                    raise CalibrationError("A sustained eye closure was not captured")
                self.calibrator.validate_closed_phase()
            except CalibrationError as error:
                self._wait_to_retry_step(7, error)
                continue
            self.alarm.notify()
            return

    def _wait_to_retry_step(self, step: int, error: CalibrationError) -> None:
        print(f"Step {step}/7 failed: {error}. Only this step will be retried.")
        self._wait_for_space(f"Step {step}/7 failed: {error}. Press SPACE to retry only this step.")

    def _wait_for_space(self, message: str) -> None:
        while True:
            frame = self.camera.read()
            if frame is None:
                continue
            face = self.tracker.track(frame)
            raw = self.measurement.measure(face) if face.status is TrackingStatus.VALID else None
            self.display.render_calibration(frame, face, raw, message, 0.0)
            key = self.display.poll_key()
            if key == "quit":
                raise UserQuit
            if key == "continue":
                return

    def _check_quit(self) -> None:
        if self.display.poll_key() == "quit":
            raise UserQuit


@dataclass
class MonitoringController:
    camera: CameraPort
    tracker: FaceTrackerPort
    measurement: EAREyeMeasurementService
    calibrator: PersonalCalibrationService
    detector: TemporalDrowsinessDetector
    alert_manager: AlertManager
    display: DisplayPort
    event_logger: EventLoggerPort

    def run(self) -> str:
        muted = False
        self.measurement.reset()
        self.detector.reset()
        self.alert_manager.reset()
        self.event_logger.begin_session()
        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    if self.camera.is_finished():
                        return "quit"
                    continue
                face = self.tracker.track(frame)
                raw = None
                relative = None
                if face.status is TrackingStatus.VALID:
                    raw = self.measurement.measure(face)
                if raw is not None and raw.reliable:
                    relative = self.calibrator.normalize(raw)
                    assessment = self.detector.update(relative)
                else:
                    assessment = self.detector.tracking_lost(frame.timestamp)
                self.event_logger.record(assessment)
                self.alert_manager.handle(assessment)
                self.display.render_monitoring(frame, face, raw, relative, assessment)
                key = self.display.poll_key()
                if key == "quit":
                    return "quit"
                if key == "recalibrate":
                    self.alert_manager.reset()
                    return "recalibrate"
                if key == "mute":
                    muted = not muted
                    self.alert_manager.set_muted(muted)
        finally:
            self.alert_manager.reset()
            self.event_logger.end_session()
