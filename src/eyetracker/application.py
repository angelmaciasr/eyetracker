from __future__ import annotations

from dataclasses import dataclass

from .adapters.alarm import SoundAlarm
from .adapters.repository import JsonCalibrationRepository
from .config import CalibrationConfig
from .domain import (
    CalibrationPhase,
    CalibrationProfile,
    TrackingStatus,
    VideoFrame,
)
from .ports import CameraPort, DisplayPort, FaceTrackerPort
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
    alarm: SoundAlarm
    display: DisplayPort
    repository: JsonCalibrationRepository
    config: CalibrationConfig

    def run(self) -> CalibrationProfile:
        wait_for_first_step = True
        while True:
            try:
                return self._run_once(wait_for_first_step)
            except CalibrationError as error:
                print(f"Calibración no válida: {error}. Se reiniciará.")
                self._wait_for_space("CALIBRACIÓN FALLIDA - ESPACIO PARA REINICIAR")
                wait_for_first_step = False

    def _run_once(self, wait_for_first_step: bool) -> CalibrationProfile:
        self.measurement.reset()
        self.calibrator.begin()
        open_started: float | None = None

        # Fase 1: aprender la apertura habitual.
        if wait_for_first_step:
            self._wait_for_space("FRONTAL - pulsa ESPACIO para empezar")
        while True:
            frame = self.camera.read()
            if frame is None:
                continue
            face = self.tracker.track(frame)
            raw = self.measurement.measure(face) if face.status is TrackingStatus.VALID else None
            if raw is not None and raw.reliable:
                open_started = open_started or raw.timestamp
                self.calibrator.add_sample(CalibrationPhase.OPEN, raw)
                elapsed = raw.timestamp - open_started
            else:
                elapsed = 0.0 if open_started is None else frame.timestamp - open_started
            progress = elapsed / self.config.open_seconds
            self.display.render_calibration(
                frame, face, raw, "Mira a la cámara con naturalidad", progress
            )
            self._check_quit()
            if open_started is not None and elapsed >= self.config.open_seconds:
                break
        self.alarm.notify()

        left_open, right_open = self.calibrator.provisional_open_ear()
        combined_open = (left_open + right_open) / 2.0
        closed_threshold = combined_open * self.config.provisional_closed_ratio
        reopened_threshold = combined_open * self.config.provisional_reopened_ratio

        # Fase 2: aprender cómo deforma el ángulo vertical la apertura aparente.
        frame = self._collect_pose_phase(
            CalibrationPhase.LOOK_DOWN,
            "Mira hacia ABAJO con los ojos abiertos",
        )
        frame = self._collect_pose_phase(
            CalibrationPhase.LOOK_UP,
            "Mira hacia ARRIBA con los ojos abiertos",
        )
        frame = self._collect_pose_phase(
            CalibrationPhase.LOOK_LEFT,
            "Mira hacia tu IZQUIERDA con los ojos abiertos",
        )
        frame = self._collect_pose_phase(
            CalibrationPhase.LOOK_RIGHT,
            "Mira hacia tu DERECHA con los ojos abiertos",
        )

        # Fase 3: medir cinco parpadeos naturales con un umbral provisional.
        self._wait_for_space("PARPADEOS - pulsa ESPACIO para empezar")
        phase_started = frame.timestamp
        in_blink = False
        blink_started = 0.0
        left_min = left_open
        right_min = right_open
        blink_count = 0
        while blink_count < self.config.required_blinks:
            frame = self.camera.read()
            if frame is None:
                continue
            face = self.tracker.track(frame)
            raw = self.measurement.measure(face) if face.status is TrackingStatus.VALID else None
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
            progress = blink_count / self.config.required_blinks
            self.display.render_calibration(
                frame,
                face,
                raw,
                f"Parpadea natural: {blink_count}/{self.config.required_blinks}",
                progress,
            )
            self._check_quit()
            if frame.timestamp - phase_started > self.config.blink_timeout_seconds:
                raise CalibrationError("No se detectaron los parpadeos a tiempo")
        self.alarm.notify()

        # Fase 4: exigir un cierre continuo y recoger su referencia geométrica.
        self._wait_for_space("OJOS CERRADOS - pulsa ESPACIO para empezar")
        phase_started = frame.timestamp
        closed_started: float | None = None
        while True:
            frame = self.camera.read()
            if frame is None:
                continue
            face = self.tracker.track(frame)
            raw = self.measurement.measure(face) if face.status is TrackingStatus.VALID else None
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
                "Mantén ambos ojos cerrados",
                closed_elapsed / self.config.closed_seconds,
            )
            self._check_quit()
            if closed_elapsed >= self.config.closed_seconds:
                break
            if frame.timestamp - phase_started > self.config.closed_timeout_seconds:
                raise CalibrationError("No se obtuvo un cierre mantenido")

        profile = self.calibrator.finish()
        self.alarm.notify()
        self.repository.save(profile)
        return profile

    def _collect_pose_phase(
        self,
        phase: CalibrationPhase,
        message: str,
    ) -> VideoFrame:
        self._wait_for_space(f"{message} - pulsa ESPACIO")
        settle_started: float | None = None
        collect_started: float | None = None
        while True:
            frame = self.camera.read()
            if frame is None:
                continue
            face = self.tracker.track(frame)
            raw = self.measurement.measure(face) if face.status is TrackingStatus.VALID else None
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
                message,
                elapsed / self.config.pose_seconds,
            )
            self._check_quit()
            if collect_started is not None and elapsed >= self.config.pose_seconds:
                self.alarm.notify()
                return frame

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
    alarm: SoundAlarm
    display: DisplayPort

    def run(self) -> str:
        muted = False
        self.measurement.reset()
        self.detector.reset()
        while True:
            frame = self.camera.read()
            if frame is None:
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
            if assessment.should_alert and not muted:
                self.alarm.activate(assessment.reason or "eyes_closed")
            else:
                self.alarm.deactivate()
            self.display.render_monitoring(frame, face, raw, relative, assessment)
            key = self.display.poll_key()
            if key == "quit":
                return "quit"
            if key == "recalibrate":
                self.alarm.deactivate()
                return "recalibrate"
            if key == "mute":
                muted = not muted
                if muted:
                    self.alarm.deactivate()
