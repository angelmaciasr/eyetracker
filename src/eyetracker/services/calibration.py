from __future__ import annotations

import statistics

from ..config import CalibrationConfig
from ..domain import (
    CalibrationPhase,
    CalibrationProfile,
    RawEyeMeasurement,
    RelativeEyeMeasurement,
)


class CalibrationError(RuntimeError):
    pass


class PersonalCalibrationService:
    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        self.profile: CalibrationProfile | None = None
        self.begin()

    def begin(self) -> None:
        self._open: list[RawEyeMeasurement] = []
        self._closed: list[RawEyeMeasurement] = []
        self._blink_samples: list[RawEyeMeasurement] = []
        self._blink_durations: list[float] = []
        self._blink_minima: list[tuple[float, float]] = []

    def add_sample(self, phase: CalibrationPhase, sample: RawEyeMeasurement) -> None:
        if not sample.reliable:
            return
        if phase is CalibrationPhase.OPEN:
            self._open.append(sample)
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

    def provisional_open_ear(self) -> tuple[float, float]:
        if len(self._open) < self.config.minimum_samples:
            raise CalibrationError("No hay suficientes muestras con ojos abiertos")
        return (
            statistics.median(sample.left_ear for sample in self._open),
            statistics.median(sample.right_ear for sample in self._open),
        )

    def finish(self) -> CalibrationProfile:
        if len(self._open) < self.config.minimum_samples:
            raise CalibrationError("Faltan muestras fiables con los ojos abiertos")
        if len(self._closed) < self.config.minimum_samples:
            raise CalibrationError("Faltan muestras fiables con los ojos cerrados")
        if not self._blink_durations:
            raise CalibrationError("No se ha detectado ningún parpadeo")

        left_open = statistics.median(s.left_ear for s in self._open)
        right_open = statistics.median(s.right_ear for s in self._open)
        left_closed = statistics.median(s.left_ear for s in self._closed)
        right_closed = statistics.median(s.right_ear for s in self._closed)
        if left_open - left_closed < 0.025 or right_open - right_closed < 0.025:
            raise CalibrationError(
                "La apertura y el cierre no se distinguen; repite la calibración"
            )
        if left_closed >= left_open * 0.82 or right_closed >= right_open * 0.82:
            raise CalibrationError("La referencia de ojos cerrados no es válida")

        typical = statistics.median(self._blink_durations)
        max_normal = min(max(typical * 2.0, 0.25), 0.60)
        left_min = statistics.median(pair[0] for pair in self._blink_minima)
        right_min = statistics.median(pair[1] for pair in self._blink_minima)
        medians = (left_open, right_open)
        deviations = [
            abs(sample.left_ear - medians[0]) + abs(sample.right_ear - medians[1])
            for sample in self._open
        ]
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
        )
        return self.profile

    def load_profile(self, profile: CalibrationProfile) -> None:
        self.profile = profile

    def is_calibrated(self) -> bool:
        return self.profile is not None

    def normalize(self, sample: RawEyeMeasurement) -> RelativeEyeMeasurement:
        if self.profile is None:
            raise CalibrationError("El sistema todavía no está calibrado")

        def relative(value: float, closed: float, opened: float) -> float:
            normalized = (value - closed) / (opened - closed)
            return min(1.2, max(0.0, normalized))

        left = relative(sample.left_ear, self.profile.left_closed_ear, self.profile.left_open_ear)
        right = relative(
            sample.right_ear,
            self.profile.right_closed_ear,
            self.profile.right_open_ear,
        )
        return RelativeEyeMeasurement(
            timestamp=sample.timestamp,
            left_openness=left,
            right_openness=right,
            combined_openness=(left + right) / 2.0,
            reliable=sample.reliable,
        )
