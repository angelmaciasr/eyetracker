from __future__ import annotations

from typing import Protocol

from .domain import (
    DrowsinessAssessment,
    FaceObservation,
    RawEyeMeasurement,
    RelativeEyeMeasurement,
    VideoFrame,
)


class CameraPort(Protocol):
    def start(self) -> None: ...
    def read(self) -> VideoFrame | None: ...
    def stop(self) -> None: ...
    def is_finished(self) -> bool: ...


class AlarmPort(Protocol):
    def activate(self, reason: str) -> None: ...
    def deactivate(self) -> None: ...
    def warn(self, reason: str) -> None: ...
    def notify(self) -> None: ...


class EventLoggerPort(Protocol):
    def begin_session(self) -> None: ...
    def record(self, assessment: DrowsinessAssessment) -> None: ...
    def end_session(self) -> None: ...


class FaceTrackerPort(Protocol):
    def initialize(self) -> None: ...
    def track(self, frame: VideoFrame) -> FaceObservation: ...
    def close(self) -> None: ...


class DisplayPort(Protocol):
    def render_monitoring(
        self,
        frame: VideoFrame,
        face: FaceObservation,
        raw: RawEyeMeasurement | None,
        relative: RelativeEyeMeasurement | None,
        assessment: DrowsinessAssessment,
    ) -> None: ...

    def render_calibration(
        self,
        frame: VideoFrame,
        face: FaceObservation,
        raw: RawEyeMeasurement | None,
        message: str,
        progress: float,
    ) -> None: ...

    def poll_key(self) -> str | None: ...
    def close(self) -> None: ...
