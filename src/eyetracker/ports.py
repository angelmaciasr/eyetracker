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
