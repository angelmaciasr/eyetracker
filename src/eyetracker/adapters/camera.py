from __future__ import annotations

import time

from ..config import CameraConfig
from ..domain import VideoFrame


class OpenCVCamera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture = None
        self._frame_id = 0

    def start(self) -> None:
        import cv2

        capture = cv2.VideoCapture(self.config.device_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera {self.config.device_index}. "
                "Check camera permissions for your terminal."
            )
        self._capture = capture

    def read(self) -> VideoFrame | None:
        if self._capture is None:
            raise RuntimeError("The camera has not been started")
        ok, image = self._capture.read()
        if not ok:
            return None
        self._frame_id += 1
        height, width = image.shape[:2]
        return VideoFrame(self._frame_id, time.monotonic(), width, height, image)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def is_finished(self) -> bool:
        return False


class VideoFileCamera:
    """Camera-compatible source for reproducible runs against recorded videos."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._capture = None
        self._frame_id = 0
        self._fps = 30.0
        self._finished = False

    def start(self) -> None:
        import cv2

        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video file: {self.path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps > 0.0:
            self._fps = fps
        self._capture = capture
        self._finished = False

    def read(self) -> VideoFrame | None:
        if self._capture is None:
            raise RuntimeError("The video source has not been started")
        ok, image = self._capture.read()
        if not ok:
            self._finished = True
            return None
        self._frame_id += 1
        height, width = image.shape[:2]
        timestamp = self._frame_id / self._fps
        return VideoFrame(self._frame_id, timestamp, width, height, image)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def is_finished(self) -> bool:
        return self._finished
