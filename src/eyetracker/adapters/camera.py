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
                f"No se pudo abrir la cámara {self.config.device_index}. "
                "Comprueba los permisos de cámara del terminal."
            )
        self._capture = capture

    def read(self) -> VideoFrame | None:
        if self._capture is None:
            raise RuntimeError("La cámara no está iniciada")
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
