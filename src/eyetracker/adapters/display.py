from __future__ import annotations

from collections import deque
from itertools import pairwise

from ..domain import (
    DrowsinessAssessment,
    DrowsinessState,
    FaceObservation,
    RawEyeMeasurement,
    RelativeEyeMeasurement,
    TrackingStatus,
    VideoFrame,
)


class OpenCVDisplay:
    WINDOW_NAME = "Eye Sentinel"

    def __init__(self, closed_threshold: float = 0.75) -> None:
        self._history: deque[float] = deque(maxlen=180)
        self._last_frame_time: float | None = None
        self._fps = 0.0
        self._closed_threshold = closed_threshold

    def render_monitoring(
        self,
        frame: VideoFrame,
        face: FaceObservation,
        raw: RawEyeMeasurement | None,
        relative: RelativeEyeMeasurement | None,
        assessment: DrowsinessAssessment,
    ) -> None:
        import cv2

        canvas = frame.image.copy()
        self._update_fps(frame.timestamp)
        self._draw_eyes(canvas, face)
        if relative is not None:
            self._history.append(relative.combined_openness)
        color = self._state_color(assessment.state)
        cv2.rectangle(canvas, (0, 0), (frame.width, 122), (18, 18, 18), -1)
        self._text(canvas, f"Estado: {assessment.state.value.upper()}", 14, 28, color, 0.72)
        self._text(canvas, f"FPS: {self._fps:4.1f}", 14, 55)
        if raw is not None:
            self._text(canvas, f"EAR I/D: {raw.left_ear:.3f} / {raw.right_ear:.3f}", 150, 55)
        if face.head_pose is not None:
            pose = face.head_pose
            self._text(
                canvas,
                f"Pose P/Y/R: {pose.pitch:+.1f} / {pose.yaw:+.1f} / {pose.roll:+.1f}",
                430,
                55,
            )
        if relative is not None:
            self._text(canvas, f"Apertura: {relative.combined_openness:.2f}", 14, 82)
        self._text(canvas, f"Cierre: {assessment.current_closure_seconds:.2f} s", 210, 82)
        self._text(canvas, f"Parpadeos: {assessment.blink_count}", 14, 109)
        self._text(canvas, "Q salir   R recalibrar   M silenciar", 210, 109, (190, 190, 190))
        self._draw_graph(canvas, y_top=max(135, frame.height - 145), width=360, height=115)
        if assessment.state is DrowsinessState.ALERT:
            cv2.rectangle(canvas, (3, 3), (frame.width - 4, frame.height - 4), color, 8)
            self._center_text(canvas, "ALERTA: ABRE LOS OJOS", frame.height // 2, color, 1.15)
        elif relative is not None and not relative.pose_valid:
            self._center_text(
                canvas,
                "ÁNGULO TODAVÍA NO CALIBRADO",
                frame.height // 2,
                (80, 210, 250),
                0.9,
            )
        cv2.imshow(self.WINDOW_NAME, canvas)

    def render_calibration(
        self,
        frame: VideoFrame,
        face: FaceObservation,
        raw: RawEyeMeasurement | None,
        message: str,
        progress: float,
    ) -> None:
        import cv2

        canvas = frame.image.copy()
        self._draw_eyes(canvas, face)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (frame.width, 112), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        self._center_text(canvas, "CALIBRACIÓN", 32, (90, 220, 255), 0.75)
        self._center_text(canvas, message, 66, (245, 245, 245), 0.75)
        left, right = 80, frame.width - 80
        cv2.rectangle(canvas, (left, 84), (right, 101), (80, 80, 80), 1)
        fill = left + int((right - left) * min(1.0, max(0.0, progress)))
        cv2.rectangle(canvas, (left + 1, 85), (fill, 100), (70, 190, 120), -1)
        if face.status is not TrackingStatus.VALID:
            self._center_text(
                canvas, "No se detecta una cara", frame.height - 40, (60, 80, 255), 0.72
            )
        elif raw is not None:
            pose_text = ""
            if face.head_pose is not None:
                pose = face.head_pose
                pose_text = f"   Pose {pose.pitch:+.1f}/{pose.yaw:+.1f}/{pose.roll:+.1f}"
            self._text(
                canvas,
                f"EAR {raw.left_ear:.3f} / {raw.right_ear:.3f}{pose_text}",
                12,
                frame.height - 15,
            )
        cv2.imshow(self.WINDOW_NAME, canvas)

    def poll_key(self) -> str | None:
        import cv2

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("r"):
            return "recalibrate"
        if key == ord("m"):
            return "mute"
        return None

    def close(self) -> None:
        import cv2

        cv2.destroyAllWindows()

    def _update_fps(self, timestamp: float) -> None:
        if self._last_frame_time is not None:
            instant = 1.0 / max(timestamp - self._last_frame_time, 1e-6)
            self._fps = instant if self._fps == 0 else self._fps * 0.9 + instant * 0.1
        self._last_frame_time = timestamp

    @staticmethod
    def _draw_eyes(canvas, face: FaceObservation) -> None:
        if face.status is not TrackingStatus.VALID:
            return
        import cv2

        height, width = canvas.shape[:2]
        for eye in (face.left_eye, face.right_eye):
            pixels = [(int(point.x * width), int(point.y * height)) for point in eye]
            for point in pixels:
                cv2.circle(canvas, point, 2, (80, 255, 160), -1)
            order = (0, 1, 2, 3, 4, 5, 0)
            for start, end in pairwise(order):
                cv2.line(canvas, pixels[start], pixels[end], (80, 220, 140), 1)

    def _draw_graph(self, canvas, y_top: int, width: int, height: int) -> None:
        import cv2

        x_left = 12
        cv2.rectangle(canvas, (x_left, y_top), (x_left + width, y_top + height), (20, 20, 20), -1)
        threshold_y = y_top + int(height * (1.0 - self._closed_threshold / 1.2))
        cv2.line(canvas, (x_left, threshold_y), (x_left + width, threshold_y), (80, 80, 220), 1)
        if len(self._history) < 2:
            return
        values = list(self._history)
        points = []
        for index, value in enumerate(values):
            x = x_left + int(index * width / max(len(values) - 1, 1))
            y = y_top + int(height * (1.0 - min(1.2, max(0.0, value)) / 1.2))
            points.append((x, y))
        for first, second in pairwise(points):
            cv2.line(canvas, first, second, (90, 235, 130), 2)

    @staticmethod
    def _state_color(state: DrowsinessState) -> tuple[int, int, int]:
        return {
            DrowsinessState.AWAKE: (90, 220, 120),
            DrowsinessState.BLINKING: (80, 210, 250),
            DrowsinessState.EYES_CLOSED: (40, 150, 255),
            DrowsinessState.ALERT: (40, 40, 255),
            DrowsinessState.TRACKING_LOST: (160, 160, 160),
        }[state]

    @staticmethod
    def _text(canvas, value: str, x: int, y: int, color=(235, 235, 235), scale=0.58) -> None:
        import cv2

        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    @staticmethod
    def _center_text(canvas, value: str, y: int, color, scale: float) -> None:
        import cv2

        size, _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = max(8, (canvas.shape[1] - size[0]) // 2)
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
