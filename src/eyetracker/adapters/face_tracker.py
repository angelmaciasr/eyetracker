from __future__ import annotations

from pathlib import Path

from ..config import TrackerConfig
from ..domain import FaceObservation, Point2D, TrackingStatus, VideoFrame
from ..services.head_pose import head_pose_from_transform

# The six points preserve the order required by EAR. "Left" and "right" refer
# to the observed person's eyes, not to the side of the image.
LEFT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_INDICES = (33, 160, 158, 133, 153, 144)


class MediaPipeFaceTracker:
    def __init__(self, config: TrackerConfig, model_path: Path) -> None:
        self.config = config
        self.model_path = model_path
        self._landmarker = None
        self._last_timestamp_ms = -1

    def initialize(self) -> None:
        import mediapipe as mp

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.config.min_face_detection_confidence,
            min_face_presence_confidence=self.config.min_face_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def track(self, frame: VideoFrame) -> FaceObservation:
        if self._landmarker is None:
            raise RuntimeError("The face tracker has not been initialized")
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(self._last_timestamp_ms + 1, int(frame.timestamp * 1000))
        self._last_timestamp_ms = timestamp_ms
        try:
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except RuntimeError:
            return FaceObservation(frame.timestamp, TrackingStatus.ERROR)
        if not result.face_landmarks:
            return FaceObservation(frame.timestamp, TrackingStatus.NO_FACE)
        landmarks = result.face_landmarks[0]
        head_pose = None
        if result.facial_transformation_matrixes:
            try:
                head_pose = head_pose_from_transform(result.facial_transformation_matrixes[0])
            except (ValueError, ArithmeticError):
                head_pose = None

        def points(indices: tuple[int, ...]) -> tuple[Point2D, ...]:
            return tuple(Point2D(landmarks[index].x, landmarks[index].y) for index in indices)

        return FaceObservation(
            timestamp=frame.timestamp,
            status=TrackingStatus.VALID,
            left_eye=points(LEFT_EYE_INDICES),
            right_eye=points(RIGHT_EYE_INDICES),
            confidence=1.0,
            head_pose=head_pose,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
