import cv2
import numpy as np

from eyetracker.adapters.display import OpenCVDisplay
from eyetracker.domain import FaceObservation, Point2D, TrackingStatus


def test_drawing_eye_contours_and_history_graph_does_not_raise():
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    eye = (
        Point2D(0.20, 0.40),
        Point2D(0.22, 0.38),
        Point2D(0.26, 0.38),
        Point2D(0.28, 0.40),
        Point2D(0.26, 0.42),
        Point2D(0.22, 0.42),
    )
    face = FaceObservation(0.0, TrackingStatus.VALID, eye, eye, 1.0)
    display = OpenCVDisplay()
    display._draw_eyes(canvas, face)
    display._history.extend((1.0, 0.5, 0.1))
    display._draw_graph(canvas, y_top=100, width=180, height=80)
    display._center_multiline(
        canvas,
        "Step 1/7: Keep your eyes open. Press SPACE to start and wait for the sound.",
        20,
        (255, 255, 255),
        0.5,
        max_width=250,
        line_height=20,
    )
    assert np.count_nonzero(canvas) > 0


def test_space_key_continues_calibration(monkeypatch):
    monkeypatch.setattr(cv2, "waitKey", lambda _: ord(" "))
    assert OpenCVDisplay().poll_key() == "continue"
