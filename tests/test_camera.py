import cv2
import numpy as np

from eyetracker.adapters.camera import VideoFileCamera


def test_video_file_camera_uses_video_time_and_finishes(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.write(np.ones((24, 32, 3), dtype=np.uint8))
    writer.release()

    camera = VideoFileCamera(str(path))
    camera.start()
    first = camera.read()
    second = camera.read()
    finished = camera.read()
    camera.stop()

    assert first is not None and first.timestamp == 0.1
    assert second is not None and second.timestamp == 0.2
    assert finished is None
    assert camera.is_finished()
