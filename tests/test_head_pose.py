import math

import numpy as np
import pytest

from eyetracker.services.head_pose import (
    angular_distance,
    head_pose_from_transform,
    signed_angular_delta,
)


def test_identity_transform_is_neutral_pose():
    pose = head_pose_from_transform(np.eye(4))
    assert pose.pitch == pytest.approx(0.0)
    assert pose.yaw == pytest.approx(0.0)
    assert pose.roll == pytest.approx(0.0)


def test_transform_extracts_yaw_in_degrees():
    angle = math.radians(20.0)
    transform = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert head_pose_from_transform(transform).yaw == pytest.approx(20.0)


def test_angular_distance_handles_wraparound():
    assert angular_distance(179.0, -179.0) == pytest.approx(2.0)
    assert signed_angular_delta(-179.0, 179.0) == pytest.approx(2.0)
