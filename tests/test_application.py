from eyetracker.application import CalibrationController
from eyetracker.services.calibration import CalibrationError


def test_failed_step_waits_to_retry_only_that_step(monkeypatch):
    controller = CalibrationController(None, None, None, None, None, None, None, None)
    waits: list[str] = []
    monkeypatch.setattr(controller, "_wait_for_space", waits.append)

    controller._wait_to_retry_step(7, CalibrationError("invalid closed-eye reference"))

    assert waits == [
        "Step 7/7 failed: invalid closed-eye reference. Press SPACE to retry only this step."
    ]
