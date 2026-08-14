from eyetracker.application import CalibrationController
from eyetracker.domain import CalibrationProfile
from eyetracker.services.calibration import CalibrationError


def test_failed_calibration_waits_and_restarts_instead_of_exiting(monkeypatch):
    controller = CalibrationController(None, None, None, None, None, None, None, None)
    profile = CalibrationProfile(0.3, 0.1, 0.31, 0.11, 0.2, 0.4, 0.09, 0.10, 0.005)
    attempts: list[bool] = []
    waits: list[str] = []

    def run_once(wait_for_first_step: bool):
        attempts.append(wait_for_first_step)
        if len(attempts) == 1:
            raise CalibrationError("pose insuficiente")
        return profile

    monkeypatch.setattr(controller, "_run_once", run_once)
    monkeypatch.setattr(controller, "_wait_for_space", waits.append)

    assert controller.run() is profile
    assert attempts == [True, False]
    assert waits == ["CALIBRACIÓN FALLIDA - ESPACIO PARA REINICIAR"]
