from eyetracker.adapters.repository import JsonCalibrationRepository
from eyetracker.domain import CalibrationProfile


def test_repository_round_trip(tmp_path):
    profile = CalibrationProfile(0.3, 0.1, 0.31, 0.11, 0.2, 0.4, 0.09, 0.10, 0.005)
    repository = JsonCalibrationRepository(tmp_path / "calibration.json")
    repository.save(profile)
    assert repository.load() == profile
