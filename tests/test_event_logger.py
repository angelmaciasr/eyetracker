import json

from eyetracker.adapters.event_logger import JsonlEventLogger
from eyetracker.domain import DrowsinessAssessment, DrowsinessState


def test_event_logger_records_transitions_and_session_summary(tmp_path):
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(path)
    awake = DrowsinessAssessment(0.0, DrowsinessState.AWAKE, 0.0, None, 0, False)
    closed = DrowsinessAssessment(1.0, DrowsinessState.EYES_CLOSED, 0.5, None, 0, False)

    logger.begin_session()
    logger.record(awake)
    logger.record(awake)
    logger.record(closed)
    logger.end_session()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "session_started",
        "state_changed",
        "state_changed",
        "session_ended",
    ]
    assert records[-1]["final_perclos_60"] == 0.0
