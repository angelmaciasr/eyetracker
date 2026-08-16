from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..domain import DrowsinessAssessment


class JsonlEventLogger:
    """Persist state transitions and a final session summary without storing video."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._started_at: datetime | None = None
        self._last_key: tuple[str, str, str | None] | None = None
        self._last_assessment: DrowsinessAssessment | None = None
        self._alert_count = 0

    def begin_session(self) -> None:
        self._started_at = datetime.now(UTC)
        self._last_key = None
        self._last_assessment = None
        self._alert_count = 0
        self._write({"event": "session_started"})

    def record(self, assessment: DrowsinessAssessment) -> None:
        self._last_assessment = assessment
        key = (assessment.state.value, assessment.level.value, assessment.reason)
        if key == self._last_key:
            return
        self._last_key = key
        if assessment.should_alert:
            self._alert_count += 1
        payload = asdict(assessment)
        payload["event"] = "state_changed"
        self._write(payload)

    def end_session(self) -> None:
        if self._started_at is None:
            return
        payload: dict[str, object] = {
            "event": "session_ended",
            "duration_seconds": (datetime.now(UTC) - self._started_at).total_seconds(),
            "alert_transitions": self._alert_count,
        }
        if self._last_assessment is not None:
            payload.update(
                {
                    "final_perclos_30": self._last_assessment.perclos_30_seconds,
                    "final_perclos_60": self._last_assessment.perclos_60_seconds,
                    "slow_blinks_last_minute": self._last_assessment.slow_blinks_last_minute,
                    "longest_closure_seconds": self._last_assessment.longest_closure_seconds,
                }
            )
        self._write(payload)
        self._started_at = None

    def _write(self, payload: dict[str, object]) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {"recorded_at": datetime.now(UTC).isoformat(), **payload}
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            self.enabled = False
