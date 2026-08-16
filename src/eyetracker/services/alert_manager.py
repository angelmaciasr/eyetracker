from __future__ import annotations

from ..config import AlarmConfig
from ..domain import AlarmSeverity, DrowsinessAssessment
from ..ports import AlarmPort


class AlertManager:
    """Translate detector severity into warning or continuous alarm behavior."""

    def __init__(self, alarm: AlarmPort, config: AlarmConfig) -> None:
        self.alarm = alarm
        self.config = config
        self._muted = False
        self._last_warning_at: float | None = None

    def reset(self) -> None:
        self._last_warning_at = None
        self.alarm.deactivate()

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            self.alarm.deactivate()

    def handle(self, assessment: DrowsinessAssessment) -> None:
        if self._muted:
            self.alarm.deactivate()
            return
        if assessment.alarm_severity is AlarmSeverity.URGENT:
            self.alarm.activate(assessment.reason or "urgent_alert")
            return
        self.alarm.deactivate()
        if assessment.alarm_severity is not AlarmSeverity.WARNING:
            return
        if (
            self._last_warning_at is None
            or assessment.timestamp - self._last_warning_at >= self.config.warning_cooldown_seconds
        ):
            self.alarm.warn(assessment.reason or "drowsiness_warning")
            self._last_warning_at = assessment.timestamp
