from eyetracker.config import AlarmConfig
from eyetracker.domain import AlarmSeverity, DrowsinessAssessment, DrowsinessState
from eyetracker.services.alert_manager import AlertManager


class FakeAlarm:
    def __init__(self):
        self.active = False
        self.activations = 0
        self.warnings = 0

    def activate(self, reason):
        self.active = True
        self.activations += 1

    def deactivate(self):
        self.active = False

    def warn(self, reason):
        self.warnings += 1

    def notify(self):
        pass


def assessment(timestamp, severity):
    return DrowsinessAssessment(
        timestamp,
        DrowsinessState.AWAKE,
        0.0,
        None,
        0,
        severity is not AlarmSeverity.NONE,
        alarm_severity=severity,
    )


def test_warning_cooldown_and_continuous_urgent_alarm_are_independent():
    alarm = FakeAlarm()
    manager = AlertManager(alarm, AlarmConfig(warning_cooldown_seconds=5.0))
    manager.handle(assessment(0.0, AlarmSeverity.WARNING))
    manager.handle(assessment(2.0, AlarmSeverity.WARNING))
    manager.handle(assessment(5.1, AlarmSeverity.WARNING))
    assert alarm.warnings == 2

    manager.handle(assessment(6.0, AlarmSeverity.URGENT))
    assert alarm.active
    manager.handle(assessment(6.1, AlarmSeverity.NONE))
    assert not alarm.active


def test_muting_stops_all_alarm_output():
    alarm = FakeAlarm()
    manager = AlertManager(alarm, AlarmConfig())
    manager.set_muted(True)
    manager.handle(assessment(0.0, AlarmSeverity.URGENT))
    manager.handle(assessment(1.0, AlarmSeverity.WARNING))
    assert not alarm.active
    assert alarm.activations == 0
    assert alarm.warnings == 0
