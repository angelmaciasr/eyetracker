from __future__ import annotations

import shutil
import subprocess
import sys
import threading

from ..config import AlarmConfig


class SoundAlarm:
    def __init__(self, config: AlarmConfig) -> None:
        self.config = config
        self._active = threading.Event()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="eye-alarm", daemon=True)
        self._thread.start()

    def activate(self, reason: str) -> None:
        del reason
        self._active.set()

    def deactivate(self) -> None:
        self._active.clear()

    def is_active(self) -> bool:
        return self._active.is_set()

    def close(self) -> None:
        self._active.clear()
        self._closed.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._closed.is_set():
            if not self._active.wait(timeout=0.1):
                continue
            self._play_once()
            self._closed.wait(self.config.repeat_seconds)

    @staticmethod
    def _play_once() -> None:
        command: list[str] | None = None
        if sys.platform == "darwin" and shutil.which("afplay"):
            command = ["afplay", "/System/Library/Sounds/Glass.aiff"]
        elif shutil.which("paplay"):
            command = ["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"]
        if command:
            try:
                subprocess.run(command, check=False, timeout=2, capture_output=True)
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        print("\a", end="", flush=True)
