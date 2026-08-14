from __future__ import annotations

import json
from pathlib import Path

from ..domain import CalibrationProfile


class JsonCalibrationRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, profile: CalibrationProfile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": 4, "profile": profile.to_dict()}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def load(self) -> CalibrationProfile | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("version") != 4:
                return None
            return CalibrationProfile.from_dict(payload["profile"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)
