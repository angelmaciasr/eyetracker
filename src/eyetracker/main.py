from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .application import UserQuit
from .bootstrap import build_application
from .config import load_config
from .services.calibration import CalibrationError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eye-sentinel",
        description="Local prolonged eye-closure detector",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--recalibrate", action="store_true", help="Ignore the saved profile")
    parser.add_argument("--camera", type=int, help="Override the camera index")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        return 2
    config = load_config(config_path)
    if args.camera is not None:
        from dataclasses import replace

        config = replace(config, camera=replace(config.camera, device_index=args.camera))
    root = config_path.parent.parent
    app = build_application(config, root)
    try:
        app.camera.start()
        app.tracker.initialize()
        profile = None if args.recalibrate else app.repository.load()
        if profile is not None:
            app.calibrator.load_profile(profile)
            print("Saved calibration loaded. Press R to run it again.")
        else:
            print(
                "Starting calibration: front, down, up, left, right, "
                "5 blinks, and sustained eye closure."
            )
            profile = app.calibration_controller.run()
        app.monitoring_controller.detector.apply_calibration(profile)
        while True:
            result = app.monitoring_controller.run()
            if result == "quit":
                break
            try:
                profile = app.calibration_controller.run()
                app.monitoring_controller.detector.apply_calibration(profile)
            except CalibrationError as error:
                print(f"Calibration failed: {error}. Try again.")
    except UserQuit:
        pass
    except CalibrationError as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
