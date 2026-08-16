from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .application import UserQuit
from .bootstrap import build_application, ensure_model
from .config import load_config
from .services.calibration import CalibrationError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eye-sentinel",
        description="Local calibrated eye and head-pose drowsiness monitor",
    )
    parser.add_argument("--config", type=Path, help="Path to a TOML configuration file")
    parser.add_argument("--recalibrate", action="store_true", help="Ignore the saved profile")
    parser.add_argument("--camera", type=int, help="Override the camera index")
    parser.add_argument("--video", type=Path, help="Process a recorded video instead of a camera")
    parser.add_argument(
        "--download-model", action="store_true", help="Download the MediaPipe model and exit"
    )
    return parser.parse_args(argv)


def _runtime_paths(config_argument: Path | None) -> tuple[Path, Path]:
    bundle_root_value = getattr(sys, "_MEIPASS", None)
    if config_argument is not None:
        config_path = config_argument.resolve()
    elif bundle_root_value is not None:
        config_path = (Path(bundle_root_value) / "config/default.toml").resolve()
    else:
        config_path = Path("config/default.toml").resolve()
    if bundle_root_value is not None:
        if sys.platform == "darwin":
            data_root = Path.home() / "Library/Application Support/Eye Sentinel"
        else:
            data_root = Path.home() / ".local/share/eye-sentinel"
    else:
        data_root = config_path.parent.parent
    return config_path, data_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.camera is not None and args.video is not None:
        print("--camera and --video cannot be used together", file=sys.stderr)
        return 2
    config_path, data_root = _runtime_paths(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        return 2
    config = load_config(config_path)
    if args.camera is not None:
        from dataclasses import replace

        config = replace(config, camera=replace(config.camera, device_index=args.camera))
    root = config_path.parent.parent
    if args.download_model:
        model_path = (root / config.tracker.model_path).resolve()
        ensure_model(model_path)
        print(f"MediaPipe model ready: {model_path}")
        return 0
    app = build_application(config, root, args.video, data_root)
    try:
        app.camera.start()
        app.tracker.initialize()
        profile = None if args.recalibrate else app.repository.load()
        if args.video is not None and profile is None:
            print(
                "Recorded-video mode requires a saved calibration. Run once with the webcam first.",
                file=sys.stderr,
            )
            return 2
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
            if args.video is not None:
                print("Recalibration is only available in webcam mode.", file=sys.stderr)
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
