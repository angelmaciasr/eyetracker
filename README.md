# Eye Sentinel

Python MVP that uses a webcam to measure the relative openness of both eyes, distinguish natural
blinks from prolonged closures, and trigger a local alarm. Video is never stored or sent to a
server.

> **Warning:** this is an experimental project, not a medical device or certified safety system.
> Do not use it as your only protection while driving or operating machinery.

## MVP features

- One person and one local webcam.
- MediaPipe Face Landmarker to locate six points around each eye.
- Eye Aspect Ratio (EAR) filtered with a short median window.
- Guided seven-step calibration: front, down, up, left, right, blinks, and closed eyes.
- Independent relative openness and head-pose correction for each eye.
- `pitch`, `yaw`, and `roll` estimation relative to the calibrated neutral pose.
- Time-based state machine with hysteresis: awake, blinking, closed, alert, and tracking lost.
- Repeating alarm when relative openness remains below `0.75` for one second.
- Debug view with landmarks, EAR, openness, closure duration, FPS, blink count, and chart.
- Persistent calibration profile in `data/calibration.json`.
- Automated tests for geometry, calibration, head pose, display, and temporal logic.

PERCLOS, gaze direction, multiple faces, and identity recognition are intentionally outside the
MVP scope.

## Requirements

- Python 3.11 or 3.12. MediaPipe does not currently provide a Python 3.14 wheel.
- macOS, Linux, or Windows with an accessible webcam.
- Internet access on first launch to download the official MediaPipe model.

Recommended setup with `uv`:

```bash
uv sync --extra dev
uv run eye-sentinel
```

The first launch downloads `models/face_landmarker.task`. macOS may ask for camera permission for
Terminal or T3 Code. If access was previously blocked, enable it in **System Settings → Privacy &
Security → Camera**, then restart the application you use to run the command.

## Usage

```bash
# Normal launch; reuse the saved calibration
uv run eye-sentinel

# Force a new calibration
uv run eye-sentinel --recalibrate

# Select a different camera
uv run eye-sentinel --camera 1

# Use a different configuration file
uv run eye-sentinel --config config/default.toml
```

Window controls:

- `Q` or `Esc`: quit.
- `R`: recalibrate.
- `M`: mute or unmute the alarm for the current session.
- `Space`: start the displayed calibration step.

## Calibration

Calibration contains seven guided steps:

1. Keep your eyes naturally open and look straight ahead.
2. Keep your eyes open and tilt your head down.
3. Keep your eyes open and tilt your head up.
4. Keep your eyes open and turn your head left.
5. Keep your eyes open and turn your head right.
6. Look straight ahead and blink naturally five times.
7. Close both eyes and keep them closed.

Before every step, the application explains what to do and waits for `Space`. A confirmation sound
marks the end of each step. Keep a stable distance and use even lighting. If the poses or open/closed
references are not distinct enough, the application explains the failure and lets you press `Space`
to restart without closing the program.

## Configuration

All thresholds are stored in [`config/default.toml`](config/default.toml). The main detector values
are:

```toml
[detector]
closed_threshold = 0.75
reopened_threshold = 0.85
minimum_blink_seconds = 0.08
maximum_blink_seconds = 0.40
alert_after_closed_seconds = 1.00
```

The closed and reopened thresholds apply to calibrated relative openness, not raw EAR. Their
difference provides hysteresis and prevents state oscillation.

Calibration learns vertical and horizontal extremes. Expected openness is interpolated separately
for each eye because turning the head can deform one eye more than the other:

```toml
[head_pose]
pitch_range_margin = 5.0
yaw_range_margin = 5.0
maximum_roll_delta = 20.0
minimum_calibration_pitch_span = 12.0
minimum_calibration_yaw_span = 15.0
```

The application accepts the measured ranges plus a five-degree margin on each axis. Beyond those
ranges it displays `ANGLE NOT COVERED BY CALIBRATION`, resets any active closure, and prevents an
alarm. Recalibration is required after upgrading from an older profile format.

## Development and tests

```bash
uv run pytest
uv run ruff check .
```

Core logic in `src/eyetracker/services/` does not import OpenCV or MediaPipe. Camera, tracking,
sound, display, and persistence implementations live in `src/eyetracker/adapters/`.

## Structure

```text
src/eyetracker/
├── application.py          # Calibration and monitoring orchestration
├── bootstrap.py            # Connects concrete implementations
├── config.py               # Typed configuration
├── domain.py               # Shared models and states
├── ports.py                # Infrastructure contracts
├── adapters/               # OpenCV, MediaPipe, sound, and JSON
└── services/               # EAR, calibration, head pose, and temporal detector
```
