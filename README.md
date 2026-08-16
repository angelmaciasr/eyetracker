# Eye Sentinel

Local Python drowsiness monitor that combines calibrated eye openness, temporal closure metrics,
PERCLOS, and head orientation. Video is never stored or sent to a server.

> **Warning:** this is an experimental project, not a medical device or certified safety system.
> Do not use it as your only protection while driving or operating machinery.

## Features

- One person and one local webcam.
- MediaPipe Face Landmarker to locate six points around each eye.
- Eye Aspect Ratio (EAR) filtered with a short median window.
- Guided seven-step calibration: front, down, up, left, right, blinks, and closed eyes.
- Independent relative openness and head-pose correction for each eye.
- `pitch`, `yaw`, and `roll` estimation relative to the calibrated neutral pose.
- Time-based state machine with hysteresis: awake, blinking, closed, alert, and tracking lost.
- Repeating alarm when relative openness remains below `0.75` for one second.
- Independent alarm when vertical head tilt exceeds `25°` for `0.75` seconds.
- Immediate alarm outside the head-pose range covered by calibration.
- Lateral head-drop alarm with independent trigger and recovery thresholds.
- Rolling PERCLOS over 30 and 60 seconds, excluding normal blinks and unreliable tracking.
- Slow-closure count, average duration, longest recent closure, and tracking confidence.
- Drowsiness levels: normal, possible drowsiness, drowsy, immediate alert, and tracking lost.
- Brief cooldown-controlled warnings and an independent repeating urgent alarm.
- Debug view with landmarks, EAR, openness, head pose, PERCLOS, confidence, FPS, and chart.
- Persistent calibration profile in `data/calibration.json`.
- Transition and session metrics in `data/events.jsonl`; no images or video are logged.
- Recorded-video input for reproducible tuning and tests.
- Automated tests for geometry, calibration, PERCLOS, alerts, video input, logging, and head pose.

Gaze direction, yawning, multiple faces, identity recognition, mobile deployment, and custom model
training remain intentionally outside scope; they were not part of the post-MVP continuation in
the original plan.

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

# Run the same detector against a recording (uses the saved calibration)
uv run eye-sentinel --video recordings/test-drive.mp4

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
head_tilt_threshold = 25.0
head_tilt_recovered_threshold = 18.0
head_side_tilt_threshold = 15.0
head_side_tilt_recovered_threshold = 10.0
head_tilt_alert_seconds = 0.75
perclos_closed_threshold = 0.20
perclos_short_window_seconds = 30.0
perclos_window_seconds = 60.0
perclos_drowsy_threshold = 0.20
perclos_minimum_observation_seconds = 10.0
slow_blink_seconds = 0.40
possible_drowsiness_slow_blinks = 3
recent_closure_window_seconds = 60.0
maximum_sample_gap_seconds = 0.50
tracking_lost_seconds = 2.0
```

The closed and reopened thresholds apply to calibrated relative openness, not raw EAR. Their
difference provides hysteresis and prevents state oscillation.

The head-tilt rule uses pitch and lateral roll relative to the neutral pose learned during
calibration. The lower recovery thresholds prevent rapid state changes near the trigger. Entering
an angle not covered by calibration triggers an immediate alarm because eye openness can no longer
be compared safely with the learned references.

PERCLOS is the proportion of reliable observed time spent below `perclos_closed_threshold`. Normal
blinks are removed after they are classified, tracking gaps are excluded from the denominator, and
intervals older than the configured rolling windows are discarded. A high PERCLOS produces a
brief warning with cooldown; a continuous eye closure or unsafe head orientation still uses the
independent repeating urgent alarm.

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

The application accepts the measured ranges plus a five-degree margin on each axis. Confidence
decreases as the head approaches those limits. Beyond them it displays
`ALERT: ANGLE NOT COVERED BY CALIBRATION`, resets any active eye closure, and sounds the alarm.

## Recorded-video evaluation

Calibrate once with the webcam, then reuse that personal profile with a recording:

```bash
mkdir -p recordings
uv run eye-sentinel --video recordings/test-drive.mp4
```

Video timestamps, rather than processing speed, drive all temporal rules, so a file produces the
same result when processed on different machines. The run ends automatically at EOF. State
transitions and the final PERCLOS/closure summary are appended to `data/events.jsonl`; recordings
remain untouched and are not copied by the application.

## Build a local macOS application

Download the model once and build the PyInstaller specification:

```bash
uv run eye-sentinel --download-model
uv run --with pyinstaller pyinstaller --clean packaging/eye-sentinel.spec
open "dist/Eye Sentinel.app"
```

The bundle includes MediaPipe, the default configuration, and the model when it is present under
`models/`. In the packaged app, calibration and event logs are written to
`~/Library/Application Support/Eye Sentinel/`. macOS will request camera permission for the built
application separately from Terminal.

## Development and tests

```bash
uv run pytest
uv run ruff check .
git diff --check
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
├── adapters/               # OpenCV, MediaPipe, sound, JSONL, and persistence
└── services/               # EAR, calibration, head pose, alerts, PERCLOS, and temporal rules
```
