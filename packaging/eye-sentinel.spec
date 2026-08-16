from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPEC).resolve().parent.parent
mediapipe_data, mediapipe_binaries, mediapipe_hidden_imports = collect_all("mediapipe")
data_files = mediapipe_data + [
    (str(project_root / "config/default.toml"), "config"),
]
model = project_root / "models/face_landmarker.task"
if model.exists():
    data_files.append((str(model), "models"))

analysis = Analysis(
    [str(project_root / "packaging/entrypoint.py")],
    pathex=[str(project_root / "src")],
    binaries=mediapipe_binaries,
    datas=data_files,
    hiddenimports=mediapipe_hidden_imports,
)
python_archive = PYZ(analysis.pure)
executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Eye Sentinel",
    console=False,
)
collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    name="Eye Sentinel",
)
app = BUNDLE(
    collection,
    name="Eye Sentinel.app",
    icon=None,
    bundle_identifier="local.eye-sentinel.app",
)
