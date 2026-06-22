"""
Experiment parameters. Values marked TBD must be fixed before data collection.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))

# One-Euro filter — TBD before data collection
ONE_EURO_MIN_CUTOFF: float = 1.0
ONE_EURO_BETA: float = 0.007
ONE_EURO_D_CUTOFF: float = 1.0

# Calibration
CALIBRATION_POINTS_REQUIRED: int = 16
CALIBRATION_HOLDOUT_FRACTION: float = 0.33

# Calibration quality thresholds (holdout MAE in normalised screen coords)
CALIB_THRESHOLD_X: float = 0.05   # ~96px on 1920px screen
CALIB_THRESHOLD_Y: float = 0.05   # ~54px on 1080px screen

# Screen resolution (pixels)
SCREEN_W_PX: int = 1920
SCREEN_H_PX: int = 1080

# OSC output
OSC_HOST: str = "127.0.0.1"
OSC_PORT: int = 9000
OSC_ADDRESS: str = "/gaze"

# Tobii Core SDK paths (relative to repo root)
TOBII_LIB_PATH: str = os.path.join(
    _REPO_ROOT, "tobiiFundamental", "Tobii.Interaction", "lib", "net45"
)
TOBII_X64_DLL_SRC: str = os.path.join(
    _REPO_ROOT, "tobiiFundamental", "Tobii.Interaction", "build", "x64",
    "Tobii.EyeX.Client.dll"
)

# CSV
CSV_FLUSH_EVERY: int = 10

# Face-position guide overlay
# Assumes standard 640 px wide webcam at ~60 cm viewing distance (focal ≈ frame_w).
# IOD_TARGET_NORM × frame_width gives the expected iris-to-iris pixel span.
# Increase to require a closer seating position; decrease to allow farther.
IOD_TARGET_NORM: float = 0.10     # target IOD as fraction of frame width  (≈60 cm)
IOD_TOLERANCE_NORM: float = 0.025 # ±25 % zone shown in green

# MediaPipe face mesh backend: "tasks" (FaceLandmarker, protobuf-compatible) or "solutions" (legacy, requires protobuf<4)
FACEMESH_BACKEND: str = "tasks"
FACEMESH_MODEL_PATH: str = os.path.join(_REPO_ROOT, "models", "face_landmarker.task")
