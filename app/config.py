"""Runtime configuration, read once from the environment."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Whisper base is the fast default; override with WHISPER_MODEL for a larger,
# more accurate model. The download helper reads the same variable so the
# fetched and loaded models stay in sync.
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "openai/whisper-base")

DATA_DIR = os.path.join(BASE_DIR, "data")
SIGNS_DIR = os.path.join(DATA_DIR, "signs")

MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "webm"}

# The most recent recorded clips are kept on disk so a translation result can be
# replayed against the original video while debugging.
DEBUG_CLIPS_DIR = os.path.join(BASE_DIR, "debug_clips")
DEBUG_CLIPS_KEEP = 3

WLASL_MODAL_APP = "wlasl-i3d"
WLASL_MODAL_CLASS = "WLASL"
