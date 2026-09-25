"""Runtime configuration, read once from the environment."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# Ceilings on the routes that call a paid service, counted in a Modal Dict so
# they hold across API restarts. The values are an operator allowance for this
# deployment, not a property of the services, so both are overridable.
BUDGET_DICT_NAME = "wlasl-budget"
MAX_TRANSLATIONS = int(os.environ.get("MAX_TRANSLATIONS", "100"))
