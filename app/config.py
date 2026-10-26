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

# Request limits per client (see extensions.client_key), in the notation of the `limits` package. A
# paid call (GPU recognition or an LLM translation) follows a recording of
# several seconds, so a person signing or speaking without pause stays under the
# paid limit; the default covers the two text calls each spoken utterance makes.
RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "60 per minute")
RATE_LIMIT_PAID = os.environ.get("RATE_LIMIT_PAID", "5 per minute")
# The team's own devices send one of these comma-separated keys in the
# X-S2S-Device-Key header and are not limited. Set in the service environment.
RATE_LIMIT_EXEMPT_KEYS = frozenset(
    k.strip() for k in os.environ.get("RATE_LIMIT_EXEMPT_KEYS", "").split(",") if k.strip())
# Proxies in front of the service that append the caller to X-Forwarded-For. The
# client address is read this many entries from the right, since entries further
# left are whatever the client sent. 0 when the service is reached directly.
TRUSTED_PROXY_HOPS = int(os.environ.get("TRUSTED_PROXY_HOPS", "0"))

WLASL_MODAL_APP = "wlasl-i3d"
WLASL_MODAL_CLASS = "WLASL"
