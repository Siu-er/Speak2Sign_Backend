"""Flask extensions, instantiated unbound here and bound in the app factory."""

import hmac
import uuid

from flask import request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO

from app import config

cors = CORS()
socketio = SocketIO()


def client_key() -> str:
    """The client a request is counted against. Visitors at a venue share one
    network address, so the app's per-browser device ID is counted when the
    request carries one; other callers are counted by address."""
    device = request.headers.get("X-S2S-Device-Id", "")
    try:
        return f"device:{uuid.UUID(device)}"
    except ValueError:
        return f"address:{get_remote_address()}"


# Counters are kept in process memory: the service runs as one gunicorn process,
# so every request is counted in the same store.
limiter = Limiter(client_key, default_limits=[config.RATE_LIMIT_DEFAULT], storage_uri="memory://")


@limiter.request_filter
def team_device() -> bool:
    key = request.headers.get("X-S2S-Device-Key", "")
    return bool(key) and any(hmac.compare_digest(key, k) for k in config.RATE_LIMIT_EXEMPT_KEYS)
