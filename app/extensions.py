"""Flask extensions, instantiated unbound here and bound in the app factory."""

from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO

from app import config

cors = CORS()
socketio = SocketIO()
# Counters are kept in process memory: the service runs as one gunicorn process,
# so every request is counted in the same store.
limiter = Limiter(get_remote_address, default_limits=[config.RATE_LIMIT_DEFAULT], storage_uri="memory://")
