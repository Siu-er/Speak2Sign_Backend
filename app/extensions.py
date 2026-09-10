"""Flask extensions, instantiated unbound here and bound in the app factory."""

from flask_cors import CORS
from flask_socketio import SocketIO

cors = CORS()
socketio = SocketIO()
