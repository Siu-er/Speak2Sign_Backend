"""Entrypoint: build the app (loads models) and serve it over Socket.IO."""

import logging

from app import create_app
from app.extensions import socketio
from app.models import models

logger = logging.getLogger(__name__)

app = create_app()

if __name__ == "__main__":
    missing = [name for name, loaded in models.status().items() if not loaded]
    if missing:
        raise RuntimeError(f"Models not loaded: {missing}")
    logger.info(f"Starting Speak2Sign API on http://localhost:5000 (device={models.device})")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
