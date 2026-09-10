"""Application factory for the Speak2Sign backend."""

from dotenv import load_dotenv

load_dotenv()

import logging

from flask import Flask

from app import config
from app.extensions import cors, socketio

logging.basicConfig(level=logging.INFO)


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    cors.init_app(app)
    socketio.init_app(app, async_mode="threading", cors_allowed_origins="*")

    from app.routes.health import health_bp
    from app.routes.pipeline import pipeline_bp
    app.register_blueprint(health_bp)
    app.register_blueprint(pipeline_bp)

    from app import relay  # noqa: F401  registers the Socket.IO handlers

    from app.models import models
    models.load()

    return app
