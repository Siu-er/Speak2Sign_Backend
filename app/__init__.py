"""Application factory for the Speak2Sign backend."""

from dotenv import load_dotenv

load_dotenv()

import logging

from flask import Flask, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

from app import config
from app.extensions import cors, limiter, socketio

logging.basicConfig(level=logging.INFO)


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    if config.TRUSTED_PROXY_HOPS:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=config.TRUSTED_PROXY_HOPS)

    cors.init_app(app)
    limiter.init_app(app)

    @app.errorhandler(429)
    def too_many_requests(e):
        logging.getLogger(__name__).warning(f"rate limit {e.description} reached by {request.remote_addr} on {request.path}")
        return jsonify({"error": f"Too many requests ({e.description}), try again shortly"}), 429

    # manage_session=False bypasses Flask-SocketIO's session propagation, which
    # sets ctx.session - a read-only property in Flask 3.x, causing AttributeError.
    socketio.init_app(app, async_mode="threading", cors_allowed_origins="*",
                      manage_session=False)

    from app.routes.health import health_bp
    from app.routes.pipeline import pipeline_bp
    app.register_blueprint(health_bp)
    app.register_blueprint(pipeline_bp)

    from app import relay  # noqa: F401  registers the Socket.IO handlers

    from app.models import models
    models.load()

    return app
