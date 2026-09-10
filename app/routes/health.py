"""Root and health-check endpoints."""

import time

import torch
from flask import Blueprint, jsonify

from app import config
from app.models import models

health_bp = Blueprint("health", __name__)


def _memory_usage():
    try:
        import psutil
        return f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB"
    except ImportError:
        return "N/A (psutil not installed)"


@health_bp.get("/")
def root():
    return jsonify({
        "service": "Speak2Sign API",
        "version": "1.0.0",
        "description": "Speech to American Sign Language conversion API",
        "endpoints": {
            "health": "/health",
            "audio_to_text": "/audio-to-text",
            "text_to_gloss": "/text-to-gloss",
            "gloss_to_sigml": "/gloss-to-sigml",
            "video_to_sentence": "/video-to-sentence",
        },
        "status": "running",
    })


@health_bp.get("/health")
def health_check():
    status = models.status()
    ready = all(status.values())
    return jsonify({
        "status": "healthy" if ready else "degraded",
        "timestamp": time.time(),
        "service": "Speak2Sign API",
        "version": "1.0.0",
        "system": {
            "device": models.device,
            "cuda_available": torch.cuda.is_available(),
            "memory_usage": _memory_usage(),
        },
        "models": {
            "status": "ready" if ready else "not_ready",
            "whisper_model": config.WHISPER_MODEL_NAME,
            "details": status,
        },
        "endpoints_available": ready,
    }), 200 if ready else 503
