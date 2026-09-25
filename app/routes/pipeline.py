"""Translation pipeline endpoints: text to gloss, gloss to SiGML, and signing
video to English. Speech-to-text is handled client-side via the browser Web
Speech API."""

import logging
import os
import tempfile
import time

from flask import Blueprint, jsonify, request

from app import config
from app.models import models
from app.services.llm import gloss_to_sentence
from app.services.recognizer import clip_slug, get_wlasl, retain_debug_clip

logger = logging.getLogger(__name__)

pipeline_bp = Blueprint("pipeline", __name__)

# /video-to-sentence is the only billable route: it runs GPU inference on Modal
# and then an LLM call. The durable spend ceiling lives in Modal so it survives
# restarts; this throttle only stops one caller burning that budget in a burst.
_RECENT_CALLS: dict = {}
RATE_LIMIT_CALLS = 5
RATE_LIMIT_WINDOW_S = 60


def _rate_limited(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _RECENT_CALLS.get(ip, []) if now - t < RATE_LIMIT_WINDOW_S]
    if len(hits) >= RATE_LIMIT_CALLS:
        _RECENT_CALLS[ip] = hits
        return True
    hits.append(now)
    _RECENT_CALLS[ip] = hits
    if len(_RECENT_CALLS) > 1000:
        for k in [k for k, v in _RECENT_CALLS.items() if not v or now - v[-1] > RATE_LIMIT_WINDOW_S]:
            _RECENT_CALLS.pop(k, None)
    return False


@pipeline_bp.post("/text-to-gloss")
def text_to_gloss():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400

        result = models.glosser.gloss(text)
        return jsonify({
            "original_text": text,
            "gloss": result.gloss,
            "gloss_tokens": result.gloss_tokens,
            "non_manual_markers": result.sentence_nmm,
            "success": True,
        })
    except Exception as e:
        logger.error(f"Text to gloss conversion failed: {e}")
        return jsonify({"error": f"Text to gloss conversion failed: {e!s}"}), 500


@pipeline_bp.post("/gloss-to-sigml")
def gloss_to_sigml():
    try:
        data = request.get_json()
        if not data or "gloss" not in data:
            return jsonify({"error": "No gloss provided"}), 400
        gloss = data["gloss"].strip()
        if not gloss:
            return jsonify({"error": "Empty gloss provided"}), 400

        sigml_xml = models.sigml.generate_sigml(gloss)
        tokens = models.sigml.gloss_to_tokens(gloss)
        token_kinds = [models.sigml.classify_token(t) for t in tokens]
        fingerspelled = [t for t, k in zip(tokens, token_kinds) if k == "fingerspell"]
        return jsonify({
            "gloss": gloss,
            "tokens": tokens,
            "token_kinds": token_kinds,
            "fingerspelled": fingerspelled,
            "sigml": sigml_xml,
            "success": True,
        })
    except Exception as e:
        logger.error(f"Gloss to SiGML conversion failed: {e}")
        return jsonify({"error": f"Gloss to SiGML conversion failed: {e!s}"}), 500


@pipeline_bp.post("/video-to-sentence")
def video_to_sentence():
    caller = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
    if _rate_limited(caller):
        return jsonify({"error": "too many requests, try again shortly"}), 429
    if "video" not in request.files:
        return jsonify({"error": "no video provided"}), 400
    file = request.files["video"]
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    data = file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        result = get_wlasl().recognize.remote(data)
        gloss = result.get("gloss", [])
        logger.info(f"video-to-sentence gloss: {gloss} ({len(data)} bytes)")
        if not gloss:
            retain_debug_clip(tmp.name, suffix, ["none"], meta={"gloss": [], "sentence": "", **result})
            return jsonify({"sentence": "", "gloss": [], "success": True})
        sentence = gloss_to_sentence(gloss)
        logger.info(f"video-to-sentence: {sentence!r}")
        retain_debug_clip(tmp.name, suffix, [clip_slug(sentence)],
                          meta={"gloss": gloss, "sentence": sentence, **result})
        return jsonify({"sentence": sentence, "gloss": gloss, "success": True})
    except Exception as e:
        logger.error(f"video-to-sentence failed: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
