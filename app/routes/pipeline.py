"""Translation pipeline endpoints: speech to text, text to gloss, gloss to SiGML,
and signing video to English."""

import logging
import os
import tempfile

import torch
from flask import Blueprint, jsonify, request

from app import config
from app.models import models
from app.services.audio import load_audio_from_bytes
from app.services.llm import gloss_to_sentence
from app.services.recognizer import clip_slug, get_wlasl, retain_debug_clip

logger = logging.getLogger(__name__)

pipeline_bp = Blueprint("pipeline", __name__)


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@pipeline_bp.post("/audio-to-text")
def audio_to_text():
    logger.info("Audio-to-text request received")
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not _allowed_file(file.filename):
            return jsonify({"error": "Invalid file format"}), 400

        # Decode the upload in-memory. A temp file holds a Windows lock between
        # write and read, so decode the bytes directly instead.
        audio_bytes = file.read()
        try:
            audio, sample_rate = load_audio_from_bytes(audio_bytes)
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            return jsonify({"error": f"Failed to load audio file: {e!s}"}), 400

        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type="kaiser_fast")
            sample_rate = 16000

        inputs = models.whisper_processor(
            audio, sampling_rate=sample_rate, return_tensors="pt", padding="max_length"
        )
        inputs = {k: v.to(models.device) for k, v in inputs.items()}

        # task="translate" makes Whisper output English from any spoken language
        # (multilingual input); otherwise transcribe English.
        task = (request.form.get("task") or "transcribe").lower()
        gen_kwargs = dict(
            input_features=inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
            max_length=224,
            min_length=1,
            num_beams=1,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=models.whisper_processor.tokenizer.eos_token_id,
        )
        if task == "translate":
            gen_kwargs["task"] = "translate"
        else:
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"

        with torch.no_grad():
            predicted_ids = models.whisper_model.generate(**gen_kwargs)

        # Content is not logged, for privacy.
        text = models.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"Transcription done ({len(text)} chars, task={task})")
        models.clear_gpu_cache()

        if not text:
            return jsonify({"error": "No speech detected in audio"}), 400
        return jsonify({"text": text, "language": "auto-detected", "success": True})

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return jsonify({"error": f"Audio processing failed: {e!s}"}), 500


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
