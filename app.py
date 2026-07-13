from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room, emit
from dotenv import load_dotenv
import secrets
import tempfile
import os
import shutil

load_dotenv()
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging
import time
from asl_glosser import ASLGlosser, GlossResult
from sigml_generator import SiGMLGenerator
from audio_utils import load_audio_from_bytes
from asl_sign_recognizer import ASLSignRecognizer

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm'}

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_processor = None
whisper_model = None
asl_glosser = None
sigml_generator = None
asl_sign_recognizer = None
models_initialized = False

# Whisper model. Base is the fast default; override with WHISPER_MODEL for a
# larger, more accurate model. The download helper reads the same variable so
# the fetched and loaded models stay in sync.
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "openai/whisper-base")

def init_models():
    """Initialize all models at server startup"""
    global whisper_processor, whisper_model, asl_glosser, sigml_generator, asl_sign_recognizer, models_initialized

    # Check if already initialized
    if models_initialized:
        logger.info("Models already initialized, skipping...")
        return

    logger.info("=" * 60)
    logger.info("INITIALIZING SPEAK2SIGN MODELS")
    logger.info("=" * 60)

    try:
        # Load Whisper model
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
        logger.info(f"Using device: {device}")

        start_time = time.time()
        whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME).to(device)

        # Optimize model for faster inference
        whisper_model.eval()  # Set to evaluation mode
        if hasattr(torch, 'compile'):
            try:
                logger.info("Compiling model for faster inference...")
                whisper_model = torch.compile(whisper_model, mode="reduce-overhead")
                logger.info("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed, continuing without: {e}")

        whisper_time = time.time() - start_time

        logger.info(f"Whisper model loaded successfully in {whisper_time:.2f}s")

        # Load ASL Glosser
        logger.info("Loading ASL Glosser...")
        start_time = time.time()
        asl_glosser = ASLGlosser("data")
        glosser_time = time.time() - start_time
        logger.info(f"ASL Glosser loaded successfully in {glosser_time:.2f}s")

        # Load SiGML Generator
        logger.info("Loading SiGML Generator...")
        start_time = time.time()
        sigml_generator = SiGMLGenerator()
        sigml_time = time.time() - start_time
        logger.info(f"SiGML Generator loaded successfully in {sigml_time:.2f}s")

        # Load ASL Sign Recognizer
        logger.info("Loading ASL Sign Recognizer...")
        start_time = time.time()
        asl_sign_recognizer = ASLSignRecognizer(os.path.join("data", "models"))
        recognizer_time = time.time() - start_time
        logger.info(f"ASL Sign Recognizer loaded successfully in {recognizer_time:.2f}s")

        total_time = whisper_time + glosser_time + sigml_time + recognizer_time
        logger.info("=" * 60)
        logger.info("ALL MODELS LOADED SUCCESSFULLY!")
        logger.info(f"Total loading time: {total_time:.2f}s")
        logger.info(f"Device: {device}")
        logger.info(f"Memory usage: {get_memory_usage()}")
        logger.info("=" * 60)

        # Mark as initialized
        models_initialized = True

    except Exception as e:
        logger.error("=" * 60)
        logger.error("FAILED TO INITIALIZE MODELS")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        raise

def get_memory_usage():
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f} MB"
    except ImportError:
        return "N/A (psutil not installed)"

def log_gpu_memory_if_cuda(stage=""):
    """Log GPU memory usage if using CUDA"""
    if device == "cuda":
        memory_mb = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"GPU memory {stage}: {memory_mb:.1f}MB")

def clear_gpu_cache():
    """Clear GPU cache to prevent memory leaks"""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Speak2Sign API',
        'version': '1.0.0',
        'description': 'Speech to American Sign Language conversion API',
        'endpoints': {
            'health': '/health',
            'audio_to_text': '/audio-to-text',
            'text_to_gloss': '/text-to-gloss',
            'gloss_to_sigml': '/gloss-to-sigml',
            'video_to_signs': '/video-to-signs',
            'sign_to_sentence': '/sign-to-sentence',
        },
        'status': 'running'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed system information"""
    try:
        # Check if all models are loaded
        models_status = {
            'whisper_processor': whisper_processor is not None,
            'whisper_model': whisper_model is not None,
            'asl_glosser': asl_glosser is not None,
            'sigml_generator': sigml_generator is not None,
            'asl_sign_recognizer': asl_sign_recognizer is not None
        }

        all_models_loaded = all(models_status.values())

        return jsonify({
            'status': 'healthy' if all_models_loaded else 'degraded',
            'timestamp': time.time(),
            'service': 'Speak2Sign API',
            'version': '1.0.0',
            'system': {
                'device': device,
                'cuda_available': torch.cuda.is_available(),
                'memory_usage': get_memory_usage()
            },
            'models': {
                'status': 'ready' if all_models_loaded else 'not_ready',
                'whisper_model': WHISPER_MODEL_NAME,
                'details': models_status
            },
            'endpoints_available': all_models_loaded
        }), 200 if all_models_loaded else 503

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'timestamp': time.time(),
            'error': str(e)
        }), 500

@app.route('/audio-to-text', methods=['POST'])
def audio_to_text():
    """Convert audio file to text using Whisper"""
    logger.info("Audio-to-text request received")
    try:

        # Check if audio file is provided
        if 'audio' not in request.files:
            logger.warning("No audio file provided in request")
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if file.filename == '':
            logger.warning("Empty filename provided")
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            logger.warning(f"Invalid file format: {file.filename}")
            return jsonify({'error': 'Invalid file format'}), 400

        logger.info(f"Processing audio file: {file.filename}")

        # Decode the upload in-memory. A temp file holds a Windows lock between
        # write and read, so decode the bytes directly instead.
        audio_bytes = file.read()
        try:
            audio, sample_rate = load_audio_from_bytes(audio_bytes)
            logger.info(f"Audio loaded: sample_rate={sample_rate}, duration={len(audio)/sample_rate:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            return jsonify({'error': f'Failed to load audio file: {str(e)}'}), 400

        # Resample audio to 16kHz if needed (Whisper requirement) - optimized
        if sample_rate != 16000:
            logger.info(f"Resampling audio from {sample_rate}Hz to 16000Hz")
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type='kaiser_fast')
            sample_rate = 16000

        # Process audio with Whisper
        logger.info(f"Processing audio with Whisper on device: {device}")
        inputs = whisper_processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logger.info(f"Moved inputs to device: {device}, input_features shape: {inputs['input_features'].shape}")

        # task="translate" makes Whisper output English from any spoken
        # language (multilingual input); otherwise transcribe English.
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
            pad_token_id=whisper_processor.tokenizer.eos_token_id,
        )
        if task == "translate":
            gen_kwargs["task"] = "translate"
        else:
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"

        with torch.no_grad():
            predicted_ids = whisper_model.generate(**gen_kwargs)

        # Decode transcription (content not logged for privacy)
        text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"Transcription done ({len(text)} chars, task={task})")

        clear_gpu_cache()

        if not text:
            logger.warning("No speech detected in audio")
            return jsonify({'error': 'No speech detected in audio'}), 400

        return jsonify({
            'text': text,
            'language': 'auto-detected',
            'success': True
        })

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return jsonify({'error': f'Audio processing failed: {str(e)}'}), 500

@app.route('/text-to-gloss', methods=['POST'])
def text_to_gloss():
    """Convert English text to ASL gloss"""
    logger.info("Text-to-gloss request received")
    try:

        data = request.get_json()
        if not data or 'text' not in data:
            logger.warning("No text provided in request")
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if not text:
            logger.warning("Empty text provided")
            return jsonify({'error': 'Empty text provided'}), 400

        logger.info(f"Converting text to gloss: '{text}'")

        # Convert to ASL gloss
        gloss_result = asl_glosser.gloss(text)
        logger.info(f"Gloss result: '{gloss_result.gloss}'")

        return jsonify({
            'original_text': text,
            'gloss': gloss_result.gloss,
            'gloss_tokens': gloss_result.gloss_tokens,
            'non_manual_markers': gloss_result.sentence_nmm,
            'success': True
        })

    except Exception as e:
        logger.error(f"Text to gloss conversion failed: {e}")
        return jsonify({'error': f'Text to gloss conversion failed: {str(e)}'}), 500

@app.route('/gloss-to-sigml', methods=['POST'])
def gloss_to_sigml():
    """Convert ASL gloss to SiGML notation"""
    logger.info("Gloss-to-SiGML request received")
    try:

        data = request.get_json()
        if not data or 'gloss' not in data:
            logger.warning("No gloss provided in request")
            return jsonify({'error': 'No gloss provided'}), 400

        gloss = data['gloss'].strip()
        if not gloss:
            logger.warning("Empty gloss provided")
            return jsonify({'error': 'Empty gloss provided'}), 400

        logger.info(f"Converting gloss to SiGML: '{gloss}'")

        # Convert to SiGML
        sigml_xml = sigml_generator.generate_sigml(gloss)
        tokens = sigml_generator.gloss_to_tokens(gloss)
        token_kinds = [sigml_generator.classify_token(t) for t in tokens]
        fingerspelled = [t for t, k in zip(tokens, token_kinds) if k == "fingerspell"]
        logger.info(f"SiGML generated for {len(tokens)} tokens, {len(fingerspelled)} fingerspelled")

        return jsonify({
            'gloss': gloss,
            'tokens': tokens,
            'token_kinds': token_kinds,
            'fingerspelled': fingerspelled,
            'sigml': sigml_xml,
            'success': True
        })

    except Exception as e:
        logger.error(f"Gloss to SiGML conversion failed: {e}")
        return jsonify({'error': f'Gloss to SiGML conversion failed: {str(e)}'}), 500

# --- Video -> sign (hold-to-record) -------------------------------------------
# The signer records one sign and sends the clip; landmarks are extracted here
# with the same MediaPipe Holistic that produced the training data, so the input
# distribution matches the recognizer (a client-side landmarker drifts from it).
_holistic = None

def get_holistic():
    global _holistic
    if _holistic is None:
        import mediapipe as mp
        _holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1, refine_face_landmarks=False,
        )
    return _holistic

_HFACE, _HLH, _HPOSE, _HRH = 468, 21, 33, 21
_HTOTAL = _HFACE + _HLH + _HPOSE + _HRH

def video_to_landmarks(path, max_frames=150):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video")
    hol = get_holistic()
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        res = hol.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        f = np.full((_HTOTAL, 3), np.nan, dtype=np.float32)
        if res.face_landmarks:
            for i, lm in enumerate(res.face_landmarks.landmark[:_HFACE]):
                f[i] = [lm.x, lm.y, lm.z]
        if res.left_hand_landmarks:
            for i, lm in enumerate(res.left_hand_landmarks.landmark[:_HLH]):
                f[_HFACE + i] = [lm.x, lm.y, lm.z]
        if res.pose_landmarks:
            off = _HFACE + _HLH
            for i, lm in enumerate(res.pose_landmarks.landmark[:_HPOSE]):
                f[off + i] = [lm.x, lm.y, lm.z]
        if res.right_hand_landmarks:
            off = _HFACE + _HLH + _HPOSE
            for i, lm in enumerate(res.right_hand_landmarks.landmark[:_HRH]):
                f[off + i] = [lm.x, lm.y, lm.z]
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError("no frames decoded from clip")
    return np.stack(frames, axis=0)


# A signed phrase is one continuous clip holding several signs back to back.
# The isolated-sign recognizer classifies a window of frames. Signs vary in
# length, so each window center is classified at several scales and the
# posteriors averaged - this covers short and long signs and is the main
# accuracy lever offline. A short temporal smooth removes single-window noise,
# and windows with no hands present are blanked (no sign there). Consecutive
# identical labels above the confidence floor collapse into runs; a run that
# holds for at least _SEQ_MIN_RUN windows is emitted as one sign. The ordered
# runs are the signs of the phrase.
_SEQ_SCALES = (30, 45, 60)
_SEQ_STRIDE = 4
_SEQ_SMOOTH = 1
_SEQ_CONF = 0.45
_SEQ_MIN_RUN = 3
_SEQ_HAND_PRESENCE = 0.3

_LH0, _LH1 = _HFACE, _HFACE + _HLH
_RH0, _RH1 = _HFACE + _HLH + _HPOSE, _HFACE + _HLH + _HPOSE + _HRH

def _hand_presence(frames):
    lh = (~np.isnan(frames[:, _LH0:_LH1, 0])).any(axis=1)
    rh = (~np.isnan(frames[:, _RH0:_RH1, 0])).any(axis=1)
    return float((lh | rh).mean())

def landmarks_to_sequence(arr):
    n = arr.shape[0]
    big = max(_SEQ_SCALES)
    probs, presence = [], []
    for c in range(0, n, _SEQ_STRIDE):
        acc, used = None, 0
        for w in _SEQ_SCALES:
            lo, hi = max(0, c - w // 2), min(n, c + w // 2)
            if hi - lo < 10:
                continue
            p = np.asarray(
                asl_sign_recognizer.predict(arr[lo:hi], return_all_probs=True)['all_probs']
            )
            acc = p if acc is None else acc + p
            used += 1
        probs.append(acc / used if used else np.zeros(asl_sign_recognizer.num_signs))
        lo, hi = max(0, c - big // 2), min(n, c + big // 2)
        presence.append(_hand_presence(arr[lo:hi]))
    probs = np.stack(probs)

    if _SEQ_SMOOTH > 0:
        smoothed = np.zeros_like(probs)
        for i in range(len(probs)):
            lo, hi = max(0, i - _SEQ_SMOOTH), min(len(probs), i + _SEQ_SMOOTH + 1)
            smoothed[i] = probs[lo:hi].mean(axis=0)
        probs = smoothed

    runs = []
    window_labels = []
    for i in range(len(probs)):
        idx = int(np.argmax(probs[i]))
        conf = float(probs[i][idx])
        window_labels.append((asl_sign_recognizer.idx_to_sign[idx], round(conf, 2), round(presence[i], 2)))
        if presence[i] < _SEQ_HAND_PRESENCE:
            runs.append(None)
            continue
        if conf < _SEQ_CONF:
            runs.append(None)
            continue
        sign = asl_sign_recognizer.idx_to_sign[idx]
        if runs and runs[-1] is not None and runs[-1]['sign'] == sign:
            runs[-1]['count'] += 1
            runs[-1]['confidence'] = max(runs[-1]['confidence'], conf)
        else:
            runs.append({'sign': sign, 'confidence': conf, 'count': 1})

    mean_presence = float(np.mean(presence)) if presence else 0.0
    logger.info("seq diag: %d windows, mean_hand_presence=%.2f, window stream: %s",
                len(window_labels), mean_presence, window_labels)

    return [r for r in runs if r and r['count'] >= _SEQ_MIN_RUN]


# The three most recent recorded clips are retained on disk so a recognition
# result can be replayed against the original video while debugging.
_DEBUG_CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_clips")
_DEBUG_CLIPS_KEEP = 3

def retain_debug_clip(src_path, ext, signs):
    os.makedirs(_DEBUG_CLIPS_DIR, exist_ok=True)
    label = "-".join(signs) if signs else "none"
    dest = os.path.join(_DEBUG_CLIPS_DIR, f"clip_{label}_{secrets.token_hex(4)}{ext}")
    shutil.copyfile(src_path, dest)
    clips = sorted(
        (os.path.join(_DEBUG_CLIPS_DIR, f) for f in os.listdir(_DEBUG_CLIPS_DIR)),
        key=os.path.getmtime, reverse=True,
    )
    for old in clips[_DEBUG_CLIPS_KEEP:]:
        try:
            os.remove(old)
        except OSError:
            pass


@app.route('/video-to-signs', methods=['POST'])
def video_to_signs():
    if asl_sign_recognizer is None:
        return jsonify({'error': 'recognizer not initialized'}), 503
    if 'video' not in request.files:
        return jsonify({'error': 'no video provided'}), 400
    file = request.files['video']
    suffix = os.path.splitext(file.filename or '')[1] or '.webm'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()
        landmarks = video_to_landmarks(tmp.name, max_frames=600)
        seq = landmarks_to_sequence(landmarks)
        signs = [r['sign'] for r in seq]
        logger.info(f"video-to-signs: {' '.join(signs) or '(none)'} "
                    f"from {landmarks.shape[0]} frames")
        retain_debug_clip(tmp.name, suffix, signs)
        return jsonify({
            'signs': signs,
            'detail': seq,
            'num_frames': int(landmarks.shape[0]),
            'success': True,
        })
    except Exception as e:
        logger.error(f"video-to-signs failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ============================================================
# ASL gloss sequence -> natural English sentence (Azure OpenAI)
# ============================================================

_llm_client = None


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import AzureOpenAI
        _llm_client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
            api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
        )
    return _llm_client


@app.route('/sign-to-sentence', methods=['POST'])
def sign_to_sentence():
    """Turn a sequence of recognized ASL glosses into one English sentence."""
    try:
        data = request.get_json() or {}
        signs = data.get('signs')
        if not isinstance(signs, list) or not signs:
            return jsonify({'error': 'signs must be a non-empty array'}), 400

        gloss = " ".join(str(s).strip() for s in signs if str(s).strip())
        deployment = os.environ.get("AZURE_OPENAI_GENERATOR_DEPLOYMENT")
        if not deployment or not os.environ.get("OPENAI_API_KEY"):
            return jsonify({'error': 'LLM not configured'}), 503

        resp = get_llm_client().chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert a sequence of recognized American Sign Language "
                        "glosses into one natural, grammatical English sentence. ASL "
                        "drops articles and tense markers and reorders words; restore "
                        "them. Reply with ONLY the sentence, no quotes or notes."
                    ),
                },
                {"role": "user", "content": gloss},
            ],
            temperature=0.2,
            max_tokens=60,
        )
        sentence = (resp.choices[0].message.content or "").strip()
        if not sentence:
            return jsonify({'error': 'empty result'}), 502
        return jsonify({'sentence': sentence, 'success': True})
    except Exception as e:
        # Log only the error type; never the content or any credential detail.
        logger.error(f"sign-to-sentence failed: {type(e).__name__}")
        return jsonify({'error': 'sign-to-sentence failed'}), 500


# ============================================================
# Two-device room relay (Socket.IO)
#
# The server holds no conversation state. It only relays messages between
# the two members of a room. State is limited to room membership.
# ============================================================

# roomId -> { sid: role }
rooms = {}


def _new_room_id():
    """Generate a room id, collision-checked against active rooms."""
    while True:
        room_id = secrets.token_hex(3).upper()  # 6 hex chars
        if room_id not in rooms:
            return room_id


@socketio.on("create_room")
def on_create_room():
    room_id = _new_room_id()
    rooms[room_id] = {}
    logger.info(f"Room created: {room_id}")
    return {"roomId": room_id}


@socketio.on("join_room")
def on_join_room(data):
    data = data or {}
    room_id = data.get("roomId")
    role = data.get("role")

    if room_id not in rooms:
        return {"ok": False, "error": "room_not_found"}
    if role not in ("speaker", "signer"):
        return {"ok": False, "error": "invalid_role"}
    if len(rooms[room_id]) >= 2:
        return {"ok": False, "error": "room_full"}

    join_room(room_id)
    rooms[room_id][request.sid] = role
    emit("peer_joined", {"role": role}, to=room_id, include_self=False)
    logger.info(f"Join {room_id} as {role} (peers={len(rooms[room_id])})")
    return {
        "ok": True,
        "result": {"roomId": room_id, "role": role, "peers": len(rooms[room_id])},
    }


@socketio.on("message")
def on_message(data):
    data = data or {}
    room_id = data.get("roomId")
    members = rooms.get(room_id)
    if not members or request.sid not in members:
        return
    emit(
        "message",
        {
            "kind": data.get("kind"),
            "text": data.get("text"),
            "fromRole": members[request.sid],
        },
        to=room_id,
        include_self=False,
    )


@socketio.on("disconnect")
def on_disconnect():
    for room_id, members in list(rooms.items()):
        if request.sid in members:
            del members[request.sid]
            leave_room(room_id)
            emit("peer_left", to=room_id)
            logger.info(f"Leave {room_id} (peers={len(members)})")
            if not members:
                rooms.pop(room_id, None)
                logger.info(f"Room closed: {room_id}")


def create_app():
    """Application factory pattern"""
    return app

def check_model_status():
    """Check if all models are loaded properly"""
    models_status = {
        'whisper_processor': whisper_processor is not None,
        'whisper_model': whisper_model is not None,
        'asl_glosser': asl_glosser is not None,
        'sigml_generator': sigml_generator is not None,
        'asl_sign_recognizer': asl_sign_recognizer is not None
    }

    all_loaded = all(models_status.values())

    if not all_loaded:
        missing = [name for name, status in models_status.items() if not status]
        logger.error(f"Missing models: {missing}")
        raise RuntimeError(f"Models not loaded properly: {missing}")

    logger.info("All models verified and ready for inference")
    return True

# Initialize models when module is imported (ensuring they're available regardless of how app is started)
logger.info("INITIALIZING MODELS ON MODULE IMPORT...")
init_models()

if __name__ == '__main__':
    logger.info("STARTING SPEAK2SIGN API SERVER")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Whisper model: {WHISPER_MODEL_NAME}")
    logger.info(f"Data directory: data/")
    logger.info("=" * 60)

    try:
        # Verify models are loaded (they should be initialized on import)
        check_model_status()

        # Create app instance
        app_instance = create_app()

        logger.info("Starting Flask server...")
        logger.info("API endpoints available at http://localhost:5000")
        logger.info("Ready for inference requests!")
        logger.info("=" * 60)

        socketio.run(app_instance, debug=True, host='0.0.0.0', port=5000,
                     allow_unsafe_werkzeug=True)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("STARTUP FAILED")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        raise