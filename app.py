from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import base64
import soundfile as sf
import logging
import time
from asl_glosser import ASLGlosser, GlossResult
from sigml_generator import SiGMLGenerator
from audio_utils import load_audio_from_bytes

app = Flask(__name__)
CORS(app)

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
models_initialized = False

# Default Whisper model - optimized for speed (base is ~6x faster than large)
WHISPER_MODEL_NAME = "openai/whisper-base"

def init_models():
    """Initialize all models at server startup"""
    global whisper_processor, whisper_model, asl_glosser, sigml_generator, models_initialized

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

        total_time = whisper_time + glosser_time + sigml_time
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
            'audio_to_text_base64': '/audio-to-text-base64',
            'text_to_gloss': '/text-to-gloss',
            'gloss_to_sigml': '/gloss-to-sigml',
            'full_pipeline': '/full-pipeline',
            'text_pipeline': '/text-pipeline',
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
            'sigml_generator': sigml_generator is not None
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

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            logger.info(f"Audio file saved to: {tmp_file.name}")

            # Load and process audio
            try:
                audio, sample_rate = sf.read(tmp_file.name)
                logger.info(f"Audio loaded: sample_rate={sample_rate}, duration={len(audio)/sample_rate:.2f}s")
            except Exception as e:
                logger.error(f"Failed to load audio file: {e}")
                os.unlink(tmp_file.name)
                return jsonify({'error': f'Failed to load audio file: {str(e)}'}), 400

            # Resample audio to 16kHz if needed (Whisper requirement) - optimized
            if sample_rate != 16000:
                logger.info(f"Resampling audio from {sample_rate}Hz to 16000Hz")
                import librosa
                # Use faster resampling with lower quality for speed
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type='kaiser_fast')
                sample_rate = 16000

            # Process audio with Whisper
            logger.info(f"Processing audio with Whisper on device: {device}")
            inputs = whisper_processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info(f"Moved inputs to device: {device}, input_features shape: {inputs['input_features'].shape}")

            # Generate transcription (force English language)
            with torch.no_grad():
                predicted_ids = whisper_model.generate(
                    input_features=inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    language="en",
                    max_length=224,   # Reduced from 448 for faster generation
                    min_length=1,     # Minimum length to avoid empty outputs
                    num_beams=1,      # Greedy search (fastest)
                    do_sample=False,  # Deterministic output
                    temperature=0.0,  # Greedy decoding
                    use_cache=True,   # Enable KV caching
                    pad_token_id=whisper_processor.tokenizer.eos_token_id
                )

            # Decode transcription
            text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            logger.info(f"Transcription result: '{text}'")

            # Clean up temp file
            os.unlink(tmp_file.name)

            # Clear GPU cache after processing
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

@app.route('/audio-to-text-base64', methods=['POST'])
def audio_to_text_base64():
    """Convert base64 encoded audio to text"""
    logger.info("Base64 audio-to-text request received")
    try:

        data = request.get_json()
        if not data or 'audio' not in data:
            logger.warning("No base64 audio data provided")
            return jsonify({'error': 'No base64 audio data provided'}), 400

        # Decode base64 audio
        try:
            audio_data = base64.b64decode(data['audio'])
            logger.info(f"Decoded base64 audio data: {len(audio_data)} bytes")
        except Exception as e:
            logger.error(f"Failed to decode base64 audio: {e}")
            return jsonify({'error': 'Invalid base64 audio data'}), 400

        # Load audio using multiple fallback methods
        try:
            audio, sample_rate = load_audio_from_bytes(audio_data)
        except Exception as e:
            logger.error(f"Failed to load audio data: {e}")
            return jsonify({'error': 'Failed to process audio format'}), 400

        # Resample audio to 16kHz if needed (Whisper requirement) - optimized
        if sample_rate != 16000:
            logger.info(f"Resampling audio from {sample_rate}Hz to 16000Hz")
            import librosa
            # Use faster resampling with lower quality for speed
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type='kaiser_fast')
            sample_rate = 16000

        # Process audio with Whisper
        logger.info(f"Processing audio with Whisper on device: {device}")
        inputs = whisper_processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logger.info(f"Moved inputs to device: {device}, input_features shape: {inputs['input_features'].shape}")

        # Generate transcription (force English language) - optimized for speed
        log_gpu_memory_if_cuda("before generation")
        generation_start = time.time()

        with torch.no_grad():
            predicted_ids = whisper_model.generate(
                input_features=inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                language="en",
                max_length=224,   # Reduced from 448 for faster generation
                min_length=1,     # Minimum length to avoid empty outputs
                num_beams=1,      # Greedy search (fastest)
                do_sample=False,  # Deterministic output
                temperature=0.0,  # Greedy decoding
                use_cache=True,   # Enable KV caching
                pad_token_id=whisper_processor.tokenizer.eos_token_id
            )

        generation_time = time.time() - generation_start
        logger.info(f"Generation completed in {generation_time:.3f}s")

        log_gpu_memory_if_cuda("after generation")

        # Decode transcription
        text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"Transcription result: '{text}'")

        # Clear GPU cache after processing
        # Clear GPU cache after processing
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
        logger.error(f"Base64 audio processing failed: {e}")
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
        logger.info(f"SiGML generated for {len(tokens)} tokens")

        return jsonify({
            'gloss': gloss,
            'tokens': tokens,
            'sigml': sigml_xml,
            'success': True
        })

    except Exception as e:
        logger.error(f"Gloss to SiGML conversion failed: {e}")
        return jsonify({'error': f'Gloss to SiGML conversion failed: {str(e)}'}), 500

@app.route('/full-pipeline', methods=['POST'])
def full_pipeline():
    """Complete pipeline: audio -> text -> gloss -> SiGML"""
    logger.info("Full pipeline request received")
    try:

        # Check if audio file is provided
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid audio file'}), 400

        # Step 1: Audio to Text
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)

            # Load and process audio
            audio, sample_rate = sf.read(tmp_file.name)

            # Resample audio to 16kHz if needed (Whisper requirement) - optimized
            if sample_rate != 16000:
                logger.info(f"Resampling audio from {sample_rate}Hz to 16000Hz")
                import librosa
                # Use faster resampling with lower quality for speed
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type='kaiser_fast')
                sample_rate = 16000

            # Process audio with Whisper
            logger.info(f"Processing audio with Whisper on device: {device}")
            inputs = whisper_processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info(f"Moved inputs to device: {device}, input_features shape: {inputs['input_features'].shape}")

            # Generate transcription (force English language)
            with torch.no_grad():
                predicted_ids = whisper_model.generate(
                    input_features=inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    language="en",
                    max_length=224,   # Reduced from 448 for faster generation
                    min_length=1,     # Minimum length to avoid empty outputs
                    num_beams=1,      # Greedy search (fastest)
                    do_sample=False,  # Deterministic output
                    temperature=0.0,  # Greedy decoding
                    use_cache=True,   # Enable KV caching
                    pad_token_id=whisper_processor.tokenizer.eos_token_id
                )

            # Decode transcription
            text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            # Clean up temp file
            os.unlink(tmp_file.name)

            if not text:
                return jsonify({'error': 'No speech detected in audio'}), 400

        # Step 2: Text to Gloss
        gloss_result = asl_glosser.gloss(text)

        # Step 3: Gloss to SiGML
        sigml_xml = sigml_generator.generate_sigml(gloss_result.gloss)
        tokens = sigml_generator.gloss_to_tokens(gloss_result.gloss)

        return jsonify({
            'original_text': text,
            'language': 'auto-detected',
            'gloss': gloss_result.gloss,
            'gloss_tokens': gloss_result.gloss_tokens,
            'non_manual_markers': gloss_result.sentence_nmm,
            'sigml_tokens': tokens,
            'sigml': sigml_xml,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': f'Pipeline processing failed: {str(e)}'}), 500

@app.route('/text-pipeline', methods=['POST'])
def text_pipeline():
    """Text pipeline: text -> gloss -> SiGML"""
    logger.info("Text pipeline request received")
    try:

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400

        # Step 1: Text to Gloss
        gloss_result = asl_glosser.gloss(text)

        # Step 2: Gloss to SiGML
        sigml_xml = sigml_generator.generate_sigml(gloss_result.gloss)
        tokens = sigml_generator.gloss_to_tokens(gloss_result.gloss)

        return jsonify({
            'original_text': text,
            'gloss': gloss_result.gloss,
            'gloss_tokens': gloss_result.gloss_tokens,
            'non_manual_markers': gloss_result.sentence_nmm,
            'sigml_tokens': tokens,
            'sigml': sigml_xml,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': f'Text pipeline processing failed: {str(e)}'}), 500


def create_app():
    """Application factory pattern"""
    return app

def check_model_status():
    """Check if all models are loaded properly"""
    models_status = {
        'whisper_processor': whisper_processor is not None,
        'whisper_model': whisper_model is not None,
        'asl_glosser': asl_glosser is not None,
        'sigml_generator': sigml_generator is not None
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

        app_instance.run(debug=True, host='0.0.0.0', port=5000)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("STARTUP FAILED")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        raise