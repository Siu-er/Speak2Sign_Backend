import os
import tempfile
import logging
import librosa

logger = logging.getLogger(__name__)

def load_audio_from_bytes(audio_data):
    """Load audio from bytes data using librosa (ML standard)"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name

    try:
        # librosa automatically normalizes to [-1, 1] and converts to mono
        audio, sample_rate = librosa.load(tmp_path, sr=None, mono=True)
        logger.info(f"Audio loaded: sample_rate={sample_rate}, duration={len(audio)/sample_rate:.2f}s")
        return audio, sample_rate
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise
    finally:
        os.unlink(tmp_path)

