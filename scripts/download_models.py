#!/usr/bin/env python3
"""
Model downloader script for Speak2Sign Backend
Downloads Whisper model with progress tracking
"""

import os
import sys

from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Matches the runtime default in app.py so the fetched and loaded models agree.
# Override both with the WHISPER_MODEL environment variable.
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "openai/whisper-base")

def download_whisper_model():
    print(f"Downloading Whisper model: {WHISPER_MODEL_NAME}")
    print("This may take several minutes depending on your internet connection...")

    try:
        print("\n1. Downloading model files...")
        snapshot_download(
            repo_id=WHISPER_MODEL_NAME,
            cache_dir=None,  # Use default cache directory
            resume_download=True,
            tqdm_class=tqdm
        )

        print("\n2. Loading processor...")
        processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
        print("✓ Processor loaded successfully")

        print("\n3. Loading model...")
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME)
        print("✓ Model loaded successfully")

        print("\nModel successfully downloaded and cached!")
        print(f"Model name: {WHISPER_MODEL_NAME}")
        print(f"Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"Cache location: {processor.name_or_path}")

        return True

    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def check_model_exists():
    try:
        from transformers import WhisperProcessor
        WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME, local_files_only=True)
        return True
    except Exception:
        return False

def main():
    print("Speak2Sign Backend - Model Downloader")
    print("=" * 50)

    if check_model_exists():
        print(f"✓ Model {WHISPER_MODEL_NAME} is already downloaded")
        response = input("Download again? (y/N): ").lower().strip()
        if response != 'y':
            print("Skipping download")
            return

    success = download_whisper_model()

    if success:
        print("\n" + "=" * 50)
        print("✓ All models downloaded successfully!")
        print("You can now run the Flask API server")
    else:
        print("\n" + "=" * 50)
        print("✗ Model download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()