#!/usr/bin/env python3
"""
Model downloader script for Speak2Sign Backend
Downloads Whisper model with progress tracking
"""

import os
import sys
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download
from tqdm import tqdm

WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"

def download_whisper_model():
    """Download Whisper model and processor with progress tracking"""
    print(f"Downloading Whisper model: {WHISPER_MODEL_NAME}")
    print("This may take several minutes depending on your internet connection...")

    try:
        # Download model files with progress bar
        print("\n1. Downloading model files...")
        snapshot_download(
            repo_id=WHISPER_MODEL_NAME,
            cache_dir=None,  # Use default cache directory
            resume_download=True,
            tqdm_class=tqdm
        )

        # Load processor (this will use cached files)
        print("\n2. Loading processor...")
        processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
        print("✓ Processor loaded successfully")

        # Load model (this will use cached files)
        print("\n3. Loading model...")
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME)
        print("✓ Model loaded successfully")

        # Display model info
        print(f"\nModel successfully downloaded and cached!")
        print(f"Model name: {WHISPER_MODEL_NAME}")
        print(f"Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"Cache location: {processor.name_or_path}")

        return True

    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def check_model_exists():
    """Check if model is already downloaded"""
    try:
        from transformers import WhisperProcessor
        WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME, local_files_only=True)
        return True
    except:
        return False

def main():
    """Main function"""
    print("Speak2Sign Backend - Model Downloader")
    print("=" * 50)

    # Check if model already exists
    if check_model_exists():
        print(f"✓ Model {WHISPER_MODEL_NAME} is already downloaded")
        response = input("Download again? (y/N): ").lower().strip()
        if response != 'y':
            print("Skipping download")
            return

    # Download model
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