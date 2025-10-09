# Speak2Sign Backend API

A Flask-based REST API for converting speech to American Sign Language (ASL) gloss notation and SiGML animation markup.

## Features

- **Audio to Text**: Convert audio files to text using OpenAI Whisper
- **Text to Gloss**: Convert English text to ASL gloss notation using rule-based processing
- **Gloss to SiGML**: Generate SiGML (Signing Gesture Markup Language) for sign animation
- **Full Pipeline**: Complete conversion from audio → text → gloss → SiGML

## API Endpoints

### Health Check
```
GET /health
```
Returns API status and model loading information.

### Audio Processing
```
POST /audio-to-text
Content-Type: multipart/form-data
Body: audio file (wav, mp3, flac, ogg, m4a, webm)
```

```
POST /audio-to-text-base64
Content-Type: application/json
Body: {"audio": "base64_encoded_audio_data"}
```

### Text Processing
```
POST /text-to-gloss
Content-Type: application/json
Body: {"text": "English text to convert"}
```

### Gloss Processing
```
POST /gloss-to-sigml
Content-Type: application/json
Body: {"gloss": "ASL gloss notation"}
```

### Pipeline Endpoints
```
POST /full-pipeline
Content-Type: multipart/form-data
Body: audio file
```

```
POST /text-pipeline
Content-Type: application/json
Body: {"text": "English text"}
```

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Create data directory structure:**
```
data/
├── lexicon.json
├── config.json
├── phrases.json
└── contractions.json
```

3. **Download required data files:**
The data files should contain ASL glossing rules and lexicon. You can download them from the original Speak2Sign repository.

4. **Run the application:**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## Usage Examples

### Convert audio to text
```bash
curl -X POST -F "audio=@sample.wav" http://localhost:5000/audio-to-text
```

### Convert text to gloss
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}' \
  http://localhost:5000/text-to-gloss
```

### Full pipeline
```bash
curl -X POST -F "audio=@sample.wav" http://localhost:5000/full-pipeline
```

## Response Format

All endpoints return JSON responses with the following structure:

```json
{
  "success": true,
  "original_text": "Hello world",
  "gloss": "HELLO WORLD",
  "gloss_tokens": ["HELLO", "WORLD"],
  "non_manual_markers": {
    "brows": null,
    "head": null,
    "qtype": null
  },
  "sigml": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>...",
  "sigml_tokens": ["HELLO", "WORLD"]
}
```

## Configuration

- **MAX_CONTENT_LENGTH**: 16MB (audio file size limit)
- **Whisper Model**: "small" (can be changed to "medium" or "large" for better accuracy)
- **Device**: Automatically detects CUDA availability

## Dependencies

- Flask & Flask-CORS for web API
- OpenAI Whisper for speech recognition
- PyTorch for ML model support
- soundfile & numpy for audio processing

## Notes

- Models are loaded lazily on first request to improve startup time
- Audio files are processed in temporary files and cleaned up automatically
- The API supports both file upload and base64 encoded audio
- CORS is enabled for cross-origin requests