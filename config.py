# config.py

# Model configuration
MODEL = "gpt-4o-audio-preview"  # Use the audio-capable model
USE_COT = False
MODALITIES = ["text"]

# Audio configuration
AUDIO = {"voice": "alloy", "format": "wav"}
SUPPORTED_AUDIO_FORMATS = ["wav"]  # Supported audio formats for client-side recording

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
ENABLE_RELOAD = True

# Session configuration
SESSION_CLEANUP_INTERVAL = 3600  # Cleanup interval in seconds
SESSION_MAX_AGE = 86400  # Session expiration time in seconds (24 hours)

# Audio recording configuration
MAX_RECORDING_DURATION = 60  # Maximum recording duration in seconds

# Welcome message
WELCOME_MESSAGE = "Hello! I'm your autonomous vehicle assistant. Press the microphone button to speak to me."

# Template and static directories
TEMPLATE_DIR = "templates"
STATIC_DIR = "static"