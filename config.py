# config.py

# Model configuration
TEXT = {
    "model": "gpt-4o",
}

# Audio configuration
AUDIO = {
    "model": "gpt-4o-mini-tts",
    "voice": "alloy", 
    "format": "wav", 
    "instructions": "Please speak in a super cheerful and welcoming with moderate tone that shows consideration."
}

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

# Vector database configuration
PERSIST_DIRECTORY = "chroma_storage/"
COLLECTION_NAME = "ndii_collection"