# config.py

# chatbot response
MODEL = "gpt-4o-mini-audio-preview"
USE_COT = False
MODALITIES = ["text", "audio"]
AUDIO = {"voice": "alloy", "format": "wav"}

# input
USER_INPUT = "What are some good foods in Singapore"

# output
OUTPUT_DIR = "output"
FILE_NAME = "food_in_Singapore"
FILE_EXT = ".wav"