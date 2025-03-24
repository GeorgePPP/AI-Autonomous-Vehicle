import asyncio
import base64
import os
import io
import re
import tempfile
from dotenv import load_dotenv
from pydub import AudioSegment

def get_api_key():
    """Get the OpenAI API key from environment variables"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def convert_audio_format(base64_audio, source_format, target_format='wav'):
    """
    Convert audio from one format to another.
    
    Args:
        base64_audio: String containing base64-encoded audio data
        source_format: Original format of the audio (e.g., 'webm')
        target_format: Desired format (must be 'wav' or 'mp3' for OpenAI)
        
    Returns:
        str: Base64-encoded audio in the target format
    """
    if target_format not in ['wav', 'mp3']:
        target_format = 'wav'  # Default to wav if invalid target format
        
    # Decode base64 to binary
    audio_binary = base64.b64decode(base64_audio)
    
    # Create temporary files for conversion
    with tempfile.NamedTemporaryFile(suffix=f'.{source_format}', delete=False) as source_file:
        source_file.write(audio_binary)
        source_path = source_file.name
    
    target_path = f"{os.path.splitext(source_path)[0]}.{target_format}"
    
    try:
        # Load audio file with pydub
        audio = AudioSegment.from_file(source_path, format=source_format)
        
        # Export to target format
        audio.export(target_path, format=target_format)
        
        # Read the converted file
        with open(target_path, 'rb') as target_file:
            converted_binary = target_file.read()
        
        # Encode back to base64
        converted_base64 = base64.b64encode(converted_binary).decode('utf-8')
        
        return converted_base64
    
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None  # Return None to indicate failure instead of returning original
    
    finally:
        # Clean up temporary files
        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(target_path):
            os.remove(target_path)


async def prepare_audio_message(base64_audio, audio_format='webm'):
    """
    Prepare audio data to be sent to OpenAI API asynchronously.
    
    Args:
        base64_audio: String containing base64-encoded audio data
        audio_format: Original format of the audio (default: 'webm')
        
    Returns:
        dict: Formatted message content with audio data for the API
    """
    # Always convert to a supported format (wav)
    target_format = 'wav'  # OpenAI supports 'wav' and 'mp3'
    
    try:
        # Convert the audio to a supported format
        print(f"Converting audio from {audio_format} to {target_format} format")
        
        # Run CPU-intensive conversion in a separate thread pool
        loop = asyncio.get_running_loop()
        converted_audio = await loop.run_in_executor(
            None, 
            lambda: convert_audio_format(base64_audio, audio_format, target_format)
        )
        
        if converted_audio is None:
            raise ValueError(f"Failed to convert audio from {audio_format} to {target_format}")
            
        # Create the audio data component
        audio_component = {
            "type": "input_audio",
            "input_audio": {
                "data": converted_audio,
                "format": target_format
            }
        }
        
        return audio_component
        
    except Exception as e:
        print(f"Audio preparation failed: {e}")
        raise


def test_base_64_string(audio_base64):
    """
    Validates that a string is properly base64 encoded.
    Returns True if valid, False otherwise.
    """
    # Check if it's a string
    if not isinstance(audio_base64, str):
        print("Error: Not a string type")
        return False
    
    # Check if the string contains valid base64 characters
    if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', audio_base64):
        print("Error: Invalid base64 characters")
        return False
    
    # Try to decode it
    try:
        decoded = base64.b64decode(audio_base64)
        # Check if it starts with common WAV header bytes
        # WAV files typically start with "RIFF" followed by file size and "WAVE"
        if len(decoded) > 12 and decoded[0:4] == b'RIFF' and decoded[8:12] == b'WAVE':
            print("Valid WAV file format")
        return True
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return False