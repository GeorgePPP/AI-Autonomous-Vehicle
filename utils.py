import asyncio
import base64
import os
import io
import re
import tempfile
import logging
from typing import Optional
from dotenv import load_dotenv
from pydub import AudioSegment

# Setup logging
logger = logging.getLogger(__name__)

def get_api_key() -> str:
    """
    Get the OpenAI API key from environment variables.
    
    Returns:
        str: The API key
        
    Raises:
        ValueError: If API key is not set
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def test_base_64_string(audio_base64: str) -> bool:
    """
    Validates that a string is properly base64 encoded.
    
    Args:
        audio_base64: The base64 string to test
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a string
    if not isinstance(audio_base64, str):
        logger.error("Base64 validation failed: Not a string type")
        return False
    
    # Check if the string contains valid base64 characters
    if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', audio_base64):
        logger.error("Base64 validation failed: Invalid base64 characters")
        return False
    
    # Try to decode it
    try:
        decoded = base64.b64decode(audio_base64)
        # For debugging - log the size of the decoded data
        size_kb = len(decoded) / 1024
        logger.info(f"Base64 decoded successfully: {size_kb:.2f} KB")
        
        # Additional check for WAV files (optional)
        if len(decoded) > 12 and decoded[0:4] == b'RIFF' and decoded[8:12] == b'WAVE':
            logger.info("Valid WAV file format detected")
            
        return True
    except Exception as e:
        logger.error(f"Base64 decoding failed: {str(e)}")
        return False

def convert_audio_format(
    base64_audio: str, 
    source_format: str, 
    target_format: str = 'wav'
) -> Optional[str]:
    """
    Convert audio from one format to another.
    
    Args:
        base64_audio: String containing base64-encoded audio data
        source_format: Original format of the audio (e.g., 'webm')
        target_format: Desired format (must be 'wav' or 'mp3' for OpenAI)
        
    Returns:
        Optional[str]: Base64-encoded audio in the target format, or None if conversion fails
    """
    if target_format not in ['wav', 'mp3']:
        target_format = 'wav'  # Default to wav if invalid target format
        
    # Log start of conversion
    logger.info(f"Converting audio from {source_format} to {target_format}")
    
    # Decode base64 to binary
    audio_binary = base64.b64decode(base64_audio)
    
    # Create temporary files for conversion
    source_path = None
    target_path = None
    
    try:
        # Create temp file for source audio
        with tempfile.NamedTemporaryFile(suffix=f'.{source_format}', delete=False) as source_file:
            source_file.write(audio_binary)
            source_path = source_file.name
        
        target_path = f"{os.path.splitext(source_path)[0]}.{target_format}"
        
        # Log audio file sizes for debugging
        logger.info(f"Source audio file size: {os.path.getsize(source_path) / 1024:.2f} KB")
        
        # Load audio file with pydub
        audio = AudioSegment.from_file(source_path, format=source_format)
        
        # Export to target format with appropriate settings
        if target_format == 'wav':
            # Use standard PCM format for best compatibility
            audio.export(
                target_path, 
                format=target_format,
                parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000"]
            )
        else:
            audio.export(target_path, format=target_format)
        
        # Log converted file size
        logger.info(f"Converted audio file size: {os.path.getsize(target_path) / 1024:.2f} KB")
        
        # Read the converted file
        with open(target_path, 'rb') as target_file:
            converted_binary = target_file.read()
        
        # Encode back to base64
        converted_base64 = base64.b64encode(converted_binary).decode('utf-8')
        
        return converted_base64
    
    except Exception as e:
        logger.error(f"Audio conversion failed: {str(e)}", exc_info=True)
        return None
    
    finally:
        # Clean up temporary files
        for path in [source_path, target_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {path}: {str(e)}")


async def prepare_audio_message(base64_audio: str, audio_format: str = 'webm') -> Optional[dict]:
    """
    Prepare audio data to be sent to OpenAI API asynchronously.
    
    Args:
        base64_audio: String containing base64-encoded audio data
        audio_format: Original format of the audio (default: 'webm')
        
    Returns:
        Optional[dict]: Formatted message content with audio data for the API
        
    Raises:
        ValueError: If audio conversion fails
    """
    # Always convert to a supported format (wav)
    target_format = 'wav'  # OpenAI supports 'wav' and 'mp3'
    
    try:
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
        
        logger.info("Audio prepared successfully for API request")
        return audio_component
        
    except Exception as e:
        logger.error(f"Audio preparation failed: {str(e)}", exc_info=True)
        raise