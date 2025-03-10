import base64
import queue
import sys
import io
import threading
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional, Tuple
from dotenv import load_dotenv

def get_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def get_audio_input(samplerate=44100, channels=1, subtype='PCM_24', device=None, max_duration=60):
    """
    Record audio from the microphone until stopped by keyboard interrupt.
    
    Args:
        samplerate: Audio sample rate (default: 44100 Hz)
        channels: Number of audio channels (default: 1 for mono)
        subtype: Audio file format subtype (default: PCM_24)
        device: Audio input device ID (default: None for system default)
        max_duration: Maximum recording duration in seconds (default: 60)
        
    Returns:
        base64_audio: Base64 encoded audio string
        format: Format of the audio (e.g., 'wav')
    """
    # Create a queue to store audio blocks
    q = queue.Queue()
    
    # Define callback function with the queue
    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())
    
    # Create a list to store all audio blocks
    audio_blocks = []
    
    # Flag to control recording
    stop_recording = threading.Event()
    
    # Thread to listen for keyboard interrupt
    def kb_listener():
        print("Press Enter or Ctrl+C to stop recording...")
        try:
            input()  # Wait for Enter key
            stop_recording.set()
        except KeyboardInterrupt:
            stop_recording.set()
    
    # Start the keyboard listener thread
    kb_thread = threading.Thread(target=kb_listener)
    kb_thread.daemon = True
    kb_thread.start()
    
    # Record audio blocks until stopped
    try:
        print("Recording... (Press Enter to stop)")
        
        # Start recording stream
        with sd.InputStream(samplerate=samplerate, device=device,
                          channels=channels, callback=callback):
            
            # Set timeout for maximum duration
            timeout = time.time() + max_duration
            
            # Record until stop signal or timeout
            while not stop_recording.is_set() and time.time() < timeout:
                try:
                    # Get audio block with a timeout to check for stop flag periodically
                    block = q.get(timeout=0.1)
                    audio_blocks.append(block)
                except queue.Empty:
                    continue
                
        print("Recording finished.")
        
        # Check if any audio was recorded
        if not audio_blocks:
            print("No audio recorded.")
            return None, None
        
        # Combine all audio blocks into a single numpy array
        audio_data = np.concatenate(audio_blocks)
        
        # Use an in-memory buffer to encode as WAV
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, samplerate, format='WAV', subtype=subtype)
        
        # Get the buffer content and encode to base64
        wav_buffer.seek(0)
        base64_audio = base64.b64encode(wav_buffer.read()).decode('utf-8')
        
        # Return the base64 encoded audio and format
        return base64_audio, 'wav'
    
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None, None
    finally:
        # Ensure the recording stops even if there's an error
        stop_recording.set()