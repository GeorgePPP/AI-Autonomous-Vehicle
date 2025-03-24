# ND II Voice Assistant

A conversational voice assistant for autonomous vehicles built with FastAPI and OpenAI's audio model.


## Features

- **Voice Input/Output**: Primary interaction through voice commands and responses
- **WebSocket Communication**: Real-time audio streaming and processing
- **Session Management**: Maintains conversation context across interactions
- **Chain of Thought Reasoning**: Can be enabled for complex queries

## Architecture

The project is structured as follows:

- `main.py`: FastAPI server setup and endpoint definitions
- `chatbot.py`: Core ND II assistant implementation using OpenAI API
- `config.py`: Configuration settings for models, server, and audio
- `utils.py`: Helper functions for audio processing and API key management
- `templates/`: HTML templates for the web interface
- `static/`: Static assets like CSS, JavaScript, and images
- `prompts/`: JSON files containing prompt templates and examples

## Setup

1. Clone the repository (current only jinja branch is working)
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in an environment variable or `.env` file
4. Ensure the `templates` and `static` directories exist
5. Run the application:
   ```
   python main.py
   ```

## Configuration

You can customize the assistant by modifying `config.py`. Key settings include:

- `MODEL`: The OpenAI model to use (default: "gpt-4o-audio-preview")
- `AUDIO_ONLY_MODE`: Set to True to only allow audio input
- `MAX_RECORDING_DURATION`: Maximum audio recording duration in seconds
- `SESSION_MAX_AGE`: How long sessions are kept active

## Usage

1. Access the web interface at `http://localhost:8000`
2. A new session will be created automatically
3. Press the microphone button to start recording
4. Speak your query/command
5. The assistant will process your audio and respond

## WebSocket API

The application uses WebSockets for real-time communication:

- Connect to `/ws/{session_id}`
- Send audio data as base64-encoded strings
- Receive responses with text and optional audio