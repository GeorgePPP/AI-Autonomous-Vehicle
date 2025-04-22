# ND II Voice Assistant

A conversational voice assistant for autonomous vehicles built with FastAPI and OpenAI's audio models.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

ND II is an AI assistant designed for autonomous vehicles that provides natural voice-based interactions. It processes audio input using OpenAI's speech recognition models, generates contextual responses with LLMs, and delivers audio output using text-to-speech.

### Key Features

- **Voice-First Interface**: Fully voice-driven interaction model
- **Real-Time Processing**: WebSocket communication for streaming audio
- **Session Management**: Maintains conversation context between interactions
- **RAG Integration**: Retrieves relevant context from a vector database
- **Context-Aware Responses**: Understands vehicle environment and user needs

## Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  Web Interface  │◄────►│  FastAPI Server │◄────►│  OpenAI APIs    │
│  (Browser)      │      │  (WebSockets)   │      │  (Audio/Chat)   │
│                 │      │                 │      │                 │
└─────────────────┘      └────────┬────────┘      └─────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │                 │
                         │  Vector Store   │
                         │  (PGVector)     │
                         │                 │
                         └─────────────────┘
```

### Components

- **`main.py`**: FastAPI server setup and endpoint definitions
- **`chatbot.py`**: Core NDII assistant implementation
- **`vector_store.py`**: PostgreSQL vector database for RAG functionality
- **`config.py`**: Configuration settings for models and environment
- **`utils.py`**: Helper functions for audio processing and API key management
- **`static/` & `templates/`**: Web interface assets and HTML templates

## Prerequisites

- Python 3.9+
- PostgreSQL with pgvector extension
- OpenAI API key

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ndii-voice-assistant.git
   cd ndii-voice-assistant
   ```

2. **Set up environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create PostgreSQL database**

   Install PostgreSQL and pgvector extension, then create a database:
   
   ```bash
   createdb ndii_db
   ```

4. **Configure environment**

   Create a `.env` file in the project root:
   
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

5. **Update configuration**

   Edit `config.py` to match your PostgreSQL settings:
   
   ```python
   PGVECTOR = {
       "connection_string": "postgresql://username:password@localhost:5432/ndii_db",
       # other settings...
   }
   ```

6. **Add knowledge sources**

   Place your knowledge base documents in the `source/` directory (PDF, DOCX, TXT formats supported).

7. **Run the application**

   ```bash
   python main.py
   ```

8. **Access the interface**

   Open your browser to `http://localhost:8000`

## Development

### Project Structure

```
ndii-voice-assistant/
├── main.py              # Main FastAPI application
├── chatbot.py           # Core assistant logic
├── vector_store.py      # Vector database implementation
├── config.py            # Configuration settings
├── utils.py             # Utility functions
├── static/              # Static assets
│   ├── script.js        # Client-side JavaScript
│   └── styles.css       # CSS styling
├── templates/           # HTML templates
│   ├── index.html       # Main interface template
│   └── _chat_messages.html  # Chat message partial
├── prompts/             # System prompts and examples
│   └── ndii_prompts.json  # Prompt templates
└── source/              # Knowledge base documents
```

## Configuration

Key configuration options in `config.py`:

- **OpenAI Models**: Change the text and audio models
- **Audio Settings**: Customize TTS voice and instructions
- **RAG Settings**: Adjust retrieval parameters and embedding models
- **Server Settings**: Modify host, port, and reload options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for speech recognition and text generation APIs
- FastAPI for the web framework
- pgvector for vector database functionality