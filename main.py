from fastapi import FastAPI, Request, Form, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

import uvicorn
import base64
import os
import json
import uuid
import asyncio
import logging
from datetime import datetime

from utils import get_api_key, test_base_64_string
from chatbot import NDII
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory=config.TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# Initialize ND II with API key
api_key = get_api_key()
nd_ii = NDII(api_key)

# Store active sessions
sessions = {}

# Session management
async def cleanup_sessions():
    """Remove expired sessions periodically"""
    while True:
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session_data in sessions.items()
            if (current_time - session_data["created_at"]).total_seconds() > config.SESSION_MAX_AGE
        ]
        
        for session_id in expired_sessions:
            del sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
        await asyncio.sleep(config.SESSION_CLEANUP_INTERVAL)

def process_nd_ii_response(response):
    """Process NDII response and extract text and audio data"""
    audio_base64 = None
    text_output = None
    
    # Handle string responses directly
    if isinstance(response, str):
        return response, None
        
    # Extract audio data if available
    if hasattr(response, 'audio') and response.audio:
        if hasattr(response.audio, 'data'):
            audio_base64 = response.audio.data
        if hasattr(response.audio, 'transcript'):
            text_output = response.audio.transcript
    else:
        logger.info("Audio data is not present in the response.")
    
    # Default fallback message
    if not text_output:
        text_output = "I received your message."
        
    return text_output, audio_base64

def add_messages_to_session(session_id, user_content, bot_content, bot_audio=None):
    """Add user and bot messages to a session"""
    if session_id not in sessions:
        logger.warning(f"Session {session_id} not found when adding messages")
        return False
        
    # Add user message
    sessions[session_id]["messages"].append({
        "sender": "user",
        "content": user_content,
        "timestamp": datetime.now()
    })
    
    # Add bot message
    sessions[session_id]["messages"].append({
        "sender": "bot",
        "content": bot_content,
        "audio_base64": bot_audio,
        "timestamp": datetime.now()
    })
    
    return True

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    asyncio.create_task(cleanup_sessions())
    logger.info("Session cleanup task started")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home page route that creates a new session"""
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "messages": [],
        "created_at": datetime.now()
    }
    
    logger.info(f"New session created: {session_id}")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "session_id": session_id,
        "welcome_message": config.WELCOME_MESSAGE,
        "max_recording_duration": config.MAX_RECORDING_DURATION,
    })

@app.get("/_chat_messages.html", response_class=HTMLResponse)
async def get_chat_messages(request: Request, session_id: str):
    """Endpoint to get just the chat messages for AJAX updates"""
    if session_id not in sessions:
        logger.warning(f"Session {session_id} not found, redirecting to home")
        return RedirectResponse(url="/")
        
    return templates.TemplateResponse("_chat_messages.html", {
        "request": request,
        "messages": sessions[session_id]["messages"],
        "session_id": session_id
    })

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    logger.info(f"WebSocket connection opened for session: {session_id}")
    
    if session_id not in sessions:
        logger.warning(f"Invalid session ID in WebSocket connection: {session_id}")
        await websocket.close()
        return
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                logger.error("Failed to parse WebSocket data as JSON")
                continue
            
            if data.get("type") == "get_config":
                # Send configuration to client
                client_config = {
                    "type": "config",
                    "recording": {
                        "maxDuration": config.MAX_RECORDING_DURATION
                    },
                }
                await websocket.send_json(client_config)
                
            elif data.get("type") == "audio_recorded":
                # Get and validate audio data
                audio_base64 = data.get("audio_data")
                audio_format = data.get("format", "wav")
                
                if not audio_base64 or not test_base_64_string(audio_base64):
                    logger.warning("Invalid audio data received")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid audio data"
                    })
                    continue
                
                logger.info(f"Processing audio input (format: {audio_format})")
                    
                try:
                    # Process audio with ND II
                    response = await nd_ii.send_message(
                        audio_base64=audio_base64,
                        audio_format=audio_format,
                        model=config.MODEL,
                        use_cot=config.USE_COT,
                        modalities=config.MODALITIES,
                        audio=config.AUDIO
                    )                    
                    
                    # Process the response
                    text_output, response_audio = process_nd_ii_response(response)
                    
                    logger.info(f"Response received - Audio output: {'Yes' if response_audio else 'No'}")
                    
                    # Add messages to session
                    success = add_messages_to_session(session_id, "[Audio Input]", text_output, response_audio)
                    
                    if success:
                        # Send success response
                        await websocket.send_json({"type": "chat_updated", "success": True})
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to update chat session"
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing audio: {str(e)}"
                    })
            else:
                logger.info(f"Received unknown message type: {data.get('type')}")
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
    finally:
        logger.info(f"WebSocket connection closed for session: {session_id}")
        await websocket.close()

if __name__ == "__main__":
    # Ensure directories exist
    for directory in [config.TEMPLATE_DIR, config.STATIC_DIR]:
        os.makedirs(directory, exist_ok=True)
        
    # Run the FastAPI application
    logger.info(f"Starting server on {config.HOST}:{config.PORT}")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=config.ENABLE_RELOAD)