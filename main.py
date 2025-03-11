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
import config
from datetime import datetime

from utils import get_api_key, test_base_64_string
from chatbot import NDII

app = FastAPI()
templates = Jinja2Templates(directory=config.TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# Initialize ND II with API key
api_key = get_api_key()
nd_ii = NDII(api_key)

# Store active sessions
sessions = {}

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
        print("Audio data is not present in the response.")
    
    # Default fallback message
    if not text_output:
        text_output = "I received your message."
        
    return text_output, audio_base64

def add_messages_to_session(session_id, user_content, bot_content, bot_audio=None):
    """Add user and bot messages to a session"""
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

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "messages": [],
        "created_at": datetime.now()
    }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "session_id": session_id,
        "welcome_message": config.WELCOME_MESSAGE,
        "max_recording_duration": config.MAX_RECORDING_DURATION,
        "audio_only_mode": config.AUDIO_ONLY_MODE
    })

@app.get("/_chat_messages.html", response_class=HTMLResponse)
async def get_chat_messages(request: Request, session_id: str):
    """Endpoint to get just the chat messages for AJAX updates"""
    if session_id not in sessions:
        return RedirectResponse(url="/")
        
    return templates.TemplateResponse("_chat_messages.html", {
        "request": request,
        "messages": sessions[session_id]["messages"],
        "session_id": session_id
    })

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print("connection open")
    
    if session_id not in sessions:
        await websocket.close()
        return
    
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            
            if data.get("type") == "get_config":
                # Send configuration to client
                client_config = {
                    "type": "config",
                    "recording": {
                        "maxDuration": config.MAX_RECORDING_DURATION
                    },
                    "audio_only_mode": config.AUDIO_ONLY_MODE
                }
                await websocket.send_json(client_config)
                
            elif data.get("type") == "audio_recorded":
                # Get and validate audio data
                audio_base64 = data.get("audio_data")
                audio_format = data.get("format", "wav")
                
                if not audio_base64 or not test_base_64_string(audio_base64):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid audio data"
                    })
                    continue
                    
                try:
                    response = await nd_ii.send_message(
                        user_input="",  # Empty for audio-only
                        audio_base64=audio_base64,
                        audio_format=audio_format,
                        model=config.MODEL,
                        use_cot=config.USE_COT,
                        modalities=config.MODALITIES,
                        audio=config.AUDIO
                    )                    
                    # Process the response
                    text_output, response_audio = process_nd_ii_response(response)
                    
                    # Add messages to session
                    add_messages_to_session(session_id, "[Audio Input]", text_output, response_audio)
                    print(f"Added to session - Audio data present: {bool(response_audio)}")
                    print(f"Session message count: {len(sessions[session_id]['messages'])}")
                    print(f"Last message has audio: {bool(sessions[session_id]['messages'][-1].get('audio_base64'))}")
                    
                    # Send success response
                    await websocket.send_json({"type": "chat_updated", "success": True})
                    
                except Exception as e:
                    print(f"Error processing audio: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing audio: {str(e)}"
                    })
                    
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()
        print("connection closed")

async def cleanup_sessions():
    """Remove expired sessions"""
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, session_data in sessions.items()
        if (current_time - session_data["created_at"]).total_seconds() > config.SESSION_MAX_AGE
    ]
    
    for session_id in expired_sessions:
        del sessions[session_id]
    
    if expired_sessions:
        print(f"Cleaned up {len(expired_sessions)} expired sessions")

if __name__ == "__main__":
    # Ensure directories exist
    for directory in [config.TEMPLATE_DIR, config.STATIC_DIR]:
        os.makedirs(directory, exist_ok=True)
        
    # Run the FastAPI application
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=config.ENABLE_RELOAD)