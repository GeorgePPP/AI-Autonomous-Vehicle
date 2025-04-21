from fastapi import FastAPI, Request, Form, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager

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

# Initialize ND II with API key
api_key = get_api_key()

# Store active sessions
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global nd_ii
    api_key = get_api_key()
    nd_ii = await NDII.create_db(api_key, max_history=4, rag_config=config.RAG)  # Keep only last 2 exchanges
    
    # Yield control back to FastAPI
    yield
    
    # Clean up resources when shutting down
    await nd_ii.close()
    logger.info("Application shutdown, resources cleaned up")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=config.TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

def process_nd_ii_response(response):
    """Process NDII response and extract text and audio data"""
    # For the updated implementation, response is now a tuple of (text_output, audio_base64)
    text_output, audio_base64 = response
    
    # Default fallback message if text is empty
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

async def save_chat_to_file(session_id: str):
    """Save chat messages to a JSON file in the chatlog directory"""
    if session_id not in sessions:
        logger.warning(f"Session {session_id} not found when saving chat log")
        return False
        
    # Create chatlog directory if it doesn't exist
    os.makedirs("chatlog", exist_ok=True)
    
    # Use a single file per session that gets updated
    filename = f"chatlog/session_{session_id}.json"
    
    # Create a clean copy of messages without audio data
    messages = []
    for msg in sessions[session_id]["messages"]:
        msg_copy = msg.copy()
        if "audio_base64" in msg_copy:
            del msg_copy["audio_base64"]
        messages.append(msg_copy)

    # Prepare the data to save
    chat_data = {
        "session_id": session_id,
        "created_at": sessions[session_id]["created_at"].isoformat(),
        "updated_at": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": messages
    }
    
    # Write to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, default=str)
        logger.info(f"Chat log updated in {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving chat log: {e}")
        return False

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
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
    await websocket.accept()
    print("connection open")
    
    if session_id not in sessions:
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
                audio_input_format = data.get("format", "wav")
                
                if not audio_base64 or not test_base_64_string(audio_base64):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid audio data"
                    })
                    continue
                    
                try:
                    # Send to ND II for processing - now returns tuple of (text, audio_base64)
                    response = await nd_ii.send_message(
                        audio_base64,
                        audio_input_format,
                        config.TEXT,
                        config.AUDIO
                    )                    
                    
                    # Process the response - unpack the tuple
                    text_output, response_audio = process_nd_ii_response(response)
                    logger.info(f"Response received - Audio output: {'Yes' if response_audio else 'No'}")
                    
                    # Add messages to session
                    add_messages_to_session(session_id, "[Audio Input]", text_output, response_audio)
                    # Save chat log after processing message
                    await save_chat_to_file(session_id)
                    print(f"Added to session - Audio data present: {bool(response_audio)}")
                    
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
        # Save chat log before removing the session
        await save_chat_to_file(session_id)
        del sessions[session_id]
    
    if expired_sessions:
        print(f"Cleaned up {len(expired_sessions)} expired sessions")

if __name__ == "__main__":
    # Ensure directories exist
    for directory in [config.TEMPLATE_DIR, config.STATIC_DIR]:
        os.makedirs(directory, exist_ok=True)
        
    # Run the FastAPI application
    logger.info(f"Starting server on {config.HOST}:{config.PORT}")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=config.ENABLE_RELOAD)