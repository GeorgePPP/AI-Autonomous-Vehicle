from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager

import uvicorn
import os
import json
import uuid
import logging
from datetime import datetime, timedelta

from utils import get_api_key, test_base_64_string
from chatbot import NDII
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active sessions
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global nd_ii, greeting_audio_base64, greeting_text_output

    api_key = get_api_key()
    nd_ii = await NDII.create_db(api_key, max_history=4, rag_config=config.RAG)  # Keep only last 2 exchanges
    
    # Generate greeting audio after DB initialization
    logger.info("Generating greeting audio after DB initialization")
    greeting_prompt_audio_base64 = await nd_ii.generate_speech(
        text=config.GREETING_PROMPT,
        voice="nova",
        format="wav"
    )

    greeting_text_output, greeting_audio_base64, _ = await nd_ii.send_message(
        audio_base64=greeting_prompt_audio_base64,
        audio_format="wav",
        text_config={"model": "gpt-4o", "temperature": 0.6, "top_p": 0.5, "max_tokens": 300, "timeout": 10},
        audio_config={"voice": "nova", "format": "wav"}
    )

    # Yield control back to FastAPI
    yield
    
    # Clean up resources when shutting down
    await nd_ii.close()
    logger.info("Application shutdown, resources cleaned up")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=config.TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

def add_messages_to_session(session_id, user_content, bot_content, bot_audio=None, message_metadata=None):
    """Add user and bot messages to a session"""
    if session_id not in sessions:
        logger.warning(f"Session {session_id} not found when adding messages")
        return False
    
    # Add user message (user_content now contains the transcribed query)
    user_message = {
        "sender": "user",
        "content": user_content,
        "timestamp": datetime.now()
    }
    
    sessions[session_id]["messages"].append(user_message)
    
    # Add bot message
    bot_message = {
        "sender": "bot",
        "content": bot_content,
        "audio_base64": bot_audio,
        "timestamp": datetime.now()
    }
    
    sessions[session_id]["messages"].append(bot_message)
    
    # Store RAG information with the session if available
    if message_metadata and message_metadata.get("retrieved_chunks"):
        # Store just the first 2-3 chunks to avoid excessive data
        max_chunks = 3
        shortened_chunks = []
        for i, chunk in enumerate(message_metadata["retrieved_chunks"][:max_chunks]):
            # Truncate long chunks to avoid massive log files
            max_length = 500
            shortened_chunk = chunk[:max_length] + "..." if len(chunk) > max_length else chunk
            
            # Add metadata if available
            chunk_info = {"content": shortened_chunk}
            if message_metadata.get("chunk_metadata") and i < len(message_metadata["chunk_metadata"]):
                chunk_info["metadata"] = message_metadata["chunk_metadata"][i]
                
            shortened_chunks.append(chunk_info)
            
        # Add the RAG data to the session
        if "rag_data" not in sessions[session_id]:
            sessions[session_id]["rag_data"] = []
            
        sessions[session_id]["rag_data"].append({
            "for_message_index": len(sessions[session_id]["messages"]) - 2,  # Index of the user message
            "chunks": shortened_chunks,
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
    
    # Use a single file per session
    filename = f"chatlog/session_{session_id}.json"
    
    # Create a clean copy of messages without audio data
    messages = []
    for msg in sessions[session_id]["messages"]:
        msg_copy = msg.copy()
        if "audio_base64" in msg_copy:
            del msg_copy["audio_base64"]
        messages.append(msg_copy)

    chat_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    
    # Add RAG data if available
    if "rag_data" in sessions[session_id]:
        chat_data["rag_data"] = sessions[session_id]["rag_data"]
    
    # Write to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, default=str)
        logger.info(f"Chat log saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving chat log: {e}")
        return False

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "messages": [],
        "created_at": datetime.now()
    }
    logger.info(f"New session created: {session_id}")  

    return templates.TemplateResponse("index.html", {
        "request": request,
        "session_id": session_id,
        "greeting_message": greeting_text_output,
        "greeting_audio": greeting_audio_base64,
        "max_recording_duration": config.MAX_RECORDING_DURATION,
    })

@app.post("/audio")
async def handle_audio_upload(request: Request):
    audio_bytes = await request.body()

    logger.info(f"Received audio file from Unreal, size: {len(audio_bytes)} bytes")

    # Convert to base64 and continue as usual
    import base64
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    text_output, response_audio_base64, _ = await nd_ii.send_message(
        audio_base64,
        "wav",
        config.TEXT,
        config.AUDIO
    )

    if not response_audio_base64:
        return JSONResponse(status_code=200, content={"message": text_output})

    response_bytes = base64.b64decode(response_audio_base64)
    save_path = os.path.join("response.wav")
    with open(save_path, "wb") as f:
        f.write(response_bytes)

    return FileResponse(save_path, media_type="audio/wav")

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
                    # Send to ND II for processing - now returns tuple of (text, audio_base64, message_metadata)
                    text_output, response_audio, message_metadata = await nd_ii.send_message(
                        audio_base64=audio_base64,
                        audio_format=audio_input_format,
                        text_config=config.TEXT,
                        audio_config=config.AUDIO
                    )
                    
                    logger.info(f"Response received - Audio output: {'Yes' if response_audio else 'No'}")
                    
                    # Use the actual transcription as the user message content if available
                    user_content = message_metadata.get("transcribed_query", "[Audio Input]")
                    
                    # Add messages to session with metadata
                    add_messages_to_session(
                        session_id, 
                        user_content,
                        text_output, 
                        response_audio,
                        message_metadata
                    )
                    
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