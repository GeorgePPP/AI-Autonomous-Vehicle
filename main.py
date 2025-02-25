from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import uvicorn
import base64
import os
import config

from utils import get_api_key
from chatbot import NDII

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize ND II with API key
api_key = get_api_key()
nd_ii = NDII(api_key)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(user_input: str = Form(...)):
    try:
        # Send message with parameters from config
        response = nd_ii.send_message(
            user_input=user_input,
            use_cot=config.USE_COT,
            model=config.MODEL,
            modalities=config.MODALITIES,
            audio=config.AUDIO
        )
        
        # Save audio if available
        audio_path = None
        if hasattr(response, 'audio') and response.audio and hasattr(response.audio, 'data'):
            wav_bytes = base64.b64decode(response.audio.data)
            file_path = os.path.join(config.OUTPUT_DIR, config.FILE_NAME + config.FILE_EXT)
            with open(file_path, "wb") as f:
                f.write(wav_bytes)
            audio_path = file_path
        
        # Return the response text and audio path if available
        # Assume audio response
        if response.audio.transcript:
            text_output = response.audio.transcript
        else:
            text_output = "There's no transcription, please find the audio below for my response." 

        return JSONResponse({
            "response": text_output,
            "audio_base64": response.audio.data,
            "audio_path": str(audio_path)
        })

    except Exception as e:
        return JSONResponse({
            "response": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
            "audio_path": None
        })

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Create templates directory if it doesn't exist
    if not os.path.exists("templates"):
        os.makedirs("templates")
    
    # Create static directory if it doesn't exist
    if not os.path.exists("static"):
        os.makedirs("static")
        
    # Run the FastAPI application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)