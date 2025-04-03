import asyncio
import json
import base64
import io
from openai import AsyncOpenAI
import time
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
from utils import prepare_audio_message

class NDII:
    def __init__(self, api_key: str, max_history: int = 2):
        self.client = AsyncOpenAI()
        self.api_key = api_key
        self.client.api_key = api_key
        self.prompt_templates = self._load_prompts()
        self.conversation_history = []
        self.current_context = {}
        self.max_history = max_history
        
    def _load_prompts(self) -> Dict:
        """Load prompt templates from the prompts directory"""
        prompt_path = Path("prompts/ndii_prompts.json")
        if not prompt_path.exists():
            raise FileNotFoundError("Prompt templates not found")
            
        with open(prompt_path, 'r') as f:
            return json.load(f)
    
    def _build_system_message(self) -> Dict[str, str]:
        """Build the system message using the COSTAR framework with embedded few-shot examples"""
        system_prompt = self.prompt_templates["system_prompt"]
        costar = self.prompt_templates["costar_framework"]
        
        # Base system message content
        text_content = f"""
            {system_prompt['context']}

            CONSTRAINTS:
            {' '.join(system_prompt['constraints'])}

            OBJECTIVES:
            {' '.join(system_prompt['objectives'])}

            OPERATING FRAMEWORK:
            Context: {costar['context']}
            Objectives: {costar['objectives']}
            Steps: {costar['steps']}
            Tools: {costar['tools']}
            Actions: {costar['actions']}
            Review: {costar['review']}

            STYLE AND PERSONALITY:
            {json.dumps(system_prompt['style'], indent=2)}
        """
        
        # Embed few-shot examples into the system message if available
        examples = self.prompt_templates.get("few_shot_examples", [])
        if examples:
            examples_text = "\n\nEXAMPLES:\n"
            for i, example in enumerate(examples):
                examples_text += f"Example {i+1}:\n"
                examples_text += f"User: {example['user']}\n"
                examples_text += f"Assistant: {example['assistant']}\n\n"
            
            text_content += examples_text
        
        return {"role": "system", "content": text_content}
    
    def _prepare_messages(
        self, 
        audio_data: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for the OpenAI API in the correct format.
        
        Args:
            audio_data: The formatted audio data
            use_cot: Whether to use chain of thought reasoning
            
        Returns:
            List of messages formatted for the OpenAI API
        """
        # Start with system message (including embedded few-shot examples)
        messages = [self._build_system_message()]
        
        # Add conversation history (only the latest ones, based on max_history)
        messages.extend(self.conversation_history)
        
        # Create the user message with proper content formatting
        user_message = {"role": "user"}
        
        # Format content as an array when using audio or both text and audio
        if audio_data:
            content = []
            
            # Add audio component
            content.append(audio_data)
            
            user_message["content"] = content
        
        # Add the user message to the messages list
        messages.append(user_message)
    
        return messages

    async def generate_speech(self, text: str = "", model:str = "gpt-4o-mini-tts", voice: str = "alloy", format: str = "wav", instructions: str = "") -> Optional[str]:
        """
        Generate text-to-speech audio using OpenAI's TTS API
        
        Args:
            text: The text to convert to speech
            voice: The voice to use
            format: The audio format
            
        Returns:
            base64-encoded audio data
        """
        try:
            print(f"Generating TTS for: '{text[:50]}...' using voice: {voice}")
            
            # Call OpenAI's TTS API
            async with self.client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                response_format=format,
                input=text,
                instructions=instructions
            ) as response:
                            
                audio_data = await response.read()
                
                # Convert to base64 for transmission
                base64_audio = base64.b64encode(audio_data).decode('utf-8')
                
                print(f"TTS generated successfully: {len(base64_audio)/1024:.2f} KB")
                return base64_audio
                
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None

    async def send_message(
        self, 
        audio_base64,
        audio_input_format,
        text_config,
        audio_config
    ) -> Tuple[str, Optional[str]]:
        """
        Send a message to ND II and get a response.
        Supports both text and audio inputs formatted for OpenAI API.
        
        Returns:
            Tuple of (text_response, audio_base64)
        """
        # Prepare audio data if provided
        audio_data = None
        if audio_base64:
            # Validate the audio data
            if not audio_base64 or len(audio_base64) < 100:
                print("Invalid audio data received")
                return "I couldn't hear your message clearly. Could you try again?", None
                
            try:
                # Prepare the audio data for the API
                audio_data = await prepare_audio_message(audio_base64, audio_input_format)
                if not audio_data:
                    print("Failed to prepare audio data")
                    return "There was an issue processing your audio. Please try again.", None
            except Exception as e:
                print(f"Error preparing audio: {e}")
                return f"There was an error processing your audio: {str(e)}", None
        
        # Prepare messages for the API
        messages = self._prepare_messages(
            audio_data=audio_data
        )
        
        try:
            print(f"Sending request with prepared messages.")
            
            # Create the chat completion
            response = await self.client.chat.completions.create(
                messages=messages,
                **text_config
            )
            
            if response.choices:
                # Get the text response
                text_output = response.choices[0].message.content
                
                if not text_output:
                    text_output = "I processed your request but couldn't generate a response."
                
                # Add user input placeholder to history
                self.conversation_history.append({
                    "role": "user",
                    "content": "[Audio Input]"  # Placeholder for user's audio input
                })
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": text_output
                })
                
                audio_base64 = await self.generate_speech(
                    text = text_output,
                    **audio_config
                )
                
                return text_output, audio_base64
            else:
                return "I didn't receive a response. Please try again.", None
            
        except Exception as e:
            print(f"Error making request: {e}")
            return f"I encountered an error while processing your request: {str(e)}", None
            
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.current_context = {}