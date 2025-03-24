import asyncio
import json
from openai import AsyncOpenAI
import time
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from utils import prepare_audio_message

class NDII:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI()
        self.api_key = api_key
        self.client.api_key = api_key
        self.prompt_templates = self._load_prompts()
        self.conversation_history = []
        self.current_context = {}
        
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
    
    def _add_few_shot_examples(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add few-shot examples to the message history"""
        examples = self.prompt_templates["few_shot_examples"]
        for example in examples:
            messages.extend([
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]}
            ])
        return messages
    
    def _prepare_messages(
        self, 
        audio_data: Optional[Dict] = None,
        use_cot: bool = False
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
        
        # Add conversation history (excluding the current user input)
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

    async def send_message(
        self, 
        audio_base64: Optional[str] = None,
        audio_format: str = "wav",
        use_cot: bool = False,
        **kwargs
    ) -> Any:
        """
        Send a message to ND II and get a response.
        Supports both text and audio inputs formatted for OpenAI API.
        """
        # Prepare audio data if provided
        audio_data = None
        if audio_base64:
            # Validate the audio data
            if not audio_base64 or len(audio_base64) < 100:
                print("Invalid audio data received")
                return "I couldn't hear your message clearly. Could you try again?"
                
            try:
                # Prepare the audio data for the API
                audio_data = await prepare_audio_message(audio_base64, audio_format)
                if not audio_data:
                    print("Failed to prepare audio data")
                    return "There was an issue processing your audio. Please try again."
            except Exception as e:
                print(f"Error preparing audio: {e}")
                return f"There was an error processing your audio: {str(e)}"
        
        # Prepare messages for the API
        messages = self._prepare_messages(
            audio_data=audio_data,
            use_cot=use_cot
        )
        
        try:
            print(f"Sending request with prepared messages.")
            # Create the chat completion
            response = await self.client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            
            if response.choices:
                # TODO: Transcribe user audio input here
                # Store the assistant's response in history
                assistant_message = response.choices[0].message
                
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_message.audio.transcript
                })

                return assistant_message
            
        except Exception as e:
            print(f"Error making request: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
            
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.current_context = {}