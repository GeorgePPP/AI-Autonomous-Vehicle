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
        self.client.api_key = api_key  # Fixed API key assignment
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
    
    def _build_chain_of_thought(self, user_input: str) -> str:
        """Build chain of thought prompt for complex queries"""
        cot = self.prompt_templates["chain_of_thought_template"]
        return f"""
            Think through this step by step:
            {' '.join(cot['steps'])}

            User Input: {user_input}

            Generate a response following this thought process.
        """
    
    def _prepare_messages(
        self, 
        user_input: str = "", 
        audio_data: Optional[Dict] = None,
        use_cot: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for the OpenAI API in the correct format.
        
        Args:
            user_input: The user's text message (can be empty for audio-only)
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
            
            # Add text component if provided
            if user_input:
                content.append({"type": "text", "text": user_input})
            
            # Add audio component
            content.append(audio_data)
            
            user_message["content"] = content
        else:
            # For text-only messages, content can be a simple string
            user_message["content"] = user_input
        
        # Add the user message to the messages list
        messages.append(user_message)
        
        return messages

    async def send_message(
        self, 
        user_input: str = "",
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
                return None
                
            try:
                # Prepare the audio data for the API
                audio_data = await prepare_audio_message(audio_base64, audio_format)
                if not audio_data:
                    print("Failed to prepare audio data")
                    return None
            except Exception as e:
                print(f"Error preparing audio: {e}")
                return None
        
        # Prepare messages for the API
        messages = self._prepare_messages(
            user_input=user_input,
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
                # Store the user's message in history
                if audio_base64 and not user_input:
                    # For audio-only input, add a placeholder
                    user_content = "[Audio Input]"
                else:
                    user_content = user_input
                
                # Add to conversation history (for future context)
                self.conversation_history.append({"role": "user", "content": user_content})
                
                # Store the assistant's response in history
                assistant_message = response.choices[0].message
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_message.content
                })
                
                return assistant_message
            
        except Exception as e:
            print(f"Error making request: {e}")
            return None
            
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.current_context = {}