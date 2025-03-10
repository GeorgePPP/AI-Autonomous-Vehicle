import json
import openai
from typing import List, Dict, Optional
from pathlib import Path
from utils import get_audio_input

class NDII:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.w = api_key
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
    
    def _load_audio_input(self, samplerate=44100, channels=1, subtype='PCM_24', device=None, duration=5) -> Dict:
        """
        Load encoded audio string from the microphone using the utils function.
        Records audio and formats it according to OpenAI API requirements.
        
        Args:
            samplerate: Audio sample rate (default: 44100 Hz)
            channels: Number of audio channels (default: 1 for mono)
            subtype: Audio file format subtype (default: PCM_24)
            device: Audio input device ID (default: None for system default)
            duration: Recording duration in seconds (default: 5)
            
        Returns:
            Dict with type and audio data formatted for OpenAI API
        """
        try:
            # Get audio from the microphone using the optimized utility function
            base64_audio, audio_format = get_audio_input(
                samplerate=samplerate,
                channels=channels,
                subtype=subtype,
                device=device,
                duration=duration
            )
            
            if base64_audio is None:
                return {"error": "Failed to record audio"}
            
            # Format the audio data according to OpenAI API requirements
            audio_data = {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_audio,
                    "format": audio_format
                }
            }
            
            return audio_data
            
        except Exception as e:
            print(f"Error recording audio: {e}")
            return {"error": str(e)}

    def _build_system_message(self) -> Dict[str, str]:
        """Build the system message using the COSTAR framework"""
        system_prompt = self.prompt_templates["system_prompt"]
        costar = self.prompt_templates["costar_framework"]
        
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
        return {"role": "system", "content": text_content}
    
    def _add_few_shot_examples(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
    
    def _prepare_messages(self, user_input: str, use_cot: bool = False, include_audio: bool = False, audio_data: Dict = None) -> List[Dict]:
        """
        Prepare the complete message history with appropriate prompting.
        Supports both text and audio content in the format expected by OpenAI API.
        
        Args:
            user_input: The user's text message
            use_cot: Whether to use chain of thought reasoning
            include_audio: Whether to include audio content
            audio_data: The formatted audio data
            
        Returns:
            List of messages formatted for the OpenAI API
        """
        messages = [self._build_system_message()]
        
        # Add few-shot examples for new conversations
        if not self.conversation_history:
            messages = self._add_few_shot_examples(messages)
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Prepare the current user input
        if use_cot:
            cot_prompt = self._build_chain_of_thought(user_input)
            user_content = cot_prompt
        else:
            user_content = user_input
        
        # Format the user message based on content type
        if include_audio and audio_data:
            # Multi-modal message with text and audio
            content = [
                {"type": "text", "text": user_content},
                audio_data  # Already formatted as {"type": "input_audio", ...}
            ]
            messages.append({"role": "user", "content": content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": user_content})
        
        return messages

    def send_message(
        self, 
        user_input: str = "",
        use_audio: bool = False,
        audio_params: dict = None,
        use_cot: bool = False,
        model: str = "gpt-4o",
        **kwargs
    ) -> Optional[str]:
        """
        Send a message to ND II and get a response.
        Supports both text and audio inputs formatted for OpenAI API.
        
        Args:
            user_input: The user's text message (optional if using audio)
            use_audio: Whether to use audio input from the microphone
            audio_params: Parameters for audio recording (sample rate, channels, etc.)
            use_cot: Whether to use chain of thought reasoning
            model: The OpenAI model to use (default: gpt-4o)
            **kwargs: Additional parameters for chat completion
            
        Returns:
            The assistant's response or None if there's an error
        """
        audio_data = None
        
        # Handle audio input if requested
        if use_audio:
            audio_params = audio_params or {}
            audio_data = self._load_audio_input(**audio_params)
            
            # If there's an error with audio recording, return the error
            if isinstance(audio_data, dict) and "error" in audio_data:
                return f"Error with audio input: {audio_data['error']}"
        
        # Prepare messages with proper format
        messages = self._prepare_messages(
            user_input=user_input, 
            use_cot=use_cot,
            include_audio=use_audio,
            audio_data=audio_data
        )
        
        try:
            # Create the chat completion
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            if response.choices:
                assistant_message = response.choices[0].message.content
                
                # Store the user's message in history
                if use_audio and not user_input:
                    # For audio-only input, add a placeholder in the history
                    self.conversation_history.append({"role": "user", "content": "[Audio Input]"})
                else:
                    # For text or text+audio
                    self.conversation_history.append({"role": "user", "content": user_input})
                    
                # Store the assistant's response in history
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message
            
        except Exception as e:
            print(f"Error making request: {e}")
            return None
            
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.current_context = {}
