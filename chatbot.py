import json
import openai
import time
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from utils import prepare_audio_message

class NDII:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key  # Fixed API key assignment
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
        Prepare the complete message history with appropriate prompting.
        Supports both text and audio content in the format expected by OpenAI API.
        
        Args:
            user_input: The user's text message (can be empty for audio-only)
            audio_data: The formatted audio data
            use_cot: Whether to use chain of thought reasoning
            
        Returns:
            List of messages formatted for the OpenAI API
        """
        messages = [self._build_system_message()]
        
        # Add few-shot examples for new conversations
        if not self.conversation_history:
            messages = self._add_few_shot_examples(messages)
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Create the user message based on inputs
        if audio_data:
            # Create content array for the user message
            content = []
            
            # Add text component if provided
            if user_input:
                content.append({"type": "text", "text": user_input})
            
            # Add audio component
            content.append(audio_data)
            
            # Append the user message with the content array
            messages.append({"role": "user", "content": content})
        
        return messages

    def send_message(
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
        
        Args:
            user_input: The user's text message (optional if using audio)
            audio_base64: Base64-encoded audio data
            audio_format: Format of the audio (default: 'wav')
            use_cot: Whether to use chain of thought reasoning
            model: The OpenAI model to use (default: gpt-4o)
            **kwargs: Additional parameters for chat completion
            
        Returns:
            The assistant's response or None if there's an error
        """
        # Prepare audio data if provided
        audio_data = None
        if audio_base64:
            audio_data = prepare_audio_message(audio_base64, audio_format)
        
        # Prepare messages
        messages = self._prepare_messages(
            user_input=user_input,
            audio_data=audio_data,
            use_cot=use_cot
        )
        
        # Essential delay for async audio recorder
        time.sleep(1)

        try:
            print(f"Model Arguments: {kwargs}")
            # Create the chat completion with audio support
            response = openai.chat.completions.create(
                messages=messages,
                **kwargs
            )
            
            if response.choices:
                # Store the user's message in history
                if audio_base64 and not user_input:
                    # For audio-only input, add a placeholder in the history
                    self.conversation_history.append({"role": "user", "content": "[Audio Input]"})
                else:
                    # For text or text+audio
                    self.conversation_history.append({"role": "user", "content": user_input})
                
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