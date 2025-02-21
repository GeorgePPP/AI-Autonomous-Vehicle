import json
import openai
from typing import List, Dict, Optional
from pathlib import Path

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
    
    def _build_system_message(self) -> Dict[str, str]:
        """Build the system message using the COSTAR framework"""
        system_prompt = self.prompt_templates["system_prompt"]
        costar = self.prompt_templates["costar_framework"]
        
        message = f"""
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
        return {"role": "system", "content": message}
    
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
    
    def _prepare_messages(self, user_input: str, use_cot: bool = False) -> List[Dict[str, str]]:
        """Prepare the complete message history with appropriate prompting"""
        messages = [self._build_system_message()]
        
        # Add few-shot examples for new conversations
        if not self.conversation_history:
            messages = self._add_few_shot_examples(messages)
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user input with optional chain of thought
        if use_cot:
            messages.append({"role": "user", "content": self._build_chain_of_thought(user_input)})
        else:
            messages.append({"role": "user", "content": user_input})
            
        return messages

    def send_message(
        self, 
        user_input: str, 
        use_cot: bool = False,
        **kwargs
    ) -> Optional[str]:
        """
        Send a message to ND II and get a response.
        
        Args:
            user_input: The user's message.
            use_cot: Whether to use chain of thought reasoning.
            **kwargs: Additional parameters for chat completion.
            
        Returns:
            The assistant's response or None if there's an error.
        """
        messages = self._prepare_messages(user_input, use_cot)
        
        try:
            response = openai.chat.completions.create(
                messages=messages,
                **kwargs  # Pass all additional parameters dynamically
            )
            
            if response.choices:
                assistant_message = response.choices[0].message
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message
            
        except Exception as e:
            print(f"Error making request: {e}")
            return None
            
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.current_context = {}
