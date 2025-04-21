import asyncio
import json
import base64
import io
import logging
from openai import AsyncOpenAI
import time
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
from utils import prepare_audio_message
import numpy as np

import config
from vector_store import VectorStore, PGVectorConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ndii.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NDII")

class NDII:
    def __init__(self, client, prompt_templates, vector_store, api_key, max_history=4):
        """Initialize the NDII chatbot.
        
        Args:
            client: AsyncOpenAI client instance
            prompt_templates: Dictionary of prompt templates
            vector_store: VectorStore instance
            api_key: OpenAI API key
            max_history: Maximum number of conversation turns to keep in history
        """
        self.client = client
        self.client.api_key = api_key
        self.prompt_templates = prompt_templates
        self.conversation_history = []
        self.current_context = {}
        self.max_history = max_history
        self.vector_store = vector_store
    
    @classmethod
    async def create_db(cls, api_key: str, max_history: int = 2, rag_config: dict = None):
        """Create and initialize the database and NDII instance.
        
        Args:
            api_key: OpenAI API key
            max_history: Maximum number of conversation turns to keep in history
            rag_config: Configuration for RAG system
            
        Returns:
            NDII instance with initialized database
        """
        logger.info("Creating NDII instance with vector database")
        client = AsyncOpenAI()
        client.api_key = api_key

        try:
            # Create PGVector configuration from config
            pg_config = PGVectorConfig(
                connection_string=config.PGVECTOR["connection_string"],
                table_name=config.PGVECTOR["table_name"],
                embedding_dim=config.PGVECTOR["embedding_dim"],
                index_method=config.PGVECTOR["index_method"],
                index_params=config.PGVECTOR["index_params"]
            )
            
            # Initialize vector store
            vector_store = VectorStore(pg_config=pg_config, rag_config=rag_config)
            
            # Process all files in the source directory
            await vector_store.process_all_files(rag_config.get("source_folder", "source"))
            logger.info("Vector database successfully processed all files")
        except Exception as e:
            logger.error(f"Vector database initialization failed: {e}")
            raise

        prompt_templates = cls._load_prompts()

        return cls(client, prompt_templates, vector_store, api_key, max_history)

    @staticmethod
    def _load_prompts() -> Dict:
        """Load prompt templates from the prompts directory.
        
        Returns:
            Dictionary of prompt templates
        
        Raises:
            FileNotFoundError: If prompt templates file is not found
        """
        prompt_path = Path("prompts/ndii_prompts.json")
        if not prompt_path.exists():
            logger.error(f"Prompt templates not found at {prompt_path}")
            raise FileNotFoundError("Prompt templates not found")
            
        try:
            with open(prompt_path, 'r') as f:
                prompts = json.load(f)
                logger.info("Prompt templates loaded successfully")
                return prompts
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse prompts JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise

    def _build_system_message(self) -> Dict[str, str]:
        """Build the system message using the Speech Act framework with embedded few-shot examples.
        
        Returns:
            Dictionary with system message role and content
        """
        system_prompt = self.prompt_templates["system_prompt"]

        # Base system message sections loaded from config
        sections = [
            system_prompt['context'],
            
            "SPEECH-ACT FRAMEWORK:",
            system_prompt['speech_act_framework']['description'],
            "- " + "\n- ".join(system_prompt['speech_act_framework']['components']),
            
            "INTERACTION PRINCIPLES:",
            "- " + "\n- ".join(system_prompt['principles']),
            
            "CONVERSATION MODES:",
            "- " + "\n- ".join([f"{k}: {v}" for k, v in system_prompt['modes'].items()]),
            
            "CONSTRAINTS:",
            "- " + "\n- ".join(system_prompt['constraints']),
            
            "PERSONA:",
            system_prompt['persona'],
            
            "RESPONSE FORMAT:",
            system_prompt['response_format'],
        ]
        
        # Content assembled with proper spacing
        text_content = "\n\n".join(sections)
        
        # Embed few-shot examples into the system message
        examples = system_prompt.get("few_shot_examples", [])
        if examples:
            examples_text = "\n\nSPEECH-ACT EXAMPLES:\n"
            for i, example in enumerate(examples):
                examples_text += f"Example {i+1} - {example['context']}:\n"
                examples_text += f"Passenger: {example['user']}\n"
                examples_text += f"ND II (Locution): {example['assistant_locution']}\n"
                examples_text += f"[Internal Illocution: {example['assistant_illocution']}]\n"
                examples_text += f"[Internal Perlocution: {example['assistant_perlocution']}]\n\n"
            
            text_content += examples_text
        
        return {"role": "system", "content": text_content}
    
    async def retrieve_context(self, user_query: str, k: int = None) -> Tuple[list[str], List[Dict]]:
        """Retrieve relevant chunks from pgvector using OpenAI embedding.
        
        Args:
            user_query: User's query text to match against the vector database
            k: Number of results to retrieve (overrides config if provided)
            
        Returns:
            Tuple of (list of document chunks, list of metadata for those chunks)
        """
        # Use k from parameters or fall back to config
        k = k or config.RAG.get("retrieval_k", 3)
        
        # Get similarity threshold from config
        threshold = config.RAG.get("similarity_threshold", 0.0)
        
        try:
            # Use vector store's retrieve_context method
            retrieved_docs, metadatas = await self.vector_store.retrieve_context(
                query=user_query,
                k=k,
                threshold=threshold
            )
            
            # Store the context in the current_context for reference
            self.current_context = {
                "chunks": retrieved_docs,
                "metadata": metadatas,
                "for_query": user_query
            }
            
            # Log the retrieved chunks (truncated for readability)
            for i, (doc, meta) in enumerate(zip(retrieved_docs, metadatas)):
                truncated_content = doc[:100] + "..." if len(doc) > 100 else doc
                logger.info(f"Retrieved chunk {i+1}: {truncated_content}")
                if meta:
                    logger.info(f"Chunk {i+1} metadata: {meta}")
                    if 'similarity_score' in meta:
                        logger.info(f"Similarity score: {meta['similarity_score']:.4f}")
            
            logger.info(f"Retrieved {len(retrieved_docs)} context chunks from vector database")
            return retrieved_docs, metadatas
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return [], []

    async def _transcribe_audio(self, audio_data: Dict) -> str:
        """Transcribe audio data using OpenAI's Whisper API.
        
        Args:
            audio_data: Dictionary containing base64-encoded audio data
            
        Returns:
            Transcribed text from the audio
            
        Raises:
            Exception: If transcription fails
        """
        try:
            audio_base64 = audio_data["input_audio"]["data"]
            audio_bytes = base64.b64decode(audio_base64)

            # Wrap in a BytesIO object to simulate a file
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "input.wav"  # Required by OpenAI

            logger.info("Transcribing audio input")
            transcription = await self.client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                language='en',
                temperature=0,
                file=audio_file
            )
            
            logger.info(f"Audio transcribed: '{transcription.text[:50]}...'")
            return transcription.text
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise
        
    async def _prepare_messages_for_llm(self, user_query: str) -> Tuple[List[Dict[str, Any]], str]:
        """Prepare messages for the LLM API call with RAG context.
        
        Args:
            user_query: User's transcribed query text
            
        Returns:
            Tuple of (list of message dictionaries for the LLM API, context text)
        """
        # Retrieve context based on user query
        context_chunks, metadata = await self.retrieve_context(user_query)
        context_text = "\n\n".join(context_chunks)

        # Start with system message
        messages = [self._build_system_message()]

        if self.max_history > 0 and self.conversation_history:
            history_start = max(0, len(self.conversation_history) - (2 * self.max_history))
            messages.extend(self.conversation_history[history_start:])
        
        # Add context as a separate message if available
        if context_text:
            messages.append({
                "role": "system",
                "content": f"RELEVANT CONTEXT:\n{context_text}"
            })
        
        # Add user query
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        logger.info(f"Prepared {len(messages)} messages for LLM")
        return messages, context_text


    async def generate_speech(self, text: str = "", model: str = "gpt-4o-mini-tts", 
                             voice: str = "alloy", format: str = "wav", 
                             instructions: str = "") -> Optional[str]:
        """Generate text-to-speech audio using OpenAI's TTS API.
        
        Args:
            text: The text to convert to speech
            model: The TTS model to use
            voice: The voice to use
            format: The audio format
            instructions: Additional instructions for the TTS model
            
        Returns:
            base64-encoded audio data or None if generation fails
        """
        try:
            logger.info(f"Generating TTS for: '{text[:50]}...' using voice: {voice}")
            
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
                
                logger.info(f"TTS generated successfully: {len(base64_audio)/1024:.2f} KB")
                return base64_audio
                
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None

    async def send_message(
        self, 
        audio_base64,
        audio_input_format,
        text_config,
        audio_config
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Send a message to ND II and get a response with optional audio.
        
        Args:
            audio_base64: Base64-encoded audio input
            audio_input_format: Format of the input audio
            text_config: Configuration for the text LLM
            audio_config: Configuration for the TTS
            
        Returns:
            Tuple of (text_response, audio_base64, message_metadata)
        """
        # Initialize metadata dict to track important information
        message_metadata = {
            "transcribed_query": None,
            "retrieved_chunks": [],
            "chunk_metadata": []
        }
        
        # Process audio input if provided
        user_query = ""
        if audio_base64:
            # Validate the audio data
            if not audio_base64 or len(audio_base64) < 100:
                logger.warning("Invalid audio data received")
                return "I couldn't hear your message clearly. Could you try again?", None, message_metadata
                
            try:
                # Prepare the audio data for the API
                audio_data = await prepare_audio_message(audio_base64, audio_input_format)
                if not audio_data:
                    logger.error("Failed to prepare audio data")
                    return "There was an issue processing your audio. Please try again.", None, message_metadata
                
                # Transcribe the audio to text
                user_query = await self._transcribe_audio(audio_data)
                message_metadata["transcribed_query"] = user_query  # Store for logging
                logger.info(f"User query transcribed: '{user_query[:50]}...'")
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                return f"There was an error processing your audio: {str(e)}", None, message_metadata
        else:
            logger.warning("No audio input provided")
            return "I need an audio input to process your request.", None, message_metadata
        
        # Prepare messages for the LLM API with transcribed text
        messages, _ = await self._prepare_messages_for_llm(user_query)
        
        # Store retrieved chunks for logging
        if self.current_context:
            message_metadata["retrieved_chunks"] = self.current_context.get("chunks", [])
            message_metadata["chunk_metadata"] = self.current_context.get("metadata", [])
        
        # Get response from LLM
        try:
            logger.info(f"Sending request to LLM with {len(messages)} messages")
            
            # Create the chat completion
            response = await self.client.chat.completions.create(
                messages=messages,
                **text_config
            )
            
            if response.choices:
                # Get the text response
                text_output = response.choices[0].message.content
                
                if not text_output:
                    logger.warning("Received empty response from LLM")
                    text_output = "I processed your request but couldn't generate a response."
                
                # Add user input to history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": text_output
                })
                
                # Trim history if it exceeds max_history * 2 * 2 (each turn is 2 messages)
                if self.max_history > 0 and len(self.conversation_history) > self.max_history * 4:
                    self.conversation_history = self.conversation_history[-(self.max_history * 4):]
                
                logger.info(f"Generated text response: '{text_output[:50]}...'")
                
                # Generate speech from text response
                audio_base64 = await self.generate_speech(
                    text=text_output,
                    **audio_config
                )
                
                return text_output, audio_base64, message_metadata
            else:
                logger.warning("No choices in LLM response")
                return "I didn't receive a response. Please try again.", None, message_metadata
            
        except Exception as e:
            logger.error(f"Error in LLM request: {e}")
            return f"I encountered an error while processing your request: {str(e)}", None, message_metadata
                
    def reset_conversation(self):
        """Reset the conversation history"""
        logger.info("Resetting conversation history")
        self.conversation_history = []
        self.current_context = {}
        
    async def close(self):
        """Close database connections"""
        try:
            await self.vector_store.close()
            logger.info("Vector store connections closed")
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")