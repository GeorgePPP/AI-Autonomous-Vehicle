import json
import base64
import io
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

from openai import AsyncOpenAI
from utils import prepare_audio_message
from opik import track
from opik.integrations.openai import track_openai

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
        client = track_openai(AsyncOpenAI())
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
        sp = self.prompt_templates["system_prompt"]
        
        # Build main content sections more concisely
        content = [
            sp['context'],
            
            "SPEECH-ACT FRAMEWORK:",
            sp['speech_act_framework']['description'],
            "- " + "\n- ".join(sp['speech_act_framework']['components']),
            
            "INTERACTION PRINCIPLES:",
            sp['principles'],
            
            "CONVERSATION MODES:",
            "- Representatives: " + sp['modes']['representatives'],
            "- Directives: " + sp['modes']['directives'],
            "- Commissive: " + sp['modes']['commissive'],
            "- Expressives: " + sp['modes']['expressives'],
            "- Declarations: " + sp['modes']['declarations'],
            
            "CONSTRAINTS:",
            "- " + "\n- ".join(sp['constraints']),
            
            "PERSONA:",
            sp['persona'],
            
            "RESPONSE FORMAT:",
            sp['response_format']
        ]
        
        # Add examples more efficiently
        if "few_shot_examples" in sp:
            content.append("SPEECH-ACT EXAMPLES:")
            for i, ex in enumerate(sp["few_shot_examples"]):
                content.append(f"Example {i+1} - {ex['context']}:")
                content.append(f"Passenger: {ex['user']}")
                content.append(f"ND II (Locution): {ex['assistant_locution']}")
                content.append(f"[Internal Illocution: {ex['assistant_illocution']}]")
                content.append(f"[Internal Perlocution: {ex['assistant_perlocution']}]")
        
        # Join with single newlines to reduce unnecessary spacing
        text_content = "\n".join(content)
        
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
            return "\n\n".join(retrieved_docs)
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return ""

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
        context_text = await self.retrieve_context(user_query)

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
    
    async def get_llm_response(self, messages: list, text_config: dict) -> str:
        """Get response from LLM.
        
        Args:
            messages: List of message dictionaries for the LLM API
            text_config: Configuration for the text LLM
            
        Returns:
            Text response from the LLM
            
        Raises:
            Exception: If LLM request fails
        """
        logger.info(f"Sending request to LLM with {len(messages)} messages")
        
        # Create the chat completion
        response = await self.client.chat.completions.create(
            messages=messages,
            **text_config
        )
        
        if not response.choices:
            logger.warning("No choices in LLM response")
            return "I didn't receive a response. Please try again."
            
        # Get the text response
        text_output = response.choices[0].message.content
        
        if not text_output:
            logger.warning("Received empty response from LLM")
            text_output = "I processed your request but couldn't generate a response."
        
        logger.info(f"Generated text response: '{text_output[:50]}...'")
        return text_output

    async def send_message(
        self,
        audio_base64=None,
        audio_format=None,  # Only used for audio
        text_config=None,
        audio_config=None,
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Send a message to ND II and get a response with optional audio.
        
        Args:
            input: Text string or base64-encoded audio string
            input_type: Type of input - "text" or "audio"
            audio_format: Format of audio input (only used when input_type is "audio")
            text_config: Configuration for the text LLM
            audio_config: Configuration for the TTS response
            mode: "inference" or "evaluation"
            
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
                audio_data = await prepare_audio_message(audio_base64, audio_format)
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
        
        # Prepare messages for the LLM API
        messages, _ = await self._prepare_messages_for_llm(user_query)
        
        # Store retrieved context information
        if self.current_context:
            message_metadata["retrieved_chunks"] = self.current_context.get("chunks", [])
            message_metadata["chunk_metadata"] = self.current_context.get("metadata", [])
        
        # Get response from LLM
        try:
            text_output = await self.get_llm_response(messages, text_config)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": text_output})
            
            # Trim history if needed
            if self.max_history > 0 and len(self.conversation_history) > self.max_history * 4:
                self.conversation_history = self.conversation_history[-(self.max_history * 4):]
            
            # Generate speech from text response
            audio_base64 = await self.generate_speech(text=text_output, **audio_config)
            
            return text_output, audio_base64, message_metadata
            
        except Exception as e:
            logger.error(f"Error in LLM request: {e}")
            return f"Error processing your request: {str(e)}", None, message_metadata
    

    async def evaluate(self, metrics=None, eval_config=None):
        """Evaluate NDII performance using Opik.
        
        Args:
            metrics: List of Opik metrics to use (defaults to Hallucination)
            
        Returns:
            Evaluation results from Opik
        """
        from opik import Opik
        from opik.evaluation import evaluate
        from opik.evaluation.metrics import (Hallucination, Moderation, AnswerRelevance, ContextPrecision, ContextRecall)
        import asyncio, nest_asyncio
        
        # Apply nest_asyncio to allow running asyncio.run inside another event loop
        nest_asyncio.apply()
        
        # Use default metrics if none provided
        if not metrics:
            metrics = [Hallucination(), Moderation(), AnswerRelevance(), ContextPrecision(), ContextRecall()]

        if not eval_config:
            eval_config = {
                "model": "gpt-4o",
                "temperature": 0.2,
                "top_p": 0.2
            }
        
        # Define evaluation task with asyncio.run as recommended in the docs
        def evaluation_task(data_item):
            # Create a separate async function to handle the send_message call

            async def process_item():
                messages, context = await self._prepare_messages_for_llm(data_item['input'])
                
                text_output = await self.get_llm_response(
                    messages, eval_config
                )
                
                return {
                    "input": data_item['input'],
                    "output": text_output,
                    "context": context,
                    "reference": data_item['expected_output']['assistant_answer']
                }
            
            # Use asyncio.run as recommended in the documentation
            return asyncio.run(process_item())
        
        client = Opik()
        dataset = client.get_or_create_dataset(name="AI_AV")

        # Run evaluation with task_threads=1 as recommended for asyncio use
        evaluation_results = evaluate(
            dataset=dataset,
            task=evaluation_task,
            scoring_metrics=metrics,
            experiment_config={
                "model": config.TEXT.get("model", "gpt-4o"),
                "max_history": self.max_history,
                "temperature": config.TEXT.get("temperature", 0.7)
            },
            task_threads=1  # Important: set to 1 as recommended in docs
        )
        
        logger.info(f"Evaluation completed. Results: {evaluation_results}")
        return evaluation_results
        
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

