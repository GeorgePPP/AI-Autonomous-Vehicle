# config.py

# Model configuration
TEXT = {
    "model": "gpt-4o",
    "temperature": 0.6,
    "top_p": 0.5
}

# Audio configuration
AUDIO = {
    "model": "gpt-4o-mini-tts",
    "voice": "alloy", 
    "format": "wav", 
    "instructions": "Please speak in a super cheerful and welcoming with moderate tone that shows consideration."
}

# PostgreSQL vector database configuration
PGVECTOR = {
    "connection_string": "postgresql://postgres:test1234@localhost:5432/ndii_db",
    "table_name": "document_embeddings",
    "embedding_dim": 1536,  # Dimension for OpenAI text-embedding-3-large
    "index_method": "hnsw",  # Options: 'hnsw' (faster but approximate), 'ivfflat' (balance), or None (exact but slower)
    "index_params": {
        "m": 16,  # Number of connections per node (HNSW)
        "ef_construction": 64  # Construction time/accuracy tradeoff (HNSW)
    }
}

# RAG configuration
RAG = {
    "embedding_model": "text-embedding-3-large",
    "chunk_size": 512,
    "chunk_overlap": 200,
    "retrieval_k": 5,
    "similarity_threshold": 0.2,  # Minimum similarity score (0-1)
    "use_metadata_filtering": True,  # Enable filtering by metadata
    "reranking_enabled": False,  # Whether to use a separate reranker model
    "source_folder": "source",
    "document_types": ["docx", "txt", "pdf"],
    "use_recursive_splitter": True,  # Use recursive text splitter for better chunks
    "batch_size": 5,  # Number of documents to embed in a batch
    "rate_limit_delay": 0.05,  # Delay between API calls in seconds
    "force_reprocess": False  # Whether to force reprocessing of all documents
}

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
ENABLE_RELOAD = True

# Session configuration
SESSION_CLEANUP_INTERVAL = 3600  # Cleanup interval in seconds
SESSION_MAX_AGE = 86400  # Session expiration time in seconds (24 hours)

# Audio recording configuration
MAX_RECORDING_DURATION = 60  # Maximum recording duration in seconds

# Greeting message
GREETING_MESSAGE = "Hello! I'm your autonomous vehicle assistant. Press the microphone button to speak to me."

# Greeting prompt
GREETING_PROMPT = "Welcome the passanger by greeting them, introducing yourself, mentioning that the destination is Terminal 1 at Changi Airport and today's weather is sunny and pleasant, and asking if they need any assistance. Use a friendly and cheerful tone. "

# Template and static directories
TEMPLATE_DIR = "templates"
STATIC_DIR = "static"