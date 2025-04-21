import os
import asyncio
import logging
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import AsyncOpenAI

# Set up logging
logger = logging.getLogger("VectorStore")

@dataclass
class PGVectorConfig:
    """Configuration for PostgreSQL vector database."""
    connection_string: str
    table_name: str = "document_embeddings"
    embedding_dim: int = 1536  # Default for OpenAI text-embedding-3-large
    index_method: str = "hnsw"  # Options: 'hnsw', 'ivfflat'
    index_params: Dict[str, Any] = None


class VectorStore:
    def __init__(self, 
                pg_config: PGVectorConfig, 
                rag_config: Optional[Dict[str, Any]] = None):
        """Initialize the Vector Store class with PostgreSQL and configuration.
        
        Args:
            pg_config: Configuration for PostgreSQL vector database
            rag_config: Configuration for RAG system
        """
        self.embedding_client = AsyncOpenAI()
        self.pg_config = pg_config
        self.rag_config = rag_config or {}  # Use empty dict if none provided
        self.pool = AsyncConnectionPool(pg_config.connection_string, min_size=1, max_size=5)
        
        # Log the configuration
        logger.info(f"VectorStore initialized with table: {pg_config.table_name}")
        logger.info(f"RAG config: {self.rag_config}")

    async def initialize_db(self):
        """Initialize the database table and extensions."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Enable pgvector extension
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create the table for document embeddings
                await cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.pg_config.table_name} (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector({self.pg_config.embedding_dim}),
                        metadata JSONB,
                        file_name TEXT,
                        chunk_index INTEGER,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create the vector index based on the configuration
                index_params = self.pg_config.index_params or {}
                index_method = self.pg_config.index_method

                # Build index parameters string if parameters exist
                param_str = ""
                if index_params:
                    param_str = "WITH (" + ", ".join(f"{k}={v}" for k, v in index_params.items()) + ")"
                
                # Create appropriate index based on method
                if index_method == "hnsw":
                    await cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.pg_config.table_name}_embedding_idx 
                        ON {self.pg_config.table_name} 
                        USING hnsw (embedding vector_l2_ops)
                        {param_str};
                    """)
                elif index_method == "ivfflat":
                    await cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.pg_config.table_name}_embedding_idx 
                        ON {self.pg_config.table_name} 
                        USING ivfflat (embedding vector_l2_ops)
                        {param_str};
                    """)
                else:
                    # Default to exact search with no special index
                    await cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.pg_config.table_name}_embedding_idx 
                        ON {self.pg_config.table_name} 
                        USING ivfflat (embedding vector_l2_ops);
                    """)
                
                await conn.commit()
                logger.info("Database initialized with vector extension and embedding table")

    def _get_document_loader(self, file_path: str):
        """Get the appropriate document loader based on file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document loader instance
        
        Raises:
            ValueError: If file type is not supported
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.docx':
            return Docx2txtLoader(file_path)
        elif file_extension == '.txt':
            return TextLoader(file_path)
        elif file_extension == '.pdf':
            return PyPDFLoader(file_path)
        else:
            supported_types = self.rag_config.get("document_types", ["docx", "txt", "pdf"])
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: {supported_types}")

    def split2chunks(self, file_path: str) -> List[Document]:
        """Split a document into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks
        """
        try:
            # Get the appropriate loader
            loader = self._get_document_loader(file_path)
            document = loader.load()
            
            # Use config parameters with fallbacks
            chunk_size = self.rag_config.get("chunk_size", 1000)
            chunk_overlap = self.rag_config.get("chunk_overlap", 400)
            
            # Choose splitter based on config
            use_recursive = self.rag_config.get("use_recursive_splitter", True)
            
            if use_recursive:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
            else:
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
            
            document_chunks = text_splitter.split_documents(document)
            
            logger.info(f"Split {file_path} into {len(document_chunks)} chunks")
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error splitting document {file_path}: {e}")
            raise

    async def get_embedding(self, text: str or List[str]):
        """Get embedding for a text or list of texts.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            Embedding response
        """
        # Use the model from config
        model = self.rag_config.get("embedding_model", "text-embedding-3-large")
        try:
            response = await self.embedding_client.embeddings.create(
                model=model,
                dimensions=self.pg_config.embedding_dim,
                input=text
            )
            return response
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def embed_text(self, texts: List[Document]):
        """Embed a list of document chunks.
        
        Args:
            texts: List of document chunks
            
        Returns:
            List of embedding responses
        """
        # Implement batching if specified in config
        batch_size = self.rag_config.get("batch_size", 20)
        all_embeddings = []
        rate_limit_delay = self.rag_config.get("rate_limit_delay", 0)

        try:
            # Process in batches (batch_size=1 is handled by the same code)
            for i in range(0, len(texts), max(1, batch_size)):
                batch = texts[i:i+batch_size]
                batch_contents = [text.page_content for text in batch]
                
                # Log batch processing
                if batch_size > 1:
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}")
                else:
                    logger.info(f"Processing document {i+1}/{len(texts)}")
                
                response = await self.get_embedding(batch_contents)
                
                # Extract embeddings from response
                if batch_size > 1:
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                else:
                    embedding = response.data[0].embedding
                    all_embeddings.append(embedding)
                
                # Rate limiting - optional sleep between batches
                if rate_limit_delay > 0 and i + max(1, batch_size) < len(texts):
                    await asyncio.sleep(rate_limit_delay)
                    
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise

    async def add_chunks_to_db(self, chunks: List[Document], embeddings: List[List[float]], file_name: str):
        """Add document chunks and their embeddings to the database.
        
        Args:
            chunks: List of document chunks
            embeddings: List of embeddings corresponding to the chunks
            file_name: Name of the source file
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    content = chunk.page_content
                    
                    # Skip empty chunks
                    if not content or content.strip() == "":
                        continue
                    
                    # Prepare metadata
                    metadata = {
                        "source": file_name,
                        "chunk_index": i
                    }
                    
                    # Add metadata from the document
                    if hasattr(chunk, 'metadata') and chunk.metadata:
                        for key, value in chunk.metadata.items():
                            metadata[key] = str(value)
                    
                    # Convert metadata to JSON string for database insertion
                    import json
                    metadata_json = json.dumps(metadata)
                    
                    # Insert into database
                    await cur.execute(
                        f"""
                        INSERT INTO {self.pg_config.table_name} 
                        (content, embedding, metadata, file_name, chunk_index)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (content, embedding, metadata_json, file_name, i)
                    )
                
                await conn.commit()
                logger.info(f"Added {len(chunks)} chunks from {file_name} to database")

    async def find_unprocessed_documents(self, source_folder: str) -> List[str]:
        """Identify documents in the source folder that haven't been processed yet.
        
        Args:
            source_folder: Path to the folder containing source files
            
        Returns:
            List of filenames that need processing
        """
        # Get list of supported file extensions
        supported_extensions = [f".{ext}" for ext in self.rag_config.get("document_types", ["docx", "txt", "pdf"])]
        
        # Get all valid source files
        source_files = []
        for filename in os.listdir(source_folder):
            file_path = os.path.join(source_folder, filename)
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
                source_files.append(filename)
        
        if not source_files:
            logger.info("No documents found in source folder")
            return []
            
        # Get list of already processed files from database
        processed_files = []
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT DISTINCT file_name 
                    FROM {self.pg_config.table_name}
                    WHERE file_name IS NOT NULL
                """)
                results = await cur.fetchall()
                processed_files = [row['file_name'] for row in results]
        
        # Find files that need processing
        unprocessed_files = [f for f in source_files if f not in processed_files]
        
        logger.info(f"Found {len(unprocessed_files)} unprocessed documents")
        return unprocessed_files

    async def process_all_files(self, source_folder: str):
        """Process all files in a folder and add them to the vector store.
        
        Args:
            source_folder: Path to the folder containing source files
        """
        # Initialize database
        await self.initialize_db()
        
        # Check if force_reprocess is enabled
        force_reprocess = self.rag_config.get("force_reprocess", False)
        
        # Get list of files to process
        files_to_process = []
        if force_reprocess:
            # Process all files if force_reprocess is enabled
            supported_extensions = [f".{ext}" for ext in self.rag_config.get("document_types", ["docx", "txt", "pdf"])]
            for filename in os.listdir(source_folder):
                file_path = os.path.join(source_folder, filename)
                if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
                    files_to_process.append(filename)
                    
            # Clear existing data if force_reprocess is enabled
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"DELETE FROM {self.pg_config.table_name}")
                    await conn.commit()
                    logger.info("Cleared existing vector store data for reprocessing")
        else:
            # Only process unprocessed files
            files_to_process = await self.find_unprocessed_documents(source_folder)
            if not files_to_process:
                logger.info("No new documents to process")
                return
                
        # Track stats
        total_files = 0
        total_chunks = 0
        
        # Process each file
        for filename in files_to_process:
            file_path = os.path.join(source_folder, filename)
            logger.info(f"Processing file: {filename}")
            total_files += 1

            # Split document into chunks
            try:
                chunks = self.split2chunks(file_path)
                
                # Skip empty documents
                if not chunks:
                    logger.warning(f"No chunks created for {filename}")
                    continue
                    
                # Get embeddings for all chunks
                embeddings = await self.embed_text(chunks)
                
                # Verify chunks and embeddings match in length
                if len(chunks) != len(embeddings):
                    logger.error(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings for {filename}")
                    continue
                    
                # Add chunks to database
                await self.add_chunks_to_db(chunks, embeddings, filename)
                
                total_chunks += len(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue

        # Log processing summary
        logger.info(f"Completed processing {total_files} files with {total_chunks} total chunks")

    async def retrieve_context(self, query: str, k: int = 3, threshold: float = 0.0) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant context based on similarity search.
        
        Args:
            query: Query text
            k: Number of results to retrieve
            threshold: Minimum similarity threshold (0-1 scale, where 1 is most similar)
            
        Returns:
            Tuple of (list of document chunks, list of metadata for those chunks)
        """
        try:
            # Get embedding for query
            embedding_response = await self.get_embedding(query)
            query_embedding = embedding_response.data[0].embedding
            
            # Convert L2 distance threshold to cosine similarity threshold if provided
            distance_threshold = ""
            if threshold > 0:
                # PGVector uses L2 distance, so we need to convert from similarity (0-1) 
                # to distance threshold. Higher similarity = lower distance.
                # This is a rough approximation for normalized vectors
                l2_threshold = 2 * (1 - threshold)  # Convert cosine similarity to L2 distance
                distance_threshold = f"AND embedding <-> %s::vector < {l2_threshold}"
            
            # Retrieve similar documents
            async with self.pool.connection() as conn:
                conn.row_factory = dict_row
                async with conn.cursor() as cur:
                    sql_query = f"""
                        SELECT 
                            content, 
                            metadata,
                            1 - (embedding <-> %s::vector) / 2 AS similarity
                        FROM {self.pg_config.table_name}
                        WHERE TRUE {distance_threshold}
                        ORDER BY embedding <-> %s::vector
                        LIMIT %s
                    """
                    
                    # Only pass the embedding twice if using threshold
                    if threshold > 0:
                        await cur.execute(
                            sql_query,
                            (query_embedding, query_embedding, query_embedding, k)
                        )
                    else:
                        await cur.execute(
                            sql_query,
                            (query_embedding, query_embedding, k)
                        )
                    
                    results = await cur.fetchall()
                    
            # Extract contents and metadata
            contents = []
            metadatas = []
            
            for result in results:
                contents.append(result['content'])
                # Add similarity score to metadata
                metadata = result['metadata'] or {}
                metadata['similarity_score'] = result['similarity']
                metadatas.append(metadata)
                
                # Log retrieved chunks (truncated for readability)
                truncated_content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                logger.info(f"Retrieved chunk with similarity {result['similarity']:.4f}: {truncated_content}")
            
            logger.info(f"Retrieved {len(contents)} context chunks from database")
            return contents, metadatas
        
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return [], []
            
    async def close(self):
        """Close database connection pool."""
        await self.pool.close()
        logger.info("Vector store connections closed")