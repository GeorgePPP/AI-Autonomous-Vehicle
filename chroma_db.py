from dotenv import load_dotenv
load_dotenv(override=True)

import os
import asyncio
from typing import List, Dict, Any, Optional
from chromadb import Client
from chromadb.config import Settings
from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import AsyncOpenAI
import logging

# Set up logging
logger = logging.getLogger("DB")

class DB:
    def __init__(self, persist_directory: str, collection_name: str, rag_config: Optional[Dict[str, Any]] = None):
        """Initialize the DB class with ChromaDB and configuration.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            rag_config: Configuration for RAG system
        """
        self.embedding_client = AsyncOpenAI()
        self.chroma_client = Client(Settings(persist_directory=persist_directory))
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.rag_config = rag_config or {}  # Use empty dict if none provided
        
        # Log the configuration
        logger.info(f"DB initialized with collection: {collection_name}")
        logger.info(f"RAG config: {self.rag_config}")

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
            use_recursive = self.rag_config.get("use_recursive_splitter", False)
            
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
        model = self.rag_config.get("embedding_model", "text-embedding-3-small")
        try:
            response = await self.embedding_client.embeddings.create(
                model=model,
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
        batch_size = self.rag_config.get("batch_size", 1)
        all_embeddings = []

        try:
            if batch_size > 1:
                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    batch_contents = [text.page_content for text in batch]
                    
                    # Log batch processing
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}")
                    
                    response = await self.get_embedding(batch_contents)
                    
                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Rate limiting - optional sleep between batches
                    rate_limit_delay = self.rag_config.get("rate_limit_delay", 0)
                    if rate_limit_delay > 0 and i + batch_size < len(texts):
                        await asyncio.sleep(rate_limit_delay)
            else:
                # Process one by one (less efficient)
                for i, text in enumerate(texts):
                    logger.info(f"Processing document {i+1}/{len(texts)}")
                    response = await self.get_embedding(text.page_content)
                    embedding = response.data[0].embedding
                    all_embeddings.append(embedding)
                    
                    # Optional rate limiting
                    rate_limit_delay = self.rag_config.get("rate_limit_delay", 0)
                    if rate_limit_delay > 0 and i < len(texts) - 1:
                        await asyncio.sleep(rate_limit_delay)
                        
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise

    async def process_all_files(self, source_folder: str):
        """Process all files in a folder and add them to the collection.
        
        Args:
            source_folder: Path to the folder containing source files
        """
        processed_flag_file = os.path.join(self.rag_config.get("persist_directory", "chroma_storage"), ".processed")
        
        # Check if already processed (unless force_reprocess is set)
        force_reprocess = self.rag_config.get("force_reprocess", False)
        if os.path.exists(processed_flag_file) and not force_reprocess:
            logger.info("ChromaDB already initialized. Skipping.")
            return
        elif os.path.exists(processed_flag_file):
            print(".processed exists but collection is empty. Reprocessing...")

        supported_extensions = [f".{ext}" for ext in self.rag_config.get("document_types", ["docx", "txt", "pdf"])]
        
        # Track stats
        total_files = 0
        total_chunks = 0
        
        # Process each file
        for filename in os.listdir(source_folder):
            file_path = os.path.join(source_folder, filename)
            
            # Skip if not a file or not a supported type
            if not os.path.isfile(file_path) or not any(filename.lower().endswith(ext) for ext in supported_extensions):
                continue
                
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
                
                # Prepare data for ChromaDB
                documents = []
                metadatas = []
                ids = []
                
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    content = chunk.page_content
                    
                    # Skip empty or invalid chunks
                    if not isinstance(content, str) or content.strip() == "":
                        continue
                        
                    # Add to lists for batch insertion
                    documents.append(content)
                    
                    # Store metadata from document if available
                    metadata = {
                        "filename": filename,
                        "chunk_index": i
                    }
                    
                    # Add any metadata from the document
                    if hasattr(chunk, 'metadata') and chunk.metadata:
                        for key, value in chunk.metadata.items():
                            metadata[key] = str(value)
                            
                    metadatas.append(metadata)
                    ids.append(f"{filename}_{i}")
                    
                # Store in ChromaDB
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_chunks += len(documents)
                logger.info(f"Added {len(documents)} chunks from {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue

        # Set processed flag
        with open(processed_flag_file, "w") as f:
            f.write(f"done - {total_files} files, {total_chunks} chunks")
            
        logger.info(f"Completed processing {total_files} files with {total_chunks} total chunks")
