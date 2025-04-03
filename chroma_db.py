from dotenv import load_dotenv
load_dotenv(override=True)

import os
from chromadb import Client
from chromadb.config import Settings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from openai import AsyncOpenAI

class DB:
    def __init__(self, persist_directory, collection_name):
        self.embedding_client = AsyncOpenAI()
        self.chroma_client = Client(Settings(persist_directory=persist_directory))
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def split2chunks(self, file_path):
        loader = Docx2txtLoader(file_path)
        document = loader.load()

        chunk_size = 1000
        chunk_overlap = 400

        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        document_chunks = text_splitter.split_documents(document)

        return document_chunks

    async def get_embedding(self, text):
        response = await self.embedding_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
        return response

    async def embed_text(self, texts):
        responses = []

        for text in texts:
            response = self.get_embedding(text.page_content)
            responses.append(response)

        return responses

    async def process_all_files(self, source_folder):
        if os.path.exists("chroma_storage/.processed"):
            print("ChromaDB already initialized. Skipping.")
            return

        for filename in os.listdir(source_folder):
            file_path = os.path.join(source_folder, filename)

            if not os.path.isfile(file_path):
                continue  # skip subfolders or invalid files

            chunks = self.split2chunks(file_path)  # returns list of text-like objects
            for i, chunk in enumerate(chunks):
                content = chunk.page_content

                # Skip empty or invalid chunks
                if not isinstance(content, str) or content.strip() == "":
                    continue

                # Embed
                try:
                    embedding_response = await self.get_embedding(content)
                    embedding = embedding_response.data[0].embedding
                except Exception as e:
                    print(f"Embedding failed for chunk {i} in {filename}: {e}")
                    continue

                # Store in ChromaDB
                self.collection.add(
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[{
                        "filename": filename,
                        "chunk_index": i
                    }],
                    ids=[f"{filename}_{i}"]
                )

        with open("chroma_storage/.processed", "w") as f:
            f.write("done")      