import os
import openai
from pinecone import Client
from uuid import uuid4

# Set OpenAI API key from environment variable (or hardcode if needed)
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "your_openai_api_key_here"  # uncomment and set if env var not used

# Pinecone API key and environment
pinecone_api_key = "pcsk_4FYWkq_2v8TPeMYDjuU2Y3mUXiL22bRjEG6BkFGTHu6wcxMn8QVG4y6erdzVaygE14zr78"
pinecone_env = "aped-4627-b74a"

# Initialize Pinecone client
pc = Client(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "clause-mind-index"

# Check existing indexes and create if missing
if index_name not in pc.list_indexes():
    pc.create_index(name=index_name, dimension=1536)

# Connect to the index
index = pc.index(index_name)

def process_and_index(content: str, filename: str):
    # Split content into chunks of 1000 chars
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

    for chunk in chunks:
        # Generate embedding from OpenAI
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embedding = response["data"][0]["embedding"]

        # Metadata for this chunk
        meta = {"filename": filename, "chunk": chunk}

        # Upsert the vector into Pinecone
        index.upsert(vectors=[(str(uuid4()), embedding, meta)])

    return {"status": "success", "chunks_indexed": len(chunks)}
