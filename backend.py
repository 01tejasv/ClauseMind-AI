import os
from uuid import uuid4
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# --- Keys from environment variables ---
# If you prefer, you can also directly set your keys here:
openai.api_key = "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"
pinecone_api_key = os.environ.get("PINECONE_API_KEY")  # Optional: set in .env
pinecone_env = os.environ.get("PINECONE_ENV", "us-east1-gcp")  # default region if not in .env

# --- Initialize Pinecone client ---
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "clause-mind-index"

# --- Create index if missing ---
existing_indexes = [idx.name() for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # adjust region if needed
    )

# --- Connect to index ---
index = pc.index(index_name)

# --- Function to process & index ---
def process_and_index(content: str, filename: str):
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
    for chunk in chunks:
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )["data"][0]["embedding"]

        meta = {"filename": filename, "chunk": chunk}
        index.upsert(vectors=[(str(uuid4()), embedding, meta)])

    return {"status": "success", "chunks_indexed": len(chunks)}
