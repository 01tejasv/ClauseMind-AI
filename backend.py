import os
import openai
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4

# --- OpenAI key ---
openai.api_key = "sk-..."

# --- Pinecone client ---
pc = Pinecone(api_key="pcsk-...", environment="aped-4627-b74a")

index_name = "clause-mind-index"

# --- Create index if missing ---
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")  # adjust region if needed
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
