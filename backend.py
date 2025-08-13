import os
import openai
import pinecone
from uuid import uuid4

# --- OpenAI key ---
openai.api_key = "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"

# --- Pinecone key & environment ---
pinecone_api_key = "pcsk_5TezHr_5FS18xfebbQxaGZwSRUELygH9RUq7Hnor9u7FdDDYoq4ztLAS2Fv2LC5W19Z4Me"
pinecone_env = "aped-4627-b74a"

# --- Initialize Pinecone ---
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "clause-mind-index"

# --- Create index if missing (using serverless spec for embeddings) ---
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        pod_type="p1"   # or "s1" for serverless; adjust if needed
    )

# --- Connect to the index ---
pc = pinecone.Index(index_name)

# --- Function to process & index ---
def process_and_index(content: str, filename: str):
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
    for chunk in chunks:
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )["data"][0]["embedding"]
        meta = {"filename": filename, "chunk": chunk}
        pc.upsert(vectors=[(str(uuid4()), embedding, meta)])
    return {"status": "success", "chunks_indexed": len(chunks)}
