import openai
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4

# --- OpenAI key ---
openai.api_key = "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"

# --- Pinecone key & environment ---
pinecone_api_key = "pcsk_5TezHr_5FS18xfebbQxaGZwSRUELygH9RUq7Hnor9u7FdDDYoq4ztLAS2Fv2LC5W19Z4Me"
pinecone_env = "aped-4627-b74a"

# --- Create Pinecone client ---
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "clause-mind-index"

# --- Create index if missing (serverless spec) ---
if index_name not in pc.list_indexes().names():
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
