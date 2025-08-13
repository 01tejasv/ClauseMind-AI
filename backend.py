import os
import openai
import pinecone
from uuid import uuid4

# 1. Set OpenAI API key (from environment variable or hardcode)
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "your_openai_api_key_here"  # Only if env var not set

# 2. Pinecone API key and environment
pinecone_api_key = os.getenv("PINECONE_API_KEY") or "pcsk_5TezHr_5FS18xfebbQxaGZwSRUELygH9RUq7Hnor9u7FdDDYoq4ztLAS2Fv2LC5W19Z4Me"
pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "aped-4627-b74a"

# 3. Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "clause-mind-index"

# 4. Create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536)

# 5. Connect to the index
index = pinecone.Index(index_name)

def process_and_index(content: str, filename: str):
    # Split content into 1000-char chunks
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

    for chunk in chunks:
        # Generate OpenAI embedding
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embedding = response["data"][0]["embedding"]
        meta = {"filename": filename, "chunk": chunk}

        # Upsert vector into Pinecone
        index.upsert(vectors=[(str(uuid4()), embedding, meta)])

    return {"status": "success", "chunks_indexed": len(chunks)}
