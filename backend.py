import os
import openai
from pinecone import Client
from uuid import uuid4

# 1. Set OpenAI API key (either from env or hardcode)
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "your_openai_api_key_here"  # Use only if env var is not set

# 2. Pinecone API key and environment
pinecone_api_key = "pcsk_5TezHr_5FS18xfebbQxaGZwSRUELygH9RUq7Hnor9u7FdDDYoq4ztLAS2Fv2LC5W19Z4Me"  # your key
pinecone_env = "aped-4627-b74a"  # your environment

# 3. Initialize Pinecone client
pc = Client(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "clause-mind-index"

# 4. Create index if not exists (using dimension 1536 for OpenAI ada embeddings)
if index_name not in pc.list_indexes():
    pc.create_index(name=index_name, dimension=1536)

# 5. Connect to the index
index = pc.index(index_name)

def process_and_index(content: str, filename: str):
    # Split content into chunks (1000 chars each)
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

    for chunk in chunks:
        # Generate embedding from OpenAI
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embedding = response["data"][0]["embedding"]

        # Metadata
        meta = {"filename": filename, "chunk": chunk}

        # Upsert vector into Pinecone index
        index.upsert(vectors=[(str(uuid4()), embedding, meta)])

    return {"status": "success", "chunks_indexed": len(chunks)}
