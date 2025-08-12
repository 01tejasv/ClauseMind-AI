import os
import openai
from pinecone import Client
from uuid import uuid4

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = "pcsk_4FYWkq_2v8TPeMYDjuU2Y3mUXiL22bRjEG6BkFGTHu6wcxMn8QVG4y6erdzVaygE14zr78"
pinecone_env = "aped-4627-b74a"

# Create Pinecone client
pc = Client(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "clause-mind-index"

# List existing indexes
indexes = pc.list_indexes()

# Create index if not exists
if index_name not in indexes:
    pc.create_index(name=index_name, dimension=1536)

# Connect to index
index = pc.index(index_name)

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
