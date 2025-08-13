import os
import openai
from pinecone import Client
from uuid import uuid4

# Use environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone client
client = Client(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "clause-mind-index"

# Create index if it doesn't exist
if index_name not in client.list_indexes():
    client.create_index(name=index_name, dimension=1536)

index = client.index(index_name)

def process_and_index(content: str, filename: str):
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

    for chunk in chunks:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embedding = response["data"][0]["embedding"]
        meta = {"filename": filename, "chunk": chunk}
        index.upsert(vectors=[(str(uuid4()), embedding, meta)])

    return {"status": "success", "chunks_indexed": len(chunks)}
