import os
import openai
from pinecone import Pinecone
from uuid import uuid4

# Keep OpenAI key from environment or set directly here if you want
openai.api_key = os.getenv("OPENAI_API_KEY")
# Or set directly like:
# openai.api_key = "your-openai-api-key"

# Use your actual Pinecone API key and environment here
pinecone_api_key = "pcsk_4FYWkq_2v8TPeMYDjuU2Y3mUXiL22bRjEG6BkFGTHu6wcxMn8QVG4y6erdzVaygE14zr78"
pinecone_env = "aped-4627-b74a"

pc = Pinecone(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

index_name = "clause-mind-index"

# Check if index exists, create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=1536)

index = pc.index(index_name)

def process_and_index(content: str, filename: str):
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
    for chunk in chunks:
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )["data"][0]["embedding"]
        meta = {"filename": filename, "chunk": chunk}
        index.upsert([(str(uuid4()), embedding, meta)])
    return {"status": "success", "chunks_indexed": len(chunks)}
