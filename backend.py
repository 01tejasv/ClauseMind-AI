import os
import openai
import pinecone
from uuid import uuid4

openai.api_key = os.getenv("OPENAI_API_KEY")

pinecone_api_key = "your_pinecone_api_key"
pinecone_env = "your_pinecone_env"

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = "clause-mind-index"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

index = pinecone.Index(index_name)

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
