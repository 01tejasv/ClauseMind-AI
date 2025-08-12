import os
from dotenv import load_dotenv
import openai
from pinecone import Client
from uuid import uuid4

load_dotenv()  # load variables from .env

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

pc = Client(api_key=pinecone_api_key, environment=pinecone_env)
index_name = "clause-mind-index"

if index_name not in pc.list_indexes():
    pc.create_index(name=index_name, dimension=1536)

index = pc.index(index_name)

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
